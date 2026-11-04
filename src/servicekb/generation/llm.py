"""多厂家大模型接口：一个统一的 ``ChatModel``，背后接不同厂家。

为什么要抽这一层？
------------------
- **不被一家厂商锁死**：价格、效果、合规要求（数据能不能出境）、可用性都会变。换厂家只改配置。
- **主备切换**：主模型超时 / 限流 / 拒答时，自动换备用厂家（例如 OpenAI → 本地 vLLM），回答不中断。
- **上层代码不关心差异**：RAG 生成、聊天记录问答抽取都只调用 ``complete(system, user, json_schema)``。

各家接口的差异都收在 adapter 里（这正是 adapter 存在的意义）：

+-------------------+----------------------+--------------------------------------------------------------+
| 厂家              | 接口                 | 主要差异                                                       |
+===================+======================+==============================================================+
| OpenAI            | 官方 SDK             | 新模型用 max_completion_tokens；JSON 用 json_schema（strict）   |
| Azure OpenAI      | 官方 SDK（Azure 版） | 需要 endpoint + api_version；model 填的是**部署名**              |
| Claude            | Anthropic 原生 SDK   | system 是单独参数；max_tokens 必填（思考也占输出）；**不支持     |
| （官方/Bedrock/   |                      | temperature**；思考深度用 effort；可能返回 stop_reason=refusal；  |
|  Vertex）         |                      | JSON 用 output_config.format                                   |
| Gemini / DeepSeek | OpenAI 兼容接口      | 用 max_tokens；JSON 只支持 json_object（不保证符合 schema）       |
| / 通义 / Kimi /   |                      |                                                              |
| 智谱 / vLLM /     |                      |                                                              |
| Ollama            |                      |                                                              |
+-------------------+----------------------+--------------------------------------------------------------+

新增一个厂家：如果它兼容 OpenAI 接口，在 ``PRESETS`` 里加一行即可；否则写一个 ``ChatModel`` 子类
（参考 ``AnthropicChatModel``），在 ``_build_one`` 里注册。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from servicekb import metrics
from servicekb.config import LLMSpec, Settings
from servicekb.logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """调用失败（网络、限流、鉴权、拒答、输出被截断……）。上层据此切换备用厂家或降级为抽取式回答。"""


class LLMRefusalError(LLMError):
    """模型 / 安全过滤**拒绝回答这个请求**。和"服务故障"不同：它说明服务是好的，只是这条内容不行，
    所以不计入熔断失败次数（否则几个敏感问题就能让所有人都用不了这家）。"""


# --------------------------------------------------------------------------- token 用量

_usage_tls = threading.local()


def record_usage(model_name: str, input_tokens: int | None, output_tokens: int | None) -> None:
    """adapter 每次调用完成后调用：累计到当前线程的 ``track_usage()`` 里，并记入监控指标。"""
    i, o = int(input_tokens or 0), int(output_tokens or 0)
    metrics.LLM_TOKENS.labels(model_name, "input").inc(i)
    metrics.LLM_TOKENS.labels(model_name, "output").inc(o)
    calls = getattr(_usage_tls, "calls", None)
    if calls is not None:
        calls.append({"model": model_name, "input_tokens": i, "output_tokens": o})


@contextmanager
def track_usage() -> Iterator[list[dict[str, Any]]]:
    """收集这段代码里所有大模型调用的 token 用量（含失败后切换备用厂家的那几次——它们也要花钱）::

        with track_usage() as calls:
            answer = chat_model.complete(...)
        summary = summarize_usage(calls)   # {"input_tokens": 1834, "output_tokens": 412, "calls": 1, ...}
    """
    prev = getattr(_usage_tls, "calls", None)
    _usage_tls.calls = calls = []
    try:
        yield calls
    finally:
        _usage_tls.calls = prev


def summarize_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not calls:
        return {}
    return {
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "calls": len(calls),
        "models": list(dict.fromkeys(c["model"] for c in calls)),
    }


# --------------------------------------------------------------------------- 厂家预设


@dataclass(frozen=True)
class Preset:
    kind: str  # openai / azure / anthropic
    base_url: str | None  # None = SDK 默认（官方地址）
    key_env: str | None  # 默认从哪个环境变量读 API Key；None = 不需要 key
    max_tokens_param: str = "max_tokens"  # OpenAI 官方新模型用 max_completion_tokens
    json_mode: str = "object"  # schema = 支持 json_schema 严格模式；object = 只支持 json_object
    example_model: str = ""  # 只用于提示，不作为默认值（各家模型更新很快，请以官方文档为准）
    default_model: str | None = None
    note: str = ""


# 地址和模型名以各厂商官方文档为准；地址变了用 SERVICEKB_LLM_BASE_URL 覆盖即可，不用改代码。
PRESETS: dict[str, Preset] = {
    "openai": Preset("openai", None, "OPENAI_API_KEY", "max_completion_tokens", "schema", "gpt-4.1-mini"),
    "azure_openai": Preset("azure", None, "AZURE_OPENAI_API_KEY", "max_completion_tokens", "schema",
                           "<你的部署名>", note="base_url 填 https://<资源名>.openai.azure.com"),
    "anthropic": Preset("anthropic", None, "ANTHROPIC_API_KEY", default_model="claude-opus-5-5",
                        example_model="claude-opus-5-5", note="原生 SDK；pip install -e \".[anthropic]\""),
    "anthropic_bedrock": Preset("anthropic", None, None, default_model="anthropic.claude-opus-5-5",
                                example_model="anthropic.claude-opus-5-5", note="需要 aws_region；AWS 凭证按 AWS 默认方式读取"),
    "anthropic_vertex": Preset("anthropic", None, None, default_model="claude-opus-5-5",
                               example_model="claude-opus-5-5", note="需要 gcp_project_id；凭证用 gcloud ADC"),
    "gemini": Preset("openai", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
                     example_model="gemini-2.5-flash", note="Google 官方的 OpenAI 兼容接口"),
    "deepseek": Preset("openai", "https://api.deepseek.com", "DEEPSEEK_API_KEY", example_model="deepseek-chat"),
    "qwen": Preset("openai", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
                   example_model="qwen-plus", note="阿里云百炼；海外账号用 dashscope-intl 域名"),
    "moonshot": Preset("openai", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", example_model="kimi-k2-turbo-preview"),
    "zhipu": Preset("openai", "https://open.bigmodel.cn/api/paas/v4/", "ZHIPUAI_API_KEY", example_model="glm-4.5"),
    "vllm": Preset("openai", "http://localhost:8000/v1", None, example_model="Qwen/Qwen3-32B",
                   note="本地部署，数据不出公司"),
    "ollama": Preset("openai", "http://localhost:11434/v1", None, example_model="qwen3:8b", note="本地部署"),
    "openai_compatible": Preset("openai", None, "OPENAI_API_KEY", note="必须填 base_url"),
}


def _strip_json(text: str) -> str:
    """有的模型会把 JSON 包在 ```json ... ``` 里，或者前后带一两句话；取出第一个完整的 JSON 对象。"""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json(text))
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {text[:200]!r}") from e
    if not isinstance(data, dict):
        raise LLMError(f"expected a JSON object, got {type(data).__name__}")
    return data


# --------------------------------------------------------------------------- 统一接口


class ChatModel(ABC):
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        """单轮对话：返回模型的文字输出。``json_schema`` 不为空时要求输出符合该 schema 的 JSON。

        失败一律抛 ``LLMError``（上层统一处理），不要让各家 SDK 的异常类型漏到业务代码里。
        """


# --------------------------------------------------------------------------- OpenAI 及兼容接口


class OpenAIChatModel(ChatModel):
    def __init__(self, provider: str, model: str, *, api_key: str | None, base_url: str | None,
                 temperature: float | None, max_tokens: int, timeout: float, max_retries: int,
                 preset: Preset, azure_api_version: str | None = None, http_client: Any = None) -> None:
        import openai

        self.provider, self.model, self.preset = provider, model, preset
        self.temperature, self.max_tokens = temperature, max_tokens
        self._errors = (openai.APIError,)
        common = dict(api_key=api_key or "EMPTY", timeout=timeout, max_retries=max_retries, http_client=http_client)
        if preset.kind == "azure":
            if not base_url:
                raise ValueError("azure_openai needs base_url = https://<resource>.openai.azure.com")
            self.client = openai.AzureOpenAI(azure_endpoint=base_url, api_version=azure_api_version, **common)
        else:
            self.client = openai.OpenAI(base_url=base_url, **common)

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.preset.max_tokens_param: self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if json_schema is not None:
            if self.preset.json_mode == "schema":
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "output", "schema": json_schema, "strict": True}}
            else:  # 兼容接口大多只支持 json_object：把 schema 写进 system 提示
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["messages"][0]["content"] = (
                    f"{system}\n\n只输出一个符合以下 JSON Schema 的 JSON 对象：\n{json.dumps(json_schema, ensure_ascii=False)}"
                )
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except self._errors as e:
            raise LLMError(f"{self.name}: {e.__class__.__name__}: {e}") from e
        u = getattr(resp, "usage", None)
        record_usage(self.name, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        choice = resp.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
            raise LLMRefusalError(f"{self.name}: refused (finish_reason={choice.finish_reason})")
        if not text.strip():
            raise LLMError(f"{self.name}: empty response (finish_reason={choice.finish_reason})")
        return text


# --------------------------------------------------------------------------- Claude（原生 SDK）


SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChatModel(ChatModel):
    """Claude：官方 API / Amazon Bedrock / Google Vertex AI，都用 Anthropic 官方 Python SDK。

    和 OpenAI 风格接口的差异（都在这里处理掉）：
    - ``system`` 是单独的参数，不是一条消息；``max_tokens`` 必填；
    - 当前模型的思考（thinking）默认开启且计入输出 token，所以 max_tokens 默认给 16000；
    - **不支持 temperature 等采样参数**（发送会 400），思考深度用 ``output_config.effort`` 控制；
    - 安全分类器可能拒答：HTTP 200 但 ``stop_reason == "refusal"``，必须先检查再读内容；
      官方 API 上默认开启服务端 fallback（``fallbacks="default"``），被拒时由服务端换模型重跑；
    - JSON 输出用 ``output_config.format``（结构化输出，保证是合法 JSON）。
    """

    def __init__(self, provider: str, model: str, *, api_key: str | None, base_url: str | None, max_tokens: int,
                 timeout: float, max_retries: int, effort: str | None, server_fallbacks: bool,
                 aws_region: str | None = None, gcp_project_id: str | None = None, gcp_region: str = "global",
                 http_client: Any = None) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError('provider=anthropic* 需要 Anthropic SDK：pip install -e ".[anthropic]"') from e

        self.provider, self.model, self.max_tokens, self.effort = provider, model, max_tokens, effort
        self._anthropic = anthropic
        self._local = threading.local()  # 记录本线程最近一次实际回答的模型（API 服务是多线程的）
        common: dict[str, Any] = dict(timeout=timeout, max_retries=max_retries)
        if http_client is not None:
            common["http_client"] = http_client
        if provider == "anthropic_bedrock":
            if not aws_region:
                raise ValueError("anthropic_bedrock needs SERVICEKB_AWS_REGION")
            self.client = anthropic.AnthropicBedrockMantle(aws_region=aws_region, base_url=base_url, **common)
        elif provider == "anthropic_vertex":
            if not gcp_project_id:
                raise ValueError("anthropic_vertex needs SERVICEKB_GCP_PROJECT_ID")
            self.client = anthropic.AnthropicVertex(project_id=gcp_project_id, region=gcp_region, base_url=base_url,
                                                    **common)
        else:
            # api_key=None 时 SDK 会自己找凭证（ANTHROPIC_API_KEY、`ant auth login` 的登录信息等）
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, **common)
        # 服务端 fallback 只在 Anthropic 官方 API 上可用（Bedrock / Vertex 不支持这个参数）
        self.server_fallbacks = server_fallbacks and provider == "anthropic"

    @property
    def name(self) -> str:
        # 服务端 fallback 生效时，实际回答的是另一个模型（resp.model），日志里要记真实的那个
        return f"{self.provider}:{getattr(self._local, 'served_model', None) or self.model}"

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        self._local.served_model = None
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        a = self._anthropic
        try:
            if self.server_fallbacks:
                resp = self.client.beta.messages.create(**kwargs, betas=[SERVER_FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except a.APIStatusError as e:
            raise LLMError(f"{self.name}: HTTP {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:  # 含超时
            raise LLMError(f"{self.name}: connection error: {e}") from e

        self._local.served_model = getattr(resp, "model", None)
        u = getattr(resp, "usage", None)
        record_usage(self.name, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(f"{self.name}: refused by safety classifier (category={category})")
        # 只取文字块：思考块（thinking）、fallback 标记块等都不是答案
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text.strip():
            raise LLMError(f"{self.name}: no text in response (stop_reason={resp.stop_reason})")
        if resp.stop_reason == "max_tokens":
            log.warning("claude output truncated by max_tokens", extra={"model": self.model, "max_tokens": self.max_tokens})
        return text


# --------------------------------------------------------------------------- 熔断 + 指标


class CircuitBreaker:
    """熔断器（三种状态）：

    - **closed（正常）**：请求照常发；连续失败达到 ``failures`` 次 → open。
    - **open（断开）**：``cooldown_s`` 秒内直接拒绝，不发请求——主模型挂了的时候，
      请求立刻切到备用厂家，而不是每个请求都先傻等一次超时（60 秒 × 重试次数）。
    - **half-open（试探）**：冷却结束后只放**一个**请求过去：成功 → closed，失败 → 再断开一个冷却期。

    ``failures=0`` 表示不熔断。线程安全。
    """

    def __init__(self, failures: int = 3, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = failures, cooldown_s, clock
        self._lock = threading.Lock()
        self._consecutive = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self.clock() - self._opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self.clock() - self._opened_at >= self.cooldown_s and not self._trial_in_flight:
                self._trial_in_flight = True  # half-open：只放一个试探请求
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._consecutive, self._opened_at, self._trial_in_flight = 0, None, False

    def failure(self) -> bool:
        """记录一次失败；返回 True 表示熔断器因此（重新）断开。"""
        with self._lock:
            self._consecutive += 1
            was_trial, self._trial_in_flight = self._trial_in_flight, False
            if self.threshold and (was_trial or (self._opened_at is None and self._consecutive >= self.threshold)):
                self._opened_at = self.clock()
                return True
            return False


class GuardedChatModel(ChatModel):
    """包在每个厂家外面：熔断 + 监控指标（调用次数 / 结果 / 耗时）。``build_chat_model`` 自动加上。"""

    def __init__(self, inner: ChatModel, breaker: CircuitBreaker) -> None:
        self.inner, self.breaker = inner, breaker
        self.provider, self.model = inner.provider, inner.model

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        base = f"{self.provider}:{self.model}"  # 指标标签用配置的名字（不随服务端 fallback 变化）
        if not self.breaker.allow():
            metrics.LLM_REQUESTS.labels(base, "circuit_open").inc()
            raise LLMError(f"{base}: circuit open (skipped for up to {self.breaker.cooldown_s:.0f}s after repeated failures)")
        t0 = time.perf_counter()
        try:
            out = self.inner.complete(system, user, json_schema)
        except LLMRefusalError:
            self.breaker.success()  # 服务正常，只是这条内容被拒
            metrics.LLM_REQUESTS.labels(base, "refused").inc()
            raise
        except Exception:
            if self.breaker.failure():
                log.warning("llm circuit opened", extra={"model": base, "cooldown_s": self.breaker.cooldown_s})
                metrics.LLM_CIRCUIT_OPEN.labels(base).set(1)
            metrics.LLM_REQUESTS.labels(base, "error").inc()
            raise
        finally:
            metrics.LLM_LATENCY.labels(base).observe(time.perf_counter() - t0)
        self.breaker.success()
        metrics.LLM_CIRCUIT_OPEN.labels(base).set(0)
        metrics.LLM_REQUESTS.labels(base, "ok").inc()
        return out


def unwrap(m: ChatModel) -> ChatModel:
    """去掉 GuardedChatModel 外壳，拿到具体厂家的 adapter（测试 / 排查用）。"""
    return unwrap(m.inner) if isinstance(m, GuardedChatModel) else m


def circuit_states(m: ChatModel | None) -> dict[str, str]:
    """每个厂家的熔断状态，给 /readyz 和 servicekb doctor 用。"""
    if m is None:
        return {}
    models = m.models if isinstance(m, FallbackChatModel) else [m]
    return {f"{x.provider}:{x.model}": x.breaker.state for x in models if isinstance(x, GuardedChatModel)}


# --------------------------------------------------------------------------- 主备切换


class FallbackChatModel(ChatModel):
    """按顺序尝试多个厂家：前一个抛 LLMError 就换下一个。``name`` 返回本线程最近一次实际回答的那一家。"""

    def __init__(self, models: list[ChatModel]) -> None:
        if not models:
            raise ValueError("FallbackChatModel needs at least one model")
        self.models = models
        self.provider, self.model = models[0].provider, models[0].model
        self._local = threading.local()

    @property
    def served_by(self) -> ChatModel | None:
        return getattr(self._local, "served_by", None)

    @property
    def name(self) -> str:
        return (self.served_by or self.models[0]).name

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        errors = []
        for m in self.models:
            try:
                out = m.complete(system, user, json_schema)
                self._local.served_by = m
                return out
            except LLMError as e:
                log.warning("llm failed, trying next provider", extra={"model": m.name, "error": str(e)[:300]})
                errors.append(str(e))
        raise LLMError("all providers failed: " + " | ".join(errors))


# --------------------------------------------------------------------------- 构造


def _api_key(spec: LLMSpec, preset: Preset) -> str | None:
    if spec.api_key:
        return spec.api_key.get_secret_value()
    env = spec.api_key_env or preset.key_env
    return os.environ.get(env) if env else None


def _build_one(spec: LLMSpec, s: Settings) -> ChatModel:
    preset = PRESETS.get(spec.provider)
    if preset is None:
        raise ValueError(f"unknown llm provider {spec.provider!r}; known: {', '.join(sorted(PRESETS))}")
    model = spec.model or preset.default_model
    if not model:
        hint = f"，例如 {preset.example_model}" if preset.example_model else ""
        raise ValueError(f"llm provider {spec.provider!r} needs a model name (SERVICEKB_LLM_MODEL{hint})")
    base_url = spec.base_url or preset.base_url
    if spec.provider == "openai_compatible" and not base_url:
        raise ValueError("openai_compatible needs SERVICEKB_LLM_BASE_URL")
    if spec.provider == "openai" and base_url and "api.openai.com" not in base_url:
        # 旧配置常见写法：provider=openai + base_url 指向 vLLM / 网关。按兼容接口处理（max_tokens、json_object）
        preset = PRESETS["openai_compatible"]
    if preset.kind == "anthropic":
        return AnthropicChatModel(
            spec.provider, model, api_key=_api_key(spec, preset), base_url=spec.base_url,
            max_tokens=spec.max_tokens or 16000, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries,
            effort=spec.effort, server_fallbacks=s.anthropic_server_fallbacks, aws_region=spec.aws_region,
            gcp_project_id=spec.gcp_project_id, gcp_region=spec.gcp_region,
        )
    return OpenAIChatModel(
        spec.provider, model, api_key=_api_key(spec, preset), base_url=base_url, temperature=spec.temperature,
        max_tokens=spec.max_tokens or 1024, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries, preset=preset,
        azure_api_version=spec.azure_api_version,
    )


def build_chat_model(s: Settings, spec: LLMSpec | None = None, with_fallbacks: bool = True) -> ChatModel | None:
    """按配置构造主模型（+ 备用模型）。``provider=extractive`` 时返回 None（不调用大模型）。

    ``spec``：临时指定另一个厂家 / 模型（例如命令行 ``ask --provider deepseek``），不影响全局配置。
    """
    primary = spec or s.primary_llm_spec()
    if primary.provider == "extractive":
        return None
    specs = [primary] + ([f for f in s.llm_fallbacks if f.provider != "extractive"] if with_fallbacks else [])
    models: list[ChatModel] = [
        GuardedChatModel(_build_one(sp, s), CircuitBreaker(s.llm_circuit_failures, s.llm_circuit_cooldown_s))
        for sp in specs
    ]
    return models[0] if len(models) == 1 else FallbackChatModel(models)
