"""多厂家大模型接口测试：用 Mock HTTP 传输层检查"发出去的请求长什么样"，不需要任何 API Key / 网络。

OpenAI SDK 基于 httpx，Anthropic SDK（2.x）基于 httpx2，两者都提供 MockTransport。
"""

from __future__ import annotations

import importlib.util
import json
import re

import httpx
import pytest

from servicekb.config import LLMSpec
from servicekb.generation.llm import (
    PRESETS,
    SERVER_FALLBACK_BETA,
    AnthropicChatModel,
    ChatModel,
    FallbackChatModel,
    LLMError,
    OpenAIChatModel,
    build_chat_model,
    parse_json,
)
from tests.conftest import memory_settings

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


# --------------------------------------------------------------------------- OpenAI 及兼容接口


def _openai_reply(text: str) -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _openai_model(provider: str, model: str = "m", status: int = 200, **kw):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "boom", "type": "server_error"}})
        return httpx.Response(200, json=_openai_reply('{"ok": true}'))

    preset = PRESETS[provider]
    kw.setdefault("base_url", preset.base_url)
    m = OpenAIChatModel(
        provider, model, api_key="sk-test", temperature=0.1, max_tokens=256, timeout=5, max_retries=0,
        preset=preset, http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw,
    )
    return m, seen


def test_openai_official_uses_max_completion_tokens_and_strict_schema():
    m, seen = _openai_model("openai", "gpt-x")
    assert parse_json(m.complete("sys", "hi", SCHEMA)) == {"ok": True}
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.host == "api.openai.com" and req.headers["authorization"] == "Bearer sk-test"
    assert body["max_completion_tokens"] == 256 and "max_tokens" not in body
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"]
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["temperature"] == 0.1
    assert m.name == "openai:gpt-x"


def test_compatible_vendor_uses_preset_url_and_json_object():
    m, seen = _openai_model("deepseek", "deepseek-chat")
    m.complete("sys", "hi", SCHEMA)
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.deepseek.com/chat/completions"
    assert body["max_tokens"] == 256 and "max_completion_tokens" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert '"additionalProperties"' in body["messages"][0]["content"]  # schema 写进了 system 提示


def test_azure_uses_deployment_url_api_version_and_api_key_header():
    m, seen = _openai_model("azure_openai", "my-deploy", base_url="https://acme.openai.azure.com",
                            azure_api_version="2024-10-21")
    m.complete("sys", "hi")
    req = seen[0]
    assert req.url.path == "/openai/deployments/my-deploy/chat/completions"
    assert req.url.params["api-version"] == "2024-10-21" and req.headers["api-key"] == "sk-test"


def test_openai_http_error_becomes_llm_error():
    m, _ = _openai_model("vllm", "qwen", status=500)
    with pytest.raises(LLMError, match="vllm:qwen"):
        m.complete("sys", "hi")


# --------------------------------------------------------------------------- Claude


def _claude_model(reply: dict | None = None, status: int = 200, server_fallbacks: bool = True, **kw):
    import httpx2

    seen: list = []

    def handler(req):
        seen.append(req)
        if status != 200:
            return httpx2.Response(status, json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}})
        return httpx2.Response(200, json=reply)

    m = AnthropicChatModel(
        "anthropic", "claude-opus-5-5", api_key="sk-ant-test", base_url=None, max_tokens=16000, timeout=5,
        max_retries=0, effort="medium", server_fallbacks=server_fallbacks,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kw,
    )
    return m, seen


def _claude_reply(content: list, stop_reason: str = "end_turn", model: str = "claude-opus-5-5", **extra) -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            **extra}


needs_anthropic = pytest.mark.skipif(importlib.util.find_spec("anthropic") is None,
                                     reason='pip install -e ".[anthropic]"')


@needs_anthropic
def test_claude_request_shape_and_text_only_output():
    reply = _claude_reply([{"type": "thinking", "thinking": "先看证据", "signature": "s"},
                           {"type": "text", "text": "按 [1] 处理"}])
    m, seen = _claude_model(reply)
    assert m.complete("系统提示", "问题", SCHEMA) == "按 [1] 处理"  # 思考块不进入答案
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages"
    # SDK 版本不同，Key 可能放在 x-api-key 或 Authorization: Bearer 里
    assert "sk-ant-test" in (req.headers.get("x-api-key") or req.headers.get("authorization", ""))
    assert SERVER_FALLBACK_BETA in req.headers["anthropic-beta"] and body["fallbacks"] == "default"
    assert body["system"] == "系统提示" and body["messages"] == [{"role": "user", "content": "问题"}]
    assert body["max_tokens"] == 16000
    assert "temperature" not in body and "top_p" not in body  # Claude 当前模型不支持采样参数
    assert body["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}


@needs_anthropic
def test_claude_without_server_fallbacks_uses_plain_messages_api():
    m, seen = _claude_model(_claude_reply([{"type": "text", "text": "ok"}]), server_fallbacks=False)
    m.complete("s", "u")
    body = json.loads(seen[0].content)
    assert "fallbacks" not in body and "anthropic-beta" not in seen[0].headers
    assert body["output_config"] == {"effort": "medium"}


@needs_anthropic
def test_claude_records_model_that_actually_answered():
    # 服务端 fallback 生效时，响应里的 model 是实际回答的模型
    m, _ = _claude_model(_claude_reply([{"type": "text", "text": "ok"}], model="claude-fallback-x"))
    m.complete("s", "u")
    assert m.name == "anthropic:claude-fallback-x"


@needs_anthropic
def test_claude_refusal_raises():
    reply = _claude_reply([], stop_reason="refusal",
                          stop_details={"type": "refusal", "category": "cyber", "explanation": None})
    m, _ = _claude_model(reply)
    with pytest.raises(LLMError, match="refused.*cyber"):
        m.complete("s", "u")


@needs_anthropic
def test_claude_http_error_becomes_llm_error():
    m, _ = _claude_model(status=529)
    with pytest.raises(LLMError, match="HTTP 529"):
        m.complete("s", "u")


@needs_anthropic
def test_claude_server_fallbacks_disabled_on_bedrock_and_vertex():
    m, _ = _claude_model(_claude_reply([{"type": "text", "text": "ok"}]))
    assert m.server_fallbacks
    v = AnthropicChatModel("anthropic_vertex", "claude-opus-5-5", api_key=None, base_url=None, max_tokens=100,
                           timeout=5, max_retries=0, effort=None, server_fallbacks=True, gcp_project_id="p")
    assert not v.server_fallbacks  # 服务端 fallback 只在 Anthropic 官方 API 上可用
    with pytest.raises(ValueError, match="AWS_REGION"):
        AnthropicChatModel("anthropic_bedrock", "x", api_key=None, base_url=None, max_tokens=100, timeout=5,
                           max_retries=0, effort=None, server_fallbacks=True)


# --------------------------------------------------------------------------- 主备切换 / 构造 / 解析


class FakeModel(ChatModel):
    def __init__(self, name: str, reply: str | None = None, error: Exception | None = None) -> None:
        self.provider, self.model, self.reply, self.error, self.calls = "fake", name, reply, error, []

    def complete(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls.append((system, user, json_schema))
        if self.error:
            raise self.error
        return self.reply or ""


def test_fallback_chain_uses_next_provider_and_reports_who_answered():
    a, b = FakeModel("a", error=LLMError("timeout")), FakeModel("b", reply="答案")
    fb = FallbackChatModel([a, b])
    assert fb.complete("s", "u") == "答案" and fb.name == "fake:b" and len(a.calls) == 1
    with pytest.raises(LLMError, match="all providers failed"):
        FallbackChatModel([FakeModel("x", error=LLMError("e1")), FakeModel("y", error=LLMError("e2"))]).complete("s", "u")


def test_build_chat_model_from_settings(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert build_chat_model(memory_settings()) is None  # 默认 extractive：不调用大模型
    m = build_chat_model(memory_settings(llm_provider="deepseek", llm_model="deepseek-chat"))
    assert isinstance(m, OpenAIChatModel) and m.preset.base_url == "https://api.deepseek.com"
    assert m.client.api_key == "sk-from-env"  # 没配 SERVICEKB_LLM_API_KEY → 读厂家自己的环境变量

    fb = build_chat_model(memory_settings(
        llm_provider="deepseek", llm_model="deepseek-chat",
        llm_fallbacks=[LLMSpec(provider="vllm", model="Qwen/Qwen3-32B", base_url="http://gpu:8000/v1")],
    ))
    assert isinstance(fb, FallbackChatModel) and [x.name for x in fb.models] == ["deepseek:deepseek-chat",
                                                                                   "vllm:Qwen/Qwen3-32B"]
    # 临时指定另一个厂家时不带备用厂家
    one = build_chat_model(memory_settings(llm_fallbacks=[LLMSpec(provider="vllm", model="q")]),
                           LLMSpec(provider="ollama", model="qwen3:8b"), with_fallbacks=False)
    assert isinstance(one, OpenAIChatModel) and one.name == "ollama:qwen3:8b"

    with pytest.raises(ValueError, match="unknown llm provider"):
        build_chat_model(memory_settings(llm_provider="nope"))
    with pytest.raises(ValueError, match="needs a model name"):
        build_chat_model(memory_settings(llm_provider="deepseek"))
    with pytest.raises(ValueError, match="BASE_URL"):
        build_chat_model(memory_settings(llm_provider="openai_compatible", llm_model="m"))


@needs_anthropic
def test_claude_defaults_and_legacy_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    m = build_chat_model(memory_settings(llm_provider="anthropic"))
    assert isinstance(m, AnthropicChatModel) and m.model == "claude-opus-5-5" and m.max_tokens == 16000
    # 旧配置 llm_backend=openai + base_url 指向 vLLM：仍然可用，按兼容接口处理
    legacy = build_chat_model(memory_settings(llm_backend="openai", llm_model="q", llm_base_url="http://gpu:8000/v1"))
    assert isinstance(legacy, OpenAIChatModel) and legacy.provider == "openai"
    assert legacy.preset.max_tokens_param == "max_tokens" and legacy.preset.json_mode == "object"


def test_fallbacks_parse_from_env_json(monkeypatch):
    monkeypatch.setenv("SERVICEKB_LLM_FALLBACKS", '[{"provider":"vllm","model":"q","base_url":"http://g:8000/v1"}]')
    s = memory_settings()
    assert s.llm_fallbacks[0].provider == "vllm" and s.llm_fallbacks[0].base_url == "http://g:8000/v1"


def test_parse_json_tolerates_fences_and_rejects_garbage():
    assert parse_json('好的：\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('结果是 {"a": [1, 2]} 。') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json("没有 JSON")
    with pytest.raises(LLMError):
        parse_json("[1, 2]")


# --------------------------------------------------------------------------- 接入 RAG 问答 / 聊天抽取


def test_rag_answer_through_any_chat_model(kb):
    from servicekb.generation.generator import LLMGenerator
    from servicekb.generation.prompts import SYSTEM_PROMPT

    fake = FakeModel("rag", reply="检查直流侧绝缘 [1]，越界引用 [99] 会被删掉。")
    saved, kb.generator = kb.generator, LLMGenerator(fake)
    try:
        r = kb.ask("E037 绝缘阻抗低怎么处理？", ["public", "internal"], log_query=False)
        assert fake.calls and fake.calls[0][0] == SYSTEM_PROMPT and "E037" in fake.calls[0][1]
        assert r.model == "fake:rag" and [c.index for c in r.citations] == [1] and "[99]" not in r.answer

        kb.generator = LLMGenerator(FakeModel("down", error=LLMError("timeout")))  # 大模型挂了 → 降级
        r = kb.ask("E037 绝缘阻抗低怎么处理？", ["public", "internal"], log_query=False)
        assert r.model == "extractive-fallback" and "llm_error" in r.reason and r.citations
    finally:
        kb.generator = saved


class TranscriptModel(ChatModel):
    """假装是大模型：取对话里第一条消息编号作证据，并故意"编造"一个对话里没有的料号。"""

    provider, model = "fake", "extract"

    def __init__(self) -> None:
        self.schemas: list = []

    def complete(self, system, user, json_schema=None):  # noqa: ANN001
        self.schemas.append(json_schema)
        first_id = re.search(r"^\[([^\]]+)\]", user, re.M).group(1)
        return json.dumps({
            "is_qa": True, "question": "E037 怎么处理", "answer": "1. 检查绝缘", "status": "resolved",
            "product_model": "", "fault_codes": [], "part_numbers": ["999-99999-Z"],
            "evidence_message_ids": [first_id], "confidence": 1.7,
        })


def test_llm_chat_extractor_validates_vendor_output():
    from servicekb.chat import ChatCleaner, ChatCleaningConfig, Roster
    from servicekb.chat.llm_extract import EXTRACTION_SCHEMA, LLMChatExtractor
    from servicekb.chat.teams_source import iter_raw_messages
    from servicekb.connectors.dictionary import EntityDictionaryConnector
    from servicekb.connectors.parts_master import PartsMasterConnector
    from servicekb.processing.entities import EntityLinker
    from tests.conftest import ROOT

    e1, r1 = EntityDictionaryConnector("d", ROOT / "data/sample/dictionaries/entity_aliases.yaml").entities()
    e2, r2 = PartsMasterConnector("p", ROOT / "data/sample/parts/parts_master.csv").entities()
    chat = ROOT / "data" / "sample" / "chat"
    c = ChatCleaner(ChatCleaningConfig(), Roster.from_csv(chat / "roster.csv"), EntityLinker(e1 + e2, r1 + r2))
    fake = TranscriptModel()
    pairs = c.run(iter_raw_messages(chat / "teams_export_sample.jsonl"), LLMChatExtractor(fake))
    assert pairs and fake.schemas[0] is EXTRACTION_SCHEMA
    p = pairs[0]
    assert p.extractor.startswith("llm:fake:extract:") and p.confidence == 1.0  # 越界置信度被截断
    assert p.needs_review and "identifier_not_in_source:999-99999-Z" in p.review_reasons  # 编造的料号被拦下


def test_extraction_schema_is_strict_compatible():
    from servicekb.chat.llm_extract import EXTRACTION_SCHEMA

    # OpenAI strict 模式 / Claude 结构化输出共同要求：所有字段 required，不允许多余字段
    assert set(EXTRACTION_SCHEMA["required"]) == set(EXTRACTION_SCHEMA["properties"])
    assert EXTRACTION_SCHEMA["additionalProperties"] is False
