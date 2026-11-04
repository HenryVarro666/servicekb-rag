"""监控指标（Prometheus）：``GET /metrics``。

日志回答"这一次发生了什么"，指标回答"整体怎么样、什么时候开始变差的"。
生产环境至少要能回答下面这些问题，所以就定义了下面这些指标：

| 问题 | 指标 |
| --- | --- |
| 接口慢不慢、错误率多少？ | ``servicekb_http_requests_total`` / ``servicekb_http_request_duration_seconds`` |
| 时间花在哪一步（召回 / 重排 / 生成）？ | ``servicekb_stage_duration_seconds{stage}`` |
| 拒答率是不是突然升高了（数据没导进来 / reranker 阈值不对）？ | ``servicekb_ask_total{outcome}`` |
| 哪个依赖在出问题（向量服务 / 重排 / 某一路召回）？ | ``servicekb_degraded_total{component}`` |
| 大模型：哪家在用、失败率、熔断、花了多少 token？ | ``servicekb_llm_*`` |
| 过载保护有没有在拒绝请求？ | ``servicekb_overload_rejected_total`` / ``servicekb_inflight_requests`` |

建议的告警（Grafana / Alertmanager）：5xx 比例 > 1%；p95 延迟 > 10 s；拒答率比上周同期高一倍；
``servicekb_llm_circuit_open == 1`` 持续 5 分钟；``degraded_total`` 增速突增。

多进程部署（``servicekb serve --workers N``）时，每个进程有自己的计数器，Prometheus 每次只抓到其中一个。
两种做法：① 容器化部署每个容器 1 个 worker，靠增加副本扩容（推荐）；② 设置环境变量
``PROMETHEUS_MULTIPROC_DIR``（每次启动前清空该目录），``/metrics`` 会自动汇总所有进程。

没装 prometheus-client 时，所有指标调用都是空操作，不影响业务功能。
"""

from __future__ import annotations

import os
from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    AVAILABLE = True
except ImportError:  # pragma: no cover - 可选依赖缺失
    AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class _Noop:
    def labels(self, *_: Any, **__: Any) -> _Noop:
        return self

    def inc(self, *_: Any) -> None: ...
    def dec(self, *_: Any) -> None: ...
    def set(self, *_: Any) -> None: ...
    def observe(self, *_: Any) -> None: ...


def _make(kind: str, name: str, doc: str, labels: tuple[str, ...] = (), **kw: Any) -> Any:
    if not AVAILABLE:
        return _Noop()
    cls = {"counter": Counter, "gauge": Gauge, "histogram": Histogram}[kind]
    if kind == "gauge" and os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        kw.setdefault("multiprocess_mode", "livesum")
    return cls(name, doc, labels, **kw)


# 延迟分桶：覆盖"毫秒级的检索"到"几十秒的大模型生成"
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)

HTTP_REQUESTS = _make("counter", "servicekb_http_requests_total", "HTTP 请求数", ("method", "route", "status"))
HTTP_LATENCY = _make("histogram", "servicekb_http_request_duration_seconds", "HTTP 请求耗时",
                     ("method", "route"), buckets=_LATENCY_BUCKETS)
ASK_TOTAL = _make("counter", "servicekb_ask_total", "问答结果：answered / refused / llm_fallback", ("outcome",))
STAGE_LATENCY = _make("histogram", "servicekb_stage_duration_seconds", "问答各阶段耗时", ("stage",),
                      buckets=_LATENCY_BUCKETS)
DEGRADED = _make("counter", "servicekb_degraded_total", "降级次数（某个依赖失败但请求仍成功）", ("component",))
LLM_REQUESTS = _make("counter", "servicekb_llm_requests_total", "大模型调用：ok / error / refused / circuit_open",
                     ("model", "outcome"))
LLM_LATENCY = _make("histogram", "servicekb_llm_request_duration_seconds", "大模型单次调用耗时", ("model",),
                    buckets=_LATENCY_BUCKETS)
LLM_TOKENS = _make("counter", "servicekb_llm_tokens_total", "大模型 token 用量", ("model", "type"))
LLM_CIRCUIT_OPEN = _make("gauge", "servicekb_llm_circuit_open", "熔断状态（1 = 断开，正在跳过该厂家）", ("model",))
EMBED_CACHE = _make("counter", "servicekb_query_embedding_cache_total", "问题向量缓存 hit / miss", ("result",))
OVERLOAD_REJECTED = _make("counter", "servicekb_overload_rejected_total", "过载保护拒绝的请求数", ("route",))
INFLIGHT = _make("gauge", "servicekb_inflight_requests", "正在处理的重请求数（ask / search / photo）")


def observe_answer(result: Any) -> None:
    """一次问答结束后记录：结果类型、各阶段耗时、降级。"""
    if result.reason.startswith("llm_error"):
        outcome = "llm_fallback"
    elif result.escalate:
        outcome = "refused"
    else:
        outcome = "answered"
    ASK_TOTAL.labels(outcome).inc()
    for stage, ms in result.timings_ms.items():
        STAGE_LATENCY.labels(stage).observe(ms / 1000)
    for d in result.degraded:
        DEGRADED.labels(d.split(":", 1)[0]).inc()


def render() -> tuple[bytes, str]:
    """返回 (内容, Content-Type)。设置了 PROMETHEUS_MULTIPROC_DIR 时汇总所有 worker 进程。"""
    if not AVAILABLE:
        return b"# prometheus-client not installed\n", CONTENT_TYPE_LATEST
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
