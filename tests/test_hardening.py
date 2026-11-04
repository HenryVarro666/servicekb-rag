"""生产加固相关测试：并发、降级、熔断、token 用量、日志脱敏、过载保护、调用方权限、请求 ID、监控指标。

见 docs/12_production_hardening.md。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from servicekb.embedding import CachedQueryEmbedder, HashingEmbedder
from servicekb.generation.llm import (
    ChatModel,
    CircuitBreaker,
    FallbackChatModel,
    GuardedChatModel,
    LLMError,
    LLMRefusalError,
    record_usage,
    summarize_usage,
    track_usage,
)
from servicekb.models import SearchFilters
from servicekb.retrieval.retriever import HybridRetriever, RetrievalError
from tests.conftest import memory_settings

FILTERS = SearchFilters(acl_groups=["public", "service", "internal"])


# --------------------------------------------------------------------------- 检索：并发与降级


class SlowStore:
    """模拟真实环境里每一路召回都要等 I/O（数据库 / 网络）。只代理检索用到的三个方法。"""

    def __init__(self, inner, delay: float = 0.03, fail: set[str] | None = None, hang: set[str] | None = None):
        self.inner, self.delay, self.fail, self.hang = inner, delay, fail or set(), hang or set()

    def _call(self, name: str, *args):
        if name in self.hang:
            time.sleep(2)
        time.sleep(self.delay)
        if name in self.fail:
            raise ConnectionError(f"{name} down")
        return getattr(self.inner, f"{name}_search")(*args)

    def dense_search(self, *a):
        return self._call("dense", *a)

    def keyword_search(self, *a):
        return self._call("keyword", *a)

    def entity_search(self, *a):
        return self._call("entity", *a)


def _retriever(kb, store, **kw) -> HybridRetriever:
    return HybridRetriever(store, kb.embedder, kb.linker, None, **kw)


def test_shared_pool_size_bounds_concurrency(kb):
    """旧版本：整个进程只有 3 个召回线程，并发请求互相排队。12 个并发请求 × 3 路 × 30 ms：
    3 线程 ≈ 12 轮 × 30 ms ≈ 360 ms；36 线程 ≈ 1 轮 ≈ 30 ms。"""
    store = SlowStore(kb.store)

    def run(workers: int) -> float:
        r = _retriever(kb, store, max_workers=workers)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(12) as ex:
            list(ex.map(lambda _: r.retrieve("E037 绝缘阻抗低", FILTERS), range(12)))
        r.close()
        return time.perf_counter() - t0

    small, large = run(3), run(36)
    assert large < small * 0.5, (small, large)


def test_embedding_failure_degrades_to_keyword_and_entity(kb):
    class Broken(HashingEmbedder):
        def embed_query(self, text):
            raise TimeoutError("embedding service timeout")

    r = HybridRetriever(kb.store, Broken(kb.embedder.dim), kb.linker, None)
    res = r.retrieve("E037 绝缘阻抗低怎么处理", FILTERS)
    assert res.results and res.degraded == ["dense:embedding_failed"]
    assert all("dense" not in sc.channel_ranks for sc in res.results)


def test_failed_or_slow_channel_is_skipped(kb):
    r = _retriever(kb, SlowStore(kb.store, delay=0, fail={"keyword"}, hang={"entity"}), channel_timeout_s=0.5)
    t0 = time.perf_counter()
    res = r.retrieve("E037 绝缘阻抗低", FILTERS)
    assert time.perf_counter() - t0 < 1.5  # 没有等那一路"卡死"的召回
    assert res.results and sorted(res.degraded) == ["entity:TimeoutError", "keyword:ConnectionError"]


def test_all_channels_failing_raises_retrieval_error(kb):
    r = _retriever(kb, SlowStore(kb.store, delay=0, fail={"dense", "keyword", "entity"}))
    with pytest.raises(RetrievalError):
        r.retrieve("E037", FILTERS)


def test_query_embedding_cache_lru():
    calls = []

    class Counting(HashingEmbedder):
        def embed_query(self, text):
            calls.append(text)
            return super().embed_query(text)

    c = CachedQueryEmbedder(Counting(64), maxsize=2)
    v1 = c.embed_query("E037")
    v1.append(999.0)  # 调用方改了返回值，不能污染缓存
    assert c.embed_query("E037") == HashingEmbedder(64).embed_query("E037") and calls == ["E037"]
    c.embed_query("E038")
    c.embed_query("E039")  # 超出容量，最久没用的 E037 被淘汰
    c.embed_query("E037")
    assert calls == ["E037", "E038", "E039", "E037"] and (c.hits, c.misses) == (1, 4)
    assert c.model_id == "hashing-v1-64" and c.dim == 64


def test_kb_uses_query_cache_by_default(kb):
    assert isinstance(kb.embedder, CachedQueryEmbedder)
    assert not isinstance(_kb_without_cache().embedder, CachedQueryEmbedder)


def _kb_without_cache():
    from servicekb.service import ServiceKB

    return ServiceKB(memory_settings(query_embedding_cache_size=0))


# --------------------------------------------------------------------------- 大模型：熔断 / 用量


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_circuit_breaker_state_machine():
    clock = Clock()
    b = CircuitBreaker(failures=3, cooldown_s=30, clock=clock)
    for _ in range(2):
        assert b.allow() and not b.failure()
    assert b.allow() and b.failure() and b.state == "open"  # 第 3 次连续失败 → 断开
    assert not b.allow()
    clock.t = 31
    assert b.state == "half_open" and b.allow()  # 冷却结束：放一个试探请求
    assert not b.allow()  # 试探还没结束，其他请求继续跳过
    assert b.failure() and b.state == "open"  # 试探失败 → 再断开一个冷却期
    clock.t = 62
    assert b.allow()
    b.success()
    assert b.state == "closed" and b.allow()
    assert not CircuitBreaker(failures=0).failure()  # 0 = 不熔断


class Flaky(ChatModel):
    provider, model = "fake", "flaky"

    def __init__(self, error: Exception | None = None, reply: str = "ok", usage: tuple[int, int] = (0, 0)):
        self.error, self.reply, self.usage, self.calls = error, reply, usage, 0

    def complete(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        record_usage(self.name, *self.usage)
        if self.error:
            raise self.error
        return self.reply


def test_open_circuit_skips_primary_without_calling_it():
    primary = Flaky(LLMError("timeout"))
    backup = Flaky(reply="来自备用厂家")
    g = GuardedChatModel(primary, CircuitBreaker(failures=2, cooldown_s=60))
    fb = FallbackChatModel([g, GuardedChatModel(backup, CircuitBreaker())])
    for _ in range(5):
        assert fb.complete("s", "u") == "来自备用厂家"
    assert primary.calls == 2 and backup.calls == 5  # 断开后主模型不再被调用（不再每次等超时）
    assert g.breaker.state == "open"


def test_refusals_do_not_open_the_circuit():
    g = GuardedChatModel(Flaky(LLMRefusalError("refused")), CircuitBreaker(failures=1))
    for _ in range(3):
        with pytest.raises(LLMRefusalError):
            g.complete("s", "u")
    assert g.breaker.state == "closed"


def test_usage_is_tracked_across_fallback_attempts():
    fb = FallbackChatModel([Flaky(LLMError("x"), usage=(100, 0)), Flaky(usage=(120, 30))])
    with track_usage() as calls:
        fb.complete("s", "u")
    assert summarize_usage(calls) == {"input_tokens": 220, "output_tokens": 30, "calls": 2, "models": ["fake:flaky"]}
    assert summarize_usage([]) == {}


def test_answer_records_usage_and_degradation(kb):
    from servicekb.generation.generator import LLMGenerator

    saved = kb.generator
    try:
        kb.generator = LLMGenerator(Flaky(reply="检查绝缘 [1]", usage=(1500, 80)))
        r = kb.ask("E037 绝缘阻抗低怎么处理？", ["public", "service"], log_query=False)
        assert r.usage["input_tokens"] == 1500 and r.usage["output_tokens"] == 80 and r.degraded == []

        kb.generator = LLMGenerator(Flaky(LLMError("down")))
        r = kb.ask("E037 绝缘阻抗低怎么处理？", ["public", "service"], log_query=False)
        assert r.model == "extractive-fallback" and r.degraded == ["generation:extractive_fallback"]
    finally:
        kb.generator = saved


# --------------------------------------------------------------------------- 查询日志：脱敏 / 清理 / 导入锁


def test_query_log_is_pii_redacted(kb):
    r = kb.ask("客户 13812345678 的 AS-110K-HV 报 E037，序列号 AS2203123456，怎么处理？", ["public", "service"],
               client="teams-bot")
    entry = kb.store.query_logs[r.query_id]
    assert "13812345678" not in entry["question"] and "[PHONE]" in entry["question"]
    assert "AS2203123456" in entry["question"] and "E037" in entry["question"]  # 序列号 / 故障码保留
    assert "13812345678" in r.question  # 返回给用户的不受影响
    assert entry["client"] == "teams-bot" and "reason" in entry and "usage" in entry


def test_prune_query_logs_keeps_feedback(fresh_kb):
    st = fresh_kb.store
    old = datetime.now(timezone.utc) - timedelta(days=400)
    for q in ("a", "b", "c"):
        st.log_query({"query_id": q, "question": q})
    st.query_logs["a"]["created_at"] = st.query_logs["b"]["created_at"] = old
    st.log_feedback("b", 1, None, "专家修正的答案")
    assert st.prune_query_logs(180) == 1 and set(st.query_logs) == {"b", "c"}
    assert st.prune_query_logs(180, keep_feedback=False) == 1 and set(st.query_logs) == {"c"}
    assert st.feedback == []


def test_concurrent_ingestion_of_same_source_is_skipped(fresh_kb):
    from servicekb.connectors import build_connector, load_sources
    from tests.conftest import SOURCES

    spec = next(s for s in load_sources(SOURCES) if s["name"] == "dictionaries")
    conn = build_connector(spec, SOURCES.resolve().parent)
    with fresh_kb.store.source_lock(conn.source_id) as got:
        assert got
        assert fresh_kb.pipeline.run(conn).status == "skipped_locked"
    assert fresh_kb.pipeline.run(conn).status == "succeeded"  # 锁释放后正常导入


# --------------------------------------------------------------------------- HTTP API


@pytest.fixture()
def api(kb):
    """返回一个工厂：按需要的配置建 TestClient，共享已导入数据的 kb。"""
    from fastapi.testclient import TestClient

    from servicekb.api import create_app

    clients = []

    def make(**kw):
        c = TestClient(create_app(memory_settings(**kw), kb=kb))
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def test_request_id_is_echoed_or_generated(api):
    c = api()
    assert c.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    generated = c.get("/healthz", headers={"X-Request-ID": "bad id\nwith newline"}).headers["X-Request-ID"]
    assert len(generated) == 32 and "\n" not in generated  # 不安全的 ID 被替换，防止日志注入


def test_metrics_endpoint(api):
    c = api()
    c.post("/v1/search", json={"question": "E037 怎么处理"})
    body = c.get("/metrics").text
    assert 'servicekb_http_requests_total{method="POST",route="/v1/search",status="200"}' in body
    assert "servicekb_stage_duration_seconds" in body
    assert api(metrics_enabled=False).get("/metrics").status_code == 404


def test_api_client_groups_are_capped(api, kb, monkeypatch):
    seen = []
    real = kb.search
    monkeypatch.setattr(kb, "search", lambda q, groups, *a, **kw: seen.append(groups) or real(q, groups, *a, **kw))
    c = api(api_keys=["gateway-key"],
            api_clients=[{"name": "teams-bot", "key": "bot-key", "groups": ["public", "service"]}])
    hdr = {"X-User-Groups": "service,internal"}
    assert c.post("/v1/search", json={"question": "E037"}, headers={**hdr, "X-API-Key": "bot-key"}).status_code == 200
    assert c.post("/v1/search", json={"question": "E037"}, headers={**hdr, "X-API-Key": "gateway-key"}).status_code == 200
    assert seen == [["service"], ["service", "internal"]]  # 机器人的 Key 不能越权拿到 internal
    assert c.post("/v1/search", json={"question": "E037"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_input_validation_returns_422_not_500(api):
    c = api()
    assert c.post("/v1/ask", json={"question": "E037", "doc_types": ["nope"]}).status_code == 422
    assert c.post("/v1/ask", json={"question": "E037", "product_models": ["x"] * 21}).status_code == 422
    assert c.post("/v1/feedback", json={"query_id": "not-a-uuid", "rating": 1}).status_code == 422
    assert c.post("/v1/feedback", json={"query_id": "00000000-0000-0000-0000-000000000000",
                                        "rating": 5}).status_code == 422
    ok = c.post("/v1/ask", json={"question": "E037 怎么处理", "doc_types": ["manual", "bulletin"]})
    assert ok.status_code == 200 and "degraded" in ok.json() and "usage" in ok.json()


def test_overload_returns_503_with_retry_after(api, kb, monkeypatch):
    started = threading.Event()

    def slow_search(*a, **kw):
        started.set()
        time.sleep(0.5)
        return real(*a, **kw)

    real = kb.search
    monkeypatch.setattr(kb, "search", slow_search)
    c = api(max_concurrent_requests=1, overload_wait_s=0.05)
    with ThreadPoolExecutor(1) as ex:
        first = ex.submit(c.post, "/v1/search", json={"question": "E037"})
        started.wait(2)
        second = c.post("/v1/search", json={"question": "E037"})
        assert first.result().status_code == 200
    assert second.status_code == 503 and second.headers["Retry-After"] == "1"


def test_store_outage_returns_503(api, kb, monkeypatch):
    def down(*a, **kw):
        raise RetrievalError("all retrieval channels failed")

    monkeypatch.setattr(kb, "ask", down)
    r = api().post("/v1/ask", json={"question": "E037"})
    assert r.status_code == 503 and r.headers["Retry-After"] == "5"


def test_readyz_reports_llm_circuit_state(api, kb):
    from servicekb.generation.generator import LLMGenerator

    saved = kb.generator
    try:
        g = GuardedChatModel(Flaky(), CircuitBreaker(failures=1))
        g.breaker.failure()
        kb.generator = LLMGenerator(g)
        body = api().get("/readyz").json()
        assert body["status"] == "ok" and body["llm"] == {"fake:flaky": "open"}  # 熔断不影响就绪
    finally:
        kb.generator = saved


# --------------------------------------------------------------------------- servicekb doctor


def test_doctor_flags_risky_config_and_broken_dependencies():
    from servicekb.doctor import config_checks, run_checks

    dev = {c.name for c in config_checks(memory_settings())}
    assert {"config.auth", "config.store", "config.embedding"} <= dev
    assert "config.retrieval_pool" not in dev and "config.db_pool" not in dev  # 默认值彼此一致

    risky = {c.name for c in config_checks(memory_settings(retrieval_max_workers=6, query_log_redact_pii=False))}
    assert {"config.retrieval_pool", "config.privacy"} <= risky

    checks = {c.name: c for c in run_checks(memory_settings(embedding_dim=256))}
    assert checks["embedding"].status == "ok" and checks["store"].status == "ok"
    broken = {c.name: c for c in run_checks(memory_settings(reranker_backend="tei", reranker_url="http://127.0.0.1:9",
                                                             llm_provider="deepseek"))}
    assert broken["reranker"].status == "fail" and broken["llm"].status == "fail"
    assert "needs a model name" in broken["llm"].detail
