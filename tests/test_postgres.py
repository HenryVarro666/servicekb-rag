"""PostgreSQL + pgvector 集成测试。

需要真实数据库，设置环境变量后运行::

    export SERVICEKB_TEST_DATABASE_URL=postgresql://servicekb:servicekb@localhost:5432/servicekb_test
    pytest -m postgres

**会清空该数据库里的 servicekb 表**，不要指向生产库！CI 里用 pgvector/pgvector 服务容器跑。
"""

import os

import pytest

from servicekb.store.base import IndexMismatchError
from tests.conftest import SOURCES

DSN = os.environ.get("SERVICEKB_TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DSN, reason="SERVICEKB_TEST_DATABASE_URL not set")]


@pytest.fixture(scope="module")
def pg_kb():
    from servicekb.config import Settings
    from servicekb.service import ServiceKB
    from servicekb.store.postgres import PostgresStore

    st = PostgresStore(DSN)
    st.reset()
    st.close()
    kb = ServiceKB(Settings(store_backend="postgres", database_url=DSN, sources_file=SOURCES, ocr_backend="metadata",
                            _env_file=None))
    kb.ingest(SOURCES)
    yield kb
    kb.close()


def test_ingest_is_idempotent(pg_kb):
    reports = pg_kb.ingest(SOURCES)
    assert all(r.added == 0 and r.updated == 0 and r.failed == 0 for r in reports)
    assert pg_kb.store.stats()["documents"]["indexed"] == 40


def test_hybrid_search_and_acl(pg_kb):
    r = pg_kb.search("310-04521-A 还能订到吗？", user_groups=["public"])
    assert r.results[0].chunk.doc_ref.startswith("parts_master:310-04521")
    assert set(r.results[0].channel_ranks) == {"dense", "keyword", "entity"}
    assert not any(sc.chunk.doc_ref.startswith(("bulletins:", "qa_archive:")) for sc in r.results)


def test_ask_logs_query_and_feedback(pg_kb):
    a = pg_kb.ask("AS-250K-HV 的直流开关坏了用哪个料号？")
    assert "220-11009-A" in a.answer and not a.escalate
    assert pg_kb.feedback(a.query_id, -1, "测试")


def test_refuses_mismatched_embedding_model(pg_kb):
    with pytest.raises(IndexMismatchError):
        pg_kb.store.init("some-other-model", 1024)


def test_tsvector_literal_roundtrip(pg_kb):
    from servicekb.store.postgres import build_tsquery_literal, build_tsvector_literal

    lit = build_tsvector_literal(["310-04521-b", "绝缘", "o'neil"], ["故障排除"])
    with pg_kb.store.pool.connection() as conn:
        assert conn.execute("SELECT %s::tsvector @@ %s::tsquery", (lit, build_tsquery_literal(["310-04521-b"]))).fetchone()[0]
        assert conn.execute("SELECT %s::tsvector @@ %s::tsquery", (lit, build_tsquery_literal(["o'neil"]))).fetchone()[0]


def test_photo_identification_on_postgres(pg_kb):
    from pathlib import Path

    photos = Path(__file__).resolve().parents[1] / "data" / "sample" / "photos"
    assert pg_kb.store.stats()["images"] == 22
    res = pg_kb.identify_photo(photos / "site_220-11009-A_label.jpg")
    assert res.identified and res.top.part_number == "220-11009-A" and res.top.ocr_confirmed
    res = pg_kb.identify_photo(photos / "site_unknown_helmet.jpg")
    assert not res.identified
    [rep] = pg_kb.ingest(SOURCES, only=["part_images"])
    assert (rep.images_indexed, rep.images_unchanged) == (0, 22)


def test_refuses_mismatched_image_model(pg_kb):
    with pytest.raises(IndexMismatchError):
        pg_kb.store.init(pg_kb.embedder.model_id, pg_kb.embedder.dim, "siglip:other:1152", 1152)


def test_query_log_ops_columns_and_prune(pg_kb):
    r = pg_kb.ask("客户 13812345678 问：E037 绝缘阻抗低怎么处理？", ["public", "service"], client="teams-bot")
    with pg_kb.store.pool.connection() as conn:
        q, reason, degraded, usage, client = conn.execute(
            "SELECT question, reason, degraded, usage, client FROM query_logs WHERE query_id = %s", (r.query_id,)
        ).fetchone()
        assert "[PHONE]" in q and client == "teams-bot" and isinstance(degraded, list) and isinstance(usage, dict)
        conn.execute("UPDATE query_logs SET created_at = now() - interval '400 days'")
    keep = pg_kb.ask("E038 怎么处理？", ["public", "service"])
    with pg_kb.store.pool.connection() as conn:
        conn.execute("UPDATE query_logs SET created_at = now() - interval '400 days' WHERE query_id = %s",
                     (keep.query_id,))
    assert pg_kb.feedback(keep.query_id, 1, None, "专家答案")
    n = pg_kb.store.prune_query_logs(180)
    assert n >= 1
    with pg_kb.store.pool.connection() as conn:
        left = {str(x[0]) for x in conn.execute("SELECT query_id FROM query_logs").fetchall()}
    assert keep.query_id in left and r.query_id not in left  # 有反馈的保留


def test_source_lock_is_exclusive_across_connections(pg_kb):
    st = pg_kb.store
    with st.source_lock("manuals") as a:
        assert a
        with st.source_lock("manuals") as b, st.source_lock("bulletins") as c:
            assert not b and c  # 同一数据源互斥，不同数据源互不影响
    with st.source_lock("manuals") as again:
        assert again  # 释放后可以再拿到
