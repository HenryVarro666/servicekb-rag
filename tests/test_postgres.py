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
    kb = ServiceKB(Settings(store_backend="postgres", database_url=DSN, sources_file=SOURCES, _env_file=None))
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
