"""导入 pipeline：增量、幂等、向量复用、故障隔离、安全删除、墓碑。"""

from collections.abc import Iterator

from servicekb.connectors.base import Connector
from servicekb.embedding import HashingEmbedder
from servicekb.models import DocType, SourceRecord, Tombstone
from servicekb.pipeline.ingest import IngestionPipeline
from servicekb.processing.chunking import ChunkingConfig
from servicekb.processing.entities import EntityLinker
from servicekb.store.memory import MemoryStore


class ListConnector(Connector):
    def __init__(self, events, full_snapshot=True):
        self.source_id = "test"
        self.events = events
        self.full_snapshot = full_snapshot

    def iter_events(self, state) -> Iterator:
        yield from self.events


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(dim=64)
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


def md(ext_id: str, body: str) -> SourceRecord:
    return SourceRecord("test", ext_id, ext_id, DocType.MANUAL, body, "text/markdown", acl_groups=["public"])


def make_pipeline():
    store, emb = MemoryStore(), CountingEmbedder()
    store.init(emb.model_id, emb.dim)
    return IngestionPipeline(store, emb, EntityLinker(), ChunkingConfig(max_tokens=60, min_tokens=1)), store, emb


DOC = "## 1 安全\n断电后等待五分钟再操作设备。\n## 2 风扇\n检查风扇连接器是否松动。\n## 3 固件\n使用 U 盘升级固件版本。\n"


def test_idempotent_and_embedding_reuse():
    p, store, emb = make_pipeline()
    r1 = p.run(ListConnector([md("a.md", DOC)]))
    assert (r1.added, r1.chunks_written, emb.calls) == (1, 3, 3)

    r2 = p.run(ListConnector([md("a.md", DOC)]))
    assert (r2.unchanged, r2.added, r2.updated, emb.calls) == (1, 0, 0, 3)  # 没变 → 不重算

    r3 = p.run(ListConnector([md("a.md", DOC.replace("检查风扇连接器是否松动。", "检查风扇连接器 J12 是否松动。"))]))
    assert r3.updated == 1
    assert (r3.embeddings_computed, r3.embeddings_reused) == (1, 2)  # 只有改动的那一块重新向量化
    assert len(store.chunks) == 3


def test_failure_is_isolated_and_old_version_kept():
    p, store, _ = make_pipeline()
    p.run(ListConnector([md("a.md", DOC), md("b.md", "## x\n内容足够长的一段话。")]))
    bad = SourceRecord("test", "a.md", "a", DocType.MANUAL, b"%PDF-broken", "application/pdf")
    r = p.run(ListConnector([bad, md("b.md", "## x\n内容足够长的一段话。")]))
    assert r.failed == 1 and r.unchanged == 1 and r.status == "partial"
    assert store.get_document_state(bad.doc_id).status == "failed"
    assert len(store.doc_chunks[bad.doc_id]) == 3  # 旧版本的块还在，用户仍能检索到


def test_full_snapshot_deletes_missing_but_has_safety_valve():
    p, store, _ = make_pipeline()
    docs = [md(f"{i}.md", f"## s\n第{i}份文档的内容。") for i in range(12)]
    p.run(ListConnector(docs))
    r = p.run(ListConnector(docs[:11]))  # 删了 1 份：正常删除
    assert r.deleted == 1
    r = p.run(ListConnector(docs[:2]))  # 一下子"消失" 9/11：拒绝删除
    assert r.deleted == 0 and "deletion_skipped" in r.connector_stats
    r = p.run(ListConnector(docs[:2]), force_delete=True)
    assert r.deleted == 9


def test_incremental_connector_uses_tombstones_only():
    p, store, _ = make_pipeline()
    p.run(ListConnector([md("a.md", DOC), md("b.md", DOC)], full_snapshot=False))
    r = p.run(ListConnector([md("a.md", DOC)], full_snapshot=False))  # b 没出现，但这是增量：不能删
    assert r.deleted == 0 and len(store.docs) == 2
    r = p.run(ListConnector([Tombstone("test", "b.md")], full_snapshot=False))
    assert r.deleted == 1 and len(store.docs) == 1


def test_sample_ingest_report(kb):
    st = kb.store.stats()
    assert st["documents"]["indexed"] == 40  # 16 料号 + 3 手册 + 2 通告 + 19 工单（2 张被质量门槛过滤）
    assert st["entities"] > 20
    run = next(r for r in kb.store.runs if r["source_id"] == "qa_archive")
    assert run["stats"]["connector_stats"]["filtered_unresolved"] == 2
