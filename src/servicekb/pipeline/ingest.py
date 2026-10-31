"""数据导入 pipeline：Connector → 解析 → 清洗/增强 → 切块 → 实体链接 → 分词 → 向量化 → 入库。

工业级 pipeline 和 demo 脚本的区别，主要在下面这几点（每一点都对应真实踩过的坑）：

1. **增量 & 幂等**：同一份数据跑多少次结果都一样；没变的文档直接跳过。
   判断依据 = ``content_hash``（内容变没变）+ ``pipeline_version``（处理逻辑 / 模型 / 词典变没变）。
2. **向量复用**：块 ID 由"文档 ID + 块内容哈希"决定。文档改了一个段落，只有那一块需要重新算向量，
   其余块的向量从数据库里取回复用——对几十万块的知识库，这能把重建时间从小时降到分钟。
   （借鉴 RAGFlow 的 task fingerprint / LlamaIndex 的 docstore 策略）
3. **故障隔离**：一份 PDF 解析失败不影响其他文档；失败的文档记录错误、保留旧版本，下次自动重试。
4. **安全删除**：全量同步时，"本次没看到的文档"才删除；如果比例异常高（网盘没挂上？），拒绝删除。
5. **可观测**：每次运行写一条 ingestion_runs，记录新增 / 更新 / 跳过 / 失败 / 删除数量。
6. **批量向量化**：攒够一批块再调用模型，GPU 利用率高得多（工单一张一块，逐条调用会非常慢）。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from servicekb.connectors.base import Connector
from servicekb.embedding import Embedder
from servicekb.logging import get_logger
from servicekb.models import Chunk, DocType, ParsedDocument, SourceRecord, Tombstone, content_hash, stable_id
from servicekb.parsing import parse
from servicekb.processing.chunking import ChunkingConfig, chunk_document
from servicekb.processing.dedup import simhash
from servicekb.processing.entities import EntityLinker, canonical_tokens
from servicekb.processing.text import estimate_tokens, tokenize
from servicekb.store.base import Store

log = get_logger(__name__)

# 修改了清洗 / 切块 / 实体链接的**代码逻辑**时，手动把这个版本号加一，所有文档会重新处理一遍
# （向量会尽量复用，成本不高）。
PIPELINE_CODE_VERSION = "1"



@dataclass
class IngestReport:
    source_id: str
    status: str = "running"
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    deleted: int = 0
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    entities: int = 0
    connector_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["errors"] = d["errors"][:50]
        return d


@dataclass
class _Pending:
    record: SourceRecord
    content_hash: str
    chunks: list[Chunk]
    is_new: bool


class IngestionPipeline:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        linker: EntityLinker,
        chunk_cfg: ChunkingConfig,
        max_delete_ratio: float = 0.5,
        batch_chunks: int = 256,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.linker = linker
        self.chunk_cfg = chunk_cfg
        self.max_delete_ratio = max_delete_ratio
        self.batch_chunks = batch_chunks

    # ------------------------------------------------------------------ 版本指纹
    def pipeline_version(self) -> str:
        """处理逻辑的指纹。任何一项变化，都意味着已有块需要重新生成。"""
        lk = self.linker
        linker_fp = stable_id(
            *sorted(f"{e.entity_id}={'|'.join(sorted(e.aliases))}" for e in lk.entities.values()),
            *sorted(f"{r.src}>{r.relation}>{r.dst}" for r in lk.relations),
        )
        return stable_id(PIPELINE_CODE_VERSION, self.chunk_cfg.fingerprint(), self.embedder.model_id, linker_fp, length=16)

    # ------------------------------------------------------------------ 单文档处理
    def _doc_models(self, record: SourceRecord, parsed: ParsedDocument) -> list[str]:
        """文档级适用机型：front matter / connector 元数据 / 标题里提到的型号。"""
        raw = list(parsed.metadata.get("product_models") or []) + list(record.metadata.get("product_models_raw") or [])
        models = [m for m in (self.linker.resolve_model(str(r)) for r in raw) if m]
        models += [
            self.linker.entities[eid].canonical_name
            for eid in self.linker.link(record.title)
            if eid.startswith("model:") and eid in self.linker.entities
        ]
        return list(dict.fromkeys(models))

    def prepare(self, record: SourceRecord) -> tuple[ParsedDocument, str, list[str]]:
        parsed = parse(record)
        if parsed.metadata.get("doc_type"):
            record.doc_type = DocType(parsed.metadata["doc_type"])
        # 注意：acl_groups 只认 sources.yaml 的配置，不认文档 front matter。
        # 权限由管理员配置，不能让文档作者自己"放开"权限。
        models = self._doc_models(record, parsed)
        fingerprint = json.dumps(
            [record.title, record.doc_type.value, sorted(record.acl_groups), models, parsed.full_text(),
             {k: v for k, v in sorted(record.metadata.items())}],
            ensure_ascii=False, default=str,
        )
        return parsed, content_hash(fingerprint), models

    def build_chunks(self, record: SourceRecord, parsed: ParsedDocument, models: list[str]) -> list[Chunk]:
        drafts = chunk_document(parsed, self.chunk_cfg)
        model_eids = [f"model:{m}" for m in models]
        occurrences: Counter[str] = Counter()
        chunks: list[Chunk] = []
        for i, d in enumerate(drafts):
            header = " > ".join(d.heading_path)
            embed_text = f"{header}\n{d.text}" if header else d.text
            h = content_hash(embed_text)
            n = occurrences[h]
            occurrences[h] += 1

            mentions = self.linker.extract(f"{header}\n{d.text}")
            entity_ids = list(dict.fromkeys([m.entity_id for m in mentions] + model_eids))
            # 块内明确提到了某些型号，就用块内的；否则继承文档级型号
            mentioned_models = [
                self.linker.entities[m.entity_id].canonical_name
                for m in mentions if m.entity_id.startswith("model:") and m.entity_id in self.linker.entities
            ]
            chunk_models = list(dict.fromkeys(mentioned_models)) or models
            meta = dict(d.metadata)
            meta["kind"] = d.kind
            unknown = [m.entity_id for m in mentions if not m.known]
            if unknown:
                meta["unknown_entities"] = unknown  # 文档里出现但物料主数据里没有的料号 → 数据质量报告
            if record.updated_at:
                meta["source_updated_at"] = record.updated_at.isoformat()

            chunks.append(
                Chunk(
                    chunk_id=stable_id(record.doc_id, h, str(n)),
                    doc_id=record.doc_id,
                    ordinal=i,
                    text=d.text,
                    heading_path=d.heading_path,
                    page=d.page,
                    doc_type=record.doc_type,
                    acl_groups=list(record.acl_groups),
                    product_models=chunk_models,
                    entity_ids=entity_ids,
                    keywords=tokenize(d.text, extra_tokens=canonical_tokens(entity_ids)),
                    heading_keywords=tokenize(header),
                    content_hash=h,
                    simhash=simhash(d.text),
                    token_count=estimate_tokens(d.text),
                    metadata=meta,
                    title=record.title,
                    uri=record.uri,
                    doc_ref=record.ref,
                )
            )
        return chunks

    # ------------------------------------------------------------------ 批量向量化 + 写入
    def _flush(self, pending: list[_Pending], report: IngestReport) -> None:
        if not pending:
            return
        todo: list[Chunk] = []
        for p in pending:
            reuse = {} if p.is_new else self.store.chunk_embeddings(p.record.doc_id)
            for c in p.chunks:
                if c.chunk_id in reuse:
                    c.embedding = reuse[c.chunk_id]
                    report.embeddings_reused += 1
                else:
                    todo.append(c)
        try:
            if todo:
                vectors = self.embedder.embed_documents([c.embed_text() for c in todo])
                for c, v in zip(todo, vectors, strict=True):
                    c.embedding = v
                report.embeddings_computed += len(todo)
        except Exception as e:  # 模型服务挂了：这一批文档全部记为失败，下次重试
            log.exception("embedding batch failed")
            for p in pending:
                self._fail(p.record, f"embedding failed: {e!r}", report)
            pending.clear()
            return

        for p in pending:
            try:
                self.store.replace_document(p.record, p.content_hash, self.pipeline_version(), p.chunks)
                report.chunks_written += len(p.chunks)
                if p.is_new:
                    report.added += 1
                else:
                    report.updated += 1
            except Exception as e:
                log.exception("write failed", extra={"doc": p.record.ref})
                self._fail(p.record, f"write failed: {e!r}", report)
        pending.clear()

    def _fail(self, record: SourceRecord, error: str, report: IngestReport) -> None:
        report.failed += 1
        report.errors.append(f"{record.ref}: {error}")
        try:
            self.store.mark_document_failed(record, error)
        except Exception:
            log.exception("could not record failure", extra={"doc": record.ref})

    # ------------------------------------------------------------------ 删除
    def _delete_missing(self, source_id: str, seen: set[str], report: IngestReport, force: bool) -> None:
        existing = self.store.list_document_ids(source_id)
        missing = existing - seen
        if not missing:
            return
        ratio = len(missing) / len(existing)
        if len(existing) >= 10 and ratio > self.max_delete_ratio and not force:
            msg = (f"refusing to delete {len(missing)}/{len(existing)} docs ({ratio:.0%}) from {source_id}; "
                   "source may be unavailable. Re-run with --force-delete if this is intended.")
            log.error(msg)
            report.errors.append(msg)
            report.connector_stats["deletion_skipped"] = len(missing)
            return
        for doc_id in missing:
            if self.store.delete_document(doc_id):
                report.deleted += 1

    # ------------------------------------------------------------------ 主流程
    def run(self, connector: Connector, *, force_delete: bool = False) -> IngestReport:
        t0 = time.perf_counter()
        source_id = connector.source_id
        report = IngestReport(source_id=source_id)
        run_id = self.store.start_run(source_id)
        error: str | None = None
        try:
            # 1) 对象层先行：实体 / 关系入库，并刷新 linker，后续文档的实体链接才能用上
            ents, rels = connector.entities()
            if ents or rels:
                self.store.replace_entities(source_id, ents, rels)
                report.entities = len(ents)
                self.reload_linker()

            # 2) 文档层
            state = self.store.get_connector_state(source_id)
            pv = self.pipeline_version()
            seen: set[str] = set()
            pending: list[_Pending] = []
            for ev in connector.iter_events(state):
                if isinstance(ev, Tombstone):
                    if self.store.delete_document(ev.doc_id):
                        report.deleted += 1
                    continue
                seen.add(ev.doc_id)
                try:
                    parsed, h, models = self.prepare(ev)
                    st = self.store.get_document_state(ev.doc_id)
                    if st and st.status == "indexed" and st.content_hash == h and st.pipeline_version == pv:
                        report.unchanged += 1
                        continue
                    chunks = self.build_chunks(ev, parsed, models)
                    if not chunks:
                        raise ValueError("document produced no chunks (empty or unparseable)")
                    pending.append(_Pending(ev, h, chunks, is_new=st is None or st.content_hash is None))
                except Exception as e:
                    log.warning("document failed", extra={"doc": ev.ref, "error": repr(e)})
                    self._fail(ev, repr(e), report)
                    continue
                if sum(len(p.chunks) for p in pending) >= self.batch_chunks:
                    self._flush(pending, report)
            self._flush(pending, report)

            # 3) 全量快照型数据源：删除源头已不存在的文档
            if connector.full_snapshot:
                self._delete_missing(source_id, seen, report, force_delete)

            # 4) 只有整体跑完才保存游标（否则下次从旧游标重来，不会漏数据）
            new_state = connector.next_state()
            if new_state:
                self.store.set_connector_state(source_id, new_state)
            report.status = "partial" if report.failed else "succeeded"
        except Exception as e:
            log.exception("ingestion run failed", extra={"source": source_id})
            report.status = "failed"
            error = repr(e)
            report.errors.append(error)
        finally:
            report.connector_stats.update(connector.stats())
            report.duration_s = round(time.perf_counter() - t0, 3)
            self.store.finish_run(run_id, report.status, report.as_dict(), error)
        log.info("ingestion finished", extra={k: v for k, v in report.as_dict().items() if k != "errors"})
        return report

    def reload_linker(self) -> None:
        """从数据库重新加载对象层（实体 + 关系），并把领域词加进分词词典。"""
        from servicekb.processing.text import add_user_words

        ents, rels = self.store.load_entities()
        self.linker.reset(ents, rels)
        add_user_words(self.linker.domain_words())
