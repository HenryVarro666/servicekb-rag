"""内存存储：用于 demo 和单元测试。逻辑与 PostgresStore 保持一致，方便对照阅读。

关键词检索在这里实现了**标准 BM25**，是理解"稀疏检索"最直观的代码：

    score(q, d) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
    IDF(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

    - tf  : 词 t 在块 d 里出现的次数（出现越多越相关，但有饱和上限，由 k1 控制）
    - |d| : 块长度；avgdl：平均块长度（b 控制"长文档惩罚"的力度）
    - IDF : 越少见的词越重要。"逆变器"到处都是，IDF 很低；"E037"只出现在少数块里，IDF 很高。
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from servicekb.models import Chunk, Entity, EntityRelation, ImageRecord, SearchFilters, SourceRecord
from servicekb.store.base import DocumentState, IndexMismatchError, Store, chunk_visible

BM25_K1 = 1.2
BM25_B = 0.75


class MemoryStore(Store):
    def __init__(self) -> None:
        self.meta: dict[str, str] = {}
        self.docs: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, Chunk] = {}
        self.doc_chunks: dict[str, list[str]] = {}
        self.entities: dict[str, tuple[str, Entity]] = {}  # entity_id -> (source_id, entity)
        self.relations: list[tuple[str, EntityRelation]] = []
        self.connector_state: dict[str, dict[str, Any]] = {}
        self.runs: list[dict[str, Any]] = []
        self.query_logs: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self.images: dict[str, ImageRecord] = {}
        self._tf_cache: dict[str, Counter[str]] = {}

    # ------------------------------------------------------------------ 生命周期
    def init(self, embedding_model_id: str, dim: int, image_model_id: str | None = None,
             image_dim: int | None = None) -> None:
        if self.meta and self.meta.get("embedding_model") != embedding_model_id:
            raise IndexMismatchError(f"index built with {self.meta['embedding_model']}, now {embedding_model_id}")
        if image_model_id and self.meta.get("image_embedding_model", image_model_id) != image_model_id:
            raise IndexMismatchError(f"images indexed with {self.meta['image_embedding_model']}, now {image_model_id}")
        self.meta = {"embedding_model": embedding_model_id, "embedding_dim": str(dim)}
        if image_model_id:
            self.meta.update(image_embedding_model=image_model_id, image_embedding_dim=str(image_dim))

    # ------------------------------------------------------------------ 文档
    def get_document_state(self, doc_id: str) -> DocumentState | None:
        d = self.docs.get(doc_id)
        if not d:
            return None
        return DocumentState(doc_id, d["content_hash"], d["pipeline_version"], d["status"], d["attempts"])

    def chunk_embeddings(self, doc_id: str) -> dict[str, list[float]]:
        return {cid: self.chunks[cid].embedding for cid in self.doc_chunks.get(doc_id, []) if self.chunks[cid].embedding}

    def replace_document(self, record: SourceRecord, content_hash: str, pipeline_version: str, chunks: list[Chunk]) -> None:
        self._drop_chunks(record.doc_id)
        self.docs[record.doc_id] = {
            "record": record,
            "content_hash": content_hash,
            "pipeline_version": pipeline_version,
            "status": "indexed",
            "attempts": 0,
            "error": None,
            "indexed_at": datetime.now(timezone.utc),
        }
        for c in chunks:
            c.title, c.uri, c.doc_ref = record.title, record.uri, record.ref
            self.chunks[c.chunk_id] = c
            self._tf_cache[c.chunk_id] = self._term_freq(c)
        self.doc_chunks[record.doc_id] = [c.chunk_id for c in chunks]

    @staticmethod
    def _term_freq(c: Chunk) -> Counter[str]:
        """正文词计 1，标题词计 0.4（和 Postgres 里 ts_rank 的 A/B 权重大致对应）。"""
        tf: Counter[str] = Counter()
        for t in c.keywords:
            tf[t] += 1.0
        for t in c.heading_keywords:
            tf[t] += 0.4
        return tf

    def mark_document_failed(self, record: SourceRecord, error: str) -> None:
        d = self.docs.get(record.doc_id)
        if d is None:
            self.docs[record.doc_id] = {
                "record": record, "content_hash": None, "pipeline_version": None,
                "status": "failed", "attempts": 1, "error": error, "indexed_at": None,
            }
        else:
            d.update(status="failed", attempts=d["attempts"] + 1, error=error)

    def _drop_chunks(self, doc_id: str) -> None:
        for cid in self.doc_chunks.pop(doc_id, []):
            self.chunks.pop(cid, None)
            self._tf_cache.pop(cid, None)

    def delete_document(self, doc_id: str) -> bool:
        self._drop_chunks(doc_id)
        return self.docs.pop(doc_id, None) is not None

    def list_document_ids(self, source_id: str) -> set[str]:
        return {i for i, d in self.docs.items() if d["record"].source_id == source_id}

    # ------------------------------------------------------------------ 对象层
    def replace_entities(self, source_id: str, entities: list[Entity], relations: list[EntityRelation]) -> None:
        self.entities = {k: v for k, v in self.entities.items() if v[0] != source_id}
        self.entities.update({e.entity_id: (source_id, e) for e in entities})
        self.relations = [r for r in self.relations if r[0] != source_id] + [(source_id, r) for r in relations]

    def load_entities(self) -> tuple[list[Entity], list[EntityRelation]]:
        return [e for _, e in self.entities.values()], [r for _, r in self.relations]

    # ------------------------------------------------------------------ 检索
    def _visible(self, f: SearchFilters) -> list[Chunk]:
        return [c for c in self.chunks.values() if chunk_visible(c, f)]

    def dense_search(self, vector: list[float], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        cands = [c for c in self._visible(filters) if c.embedding is not None]
        if not cands:
            return []
        m = np.asarray([c.embedding for c in cands], dtype=np.float32)
        sims = m @ np.asarray(vector, dtype=np.float32)  # 向量已归一化：点积 = 余弦相似度
        order = np.argsort(-sims)[:k]
        return [(cands[i], float(sims[i])) for i in order]

    def keyword_search(self, tokens: list[str], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        q = set(tokens)
        if not q or not self.chunks:
            return []
        n = len(self.chunks)
        avgdl = sum(sum(tf.values()) for tf in self._tf_cache.values()) / n or 1.0
        df = {t: sum(1 for tf in self._tf_cache.values() if t in tf) for t in q}
        scored: list[tuple[Chunk, float]] = []
        for c in self._visible(filters):
            tf = self._tf_cache[c.chunk_id]
            dl = sum(tf.values()) or 1
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
                    s += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def entity_search(self, entity_weights: dict[str, float], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        scored = []
        for c in self._visible(filters):
            s = sum(entity_weights.get(e, 0.0) for e in c.entity_ids)
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def chunks_for_documents(self, doc_ids: list[str], filters: SearchFilters, limit: int = 20) -> list[Chunk]:
        out = [self.chunks[cid] for d in doc_ids for cid in self.doc_chunks.get(d, [])]
        return [c for c in out if chunk_visible(c, filters)][:limit]

    def token_document_frequency(self, tokens: list[str]) -> tuple[dict[str, int], int]:
        return {t: sum(1 for tf in self._tf_cache.values() if t in tf) for t in set(tokens)}, len(self.chunks)

    # ------------------------------------------------------------------ 图片
    def image_state(self, source_id: str) -> dict[str, str]:
        return {i: r.content_hash for i, r in self.images.items() if r.source_id == source_id}

    def upsert_images(self, records: list[ImageRecord]) -> None:
        self.images.update({r.image_id: r for r in records})

    def delete_images(self, image_ids: list[str]) -> int:
        return sum(1 for i in image_ids if self.images.pop(i, None) is not None)

    def image_search(self, vector: list[float], k: int, acl_groups: list[str]) -> list[tuple[ImageRecord, float]]:
        cands = [r for r in self.images.values() if r.embedding is not None and set(r.acl_groups) & set(acl_groups)]
        if not cands:
            return []
        sims = np.asarray([r.embedding for r in cands], dtype=np.float32) @ np.asarray(vector, dtype=np.float32)
        return [(cands[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    # ------------------------------------------------------------------ 运维
    def get_connector_state(self, source_id: str) -> dict[str, Any] | None:
        return self.connector_state.get(source_id)

    def set_connector_state(self, source_id: str, state: dict[str, Any]) -> None:
        self.connector_state[source_id] = state

    def start_run(self, source_id: str) -> int:
        self.runs.append({"source_id": source_id, "status": "running"})
        return len(self.runs)

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        self.runs[run_id - 1].update(status=status, stats=stats, error=error)

    def log_query(self, entry: dict[str, Any]) -> None:
        self.query_logs[str(entry["query_id"])] = {"created_at": datetime.now(timezone.utc), **entry}

    def prune_query_logs(self, older_than_days: int, keep_feedback: bool = True) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        with_feedback = {f["query_id"] for f in self.feedback}
        old = [q for q, e in self.query_logs.items()
               if e["created_at"] < cutoff and not (keep_feedback and q in with_feedback)]
        for q in old:
            del self.query_logs[q]
        self.feedback = [f for f in self.feedback if f["query_id"] in self.query_logs]
        return len(old)

    def log_feedback(self, query_id: str, rating: int, comment: str | None, corrected_answer: str | None) -> bool:
        if query_id not in self.query_logs:
            return False
        self.feedback.append({"id": str(uuid.uuid4()), "query_id": query_id, "rating": rating,
                              "comment": comment, "corrected_answer": corrected_answer})
        return True

    def stats(self) -> dict[str, Any]:
        by_status = Counter(d["status"] for d in self.docs.values())
        by_type = Counter(c.doc_type.value for c in self.chunks.values())
        return {"documents": dict(by_status), "chunks": len(self.chunks), "chunks_by_type": dict(by_type),
                "entities": len(self.entities), "relations": len(self.relations), "images": len(self.images),
                **self.meta}
