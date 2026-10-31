"""PostgreSQL + pgvector 存储（生产用）。

几个值得注意的实现细节
----------------------
1. **中文全文检索**：Postgres 内置的分词器不支持中文（会把一整句中文当成一个词）。常见方案是装
   zhparser / pg_jieba 扩展，但很多托管数据库（RDS 等）装不了。本项目在 **Python 侧用 jieba 分好词**，
   直接构造 tsvector 字面量写入（``'绝缘':1A '阻抗':2A``），查询时也构造 tsquery 字面量，
   绕开 Postgres 的分词器。优点：零扩展、分词逻辑和 MemoryStore 完全一致。
   局限：``ts_rank`` 不是真正的 BM25（没有 IDF），见 docs/03_database.md 的改进方案。

2. **HNSW + 过滤**：带 WHERE 条件的向量查询，HNSW 是"先找近邻、再过滤"，过滤掉太多时
   返回结果会不足 k 条。pgvector >= 0.8 支持 ``hnsw.iterative_scan``，会自动继续扫描；
   本模块检测到 0.8+ 时自动开启。

3. **原子替换**：``replace_document`` 在一个事务里完成"更新文档行 + 删除过期块 + 插入新块"，
   检索永远看不到半份文档。

4. **迁移**：``migrations/*.sql`` 按文件名顺序执行，``schema_migrations`` 表记录已执行的版本；
   用 advisory lock 防止多个进程同时迁移。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from servicekb.logging import get_logger
from servicekb.models import Chunk, DocType, Entity, EntityRelation, SearchFilters, SourceRecord
from servicekb.store.base import DocumentState, IndexMismatchError, Store

log = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_LOCK_ID = 73_100_001  # 任意常量，用于 pg_advisory_xact_lock

_CHUNK_COLS = (
    "c.chunk_id, c.doc_id, c.ordinal, c.text, c.heading_path, c.page, c.doc_type, c.acl_groups, "
    "c.product_models, c.entity_ids, c.content_hash, c.simhash, c.token_count, c.metadata, "
    "d.title, d.uri, d.source_id, d.external_id"
)


# --------------------------------------------------------------------------- tsvector / tsquery 字面量


def _quote_lexeme(t: str) -> str:
    return "'" + t.replace("\\", "\\\\").replace("'", "''") + "'"


def build_tsvector_literal(body_tokens: list[str], heading_tokens: list[str]) -> str:
    """构造 tsvector 字面量。正文词权重 A，标题词权重 B（标题词出现在每个块里，权重低一些）。

    形如 ``'e037':3A,9A '绝缘':4A '故障排除':12B``。位置上限 16383（Postgres 限制）。
    """
    positions: dict[str, list[str]] = {}
    pos = 0
    for weight, toks in (("A", body_tokens), ("B", heading_tokens)):
        for t in toks:
            pos = min(pos + 1, 16383)
            positions.setdefault(t, []).append(f"{pos}{weight}")
    return " ".join(f"{_quote_lexeme(t)}:{','.join(p[:256])}" for t, p in positions.items())


def build_tsquery_literal(tokens: list[str]) -> str:
    """OR 语义：任意词命中即可召回，由 ts_rank 排序。"""
    return " | ".join(_quote_lexeme(t) for t in dict.fromkeys(tokens))


# --------------------------------------------------------------------------- Store


class PostgresStore(Store):
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10, ef_search: int = 100) -> None:
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self.dsn = dsn
        self.ef_search = ef_search
        # 先确保扩展存在，否则 register_vector 找不到 vector 类型
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        ver = tuple(int(x) for x in row[0].split(".")[:2]) if row else (0, 0)
        self.iterative_scan = ver >= (0, 8)
        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, configure=register_vector, open=True)

    def close(self) -> None:
        self.pool.close()

    # ------------------------------------------------------------------ 迁移 + 元信息
    def init(self, embedding_model_id: str, dim: int) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            applied = dict(conn.execute("SELECT version, checksum FROM schema_migrations").fetchall())
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                template = path.read_text("utf-8")
                checksum = hashlib.sha256(template.encode()).hexdigest()
                if path.stem in applied:
                    if applied[path.stem] != checksum:
                        log.warning("migration file changed after being applied", extra={"version": path.stem})
                    continue
                log.info("applying migration", extra={"version": path.stem})
                conn.execute(template.replace("{{EMBEDDING_DIM}}", str(int(dim))))
                conn.execute("INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)", (path.stem, checksum))

            meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            if not meta:
                conn.execute(
                    "INSERT INTO index_meta (key, value) VALUES ('embedding_model', %s), ('embedding_dim', %s)",
                    (embedding_model_id, str(dim)),
                )
            elif meta.get("embedding_model") != embedding_model_id or meta.get("embedding_dim") != str(dim):
                raise IndexMismatchError(
                    f"database was built with {meta.get('embedding_model')} (dim={meta.get('embedding_dim')}), "
                    f"but config says {embedding_model_id} (dim={dim}). "
                    "Changing the embedding model requires a full rebuild: `servicekb db reset` then re-ingest."
                )

    def reset(self) -> None:
        """删除所有表（危险！只在换向量模型或开发时使用）。"""
        with self.pool.connection() as conn:
            conn.execute(
                "DROP TABLE IF EXISTS feedback, query_logs, ingestion_runs, connector_state, entity_relations, "
                "entities, chunks, documents, index_meta, schema_migrations CASCADE"
            )

    # ------------------------------------------------------------------ 过滤条件 → SQL
    @staticmethod
    def _filter_sql(f: SearchFilters) -> tuple[str, list[Any]]:
        clauses = ["c.acl_groups && %s::text[]"]
        params: list[Any] = [list(f.acl_groups)]
        if f.doc_types:
            clauses.append("c.doc_type = ANY(%s::text[])")
            params.append([d.value for d in f.doc_types])
        if f.product_models:
            clauses.append("(c.product_models && %s::text[] OR c.product_models = '{}')")
            params.append(list(f.product_models))
        if f.source_ids:
            clauses.append("c.doc_id IN (SELECT doc_id FROM documents WHERE source_id = ANY(%s::text[]))")
            params.append(list(f.source_ids))
        return " AND ".join(clauses), params

    @staticmethod
    def _row_to_chunk(r: tuple[Any, ...]) -> Chunk:
        return Chunk(
            chunk_id=r[0], doc_id=r[1], ordinal=r[2], text=r[3], heading_path=list(r[4] or []), page=r[5],
            doc_type=DocType(r[6]), acl_groups=list(r[7]), product_models=list(r[8]), entity_ids=list(r[9]),
            content_hash=r[10], simhash=int(r[11]), token_count=r[12], metadata=r[13] or {},
            title=r[14], uri=r[15], doc_ref=f"{r[16]}:{r[17]}",
        )

    def _search(self, hits_sql: str, params: list[Any], order: str) -> list[tuple[Chunk, float]]:
        sql = (
            f"WITH hits AS ({hits_sql}) "
            f"SELECT {_CHUNK_COLS}, hits.score FROM hits "
            "JOIN chunks c ON c.chunk_id = hits.chunk_id JOIN documents d ON d.doc_id = c.doc_id "
            f"ORDER BY hits.score {order}, c.chunk_id"
        )
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(self.ef_search),))
            if self.iterative_scan:
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            rows = conn.execute(sql, params).fetchall()
        return [(self._row_to_chunk(r[:-1]), float(r[-1])) for r in rows]

    # ------------------------------------------------------------------ 三路检索
    def dense_search(self, vector: list[float], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        where, params = self._filter_sql(filters)
        vec = np.asarray(vector, dtype=np.float32)
        hits = (
            "SELECT c.chunk_id, 1 - (c.embedding <=> %s) AS score FROM chunks c "
            f"WHERE {where} ORDER BY c.embedding <=> %s LIMIT %s"
        )
        return self._search(hits, [vec, *params, vec, k], "DESC")

    def keyword_search(self, tokens: list[str], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        if not tokens:
            return []
        where, params = self._filter_sql(filters)
        hits = (
            "SELECT c.chunk_id, ts_rank(c.search_tsv, q.query, 1) AS score "
            "FROM chunks c, (SELECT %s::tsquery AS query) q "
            f"WHERE c.search_tsv @@ q.query AND {where} ORDER BY score DESC, c.chunk_id LIMIT %s"
        )
        return self._search(hits, [build_tsquery_literal(tokens), *params, k], "DESC")

    def entity_search(self, entity_weights: dict[str, float], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        if not entity_weights:
            return []
        where, params = self._filter_sql(filters)
        ids, weights = list(entity_weights), [float(w) for w in entity_weights.values()]
        hits = (
            "SELECT c.chunk_id, sum(q.w) AS score FROM chunks c "
            "JOIN unnest(%s::text[], %s::float8[]) AS q(eid, w) ON q.eid = ANY(c.entity_ids) "
            f"WHERE c.entity_ids && %s::text[] AND {where} "
            "GROUP BY c.chunk_id ORDER BY score DESC, c.chunk_id LIMIT %s"
        )
        return self._search(hits, [ids, weights, ids, *params, k], "DESC")

    def chunks_for_documents(self, doc_ids: list[str], filters: SearchFilters, limit: int = 20) -> list[Chunk]:
        if not doc_ids:
            return []
        where, params = self._filter_sql(filters)
        sql = (
            f"SELECT {_CHUNK_COLS} FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.doc_id = ANY(%s::text[]) AND {where} ORDER BY c.doc_id, c.ordinal LIMIT %s"
        )
        with self.pool.connection() as conn:
            rows = conn.execute(sql, [doc_ids, *params, limit]).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def token_document_frequency(self, tokens: list[str]) -> tuple[dict[str, int], int]:
        toks = list(dict.fromkeys(tokens))
        with self.pool.connection() as conn:
            total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if not toks:
                return {}, total
            rows = conn.execute(
                "SELECT t, (SELECT count(*) FROM chunks c WHERE c.search_tsv @@ quote_literal(t)::tsquery) "
                "FROM unnest(%s::text[]) AS t",
                [toks],
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}, int(total)

    # ------------------------------------------------------------------ 文档
    def get_document_state(self, doc_id: str) -> DocumentState | None:
        with self.pool.connection() as conn:
            r = conn.execute(
                "SELECT doc_id, content_hash, pipeline_version, status, attempts FROM documents WHERE doc_id = %s",
                (doc_id,),
            ).fetchone()
        return DocumentState(*r) if r else None

    def chunk_embeddings(self, doc_id: str) -> dict[str, list[float]]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT chunk_id, embedding FROM chunks WHERE doc_id = %s", (doc_id,)).fetchall()
        # pgvector 的 psycopg 适配器返回 numpy 数组或 Vector 对象（取决于版本），统一转成 list[float]
        return {r[0]: (r[1].to_list() if hasattr(r[1], "to_list") else np.asarray(r[1], dtype=np.float32).tolist())
                for r in rows}

    def replace_document(self, record: SourceRecord, content_hash: str, pipeline_version: str, chunks: list[Chunk]) -> None:
        doc_id = record.doc_id
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO documents (doc_id, source_id, external_id, title, doc_type, uri, acl_groups, metadata,
                    content_hash, pipeline_version, status, attempts, error, source_updated_at, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'indexed', 0, NULL, %s, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, doc_type = EXCLUDED.doc_type, uri = EXCLUDED.uri,
                    acl_groups = EXCLUDED.acl_groups, metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash, pipeline_version = EXCLUDED.pipeline_version,
                    status = 'indexed', attempts = 0, error = NULL,
                    source_updated_at = EXCLUDED.source_updated_at, indexed_at = now(), updated_at = now()
                """,
                (doc_id, record.source_id, record.external_id, record.title, record.doc_type.value, record.uri,
                 record.acl_groups, Jsonb(record.metadata), content_hash, pipeline_version, record.updated_at),
            )
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = %s AND NOT (chunk_id = ANY(%s::text[]))",
                (doc_id, [c.chunk_id for c in chunks]),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, ordinal, text, heading_path, page, doc_type, acl_groups,
                        product_models, entity_ids, content_hash, simhash, token_count, metadata, embedding, search_tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::tsvector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        ordinal = EXCLUDED.ordinal, text = EXCLUDED.text, heading_path = EXCLUDED.heading_path,
                        page = EXCLUDED.page, doc_type = EXCLUDED.doc_type, acl_groups = EXCLUDED.acl_groups,
                        product_models = EXCLUDED.product_models, entity_ids = EXCLUDED.entity_ids,
                        simhash = EXCLUDED.simhash, token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, search_tsv = EXCLUDED.search_tsv
                    """,
                    [
                        (c.chunk_id, doc_id, c.ordinal, c.text, c.heading_path, c.page, c.doc_type.value,
                         c.acl_groups, c.product_models, c.entity_ids, c.content_hash, c.simhash, c.token_count,
                         Jsonb(c.metadata), np.asarray(c.embedding, dtype=np.float32),
                         build_tsvector_literal(c.keywords, c.heading_keywords))
                        for c in chunks
                    ],
                )

    def mark_document_failed(self, record: SourceRecord, error: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, source_id, external_id, title, doc_type, uri, acl_groups, metadata,
                    status, attempts, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'failed', 1, %s)
                ON CONFLICT (doc_id) DO UPDATE SET status = 'failed', attempts = documents.attempts + 1,
                    error = EXCLUDED.error, updated_at = now()
                """,
                (record.doc_id, record.source_id, record.external_id, record.title, record.doc_type.value,
                 record.uri, record.acl_groups, Jsonb(record.metadata), error[:4000]),
            )

    def delete_document(self, doc_id: str) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,)).rowcount > 0

    def list_document_ids(self, source_id: str) -> set[str]:
        with self.pool.connection() as conn:
            return {r[0] for r in conn.execute("SELECT doc_id FROM documents WHERE source_id = %s", (source_id,))}

    # ------------------------------------------------------------------ 对象层
    def replace_entities(self, source_id: str, entities: list[Entity], relations: list[EntityRelation]) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM entity_relations WHERE source_id = %s", (source_id,))
            conn.execute("DELETE FROM entities WHERE source_id = %s", (source_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO entities (entity_id, entity_type, canonical_name, display_name, aliases, attributes, source_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE SET entity_type = EXCLUDED.entity_type,
                        canonical_name = EXCLUDED.canonical_name, display_name = EXCLUDED.display_name,
                        aliases = EXCLUDED.aliases, attributes = EXCLUDED.attributes,
                        source_id = EXCLUDED.source_id, updated_at = now()
                    """,
                    [(e.entity_id, e.entity_type, e.canonical_name, e.display_name, e.aliases, Jsonb(e.attributes),
                      source_id) for e in entities],
                )
                cur.executemany(
                    "INSERT INTO entity_relations (src_entity_id, relation, dst_entity_id, attributes, source_id) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    [(r.src, r.relation, r.dst, Jsonb(r.attributes), source_id) for r in relations],
                )

    def load_entities(self) -> tuple[list[Entity], list[EntityRelation]]:
        with self.pool.connection() as conn:
            ents = [
                Entity(r[0], r[1], r[2], r[3], list(r[4]), r[5] or {})
                for r in conn.execute(
                    "SELECT entity_id, entity_type, canonical_name, display_name, aliases, attributes FROM entities"
                )
            ]
            rels = [
                EntityRelation(r[0], r[1], r[2], r[3] or {})
                for r in conn.execute("SELECT src_entity_id, relation, dst_entity_id, attributes FROM entity_relations")
            ]
        return ents, rels

    # ------------------------------------------------------------------ 运维
    def get_connector_state(self, source_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            r = conn.execute("SELECT state FROM connector_state WHERE source_id = %s", (source_id,)).fetchone()
        return r[0] if r else None

    def set_connector_state(self, source_id: str, state: dict[str, Any]) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO connector_state (source_id, state) VALUES (%s, %s) "
                "ON CONFLICT (source_id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
                (source_id, Jsonb(state)),
            )

    def start_run(self, source_id: str) -> int:
        with self.pool.connection() as conn:
            return int(conn.execute(
                "INSERT INTO ingestion_runs (source_id, status) VALUES (%s, 'running') RETURNING run_id", (source_id,)
            ).fetchone()[0])

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET status = %s, stats = %s, error = %s, finished_at = now() WHERE run_id = %s",
                (status, Jsonb(stats), error, run_id),
            )

    def log_query(self, entry: dict[str, Any]) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (query_id, question, user_groups, filters, retrieved, answer, citations,
                    escalate, model, prompt_version, timings_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (entry["query_id"], entry["question"], entry.get("user_groups", []), Jsonb(entry.get("filters", {})),
                 Jsonb(entry.get("retrieved", [])), entry.get("answer"), Jsonb(entry.get("citations", [])),
                 entry.get("escalate", False), entry.get("model"), entry.get("prompt_version"),
                 Jsonb(entry.get("timings_ms", {}))),
            )

    def log_feedback(self, query_id: str, rating: int, comment: str | None, corrected_answer: str | None) -> bool:
        with self.pool.connection() as conn:
            exists = conn.execute("SELECT 1 FROM query_logs WHERE query_id = %s", (query_id,)).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO feedback (query_id, rating, comment, corrected_answer) VALUES (%s, %s, %s, %s)",
                (query_id, rating, comment, corrected_answer),
            )
        return True

    def stats(self) -> dict[str, Any]:
        with self.pool.connection() as conn:
            docs = dict(conn.execute("SELECT status, count(*) FROM documents GROUP BY status").fetchall())
            by_type = dict(conn.execute("SELECT doc_type, count(*) FROM chunks GROUP BY doc_type").fetchall())
            n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            n_ent = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
            n_rel = conn.execute("SELECT count(*) FROM entity_relations").fetchone()[0]
            meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
            size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'))").fetchone()[0]
        return {"documents": docs, "chunks": n_chunks, "chunks_by_type": by_type, "entities": n_ent,
                "relations": n_rel, "chunks_table_size": size, **meta}
