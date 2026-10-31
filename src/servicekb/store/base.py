"""存储接口。

为什么要抽象一层？
    - 单元测试和新手 demo 用 ``MemoryStore``（不装数据库也能跑）；
    - 生产用 ``PostgresStore``；
    - 将来数据量大了想换 Milvus / Elasticsearch，只需实现这个接口，上层代码不动。
    （借鉴 Haystack 的 DocumentStore Protocol：接口越小越好替换）

三路检索接口返回的都是 ``list[tuple[Chunk, float]]``，按分数降序。分数的**量纲不同**
（余弦相似度 / BM25 / 实体命中权重），所以融合时只看排名，不看分数（RRF，见 retrieval/fusion.py）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from servicekb.models import Chunk, Entity, EntityRelation, SearchFilters, SourceRecord


@dataclass
class DocumentState:
    doc_id: str
    content_hash: str | None
    pipeline_version: str | None
    status: str
    attempts: int = 0


class IndexMismatchError(RuntimeError):
    """数据库里的向量是用另一个模型生成的。"""


class Store(ABC):
    # ------------------------------------------------------------------ 生命周期
    @abstractmethod
    def init(self, embedding_model_id: str, dim: int) -> None:
        """建表 / 迁移，并校验 index_meta 里的模型与当前配置一致。"""

    def close(self) -> None:  # noqa: B027  可选实现
        pass

    # ------------------------------------------------------------------ 文档
    @abstractmethod
    def get_document_state(self, doc_id: str) -> DocumentState | None: ...

    @abstractmethod
    def chunk_embeddings(self, doc_id: str) -> dict[str, list[float]]:
        """该文档现有块的 {chunk_id: embedding}，用于增量更新时复用向量（不重复计算）。"""

    @abstractmethod
    def replace_document(
        self, record: SourceRecord, content_hash: str, pipeline_version: str, chunks: list[Chunk]
    ) -> None:
        """**原子地**用新块替换文档的旧块。要么全部成功，要么保持原样——检索永远不会看到"半份文档"。"""

    @abstractmethod
    def mark_document_failed(self, record: SourceRecord, error: str) -> None:
        """记录失败但**保留旧块**：新版本解析失败时，用户仍能检索到旧版本。"""

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool: ...

    @abstractmethod
    def list_document_ids(self, source_id: str) -> set[str]: ...

    # ------------------------------------------------------------------ 对象层
    @abstractmethod
    def replace_entities(self, source_id: str, entities: list[Entity], relations: list[EntityRelation]) -> None: ...

    @abstractmethod
    def load_entities(self) -> tuple[list[Entity], list[EntityRelation]]: ...

    # ------------------------------------------------------------------ 检索
    @abstractmethod
    def dense_search(self, vector: list[float], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def keyword_search(self, tokens: list[str], k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def entity_search(
        self, entity_weights: dict[str, float], k: int, filters: SearchFilters
    ) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def chunks_for_documents(self, doc_ids: list[str], filters: SearchFilters, limit: int = 20) -> list[Chunk]: ...

    @abstractmethod
    def token_document_frequency(self, tokens: list[str]) -> tuple[dict[str, int], int]:
        """返回 ({token: 包含该 token 的块数}, 总块数)，用于计算 IDF（拒答判断用）。"""

    # ------------------------------------------------------------------ 运维
    @abstractmethod
    def get_connector_state(self, source_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def set_connector_state(self, source_id: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def start_run(self, source_id: str) -> int: ...

    @abstractmethod
    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None: ...

    @abstractmethod
    def log_query(self, entry: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_feedback(
        self, query_id: str, rating: int, comment: str | None, corrected_answer: str | None
    ) -> bool: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...


def chunk_visible(c: Chunk, f: SearchFilters) -> bool:
    """过滤条件的 Python 版本（MemoryStore 用；PostgresStore 在 SQL 里实现同样的语义）。"""
    if not set(c.acl_groups) & set(f.acl_groups):
        return False
    if f.doc_types and c.doc_type not in f.doc_types:
        return False
    if f.product_models and c.product_models and not set(c.product_models) & set(f.product_models):
        return False  # 注意：没有标注机型的块（通用内容）对所有机型可见
    return not (f.source_ids and c.doc_ref.split(":", 1)[0] not in f.source_ids)
