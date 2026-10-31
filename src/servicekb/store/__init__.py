"""存储层。用 ``build_store(settings)`` 按配置构造。"""

from __future__ import annotations

from servicekb.config import Settings
from servicekb.store.base import DocumentState, IndexMismatchError, Store
from servicekb.store.memory import MemoryStore


def build_store(s: Settings) -> Store:
    if s.store_backend == "memory":
        return MemoryStore()
    from servicekb.store.postgres import PostgresStore

    return PostgresStore(s.database_url, s.db_pool_min_size, s.db_pool_max_size, s.hnsw_ef_search)


__all__ = ["DocumentState", "IndexMismatchError", "MemoryStore", "Store", "build_store"]
