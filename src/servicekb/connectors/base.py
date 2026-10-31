"""Connector 接口：负责"从哪里读数据"，不关心"怎么处理"。

每种数据源一个 Connector。pipeline 只和这个接口打交道，所以新增数据源（Jira、ServiceNow、
企业微信……）只需要实现一个新类，并在 connectors/__init__.py 的 ``build_connector`` 里注册。

两种同步语义（**很重要，决定了删除逻辑**）：

- ``full_snapshot = True``：每次 ``iter_events`` 返回数据源的**全量**（例如扫描整个文件夹）。
  本次没出现的文档 = 已在源头删除 → pipeline 会删除它们（有安全阈值保护，见 max_delete_ratio）。
- ``full_snapshot = False``：增量（例如 SharePoint delta 查询）。只返回变化的记录，
  删除必须由数据源显式给出 ``Tombstone``。**绝不能**因为"本批次没出现"就删除。
  （LlamaIndex 的 UPSERTS_AND_DELETE 在增量场景下会误删数据，这是一个常见坑。）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from servicekb.models import Entity, EntityRelation, SourceEvent


class Connector(ABC):
    source_id: str
    full_snapshot: bool = True

    @abstractmethod
    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        """产出 SourceRecord（新增 / 可能修改）或 Tombstone（删除）。

        ``state`` 是上次成功运行后保存的游标（例如 SharePoint 的 deltaLink），首次运行为 None。
        """

    def next_state(self) -> dict[str, Any] | None:
        """本次运行**成功结束后**要保存的新游标。默认无状态。"""
        return None

    def entities(self) -> tuple[list[Entity], list[EntityRelation]]:
        """对象层数据（实体 + 关系）。只有物料主数据、词典这类数据源需要实现。"""
        return [], []

    def stats(self) -> dict[str, int]:
        """Connector 自己的统计（例如被质量门槛过滤掉的工单数），会写进 ingestion_runs。"""
        return {}
