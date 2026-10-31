"""根据 configs/sources.yaml 构造 connector。

sources.yaml 示例::

    sources:
      - name: manuals            # = source_id，入库后不要改名（改名 = 全部重建）
        type: filesystem
        path: data/sample/manuals
        doc_type: manual
        acl_groups: [public]

**顺序很重要**：词典、物料主数据（对象层）要排在文档前面，这样文档入库时实体链接才能用上最新的对象。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from servicekb.connectors.base import Connector
from servicekb.connectors.dictionary import EntityDictionaryConnector
from servicekb.connectors.filesystem import FilesystemConnector
from servicekb.connectors.parts_master import PartsMasterConnector
from servicekb.connectors.qa_archive import QAArchiveConnector
from servicekb.connectors.sharepoint import SharePointConnector

CONNECTOR_TYPES: dict[str, type[Connector]] = {
    "filesystem": FilesystemConnector,
    "qa_archive": QAArchiveConnector,
    "parts_master": PartsMasterConnector,
    "entity_dictionary": EntityDictionaryConnector,
    "sharepoint": SharePointConnector,
}


def register_connector(type_name: str, cls: type[Connector]) -> None:
    """注册自定义 connector（例如 chat 包里的 Teams 导出）。"""
    CONNECTOR_TYPES[type_name] = cls


def build_connector(spec: dict[str, Any], base_dir: Path | None = None) -> Connector:
    spec = dict(spec)
    type_name = spec.pop("type")
    name = spec.pop("name")
    spec.pop("enabled", None)
    if "path" in spec and base_dir is not None and not Path(spec["path"]).is_absolute():
        spec["path"] = str(base_dir / spec["path"])
    try:
        cls = CONNECTOR_TYPES[type_name]
    except KeyError as e:
        raise ValueError(f"unknown connector type {type_name!r}; known: {sorted(CONNECTOR_TYPES)}") from e
    return cls(source_id=name, **spec)


def load_sources(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    return [s for s in data.get("sources", []) if s.get("enabled", True)]


__all__ = ["CONNECTOR_TYPES", "Connector", "build_connector", "load_sources", "register_connector"]
