"""物料主数据（Parts Master / BOM 导出）。

这是"对象层"的主要来源。一行料号同时产出两样东西：
1. **实体**（Entity）+ **关系**（替代 / 互换 / 适用机型）→ 写入 entities / entity_relations 表，
   用于实体链接、查询扩展、结构化事实注入。
2. **文档**（每个料号一条记录，渲染成 "- 料号: ..." 文本）→ 可被检索、可被引用。
   借鉴 RAGFlow table 模板：一行一块。

真实数据里常见的问题，本 connector 都做了处理：
- 同一料号出现多行、名字不一样 → 合并，所有名字都作为别名，并记入 warnings（数据质量报告）
- 料号写法不统一（空格、小写）→ 规范化
- 替代料号用 ";"、"；"、"," 分隔 → 统一拆分

CSV 列（可通过 column_map 映射）::

    part_number,name_zh,name_en,category,applicable_models,status,substituted_by,
    interchangeable_with,interchange_note,aliases,notes
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from servicekb.connectors.base import Connector
from servicekb.models import DocType, Entity, EntityRelation, SourceEvent, SourceRecord
from servicekb.processing.text import normalize_text

RECORD_MIME = "application/x-servicekb-record+json"
_SPLIT = re.compile(r"[;；,，/|]")

DEFAULT_COLUMNS = {
    "part_number": "part_number",
    "name_zh": "name_zh",
    "name_en": "name_en",
    "category": "category",
    "applicable_models": "applicable_models",
    "status": "status",
    "substituted_by": "substituted_by",
    "interchangeable_with": "interchangeable_with",
    "interchange_note": "interchange_note",
    "aliases": "aliases",
    "notes": "notes",
}

STATUS_LABEL = {"active": "在售", "obsolete": "停产", "restricted": "限制使用"}


def _split(v: Any) -> list[str]:
    return [normalize_text(x) for x in _SPLIT.split(str(v or "")) if x.strip()]


def canonical_part_number(pn: str) -> str:
    return re.sub(r"\s+", "", normalize_text(pn)).upper()


class PartsMasterConnector(Connector):
    full_snapshot = True

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        acl_groups: list[str] | None = None,
        column_map: dict[str, str] | None = None,
    ) -> None:
        self.source_id = source_id
        self.path = Path(path)
        self.acl_groups = acl_groups or ["public"]
        self.cm = {**DEFAULT_COLUMNS, **(column_map or {})}
        self.warnings: list[str] = []
        self._parts: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ 读取 + 合并

    def _read_rows(self) -> list[dict[str, Any]]:
        if self.path.suffix.lower() == ".xlsx":
            from openpyxl import load_workbook

            ws = load_workbook(self.path, read_only=True, data_only=True).worksheets[0]
            rows = list(ws.iter_rows(values_only=True))
            header = [str(h).strip() for h in rows[0]]
            return [dict(zip(header, r, strict=False)) for r in rows[1:] if any(r)]
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._parts is not None:
            return self._parts
        cm = self.cm
        parts: dict[str, dict[str, Any]] = {}
        for row in self._read_rows():
            pn = canonical_part_number(str(row.get(cm["part_number"]) or ""))
            if not pn:
                continue
            name_zh = normalize_text(str(row.get(cm["name_zh"]) or ""))
            name_en = normalize_text(str(row.get(cm["name_en"]) or ""))
            if pn in parts:
                p = parts[pn]
                if name_zh and name_zh != p["name_zh"]:
                    self.warnings.append(f"duplicate part {pn}: names {p['name_zh']!r} vs {name_zh!r} (merged as alias)")
                    p["aliases"].append(name_zh)
                for key in ("applicable_models", "substituted_by", "interchangeable_with"):
                    p[key] = list(dict.fromkeys(p[key] + _split(row.get(cm[key]))))
                continue
            parts[pn] = {
                "part_number": pn,
                "name_zh": name_zh,
                "name_en": name_en,
                "category": normalize_text(str(row.get(cm["category"]) or "")),
                "applicable_models": _split(row.get(cm["applicable_models"])),
                "status": (str(row.get(cm["status"]) or "active").strip().lower()),
                "substituted_by": [canonical_part_number(x) for x in _split(row.get(cm["substituted_by"]))],
                "interchangeable_with": [
                    canonical_part_number(x) for x in _split(row.get(cm["interchangeable_with"]))
                ],
                "interchange_note": normalize_text(str(row.get(cm["interchange_note"]) or "")),
                "aliases": _split(row.get(cm["aliases"])),
                "notes": normalize_text(str(row.get(cm["notes"]) or "")),
            }
        for pn, p in parts.items():
            for ref in p["substituted_by"] + p["interchangeable_with"]:
                if ref not in parts:
                    self.warnings.append(f"part {pn} references unknown part {ref}")
        self._parts = parts
        return parts

    # ------------------------------------------------------------------ Connector 接口

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        for pn, p in self._load().items():
            doc = {
                "料号": pn,
                "名称": p["name_zh"],
                "英文名称": p["name_en"],
                "类别": p["category"],
                "适用机型": p["applicable_models"],
                "状态": STATUS_LABEL.get(p["status"], p["status"]),
                "替代料号": p["substituted_by"],
                "可互换料号": p["interchangeable_with"],
                "互换条件": p["interchange_note"],
                "别名": p["aliases"],
                "备注": p["notes"],
            }
            yield SourceRecord(
                source_id=self.source_id,
                external_id=pn,
                title=f"{pn} {p['name_zh']}".strip(),
                doc_type=DocType.PART,
                content=json.dumps(doc, ensure_ascii=False),
                mime_type=RECORD_MIME,
                uri=f"part://{pn}",
                metadata={"product_models_raw": p["applicable_models"], "part_status": p["status"]},
                acl_groups=list(self.acl_groups),
            )

    def entities(self) -> tuple[list[Entity], list[EntityRelation]]:
        ents: list[Entity] = []
        rels: list[EntityRelation] = []
        for pn, p in self._load().items():
            eid = f"part:{pn}"
            ents.append(
                Entity(
                    entity_id=eid,
                    entity_type="part",
                    canonical_name=pn,
                    display_name=p["name_zh"] or p["name_en"],
                    aliases=[a for a in [p["name_zh"], p["name_en"], *p["aliases"]] if a],
                    attributes={
                        "status": STATUS_LABEL.get(p["status"], p["status"]),
                        "category": p["category"],
                        "applicable_models": p["applicable_models"],
                        "interchange_note": p["interchange_note"],
                    },
                )
            )
            rels += [EntityRelation(eid, "substituted_by", f"part:{s}") for s in p["substituted_by"]]
            rels += [EntityRelation(eid, "interchangeable_with", f"part:{s}") for s in p["interchangeable_with"]]
            rels += [EntityRelation(eid, "fits", f"model:{m}") for m in p["applicable_models"]]
        return ents, rels

    def stats(self) -> dict[str, int]:
        return {"parts": len(self._load()), "warnings": len(self.warnings)}
