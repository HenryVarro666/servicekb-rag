"""领域词典（YAML）：产品型号别名、故障码。

为什么要单独维护一份词典？
    型号别名（"AS110K" = "AS-110K-HV"）、故障码含义这类知识**不在任何一个系统里**，
    而是在老师傅脑子里。把它们沉淀成一份受版本控制的 YAML，是成本最低的"对象层"建设方式。
    业务人员可以直接改这个文件提 PR，不需要懂代码。

格式见 data/sample/dictionaries/entity_aliases.yaml。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from servicekb.connectors.base import Connector
from servicekb.models import Entity, EntityRelation, SourceEvent


class EntityDictionaryConnector(Connector):
    full_snapshot = True

    def __init__(self, source_id: str, path: str | Path, **_: Any) -> None:
        self.source_id = source_id
        self.path = Path(path)

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        return iter(())  # 词典只产出实体，不产出文档

    def entities(self) -> tuple[list[Entity], list[EntityRelation]]:
        data = yaml.safe_load(self.path.read_text("utf-8")) or {}
        ents: list[Entity] = []
        rels: list[EntityRelation] = []
        for m in data.get("product_models", []):
            ents.append(
                Entity(
                    entity_id=f"model:{m['name']}",
                    entity_type="product_model",
                    canonical_name=m["name"],
                    display_name=m.get("display_name", m["name"]),
                    aliases=m.get("aliases", []),
                    attributes={k: v for k, v in m.items() if k not in ("name", "aliases", "display_name")},
                )
            )
        for f in data.get("fault_codes", []):
            code = f["code"].upper()
            ents.append(
                Entity(
                    entity_id=f"fault:{code}",
                    entity_type="fault_code",
                    canonical_name=code,
                    display_name=f.get("name_zh", code),
                    aliases=[a for a in [f.get("name_zh"), f.get("name_en"), *f.get("aliases", [])] if a],
                    attributes={k: v for k, v in f.items() if k not in ("code", "aliases")},
                )
            )
            rels += [EntityRelation(f"fault:{code}", "related_fault", f"fault:{r}") for r in f.get("related", [])]
        return ents, rels
