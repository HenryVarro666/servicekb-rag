"""实体链接（Entity Linking）：把文本里提到的型号、料号、故障码，映射到对象层里的唯一 ID。

为什么这一步对工业场景特别重要？
------------------------------------
1. **向量检索不擅长精确标识符**。"310-04521-A" 和 "310-04521-B" 在向量空间里几乎一样，
   但一个已停产、一个是替代件——答错了就是发错货。
2. **名字不统一**。同一个型号在工单里写成 "AS110K"、"as-110k-hv"、"110K HV"……
   用 ``normalize_key`` + 别名表把它们都映射到 ``model:AS-110K-HV``。
3. **把"记录"串成"对象"**。每个文档块都打上 entity_ids，查询 "E037" 时就能把手册章节、
   历史工单、相关料号全部拉出来——这就是检索里的第三路"实体召回"。

实现：正则抽候选 + 别名表查找（精确、可解释、零 LLM 成本）。
后续可升级为 NER 模型 / LLM 抽取，但**最终都要落到规范化 ID 上**（见 docs/08_limitations_roadmap.md）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from servicekb.models import Entity, EntityRelation
from servicekb.processing.text import CJK_CHAR, normalize_key

# 中文别名匹配前统一括号（全角 / 半角），长度不变，位置不受影响
_PARENS = str.maketrans({"（": "(", "）": ")"})

# 型号候选：字母前缀 + 数字 + K（功率）+ 可选后缀。例："AS-110K-HV"、"as 8k rs"、"AS110KHV"
# 这是示例数据的格式，**接入真实数据时必须按你们的型号命名规则改**。
_MODEL_CANDIDATE = re.compile(r"\b[A-Z]{2}[\s\-_]?\d{1,3}\s?K(?:[\s\-_]?(?:HV|RS|LV))?\b", re.IGNORECASE)


# 实体类型 → 检索时的权重（料号、故障码比型号更有区分度：一个型号会出现在成百上千个块里）
ENTITY_TYPE_WEIGHT = {"part": 1.0, "fault": 1.0, "model": 0.3}


def canonical_tokens(entity_ids: Iterable[str]) -> list[str]:
    """实体 ID → 额外的关键词 token。例如 model:AS-110K-HV → "as-110k-hv"。

    这样工单里写成 "AS110K" 的块，和问题里写成 "AS-110K-HV" 的查询，在关键词检索里也能对上。
    """
    return [eid.split(":", 1)[1].lower() for eid in entity_ids]


@dataclass(frozen=True)
class Mention:
    entity_id: str
    surface: str  # 原文里的写法
    known: bool  # 是否在对象层里登记过（未登记的料号要进数据质量报告）


class EntityLinker:
    """持有实体词典，负责：抽取 → 规范化 → 查表 → 返回实体 ID。"""

    def __init__(
        self,
        entities: Iterable[Entity] = (),
        relations: Iterable[EntityRelation] = (),
        part_number_pattern: str = r"\b\d{3}-\d{5}-[A-Z]\b",
        fault_code_pattern: str = r"\b[EWF]\d{3}\b",
    ) -> None:
        self.part_re = re.compile(part_number_pattern, re.IGNORECASE)
        self.fault_re = re.compile(fault_code_pattern, re.IGNORECASE)
        self.reset(entities, relations)

    # ------------------------------------------------------------------ 构建

    def reset(self, entities: Iterable[Entity] = (), relations: Iterable[EntityRelation] = ()) -> None:
        """原地替换全部词典。linker 被 pipeline 和 retriever 共享，原地更新保证两边一致。"""
        self.entities: dict[str, Entity] = {}
        self.alias_index: dict[str, str] = {}  # normalize_key(alias) -> entity_id
        self.cjk_aliases: list[tuple[str, str]] = []  # (中文别名, entity_id)，按长度降序
        self.relations: list[EntityRelation] = list(relations)
        self.add_entities(entities)

    def add_entities(self, entities: Iterable[Entity]) -> None:
        for e in entities:
            self.entities[e.entity_id] = e
            for alias in {e.canonical_name, e.display_name, *e.aliases}:
                if not alias:
                    continue
                if CJK_CHAR.search(alias):
                    # 中文别名（零件名称）：太短的（<4 字）容易误匹配，例如"风扇"会命中所有风扇
                    if len(alias) >= 4:
                        self.cjk_aliases.append((alias.translate(_PARENS), e.entity_id))
                else:
                    key = normalize_key(alias)
                    if len(key) >= 3:
                        self.alias_index.setdefault(key, e.entity_id)
        # 最长匹配优先："散热风扇组件(带防尘网)" 应优先于 "散热风扇组件"
        self.cjk_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    def domain_words(self) -> list[str]:
        """供 jieba 加载的领域词。"""
        return [a for a, _ in self.cjk_aliases]

    # ------------------------------------------------------------------ 抽取

    def extract(self, text: str) -> list[Mention]:
        """从文本中抽取实体提及（去重，保持出现顺序）。"""
        found: dict[str, Mention] = {}

        def add(entity_id: str, surface: str, known: bool) -> None:
            if entity_id not in found:
                found[entity_id] = Mention(entity_id, surface, known)

        for m in self.part_re.finditer(text):
            surface = m.group(0)
            eid = self.alias_index.get(normalize_key(surface))
            add(eid or f"part:{surface.upper()}", surface, eid is not None)

        for m in self.fault_re.finditer(text):
            code = m.group(0).upper()
            eid = self.alias_index.get(normalize_key(code))
            add(eid or f"fault:{code}", m.group(0), eid is not None)

        for m in _MODEL_CANDIDATE.finditer(text):
            eid = self.alias_index.get(normalize_key(m.group(0)))
            if eid:  # 型号只认登记过的，否则噪声太多
                add(eid, m.group(0), True)

        # 中文别名：最长匹配、不重叠
        text = text.translate(_PARENS)
        taken = [False] * len(text)
        for alias, eid in self.cjk_aliases:
            start = text.find(alias)
            while start != -1:
                end = start + len(alias)
                if not any(taken[start:end]):
                    for i in range(start, end):
                        taken[i] = True
                    add(eid, alias, True)
                start = text.find(alias, end)
        return list(found.values())

    def link(self, text: str) -> list[str]:
        return [m.entity_id for m in self.extract(text)]

    def resolve_model(self, raw: str) -> str | None:
        """把一个写法随意的型号字符串（例如工单里的 "as110k"）解析成规范化型号名。"""
        if not raw:
            return None
        eid = self.alias_index.get(normalize_key(raw))
        if eid and eid.startswith("model:"):
            return self.entities[eid].canonical_name
        return None

    # ------------------------------------------------------------------ 关系 / 事实

    def neighbors(self, entity_id: str, relations: Iterable[str]) -> list[EntityRelation]:
        rels = set(relations)
        return [r for r in self.relations if r.relation in rels and (r.src == entity_id or r.dst == entity_id)]

    def expand(self, entity_ids: Iterable[str]) -> list[str]:
        """查询扩展：料号 → 加上它的替代件 / 可互换件（双向）。

        用户问"310-04521-A 有货吗"，而新文档只写了 "310-04521-B"，不扩展就搜不到。
        """
        out = list(dict.fromkeys(entity_ids))
        for eid in list(out):
            for r in self.neighbors(eid, ("substituted_by", "interchangeable_with")):
                other = r.dst if r.src == eid else r.src
                if other not in out:
                    out.append(other)
        return out

    def facts_for(self, entity_ids: Iterable[str], only_inactive: bool = False) -> list[str]:
        """为 LLM 生成"结构化事实"（来自物料主数据，比文档里的描述更权威、更新）。

        例如：“料号 310-04521-A（散热风扇组件）状态：停产；已被 310-04521-B 替代。”
        这些事实会放在 prompt 的证据之前，避免 LLM 推荐已停产的零件。

        ``only_inactive=True``：只为停产 / 限制使用的料号生成事实（用于"证据里提到的料号"，
        避免把每个出现过的料号都塞进 prompt）。
        """
        facts: list[str] = []
        for eid in dict.fromkeys(entity_ids):
            e = self.entities.get(eid)
            if not e or e.entity_type != "part":
                continue
            attrs = e.attributes
            if only_inactive and attrs.get("status") in ("在售", "active"):
                continue
            line = f"料号 {e.canonical_name}（{e.display_name}）状态：{attrs.get('status', '未知')}"
            subs = [r.dst.split(":", 1)[1] for r in self.relations if r.src == eid and r.relation == "substituted_by"]
            replaced = [r.src.split(":", 1)[1] for r in self.relations if r.dst == eid and r.relation == "substituted_by"]
            inter = [
                (r.dst if r.src == eid else r.src).split(":", 1)[1]
                for r in self.neighbors(eid, ("interchangeable_with",))
            ]
            if subs:
                line += f"；已被 {', '.join(subs)} 替代"
            if replaced:
                line += f"；替代了旧料号 {', '.join(replaced)}"
            if inter:
                cond = attrs.get("interchange_note")
                line += f"；可与 {', '.join(inter)} 互换" + (f"（条件：{cond}）" if cond else "")
            models = attrs.get("applicable_models")
            if models:
                line += f"；适用机型：{', '.join(models)}"
            facts.append(line + "。")
        return facts
