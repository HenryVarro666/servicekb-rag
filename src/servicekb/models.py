"""核心数据结构。

整条 RAG 链路里流动的就是这几种对象，先把它们看懂，后面的代码就好读了::

    数据源 ──Connector──▶ SourceRecord（原始记录：一份手册 / 一张工单 / 一行料号）
                │
             Parser
                ▼
          ParsedDocument（结构化内容：标题、段落、表格、问答对……= Block 列表）
                │
       清洗 → 切块 → 实体链接 → 向量化
                ▼
              Chunk（检索的最小单位，带向量、关键词、实体、权限）
                │
             存进 Store，查询时被召回
                ▼
           ScoredChunk（带各路检索分数的块）──▶ LLM ──▶ AnswerResult（答案 + 引用）

为什么用 dataclass 而不是 pydantic？
    内部对象量大（几十万个 Chunk），dataclass 更轻；对外 API 的入参 / 出参才用 pydantic 做校验。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class DocType(str, Enum):
    """文档类型。检索时可按类型过滤，生成时也会告诉 LLM 证据来自哪类文档。"""

    MANUAL = "manual"  # 产品手册 / 维修手册
    BULLETIN = "bulletin"  # 服务通告（Service Bulletin）：通常比手册更新、优先级更高
    QA = "qa"  # 历史问答 / 工单
    PART = "part"  # 物料主数据中的一个料号
    CHAT = "chat"  # 从 Teams 等聊天记录里提炼出的问答
    RECORD = "record"  # 维修记录 / 现场记录等
    OTHER = "other"


def stable_id(*parts: str, length: int = 32) -> str:
    """把若干字符串拼起来做 SHA-256，得到**确定性** ID。

    "确定性"是增量更新的基础：同一份文档无论第几次导入，ID 都一样，
    所以我们能判断"这份文档以前见过没有、内容变没变"。
    """
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- 原始记录


@dataclass
class SourceRecord:
    """Connector 从数据源读出的一条原始记录（还没解析）。"""

    source_id: str  # 数据源名称，对应 configs/sources.yaml 里的 name
    external_id: str  # 在源系统里的唯一标识：文件相对路径 / 工单号 / SharePoint item id / 料号
    title: str
    doc_type: DocType
    content: bytes | str  # 原始内容；文本类给 str，二进制文件（pdf/docx）给 bytes
    mime_type: str  # 决定用哪个 Parser，例如 "text/markdown"、"application/pdf"
    uri: str = ""  # 引用时给用户点击跳转的地址
    metadata: dict[str, Any] = field(default_factory=dict)
    acl_groups: list[str] = field(default_factory=lambda: ["public"])  # 谁能看到这份文档
    updated_at: datetime | None = None

    @property
    def doc_id(self) -> str:
        return stable_id(self.source_id, self.external_id)

    @property
    def ref(self) -> str:
        """人类可读的文档引用，形如 ``qa_archive:TK-2024-0103``。评估集里用它标注相关文档。"""
        return f"{self.source_id}:{self.external_id}"


@dataclass
class Tombstone:
    """"墓碑"：数据源明确告诉我们某条记录被删除了（例如 SharePoint delta 返回 deleted）。"""

    source_id: str
    external_id: str

    @property
    def doc_id(self) -> str:
        return stable_id(self.source_id, self.external_id)


SourceEvent = SourceRecord | Tombstone


# --------------------------------------------------------------------------- 解析结果

BlockKind = Literal["heading", "paragraph", "list", "table", "qa", "record"]


@dataclass
class Block:
    """文档里的一个结构单元。切块器（chunker）根据 Block 的类型决定怎么切。

    - heading  : 标题，``level`` 表示层级（1=一级标题）
    - paragraph: 正文段落
    - list     : 列表项
    - table    : 表格，``rows[0]`` 是表头
    - qa       : 一组问答（question/answer 放在 ``fields`` 里）
    - record   : 一条结构化记录（例如料号主数据的一行，键值对放在 ``fields`` 里）
    """

    kind: BlockKind
    text: str = ""
    level: int = 0
    page: int | None = None
    rows: list[list[str]] | None = None
    fields: dict[str, str] | None = None


@dataclass
class ParsedDocument:
    record: SourceRecord
    blocks: list[Block]
    # 解析器从文档里读到的额外元数据（例如 Markdown front matter 里的 product_models）
    metadata: dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        parts: list[str] = []
        for b in self.blocks:
            if b.kind == "table" and b.rows:
                parts.extend(" | ".join(r) for r in b.rows)
            elif b.fields:
                parts.extend(f"{k}: {v}" for k, v in b.fields.items())
            else:
                parts.append(b.text)
        return "\n".join(parts)


# --------------------------------------------------------------------------- 检索单元


@dataclass
class Chunk:
    """检索的最小单位。

    几个容易混淆的文本字段：
    - ``text``       : 块正文。展示给用户、送给 LLM、做引用片段。
    - ``heading_path``: 所在章节路径，例如 ["AS-110K-HV 维修手册", "4 故障排除", "4.1 E037 绝缘阻抗低"]。
    - 向量化时用的是 ``embed_text()`` = 标题路径 + 正文（"上下文化"的块，召回效果明显更好），
      但标题**不**写进 ``text``，避免污染引用片段（借鉴 docling 的 contextualize 思路）。
    """

    chunk_id: str
    doc_id: str
    ordinal: int  # 在文档内的顺序
    text: str
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    doc_type: DocType = DocType.OTHER
    acl_groups: list[str] = field(default_factory=lambda: ["public"])
    product_models: list[str] = field(default_factory=list)  # 适用机型（规范化后的型号）
    entity_ids: list[str] = field(default_factory=list)  # 关联的对象：part:xxx / fault:E037 / model:xxx
    keywords: list[str] = field(default_factory=list)  # 正文分词结果（关键词检索用）
    heading_keywords: list[str] = field(default_factory=list)  # 标题路径分词结果（权重较低）
    content_hash: str = ""
    simhash: int = 0  # 用于近似去重
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    # 以下字段来自所属文档，检索结果展示 / 引用时使用（存储层 join 回来）
    title: str = ""
    uri: str = ""
    doc_ref: str = ""

    def embed_text(self) -> str:
        header = " > ".join(self.heading_path)
        return f"{header}\n{self.text}" if header else self.text


@dataclass
class SearchFilters:
    """检索过滤条件。``acl_groups`` 是**强制**的：用户只能检索到自己所在组可见的块。"""

    acl_groups: list[str]
    doc_types: list[DocType] | None = None
    product_models: list[str] | None = None  # 硬过滤：只要这些型号 + 通用（无型号）的块
    source_ids: list[str] | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float = 0.0  # 最终排序分数（融合分 或 重排分）
    channel_scores: dict[str, float] = field(default_factory=dict)  # 各路原始分数
    channel_ranks: dict[str, int] = field(default_factory=dict)  # 各路排名（从 1 开始）
    rerank_score: float | None = None
    duplicates: list[str] = field(default_factory=list)  # 被折叠掉的重复块 doc_ref


# --------------------------------------------------------------------------- 对象层（实体）


@dataclass
class Entity:
    """对象层：把"按记录存"的数据整理成"按对象存"。

    现场数据常见问题：同一个零件在不同表里叫法不同、料号写法不一致（"310-04521-B" /
    "31004521B" / "310 04521 B"）。实体层给每个对象一个**唯一 ID** 和一组**别名**，
    文档块通过 entity_ids 关联到对象，从而把散落的记录串起来。
    """

    entity_id: str  # 形如 "part:310-04521-B"、"model:AS-110K-HV"、"fault:E037"
    entity_type: Literal["part", "product_model", "fault_code"]
    canonical_name: str
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityRelation:
    """对象之间的关系，例如 part:A --substituted_by--> part:B，part:X --fits--> model:Y。

    这是知识图谱的雏形：先用一张关系表把最重要的关系存下来，等真需要多跳推理时再上图数据库。
    """

    src: str
    relation: Literal["substituted_by", "interchangeable_with", "fits", "related_fault"]
    dst: str
    attributes: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- 回答


@dataclass
class Citation:
    index: int  # 答案里的 [n]
    chunk_id: str
    doc_ref: str
    title: str
    uri: str
    heading_path: list[str]
    page: int | None
    snippet: str


@dataclass
class AnswerResult:
    query_id: str
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    # escalate=True 表示知识库证据不足，应转人工 / 升级处理（对应"未回复请求的跟踪升级"）
    escalate: bool = False
    reason: str = ""
    model: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)  # 从对象层注入的结构化事实（例如替代件关系）
