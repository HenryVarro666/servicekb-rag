"""聊天记录清洗用到的数据结构。

一条 Teams 消息从原始 JSON 到知识库问答对，会经历::

    Graph chatMessage JSON ──解析──▶ ChatMessage ──过滤/清洗──▶ ChatMessage(text 已清洗)
        ──合并连发 / 组装线程──▶ Thread ──质量评估 + 抽取──▶ QAPair ──▶ teams_qa.jsonl ──▶ 知识库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Role = Literal["engineer", "field_tech", "service_partner", "customer", "bot", "unknown"]


@dataclass
class ChatMessage:
    key: str  # 复合主键：conversation_id + message id（Graph 的 id 只在一个对话内唯一！）
    message_id: str
    conversation_id: str  # 频道消息：teamId/channelId；群聊：chatId
    reply_to: str | None  # 频道回复：父消息 id；群聊没有
    created_at: datetime
    author_id: str
    author_name: str
    role: Role
    html: str  # 原始 body.content
    text: str = ""  # 清洗后的纯文本
    message_type: str = "message"
    deleted: bool = False
    edited: bool = False
    is_bot: bool = False
    policy_violation: bool = False
    web_url: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reactions: list[dict[str, Any]] = field(default_factory=list)
    quoted_message_ids: list[str] = field(default_factory=list)  # 本消息引用（回复）了哪些消息
    merged_ids: list[str] = field(default_factory=list)  # 连发合并时被并进来的消息


@dataclass
class Thread:
    thread_id: str
    conversation_id: str
    messages: list[ChatMessage]
    kind: Literal["channel_thread", "chat_session"]

    @property
    def root(self) -> ChatMessage:
        return self.messages[0]

    @property
    def started_at(self) -> datetime:
        return self.messages[0].created_at

    def transcript(self, with_ids: bool = False) -> str:
        lines = []
        for m in self.messages:
            prefix = f"[{m.message_id}] " if with_ids else ""
            lines.append(f"{prefix}{m.author_name}（{m.role}）: {m.text}")
        return "\n".join(lines)


@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    status: Literal["resolved", "answered_unconfirmed", "unresolved"]
    confidence: float
    created_at: str
    product_model: str = ""
    fault_codes: list[str] = field(default_factory=list)
    part_numbers: list[str] = field(default_factory=list)
    evidence_message_ids: list[str] = field(default_factory=list)
    source_url: str = ""
    conversation_id: str = ""
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    extractor: str = "rule-v1"
    duplicates: list[str] = field(default_factory=list)

    def to_archive_row(self) -> dict[str, Any]:
        """转成 QAArchiveConnector 能直接读的格式（和工单归档共用同一套入库逻辑）。"""
        return {
            "ticket_id": self.qa_id,
            "created_at": self.created_at,
            "product_model": self.product_model,
            "subject": "",
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "tags": self.fault_codes,
            "source_url": self.source_url,
            "evidence_message_ids": self.evidence_message_ids,
            "confidence": round(self.confidence, 2),
            "extractor": self.extractor,
            "duplicates": self.duplicates,
        }
