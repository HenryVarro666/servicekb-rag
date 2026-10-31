"""历史问答 / 工单归档（JSONL 或 CSV 导出）。

工单是售后知识库里**价值最高、也最脏**的数据：
- 价值高：每一张已解决的工单都是"真实问题 + 被验证过的解决方案"。
- 脏：邮件往来的引用和签名、客户电话、型号写法五花八门、大量重复提问、很多工单没有结论。

本 connector 做三件事：
1. 字段映射：不同系统导出的字段名不同，用 ``field_map`` 统一。
2. 清洗：去邮件噪声、PII 脱敏（processing/cleaning.py）。
3. **质量门槛**：只收"已解决且有答案"的工单。没有结论的工单进知识库只会误导 LLM。
   被过滤的数量会记录在 ingestion_runs.stats 里，方便和业务方对账。

输入格式（每行一个 JSON，字段名可通过 field_map 映射）::

    {"ticket_id": "TK-2024-0103", "created_at": "2024-03-02", "product_model": "AS110K",
     "subject": "...", "question": "...", "answer": "...", "status": "resolved", "tags": ["E037"]}
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from servicekb.connectors.base import Connector
from servicekb.models import DocType, SourceEvent, SourceRecord
from servicekb.processing.cleaning import clean_text

QA_MIME = "application/x-servicekb-qa+json"

DEFAULT_FIELD_MAP = {
    "ticket_id": "ticket_id",
    "created_at": "created_at",
    "product_model": "product_model",
    "subject": "subject",
    "question": "question",
    "answer": "answer",
    "status": "status",
    "tags": "tags",
}


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


class QAArchiveConnector(Connector):
    full_snapshot = True

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        acl_groups: list[str] | None = None,
        field_map: dict[str, str] | None = None,
        resolved_statuses: list[str] | None = None,
        min_answer_chars: int = 15,
        uri_template: str = "",
        doc_type: str = "qa",
        allow_row_acl: bool = False,
    ) -> None:
        self.source_id = source_id
        self.path = Path(path)
        self.acl_groups = acl_groups or ["service"]
        self.fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
        self.resolved = {s.lower() for s in (resolved_statuses or ["resolved", "closed", "已解决", "已关闭"])}
        self.min_answer_chars = min_answer_chars
        self.uri_template = uri_template
        self.doc_type = DocType(doc_type)
        # 行级权限：只对"我们自己的 pipeline 生成的文件"开启（例如聊天记录按频道成员授权），
        # 不要对业务系统直接导出的文件开启。
        self.allow_row_acl = allow_row_acl
        self._stats = {"total": 0, "filtered_unresolved": 0, "filtered_no_answer": 0}

    def _rows(self) -> Iterator[dict[str, Any]]:
        if self.path.suffix.lower() == ".csv":
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                yield from csv.DictReader(f)
        else:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        fm = self.fm
        for row in self._rows():
            self._stats["total"] += 1
            tid = str(row.get(fm["ticket_id"], "")).strip()
            status = str(row.get(fm["status"], "")).strip().lower()
            if status not in self.resolved:
                self._stats["filtered_unresolved"] += 1
                continue
            question = clean_text(str(row.get(fm["question"], "")), email_like=True)
            answer = clean_text(str(row.get(fm["answer"], "")), email_like=True)
            if len(answer) < self.min_answer_chars:
                self._stats["filtered_no_answer"] += 1
                continue
            subject = clean_text(str(row.get(fm["subject"], "") or ""))
            model_raw = str(row.get(fm["product_model"], "") or "").strip()
            tags = row.get(fm["tags"]) or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.replace("；", ";").split(";") if t.strip()]
            payload = {
                "question": f"{subject}\n{question}".strip() if subject and subject not in question else question,
                "answer": answer,
                "ticket_id": tid,
                "product_model": model_raw,
                "created_at": str(row.get(fm["created_at"], "") or ""),
            }
            if row.get("source_url"):
                uri = str(row["source_url"])  # 例如 Teams 消息的 webUrl，引用时可直接跳回原对话
            elif self.uri_template:
                uri = self.uri_template.format(ticket_id=tid)
            else:
                uri = f"ticket://{tid}"
            acl = list(self.acl_groups)
            if self.allow_row_acl and row.get("acl_groups"):
                acl = [str(g) for g in row["acl_groups"]]
            yield SourceRecord(
                source_id=self.source_id,
                external_id=tid,
                title=subject or question[:40],
                doc_type=self.doc_type,
                content=json.dumps(payload, ensure_ascii=False),
                mime_type=QA_MIME,
                uri=uri,
                metadata={"product_models_raw": [model_raw] if model_raw else [], "tags": tags, "status": status},
                acl_groups=acl,
                updated_at=_parse_date(row.get(fm["created_at"])),
            )

    def stats(self) -> dict[str, int]:
        return dict(self._stats)
