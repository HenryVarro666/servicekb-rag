"""结构化记录解析：工单问答、物料主数据行。

这两类数据由 connector 预先整理成 JSON（见 connectors/qa_archive.py、connectors/parts_master.py），
解析器只负责把字段变成 Block。清洗（去邮件引用、脱敏）在 connector 里已经做过了。
"""

from __future__ import annotations

import json

from servicekb.models import Block, ParsedDocument, SourceRecord


def _load(record: SourceRecord) -> dict:
    raw = record.content.decode("utf-8") if isinstance(record.content, bytes) else record.content
    return json.loads(raw)


def parse_qa_json(record: SourceRecord) -> ParsedDocument:
    data = _load(record)
    fields = {k: str(v) for k, v in data.items() if v not in (None, "", [])}
    return ParsedDocument(record=record, blocks=[Block(kind="qa", fields=fields)])


def parse_record_json(record: SourceRecord) -> ParsedDocument:
    data = _load(record)
    fields = {k: (", ".join(v) if isinstance(v, list) else str(v)) for k, v in data.items()}
    return ParsedDocument(record=record, blocks=[Block(kind="record", fields=fields)])
