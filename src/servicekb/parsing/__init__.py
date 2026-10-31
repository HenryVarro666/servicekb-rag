"""解析（Parsing）：把各种格式的原始内容变成统一的 ``ParsedDocument``（Block 列表）。

"统一的中间表示"是整个 pipeline 解耦的关键（借鉴 docling 的 DoclingDocument、MinerU 的 content list）：
不管来源是 PDF、Word、Excel 还是工单 JSON，切块器只认 Block，新增格式只需加一个 Parser。

新增一个格式的步骤：
    1. 写一个函数 ``def parse_xxx(record: SourceRecord) -> ParsedDocument``
    2. 在下面的 ``PARSERS`` 里按 mime_type 注册
"""

from __future__ import annotations

from collections.abc import Callable

from servicekb.models import ParsedDocument, SourceRecord
from servicekb.parsing.markdown import parse_markdown, parse_plaintext
from servicekb.parsing.office import parse_docx, parse_pdf, parse_xlsx
from servicekb.parsing.structured import parse_qa_json, parse_record_json

Parser = Callable[[SourceRecord], ParsedDocument]

PARSERS: dict[str, Parser] = {
    "text/markdown": parse_markdown,
    "text/plain": parse_plaintext,
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": parse_xlsx,
    "application/x-servicekb-qa+json": parse_qa_json,
    "application/x-servicekb-record+json": parse_record_json,
}

EXTENSION_MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class UnsupportedFormatError(ValueError):
    pass


def parse(record: SourceRecord) -> ParsedDocument:
    parser = PARSERS.get(record.mime_type)
    if parser is None:
        raise UnsupportedFormatError(f"no parser for mime_type={record.mime_type!r} ({record.ref})")
    return parser(record)


__all__ = ["EXTENSION_MIME", "PARSERS", "UnsupportedFormatError", "parse"]
