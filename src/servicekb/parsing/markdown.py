"""Markdown / 纯文本解析。

Markdown 是最理想的中间格式：标题层级清晰、表格结构明确。实际项目里很多团队会先用
docling / MinerU 把 PDF 转成 Markdown，再走这个解析器（见 office.parse_pdf 的 docling 分支）。

支持 YAML front matter 携带元数据::

    ---
    title: AS-110K-HV 维修手册
    product_models: [AS-110K-HV]
    doc_type: manual
    acl_groups: [public]
    ---
"""

from __future__ import annotations

import re

import yaml

from servicekb.models import Block, ParsedDocument, SourceRecord
from servicekb.processing.text import normalize_text

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\|.*\|$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_LIST_ITEM = re.compile(r"^(\s*[-*+]|\s*\d+[.)])\s+")


def _as_text(record: SourceRecord) -> str:
    if isinstance(record.content, bytes):
        return record.content.decode("utf-8", errors="replace")
    return record.content


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown_text(text: str) -> tuple[list[Block], dict]:
    """返回 (blocks, front_matter)。单独拆出来，方便 PDF→Markdown 后复用。"""
    meta: dict = {}
    m = _FRONT_MATTER.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        text = text[m.end() :]

    blocks: list[Block] = []
    para: list[str] = []
    table: list[list[str]] = []
    in_code = False

    def flush_para() -> None:
        if para:
            joined = normalize_text("\n".join(para))
            if joined:
                kind = "list" if all(_LIST_ITEM.match(p) for p in para) else "paragraph"
                blocks.append(Block(kind=kind, text=joined))
            para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block(kind="table", rows=[list(r) for r in table]))
            table.clear()

    for raw in text.split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            para.append(line)
            continue
        if in_code:
            para.append(line)
            continue
        if _TABLE_ROW.match(line.strip()):
            flush_para()
            if not _TABLE_SEP.match(line.strip()):
                table.append(_split_row(line))
            continue
        flush_table()
        h = _HEADING.match(line)
        if h:
            flush_para()
            blocks.append(Block(kind="heading", text=normalize_text(h.group(2)), level=len(h.group(1))))
        elif not line.strip():
            flush_para()
        else:
            para.append(line)
    flush_para()
    flush_table()
    return blocks, meta


def parse_markdown(record: SourceRecord) -> ParsedDocument:
    blocks, meta = parse_markdown_text(_as_text(record))
    # 标题优先级：front matter > 文档第一个一级标题 > 文件名
    if meta.get("title"):
        record.title = str(meta["title"])
    if blocks and blocks[0].kind == "heading" and blocks[0].level == 1:
        if not meta.get("title"):
            record.title = blocks[0].text
        # 一级标题就是文档标题时去掉，避免标题路径里出现两次
        if blocks[0].text == record.title:
            blocks = blocks[1:]
    return ParsedDocument(record=record, blocks=blocks, metadata=meta)


def parse_plaintext(record: SourceRecord) -> ParsedDocument:
    text = normalize_text(_as_text(record))
    blocks = [Block(kind="paragraph", text=p) for p in text.split("\n\n") if p.strip()]
    return ParsedDocument(record=record, blocks=blocks)
