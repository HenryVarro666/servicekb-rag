"""切块（Chunking）：把一份文档切成适合检索的小块。

切块是 RAG 里**最影响效果、又最容易被忽视**的一步。块太大：一块里混了好几件事，向量"平均"
掉了，召回不准，还浪费 LLM 上下文；块太小：丢了上下文，"拧紧到 2.5 N·m"不知道是在拧哪个螺丝。

本模块按"文档结构"切，而不是按固定字数硬切（借鉴 RAGFlow 的模板化切块、docling HybridChunker、
Unstructured chunk_by_title）：

+-----------+-----------------------------------------------------------------+
| Block 类型 | 切法                                                            |
+===========+=================================================================+
| 标题      | 不单独成块，而是维护"标题路径"栈，挂到后续每个块的 heading_path 上        |
| 段落/列表  | 同一章节内贪心打包到 max_tokens；超长段落按句切；块之间保留 overlap      |
| 表格      | 单独成块；大表按**整行**拆分，每块重复表头（否则第二块的数字没人看得懂）  |
| 问答对    | 一问一答是一个**原子块**；答案太长时拆分，但每块都带上问题              |
| 记录      | 一行记录一块，渲染成 "- 字段: 值" 形式                                 |
+-----------+-----------------------------------------------------------------+

章节边界不跨越：不同小节的内容不会被打包进同一块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from servicekb.models import Block, ParsedDocument
from servicekb.processing.text import estimate_tokens, split_sentences


@dataclass
class ChunkingConfig:
    max_tokens: int = 450
    overlap_tokens: int = 60
    min_tokens: int = 15

    def fingerprint(self) -> str:
        """配置指纹。配置一变，所有文档都需要重新切块（见 pipeline 的 pipeline_version）。"""
        return f"chunk:{self.max_tokens}:{self.overlap_tokens}:{self.min_tokens}"


@dataclass
class ChunkDraft:
    """切块结果（还没有 ID / 向量 / 实体）。"""

    text: str
    heading_path: list[str]
    page: int | None = None
    kind: str = "text"  # text / table / qa / record
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- 内部：打包


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int | None


def _pack(units: list[_Unit], cfg: ChunkingConfig) -> list[tuple[str, int | None]]:
    """把句子 / 段落单元贪心打包成块，相邻块之间带 overlap。

    overlap 的意义：一个关键信息恰好落在两块边界时，至少有一块是完整的。
    """
    chunks: list[tuple[str, int | None]] = []
    cur: list[_Unit] = []
    cur_tokens = 0
    for u in units:
        if cur and cur_tokens + u.tokens > cfg.max_tokens:
            chunks.append(("\n".join(x.text for x in cur), cur[0].page))
            # 从当前块尾部回取若干单元作为下一块的开头
            tail: list[_Unit] = []
            t = 0
            for x in reversed(cur):
                if t + x.tokens > cfg.overlap_tokens:
                    break
                tail.insert(0, x)
                t += x.tokens
            cur, cur_tokens = tail, t
        cur.append(u)
        cur_tokens += u.tokens
    if cur:
        chunks.append(("\n".join(x.text for x in cur), cur[0].page))
    return chunks


def _units_from_text(text: str, page: int | None, cfg: ChunkingConfig) -> list[_Unit]:
    """一个段落如果不超长就整体作为一个单元；超长就按句子拆。"""
    tok = estimate_tokens(text)
    if tok <= cfg.max_tokens:
        return [_Unit(text, tok, page)]
    return [_Unit(s, estimate_tokens(s), page) for s in split_sentences(text)]


def _render_table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(c.strip() for c in r) + " |" for r in rows)


def _table_chunks(block: Block, path: list[str], cfg: ChunkingConfig) -> list[ChunkDraft]:
    rows = [r for r in (block.rows or []) if any(c.strip() for c in r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    header_tokens = estimate_tokens(" ".join(header))
    out: list[ChunkDraft] = []
    batch: list[list[str]] = []
    tokens = header_tokens
    for r in body:
        rt = estimate_tokens(" ".join(r))
        if batch and tokens + rt > cfg.max_tokens:
            out.append(ChunkDraft(_render_table([header, *batch]), path, block.page, "table"))
            batch, tokens = [], header_tokens
        batch.append(r)
        tokens += rt
    if batch or not body:
        out.append(ChunkDraft(_render_table([header, *batch]), path, block.page, "table"))
    return out


def _qa_chunks(block: Block, path: list[str], cfg: ChunkingConfig) -> list[ChunkDraft]:
    f = block.fields or {}
    q, a = f.get("question", "").strip(), f.get("answer", "").strip()
    if not q or not a:
        return []
    meta = {k: v for k, v in f.items() if k not in ("question", "answer")}
    text = f"问题：{q}\n回答：{a}"
    if estimate_tokens(text) <= cfg.max_tokens:
        return [ChunkDraft(text, path, None, "qa", meta)]
    # 答案太长：拆答案，每块都带问题（否则后面几块不知道在回答什么）
    budget = max(cfg.max_tokens - estimate_tokens(q) - 10, cfg.max_tokens // 3)
    sub = ChunkingConfig(max_tokens=budget, overlap_tokens=cfg.overlap_tokens, min_tokens=0)
    parts = _pack(_units_from_text(a, None, sub), sub)
    n = len(parts)
    return [
        ChunkDraft(f"问题：{q}\n回答（{i}/{n}）：{p}", path, None, "qa", meta)
        for i, (p, _) in enumerate(parts, 1)
    ]


def _record_chunk(block: Block, path: list[str]) -> ChunkDraft | None:
    f = {k: v for k, v in (block.fields or {}).items() if str(v).strip()}
    if not f:
        return None
    text = "\n".join(f"- {k}: {v}" for k, v in f.items())
    return ChunkDraft(text, path, block.page, "record")


# --------------------------------------------------------------------------- 入口


def chunk_document(doc: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
    title = doc.record.title
    stack: list[tuple[int, str]] = []  # (level, heading text)
    section_units: list[_Unit] = []
    out: list[ChunkDraft] = []

    def path() -> list[str]:
        return [title, *(h for _, h in stack)] if title else [h for _, h in stack]

    def flush() -> None:
        nonlocal section_units
        if section_units:
            p = path()
            for text, page in _pack(section_units, cfg):
                out.append(ChunkDraft(text, p, page, "text"))
        section_units = []

    for b in doc.blocks:
        if b.kind == "heading":
            flush()
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text.strip()))
        elif b.kind in ("paragraph", "list"):
            if b.text.strip():
                section_units.extend(_units_from_text(b.text.strip(), b.page, cfg))
        elif b.kind == "table":
            flush()
            out.extend(_table_chunks(b, path(), cfg))
        elif b.kind == "qa":
            flush()
            out.extend(_qa_chunks(b, path(), cfg))
        elif b.kind == "record":
            flush()
            rc = _record_chunk(b, path())
            if rc:
                out.append(rc)
    flush()

    # 过滤过短的文本块（孤立的"注意"、页眉页脚残留等）；结构化块（问答 / 记录 / 表格）一律保留
    kept = [c for c in out if c.kind != "text" or estimate_tokens(c.text) >= cfg.min_tokens]
    return kept or out
