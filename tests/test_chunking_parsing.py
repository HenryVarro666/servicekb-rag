"""解析器与切块器测试。"""

import io
import json

import pytest

from servicekb.models import Block, DocType, ParsedDocument, SourceRecord
from servicekb.parsing import UnsupportedFormatError, parse
from servicekb.processing.chunking import ChunkingConfig, chunk_document


def _rec(content, mime, title="Doc", doc_type=DocType.MANUAL):
    return SourceRecord("src", "x", title, doc_type, content, mime)


def test_markdown_front_matter_headings_and_tables():
    md = "---\ntitle: T\nproduct_models: [AS-110K-HV]\n---\n# T\n## 1 概述\n正文一。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n### 1.1 子节\n正文二。\n"
    doc = parse(_rec(md, "text/markdown"))
    assert doc.metadata["product_models"] == ["AS-110K-HV"]
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "table", "heading", "paragraph"]
    assert doc.blocks[2].rows == [["a", "b"], ["1", "2"]]


def test_chunk_heading_path_and_sections_not_mixed():
    md = "# 手册\n## 4 故障排除\n### 4.1 E037\n检查绝缘。\n### 4.2 E051\n检查风扇。\n"
    doc = parse(_rec(md, "text/markdown", title="手册"))
    chunks = chunk_document(doc, ChunkingConfig(max_tokens=200, min_tokens=1))
    assert [c.heading_path for c in chunks] == [["手册", "4 故障排除", "4.1 E037"], ["手册", "4 故障排除", "4.2 E051"]]
    assert chunks[0].text == "检查绝缘。"  # 标题不写进正文


def test_large_table_split_repeats_header():
    rows = [["代码", "名称"]] + [[f"E{i:03d}", "某个比较长的故障名称" * 3] for i in range(40)]
    doc = ParsedDocument(_rec("", "text/markdown"), [Block(kind="table", rows=rows)])
    chunks = chunk_document(doc, ChunkingConfig(max_tokens=120))
    assert len(chunks) > 1
    assert all(c.text.startswith("| 代码 | 名称 |") for c in chunks)
    assert sum(c.text.count("| E") for c in chunks) == 40  # 每一行都在且只在一个块里


def test_qa_is_atomic_and_long_answers_keep_question():
    doc = ParsedDocument(_rec("", "x", doc_type=DocType.QA), [Block(kind="qa", fields={"question": "E037？", "answer": "检查。"})])
    [c] = chunk_document(doc, ChunkingConfig())
    assert c.text == "问题：E037？\n回答：检查。" and c.kind == "qa"
    long_answer = "。".join(f"第{i}步检查组串绝缘并记录数据" for i in range(80))
    doc = ParsedDocument(_rec("", "x"), [Block(kind="qa", fields={"question": "E037 怎么办？", "answer": long_answer})])
    parts = chunk_document(doc, ChunkingConfig(max_tokens=100, overlap_tokens=20))
    assert len(parts) > 2 and all(p.text.startswith("问题：E037 怎么办？") for p in parts)


def test_long_paragraph_overlap():
    para = "".join(f"第{i}句内容比较长一些。" for i in range(60))
    doc = ParsedDocument(_rec("", "x"), [Block(kind="paragraph", text=para)])
    chunks = chunk_document(doc, ChunkingConfig(max_tokens=80, overlap_tokens=20))
    assert len(chunks) > 3
    last_sentence_of_first = chunks[0].text.split("\n")[-1]
    assert chunks[1].text.startswith(last_sentence_of_first)  # 相邻块有重叠


def test_record_json():
    doc = parse(_rec(json.dumps({"料号": "310-04521-B", "替代料号": [], "名称": "风扇"}, ensure_ascii=False),
                     "application/x-servicekb-record+json"))
    [c] = chunk_document(doc, ChunkingConfig())
    assert c.text == "- 料号: 310-04521-B\n- 名称: 风扇"  # 空字段被去掉


def test_docx_parser_keeps_order(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("4 故障排除", level=1)
    d.add_paragraph("先检查连接器。")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "代码", "名称", "E037", "绝缘阻抗低"
    d.add_paragraph("最后上电。")
    buf = io.BytesIO()
    d.save(buf)
    doc = parse(_rec(buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
    assert [b.kind for b in doc.blocks] == ["heading", "paragraph", "table", "paragraph"]
    assert doc.blocks[2].rows[1] == ["E037", "绝缘阻抗低"]


def test_xlsx_parser(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "故障码"
    ws.append(["代码", "名称"])
    ws.append(["E051", "风扇故障"])
    buf = io.BytesIO()
    wb.save(buf)
    doc = parse(_rec(buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    assert doc.blocks[0].text == "故障码" and doc.blocks[1].rows == [["代码", "名称"], ["E051", "风扇故障"]]


def test_pdf_parser_heading_detection():
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in ["3 Troubleshooting", "3.1 E037 Low insulation", "Measure each string to ground.",
                 "Values above 1 MOhm are normal."]:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    doc = parse(_rec(bytes(pdf.output()), "application/pdf"))
    heads = [(b.text, b.level) for b in doc.blocks if b.kind == "heading"]
    assert heads == [("3 Troubleshooting", 1), ("3.1 E037 Low insulation", 2)]
    assert doc.blocks[-1].page == 1


def test_scanned_pdf_is_rejected():
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()  # 空白页 = 模拟扫描件（没有文本层）
    with pytest.raises(ValueError, match="scanned"):
        parse(_rec(bytes(pdf.output()), "application/pdf"))


def test_unknown_mime():
    with pytest.raises(UnsupportedFormatError):
        parse(_rec("x", "application/zip"))
