"""生成 README / 文档用的示意图（中英文两版 SVG）。

    python docs/images/make_diagrams.py        # 输出到 docs/images/*.zh.svg 和 *.en.svg

为什么用脚本而不是画图软件？
    - 图里的文字、数字和代码 / 评估结果保持一致，改了只需重新运行；
    - 中英文两版共用同一套坐标，不会出现两版不一致；
    - 纯标准库，不需要安装任何东西；SVG 是文本，git diff 能看出改了什么。

每个标签写成 T("中文", "English")，按语言取值。
"""

from __future__ import annotations

import html
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
        "'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', Helvetica, Arial, sans-serif")
MONO = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', 'WenQuanYi Zen Hei Mono', monospace"

INK, MUTED, LINE = "#111827", "#4B5563", "#6B7280"
# 每一类元素一个颜色：(填充, 描边)
C = {
    "source": ("#EFF6FF", "#3B82F6"),
    "pipe": ("#ECFDF5", "#10B981"),
    "store": ("#F5F3FF", "#8B5CF6"),
    "query": ("#FFF7ED", "#F97316"),
    "good": ("#F0FDF4", "#16A34A"),
    "warn": ("#FFFBEB", "#D97706"),
    "bad": ("#FEF2F2", "#DC2626"),
    "gray": ("#F9FAFB", "#9CA3AF"),
    "frame": ("#FCFCFD", "#D1D5DB"),
}

_CJK = re.compile(r"[⺀-鿿＀-￯　-〿]")


def text_width(s: str, size: float) -> float:
    """粗略估算文字宽度（用来检查文字有没有超出方框）。"""
    w = 0.0
    for ch in s:
        if _CJK.match(ch):
            w += size
        elif ch.isupper() or ch.isdigit():
            w += size * 0.62
        elif ch in " ·.,:;|!'()[]-":
            w += size * 0.32
        else:
            w += size * 0.54
    return w


class SVG:
    def __init__(self, w: int, h: int, name: str) -> None:
        self.w, self.h, self.name = w, h, name
        self.body: list[str] = []
        self.markers: set[str] = set()
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ 基础图元
    def rect(self, x, y, w, h, fill, stroke, rx=10, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=14, weight=400, anchor="middle", fill=INK, mono=False, italic=False):
        fam = MONO if mono else FONT
        st = ' font-style="italic"' if italic else ""
        self.body.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                         f'text-anchor="{anchor}" fill="{fill}"{st}>{html.escape(s)}</text>')

    def lines(self, cx, cy, lines, sizes=None, weights=None, fills=None, gap=1.35, mono=False, anchor="middle"):
        """多行文字，整体在 (cx, cy) 垂直居中。"""
        n = len(lines)
        sizes = (sizes or [14] + [12] * (n - 1))[:n]
        weights = (weights or [600] + [400] * (n - 1))[:n]
        fills = (fills or [INK] + [MUTED] * (n - 1))[:n]
        total = sum(s * gap for s in sizes) - (sizes[-1] * (gap - 1))
        y = cy - total / 2
        for i, line in enumerate(lines):
            y += sizes[i] * (1 if i == 0 else gap)
            self.text(cx, y - sizes[i] * 0.18, line, sizes[i], weights[i], anchor, fills[i], mono)

    def box(self, x, y, w, h, lines, kind="gray", rx=10, sizes=None, weights=None, dash=None, mono=False):
        fill, stroke = C[kind]
        self.rect(x, y, w, h, fill, stroke, rx=rx, dash=dash)
        lines = [lines] if isinstance(lines, str) else lines
        sizes = sizes or [14] + [12] * (len(lines) - 1)
        for ln, sz in zip(lines, sizes[: len(lines)], strict=False):
            if text_width(ln, sz) > w - 12:
                self.warnings.append(f"{self.name}: text overflows box ({w}px): {ln!r}")
        self.lines(x + w / 2, y + h / 2, lines, sizes, weights, mono=mono)

    def frame(self, x, y, w, h, title, kind="frame", title_x=None, title_anchor="start", sub=None):
        fill, stroke = C[kind]
        self.rect(x, y, w, h, fill, stroke, rx=14, sw=1.2, dash="6 4")
        tx = title_x if title_x is not None else x + 16
        self.text(tx, y + 24, title, 15, 700, title_anchor, INK)
        if sub:
            self.text(tx, y + 44, sub, 12, 400, title_anchor, MUTED)

    def diamond(self, cx, cy, w, h, lines, kind="query"):
        fill, stroke = C[kind]
        pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        self.body.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.lines(cx, cy, lines, [13] * len(lines), [600] * len(lines))

    def chip(self, x, y, s, kind="gray", size=12, mono=True):
        w = text_width(s, size) + 16
        fill, stroke = C[kind]
        self.rect(x, y, w, size + 10, fill, stroke, rx=(size + 10) / 2, sw=1)
        self.text(x + w / 2, y + size + 3, s, size, 500, "middle", INK, mono)
        return w

    def arrow(self, pts, color=LINE, dash=None, label=None, label_at=None, label_anchor="middle", width=1.6):
        mid = "m" + color.lstrip("#")
        self.markers.add(color)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.body.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{width}"{d} '
                         f'stroke-linejoin="round" marker-end="url(#{mid})"/>')
        if label:
            lx, ly = label_at or pts[len(pts) // 2]
            self.text(lx, ly, label, 12, 500, label_anchor, color if color != LINE else MUTED)

    # ------------------------------------------------------------------ 输出
    def save(self, path: Path) -> None:
        defs = "".join(
            f'<marker id="m{c.lstrip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self.markers)
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img">'
            f"<defs>{defs}</defs>"
            # 白色背景：GitHub 深色模式下也能看清
            f'<rect x="0" y="0" width="{self.w}" height="{self.h}" rx="16" fill="#FFFFFF"/>'
            + "".join(self.body)
            + "</svg>\n"
        )
        path.write_text(svg, "utf-8")


# =========================================================================== 1. 总体架构


def architecture(lang: str) -> SVG:
    def T(zh, en):
        return zh if lang == "zh" else en

    s = SVG(1200, 760, "architecture")
    s.text(600, 38, T("servicekb 总体架构", "servicekb architecture"), 22, 700)

    # ---- 数据源
    s.frame(24, 62, 520, 200, T("数据源", "Data sources"))
    srcs = [
        (40, 98, "source", [T("手册 · 服务通告", "Manuals · bulletins"), "MD · PDF · DOCX · XLSX"]),
        (204, 98, "source", [T("历史工单", "Ticket archive"), T("只收已解决、有答案的", "resolved tickets only")]),
        (368, 98, "source", [T("Teams 聊天记录", "Teams chats"), T("chat clean → 问答对", "chat clean → Q&A pairs")]),
        (40, 182, "store", [T("领域词典", "Domain dictionary"), T("型号别名 · 故障码", "model aliases · faults")]),
        (204, 182, "store", [T("物料主数据", "Parts master"), T("料号 · 替代件 · 状态", "parts · substitutes")]),
        (368, 182, "source", ["SharePoint", T("Graph delta 增量同步", "Graph delta sync")]),
    ]
    for x, y, k, lb in srcs:
        s.box(x, y, 156, 64, lb, k)

    # ---- 导入 pipeline
    s.frame(572, 62, 604, 200, T("导入 pipeline", "Ingestion pipeline"),
            sub=T("增量 · 幂等 · 向量复用 · 故障隔离 · 安全删除",
                  "incremental · idempotent · vector reuse · fault isolation · safe deletes"))
    steps = [T("解析", "Parse"), T("清洗\n脱敏", "Clean\n& redact"), T("结构化\n切块", "Chunk"),
             T("实体\n链接", "Link\nentities"), T("中英\n分词", "Tokenize"), T("向量化", "Embed")]
    for i, st in enumerate(steps):
        x = 588 + i * 98
        s.box(x, 120, 84, 60, st.split("\n"), "pipe", sizes=[14, 14], weights=[600, 600])
        if i:
            s.arrow([(x - 14, 150), (x - 1, 150)], C["pipe"][1])
    s.text(874, 212, T("内容哈希 + pipeline 版本没变 → 跳过", "content hash + pipeline version unchanged → skip"),
           12, 400, "middle", MUTED)
    s.text(874, 232, T("块 ID = hash(文档, 内容) → 只重算改动的块", "chunk id = hash(doc, content) → re-embed changed chunks only"),
           12, 400, "middle", MUTED)
    s.arrow([(544, 130), (571, 130)], C["source"][1])

    # ---- PostgreSQL
    s.frame(24, 296, 1152, 162, T("PostgreSQL + pgvector（关系库 · 向量库 · 全文库）",
                                  "PostgreSQL + pgvector (relational · vector · full-text)"),
            title_x=600, title_anchor="middle")
    s.box(40, 344, 330, 96, [T("对象层", "Object layer"), "entities · entity_relations",
                            T("别名归一 · 替代关系 · 停产状态", "aliases · substitutes · part status")], "store",
          sizes=[15, 12, 12])
    s.box(390, 344, 420, 96, [T("chunks（检索单元）", "chunks (retrieval units)"),
                             "embedding vector + HNSW · search_tsv + GIN",
                             "entity_ids · acl_groups · product_models"], "store", sizes=[15, 12, 12])
    s.box(830, 344, 330, 96, [T("运维与闭环", "Ops & feedback loop"), "ingestion_runs · connector_state",
                             "query_logs · feedback"], "store", sizes=[15, 12, 12])
    s.arrow([(118, 246), (118, 343)], C["store"][1])
    s.arrow([(282, 246), (282, 343)], C["store"][1], label=T("对象层先行", "objects first"), label_at=(200, 284))
    s.arrow([(796, 262), (796, 343)], C["pipe"][1], label=T("原子替换", "atomic replace"), label_at=(806, 288),
            label_anchor="start")

    # ---- 问答
    s.frame(24, 492, 1152, 244, T("问答：检索 → 证据门槛 → 生成", "Ask: retrieve → evidence gate → generate"),
            title_x=600)
    s.box(40, 584, 132, 64, [T("现场工程师", "Field engineer"), T("提问", "asks")], "gray")
    s.box(196, 580, 160, 72, [T("查询分析", "Query analysis"), T("实体 · 替代件扩展", "entities · expansion"),
                             T("中英分词", "tokens")], "query", sizes=[14, 12, 12])
    lanes = [T("向量 · 语义", "Dense · semantic"), T("关键词 · BM25", "Keyword · BM25"),
             T("实体 · 料号 / 故障码", "Entity · part / fault IDs")]
    for i, ln in enumerate(lanes):
        y = 546 + i * 50
        s.box(384, y, 176, 40, [ln], "query", sizes=[13])
        s.arrow([(356, 616), (370, 616), (370, y + 20), (383, y + 20)], C["query"][1])
        s.arrow([(560, y + 20), (574, y + 20), (574, 616), (587, 616)], C["query"][1])
    s.box(588, 580, 132, 72, [T("RRF 融合", "RRF fusion"), T("折叠近似重复", "collapse near-dups")], "query")
    s.box(744, 580, 112, 72, [T("重排", "Rerank"), "cross-encoder", T("（可选）", "(optional)")], "query",
          sizes=[14, 11, 11])
    s.diamond(934, 616, 124, 96, [T("证据", "Enough"), T("够吗？", "evidence?")])
    s.box(1010, 528, 152, 76, [T("LLM 生成", "LLM answer"), T("+ 结构化事实", "+ structured facts"),
                              T("强制引用 [n]", "citations [n]")], "good", sizes=[14, 12, 12])
    s.box(1010, 640, 152, 64, [T("拒答 + 升级", "Refuse + escalate"), T("不调用 LLM", "no LLM call")], "bad")

    s.arrow([(172, 616), (195, 616)])
    s.arrow([(720, 616), (743, 616)], C["query"][1])
    s.arrow([(856, 616), (871, 616)], C["query"][1])
    s.arrow([(934, 568), (934, 566), (1009, 566)], C["good"][1], label=T("是", "yes"), label_at=(960, 558))
    s.arrow([(934, 664), (934, 672), (1009, 672)], C["bad"][1], label=T("否", "no"), label_at=(958, 690))
    s.arrow([(472, 440), (472, 545)], C["store"][1], label=T("三路召回", "3-lane recall"), label_at=(520, 470))
    s.arrow([(276, 440), (276, 579)], C["store"][1], label=T("实体词典", "entity lookup"), label_at=(316, 520))
    s.arrow([(1086, 527), (1086, 441)], C["good"][1], label=T("写日志", "log"), label_at=(1120, 476))
    return s


# =========================================================================== 2. 一次检索的细节


def query_flow(lang: str) -> SVG:
    def T(zh, en):
        return zh if lang == "zh" else en

    s = SVG(1200, 560, "query_flow")
    s.text(600, 38, T("一次问答在内部发生了什么", "What happens inside one question"), 22, 700)

    # 阶段 1：问题 + 查询分析
    s.box(24, 70, 236, 60, [T("“310-04521-A 还能订到吗？”", "“Can I still order 310-04521-A?”")], "gray",
          sizes=[13])
    s.rect(24, 150, 236, 230, *C["query"], rx=10)
    s.text(142, 176, T("① 查询分析", "① Query analysis"), 15, 700)
    s.text(40, 204, T("识别实体", "entities"), 12, 600, "start", MUTED)
    s.chip(40, 212, "part:310-04521-A", "query")
    s.text(40, 258, T("替代件扩展", "substitute expansion"), 12, 600, "start", MUTED)
    s.chip(40, 266, "+ part:310-04521-B", "good")
    s.text(40, 312, T("分词", "tokens"), 12, 600, "start", MUTED)
    w = s.chip(40, 320, "310-04521-a", "gray")
    s.chip(48 + w, 320, T("订到", "order"), "gray")
    s.arrow([(142, 130), (142, 149)])

    # 阶段 2：三路召回
    s.frame(284, 70, 430, 310, T("② 三路并行召回（各取 top 40）", "② Three lanes in parallel (top 40 each)"))
    cols = [
        (T("向量", "Dense"), [("1", "310-04521-A"), ("2", "SB-2025-014 §3"), ("3", "310-04521-B")]),
        (T("关键词", "Keyword"), [("1", "SB-2025-014 §3"), ("2", "310-04521-A"), ("3", "310-04521-B")]),
        (T("实体", "Entity"), [("1", "310-04521-A"), ("2", "310-04521-B"), ("…", ""), ("7", "SB-2025-014 §3")]),
    ]
    for ci, (head, rows) in enumerate(cols):
        x = 296 + ci * 140
        s.rect(x, 110, 136, 254, "#FFFFFF", C["query"][1], rx=8, sw=1.2)
        s.text(x + 68, 134, head, 14, 700)
        for ri, (rank, label) in enumerate(rows):
            y = 156 + ri * 48
            if rank == "…":
                s.text(x + 68, y + 22, "…", 14, 700, "middle", MUTED)
                continue
            s.rect(x + 7, y, 122, 38, "#FFF7ED", "#FDBA74", rx=6, sw=1)
            s.text(x + 19, y + 24, rank, 13, 700, "middle", C["query"][1])
            s.text(x + 30, y + 24, label, 11, 500, "start", INK, mono=True)
            if text_width(label, 11) > 96:
                s.warnings.append(f"query_flow: lane label overflow {label!r}")
    s.arrow([(260, 265), (283, 265)])

    # 阶段 3：RRF
    s.rect(738, 70, 214, 310, *C["query"], rx=10)
    s.text(845, 98, T("③ RRF 融合", "③ RRF fusion"), 15, 700)
    s.text(845, 126, "score = Σ 1 / (60 + rank)", 13, 600, "middle", INK, mono=True)
    s.text(845, 146, T("只看排名，不看分数", "uses ranks, not raw scores"), 11, 400, "middle", MUTED)
    fused = [(T("料号 A", "part A"), "1/61+1/62+1/61", "0.0489"),
             (T("料号 B", "part B"), "1/63+1/63+1/62", "0.0479"),
             ("SB §3", "1/62+1/61+1/67", "0.0474")]
    for i, (lab, calc, sc) in enumerate(fused):
        y = 166 + i * 64
        s.rect(752, y, 186, 54, "#FFFFFF", "#FDBA74", rx=8, sw=1)
        s.text(764, y + 20, f"{i + 1}. {lab}", 13, 700, "start")
        s.text(926, y + 20, sc, 13, 700, "end", C["query"][1], mono=True)
        s.text(764, y + 40, calc, 11, 400, "start", MUTED, mono=True)
    s.text(845, 368, T("被多路同时召回的排在前面", "found by several lanes ⇒ ranked higher"), 11, 500, "middle", MUTED)
    s.arrow([(714, 225), (737, 225)])

    # 阶段 4：门槛 + 事实 + 生成
    s.rect(976, 70, 200, 132, *C["store"], rx=10)
    s.text(1076, 96, T("④ 结构化事实", "④ Structured facts"), 15, 700)
    s.lines(1076, 150, [T("310-04521-A：停产", "310-04521-A: obsolete"),
                        T("已被 310-04521-B 替代", "replaced by 310-04521-B"),
                        T("来自物料主数据", "from the parts master")],
            [13, 13, 11], [600, 600, 400], [INK, INK, MUTED])
    s.rect(976, 222, 200, 158, *C["good"], rx=10)
    s.text(1076, 248, T("⑤ LLM 回答", "⑤ LLM answer"), 15, 700)
    s.lines(1076, 310, [T("310-04521-A 已停产，", "310-04521-A is obsolete;"),
                        T("请改用 310-04521-B [1][3]", "use 310-04521-B [1][3]"),
                        T("证据不足时直接拒答", "refuses when evidence is weak"),
                        T("并标记升级", "and flags for escalation")],
            [13, 13, 11, 11], [600, 600, 400, 400], [INK, INK, MUTED, MUTED])
    s.arrow([(952, 225), (975, 225)])
    s.arrow([(1076, 202), (1076, 221)], C["store"][1])

    # 底部说明
    s.rect(24, 404, 1152, 132, "#F9FAFB", "#E5E7EB", rx=12, sw=1)
    notes = [
        T("向量路：擅长同义表达和口语描述，但分不清 310-04521-A / 310-04521-B 这种只差一位的标识符。",
          "Dense lane: great at paraphrases and fuzzy symptoms, but can't tell 310-04521-A from 310-04521-B."),
        T("关键词路：字面匹配故障码、术语；中文先用 jieba 分词，料号整体保留。",
          "Keyword lane: literal matches on fault codes and terms; Chinese is segmented with jieba, IDs kept whole."),
        T("实体路：按对象 ID 精确匹配，型号别名已归一（AS110K = AS-110K-HV），还会带上替代件。",
          "Entity lane: exact match on object IDs; model aliases normalized (AS110K = AS-110K-HV), substitutes included."),
        T("分数来自示例数据的真实运行（servicekb search \"310-04521-A 还能订到吗？\"）。",
          "Scores are from a real run on the sample data (servicekb search ...)."),
    ]
    for i, n in enumerate(notes):
        s.text(44, 434 + i * 26, ("• " if i < 3 else "") + n, 13 if i < 3 else 12, 400, "start",
               INK if i < 3 else MUTED, italic=i == 3)
    return s


# =========================================================================== 3. 导入 pipeline（增量）


def ingest_pipeline(lang: str) -> SVG:
    def T(zh, en):
        return zh if lang == "zh" else en

    s = SVG(1200, 520, "ingest_pipeline")
    s.text(600, 38, T("导入 pipeline：增量、幂等、故障隔离", "Ingestion: incremental, idempotent, fault-isolated"), 22, 700)

    y0, h = 96, 72
    s.box(24, y0, 130, h, ["Connector", T("读取 + 质量门槛", "read + quality gate")], "source")
    s.box(178, y0, 112, h, [T("解析", "Parse"), T("→ Block 列表", "→ Blocks")], "pipe")
    s.diamond(372, y0 + h / 2, 132, 104, [T("内容 / 版本", "content or"), T("变了吗？", "version"), ]
              + ([] if lang == "zh" else ["changed?"]), "warn")
    s.box(462, y0, 112, h, [T("切块", "Chunk"), T("章节 · 表格 · 问答", "section · table · Q&A")], "pipe",
          sizes=[14, 11])
    s.box(598, y0, 112, h, [T("实体链接", "Link entities"), T("料号 · 型号 · 故障码", "part · model · fault")], "pipe",
          sizes=[14, 11])
    s.box(734, y0, 112, h, [T("分词", "Tokenize"), T("正文 + 标题路径", "body + headings")], "pipe", sizes=[14, 11])
    s.box(870, y0, 130, h, [T("向量化", "Embed"), T("复用未变块的向量", "reuse unchanged")], "pipe", sizes=[14, 11])
    s.box(1024, y0, 152, h, [T("原子替换", "Atomic replace"), T("一个事务：文档 + 块", "one transaction")], "store",
          sizes=[14, 11])
    for a, b in [(154, 177), (290, 305), (439, 461), (574, 597), (710, 733), (846, 869), (1000, 1023)]:
        s.arrow([(a, y0 + h / 2), (b, y0 + h / 2)], C["pipe"][1])
    s.text(450, y0 + h / 2 - 10, T("是", "yes"), 12, 600, "middle", C["warn"][1])
    s.arrow([(372, y0 + h / 2 + 52), (372, 225)], C["warn"][1], label=T("否", "no"), label_at=(388, 204),
            label_anchor="start")
    s.box(312, 226, 120, 44, [T("跳过", "Skip"), "unchanged"], "gray", sizes=[14, 11])

    # 向量复用的示例
    s.rect(822, 196, 230, 92, *C["good"], rx=10)
    s.lines(937, 242, [T("改了手册里一句话：", "Edited one sentence:"), T("1 块重新向量化", "1 chunk re-embedded"),
                       T("10 块复用旧向量", "10 chunks reuse vectors")], [12, 14, 14], [400, 700, 700],
            [MUTED, INK, INK])
    s.arrow([(935, 195), (935, y0 + h + 1)], C["good"][1], dash="4 3")

    # 故障隔离
    s.rect(470, 196, 330, 92, *C["bad"], rx=10)
    s.lines(635, 242, [T("任何一步失败（坏 PDF、模型超时…）", "Any step fails (bad PDF, model timeout…)"),
                       T("→ 文档记为 failed，保留旧版本的块", "→ doc marked failed, old chunks kept"),
                       T("→ 继续处理其他文档；下次自动重试", "→ others continue; retried next run")],
            [13, 12, 12], [600, 400, 400], [INK, MUTED, MUTED])
    s.arrow([(518, y0 + h), (518, 195)], C["bad"][1], dash="4 3")

    # 版本指纹
    s.rect(24, 312, 560, 92, *C["warn"], rx=10)
    s.text(40, 338, T("pipeline_version（任何一项变化 → 全部文档重新处理，向量尽量复用）",
                      "pipeline_version (any change → every doc reprocessed, vectors reused)"), 13, 700, "start")
    x = 40
    for lab in [T("代码版本", "code version"), T("切块参数", "chunk params"), T("向量模型", "embed model"),
                T("词典 / 料号指纹", "dictionary fingerprint")]:
        x += s.chip(x, 358, lab, "warn", 12, mono=False) + 10

    # 删除语义
    s.rect(608, 312, 568, 184, *C["frame"], rx=10)
    s.text(624, 338, T("删除：全量 vs 增量（最容易出事故的地方）", "Deletes: full snapshot vs incremental"), 13, 700, "start")
    s.box(624, 352, 262, 128, [T("全量快照", "Full snapshot"), T("文件夹扫描、工单导出", "folder scans, exports"),
                              T("本次没出现 → 删除", "missing this run → delete"),
                              T("消失 > 50% → 拒绝删除", "> 50% missing → refuse"),
                              T("（网盘没挂上？）", "(drive not mounted?)")], "warn", sizes=[14, 12, 12, 12, 11])
    s.box(898, 352, 262, 128, [T("增量", "Incremental"), T("SharePoint delta 等", "SharePoint delta, etc."),
                              T("只认显式墓碑 Tombstone", "only explicit tombstones"),
                              T("“本批没出现”≠ 删除", "absent from batch ≠ deleted"),
                              T("游标在成功后才保存", "cursor saved on success")], "source",
          sizes=[14, 12, 12, 12, 11])
    s.text(40, 434, T("每次运行写一条 ingestion_runs：新增 / 更新 / 跳过 / 失败 / 删除 / 向量新算 / 复用",
                      "Each run writes ingestion_runs: added / updated / unchanged / failed / deleted / embeddings"),
           12, 400, "start", MUTED)
    s.text(40, 458, T("命令：servicekb ingest（增量，可反复运行）", "Command: servicekb ingest (safe to re-run)"),
           12, 600, "start", INK)
    return s


# =========================================================================== 4. 对象层


def object_layer(lang: str) -> SVG:
    def T(zh, en):
        return zh if lang == "zh" else en

    s = SVG(1200, 560, "object_layer")
    s.text(600, 38, T("对象层：从“按记录存”到“按对象存”", "Object layer: from records to objects"), 22, 700)

    # 左：原始记录
    s.frame(24, 62, 420, 470, T("原始记录（名字不统一、重复、过时）", "Raw records (inconsistent, duplicated, stale)"))
    cards = [
        (T("工单 TK-2024-0211", "Ticket TK-2024-0211"), T("“AS-110K HV 报E051 … 更换 310-04521-A”", "“AS-110K HV E051 … replace 310-04521-A”")),
        (T("工单 TK-2025-0042", "Ticket TK-2025-0042"), T("“AS110KHV 风扇异响 … 用 310-04521-B”", "“AS110KHV fan noise … use 310-04521-B”")),
        (T("物料表第 1 行", "Parts master row 1"), T("310-04521-A · 散热风扇组件 · 停产", "310-04521-A · cooling fan · obsolete")),
        (T("物料表第 17 行（重复录入）", "Parts master row 17 (duplicate)"), T("310-04521-A · 风扇总成", "310-04521-A · fan assy")),
        (T("服务通告 SB-2025-014", "Bulletin SB-2025-014"), T("“310-04521-A 停产，由 310-04521-B 替代”", "“310-04521-A discontinued → 310-04521-B”")),
        (T("手册 §4.2 E051", "Manual §4.2 E051"), T("“更换散热风扇组件 310-04521-B”", "“replace fan assembly 310-04521-B”")),
    ]
    for i, (h1, h2) in enumerate(cards):
        y = 100 + i * 70
        s.box(40, y, 388, 58, [h1, h2], "gray", sizes=[13, 12])

    # 中：箭头
    s.arrow([(452, 296), (520, 296)], C["store"][1], width=2.4)
    s.lines(486, 340, [T("规范化", "normalize"), T("实体链接", "link"), T("合并别名", "merge aliases")],
            [12, 12, 12], [600, 400, 400], [C["store"][1], MUTED, MUTED])

    # 右：对象图
    s.frame(528, 62, 648, 470, T("对象层（entities + entity_relations）", "Object layer (entities + entity_relations)"),
            kind="frame")
    s.box(722, 100, 260, 74, ["model:AS-110K-HV", T("别名：AS110K · as-110k-hv · 110K-HV", "aliases: AS110K · as-110k-hv · 110K-HV")],
          "source", sizes=[14, 11], mono=False)
    s.box(560, 262, 230, 86, ["part:310-04521-A", T("🔴 停产", "🔴 obsolete"),
                             T("别名：散热风扇组件 · 风扇总成", "aliases: cooling fan · fan assy")], "bad", sizes=[14, 12, 11])
    s.box(920, 262, 230, 86, ["part:310-04521-B", T("🟢 在售", "🟢 active"), T("带防尘网", "with dust filter")], "good",
          sizes=[14, 12, 11])
    s.box(752, 420, 200, 64, ["fault:E051", T("风扇故障", "fan fault")], "warn", sizes=[14, 12])

    s.arrow([(791, 305), (919, 305)], C["bad"][1], width=2, label="substituted_by", label_at=(855, 296))
    s.arrow([(700, 261), (700, 216), (790, 216), (790, 175)], LINE, label="fits", label_at=(730, 210))
    s.arrow([(1030, 261), (1030, 216), (912, 216), (912, 175)], LINE, label="fits", label_at=(1000, 210))
    s.arrow([(772, 349), (772, 419)], LINE, dash="4 3")
    s.arrow([(932, 349), (932, 419)], LINE, dash="4 3")
    s.text(852, 392, T("在同一批文档块里被提及", "co-mentioned in chunks"), 12, 500, "middle", MUTED)

    s.rect(984, 388, 176, 108, "#FFFFFF", "#D1D5DB", rx=10, sw=1)
    s.lines(1072, 442, [T("每个文档块都带", "Every chunk carries"), "entity_ids", T("→ 实体检索", "→ entity lane"),
                        T("→ 替代件扩展", "→ substitute expansion"), T("→ 结构化事实", "→ structured facts")],
            [12, 12, 11, 11, 11], [400, 700, 400, 400, 400], [MUTED, INK, MUTED, MUTED, MUTED])
    s.text(560, 516, T("问“310-04521-A 还能订到吗？” → 自动扩展到 B，并注入事实“A 已停产，被 B 替代”",
                       "Ask “Can I order 310-04521-A?” → expands to B and injects “A is obsolete, replaced by B”"),
           12, 600, "start", C["store"][1])
    return s


# =========================================================================== 5. 聊天记录清洗漏斗


def chat_cleaning(lang: str) -> SVG:
    def T(zh, en):
        return zh if lang == "zh" else en

    s = SVG(1200, 600, "chat_cleaning")
    s.text(600, 38, T("Teams 聊天记录清洗漏斗（示例数据真实运行结果）",
                      "Teams chat cleaning funnel (real run on the sample data)"), 22, 700)
    stages = [
        (T("原始消息", "Raw messages"), 34, T("Graph API 导出，原样落地", "exported via Graph, stored raw"), "source"),
        (T("过滤后", "After filters"), 30, T("−1 系统事件 · −1 机器人 · −1 已删除 · −1 DLP 拦截",
                                            "−1 system event · −1 bot · −1 deleted · −1 DLP-blocked"), "pipe"),
        (T("清洗", "Cleaned"), 30, T("HTML→文本 · 去引用回复 · 去粘贴邮件 · PII 脱敏 · 人名化名",
                                    "HTML→text · drop quotes · strip pasted email · redact PII · pseudonyms"), "pipe"),
        (T("合并连发", "Bursts merged"), 28, T("同一人 3 分钟内连发的消息合并", "same author within 3 min"), "pipe"),
        (T("线程 / 会话", "Threads"), 10, T("频道按 replyToId；群聊按时间间隔切分",
                                         "channels: replyToId · group chats: time-gap sessions"), "query"),
        (T("问答抽取", "Q&A extracted"), 10, T("问题 + 工程师答案 + 是否解决（提问者确认 / 点赞）",
                                             "question + engineer answer + resolved? (asker confirms / likes)"), "query"),
    ]
    cx, maxw = 300, 460
    for i, (name, n, note, kind) in enumerate(stages):
        y = 70 + i * 62
        w = max(160, maxw * n / 34)
        fill, stroke = C[kind]
        s.rect(cx - w / 2, y, w, 48, fill, stroke, rx=8)
        s.text(cx, y + 30, f"{name}  {n}", 15, 700)
        s.text(560, y + 30, note, 13, 400, "start", MUTED)
        if i:
            s.arrow([(cx, y - 13), (cx, y - 1)], LINE)

    y = 70 + len(stages) * 62 + 14
    outs = [
        (T("✅ 可直接入库", "✅ Ready to ingest"), "7", T("已解决且无需审核 → teams_qa.jsonl", "resolved, no review needed"), "good"),
        (T("📝 人工审核", "📝 Human review"), "2", T("未确认解决 / 涉及 1500V 安全操作", "unconfirmed / 1500V safety"), "warn"),
        (T("❓ 知识缺口", "❓ Knowledge gap"), "1", T("没人回答 → 交给技术支持补文档", "unanswered → write the doc"), "bad"),
    ]
    for i, (name, n, note, kind) in enumerate(outs):
        x = 24 + i * 390
        s.box(x, y + 18, 370, 70, [f"{name}  {n}", note], kind, sizes=[16, 12])
        s.arrow([(cx, y - 14), (cx, y + 2), (x + 185, y + 2), (x + 185, y + 17)], LINE)
    s.text(600, 580, T("命令：servicekb chat clean ｜ 详细指南：docs/09_chat_log_cleaning.md",
                       "Command: servicekb chat clean | Guide: docs/09_chat_log_cleaning.md"), 13, 600, "middle", INK)
    return s


DIAGRAMS = [architecture, query_flow, ingest_pipeline, object_layer, chat_cleaning]


def main() -> None:
    warnings: list[str] = []
    for fn in DIAGRAMS:
        for lang in ("zh", "en"):
            svg = fn(lang)
            svg.save(OUT / f"{fn.__name__}.{lang}.svg")
            warnings += [f"[{lang}] {w}" for w in svg.warnings]
            print(f"wrote {fn.__name__}.{lang}.svg")
    for w in warnings:
        print("WARN", w)


if __name__ == "__main__":
    main()
