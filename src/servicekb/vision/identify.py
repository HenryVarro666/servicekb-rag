"""拍照识别备件：照片 → 候选料号（带证据）→ 交给知识库回答。

三路信号（和文本检索的"三路召回"是同一个思路：各有所长，互相补短板）::

    照片 ──┬─ ① 视觉：整图 + 切块 → 图像向量 → 最像的参考图 → 料号候选（擅长"大概是什么"）
           ├─ ② OCR：标签 / 铭牌文字 → 实体链接 → 料号、型号（擅长"到底是哪一个"）
           └─ ③ 序列号：按规则解析出厂年月（决定是否需要转接件等）
                     ↓ 融合
        候选列表 + 长得像提醒 + 未识别判断 + 结构化事实（停产 / 替代件）

最重要的设计原则：**宁可说"请拍铭牌确认"，也不要自信地给出一个错误料号。**
外观几乎相同的零件（新旧版本、1100V / 1500V）只靠视觉是分不开的，系统必须把这一点说出来。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from servicekb.logging import get_logger
from servicekb.processing.entities import EntityLinker
from servicekb.store.base import Store
from servicekb.vision.embedders import ImageEmbedder
from servicekb.vision.images import border_color, load_image, make_crops
from servicekb.vision.ocr import OCREngine, normalize_ocr_text

log = get_logger(__name__)

OCR_BONUS = 1.0  # 标签上读到了料号：这是最强的证据，直接排到视觉候选前面
SPEC_BONUS = 0.05  # 零件本体上印的规格（例如 "1500V"）与候选料号名称一致：小幅加分，用来区分外观相同的件
_SPEC = re.compile(r"\d+(?:\.\d+)?\s?(?:KV|V|A|KW|W|MM|HZ)\b", re.IGNORECASE)


@dataclass
class PartCandidate:
    entity_id: str
    part_number: str
    name: str
    status: str
    score: float
    visual_score: float | None = None
    ocr_confirmed: bool = False
    best_reference: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class IdentifyResult:
    candidates: list[PartCandidate]
    identified: bool
    ocr_text: str = ""
    ocr_entities: list[str] = field(default_factory=list)
    serial_number: str | None = None
    manufacture_date: str | None = None
    lookalikes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top(self) -> PartCandidate | None:
        return self.candidates[0] if self.candidates and self.identified else None

    def hint_entities(self) -> list[str]:
        """交给知识库检索的实体：OCR 确认的料号；否则视觉第一名 + 长得像的候选。"""
        confirmed = [c.entity_id for c in self.candidates if c.ocr_confirmed]
        if confirmed:
            return confirmed
        if not self.identified:
            return []
        return list(dict.fromkeys([self.candidates[0].entity_id, *self.lookalikes]))[:3]


class PartIdentifier:
    def __init__(
        self,
        store: Store,
        embedder: ImageEmbedder,
        ocr: OCREngine,
        linker: EntityLinker,
        *,
        tile_grid: int = 2,
        tile_min_side: int = 1200,
        top_k_refs: int = 20,
        min_visual_score: float = 0.80,
        lookalike_margin: float = 0.03,
        serial_pattern: str = r"\bAS\d{2}(?P<year>\d{2})(?P<month>0[1-9]|1[0-2])\d{6}\b",
    ) -> None:
        self.store, self.embedder, self.ocr, self.linker = store, embedder, ocr, linker
        self.tile_grid, self.tile_min_side, self.top_k_refs = tile_grid, tile_min_side, top_k_refs
        self.min_visual_score, self.lookalike_margin = min_visual_score, lookalike_margin
        self.serial_re = re.compile(serial_pattern)

    # ------------------------------------------------------------------ 各路信号
    def _visual(self, img: Image.Image, acl_groups: list[str]) -> dict[str, tuple[float, str]]:
        """每个料号取"所有切块 × 所有参考图"里的最高相似度。"""
        crops = make_crops(img, self.tile_grid, self.tile_min_side)
        best: dict[str, tuple[float, str]] = {}
        for vec in self.embedder.embed_images(crops):
            for rec, score in self.store.image_search(vec, self.top_k_refs, acl_groups):
                if score > best.get(rec.entity_id, (-1.0, ""))[0]:
                    best[rec.entity_id] = (score, rec.uri)
        return best

    def _is_label_text(self, line: str) -> bool:
        return bool(self.linker.part_re.search(line) or self.serial_re.search(line.upper())
                    or re.search(r"\b(P/N|S/N|PN|SN|MODEL)\b|料号|序列号|型号", line, re.IGNORECASE))

    @staticmethod
    def _mask_text(img: Image.Image, boxes: list[tuple[int, int, int, int]], pad: int = 12) -> Image.Image:
        """把 OCR 找到的文字区域（向外扩一圈，盖住贴纸边框）涂成背景色。"""
        if not boxes:
            return img
        from PIL import ImageDraw

        out = img.copy()
        d = ImageDraw.Draw(out)
        fill = border_color(img)
        for x0, y0, x1, y1 in boxes:
            d.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=fill)
        return out

    def _serial(self, text: str) -> tuple[str | None, str | None]:
        m = self.serial_re.search(text.upper())
        if not m:
            return None, None
        date = None
        if "year" in m.groupdict() and "month" in m.groupdict():
            date = f"20{m.group('year')}-{m.group('month')}"
        return m.group(0), date

    # ------------------------------------------------------------------ 主流程
    def identify(self, image: bytes | str | Path | Image.Image, acl_groups: list[str], top_k: int = 5) -> IdentifyResult:
        timings: dict[str, float] = {}
        img = load_image(image)

        # 先 OCR：既要读文字，也要知道文字在哪——视觉匹配前把标签 / 贴纸涂掉。
        # 标签、贴纸、手写字每台机器都不一样，留着只会干扰"长得像哪个零件"的判断。
        t = time.perf_counter()
        warnings: list[str] = []
        try:
            raw, lines = self.ocr.read_with_boxes(img)
            text = normalize_ocr_text(raw)
        except Exception as e:  # OCR 挂了不影响视觉结果
            log.exception("ocr failed")
            text, lines, warnings = "", [], [f"OCR 失败：{e!r}"]
        timings["ocr"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        # 只涂掉"像标签"的文字（含料号 / 序列号 / 标签键）。OCR 的误检、零件本体上印的规格（如 1500V）
        # 都属于零件外观的一部分，不能涂。
        label_boxes = [box for line, box in lines if self._is_label_text(normalize_ocr_text(line))]
        visual = self._visual(self._mask_text(img, label_boxes), acl_groups)
        timings["visual"] = (time.perf_counter() - t) * 1000
        mentions = self.linker.extract(text) if text else []
        ocr_parts = [m.entity_id for m in mentions if m.entity_id.startswith("part:")]
        serial, mdate = self._serial(text)

        # ---- 融合
        specs = {m.group(0).replace(" ", "").upper() for m in _SPEC.finditer(text)}
        cands: list[PartCandidate] = []
        for eid in dict.fromkeys([*ocr_parts, *visual]):
            e = self.linker.entities.get(eid)
            vs, ref = visual.get(eid, (None, None))
            ocr_ok = eid in ocr_parts
            ev = []
            if ocr_ok:
                ev.append(f"标签文字识别到 {eid.split(':', 1)[1]}")
            if vs is not None:
                ev.append(f"外观与参考图相似度 {vs:.2f}")
            names = " ".join([e.display_name, *e.aliases]).upper().replace(" ", "") if e else ""
            spec_hit = sorted(sp for sp in specs if sp in names)
            if spec_hit:
                ev.append(f"本体上印的规格 {'/'.join(spec_hit)} 与该料号一致")
            cands.append(PartCandidate(
                entity_id=eid,
                part_number=eid.split(":", 1)[1],
                name=e.display_name if e else "（物料主数据中没有）",
                status=str(e.attributes.get("status", "未知")) if e else "未知",
                score=(vs or 0.0) + (OCR_BONUS if ocr_ok else 0.0) + (SPEC_BONUS if spec_hit else 0.0),
                visual_score=vs, ocr_confirmed=ocr_ok, best_reference=ref, evidence=ev,
            ))
            if ocr_ok and e is None:
                warnings.append(f"标签上的料号 {eid.split(':', 1)[1]} 不在物料主数据中（新料号？识别错误？），请人工确认。")
        cands.sort(key=lambda c: (-c.score, c.entity_id))
        cands = cands[:top_k]

        best_vis = max((c.visual_score or 0.0) for c in cands) if cands else 0.0
        top = cands[0] if cands else None
        identified = bool(top and (top.ocr_confirmed or (top.visual_score or 0.0) >= self.min_visual_score))

        # ---- 长得像：没有 OCR 确认时，和第一名分数接近的候选都要提醒
        lookalikes: list[str] = []
        if top and identified and not top.ocr_confirmed and top.visual_score is not None:
            # 规则 1：分数（外观 + 本体规格）非常接近
            # 规则 2：物料主数据里有替代 / 互换关系的新旧版本，外观通常很像，放宽到 0.10
            related = {(r.dst if r.src == top.entity_id else r.src)
                       for r in self.linker.neighbors(top.entity_id, ("substituted_by", "interchangeable_with"))}
            for c in cands[1:]:
                if c.visual_score is None:
                    continue
                gap = top.score - c.score
                if gap <= self.lookalike_margin or (c.entity_id in related and gap <= 0.10):
                    lookalikes.append(c.entity_id)
            if lookalikes:
                names = " / ".join(x.split(":", 1)[1] for x in [top.entity_id, *lookalikes])
                why = "（物料主数据中为新旧版本 / 可互换件）" if any(x in related for x in lookalikes) else ""
                warnings.append(f"以下料号外观几乎相同{why}：{names}。请拍摄标签或铭牌确认，不要仅凭外观下单。")
        if top and top.ocr_confirmed and top.visual_score is not None and best_vis - top.visual_score > 0.15:
            warnings.append("标签文字与外观不太一致（标签贴错？拍到了别的零件的标签？），请人工确认。")
        if not identified:
            warnings.append(
                f"未能可靠识别：外观最高相似度 {best_vis:.2f} 低于阈值 {self.min_visual_score:.2f}，"
                "且没有读到料号标签。请拍摄标签 / 铭牌，或转人工。"
            )

        # ---- 结构化事实（交给 LLM / 展示给用户）
        facts: list[str] = []
        if identified:
            ids = [c.entity_id for c in cands if c.ocr_confirmed] or [top.entity_id, *lookalikes]  # type: ignore[union-attr]
            facts += self.linker.facts_for(ids)
        if serial:
            facts.append(f"照片中的序列号 {serial}" + (f"，按序列号规则解析出厂年月为 {mdate}。" if mdate else "。"))

        return IdentifyResult(
            candidates=cands, identified=identified, ocr_text=text,
            ocr_entities=[m.entity_id for m in mentions], serial_number=serial, manufacture_date=mdate,
            lookalikes=lookalikes, warnings=warnings, facts=facts,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )
