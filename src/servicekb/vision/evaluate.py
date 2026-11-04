"""拍照识别评估。

标注格式（data/sample/photos/labels.jsonl）::

    {"file": "site_310-04521-B_label.jpg", "part_number": "310-04521-B", "has_label": true}
    {"file": "site_unknown_helmet.jpg", "part_number": null}       # 图库里没有的东西 → 应该"未识别"

指标（工业场景最关心的是最后一个）：
- top1 / top3      ：正确料号排第一 / 进前三
- unknown_rejected ：图库里没有的东西，系统是否正确地说"未识别"
- **confident_wrong**：系统给出了一个料号、没有任何"请确认"的提醒，而且是错的。
  这是会导致发错货、装错件的情况，必须接近 0。宁可多提醒，也不能自信地答错。

每个指标分"有标签 / 无标签"两组统计，并且可以关掉 OCR 只看视觉（--no-ocr），
用来回答"OCR 到底贡献了多少"。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class PhotoEvalReport:
    n: int
    metrics: dict[str, float]
    by_group: dict[str, dict[str, float]]
    per_photo: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# 拍照识别评估（{self.n} 张）", "", "| 指标 | 全部 | 有标签 | 无标签 |", "| --- | --- | --- | --- |"]
        for k in self.metrics:
            g = [self.by_group.get(x, {}).get(k) for x in ("label", "nolabel")]
            fmt = [f"{v:.3f}" if v is not None else "-" for v in [self.metrics[k], *g]]
            lines.append(f"| {k} | " + " | ".join(fmt) + " |")
        lines += ["", "| 照片 | 期望 | 系统第一名 | 结果 | 提醒 |", "| --- | --- | --- | --- | --- |"]
        for p in self.per_photo:
            lines.append(f"| {p['file']} | {p['expected'] or '（图库外）'} | {p['top'] or '未识别'} | {p['outcome']} | "
                         f"{'; '.join(p['warnings'])[:80]} |")
        return "\n".join(lines) + "\n"


def run_photo_eval(kb: Any, labels: Path, photo_dir: Path | None = None, use_ocr: bool = True) -> PhotoEvalReport:
    from servicekb.vision.ocr import NullOCR

    photo_dir = photo_dir or labels.parent
    ident = kb.identifier
    saved = ident.ocr
    if not use_ocr:
        ident.ocr = NullOCR()
    rows = [json.loads(line) for line in labels.read_text("utf-8").splitlines() if line.strip()]
    per: list[dict[str, Any]] = []
    try:
        for row in rows:
            res = kb.identify_photo(photo_dir / row["file"])
            exp = f"part:{row['part_number']}" if row.get("part_number") else None
            top = res.top.entity_id if res.top else None
            ranked = [c.entity_id for c in res.candidates] if res.identified else []
            warned = bool(res.lookalikes) or not res.identified
            if exp is None:
                outcome = "正确拒识" if not res.identified else ("错误识别" if not warned else "提醒确认")
            elif top == exp:
                outcome = "正确" if not res.lookalikes else "正确（附相似件提醒）"
            elif warned and exp in [top, *res.lookalikes]:
                outcome = "提醒确认"
            elif not res.identified:
                outcome = "未识别"
            else:
                outcome = "自信答错"
            per.append({
                "file": row["file"], "expected": exp and exp.split(":", 1)[1], "top": top and top.split(":", 1)[1],
                "group": "label" if row.get("has_label") else "nolabel", "outcome": outcome,
                "top1": float(top == exp) if exp else None,
                "top3": float(exp in ranked[:3]) if exp else None,
                "unknown_rejected": float(not res.identified) if exp is None else None,
                "confident_wrong": float(res.identified and not warned and top != exp),
                "warnings": res.warnings,
            })
    finally:
        ident.ocr = saved

    def agg(items: list[dict[str, Any]]) -> dict[str, float]:
        out = {}
        for k in ("top1", "top3", "unknown_rejected", "confident_wrong"):
            vals = [p[k] for p in items if p[k] is not None]
            out[k] = round(mean(vals), 4) if vals else 0.0
        return out

    return PhotoEvalReport(
        n=len(per), metrics=agg(per),
        by_group={g: agg([p for p in per if p["group"] == g]) for g in ("label", "nolabel")},
        per_photo=per,
    )


def save_photo_report(report: PhotoEvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "photo_eval_report.md"
    md.write_text(report.to_markdown(), "utf-8")
    (out_dir / "photo_eval_report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), "utf-8")
    return md
