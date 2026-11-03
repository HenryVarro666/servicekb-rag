"""生成拍照识别的示例数据（全部是程序画的虚构零件，不是真实产品照片）。

    python scripts/make_sample_part_images.py

输出：
- data/sample/part_images/<料号>/*.png   参考图库（"目录照片 / CAD 渲染图"），白底
- data/sample/photos/*.jpg               "现场照片"：旋转、缩放、杂乱背景、模糊、噪声，部分贴了铭牌贴纸
- data/sample/photos/labels.jsonl        现场照片的标注（正确料号），用于 servicekb photo eval

刻意设计了两组"长得几乎一样"的零件，用来演示为什么必须结合 OCR：
- 310-04521-A / 310-04521-B 散热风扇（B 只多了一层细防尘网）
- 220-11008-A / 220-11009-A 直流开关（1100V / 1500V，外形相同，装错有安全风险）

贴纸上的文字同时写进 JPEG 的 comment 字段，供 MetadataOCR（测试替身）读取；
装了 RapidOCR 时会直接从像素里识别。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "sample" / "part_images"
PHOTO_DIR = ROOT / "data" / "sample" / "photos"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
S = 384  # 参考图尺寸


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- 零件画法


def _fan(d: ImageDraw.ImageDraw, dust_filter: bool) -> None:
    d.rounded_rectangle((42, 42, 342, 342), 26, fill=(62, 64, 70), outline=(35, 35, 40), width=4)
    for cx, cy in [(68, 68), (316, 68), (68, 316), (316, 316)]:
        d.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=(235, 235, 235))
    d.ellipse((62, 62, 322, 322), fill=(28, 28, 32))
    for k in range(7):
        a = 2 * math.pi * k / 7
        pts = [(192 + 34 * math.cos(a), 192 + 34 * math.sin(a)),
               (192 + 125 * math.cos(a + 0.25), 192 + 125 * math.sin(a + 0.25)),
               (192 + 118 * math.cos(a + 0.72), 192 + 118 * math.sin(a + 0.72)),
               (192 + 34 * math.cos(a + 0.6), 192 + 34 * math.sin(a + 0.6))]
        d.polygon(pts, fill=(88, 90, 98))
    d.ellipse((152, 152, 232, 232), fill=(110, 112, 120), outline=(50, 50, 55), width=3)
    if dust_filter:  # 新版：带防尘网
        for x in range(66, 320, 9):
            d.line((x, 66, x, 318), fill=(150, 152, 158), width=1)
        for y in range(66, 320, 9):
            d.line((66, y, 318, y), fill=(150, 152, 158), width=1)


def _ext_fan(d: ImageDraw.ImageDraw) -> None:
    d.ellipse((30, 30, 354, 354), fill=(52, 78, 112), outline=(25, 40, 60), width=5)
    d.ellipse((70, 70, 314, 314), fill=(20, 28, 40))
    for k in range(5):
        a = 2 * math.pi * k / 5
        pts = [(192 + 30 * math.cos(a), 192 + 30 * math.sin(a)),
               (192 + 118 * math.cos(a + 0.4), 192 + 118 * math.sin(a + 0.4)),
               (192 + 105 * math.cos(a + 1.0), 192 + 105 * math.sin(a + 1.0))]
        d.polygon(pts, fill=(120, 140, 170))
    for r in (40, 75, 110):  # 金属防护网圈
        d.ellipse((192 - r, 192 - r, 192 + r, 192 + r), outline=(200, 205, 210), width=3)
    d.line((70, 192, 314, 192), fill=(200, 205, 210), width=3)
    d.line((192, 70, 192, 314), fill=(200, 205, 210), width=3)


def _board(d: ImageDraw.ImageDraw, rev: str) -> None:
    d.rectangle((30, 70, 354, 314), fill=(22, 112, 62), outline=(12, 70, 38), width=4)
    for cx, cy in [(46, 86), (338, 86), (46, 298), (338, 298)]:
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(220, 190, 90))
    chips = [(80, 110, 160, 190), (190, 110, 240, 150), (190, 170, 300, 230), (80, 220, 140, 270)]
    if rev == "D":  # 新版：布局略有不同，多一颗芯片
        chips = [(80, 110, 170, 200), (200, 110, 250, 150), (200, 170, 310, 230), (80, 225, 140, 275),
                 (260, 110, 310, 150)]
    for c in chips:
        d.rectangle(c, fill=(25, 25, 28))
    for x in range(60, 330, 14):
        d.rectangle((x, 290, x + 7, 312), fill=(215, 180, 70))
    for x in range(250, 340, 18):
        d.ellipse((x, 250, x + 10, 260), fill=(200, 200, 200))


def _comm_board(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((40, 90, 344, 294), fill=(32, 72, 150), outline=(18, 40, 90), width=4)
    for i in range(6):  # 天线
        y = 110 + i * 12
        d.line((60, y, 170, y), fill=(220, 190, 90), width=3)
    d.rectangle((210, 200, 270, 280), fill=(190, 192, 196))
    d.rectangle((280, 200, 334, 280), fill=(190, 192, 196))
    d.rectangle((70, 200, 150, 250), fill=(25, 25, 28))


def _dc_switch(d: ImageDraw.ImageDraw, rating: str) -> None:
    top = 60 if rating == "1500V" else 80
    d.rounded_rectangle((90, top, 294, 330), 14, fill=(150, 152, 158), outline=(80, 80, 86), width=4)
    d.ellipse((132, top + 40, 252, top + 160), fill=(200, 30, 30), outline=(120, 10, 10), width=4)
    d.rounded_rectangle((182, top + 30, 202, top + 170), 8, fill=(160, 20, 20))
    d.rectangle((120, 260, 264, 300), fill=(250, 210, 40))
    d.text((192, 280), rating, fill=(20, 20, 20), font=font(26), anchor="mm")


def _spd(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((132, 40, 252, 344), fill=(214, 214, 208), outline=(120, 120, 115), width=4)
    d.rectangle((160, 150, 224, 200), fill=(40, 170, 70), outline=(60, 60, 60), width=2)
    for y in (60, 314):
        d.ellipse((176, y - 2, 208, y + 30), fill=(160, 160, 160), outline=(90, 90, 90), width=2)
    d.rectangle((150, 230, 234, 250), fill=(60, 60, 60))


def _side_cover(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle((24, 54, 360, 330), 10, fill=(188, 190, 194), outline=(110, 112, 116), width=4)
    for x in (44, 192, 340):
        for y in (74, 310):
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(90, 90, 95))
    for x in (44, 340):
        d.ellipse((x - 8, 184, x + 8, 200), fill=(90, 90, 95))
    for i in range(7):
        y = 120 + i * 24
        d.rounded_rectangle((110, y, 274, y + 10), 4, fill=(120, 122, 126))


def _mc4(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle((40, 160, 260, 224), 20, fill=(24, 24, 26))
    d.rounded_rectangle((250, 168, 350, 216), 12, fill=(36, 36, 40))
    for x in range(70, 140, 12):
        d.line((x, 162, x, 222), fill=(70, 70, 75), width=4)
    d.ellipse((20, 176, 52, 208), fill=(60, 60, 64))


PARTS = {
    "310-04521-A": lambda d: _fan(d, dust_filter=False),
    "310-04521-B": lambda d: _fan(d, dust_filter=True),
    "310-04522-A": _ext_fan,
    "410-20031-C": lambda d: _board(d, "C"),
    "410-20031-D": lambda d: _board(d, "D"),
    "410-20077-A": _comm_board,
    "220-11008-A": lambda d: _dc_switch(d, "1100V"),
    "220-11009-A": lambda d: _dc_switch(d, "1500V"),
    "520-00310-A": _spd,
    "730-00012-A": _side_cover,
    "610-00045-A": _mc4,
}
MODEL_CODE = {"310-04521-A": "11", "310-04521-B": "11", "310-04522-A": "25", "410-20031-C": "11",
              "410-20031-D": "11", "410-20077-A": "08", "220-11008-A": "11", "220-11009-A": "25",
              "520-00310-A": "50", "730-00012-A": "11", "610-00045-A": "11"}


def render(pn: str, bg=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", (S, S), bg)
    PARTS[pn](ImageDraw.Draw(img))
    return img


# --------------------------------------------------------------------------- 现场照片


def background(w: int, h: int, rng: np.random.Generator, kind: int) -> Image.Image:
    if kind == 0:  # 水泥地
        base = rng.normal(150, 18, (h // 4, w // 4, 1)).repeat(3, axis=2)
    elif kind == 1:  # 米色机柜
        base = np.ones((h // 4, w // 4, 3)) * np.array([214, 204, 178]) + rng.normal(0, 6, (h // 4, w // 4, 3))
    else:  # 深色地垫
        base = np.ones((h // 4, w // 4, 3)) * np.array([70, 76, 84]) + rng.normal(0, 10, (h // 4, w // 4, 3))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return img.filter(ImageFilter.GaussianBlur(1.5))


def site_photo(pn: str, seed: int, label: str | None) -> Image.Image:
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    W, H = 640, 480
    bg = background(W, H, rng, seed % 3)
    part = render(pn, bg=(255, 255, 255)).convert("RGBA")
    # 白底抠掉（参考图是白底，现场照片里零件放在别的背景上）
    a = np.asarray(part).copy()
    white = (a[:, :, :3] > 245).all(axis=2)
    a[white, 3] = 0
    part = Image.fromarray(a)
    part = part.rotate(pyrng.uniform(-18, 18), resample=Image.BICUBIC, expand=True)
    scale = pyrng.uniform(0.85, 1.15) * H / part.height
    part = part.resize((int(part.width * scale), int(part.height * scale)), Image.BICUBIC)
    x = pyrng.randint(10, max(11, W - part.width - 10))
    y = pyrng.randint(5, max(6, H - part.height - 5))
    bg.paste(part, (x, y), part)
    img = ImageEnhance.Brightness(bg).enhance(pyrng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(pyrng.uniform(0.85, 1.1))
    if label:  # 铭牌贴纸
        d = ImageDraw.Draw(img)
        lines = label.split("\n")
        f = font(20)
        tw = max(int(d.textlength(ln, font=f)) for ln in lines) + 24
        th = 28 * len(lines) + 16
        sx, sy = (W - tw - 12, H - th - 12) if x < W / 2 else (12, H - th - 12)
        d.rectangle((sx, sy, sx + tw, sy + th), fill=(248, 248, 244), outline=(40, 40, 40), width=2)
        for i, ln in enumerate(lines):
            d.text((sx + 12, sy + 10 + 28 * i), ln, fill=(15, 15, 15), font=f)
    img = img.filter(ImageFilter.GaussianBlur(pyrng.uniform(0.0, 0.9)))
    noise = rng.normal(0, 5, (H, W, 3))
    return Image.fromarray(np.clip(np.asarray(img, dtype=np.float64) + noise, 0, 255).astype(np.uint8))


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    for pn in PARTS:
        d = REF_DIR / pn
        d.mkdir(exist_ok=True)
        render(pn).save(d / "front.png", optimize=True)
        alt = render(pn).rotate(8, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        ImageEnhance.Brightness(alt).enhance(0.93).save(d / "render_rotated.png", optimize=True)

    labels = []
    for i, pn in enumerate(PARTS):
        f1 = f"site_{pn}_nolabel.jpg"
        site_photo(pn, 100 + i, None).save(PHOTO_DIR / f1, quality=88)
        labels.append({"file": f1, "part_number": pn, "has_label": False})
        yymm = f"{22 + (i * 2) // 12:02d}{(i * 2) % 12 + 1:02d}"  # 出厂年月：2022-01 起每隔两个月一台
        serial = f"AS{MODEL_CODE[pn]}{yymm}{4560 + i:06d}"
        text = f"P/N {pn}\nSN {serial}"
        f2 = f"site_{pn}_label.jpg"
        site_photo(pn, 200 + i, text).save(PHOTO_DIR / f2, quality=92, comment=text.encode())
        labels.append({"file": f2, "part_number": pn, "has_label": True, "serial": serial})
    # 一张"图库里没有"的东西：安全帽。系统应当回答"未识别"，而不是硬猜一个料号
    unk = background(640, 480, np.random.default_rng(7), 0)
    d = ImageDraw.Draw(unk)
    d.pieslice((200, 120, 440, 360), 180, 360, fill=(245, 200, 20), outline=(150, 120, 0), width=4)
    d.rectangle((180, 236, 460, 256), fill=(230, 185, 15))
    unk.save(PHOTO_DIR / "site_unknown_helmet.jpg", quality=88)
    labels.append({"file": "site_unknown_helmet.jpg", "part_number": None, "has_label": False})
    with (PHOTO_DIR / "labels.jsonl").open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(PARTS)} parts, {len(labels)} photos")


if __name__ == "__main__":
    main()
