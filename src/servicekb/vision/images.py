"""图片基础处理：读取、方向校正、补成正方形、高分辨率切块。

这些看起来不起眼的步骤，对识别效果影响很大：
- **EXIF 方向**：手机拍的照片像素是横的，靠 EXIF 标记告诉看图软件"请旋转 90°"。
  不处理的话模型看到的是一张躺倒的零件。
- **补成正方形**：SigLIP 的预处理会把图片直接拉伸成正方形（例如 384×384），
  长条形的零件会被压扁。先用背景色补成正方形再缩放，形状不变。
- **切块（tiling）**：现场照片动辄 3000~4000 像素，而 SigLIP 输入只有 224~512 像素。
  整图缩放后，一个占画面 1/10 的零件只剩几十个像素，标签上的小字更是完全看不见。
  所以除了整图，再把图片切成 grid×grid 块（带重叠）分别编码，取最相似的那块。
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 80_000_000  # 防止"解压炸弹"图片拖垮服务（约 9000×9000）


def load_image(src: bytes | str | Path | Image.Image) -> Image.Image:
    """读取图片 → 按 EXIF 方向转正 → RGB。"""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    img.load()
    info = dict(img.info)  # exif_transpose 会丢掉 info（里面可能有样例图片的标签文字）
    img = ImageOps.exif_transpose(img) or img
    img = img.convert("RGB")
    img.info.update(info)
    return img


def image_bytes(src: bytes | str | Path) -> bytes:
    return bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def border_color(img: Image.Image) -> tuple[int, int, int]:
    """取图片四周一圈像素的中位数作为"背景色"。"""
    a = np.asarray(img)
    b = max(2, min(a.shape[0], a.shape[1]) // 20)
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return tuple(int(x) for x in np.median(border, axis=0))  # type: ignore[return-value]


def pad_to_square(img: Image.Image, fill: tuple[int, int, int] | None = None) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill or border_color(img))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def has_content(img: Image.Image, min_fraction: float = 0.01, delta: float = 25.0) -> bool:
    """几乎纯色的切块（只有墙面 / 地面）不参与匹配，否则会和"大块平板"类零件误匹配。

    不用整块的标准差：一张 2000 像素的切块里只有一个小零件时，标准差很低，但它恰恰是最有用的那块。
    改为看"和主色明显不同的像素占比"。
    """
    a = np.asarray(img.convert("L").reduce(4), dtype=np.float32)
    return float((np.abs(a - np.median(a)) > delta).mean()) >= min_fraction


def make_crops(img: Image.Image, grid: int = 2, min_side: int = 1200, overlap: float = 0.25) -> list[Image.Image]:
    """整图 + （大图时）grid×grid 个带重叠的切块。"""
    crops = [img]
    w, h = img.size
    if grid < 2 or max(w, h) < min_side:
        return crops
    tw, th = int(w / grid * (1 + overlap)), int(h / grid * (1 + overlap))
    for gy in range(grid):
        for gx in range(grid):
            x0 = min(int(gx * w / grid - tw * overlap / 2), w - tw)
            y0 = min(int(gy * h / grid - th * overlap / 2), h - th)
            x0, y0 = max(0, x0), max(0, y0)
            tile = img.crop((x0, y0, x0 + tw, y0 + th))
            if has_content(tile):
                crops.append(tile)
    return crops
