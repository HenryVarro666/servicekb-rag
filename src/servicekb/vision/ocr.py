"""OCR：读出照片里的铭牌、板卡标签、贴纸上的文字。

为什么备件识别离不开 OCR？
    同一个零件的新旧版本（例如 310-04521-A / -B）外观几乎一样，额定 1100V 和 1500V 的直流开关
    也长得一模一样——**图像模型无法、也不应该替你区分它们**。真正决定"是哪一个料号"的是标签上印的字。
    视觉负责"大概是哪一类、哪几个候选"，OCR 负责"到底是哪一个"。

实现：
- ``RapidOCREngine``：RapidOCR（PaddleOCR 模型的 ONNX 版本），中英文都好、CPU 就能跑，``pip install -e ".[ocr]"``。
- ``MetadataOCR``：**测试替身**。只读取样例图片里预先写入的标签文字（JPEG comment / PNG 文本块），
  真实照片上永远返回空字符串。用于没装 OCR 时的离线 demo 和 CI。
- ``NullOCR``：关闭 OCR。

条码 / 二维码（序列号常以条码形式出现）可以加一个 pyzbar / zxing-cpp 的实现，接口相同，见 docs/10。
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from servicekb.logging import get_logger

log = get_logger(__name__)

METADATA_KEY = "servicekb:ocr"


Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)


class OCREngine(ABC):
    name: str

    @abstractmethod
    def read(self, img: Image.Image) -> str:
        """返回图片里识别出的全部文字（多行用换行连接）。"""

    def read_with_boxes(self, img: Image.Image) -> tuple[str, list[tuple[str, Box]]]:
        """全部文字 + 每一行文字及其矩形框。框用于在视觉匹配前把标签 / 贴纸涂掉（见 identify.py）。"""
        return self.read(img), []


class NullOCR(OCREngine):
    name = "none"

    def read(self, img: Image.Image) -> str:
        return ""


class MetadataOCR(OCREngine):
    name = "metadata"

    def read(self, img: Image.Image) -> str:
        raw = img.info.get(METADATA_KEY) or img.info.get("comment") or ""
        return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


class RapidOCREngine(OCREngine):
    name = "rapidocr"

    def __init__(self, min_confidence: float = 0.5) -> None:
        from rapidocr_onnxruntime import RapidOCR  # 可选依赖

        self.engine = RapidOCR()
        self.min_confidence = min_confidence

    def read(self, img: Image.Image) -> str:
        return self.read_with_boxes(img)[0]

    def read_with_boxes(self, img: Image.Image) -> tuple[str, list[tuple[str, Box]]]:
        result, _ = self.engine(np.asarray(img.convert("RGB"))[:, :, ::-1])  # RapidOCR 期望 BGR
        lines: list[tuple[str, Box]] = []
        for pts, text, conf in result or []:
            if float(conf) < self.min_confidence:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            lines.append((text, (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
        return "\n".join(t for t, _ in lines), lines


# 标签上常见的"键"。OCR 经常把键和值粘在一起（"P/N310-04521-B"、"SNAS112203004561"），
# 导致正则里的 \b 失效、料号认不出来。在键后面补一个空格。
_LABEL_KEYS = re.compile(r"(P\s*/\s*N|S\s*/\s*N|PN|SN|料号|物料号|序列号|型号|MODEL|PART\s*NO\.?)\s*[:：#]?\s*",
                         re.IGNORECASE)
# 料号里数字位置上常见的 OCR 混淆：O→0、I / l→1（只在"像料号"的片段里替换，不动普通文字）
_PN_LIKE = re.compile(r"(?<![A-Za-z0-9])([0-9OIl]{3})-([0-9OIl]{5})-([A-Z])(?![A-Za-z0-9])")
_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "l": "1"})


def normalize_ocr_text(text: str) -> str:
    """OCR 结果的后处理：全角转半角、拆开粘连的"键+值"、修正料号里的常见混淆字符。"""
    text = unicodedata.normalize("NFKC", text)
    text = _LABEL_KEYS.sub(lambda m: m.group(1).replace(" ", "").upper() + " ", text)
    return _PN_LIKE.sub(lambda m: f"{m.group(1).translate(_DIGIT_FIX)}-{m.group(2).translate(_DIGIT_FIX)}-{m.group(3)}",
                        text)


def build_ocr(backend: str) -> OCREngine:
    if backend == "none":
        return NullOCR()
    if backend == "metadata":
        return MetadataOCR()
    try:
        return RapidOCREngine()
    except ImportError:
        if backend == "rapidocr":
            raise
        log.warning("rapidocr not installed; OCR falls back to the metadata test double (real photos get no OCR). "
                    'Install with: pip install -e ".[ocr]"')
        return MetadataOCR()
