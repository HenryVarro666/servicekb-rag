"""图像向量模型。

SigLIP 是什么？
---------------
SigLIP（Sigmoid Loss for Language-Image Pre-training，Google 2023；SigLIP 2 于 2025 年发布）是和
CLIP 同类的"图文双塔"模型：一个图像编码器、一个文本编码器，在海量"图片-描述"对上训练，
让配对的图片和文字向量靠近。和 CLIP 的区别是把 softmax 对比损失换成了逐对的 sigmoid 损失，
训练更省显存、小 batch 也稳定，图像特征质量通常更好。

在备件识别里我们主要用它的**图像编码器**：把参考图（目录照片、CAD 渲染图）和现场照片都编码成向量，
"现场照片 → 最像的参考图 → 对应料号"。文本编码器只适合短描述（默认 64 token），
**不要**拿它替代知识库的文本向量模型（bge-m3）。

三个实现：
- ``HandcraftedImageEmbedder``：颜色直方图 + LBP 纹理 + 径向形状特征。零依赖、确定性、**没有语义**，
  只能区分示例数据里程序画出来的零件；用于 demo 和 CI，不能用于真实照片。
- ``SiglipImageEmbedder``：transformers 加载 SigLIP / SigLIP 2（推荐 google/siglip-so400m-patch14-384）。
- ``HTTPImageEmbedder``：调用独立部署的图像编码服务（团队已有 SigLIP worker 时直接复用）。
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

import httpx
import numpy as np
from PIL import Image, ImageFilter

from servicekb.config import Settings
from servicekb.vision.images import border_color, pad_to_square


class ImageEmbedder(ABC):
    dim: int

    @property
    @abstractmethod
    def model_id(self) -> str:
        """唯一标识这个向量空间，写进 index_meta。换模型 = 重建图片索引。"""

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


# --------------------------------------------------------------------------- 手工特征（离线 demo）


def _lbp_riu2(gray: np.ndarray) -> np.ndarray:
    """旋转不变的"均匀 LBP"（10 个取值）：描述局部纹理，例如防尘网的细格子。"""
    c = gray[1:-1, 1:-1]
    h, w = gray.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = np.stack([gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] >= c + 3 for dy, dx in offs]).astype(np.uint8)
    transitions = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(transitions <= 2, bits.sum(axis=0), 9)


def _soft_hist(vals: np.ndarray, bins: tuple[int, int, int], hue_cyclic: bool) -> np.ndarray:
    """三维软分箱直方图（三线性插值）。vals: N×3，取值 0~255。"""
    lo, frac = [], []
    for d, n in enumerate(bins):
        pos = vals[:, d] / 256 * n - 0.5
        f = np.floor(pos)
        lo.append(f.astype(int))
        frac.append(pos - f)
    hist = np.zeros(bins, dtype=np.float64)
    for corner in range(8):
        idx, w = [], np.ones(len(vals))
        for d, n in enumerate(bins):
            up = (corner >> d) & 1
            i = lo[d] + up
            i = np.mod(i, n) if (d == 0 and hue_cyclic) else np.clip(i, 0, n - 1)
            idx.append(i)
            w = w * (frac[d] if up else 1 - frac[d])
        np.add.at(hist, tuple(idx), w)
    return hist.ravel()


class HandcraftedImageEmbedder(ImageEmbedder):
    SIZE = 128

    def __init__(self) -> None:
        self.dim = 12 * 3 * 3 + 10 + 16 + 3

    @property
    def model_id(self) -> str:
        return f"handcrafted-v2-{self.dim}"

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """在缩小 4 倍的掩码上做连通域分析，只保留面积最大的一块（去掉贴纸、杂物、噪点）。"""
        h, w = mask.shape
        small = mask[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3)) > 0.3
        labels = np.zeros(small.shape, dtype=np.int32)
        best, best_size, cur = 0, 0, 0
        for y0, x0 in zip(*np.nonzero(small), strict=True):
            if labels[y0, x0]:
                continue
            cur += 1
            stack, size = [(y0, x0)], 0
            labels[y0, x0] = cur
            while stack:
                y, x = stack.pop()
                size += 1
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < small.shape[0] and 0 <= nx < small.shape[1] and small[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = cur
                        stack.append((ny, nx))
            if size > best_size:
                best, best_size = cur, size
        keep = np.kron(labels == best, np.ones((4, 4), dtype=bool)) if best else np.zeros_like(small)
        out = np.zeros_like(mask)
        out[: keep.shape[0], : keep.shape[1]] = keep
        return out & mask

    def _foreground(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """按"和四周背景色差得多"找出零件区域，裁出来补成正方形。相当于一个最简陋的目标检测。

        白色低饱和的像素（标签贴纸、反光）不算前景；再只保留最大的连通块，避免贴纸把裁剪框撑大。
        """
        small = img.copy()
        small.thumbnail((512, 512))
        bg = np.array(border_color(small), dtype=np.int16)
        blurred = np.asarray(small.filter(ImageFilter.GaussianBlur(2)), dtype=np.int16)
        hsv = np.asarray(small.convert("HSV"))
        whiteish = (hsv[:, :, 2] > 225) & (hsv[:, :, 1] < 30)
        diff = np.abs(blurred - bg).max(axis=2)
        # 阈值按"背景本身有多花"自适应：用四周一圈像素的差异分布估计噪声水平
        b = max(2, min(diff.shape) // 20)
        border = np.concatenate([diff[:b].ravel(), diff[-b:].ravel(), diff[:, :b].ravel(), diff[:, -b:].ravel()])
        thr = float(np.clip(np.percentile(border, 99) + 8, 18, 60))
        mask = (diff > thr) & ~whiteish
        if mask.mean() > 0.005:
            mask = self._largest_component(mask) if mask.shape[0] >= 8 and mask.shape[1] >= 8 else mask
        if mask.mean() > 0.005:
            ys, xs = np.nonzero(mask)
            x0, x1 = np.percentile(xs, [1, 99]).astype(int)
            y0, y1 = np.percentile(ys, [1, 99]).astype(int)
            small = small.crop((max(0, x0 - 2), max(0, y0 - 2), x1 + 3, y1 + 3))
        crop = pad_to_square(small, tuple(int(x) for x in bg)).resize((self.SIZE, self.SIZE), Image.BILINEAR)  # type: ignore[arg-type]
        a = np.asarray(crop)
        chsv = np.asarray(crop.convert("HSV"))
        fg = (np.abs(np.asarray(crop.filter(ImageFilter.GaussianBlur(1)), dtype=np.int16) - bg).max(axis=2) > thr) \
            & ~((chsv[:, :, 2] > 225) & (chsv[:, :, 1] < 30))
        if fg.mean() < 0.02:
            fg = np.ones(fg.shape, dtype=bool)
        return a, fg

    def _one(self, img: Image.Image) -> np.ndarray:
        rgb, fg = self._foreground(img)
        pil = Image.fromarray(rgb)
        hsv = np.asarray(pil.convert("HSV")).reshape(-1, 3)[fg.reshape(-1)]
        # 1) 颜色：H 12 × S 3 × V 3 的联合直方图，**软分箱**（每个像素按距离分给相邻两个箱子），
        #    否则光线稍亮一点，大片颜色就会整体"跳"到隔壁箱子。开平方 = Hellinger 距离。
        color = _soft_hist(hsv.astype(np.float64), (12, 3, 3), hue_cyclic=True)
        color = np.sqrt(color / max(color.sum(), 1))
        # 2) 纹理：LBP 直方图
        gray = np.asarray(pil.convert("L"), dtype=np.float64)
        lbp = _lbp_riu2(gray)[fg[1:-1, 1:-1]]
        tex = np.bincount(lbp.reshape(-1), minlength=10).astype(np.float64)
        tex = np.sqrt(tex / max(tex.sum(), 1))
        # 3) 形状：以零件中心为圆心，8 个同心环上的平均亮度和边缘密度（旋转不变，圆形风扇 vs 方形板卡）
        ys, xs = np.nonzero(fg)
        cy, cx = ys.mean(), xs.mean()
        yy, xx = np.mgrid[0 : self.SIZE, 0 : self.SIZE]
        r = np.hypot(yy - cy, xx - cx)
        rmax = max(float(r[fg].max()), 1.0)
        ring = np.minimum((r / rmax * 8).astype(int), 7)
        gy, gx = np.gradient(gray)
        edge = np.hypot(gx, gy) > 25
        radial = []
        for k in range(8):
            sel = fg & (ring == k)
            radial.append(gray[sel].mean() / 255 if sel.any() else 0.0)
        for k in range(8):
            sel = fg & (ring == k)
            radial.append(edge[sel].mean() if sel.any() else 0.0)
        shape = np.array([fg.mean(), fg.sum() / (np.pi * rmax**2), gray[fg].std() / 128])

        blocks = [(color, 1.0), (tex, 0.8), (np.array(radial), 0.8), (shape, 0.5)]
        v = np.concatenate([b / (np.linalg.norm(b) or 1.0) * w for b, w in blocks])
        return v

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        return _l2(np.stack([self._one(im) for im in images])).astype(np.float32).tolist()


# --------------------------------------------------------------------------- SigLIP


class SiglipImageEmbedder(ImageEmbedder):
    """transformers 加载 SigLIP / SigLIP 2。

    几个容易踩的坑：
    - 先 ``pad_to_square`` 再交给 processor，否则长条零件会被拉伸变形；
    - fp16 只在 GPU 上用，CPU 上用 fp32；
    - 输出向量要 L2 归一化，pgvector 用余弦距离检索；
    - ``get_image_features`` 在不同 transformers 版本里返回张量或输出对象，这里都兼容。
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16,
                 model=None, processor=None) -> None:  # model / processor 可注入，便于测试
        import torch

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        if model is None or processor is None:
            from transformers import AutoModel, AutoProcessor

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            model = model or AutoModel.from_pretrained(model_name, torch_dtype=dtype)
            processor = processor or AutoProcessor.from_pretrained(model_name)
        self.model = model.to(self.device).eval()
        self.processor = processor
        self.dim = len(self._encode([Image.new("RGB", (64, 64), (128, 128, 128))])[0])  # 用一张空图探测输出维度

    @property
    def model_id(self) -> str:
        return f"siglip:{self.model_name}:{self.dim}"

    def _encode(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        inputs = {k: (v.to(self.device, dtype=dtype) if k == "pixel_values" else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = [pad_to_square(im) for im in images[i : i + self.batch_size]]
            out.extend(self._encode(batch).tolist())
        return out


# --------------------------------------------------------------------------- HTTP 服务


class HTTPImageEmbedder(ImageEmbedder):
    """调用已部署的图像编码服务。约定的接口（按你们的服务改 ``_post`` 即可）::

        POST {url}   {"images": ["<base64 PNG>", ...]}
        200          {"embeddings": [[0.01, ...], ...]}
    """

    def __init__(self, url: str, dim: int, model_name: str = "remote", timeout: float = 60.0) -> None:
        self.url, self.dim, self.model_name = url, dim, model_name
        self.http = httpx.Client(timeout=timeout)

    @property
    def model_id(self) -> str:
        return f"http:{self.url}:{self.model_name}:{self.dim}"

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        payload = []
        for im in images:
            buf = io.BytesIO()
            pad_to_square(im).save(buf, format="PNG")
            payload.append(base64.b64encode(buf.getvalue()).decode())
        r = self.http.post(self.url, json={"images": payload})
        r.raise_for_status()
        vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"image service returned dim {vecs.shape[1]}, expected {self.dim}")
        return _l2(vecs).tolist()


def build_image_embedder(s: Settings) -> ImageEmbedder:
    if s.image_embedding_backend == "handcrafted":
        return HandcraftedImageEmbedder()
    if s.image_embedding_backend == "siglip":
        return SiglipImageEmbedder(s.image_embedding_model, s.embedding_device)
    if not s.image_embedding_url:
        raise ValueError("SERVICEKB_IMAGE_EMBEDDING_URL is required for image_embedding_backend=http")
    return HTTPImageEmbedder(s.image_embedding_url, s.image_embedding_dim, s.image_embedding_model)
