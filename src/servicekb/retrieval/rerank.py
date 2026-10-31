"""重排（Rerank）：用更准但更慢的模型，对召回的几十个候选重新打分。

为什么召回之后还要重排？
    召回阶段（bi-encoder 向量 + 关键词）追求"别漏"，要快，所以问题和文档是**分开**编码的。
    重排用 **cross-encoder**：把"问题 + 候选块"拼在一起送进模型，模型能逐词对照两者，
    判断"这块到底能不能回答这个问题"。精度高很多，但每个候选都要过一遍模型，
    所以只对 top 30~50 做。

    典型收益：top-5 命中率提升 10~20 个百分点；另一个重要用途是**拒答**——
    重排分数（sigmoid 后 0~1）有绝对意义，最高分都很低说明知识库里没有答案。

推荐模型：BAAI/bge-reranker-v2-m3（中英混合，568M 参数，48GB 卡上 fp16 跑很轻松）。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import httpx

from servicekb.config import Settings


def _sigmoid_if_logits(scores: list[float]) -> list[float]:
    """有的实现返回 logits（可能是 -8.2、5.3），统一转成 0~1，阈值才有可比性。"""
    if scores and (min(scores) < 0 or max(scores) > 1):
        return [1 / (1 + math.exp(-s)) for s in scores]
    return scores


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """返回与 passages 一一对应的相关性分数（0~1，越大越相关）。"""


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # 可选依赖

        self.name = f"cross_encoder:{model_name}"
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False)
        return _sigmoid_if_logits([float(x) for x in raw])


class TEIReranker(Reranker):
    """调用 HuggingFace text-embeddings-inference 部署的 reranker（推荐生产用：独立服务、可扩缩容）。"""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.name = f"tei:{url}"
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        r = self.http.post(f"{self.url}/rerank", json={"query": query, "texts": passages, "truncate": True})
        r.raise_for_status()
        scores = [0.0] * len(passages)
        for item in r.json():
            scores[item["index"]] = float(item["score"])
        return _sigmoid_if_logits(scores)


def build_reranker(s: Settings) -> Reranker | None:
    if s.reranker_backend == "none":
        return None
    if s.reranker_backend == "cross_encoder":
        return CrossEncoderReranker(s.reranker_model, s.embedding_device)
    if not s.reranker_url:
        raise ValueError("SERVICEKB_RERANKER_URL is required for reranker_backend=tei")
    return TEIReranker(s.reranker_url)
