"""向量模型（Embedding）：把文本变成向量，语义相近的文本向量也相近。

RAG 小知识
----------
- **Bi-encoder（双塔）**：问题和文档**分别**编码成向量，用余弦相似度比较。优点是文档向量可以
  提前算好存进数据库，查询时只需编码问题 → 快。缺点是问题和文档之间没有"交互"，精度有上限。
  （精度更高的 cross-encoder 用在重排阶段，见 retrieval/rerank.py）
- **归一化**：向量 L2 归一化后，余弦相似度 = 点积，计算更快；pgvector 用 ``<=>`` 计算余弦距离。
- **query / passage 不对称**：有的模型（e5、bge 英文 v1.5）要求给问题加指令前缀，见配置里的
  embedding_query_prefix。用错了召回会明显下降。**bge-m3 不需要前缀**。
- **换模型 = 重建索引**：不同模型的向量空间不兼容，混在一起检索结果是随机的。
  本项目在数据库 ``index_meta`` 表里记录了建库时的模型 ID，不一致会拒绝启动。

三个实现：
- ``HashingEmbedder``：字符 n-gram 哈希。零依赖、确定性、**没有语义理解能力**，只用于 demo / CI。
- ``SentenceTransformerEmbedder``：本地 GPU 跑开源模型，推荐 BAAI/bge-m3（中英混合、8192 长度、1024 维）。
- ``OpenAIEmbedder``：OpenAI 官方，或任何 OpenAI-compatible 服务（HuggingFace TEI、vLLM、Ollama）。
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from servicekb.config import Settings


class Embedder(ABC):
    dim: int

    @property
    @abstractmethod
    def model_id(self) -> str:
        """唯一标识"这个向量空间"，写进 index_meta 和 pipeline_version。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder(Embedder):
    """把文本的特征（中文字 1/2-gram、英文单词、英文字符 3-gram）哈希到固定维度。

    它本质上是"带一点模糊匹配能力的词袋"，能让 demo 跑出合理结果，但**不懂同义词**：
    "风扇不转" 和 "散热器停止工作" 在它眼里毫无关系。生产必须换成真正的语义模型。
    """

    _TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing-v1-{self.dim}"

    def _features(self, text: str) -> list[str]:
        toks = self._TOKEN.findall(text.lower())
        feats = list(toks)
        feats += [a + b for a, b in zip(toks, toks[1:], strict=False)]  # 相邻 bigram，捕获"绝缘+阻抗"这类搭配
        for t in toks:
            if len(t) > 4:
                feats += [t[i : i + 3] for i in range(len(t) - 2)]
        return feats

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                m[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2_normalize(m).tolist()


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer  # 可选依赖：pip install -e ".[local]"

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix

    @property
    def model_id(self) -> str:
        return f"st:{self.model_name}:{self.dim}"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        dim: int,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self.model, self.dim, self.batch_size = model, dim, batch_size
        self.base_url = base_url or "openai"
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix

    @property
    def model_id(self) -> str:
        return f"openai:{self.base_url}:{self.model}:{self.dim}"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # text-embedding-3-* 支持 dimensions 参数截断维度；自托管服务一般不支持，故只在官方接口传
            kwargs = {"dimensions": self.dim} if self.base_url == "openai" and "text-embedding-3" in self.model else {}
            resp = self.client.embeddings.create(model=self.model, input=batch, **kwargs)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return _l2_normalize(np.asarray(out, dtype=np.float32)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_backend == "hashing":
        return HashingEmbedder(dim=s.embedding_dim)
    if s.embedding_backend == "sentence_transformers":
        e = SentenceTransformerEmbedder(
            s.embedding_model, s.embedding_device, s.embedding_batch_size,
            s.embedding_query_prefix, s.embedding_passage_prefix,
        )
        if e.dim != s.embedding_dim:
            raise ValueError(f"SERVICEKB_EMBEDDING_DIM={s.embedding_dim} but model {s.embedding_model} outputs {e.dim}")
        return e
    return OpenAIEmbedder(
        s.embedding_model, s.embedding_dim,
        api_key=s.embedding_api_key.get_secret_value() if s.embedding_api_key else None,
        base_url=s.embedding_base_url, batch_size=s.embedding_batch_size,
        query_prefix=s.embedding_query_prefix, passage_prefix=s.embedding_passage_prefix,
    )
