"""混合检索器：查询分析 → 三路并行召回 → RRF 融合 → 去重折叠 → 重排 → 截断。

                         ┌─ 向量检索（语义："风扇不转" ≈ "散热故障"）──────┐
    问题 ─ 查询分析 ─────┼─ 关键词检索（字面："E037"、"直流开关"）────────┼─ RRF 融合 ─ 折叠重复 ─ 重排 ─ top-k
       (实体 / 分词 / 扩展) └─ 实体检索（对象："part:310-04521-B" 及其替代件）┘

三路各有所长、互相补短板：
- 只用向量：精确标识符（料号、故障码）经常召回错；
- 只用关键词：同义表达、中英混问召回不到；
- 实体路：利用对象层，把"料号 A 已被料号 B 替代"这类结构化关系用在检索上。

容错（生产环境里"某个依赖挂了"是常态，不是例外）：
- 问题向量算不出来（向量服务挂了 / 超时）→ 跳过向量路，关键词 + 实体两路照常；
- 某一路查询报错 / 超时 → 跳过这一路；
- 重排服务挂了 → 用融合排序；
- 降级情况记在 ``RetrievalResult.degraded`` 里（进日志和监控指标），**只有所有路都失败才报错**。
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from servicekb.embedding import Embedder
from servicekb.logging import get_logger
from servicekb.models import ScoredChunk, SearchFilters
from servicekb.processing.entities import ENTITY_TYPE_WEIGHT, EntityLinker, Mention, canonical_tokens
from servicekb.processing.text import normalize_text, tokenize
from servicekb.retrieval.fusion import collapse_near_duplicates, reciprocal_rank_fusion
from servicekb.retrieval.rerank import Reranker
from servicekb.store.base import Store

log = get_logger(__name__)


@dataclass
class QueryAnalysis:
    question: str
    mentions: list[Mention]
    entity_ids: list[str]  # 问题里直接提到的
    expanded_ids: list[str]  # + 替代件 / 互换件
    entity_weights: dict[str, float]
    tokens: list[str]
    product_models: list[str]


class RetrievalError(RuntimeError):
    """所有召回通道都失败了（通常是数据库不可用）。API 层返回 503。"""


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    results: list[ScoredChunk]
    candidates: int = 0
    reranked: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)  # 例如 ["dense:embedding_failed", "rerank:failed"]


def analyze_query(question: str, linker: EntityLinker, extra_entities: list[str] | None = None) -> QueryAnalysis:
    """``extra_entities``：问题文字之外已知的对象（例如拍照识别出的料号），和"问题里直接提到的"同等对待。"""
    q = normalize_text(question)
    mentions = linker.extract(q)
    direct = list(dict.fromkeys([m.entity_id for m in mentions] + list(extra_entities or [])))
    expanded = linker.expand(direct)
    weights: dict[str, float] = {}
    for eid in expanded:
        w = ENTITY_TYPE_WEIGHT.get(eid.split(":", 1)[0], 0.5)
        weights[eid] = w if eid in direct else w * 0.8  # 扩展出来的实体权重略低
    models = [linker.entities[e].canonical_name for e in direct if e.startswith("model:") and e in linker.entities]
    tokens = tokenize(q, extra_tokens=canonical_tokens(expanded))
    return QueryAnalysis(q, mentions, direct, expanded, weights, tokens, models)


class HybridRetriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        linker: EntityLinker,
        reranker: Reranker | None = None,
        *,
        top_k_dense: int = 40,
        top_k_keyword: int = 40,
        top_k_entity: int = 30,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        rerank_candidates: int = 40,
        near_dup_hamming: int = 3,
        max_workers: int = 48,
        channel_timeout_s: float = 10.0,
    ) -> None:
        self.store, self.embedder, self.linker, self.reranker = store, embedder, linker, reranker
        self.k_dense, self.k_kw, self.k_ent = top_k_dense, top_k_keyword, top_k_entity
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "keyword": 1.0, "entity": 1.0}
        self.rerank_candidates = rerank_candidates
        self.near_dup_hamming = near_dup_hamming
        self.channel_timeout_s = channel_timeout_s
        # 线程池是**进程级共享**的：并发 N 个请求就有 3N 个召回任务。池子太小时请求之间互相排队，
        # 延迟随并发线性上涨（旧版本固定 3 个线程就有这个问题）。
        self._pool = ThreadPoolExecutor(max_workers=max(3, max_workers), thread_name_prefix="retrieve")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def retrieve(self, question: str, filters: SearchFilters, top_k: int = 6,
                 extra_entities: list[str] | None = None) -> RetrievalResult:
        timings: dict[str, float] = {}
        degraded: list[str] = []
        t = time.perf_counter()
        qa = analyze_query(question, self.linker, extra_entities)
        # 关键词路和实体路不依赖向量，先提交，和"算问题向量"并行
        futures: dict[str, Future] = {
            "keyword": self._pool.submit(self.store.keyword_search, qa.tokens, self.k_kw, filters),
            "entity": self._pool.submit(self.store.entity_search, qa.entity_weights, self.k_ent, filters),
        }
        try:
            qvec = self.embedder.embed_query(qa.question)
            futures["dense"] = self._pool.submit(self.store.dense_search, qvec, self.k_dense, filters)
        except Exception as e:  # 向量服务挂了：不能让整个请求失败
            log.warning("query embedding failed, dense channel skipped", extra={"error": repr(e)[:300]})
            degraded.append("dense:embedding_failed")
        timings["analyze_embed"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()

        # 三路并行（Postgres 连接池支持并发）。总耗时 ≈ 最慢的一路，而不是三路之和。
        channels: dict[str, list] = {}
        deadline = time.perf_counter() + self.channel_timeout_s
        for name in ("dense", "keyword", "entity"):
            if name not in futures:
                continue
            try:
                channels[name] = futures[name].result(timeout=max(0.0, deadline - time.perf_counter()))
            except Exception as e:  # noqa: BLE001  超时 / 数据库报错：跳过这一路
                log.warning("retrieval channel failed", extra={"channel": name, "error": repr(e)[:300]})
                degraded.append(f"{name}:{type(e).__name__}")
        if not channels:
            raise RetrievalError("all retrieval channels failed: " + ", ".join(degraded))
        timings["recall"] = (time.perf_counter() - t) * 1000

        fused = reciprocal_rank_fusion(channels, k=self.rrf_k, weights=self.weights)
        fused = collapse_near_duplicates(fused, self.near_dup_hamming)
        n_candidates = len(fused)

        reranked = False
        if self.reranker and fused:
            t = time.perf_counter()
            cands = fused[: self.rerank_candidates]
            try:
                scores = self.reranker.score(qa.question, [c.chunk.embed_text() for c in cands])
            except Exception:  # 重排服务挂了：降级为只用融合排序，而不是整个请求失败
                log.exception("reranker failed, falling back to fused ranking")
                scores = None
                degraded.append("rerank:failed")
            if scores is not None:
                for c, s in zip(cands, scores, strict=True):
                    c.rerank_score = s
                    c.score = s
                fused = sorted(cands, key=lambda c: -c.score)
                reranked = True
            timings["rerank"] = (time.perf_counter() - t) * 1000

        return RetrievalResult(qa, fused[:top_k], n_candidates, reranked, timings, degraded)
