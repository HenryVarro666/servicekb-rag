"""混合检索器：查询分析 → 三路并行召回 → RRF 融合 → 去重折叠 → 重排 → 截断。

                         ┌─ 向量检索（语义："风扇不转" ≈ "散热故障"）──────┐
    问题 ─ 查询分析 ─────┼─ 关键词检索（字面："E037"、"直流开关"）────────┼─ RRF 融合 ─ 折叠重复 ─ 重排 ─ top-k
       (实体 / 分词 / 扩展) └─ 实体检索（对象："part:310-04521-B" 及其替代件）┘

三路各有所长、互相补短板：
- 只用向量：精确标识符（料号、故障码）经常召回错；
- 只用关键词：同义表达、中英混问召回不到；
- 实体路：利用对象层，把"料号 A 已被料号 B 替代"这类结构化关系用在检索上。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    results: list[ScoredChunk]
    candidates: int = 0
    reranked: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


def analyze_query(question: str, linker: EntityLinker) -> QueryAnalysis:
    q = normalize_text(question)
    mentions = linker.extract(q)
    direct = [m.entity_id for m in mentions]
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
    ) -> None:
        self.store, self.embedder, self.linker, self.reranker = store, embedder, linker, reranker
        self.k_dense, self.k_kw, self.k_ent = top_k_dense, top_k_keyword, top_k_entity
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "keyword": 1.0, "entity": 1.0}
        self.rerank_candidates = rerank_candidates
        self.near_dup_hamming = near_dup_hamming
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="retrieve")

    def retrieve(self, question: str, filters: SearchFilters, top_k: int = 6) -> RetrievalResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        qa = analyze_query(question, self.linker)
        qvec = self.embedder.embed_query(qa.question)
        timings["analyze_embed"] = (time.perf_counter() - t) * 1000

        # 三路并行（Postgres 连接池支持并发）。总耗时 ≈ 最慢的一路，而不是三路之和。
        t = time.perf_counter()
        f_dense = self._pool.submit(self.store.dense_search, qvec, self.k_dense, filters)
        f_kw = self._pool.submit(self.store.keyword_search, qa.tokens, self.k_kw, filters)
        f_ent = self._pool.submit(self.store.entity_search, qa.entity_weights, self.k_ent, filters)
        channels = {"dense": f_dense.result(), "keyword": f_kw.result(), "entity": f_ent.result()}
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
            if scores is not None:
                for c, s in zip(cands, scores, strict=True):
                    c.rerank_score = s
                    c.score = s
                fused = sorted(cands, key=lambda c: -c.score)
                reranked = True
            timings["rerank"] = (time.perf_counter() - t) * 1000

        return RetrievalResult(qa, fused[:top_k], n_candidates, reranked, timings)
