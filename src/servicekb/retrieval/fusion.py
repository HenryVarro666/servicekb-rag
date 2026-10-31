"""融合（Fusion）与去重折叠。

RRF（Reciprocal Rank Fusion，倒数排名融合）
-------------------------------------------
三路检索的分数量纲完全不同：余弦相似度在 0~1，BM25 可能是 0~30，实体命中是 0~3。
直接相加没有意义。RRF 只看**排名**：

    RRF(d) = Σ_i  w_i / (k + rank_i(d))

- rank_i(d)：文档 d 在第 i 路结果里的名次（从 1 开始）；没出现就不加分。
- k：平滑常数，惯例 60。k 越大，"第 1 名"和"第 10 名"的差距越小，越看重"多路都出现"。
- w_i：每一路的权重。

举例（k=60）：块 A 在向量路第 1、关键词路第 3 → 1/61 + 1/63 = 0.0323；
           块 B 只在向量路第 1          → 1/61        = 0.0164。
A 胜出：**被多路同时召回的结果更可信**，这是混合检索比单路检索稳的根本原因。

参考：Cormack et al., SIGIR 2009；Haystack ``DocumentJoiner``、R2R ``hybrid_search``。
"""

from __future__ import annotations

from servicekb.models import Chunk, ScoredChunk
from servicekb.processing.dedup import hamming


def reciprocal_rank_fusion(
    results: dict[str, list[tuple[Chunk, float]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    weights = weights or {}
    fused: dict[str, ScoredChunk] = {}
    for channel, hits in results.items():
        w = weights.get(channel, 1.0)
        for rank, (chunk, score) in enumerate(hits, start=1):
            sc = fused.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk))
            sc.score += w / (k + rank)
            sc.channel_scores[channel] = score
            sc.channel_ranks[channel] = rank
    return sorted(fused.values(), key=lambda s: (-s.score, s.chunk.chunk_id))


def collapse_near_duplicates(items: list[ScoredChunk], max_hamming: int = 3) -> list[ScoredChunk]:
    """按排序顺序遍历，和已保留的结果近似重复的折叠进去（只记录 doc_ref）。

    "入库不删、检索折叠"：历史工单里同一个问题被问了 20 次，检索结果只展示 1 条，
    并告诉用户"另有 19 条类似工单"——既不挤占 top-k，又不丢信息。
    """
    kept: list[ScoredChunk] = []
    for it in items:
        dup_of = next(
            (
                k for k in kept
                if it.chunk.content_hash == k.chunk.content_hash
                or (it.chunk.simhash and k.chunk.simhash and hamming(it.chunk.simhash, k.chunk.simhash) <= max_hamming)
            ),
            None,
        )
        if dup_of is None:
            kept.append(it)
        else:
            dup_of.duplicates.append(it.chunk.doc_ref)
    return kept
