"""应用容器：把所有组件按配置组装起来，对外提供 ingest / search / ask 三个能力。

CLI 和 HTTP API 都只调用这里，保证两边行为一致。

一次 ask 的完整流程::

    问题 ──▶ 混合检索（retrieval/retriever.py）
          ──▶ 证据门槛：证据不足 → 直接拒答并标记升级（不调用 LLM，省钱也防幻觉）
          ──▶ 结构化事实（对象层：料号状态 / 替代关系）
          ──▶ LLM 生成（generation/）
          ──▶ 引用校验
          ──▶ 写 query_logs（用于反馈、评估、排查）
"""

from __future__ import annotations

import math
import re
import time
import uuid
from pathlib import Path
from typing import Any

from servicekb.config import Settings, get_settings
from servicekb.connectors import build_connector, load_sources
from servicekb.embedding import build_embedder
from servicekb.generation.generator import build_generator, extract_citations
from servicekb.generation.prompts import PROMPT_VERSION, REFUSAL_EN, REFUSAL_ZH
from servicekb.logging import get_logger
from servicekb.models import AnswerResult, DocType, ScoredChunk, SearchFilters
from servicekb.pipeline.ingest import IngestionPipeline, IngestReport
from servicekb.processing.chunking import ChunkingConfig
from servicekb.processing.entities import EntityLinker, canonical_tokens
from servicekb.processing.text import tokenize
from servicekb.retrieval.rerank import build_reranker
from servicekb.retrieval.retriever import HybridRetriever, RetrievalResult
from servicekb.store import build_store

log = get_logger(__name__)

_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿]")


def is_english(text: str) -> bool:
    return len(_LATIN.findall(text)) > 3 * len(_CJK.findall(text))


def keyword_coverage(tokens: list[str], contexts: list[ScoredChunk], df: dict[str, int], n: int) -> float:
    """问题关键词被"最佳证据块"覆盖的比例（按 IDF 加权）。没有 reranker 时用于判断要不要拒答。

    例："逆变器支持接入风力发电机吗？" → "逆变器"到处都有（IDF 低），"风力""发电机"在知识库里
    一次都没出现（IDF 最高）→ 覆盖率很低 → 拒答。
    """
    q = set(tokens)
    if not q or not contexts:
        return 0.0
    idf = {t: math.log((n + 1) / (df.get(t, 0) + 1)) + 1 for t in q}
    total = sum(idf.values())
    best = 0.0
    for sc in contexts[:3]:
        c = sc.chunk
        ctoks = set(tokenize(" ".join(c.heading_path) + "\n" + c.text, extra_tokens=canonical_tokens(c.entity_ids)))
        best = max(best, sum(idf[t] for t in q & ctoks) / total)
    return best


class ServiceKB:
    def __init__(self, settings: Settings | None = None) -> None:
        s = self.settings = settings or get_settings()
        self.store = build_store(s)
        self.embedder = build_embedder(s)
        self.store.init(self.embedder.model_id, self.embedder.dim)
        self.linker = EntityLinker(part_number_pattern=s.part_number_pattern, fault_code_pattern=s.fault_code_pattern)
        self.reranker = build_reranker(s)
        self.generator = build_generator(s)
        self.pipeline = IngestionPipeline(
            self.store, self.embedder, self.linker,
            ChunkingConfig(s.chunk_max_tokens, s.chunk_overlap_tokens, s.chunk_min_tokens),
            max_delete_ratio=s.max_delete_ratio,
        )
        self.pipeline.reload_linker()
        self.retriever = HybridRetriever(
            self.store, self.embedder, self.linker, self.reranker,
            top_k_dense=s.top_k_dense, top_k_keyword=s.top_k_keyword, top_k_entity=s.top_k_entity,
            rrf_k=s.rrf_k,
            weights={"dense": s.weight_dense, "keyword": s.weight_keyword, "entity": s.weight_entity},
            rerank_candidates=s.reranker_candidates, near_dup_hamming=s.near_dup_hamming,
        )

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------------ 导入
    def ingest(
        self, sources_file: Path | None = None, only: list[str] | None = None, force_delete: bool = False
    ) -> list[IngestReport]:
        path = Path(sources_file or self.settings.sources_file)
        base_dir = path.resolve().parent  # sources.yaml 里的相对路径，相对于 yaml 文件所在目录
        reports = []
        for spec in load_sources(path):
            if only and spec["name"] not in only:
                continue
            reports.append(self.pipeline.run(build_connector(spec, base_dir), force_delete=force_delete))
        return reports

    # ------------------------------------------------------------------ 检索
    def _filters(
        self, user_groups: list[str] | None, doc_types: list[str] | None, product_models: list[str] | None
    ) -> SearchFilters:
        models = [self.linker.resolve_model(m) or m for m in product_models] if product_models else None
        return SearchFilters(
            acl_groups=list(user_groups or self.settings.default_user_groups),
            doc_types=[DocType(d) for d in doc_types] if doc_types else None,
            product_models=models,
        )

    def search(
        self,
        question: str,
        user_groups: list[str] | None = None,
        doc_types: list[str] | None = None,
        product_models: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        filters = self._filters(user_groups, doc_types, product_models)
        return self.retriever.retrieve(question, filters, top_k or self.settings.final_top_k)

    # ------------------------------------------------------------------ 问答
    def _evidence_ok(self, r: RetrievalResult) -> tuple[bool, str]:
        if not r.results:
            return False, "no_results"
        if r.reranked:
            best = max(c.rerank_score or 0.0 for c in r.results)
            if best < self.settings.reranker_min_score:
                return False, f"low_rerank_score:{best:.3f}"
            return True, ""
        # 没有 reranker 时的启发式（离线 demo 用；生产请开 reranker，用分数阈值判断）：
        # 1) 问题里有登记过的型号 / 料号 / 故障码，且实体路命中了 → 领域问题，证据可信
        known = [e for e in r.analysis.entity_ids if e in self.linker.entities]
        if known and any("entity" in sc.channel_ranks for sc in r.results[:3]):
            return True, ""
        # 2) 否则看关键词覆盖率（IDF 加权）
        df, n = self.store.token_document_frequency(r.analysis.tokens)
        cov = keyword_coverage(r.analysis.tokens, r.results, df, n)
        if cov < self.settings.min_keyword_coverage:
            return False, f"low_keyword_coverage:{cov:.2f}"
        return True, ""

    def _drop_weak_tail(self, results: list[ScoredChunk]) -> list[ScoredChunk]:
        """重排后去掉低分尾巴：低于阈值，或比最高分低一半以上的（借鉴 QAnything 的做法）。"""
        if not results or results[0].rerank_score is None:
            return results
        top = results[0].rerank_score or 0.0
        floor = max(self.settings.reranker_min_score, top * 0.5)
        return [c for c in results if (c.rerank_score or 0.0) >= floor] or results[:1]

    def ask(
        self,
        question: str,
        user_groups: list[str] | None = None,
        doc_types: list[str] | None = None,
        product_models: list[str] | None = None,
        log_query: bool = True,
    ) -> AnswerResult:
        t0 = time.perf_counter()
        query_id = str(uuid.uuid4())
        groups = list(user_groups or self.settings.default_user_groups)
        r = self.search(question, groups, doc_types, product_models)
        timings = dict(r.timings_ms)
        contexts = self._drop_weak_tail(r.results)
        # 结构化事实：① 问题里提到的料号（含替代件）；② 证据里提到的**停产 / 限制使用**料号。
        # ② 很重要：旧工单可能推荐了后来停产的料号，即使用户没问料号，也要提醒替代关系。
        facts = self.linker.facts_for(r.analysis.expanded_ids)
        evidence_parts = [e for c in contexts for e in c.chunk.entity_ids
                          if e.startswith("part:") and e not in r.analysis.expanded_ids]
        facts += self.linker.facts_for(evidence_parts, only_inactive=True)

        ok, reason = self._evidence_ok(r)
        if not ok:
            answer, citations, escalate, model = (REFUSAL_EN if is_english(question) else REFUSAL_ZH), [], True, "none"
        else:
            t = time.perf_counter()
            try:
                raw = self.generator.generate(question, contexts, facts)
                model = self.generator.name
            except Exception as e:  # LLM 服务故障 → 降级为抽取式回答，而不是直接报错
                log.exception("generation failed, falling back to extractive")
                from servicekb.generation.generator import ExtractiveGenerator

                raw, model, reason = ExtractiveGenerator().generate(question, contexts, facts), "extractive-fallback", f"llm_error:{e!r}"
            timings["generate"] = (time.perf_counter() - t) * 1000
            answer, citations = extract_citations(raw, contexts)
            escalate = "没有找到相关依据" in answer or "does not contain enough" in answer
            if not citations:
                reason = reason or "no_valid_citations"
        timings["total"] = (time.perf_counter() - t0) * 1000

        result = AnswerResult(
            query_id=query_id, question=question, answer=answer, citations=citations, retrieved=contexts,
            escalate=escalate, reason=reason, model=model, timings_ms={k: round(v, 1) for k, v in timings.items()},
            facts=facts,
        )
        if log_query:
            self._log(result, groups, {"doc_types": doc_types, "product_models": product_models})
        return result

    def _log(self, r: AnswerResult, groups: list[str], filters: dict[str, Any]) -> None:
        try:
            self.store.log_query({
                "query_id": r.query_id, "question": r.question, "user_groups": groups, "filters": filters,
                "retrieved": [
                    {"chunk_id": c.chunk.chunk_id, "doc_ref": c.chunk.doc_ref, "score": round(c.score, 5),
                     "rerank": c.rerank_score, "ranks": c.channel_ranks} for c in r.retrieved
                ],
                "answer": r.answer,
                "citations": [{"index": c.index, "chunk_id": c.chunk_id, "doc_ref": c.doc_ref} for c in r.citations],
                "escalate": r.escalate, "model": r.model, "prompt_version": PROMPT_VERSION, "timings_ms": r.timings_ms,
            })
        except Exception:  # 日志失败不能影响回答
            log.exception("failed to log query")

    def feedback(self, query_id: str, rating: int, comment: str | None = None, corrected_answer: str | None = None) -> bool:
        return self.store.log_feedback(query_id, rating, comment, corrected_answer)
