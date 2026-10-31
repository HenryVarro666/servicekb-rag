"""检索、融合、权限、拒答、引用。"""

from servicekb.generation.generator import extract_citations
from servicekb.models import Chunk, ScoredChunk
from servicekb.retrieval.fusion import collapse_near_duplicates, reciprocal_rank_fusion


def _c(cid, text="t", sh=0):
    return Chunk(chunk_id=cid, doc_id="d", ordinal=0, text=text, content_hash=cid, simhash=sh, doc_ref=f"s:{cid}")


def test_rrf_prefers_items_found_by_multiple_channels():
    a, b, c = _c("a"), _c("b"), _c("c")
    fused = reciprocal_rank_fusion({"dense": [(b, 0.9), (a, 0.8)], "keyword": [(a, 12.0), (c, 3.0)]}, k=60)
    assert [x.chunk.chunk_id for x in fused] == ["a", "b", "c"]
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9
    assert fused[0].channel_ranks == {"dense": 2, "keyword": 1}


def test_collapse_near_duplicates_keeps_first_and_records_others():
    items = [ScoredChunk(_c("a", sh=0b1111)), ScoredChunk(_c("b", sh=0b1110)), ScoredChunk(_c("c", sh=0xFFFF0000))]
    kept = collapse_near_duplicates(items, max_hamming=1)
    assert [k.chunk.chunk_id for k in kept] == ["a", "c"] and kept[0].duplicates == ["s:b"]


def test_extract_citations_drops_out_of_range():
    ctx = [ScoredChunk(_c("a")), ScoredChunk(_c("b"))]
    text, cites = extract_citations("结论 [1]。另见【2】和 [7]，以及 [1, 2]。", ctx)
    assert text == "结论 [1]。另见[2]和 ，以及 [1][2]。"
    assert [c.index for c in cites] == [1, 2]


def test_part_question_gets_substitute_fact(kb):
    r = kb.ask("310-04521-A 还能订到吗？")
    assert not r.escalate
    assert any("已被 310-04521-B 替代" in f for f in r.facts)
    assert r.citations and r.retrieved[0].chunk.doc_ref.startswith("parts_master:310-04521")


def test_alias_spelling_matches_canonical_model(kb):
    r = kb.search("as110khv E037 早上报警")
    assert "model:AS-110K-HV" in r.analysis.entity_ids
    assert any(sc.chunk.doc_ref.startswith("qa_archive:TK-2024-0103") for sc in r.results)


def test_acl_public_user_cannot_see_internal_sources(kb):
    for q in ["E051 风扇故障换哪个料号", "主控板更换后 E077"]:
        refs = [sc.chunk.doc_ref for sc in kb.search(q, user_groups=["public"], top_k=20).results]
        assert refs and not any(r.startswith(("bulletins:", "qa_archive:")) for r in refs)
    refs = [sc.chunk.doc_ref for sc in kb.search("E051 风扇故障换哪个料号", user_groups=["service"], top_k=20).results]
    assert any(r.startswith("bulletins:") for r in refs)


def test_product_model_hard_filter(kb):
    r = kb.search("E037 绝缘阻抗低", product_models=["AS-250K-HV"], top_k=20)
    for sc in r.results:
        assert not sc.chunk.product_models or "AS-250K-HV" in sc.chunk.product_models


def test_duplicate_tickets_are_collapsed(kb):
    r = kb.search("所有组串绝缘都正常但还是报 E037", top_k=10)
    refs = [sc.chunk.doc_ref for sc in r.results]
    assert not ("qa_archive:TK-2025-0215" in refs and "qa_archive:TK-2026-0031" in refs)
    assert any(sc.duplicates for sc in r.results)


def test_out_of_scope_question_is_refused(kb):
    for q in ["逆变器支持接入风力发电机吗？", "公司年假政策是什么？"]:
        r = kb.ask(q)
        assert r.escalate and not r.citations


def test_query_is_logged_and_feedback_accepted(kb):
    r = kb.ask("AS-110K-HV 侧盖螺丝扭矩")
    assert r.query_id in kb.store.query_logs
    assert kb.feedback(r.query_id, 1, "有用")
    assert not kb.feedback("00000000-0000-0000-0000-000000000000", 1)


def test_reranker_failure_degrades_gracefully(kb):
    from servicekb.retrieval.rerank import Reranker

    class Broken(Reranker):
        name = "broken"

        def score(self, query, passages):
            raise ConnectionError("TEI down")

    old = kb.retriever.reranker
    kb.retriever.reranker = Broken()
    try:
        r = kb.search("310-04521-A 还能订到吗？")
        assert r.results and not r.reranked
    finally:
        kb.retriever.reranker = old


def test_reranker_scores_reorder_and_gate(kb):
    from servicekb.retrieval.rerank import Reranker

    class Fake(Reranker):
        name = "fake"

        def score(self, query, passages):  # 含 "220-11009-A" 的块打高分，其余打低分
            return [0.9 if "220-11009-A" in p else 0.05 for p in passages]

    old = kb.retriever.reranker
    kb.retriever.reranker = Fake()
    try:
        a = kb.ask("AS-250K-HV 的直流开关用哪个料号？")
        assert "220-11009-A" in a.retrieved[0].chunk.text and not a.escalate
        assert all((c.rerank_score or 0) >= 0.45 for c in a.retrieved)  # 低于最高分一半的弱尾被截掉
    finally:
        kb.retriever.reranker = old


def test_obsolete_part_in_evidence_triggers_substitute_fact(kb):
    # 问题里没有料号，但证据里的旧工单推荐了已停产的 310-04521-A → 必须提醒替代件
    r = kb.ask("AS-110K-HV 报 E051 风扇故障，风沙很大，要换什么？")
    assert any(f.startswith("料号 310-04521-A") and "已被 310-04521-B 替代" in f for f in r.facts)
    # 证据里提到的在售料号不注入（只提醒停产 / 限制使用的），避免 prompt 膨胀
    assert not any(f.startswith("料号 730-00014-A") for f in r.facts)
