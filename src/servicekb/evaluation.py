"""评估：没有评估的 RAG 优化都是"凭感觉"。

先做**不需要 LLM 裁判**的指标（便宜、稳定、可以放进 CI 做回归测试），再做需要 LLM 的指标。

检索指标（看"证据找得对不对"），基于人工标注的相关文档 doc_ref：
- Hit@k     ：top-k 里至少有一个相关文档的问题占比。最直观。
- Recall@k  ：top-k 覆盖了多少比例的相关文档。
- MRR@k     ：第一个相关文档排名的倒数的平均值（第 1 名=1，第 2 名=0.5…）。看"排得靠不靠前"。
- nDCG@k    ：考虑所有相关文档位置的综合排序质量。

回答指标（看"答得对不对"）：
- must_include 召回：答案里是否包含标注的关键事实（料号、数值）——工业场景最实用的硬指标。
- 引用有效率：答案中的 [n] 都指向真实证据的比例。
- 拒答准确率：不该回答的问题（知识库里没有）是否正确拒答 / 升级；该回答的是否没有误拒。
- （可选）LLM 裁判的忠实度：答案中的每个陈述能否被证据支持（参考 ragas Faithfulness）。

评估集格式（eval/golden_qa.jsonl，每行一个）::

    {"id": "q01", "question": "...", "relevant": ["manuals:AS-110K-HV_service_manual_zh.md", "qa_archive:TK-2024-0103"],
     "must_include": ["310-04521-B"], "answerable": true, "user_groups": ["service"]}
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from servicekb.service import ServiceKB


@dataclass
class EvalCase:
    id: str
    question: str
    relevant: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    answerable: bool = True
    user_groups: list[str] | None = None
    notes: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            cases.append(EvalCase(**json.loads(line)))
    return cases


# --------------------------------------------------------------------------- 检索指标（纯函数，便于单测）


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant and r not in seen:  # 同一文档的多个块只算一次
            dcg += 1 / math.log2(i + 1)
            seen.add(r)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _dedup_docs(doc_refs: list[str]) -> list[str]:
    """检索结果是"块"，评估按"文档"算：同一文档的多个块只保留第一次出现的位置。"""
    return list(dict.fromkeys(doc_refs))


# --------------------------------------------------------------------------- 运行


@dataclass
class EvalReport:
    k: int
    n_cases: int
    retrieval: dict[str, float]
    answer: dict[str, float]
    per_case: list[dict[str, Any]]

    def to_markdown(self) -> str:
        lines = [f"# 评估报告（k={self.k}，{self.n_cases} 个问题）", "", "## 检索", "", "| 指标 | 值 |", "| --- | --- |"]
        lines += [f"| {k} | {v:.3f} |" for k, v in self.retrieval.items()]
        lines += ["", "## 回答", "", "| 指标 | 值 |", "| --- | --- |"]
        lines += [f"| {k} | {v:.3f} |" for k, v in self.answer.items()]
        lines += ["", "## 明细", "", "| id | hit | mrr | must_include | 拒答正确 | 首个相关排名 | 问题 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for c in self.per_case:
            lines.append(
                f"| {c['id']} | {c.get('hit', '-')} | {c.get('mrr', '-')} | {c.get('must_include', '-')} | "
                f"{c['refusal_correct']} | {c.get('first_rank', '-')} | {c['question'][:40]} |"
            )
        return "\n".join(lines) + "\n"


def run_eval(kb: ServiceKB, cases: list[EvalCase], k: int = 5) -> EvalReport:
    per_case: list[dict[str, Any]] = []
    hits, recalls, mrrs, ndcgs, musts, cites, refusal = [], [], [], [], [], [], []
    for c in cases:
        res = kb.ask(c.question, user_groups=c.user_groups, log_query=False)
        # 检索指标单独用更大的 k 再取一次，避免 ask 里的"弱尾截断"影响检索评估
        r = kb.search(c.question, user_groups=c.user_groups, top_k=max(k, 10))
        docs = _dedup_docs([sc.chunk.doc_ref for sc in r.results])
        row: dict[str, Any] = {"id": c.id, "question": c.question, "retrieved_docs": docs[:k],
                               "answer": res.answer, "escalate": res.escalate, "reason": res.reason}
        rel = set(c.relevant)
        if c.answerable and rel:
            row["hit"] = hit_at_k(docs, rel, k)
            row["mrr"] = round(mrr_at_k(docs, rel, k), 3)
            row["first_rank"] = next((i for i, d in enumerate(docs, 1) if d in rel), None)
            hits.append(row["hit"])
            recalls.append(recall_at_k(docs, rel, k))
            mrrs.append(row["mrr"])
            ndcgs.append(ndcg_at_k(docs, rel, k))
        if c.answerable and c.must_include and not res.escalate:
            ans = res.answer.upper()
            m = sum(1 for x in c.must_include if x.upper() in ans) / len(c.must_include)
            row["must_include"] = round(m, 2)
            musts.append(m)
        if not res.escalate and res.answer:
            cites.append(1.0 if res.citations else 0.0)
        row["refusal_correct"] = (res.escalate is (not c.answerable))
        refusal.append(1.0 if row["refusal_correct"] else 0.0)
        per_case.append(row)

    def avg(xs: list[float]) -> float:
        return round(mean(xs), 4) if xs else 0.0

    return EvalReport(
        k=k,
        n_cases=len(cases),
        retrieval={f"hit@{k}": avg(hits), f"recall@{k}": avg(recalls), f"mrr@{k}": avg(mrrs), f"ndcg@{k}": avg(ndcgs)},
        answer={"must_include_recall": avg(musts), "answers_with_citations": avg(cites),
                "refusal_accuracy": avg(refusal)},
        per_case=per_case,
    )


def save_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "eval_report.md", out_dir / "eval_report.json"
    md.write_text(report.to_markdown(), "utf-8")
    js.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), "utf-8")
    return md, js
