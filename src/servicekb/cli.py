"""命令行入口：``servicekb --help``

常用命令::

    servicekb demo                      # 一键体验（内存存储，无需数据库 / GPU / API Key）
    servicekb db init                   # 建表（Postgres）
    servicekb ingest                    # 按 configs/sources.yaml 导入所有数据源
    servicekb ask "E037 怎么处理？"      # 问答
    servicekb search "E037"             # 只看检索结果（调试用：每一路的排名都会显示）
    servicekb eval                      # 跑评估集
    servicekb audit parts               # 物料主数据质量报告
    servicekb chat clean --input ...    # 清洗 Teams 聊天记录
    servicekb serve                     # 启动 HTTP API

注意：``store_backend=memory`` 时每条命令都是新进程、数据不保留，所以 ask / search / eval
会先自动导入一遍示例数据。生产请用 postgres。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from servicekb.config import Settings, get_settings
from servicekb.logging import setup_logging

app = typer.Typer(add_completion=False, help="servicekb：面向售后 / 现场服务知识库的 RAG 框架")
db_app = typer.Typer(help="数据库管理")
audit_app = typer.Typer(help="数据质量审计")
chat_app = typer.Typer(help="聊天记录（Teams）清洗")
app.add_typer(db_app, name="db")
app.add_typer(audit_app, name="audit")
app.add_typer(chat_app, name="chat")
console = Console()


def _settings() -> Settings:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    return s


def _kb(auto_ingest: bool = True):
    from servicekb.service import ServiceKB

    s = _settings()
    kb = ServiceKB(s)
    if s.store_backend == "memory" and auto_ingest:
        console.print("[dim]（memory 模式：先自动导入 configs/sources.yaml 里的数据）[/dim]")
        kb.ingest()
    return kb


def _groups(groups: str | None) -> list[str] | None:
    return [g.strip() for g in groups.split(",") if g.strip()] if groups else None


# --------------------------------------------------------------------------- 导入


@app.command()
def ingest(
    sources: Path = typer.Option(None, help="数据源配置文件，默认 configs/sources.yaml"),
    only: list[str] | None = typer.Option(None, help="只导入指定的数据源（可重复）"),
    force_delete: bool = typer.Option(False, help="允许删除比例超过安全阈值（确认数据源确实删了大量文件时用）"),
) -> None:
    """按 sources.yaml 导入数据（增量：没变化的文档会跳过）。"""
    kb = _kb(auto_ingest=False)
    reports = kb.ingest(sources, only, force_delete)
    t = Table("source", "status", "added", "updated", "unchanged", "failed", "deleted", "chunks", "emb new/reused", "s")
    for r in reports:
        t.add_row(r.source_id, r.status, str(r.added), str(r.updated), str(r.unchanged), str(r.failed), str(r.deleted),
                  str(r.chunks_written), f"{r.embeddings_computed}/{r.embeddings_reused}", str(r.duration_s))
    console.print(t)
    for r in reports:
        for e in r.errors[:5]:
            console.print(f"[red]{r.source_id}: {e}[/red]")
    if any(r.status == "failed" for r in reports):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- 查询


@app.command()
def search(
    question: str,
    groups: str | None = typer.Option(None, help="用户组，逗号分隔，例如 public,service"),
    model: list[str] | None = typer.Option(None, help="按机型硬过滤"),
    top_k: int = typer.Option(10),
) -> None:
    """只做检索，打印每个结果在三路检索里的排名（调试检索效果用）。"""
    kb = _kb()
    r = kb.search(question, _groups(groups), product_models=model, top_k=top_k)
    a = r.analysis
    console.print(f"[bold]实体[/bold]: {a.entity_ids}  [bold]扩展[/bold]: {a.expanded_ids}")
    console.print(f"[bold]关键词[/bold]: {a.tokens[:20]}")
    t = Table("#", "score", "dense", "kw", "entity", "rerank", "doc_ref", "section", "dups")
    for i, sc in enumerate(r.results, 1):
        rk = sc.channel_ranks
        t.add_row(str(i), f"{sc.score:.4f}", str(rk.get("dense", "")), str(rk.get("keyword", "")),
                  str(rk.get("entity", "")), f"{sc.rerank_score:.3f}" if sc.rerank_score is not None else "",
                  sc.chunk.doc_ref, " > ".join(sc.chunk.heading_path[1:])[:40], str(len(sc.duplicates)))
    console.print(t)
    console.print(f"[dim]{r.timings_ms}[/dim]")


@app.command()
def ask(
    question: str,
    groups: str | None = typer.Option(None, help="用户组，逗号分隔"),
    model: list[str] | None = typer.Option(None, help="按机型硬过滤"),
    as_json: bool = typer.Option(False, "--json", help="输出 JSON"),
) -> None:
    """问答：检索 + 生成，带引用。"""
    kb = _kb()
    r = kb.ask(question, _groups(groups), product_models=model)
    if as_json:
        print(json.dumps({
            "query_id": r.query_id, "answer": r.answer, "escalate": r.escalate, "reason": r.reason,
            "citations": [c.__dict__ for c in r.citations], "facts": r.facts, "timings_ms": r.timings_ms,
        }, ensure_ascii=False, indent=2))
        return
    console.print(f"\n[bold]{question}[/bold]\n")
    console.print(r.answer)
    if r.citations:
        console.print("\n[bold]引用[/bold]")
        for c in r.citations:
            section = " > ".join(c.heading_path[1:])
            console.print(f"  [{c.index}] {c.title}  {section}  [dim]{c.uri}[/dim]")
    if r.escalate:
        console.print(f"\n[yellow]⚠ 已标记升级：{r.reason}[/yellow]")
    console.print(f"\n[dim]model={r.model} timings={r.timings_ms} query_id={r.query_id}[/dim]")


# --------------------------------------------------------------------------- 评估


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(Path("eval/golden_qa.jsonl")),
    k: int = typer.Option(5),
    out: Path = typer.Option(Path("eval/reports")),
    min_hit: float = typer.Option(0.0, help="hit@k 低于该值时返回非零退出码（CI 回归门禁）"),
    min_refusal: float = typer.Option(0.0, help="拒答准确率低于该值时返回非零退出码"),
) -> None:
    """跑评估集，输出 Markdown + JSON 报告。"""
    from servicekb.evaluation import load_cases, run_eval, save_report

    kb = _kb()
    rep = run_eval(kb, load_cases(golden), k)
    md, _ = save_report(rep, out)
    console.print(rep.to_markdown())
    console.print(f"[green]报告已写入 {md}[/green]")
    hit = rep.retrieval[f"hit@{k}"]
    if hit < min_hit or rep.answer["refusal_accuracy"] < min_refusal:
        console.print(f"[red]未达到门槛：hit@{k}={hit} (>= {min_hit}), refusal={rep.answer['refusal_accuracy']} (>= {min_refusal})[/red]")
        raise typer.Exit(1)


@app.command()
def demo() -> None:
    """一键演示：导入示例数据并回答几个典型问题（含一个应该拒答的问题）。"""
    kb = _kb()
    console.print(kb.store.stats())
    for q in [
        "AS-110K-HV 早上报 E037，中午自己恢复，怎么处理？",
        "310-04521-A 还能订到吗？",
        "AS-250K-HV 的直流开关坏了，仓库里的 220-11008-A 能用吗？",
        "逆变器支持接入风力发电机吗？",
    ]:
        r = kb.ask(q)
        console.rule(q)
        console.print(r.answer)
        for c in r.citations:
            console.print(f"  [{c.index}] {c.doc_ref}  {' > '.join(c.heading_path[1:])}", style="dim")
        if r.escalate:
            console.print(f"[yellow]⚠ escalate: {r.reason}[/yellow]")


# --------------------------------------------------------------------------- 数据库


@db_app.command("init")
def db_init() -> None:
    """建表 / 执行迁移，并写入向量模型信息。"""
    kb = _kb(auto_ingest=False)
    console.print(kb.store.stats())


@db_app.command("stats")
def db_stats() -> None:
    kb = _kb(auto_ingest=False)
    console.print_json(json.dumps(kb.store.stats(), ensure_ascii=False, default=str))


@db_app.command("reset")
def db_reset(yes: bool = typer.Option(False, "--yes", help="确认删除所有数据")) -> None:
    """删除所有表（换向量模型时需要）。危险操作！"""
    s = _settings()
    if s.store_backend != "postgres":
        console.print("memory 模式无需 reset")
        return
    if not yes:
        console.print("[red]这会删除所有数据。确认请加 --yes[/red]")
        raise typer.Exit(1)
    from servicekb.store.postgres import PostgresStore

    st = PostgresStore(s.database_url)
    st.reset()
    st.close()
    console.print("[green]已删除所有表。执行 `servicekb db init` 和 `servicekb ingest` 重建。[/green]")


# --------------------------------------------------------------------------- 审计


@audit_app.command("parts")
def audit_parts(
    path: Path = typer.Option(Path("data/sample/parts/parts_master.csv")),
    threshold: int = typer.Option(85, help="名称相似度阈值（0-100）"),
) -> None:
    """物料主数据质量：重复料号、相似名称、引用了不存在的料号。"""
    from rapidfuzz import fuzz

    from servicekb.connectors.parts_master import PartsMasterConnector
    from servicekb.processing.text import normalize_key

    pm = PartsMasterConnector("parts_master", path)
    parts = pm._load()
    console.print(f"[bold]{len(parts)} 个料号[/bold]")
    for w in pm.warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    items = [(pn, p["name_zh"]) for pn, p in parts.items() if p["name_zh"]]
    t = Table("料号 A", "名称 A", "料号 B", "名称 B", "相似度")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            score = fuzz.ratio(normalize_key(a[1]), normalize_key(b[1]))
            if score >= threshold:
                t.add_row(a[0], a[1], b[0], b[1], f"{score:.0f}")
    console.print("[bold]名称相似、料号不同（可能是同一零件的不同版本，也可能是重复建档）[/bold]")
    console.print(t)


@audit_app.command("qa")
def audit_qa(
    path: Path = typer.Option(Path("data/sample/qa_archive/tickets.jsonl")),
    max_distance: int = typer.Option(3),
) -> None:
    """历史问答：近似重复工单簇、被过滤的工单统计。"""
    from servicekb.connectors.qa_archive import QAArchiveConnector
    from servicekb.processing.dedup import cluster_near_duplicates, simhash

    c = QAArchiveConnector("qa_archive", path)
    recs = list(c.iter_events(None))
    items = [(r.external_id, simhash(json.loads(r.content)["question"])) for r in recs]  # type: ignore[union-attr]
    console.print(c.stats())
    for cl in cluster_near_duplicates(items, max_distance):
        console.print(f"[yellow]近似重复：{', '.join(cl)}[/yellow]")


# --------------------------------------------------------------------------- 聊天记录


@chat_app.command("clean")
def chat_clean(
    input: Path = typer.Option(Path("data/sample/chat/teams_export_sample.jsonl"), help="Graph 导出的 JSONL / JSON / 目录"),
    roster: Path = typer.Option(Path("data/sample/chat/roster.csv"), help="人员名册 user_id,display_name,role"),
    out_dir: Path = typer.Option(Path("data/sample/chat"), help="输出目录"),
    gap_minutes: int = typer.Option(120, help="群聊会话切分的时间间隔（分钟）"),
    llm: bool = typer.Option(False, help="用 LLM 抽取问答（需要配置 SERVICEKB_LLM_*）"),
) -> None:
    """Teams 聊天记录 → 清洗 → 线程 → 问答对（三路分流 + 清洗报告）。"""
    from servicekb.chat import ChatCleaner, ChatCleaningConfig, Roster, write_outputs
    from servicekb.chat.teams_source import iter_raw_messages

    s = _settings()
    kb = _kb(auto_ingest=False)
    if s.store_backend == "memory":  # 需要对象层来识别型号 / 料号
        kb.ingest(only=["dictionaries", "parts_master"])
    extractor = None
    if llm:
        from servicekb.chat.llm_extract import LLMChatExtractor

        extractor = LLMChatExtractor(s.llm_model, s.llm_api_key.get_secret_value() if s.llm_api_key else None,
                                     s.llm_base_url)
    cleaner = ChatCleaner(ChatCleaningConfig(session_gap_minutes=gap_minutes), Roster.from_csv(roster), kb.linker)
    pairs = cleaner.run(iter_raw_messages(input), extractor)
    paths = write_outputs(pairs, cleaner, out_dir)
    console.print(cleaner.report.to_markdown())
    for k, p in paths.items():
        console.print(f"[green]{k}: {p}[/green]")


@chat_app.command("export")
def chat_export(
    team_id: str = typer.Option(...),
    channel_id: str = typer.Option(...),
    out: Path = typer.Option(Path("raw/teams_channel.jsonl")),
) -> None:
    """从 Microsoft Graph 导出频道消息（需要 SERVICEKB_GRAPH_TENANT_ID / CLIENT_ID / CLIENT_SECRET）。"""
    import os

    from servicekb.chat.teams_source import TeamsChannelExporter
    from servicekb.connectors.sharepoint import GraphClient

    client = GraphClient(os.environ["SERVICEKB_GRAPH_TENANT_ID"], os.environ["SERVICEKB_GRAPH_CLIENT_ID"],
                         os.environ["SERVICEKB_GRAPH_CLIENT_SECRET"])
    n = TeamsChannelExporter(client).export(team_id, channel_id, out)
    console.print(f"[green]导出 {n} 条消息 → {out}[/green]")


# --------------------------------------------------------------------------- 服务


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:
    """启动 HTTP API（FastAPI）。文档：http://localhost:8000/docs"""
    import uvicorn

    s = _settings()
    if s.store_backend == "memory" and workers > 1:
        console.print("[red]memory 模式不支持多 worker（每个进程各有一份数据）[/red]")
        sys.exit(1)
    uvicorn.run("servicekb.api:create_app", host=host, port=port, workers=workers, factory=True)


if __name__ == "__main__":
    app()
