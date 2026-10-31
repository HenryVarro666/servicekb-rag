"""聊天记录（Teams 等）清洗与问答抽取。入口：``servicekb chat clean``，文档：docs/09_chat_log_cleaning.md。"""

from __future__ import annotations

import json
from pathlib import Path

from servicekb.chat.cleaner import ChatCleaner, ChatCleaningConfig, CleaningReport, Roster
from servicekb.chat.models import ChatMessage, QAPair, Thread


def write_outputs(pairs: list[QAPair], cleaner: ChatCleaner, out_dir: Path) -> dict[str, Path]:
    """三路分流输出 + 清洗报告。

    - teams_qa.jsonl        ：已解决、无需审核 → 直接作为知识库数据源（QAArchiveConnector 可读）
    - review_queue.jsonl    ：需要人工审核（未确认解决 / 低置信度 / 涉及安全操作 / LLM 校验不过）
    - knowledge_gaps.jsonl  ：没人回答的问题 → 知识缺口清单，交给技术支持补充文档
    - cleaning_report.md    ：漏斗统计、丢弃原因、名册缺失、未知料号、重复簇
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / v for k, v in {
        "ready": "teams_qa.jsonl", "review": "review_queue.jsonl",
        "gaps": "knowledge_gaps.jsonl", "report": "cleaning_report.md",
    }.items()}
    with paths["ready"].open("w", encoding="utf-8") as fr, paths["review"].open("w", encoding="utf-8") as fv, \
            paths["gaps"].open("w", encoding="utf-8") as fg:
        for p in pairs:
            row = p.to_archive_row()
            row["acl_groups"] = cleaner.acl_for(p)
            row["review_reasons"] = p.review_reasons
            if p.status == "unresolved":
                fg.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif p.needs_review:
                fv.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                fr.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["report"].write_text(cleaner.report.to_markdown(), "utf-8")
    return paths


__all__ = ["ChatCleaner", "ChatCleaningConfig", "ChatMessage", "CleaningReport", "QAPair", "Roster", "Thread",
           "write_outputs"]
