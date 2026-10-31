"""Teams 聊天记录清洗测试。"""

from pathlib import Path

from servicekb.chat import ChatCleaner, ChatCleaningConfig, Roster, write_outputs
from servicekb.chat.teams_html import teams_html_to_text
from servicekb.chat.teams_source import iter_raw_messages

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "chat"


def test_html_conversion_handles_teams_markup():
    html = (
        '<blockquote itemscope="" itemtype="http://schema.skype.com/Reply" itemid="111"><p>旧消息</p></blockquote>'
        '<p><at id="0">王工</at> 看下 <emoji id="like" alt="👍" title="Like"></emoji></p>'
        '<p>文件 <attachment id="a1"></attachment><img src="https://graph.microsoft.com/x/hostedContents/1/$value"></p>'
        '<codeblock class="plaintext"><code>V2.3.0</code></codeblock><ul><li>一</li><li>二</li></ul>'
    )
    text, quoted = teams_html_to_text(html, "html", [{"id": "a1", "contentType": "reference", "name": "SB.pdf"}])
    assert "旧消息" not in text and quoted == ["111"]
    assert "@王工 看下 👍" in text and "[附件: SB.pdf]" in text and "[图片]" in text
    assert "```\nV2.3.0\n```" in text and "- 一\n- 二" in text


def test_message_reference_attachment_is_removed():
    text, quoted = teams_html_to_text(
        '<attachment id="q1"></attachment><p>同意</p>', "html",
        [{"id": "q1", "contentType": "messageReference", "content": '{"messageId":"999","messagePreview":"原文"}'}],
    )
    assert text == "同意" and quoted == ["999"]


def _run(**cfg):
    from servicekb.connectors.dictionary import EntityDictionaryConnector
    from servicekb.connectors.parts_master import PartsMasterConnector
    from servicekb.processing.entities import EntityLinker

    e1, r1 = EntityDictionaryConnector("d", ROOT / "data/sample/dictionaries/entity_aliases.yaml").entities()
    e2, r2 = PartsMasterConnector("p", ROOT / "data/sample/parts/parts_master.csv").entities()
    c = ChatCleaner(ChatCleaningConfig(**cfg), Roster.from_csv(SAMPLE / "roster.csv"), EntityLinker(e1 + e2, r1 + r2))
    pairs = c.run(iter_raw_messages(SAMPLE / "teams_export_sample.jsonl"))
    return c, {p.qa_id.rsplit(":", 1)[-1]: p for p in pairs}


def test_filters_and_funnel():
    c, pairs = _run()
    d = c.report.dropped
    assert d["system_event"] == 1 and d["bot_or_app"] == 1 and d["deleted"] == 1 and d["policy_violation"] == 1
    assert c.report.unknown_authors["u-9999"] == 2  # 名册缺失 → 报告提醒
    assert c.report.counts["qa_ready"] >= 5 and c.report.counts["qa_gaps"] == 1


def test_burst_merge_pii_and_pseudonym():
    _, pairs = _run()
    q = pairs["1001"].question
    assert "补充：那边已经连续下了一周雨" in q  # 连发合并
    assert "[PHONE]" in q and "13800138000" not in q
    assert "王强" not in q  # 人名化名


def test_resolution_signals():
    _, pairs = _run()
    assert pairs["1001"].status == "resolved"  # 提问者说"好了"
    assert pairs["3001"].status == "resolved"  # 提问者给工程师的回复点了赞
    assert pairs["5001"].status == "unresolved"  # 没人回答 → 知识缺口


def test_non_expert_answer_excluded_and_safety_review():
    _, pairs = _run()
    torque = pairs["7001"]
    assert "3 N·m" not in torque.answer.replace("不是 3", "") and "2.5 N·m" in torque.answer
    assert "non_expert_replies_excluded" in torque.review_reasons
    polarity = pairs["10001"]
    assert polarity.needs_review and any(r.startswith("safety:") for r in polarity.review_reasons)


def test_group_chat_sessionization():
    _, pairs = _run(session_gap_minutes=120)
    assert "c1" in pairs and "c6" in pairs  # 隔天的消息被切成两个会话
    assert pairs["c1"].status == "resolved"
    _, merged = _run(session_gap_minutes=60 * 48)
    assert "c6" not in merged  # 间隔放大到 48 小时 → 合成一个会话


def test_pasted_email_is_stripped():
    _, pairs = _run()
    assert "From:" not in pairs["11001"].question and "customer@" not in pairs["11001"].question


def test_outputs_feed_qa_connector(tmp_path):
    from servicekb.connectors.qa_archive import QAArchiveConnector

    c, pairs = _run()
    paths = write_outputs(list(pairs.values()), c, tmp_path)
    recs = list(QAArchiveConnector("teams", paths["ready"], resolved_statuses=["resolved"], allow_row_acl=True,
                                   doc_type="chat").iter_events(None))
    assert recs and all(r.uri.startswith("https://teams.microsoft.com/") for r in recs)
    assert "漏斗" in paths["report"].read_text("utf-8")
