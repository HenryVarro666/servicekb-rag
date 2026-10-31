"""聊天记录清洗 pipeline（Teams 为例，Slack / 企业微信导出改一下 ``parse_message`` 即可复用）。

为什么聊天记录比工单更难清洗？
    工单天然是"一问一答"的结构；聊天记录是**流**：系统消息、机器人、表情回复、"收到"、
    一句话分三条发、几个话题交错、引用回复把原文又贴一遍、非专家的错误回答、
    没有结论就不了了之……直接把聊天记录切块入库，检索结果会被噪声淹没，还会把错误答案当成知识。

本 pipeline 的 8 个阶段（每个阶段的丢弃原因都会计数，写进清洗报告）::

    1 解析     Graph JSON → ChatMessage；复合主键（Graph 的 id 只在单个对话内唯一）
    2 过滤     系统事件 / 已删除 / 机器人 / DLP 拦截 / 清洗后为空
    3 清洗     HTML→文本、去引用回复、粘贴邮件去引用、PII 脱敏、人名→角色化名、规范化
    4 识别噪声  "收到""👍""thanks" 这类确认消息：不进问答正文，但用于判断"问题是否解决"
    5 组线程   频道：按 replyToId；群聊：按时间间隔切会话（sessionization）
    6 合并连发  同一人短时间内连发的多条消息合并成一条
    7 抽取问答  规则抽取（默认）或 LLM 抽取（--llm）：问题、专家答案、是否解决、置信度
    8 分流     可直接入库 / 需要人工审核 / 知识缺口（没人回答的问题）

详见 docs/09_chat_log_cleaning.md。
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from servicekb.chat.models import ChatMessage, QAPair, Role, Thread
from servicekb.chat.teams_html import teams_html_to_text
from servicekb.processing.cleaning import redact_pii, strip_quoted_email
from servicekb.processing.dedup import cluster_near_duplicates, simhash
from servicekb.processing.entities import EntityLinker
from servicekb.processing.text import normalize_text

# --------------------------------------------------------------------------- 配置


@dataclass
class ChatCleaningConfig:
    session_gap_minutes: int = 120  # 群聊：相邻消息间隔超过它就切成新会话
    merge_window_seconds: int = 180  # 同一人连发合并的时间窗口
    min_question_chars: int = 8
    min_answer_chars: int = 15
    expert_roles: tuple[str, ...] = ("engineer",)  # 谁的回答算"专家答案"
    drop_bots: bool = True
    pseudonymize: bool = True  # 人名 → "工程师A"、"现场B"
    review_confidence: float = 0.7  # 置信度低于它进人工审核
    # 答案涉及这些内容时强制人工审核（错误答案可能导致人身伤害或设备损坏）
    safety_keywords: tuple[str, ...] = (
        "高压", "带电", "拆机", "放电", "1500V", "1100V", "刷机", "短接", "绕过", "bypass", "live", "polarity",
    )
    # 对话 → 权限组。频道成员即可见范围，必须映射过来，否则私有频道的内容会被所有人检索到
    acl_by_conversation: dict[str, list[str]] = field(default_factory=dict)
    default_acl: list[str] = field(default_factory=lambda: ["service"])


_ROLE_LABEL = {
    "engineer": "工程师", "field_tech": "现场", "service_partner": "服务商",
    "customer": "客户", "bot": "机器人", "unknown": "用户",
}


class Roster:
    """人员名册：user_id → (显示名, 角色)。来源可以是 Entra ID 分组导出、HR 系统，或手工维护的 CSV。

    角色决定了"谁的回答可信"。名册缺失的人记为 unknown，会出现在清洗报告里提醒补全。
    """

    def __init__(self, people: dict[str, tuple[str, Role]] | None = None) -> None:
        self.people = people or {}
        self._pseudo: dict[str, str] = {}
        self._counters: Counter[str] = Counter()

    @classmethod
    def from_csv(cls, path: Path) -> Roster:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return cls({r["user_id"]: (r["display_name"], r["role"]) for r in csv.DictReader(f)})  # type: ignore[misc]

    def role(self, user_id: str) -> Role:
        return self.people.get(user_id, ("", "unknown"))[1]

    def pseudonym(self, user_id: str) -> str:
        """稳定化名：同一个人在所有对话里都是同一个化名，便于阅读，又不暴露真名。"""
        if user_id not in self._pseudo:
            label = _ROLE_LABEL[self.role(user_id)]
            idx = self._counters[label]
            self._counters[label] += 1
            self._pseudo[user_id] = f"{label}{chr(ord('A') + idx % 26)}{'' if idx < 26 else idx // 26}"
        return self._pseudo[user_id]


# --------------------------------------------------------------------------- 清洗报告


@dataclass
class CleaningReport:
    counts: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unknown_authors: Counter[str] = field(default_factory=Counter)
    unknown_entities: Counter[str] = field(default_factory=Counter)
    duplicate_clusters: list[list[str]] = field(default_factory=list)

    def drop(self, reason: str, example: str = "") -> None:
        self.dropped[reason] += 1
        if example and len(self.examples[reason]) < 3:
            self.examples[reason].append(example[:120].replace("\n", " "))

    def to_markdown(self) -> str:
        c = self.counts
        lines = [
            "# 聊天记录清洗报告", "",
            "## 漏斗", "",
            "| 阶段 | 数量 |", "| --- | --- |",
            f"| 原始消息 | {c['raw_messages']} |",
            f"| 过滤后消息 | {c['kept_messages']} |",
            f"| 连发合并后消息 | {c['merged_messages']} |",
            f"| 线程 / 会话 | {c['threads']} |",
            f"| 问答候选 | {c['qa_candidates']} |",
            f"| ✅ 可直接入库（已解决且无需审核） | {c['qa_ready']} |",
            f"| 📝 需人工审核 | {c['qa_review']} |",
            f"| ❓ 知识缺口（无人回答） | {c['qa_gaps']} |",
            "", "## 丢弃原因", "", "| 原因 | 数量 | 示例 |", "| --- | --- | --- |",
        ]
        for reason, n in self.dropped.most_common():
            ex = "<br>".join(self.examples.get(reason, [])) or "-"
            lines.append(f"| {reason} | {n} | {ex} |")
        if self.unknown_authors:
            lines += ["", "## 名册中缺失的人员（请补全 roster，否则其回答不会被当作专家答案）", ""]
            lines += [f"- `{uid}`：{n} 条消息" for uid, n in self.unknown_authors.most_common()]
        if self.unknown_entities:
            lines += ["", "## 物料主数据中不存在的料号 / 故障码（数据质量问题）", ""]
            lines += [f"- `{e}`：出现 {n} 次" for e, n in self.unknown_entities.most_common()]
        if self.duplicate_clusters:
            lines += ["", "## 近似重复的问答（检索时会自动折叠）", ""]
            lines += [f"- {', '.join(cl)}" for cl in self.duplicate_clusters]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- 规则

_ACK = re.compile(
    r"^(ok|okay|k|收到|好的?|嗯+|行|谢谢|多谢|感谢|thanks?|thx|ty|got it|noted|👍|🙏|👌|\+1|1|了解|明白)+[!！。.~～\s]*$",
    re.IGNORECASE,
)
_RESOLVED = re.compile(
    r"(解决了|好了|可以了|恢复了|恢复正常|搞定|正常了|没问题了|已解决|成功了|works now|it works|worked|fixed|solved|resolved|back online)",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"[?？]|吗|怎么|如何|怎样|为什么|为啥|能否|是否|哪个|多少|求助|请教|help|how|what|why|which|can i|should", re.IGNORECASE)
_POSITIVE_REACTIONS = {"like", "heart", "👍", "❤️", "✅"}
_MENTION = re.compile(r"@\S+\s*")  # 化名后的 @提及（"@工程师A "），对知识内容是噪声
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def _parse_dt(s: str | None) -> datetime:
    return datetime.fromisoformat((s or "1970-01-01T00:00:00Z").replace("Z", "+00:00"))


class ChatCleaner:
    def __init__(self, cfg: ChatCleaningConfig | None = None, roster: Roster | None = None,
                 linker: EntityLinker | None = None) -> None:
        self.cfg = cfg or ChatCleaningConfig()
        self.roster = roster or Roster()
        self.linker = linker or EntityLinker()
        self.report = CleaningReport()

    # ------------------------------------------------------------------ 阶段 1+2：解析 & 过滤
    def parse_message(self, raw: dict[str, Any]) -> ChatMessage | None:
        r = self.report
        mtype = raw.get("messageType", "message")
        body = raw.get("body") or {}
        preview = str(body.get("content", ""))
        if mtype != "message":
            r.drop("system_event", preview)  # 成员加入 / 改名 / 会议开始等
            return None
        if raw.get("deletedDateTime"):
            r.drop("deleted")  # 已删除：内容通常为空，且用户撤回意味着不应进入知识库
            return None
        if raw.get("policyViolation"):
            r.drop("policy_violation")  # 被 DLP 策略拦截的消息，绝不能入库
            return None
        sender = raw.get("from") or {}
        user = sender.get("user") or {}
        if (sender.get("application") or not user) and self.cfg.drop_bots:
            r.drop("bot_or_app", preview)  # 机器人、连接器、webhook 推送
            return None

        ch = raw.get("channelIdentity") or {}
        conv = f"{ch.get('teamId')}/{ch.get('channelId')}" if ch.get("channelId") else str(raw.get("chatId", ""))
        uid = str(user.get("id", "unknown"))
        role = self.roster.role(uid)
        if role == "unknown":
            r.unknown_authors[uid] += 1
        text, quoted = teams_html_to_text(preview, body.get("contentType", "html"), raw.get("attachments"))
        return ChatMessage(
            key=f"{conv}:{raw['id']}", message_id=str(raw["id"]), conversation_id=conv,
            reply_to=raw.get("replyToId"), created_at=_parse_dt(raw.get("createdDateTime")),
            author_id=uid, author_name=str(user.get("displayName", "")), role=role,
            html=preview, text=text, message_type=mtype, edited=bool(raw.get("lastEditedDateTime")),
            web_url=str(raw.get("webUrl", "")), attachments=raw.get("attachments") or [],
            reactions=raw.get("reactions") or [], quoted_message_ids=quoted,
        )

    # ------------------------------------------------------------------ 阶段 3：文本清洗
    def clean_message(self, m: ChatMessage) -> ChatMessage | None:
        text = strip_quoted_email(m.text)  # 有人会把整封邮件粘进聊天
        if self.cfg.pseudonymize:
            for uid, (name, _) in self.roster.people.items():
                if name and name in text:
                    text = text.replace(name, self.roster.pseudonym(uid))
            m.author_name = self.roster.pseudonym(m.author_id)
        text = normalize_text(redact_pii(text))
        if not text:  # 只有图片 / 引用卡片的消息，清洗后为空
            self.report.drop("empty_after_cleaning", m.html)
            return None
        m.text = text
        return m

    @staticmethod
    def is_ack(text: str) -> bool:
        """纯确认 / 纯表情消息。"""
        rest = _EMOJI.sub("", text).strip()
        return not rest or bool(_ACK.match(rest))

    # ------------------------------------------------------------------ 阶段 5：组线程
    def build_threads(self, msgs: list[ChatMessage]) -> list[Thread]:
        threads: list[Thread] = []
        by_conv: dict[str, list[ChatMessage]] = defaultdict(list)
        for m in msgs:
            by_conv[m.conversation_id].append(m)
        for conv, ms in by_conv.items():
            ms.sort(key=lambda x: x.created_at)
            if any(m.reply_to for m in ms) or "/" in conv:
                # 频道：天然有线程结构。根消息 replyToId 为空，回复的 replyToId = 根消息 id
                roots: dict[str, list[ChatMessage]] = {}
                for m in ms:
                    tid = m.reply_to or m.message_id
                    roots.setdefault(tid, []).append(m)
                for tid, tm in roots.items():
                    if tm[0].message_id != tid:
                        self.report.counts["orphan_threads"] += 1  # 根消息被删了，只剩回复
                    threads.append(Thread(f"{conv}:{tid}", conv, tm, "channel_thread"))
            else:
                # 群聊：没有线程，按时间间隔切会话。更精细的做法（按 @提及、引用、故障码切分）见文档
                gap = timedelta(minutes=self.cfg.session_gap_minutes)
                cur: list[ChatMessage] = []
                for m in ms:
                    if cur and m.created_at - cur[-1].created_at > gap:
                        threads.append(Thread(f"{conv}:{cur[0].message_id}", conv, cur, "chat_session"))
                        cur = []
                    cur.append(m)
                if cur:
                    threads.append(Thread(f"{conv}:{cur[0].message_id}", conv, cur, "chat_session"))
        return threads

    # ------------------------------------------------------------------ 阶段 6：合并连发
    def merge_bursts(self, t: Thread) -> Thread:
        window = timedelta(seconds=self.cfg.merge_window_seconds)
        out: list[ChatMessage] = []
        for m in t.messages:
            prev = out[-1] if out else None
            if (prev and prev.author_id == m.author_id and m.created_at - prev.created_at <= window
                    and not self.is_ack(m.text) and not self.is_ack(prev.text)):
                prev.text = f"{prev.text}\n{m.text}"
                prev.merged_ids.append(m.message_id)
                prev.reactions += m.reactions
                prev.created_at = m.created_at  # 以最后一条的时间作为"连发结束时间"，便于连续合并
            else:
                out.append(m)
        t.messages = out
        return t

    # ------------------------------------------------------------------ 阶段 7：规则抽取问答
    def extract_qa(self, t: Thread) -> QAPair | None:
        cfg = self.cfg
        msgs = t.messages
        asker = msgs[0]
        question_parts = [asker.text]
        i = 1
        while i < len(msgs) and msgs[i].author_id == asker.author_id:  # 提问者在专家回复前的补充
            if not self.is_ack(msgs[i].text):
                question_parts.append(msgs[i].text)
            i += 1
        question = _MENTION.sub("", "\n".join(question_parts)).strip()
        if len(question) < cfg.min_question_chars or not (_QUESTION.search(question) or self.linker.fault_re.search(question)):
            self.report.drop("thread_not_a_question", question)
            return None

        expert_msgs = [m for m in msgs[1:] if m.role in cfg.expert_roles and m.author_id != asker.author_id
                       and not self.is_ack(m.text)]
        answer = _MENTION.sub("", "\n".join(m.text for m in expert_msgs)).strip()
        expert_ids = {m.message_id for m in expert_msgs}

        # 是否解决：提问者在专家回复之后说"好了 / 解决了"，或给专家的回复点了赞
        resolved = False
        if expert_msgs:
            after = [m for m in msgs if m.author_id == asker.author_id and m.created_at > expert_msgs[0].created_at]
            resolved = any(_RESOLVED.search(m.text) for m in after)
            for m in expert_msgs:
                for rc in m.reactions:
                    who = ((rc.get("user") or {}).get("user") or {}).get("id")
                    if who == asker.author_id and str(rc.get("reactionType", "")).lower() in _POSITIVE_REACTIONS:
                        resolved = True
        if not expert_msgs or len(answer) < cfg.min_answer_chars:
            status = "unresolved"
        else:
            status = "resolved" if resolved else "answered_unconfirmed"

        mentions = self.linker.extract(f"{question}\n{answer}")
        for mm in mentions:
            if not mm.known and mm.entity_id.startswith("part:"):
                self.report.unknown_entities[mm.entity_id] += 1
        models = [self.linker.entities[mm.entity_id].canonical_name for mm in mentions
                  if mm.entity_id.startswith("model:") and mm.entity_id in self.linker.entities]
        faults = [mm.entity_id.split(":", 1)[1] for mm in mentions if mm.entity_id.startswith("fault:")]
        parts = [mm.entity_id.split(":", 1)[1] for mm in mentions if mm.entity_id.startswith("part:")]

        non_expert_answers = [m for m in msgs[1:] if m.author_id != asker.author_id and m.role not in cfg.expert_roles
                              and not self.is_ack(m.text)]

        # 置信度：启发式打分，用于分流（不是概率！上线后用人工审核结果来校准）
        conf = 0.3
        conf += 0.35 if status == "resolved" else 0.0
        conf += 0.1 if (models or faults or parts) else 0.0
        conf += 0.1 if len(answer) >= 50 else 0.0
        conf += 0.1 if not non_expert_answers else 0.0  # 有非专家插话时，抽取更容易出错
        conf = min(conf, 1.0)

        reasons = []
        if status != "resolved":
            reasons.append(status)
        if conf < cfg.review_confidence:
            reasons.append("low_confidence")
        hits = [k for k in cfg.safety_keywords if k.lower() in answer.lower()]
        if hits:
            reasons.append("safety:" + ",".join(hits))
        if non_expert_answers:
            reasons.append("non_expert_replies_excluded")  # 不阻塞入库，但提示审核人看一眼

        root = t.root
        return QAPair(
            qa_id=f"teams:{t.thread_id}",
            question=question,
            answer=answer,
            status=status,  # type: ignore[arg-type]
            confidence=conf,
            created_at=root.created_at.date().isoformat(),
            product_model=models[0] if models else "",
            fault_codes=list(dict.fromkeys(faults)),
            part_numbers=list(dict.fromkeys(parts)),
            evidence_message_ids=[root.message_id, *sorted(expert_ids)],
            source_url=root.web_url,
            conversation_id=t.conversation_id,
            needs_review=bool([r for r in reasons if r != "non_expert_replies_excluded"]),
            review_reasons=reasons,
        )

    # ------------------------------------------------------------------ 主流程
    def run(self, raw_messages: Iterable[dict[str, Any]], llm_extractor: Any = None) -> list[QAPair]:
        r = self.report
        msgs: list[ChatMessage] = []
        for raw in raw_messages:
            r.counts["raw_messages"] += 1
            m = self.parse_message(raw)
            if m is not None:
                m = self.clean_message(m)
            if m is not None:
                msgs.append(m)
        r.counts["kept_messages"] = len(msgs)

        threads = [self.merge_bursts(t) for t in self.build_threads(msgs)]
        r.counts["threads"] = len(threads)
        r.counts["merged_messages"] = sum(len(t.messages) for t in threads)

        pairs: list[QAPair] = []
        for t in threads:
            qa = llm_extractor.extract(t, self) if llm_extractor else self.extract_qa(t)
            if qa is not None:
                pairs.append(qa)
        r.counts["qa_candidates"] = len(pairs)

        # 近似重复：不删除，只在报告里列出（检索时 collapse_near_duplicates 会折叠）
        clusters = cluster_near_duplicates([(p.qa_id, simhash(p.question)) for p in pairs], max_distance=6)
        r.duplicate_clusters = clusters
        by_id = {p.qa_id: p for p in pairs}
        for cl in clusters:
            for qid in cl:
                by_id[qid].duplicates = [x for x in cl if x != qid]

        r.counts["qa_ready"] = sum(1 for p in pairs if p.status == "resolved" and not p.needs_review)
        r.counts["qa_review"] = sum(1 for p in pairs if p.status != "unresolved" and p.needs_review)
        r.counts["qa_gaps"] = sum(1 for p in pairs if p.status == "unresolved")
        return pairs

    def acl_for(self, qa: QAPair) -> list[str]:
        return self.cfg.acl_by_conversation.get(qa.conversation_id, self.cfg.default_acl)
