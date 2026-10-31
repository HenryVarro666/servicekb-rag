"""清洗：去噪 + 脱敏。

工业数据的"脏"主要来自三处：
1. **邮件 / 工单往来**：大量引用的历史邮件、签名、免责声明。如果不去掉，同一段话会在
   几十张工单里重复出现，检索结果被"噪声"占满。
2. **个人信息（PII）**：客户手机号、邮箱。知识库会被很多人检索，甚至可能把内容发给外部
   LLM API，必须在入库前脱敏。
3. **格式垃圾**：零宽字符、全角半角混用（见 text.normalize_text）。

聊天记录（Teams 等）的清洗更复杂，单独放在 ``servicekb.chat`` 包里，见 docs/09_chat_log_cleaning.md。

这里全部用正则实现，零依赖、可解释。规模变大或要求更高时可以换成：
- 邮件引用 / 签名：mailgun/talon（基于机器学习的签名识别）、email-reply-parser
- PII：microsoft/presidio（支持自定义识别器，可接中文 NER 模型）
"""

from __future__ import annotations

import re

from servicekb.processing.text import normalize_text

# --------------------------------------------------------------------------- 邮件 / 工单去噪

# 出现这些行，说明后面是"被引用的历史邮件"，整段截掉
_QUOTE_HEADERS = [
    r"^-{2,}\s*Original Message\s*-{2,}",
    r"^-{2,}\s*原始邮件\s*-{2,}",
    r"^From:\s.+",  # Outlook 英文引用头
    r"^发件人[:：].+",  # Outlook 中文引用头
    r"^On .{5,80} wrote:$",  # Gmail 英文
    r"^在 .{5,80}写道[:：]$",  # Gmail 中文
    r"^_{10,}$",  # Outlook 分隔线
]
_QUOTE_HEADER_RE = re.compile("|".join(_QUOTE_HEADERS), re.IGNORECASE | re.MULTILINE)

# 签名开始的标志行（从这一行开始到结尾都视为签名）
_SIGNATURE_STARTS = re.compile(
    r"^(best regards|kind regards|regards|thanks(?: and regards)?|thank you|br|cheers|"
    r"此致|敬礼|顺祝商祺|祝好|谢谢|--|——)[,，!！.。\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# 免责声明：通常是一大段固定文字
_DISCLAIMER = re.compile(
    r"(this (e-?mail|message) and any attachments? (is|are) confidential.*|"
    r"本邮件(及其附件)?(含有|包含).{0,20}(保密|机密)信息.*)",
    re.IGNORECASE | re.DOTALL,
)


def strip_quoted_email(text: str) -> str:
    """只去掉"被引用的历史邮件"和 "> " 引用行（聊天记录里粘贴的邮件也用这个）。"""
    m = _QUOTE_HEADER_RE.search(text)
    if m and m.start() > 0:
        text = text[: m.start()]
    return "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith(">"))


def strip_email_noise(text: str) -> str:
    """去掉引用的历史邮件、签名、免责声明、"> " 开头的引用行。

    注意顺序：先截引用（引用里也可能有签名），再截签名，最后去免责声明。
    签名规则只在**正文后半部分**生效，避免把正文第一行的"谢谢"当成签名截掉。
    """
    text = strip_quoted_email(text)

    for m in _SIGNATURE_STARTS.finditer(text):
        if m.start() > len(text) * 0.4:
            text = text[: m.start()]
            break

    text = _DISCLAIMER.sub("", text)
    return normalize_text(text)


# --------------------------------------------------------------------------- PII 脱敏

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 中国大陆手机号 / 美国电话 / 带国家码的国际号码
_PHONE = re.compile(
    r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)"
    r"|(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
# 中国大陆身份证号
_CN_ID = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact_pii(text: str) -> str:
    """把邮箱、电话、身份证号替换成占位符。

    为什么用占位符而不是直接删除？保留"这里原来有个电话"的语义，LLM 读起来更通顺，
    也方便审计时知道哪里被脱敏了。

    **不脱敏**：设备序列号、料号——它们是售后知识的关键信息。
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CN_ID.sub("[ID]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def clean_text(text: str, *, email_like: bool = False) -> str:
    """清洗总入口。``email_like=True`` 时额外去除邮件噪声（工单、邮件往来用）。"""
    text = normalize_text(text)
    if email_like:
        text = strip_email_noise(text)
    return redact_pii(text)
