"""Teams 消息 HTML → 纯文本。

Teams 的 ``body.content`` 通常是 HTML（只要有 @提及，就一定是 HTML），里面有几种 Teams 特有的标记，
处理不好会严重污染知识库：

+-------------------------------------------+--------------------------------------------------+
| 标记                                       | 处理                                              |
+===========================================+==================================================+
| ``<at id="0">王工</at>``                   | 保留为 "@王工"（或替换成角色名，见 pseudonymize）    |
| ``<attachment id="..."></attachment>``    | 按 attachments[] 换成 "[附件: 文件名]" / 删除引用卡片 |
| 旧版引用回复                                | 整块删除（被引用的原文在原消息里已经有了，重复会污染）  |
| ``<blockquote itemtype=".../Reply">``      |                                                  |
| ``<emoji alt="👍">``                       | 换成 alt 里的 emoji 字符                            |
| ``<img src=".../hostedContents/...">``    | "[图片]"（图片内容需要单独走 OCR / 多模态，见文档）    |
| ``<codeblock>`` / ``<pre>`` / ``<code>``  | 保留内容，前后加 ``` 标记                           |
| ``<p> <div> <br> <li>``                    | 换行 / 列表符号                                    |
+-------------------------------------------+--------------------------------------------------+

只用标准库 ``html.parser``，不依赖 BeautifulSoup。
"""

from __future__ import annotations

import html
import json
from html.parser import HTMLParser
from typing import Any

_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
_REPLY_ITEMTYPE = "schema.skype.com/Reply"


class _TeamsHTMLToText(HTMLParser):
    def __init__(self, attachments: dict[str, dict[str, Any]]) -> None:
        super().__init__(convert_charrefs=True)
        self.attachments = attachments
        self.out: list[str] = []
        self.skip_depth = 0  # >0 时处于"要丢弃的块"内部（引用回复）
        self.in_code = False
        self.quoted_ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if self.skip_depth:
            if tag == "blockquote":
                self.skip_depth += 1
            return
        if tag == "blockquote" and _REPLY_ITEMTYPE in (a.get("itemtype") or ""):
            self.skip_depth = 1
            if a.get("itemid"):
                self.quoted_ids.append(str(a["itemid"]))
            return
        if tag == "at":
            self.out.append("@")
        elif tag in ("emoji", "customemoji"):
            self.out.append(a.get("alt") or a.get("title") or "")
        elif tag == "img":
            self.out.append("[图片]")
        elif tag == "attachment":
            att = self.attachments.get(a.get("id") or "", {})
            ctype = att.get("contentType", "")
            if ctype == "messageReference":
                # 新版"引用回复"：内容是 JSON，记下被引用的消息 id，正文不保留
                try:
                    ref = json.loads(att.get("content") or "{}")
                    if ref.get("messageId"):
                        self.quoted_ids.append(str(ref["messageId"]))
                except json.JSONDecodeError:
                    pass
            elif ctype == "reference":
                self.out.append(f"[附件: {att.get('name') or '文件'}]")
            elif ctype.startswith("application/vnd.microsoft.card"):
                self.out.append("[卡片]")
            elif ctype:
                self.out.append(f"[附件: {att.get('name') or ctype}]")
        elif tag in ("codeblock", "pre") and not self.in_code:
            self.in_code = True
            self.out.append("\n```\n")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag == "blockquote":
                self.skip_depth -= 1
            return
        if tag in ("codeblock", "pre") and self.in_code:
            self.in_code = False
            self.out.append("\n```\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.out.append(data)


def teams_html_to_text(content: str, content_type: str = "html", attachments: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    """返回 (纯文本, 被引用的消息 id 列表)。"""
    if content_type != "html":
        return html.unescape(content or ""), []
    by_id = {str(a.get("id")): a for a in (attachments or [])}
    p = _TeamsHTMLToText(by_id)
    p.feed(content or "")
    p.close()
    text = "".join(p.out)
    # 聊天消息很少需要段落结构，空行全部去掉
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln), p.quoted_ids
