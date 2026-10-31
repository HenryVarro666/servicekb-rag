"""Teams 数据获取：从 Microsoft Graph 导出原始消息 / 读取导出文件。

**原则：先原样落地（raw），再清洗。** 原始 JSON 存一份（对象存储 / 受控目录，带保留期限），
清洗规则改了可以随时重跑，不用重新调用 Graph（Graph 有限流，而且 Teams 消息可能已被删除）。

Graph 端点（v1.0，2026-10 核对 microsoftgraph/microsoft-graph-docs-contrib）
-----------------------------------------------------------------------
- 频道根消息：``GET /teams/{team-id}/channels/{channel-id}/messages``（$top 最大 50，``$expand=replies``）
- 某条消息的回复：``GET /teams/{team-id}/channels/{channel-id}/messages/{message-id}/replies``
- 批量导出（仅应用权限）：``GET /teams/{team-id}/channels/getAllMessages``——回复是"平铺"返回的，带 replyToId
- 群聊 / 私聊：``GET /chats/{chat-id}/messages``；``/users/{id}/chats/getAllMessages``（覆盖该用户所有聊天，
  权限很大，**不建议**用于知识库）

权限（最小权限优先）
    推荐资源级授权（RSC）``ChannelMessage.Read.Group``：由每个 Team 的 owner 单独授权，只能读该 Team；
    而不是租户级 ``ChannelMessage.Read.All``（能读全公司所有频道）。

限流
    GET 频道 / 聊天消息：每应用每租户 20 rps，**每个频道 / 聊天 1 rps**；返回 429 时遵守 Retry-After。
    微软建议同一资源轮询不超过一天一次，更高频率请用 change notifications（webhook）。

完整说明见 docs/09_chat_log_cleaning.md。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from servicekb.connectors.sharepoint import GRAPH, GraphClient


def iter_raw_messages(path: Path) -> Iterator[dict[str, Any]]:
    """读取导出文件。支持三种格式：
    - JSONL：每行一个 chatMessage（推荐，本模块的导出器就输出这种）
    - JSON 数组：``[{...}, {...}]``
    - Graph 原始分页：``{"value": [...]}``（可以直接保存 Graph Explorer 的响应）
    """
    files = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    for f in files:
        text = f.read_text("utf-8").strip()
        if not text:
            continue
        if f.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
            continue
        data = json.loads(text)
        if isinstance(data, dict):
            yield from data.get("value", [])
        else:
            yield from data


class TeamsChannelExporter:
    """把指定频道的根消息 + 回复导出为 JSONL（原样保存 Graph 返回的 chatMessage）。

    用法::

        client = GraphClient(tenant_id, client_id, client_secret)
        TeamsChannelExporter(client).export(team_id, channel_id, Path("raw/teams_support.jsonl"))

    注意：要拿到 ``systemEventMessage`` 类型，需要请求头 ``Prefer: include-unknown-enum-members``；
    我们本来就要过滤掉系统消息，所以不加这个头也没关系。
    """

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    def _paged(self, url: str) -> Iterator[dict[str, Any]]:
        while url:
            page = self.client.get(url)
            yield from page.get("value", [])
            url = page.get("@odata.nextLink")

    def export(self, team_id: str, channel_id: str, out: Path) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        base = f"{GRAPH}/teams/{team_id}/channels/{channel_id}/messages"
        with out.open("w", encoding="utf-8") as f:
            for root in self._paged(f"{base}?$top=50"):
                f.write(json.dumps(root, ensure_ascii=False) + "\n")
                n += 1
                for reply in self._paged(f"{base}/{root['id']}/replies?$top=50"):
                    f.write(json.dumps(reply, ensure_ascii=False) + "\n")
                    n += 1
        return n
