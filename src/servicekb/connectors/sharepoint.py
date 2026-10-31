"""SharePoint / OneDrive 文档库（通过 Microsoft Graph API 增量同步）。

关键概念：**delta query（增量查询）**
    第一次调用 ``GET /drives/{drive-id}/root/delta`` 会返回全部文件；响应最后一页带一个
    ``@odata.deltaLink``。把它存下来，下次直接请求这个链接，Graph 只返回"自上次以来变化的文件"，
    其中被删除的文件带 ``deleted`` 字段。这样每天同步一次也只需要几秒钟。
    文档：https://learn.microsoft.com/graph/api/driveitem-delta

鉴权：Azure AD（Entra ID）应用注册 + client credentials（应用权限 ``Sites.Read.All`` 或更细的
``Sites.Selected``）。**生产环境优先用 Sites.Selected**，只授权知识库所在站点，最小权限原则。

限流：Graph 返回 429 / 503 时带 ``Retry-After`` 头，必须遵守，否则会被封禁更久。

尚未实现（见 docs/08_limitations_roadmap.md）：
    - 把 SharePoint 的文件权限（``/permissions``）映射成 acl_groups；目前按数据源统一配置 acl。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

import httpx

from servicekb.connectors.base import Connector
from servicekb.logging import get_logger
from servicekb.models import DocType, SourceEvent, SourceRecord, Tombstone
from servicekb.parsing import EXTENSION_MIME

log = get_logger(__name__)

GRAPH = "https://graph.microsoft.com/v1.0"


class GraphClient:
    """极简 Graph 客户端：拿 token、GET（带限流重试）、下载文件。"""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        http: httpx.Client | None = None,
        max_retries: int = 5,
    ) -> None:
        self.tenant_id, self.client_id, self.client_secret = tenant_id, client_id, client_secret
        self.http = http or httpx.Client(timeout=60)
        self.max_retries = max_retries
        self._token: str | None = None
        self._token_exp = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        r = self.http.post(
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        self._token_exp = time.time() + int(body.get("expires_in", 3600))
        return self._token

    def get(self, url: str, *, raw: bool = False) -> Any:
        for attempt in range(self.max_retries + 1):
            r = self.http.get(url, headers={"Authorization": f"Bearer {self._get_token()}"}, follow_redirects=True)
            if r.status_code in (429, 503, 504) and attempt < self.max_retries:
                wait = float(r.headers.get("Retry-After", 2**attempt))
                log.warning("graph throttled, retrying", extra={"status": r.status_code, "wait_s": wait})
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content if raw else r.json()
        raise RuntimeError("unreachable")


class SharePointConnector(Connector):
    full_snapshot = False  # delta 是增量：删除只认 Graph 显式返回的 deleted

    def __init__(
        self,
        source_id: str,
        drive_id: str,
        tenant_id: str = "",
        client_id: str = "",
        client_secret_env: str = "SERVICEKB_GRAPH_CLIENT_SECRET",
        folder: str = "",
        doc_type: str = "manual",
        acl_groups: list[str] | None = None,
        client: GraphClient | None = None,
    ) -> None:
        self.source_id = source_id
        self.drive_id = drive_id
        self.folder = folder.strip("/")
        self.doc_type = DocType(doc_type)
        self.acl_groups = acl_groups or ["service"]
        # 密钥只从环境变量读，绝不写进 sources.yaml
        self.client = client or GraphClient(
            tenant_id or os.environ.get("SERVICEKB_GRAPH_TENANT_ID", ""),
            client_id or os.environ.get("SERVICEKB_GRAPH_CLIENT_ID", ""),
            os.environ.get(client_secret_env, ""),
        )
        self._delta_link: str | None = None

    def _in_scope(self, item: dict[str, Any]) -> bool:
        if not self.folder:
            return True
        parent = item.get("parentReference", {}).get("path", "")  # 形如 /drives/{id}/root:/KB/Manuals
        return f"root:/{self.folder}" in parent

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        url = (state or {}).get("delta_link") or f"{GRAPH}/drives/{self.drive_id}/root/delta"
        while url:
            page = self.client.get(url)
            for item in page.get("value", []):
                if "deleted" in item:
                    yield Tombstone(self.source_id, item["id"])
                    continue
                if "file" not in item or not self._in_scope(item):
                    continue
                name = item.get("name", "")
                mime = EXTENSION_MIME.get(PurePosixPath(name).suffix.lower())
                if mime is None:
                    continue
                content = self.client.get(f"{GRAPH}/drives/{self.drive_id}/items/{item['id']}/content", raw=True)
                if mime.startswith("text/"):
                    content = content.decode("utf-8", errors="replace")
                modified = item.get("lastModifiedDateTime")
                yield SourceRecord(
                    source_id=self.source_id,
                    external_id=item["id"],  # 用 item id 而不是路径：文件改名 / 移动后 ID 不变
                    title=PurePosixPath(name).stem,
                    doc_type=self.doc_type,
                    content=content,
                    mime_type=mime,
                    uri=item.get("webUrl", ""),
                    metadata={"sharepoint_path": item.get("parentReference", {}).get("path", ""), "etag": item.get("eTag")},
                    acl_groups=list(self.acl_groups),
                    updated_at=datetime.fromisoformat(modified.replace("Z", "+00:00")) if modified else None,
                )
            if "@odata.deltaLink" in page:
                self._delta_link = page["@odata.deltaLink"]
            url = page.get("@odata.nextLink")

    def next_state(self) -> dict[str, Any] | None:
        return {"delta_link": self._delta_link} if self._delta_link else None
