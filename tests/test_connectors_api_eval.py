"""SharePoint（mock Graph）、HTTP API、评估指标。"""

import httpx
import pytest

from servicekb.connectors.sharepoint import GRAPH, GraphClient, SharePointConnector
from servicekb.evaluation import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from servicekb.models import SourceRecord, Tombstone


def _graph_transport():
    calls = {"delta": 0, "throttled": False}

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "oauth2/v2.0/token" in url:
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        if url.endswith("/root/delta"):
            calls["delta"] += 1
            if not calls["throttled"]:
                calls["throttled"] = True
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={
                "value": [
                    {"id": "1", "name": "manual.md", "file": {}, "webUrl": "https://sp/manual.md",
                     "lastModifiedDateTime": "2026-01-01T00:00:00Z", "parentReference": {"path": "/drives/d/root:/KB"}},
                    {"id": "f", "name": "KB", "folder": {}},
                ],
                "@odata.nextLink": f"{GRAPH}/drives/d/root/delta?token=p2",
            })
        if "token=p2" in url:
            return httpx.Response(200, json={
                "value": [{"id": "2", "deleted": {"state": "deleted"}}, {"id": "3", "name": "x.zip", "file": {}}],
                "@odata.deltaLink": f"{GRAPH}/drives/d/root/delta?token=next",
            })
        if url.endswith("/items/1/content"):
            return httpx.Response(200, content="# 手册\n内容".encode())
        return httpx.Response(404)

    return httpx.MockTransport(handler), calls


def test_sharepoint_delta_paging_tombstones_and_throttling():
    transport, calls = _graph_transport()
    client = GraphClient("tenant", "cid", "secret", http=httpx.Client(transport=transport))
    conn = SharePointConnector("sp", drive_id="d", client=client)
    events = list(conn.iter_events(None))
    assert isinstance(events[0], SourceRecord) and events[0].content.startswith("# 手册")
    assert events[0].uri == "https://sp/manual.md"
    assert isinstance(events[1], Tombstone) and events[1].external_id == "2"
    assert len(events) == 2  # 文件夹和不支持的 .zip 被跳过
    assert conn.next_state() == {"delta_link": f"{GRAPH}/drives/d/root/delta?token=next"}
    assert calls["delta"] == 2  # 429 之后重试成功
    assert conn.full_snapshot is False


def test_retrieval_metrics():
    ret, rel = ["a", "x", "b"], {"a", "b", "c"}
    assert hit_at_k(ret, rel, 1) == 1.0
    assert recall_at_k(ret, rel, 3) == pytest.approx(2 / 3)
    assert mrr_at_k(["x", "a"], rel, 5) == 0.5
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rel, 2) == 0.0


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from servicekb.api import create_app
    from tests.conftest import memory_settings

    app = create_app(memory_settings(api_keys=["secret-key"]))
    with TestClient(app) as c:
        yield c


def test_api_auth_and_groups(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    body = {"question": "E051 风扇故障换哪个料号"}
    assert client.post("/v1/ask", json=body).status_code == 401
    h = {"X-API-Key": "secret-key"}
    public = client.post("/v1/search", json=body, headers=h).json()
    assert not any(x["doc_ref"].startswith("bulletins:") for x in public)  # 缺省只有 public 权限
    service = client.post("/v1/search", json=body, headers={**h, "X-User-Groups": "service"}).json()
    assert any(x["doc_ref"].startswith("bulletins:") for x in service)


def test_api_ask_and_feedback(client):
    h = {"X-API-Key": "secret-key", "X-User-Groups": "public,service"}
    r = client.post("/v1/ask", json={"question": "310-04521-A 还能订到吗？"}, headers=h).json()
    assert r["citations"] and not r["escalate"]
    assert client.post("/v1/feedback", json={"query_id": r["query_id"], "rating": 1}, headers=h).json() == {"ok": True}
    assert client.post("/v1/feedback", json={"query_id": r["query_id"], "rating": 5}, headers=h).status_code == 422
