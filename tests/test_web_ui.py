"""网页问答（/ui）的服务端测试：路由、安全响应头、页面资源、配置接口。

页面交互（提问、引用跳转、拍照、反馈、鉴权弹窗、手机布局、XSS）用真实浏览器做端到端检查，
见 docs/12 / README 里的说明；这里锁住服务端的约定，保证页面能被正确地提供出来。
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from servicekb.api import WEB_DIR, create_app
from tests.conftest import memory_settings


@pytest.fixture()
def client(kb):
    with TestClient(create_app(memory_settings(), kb=kb)) as c:
        yield c


def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "ui/"  # 相对地址：网关子路径下也能跳对
    assert client.get("/ui", follow_redirects=False).headers["location"].endswith("/ui/")


def test_page_and_assets_are_served_with_security_headers(client):
    page = client.get("/ui/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert page.headers["x-content-type-options"] == "nosniff" and page.headers["x-frame-options"] == "DENY"
    assert "x-request-id" in page.headers
    for asset, ctype in (("app.js", "javascript"), ("style.css", "text/css")):
        r = client.get(f"/ui/{asset}")
        assert r.status_code == 200 and ctype in r.headers["content-type"]
    assert client.get("/ui/missing.js").status_code == 404
    # API 响应不需要页面的 CSP（不影响其他调用方）
    assert "content-security-policy" not in client.get("/healthz").headers


def test_page_only_loads_local_assets():
    html = (WEB_DIR / "index.html").read_text("utf-8")
    srcs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert srcs and all(s.startswith("./") or s.startswith("data:") for s in srcs), srcs  # 不引用任何外部资源


def test_page_script_never_uses_innerhtml():
    """服务端返回的文字（证据、答案）必须当纯文本插入，防止 XSS。"""
    js = (WEB_DIR / "app.js").read_text("utf-8")
    assert ".innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js
    assert "eval(" not in js and "new Function" not in js


def test_ui_config_reports_auth(kb):
    with TestClient(create_app(memory_settings(), kb=kb)) as c:
        assert c.get("/ui/config").json()["auth_required"] is False
    with TestClient(create_app(memory_settings(api_keys=["k"]), kb=kb)) as c:
        body = c.get("/ui/config").json()  # 配置接口不需要 Key（页面要靠它决定是否提示填写）
        assert body["auth_required"] is True and body["max_upload_mb"] == 15
        assert set(body) == {"auth_required", "version", "max_upload_mb"}  # 不泄露其他信息


def test_ui_can_be_disabled(kb):
    with TestClient(create_app(memory_settings(ui_enabled=False), kb=kb)) as c:
        assert c.get("/ui/").status_code == 404 and c.get("/").status_code == 404
        assert c.get("/healthz").status_code == 200


def test_refusal_does_not_carry_facts_from_irrelevant_evidence(kb):
    """拒答时，不能把"不相关证据"里摘出来的料号事实展示给用户（页面上会显得莫名其妙）。"""
    r = kb.ask("公司年假政策是什么？", ["public", "service"], log_query=False)
    assert r.escalate and r.facts == []
    r = kb.ask("310-04521-A 还能订到吗？", ["public", "service"], log_query=False)
    assert any("310-04521-B" in f for f in r.facts)  # 问题里提到的料号照常给出事实
