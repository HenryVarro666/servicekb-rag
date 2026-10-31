"""HTTP API（FastAPI）。启动：``servicekb serve``，交互式文档：http://localhost:8000/docs

鉴权与权限（**上线前必读**）
----------------------------
- ``X-API-Key``：服务间调用的密钥，配置在 SERVICEKB_API_KEYS。为空时不鉴权（仅限本地开发，启动时会告警）。
- ``X-User-Groups``：当前用户所属的组，决定能检索到哪些文档。
  **这个头必须由可信的网关 / 后端在完成用户登录（Entra ID / OIDC）后设置，绝不能由浏览器直接传。**
  否则任何人都能自称 "service" 组看到内部资料。缺省时只给 ``public``（最小权限）。
- 永远不要让 LLM 决定权限过滤条件（例如"从问题里抽取用户组"）。
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from servicekb.config import Settings, get_settings
from servicekb.logging import get_logger, setup_logging
from servicekb.service import ServiceKB

log = get_logger(__name__)


# --------------------------------------------------------------------------- 请求 / 响应模型


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000, examples=["AS-110K-HV 早上报 E037 怎么处理？"])
    product_models: list[str] | None = Field(None, description="按机型硬过滤，例如 ['AS-110K-HV']")
    doc_types: list[str] | None = Field(None, description="manual / bulletin / qa / part / chat")


class CitationOut(BaseModel):
    index: int
    doc_ref: str
    title: str
    uri: str
    section: str
    page: int | None
    snippet: str


class AskResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[CitationOut]
    facts: list[str]
    escalate: bool
    reason: str
    model: str
    timings_ms: dict[str, float]


class SearchHit(BaseModel):
    rank: int
    score: float
    rerank_score: float | None
    channel_ranks: dict[str, int]
    doc_ref: str
    title: str
    section: str
    text: str
    duplicates: list[str]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., description="1 = 有用，-1 = 没用")
    comment: str | None = None
    corrected_answer: str | None = Field(None, description="专家给出的正确答案（会被沉淀为精选问答）")


# --------------------------------------------------------------------------- 应用


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level, s.log_json)
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        kb = ServiceKB(s)
        if s.store_backend == "memory":
            log.warning("memory store: ingesting sample sources at startup (dev only)")
            kb.ingest()
        if not s.api_keys:
            log.warning("SERVICEKB_API_KEYS is empty: API authentication is DISABLED (dev only)")
        state["kb"] = kb
        yield
        kb.close()

    app = FastAPI(title="servicekb", version="0.1.0", lifespan=lifespan)

    def kb() -> ServiceKB:
        return state["kb"]

    def auth(x_api_key: str | None = Header(default=None)) -> None:
        if not s.api_keys:
            return
        if not x_api_key or not any(secrets.compare_digest(x_api_key, k.get_secret_value()) for k in s.api_keys):
            raise HTTPException(status_code=401, detail="invalid api key")

    def user_groups(x_user_groups: str | None = Header(default=None)) -> list[str]:
        groups = [g.strip() for g in (x_user_groups or "").split(",") if g.strip()]
        return groups or ["public"]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """存活探针：进程活着就返回 ok。"""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(k: ServiceKB = Depends(kb)) -> dict[str, Any]:
        """就绪探针：数据库可用才返回 ok（K8s / 负载均衡用它决定是否转发流量）。"""
        try:
            st = k.store.stats()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"store unavailable: {e!r}") from e
        return {"status": "ok", "chunks": st.get("chunks"), "embedding_model": st.get("embedding_model")}

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(auth)])
    def ask(req: AskRequest, groups: list[str] = Depends(user_groups), k: ServiceKB = Depends(kb)) -> AskResponse:
        r = k.ask(req.question, groups, req.doc_types, req.product_models)
        return AskResponse(
            query_id=r.query_id, answer=r.answer, facts=r.facts, escalate=r.escalate, reason=r.reason,
            model=r.model, timings_ms=r.timings_ms,
            citations=[CitationOut(index=c.index, doc_ref=c.doc_ref, title=c.title, uri=c.uri,
                                   section=" > ".join(c.heading_path[1:]), page=c.page, snippet=c.snippet)
                       for c in r.citations],
        )

    @app.post("/v1/search", response_model=list[SearchHit], dependencies=[Depends(auth)])
    def search(req: AskRequest, groups: list[str] = Depends(user_groups), k: ServiceKB = Depends(kb)) -> list[SearchHit]:
        r = k.search(req.question, groups, req.doc_types, req.product_models, top_k=10)
        return [
            SearchHit(rank=i, score=sc.score, rerank_score=sc.rerank_score, channel_ranks=sc.channel_ranks,
                      doc_ref=sc.chunk.doc_ref, title=sc.chunk.title, section=" > ".join(sc.chunk.heading_path[1:]),
                      text=sc.chunk.text, duplicates=sc.duplicates)
            for i, sc in enumerate(r.results, 1)
        ]

    @app.post("/v1/feedback", dependencies=[Depends(auth)])
    def feedback(req: FeedbackRequest, k: ServiceKB = Depends(kb)) -> dict[str, bool]:
        if req.rating not in (-1, 1):
            raise HTTPException(status_code=422, detail="rating must be 1 or -1")
        if not k.feedback(req.query_id, req.rating, req.comment, req.corrected_answer):
            raise HTTPException(status_code=404, detail="unknown query_id")
        return {"ok": True}

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats(k: ServiceKB = Depends(kb)) -> dict[str, Any]:
        return k.store.stats()

    return app
