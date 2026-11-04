"""HTTP API（FastAPI）。启动：``servicekb serve``，交互式文档：http://localhost:8000/docs

鉴权与权限（**上线前必读**）
----------------------------
- ``X-API-Key``：调用方的密钥。两种配置：
  - ``SERVICEKB_API_KEYS``：可信网关用，可以访问任意用户组；
  - ``SERVICEKB_API_CLIENTS``：按调用方分配 Key + **允许的用户组**（最小权限）。例如 Teams 机器人只允许
    ``public, service``，即使它被攻破、在 ``X-User-Groups`` 里自称 ``internal``，也看不到内部资料。
  两者都为空时不鉴权（仅限本地开发，启动时会告警）。
- ``X-User-Groups``：当前用户所属的组，决定能检索到哪些文档。
  **这个头必须由可信的网关 / 后端在完成用户登录（Entra ID / OIDC）后设置，绝不能由浏览器直接传。**
  缺省时只给 ``public``（最小权限）。
- 永远不要让 LLM 决定权限过滤条件（例如"从问题里抽取用户组"）。

生产防护（见 docs/12_production_hardening.md）
----------------------------------------------
- **请求 ID**：每个请求有一个 ID（调用方可以用 ``X-Request-ID`` 传入，否则自动生成），写进所有日志、
  查询日志和响应头。用户报障时让对方提供这个 ID。
- **过载保护**：同时处理的 ask / search / photo 请求数有上限，超过时短暂排队，仍满就返回
  ``503 + Retry-After``——宁可让一部分请求快速失败重试，也不让所有请求一起变慢、一起超时。
- **阻塞调用不进事件循环**：识别照片（OCR / 图像模型）是 CPU / GPU 密集的同步调用，放在线程池里执行；
  如果直接在 ``async def`` 里调用，会卡住整个事件循环，连 /healthz 都没法响应。
- **依赖故障**：数据库不可用时返回 503（可重试），而不是 500。
- **监控**：``GET /metrics``（Prometheus 格式）。不要暴露到公网。
"""

from __future__ import annotations

import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from servicekb import metrics
from servicekb.config import Settings, get_settings
from servicekb.logging import get_logger, request_id_var, setup_logging
from servicekb.models import DocType
from servicekb.retrieval.retriever import RetrievalError
from servicekb.service import ServiceKB

log = get_logger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")  # 调用方传入的 ID 只接受安全字符，防止日志注入
_QUIET_PATHS = {"/healthz", "/readyz", "/metrics"}  # 探针 / 抓取请求太频繁，不写访问日志


# --------------------------------------------------------------------------- 请求 / 响应模型


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000, examples=["AS-110K-HV 早上报 E037 怎么处理？"])
    product_models: list[Annotated[str, Field(max_length=64)]] | None = Field(
        None, max_length=20, description="按机型硬过滤，例如 ['AS-110K-HV']")
    # 用枚举类型：传了不存在的类型返回 422（以前会在服务内部抛 ValueError，变成 500）
    doc_types: list[DocType] | None = Field(None, max_length=len(DocType), description="manual / bulletin / qa / part / chat")


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
    degraded: list[str] = Field(default_factory=list, description="失败但已降级的依赖，例如 rerank:failed")
    usage: dict[str, Any] = Field(default_factory=dict, description="大模型 token 用量")


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


class PartCandidateOut(BaseModel):
    part_number: str
    name: str
    status: str
    score: float
    visual_score: float | None
    ocr_confirmed: bool
    evidence: list[str]


class IdentifyResponse(BaseModel):
    identified: bool
    candidates: list[PartCandidateOut]
    lookalikes: list[str]
    warnings: list[str]
    facts: list[str]
    ocr_text: str
    serial_number: str | None
    manufacture_date: str | None
    timings_ms: dict[str, float]


class PhotoAskResponse(BaseModel):
    identification: IdentifyResponse
    answer: AskResponse


class FeedbackRequest(BaseModel):
    query_id: uuid.UUID  # 非法格式直接 422（以前会让 Postgres 报 uuid 语法错误 → 500）
    rating: Literal[-1, 1] = Field(..., description="1 = 有用，-1 = 没用")
    comment: str | None = Field(None, max_length=2000)
    corrected_answer: str | None = Field(None, max_length=10000, description="专家给出的正确答案（会被沉淀为精选问答）")


# --------------------------------------------------------------------------- 调用方 / 鉴权


@dataclass(frozen=True)
class Caller:
    name: str
    allowed_groups: frozenset[str] | None  # None = 不限制（可信网关 / 本地开发）


def _build_callers(s: Settings) -> list[tuple[bytes, Caller]]:
    callers = [(k.get_secret_value().encode(), Caller(f"key-{i}", None)) for i, k in enumerate(s.api_keys, 1)]
    for c in s.api_clients:
        groups = frozenset(c.groups) if c.groups is not None else None
        callers.append((c.key.get_secret_value().encode(), Caller(c.name, groups)))
    return callers


def _to_ask_response(r: Any) -> AskResponse:
    return AskResponse(
        query_id=r.query_id, answer=r.answer, facts=r.facts, escalate=r.escalate, reason=r.reason,
        model=r.model, timings_ms=r.timings_ms, degraded=r.degraded, usage=r.usage,
        citations=[CitationOut(index=c.index, doc_ref=c.doc_ref, title=c.title, uri=c.uri,
                               section=" > ".join(c.heading_path[1:]), page=c.page, snippet=c.snippet)
                   for c in r.citations],
    )


def _to_identify_response(res: Any) -> IdentifyResponse:
    return IdentifyResponse(
        identified=res.identified, lookalikes=[x.split(":", 1)[1] for x in res.lookalikes],
        warnings=res.warnings, facts=res.facts, ocr_text=res.ocr_text, serial_number=res.serial_number,
        manufacture_date=res.manufacture_date, timings_ms=res.timings_ms,
        candidates=[PartCandidateOut(part_number=c.part_number, name=c.name, status=c.status, score=c.score,
                                     visual_score=c.visual_score, ocr_confirmed=c.ocr_confirmed,
                                     evidence=c.evidence) for c in res.candidates],
    )


# --------------------------------------------------------------------------- 应用


def create_app(settings: Settings | None = None, kb: ServiceKB | None = None) -> FastAPI:
    """``kb``：测试时可以传入已经建好的 ServiceKB（省掉启动时导入数据）。"""
    s = settings or get_settings()
    setup_logging(s.log_level, s.log_json)
    state: dict[str, Any] = {}
    callers = _build_callers(s)
    slots = threading.BoundedSemaphore(s.max_concurrent_requests) if s.max_concurrent_requests > 0 else None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        k = kb or ServiceKB(s)
        if kb is None and s.store_backend == "memory":
            log.warning("memory store: ingesting sample sources at startup (dev only)")
            k.ingest()
        if not callers:
            log.warning("SERVICEKB_API_KEYS / SERVICEKB_API_CLIENTS are empty: API authentication is DISABLED (dev only)")
        state["kb"] = k
        yield
        if kb is None:
            k.close()

    app = FastAPI(title="servicekb", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------------ 中间件：请求 ID + 访问日志 + 指标
    # 纯 ASGI 写法（不用 @app.middleware("http") / BaseHTTPMiddleware）：后者每个请求多建一个任务、
    # 多一层响应包装，压测里能看到明显的额外开销。
    async def observe(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await inner_app(scope, receive, send)
            return
        rid = ""
        for k, v in scope.get("headers", []):
            if k == b"x-request-id":
                rid = v.decode("latin-1")
                break
        if not _REQUEST_ID.match(rid):
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await inner_app(scope, receive, send_wrapper)
        finally:
            dt = time.perf_counter() - t0
            # 用路由模板（/v1/ask）而不是实际路径做标签，避免指标基数爆炸
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope.get("method", "")
            metrics.HTTP_REQUESTS.labels(method, route, str(status)).inc()
            metrics.HTTP_LATENCY.labels(method, route).observe(dt)
            if route not in _QUIET_PATHS:
                log.info("http request", extra={"method": method, "path": route, "status": status,
                                                "duration_ms": round(dt * 1000, 1)})
            request_id_var.reset(token)

    @app.exception_handler(RetrievalError)
    async def retrieval_unavailable(_: Request, exc: RetrievalError) -> JSONResponse:
        log.error("retrieval unavailable", extra={"error": str(exc)[:300]})
        return JSONResponse({"detail": "knowledge base temporarily unavailable"}, status_code=503,
                            headers={"Retry-After": "5"})

    # ------------------------------------------------------------------ 依赖
    def get_kb() -> ServiceKB:
        return state["kb"]

    def auth(x_api_key: str | None = Header(default=None)) -> Caller:
        if not callers:
            return Caller("anonymous", None)
        given = (x_api_key or "").encode()
        match: Caller | None = None
        for key, caller in callers:  # 逐个比较、不提前退出：耗时与"匹配的是第几个 Key"无关
            if given and secrets.compare_digest(given, key):
                match = caller
        if match is None:
            raise HTTPException(status_code=401, detail="invalid api key")
        return match

    def user_groups(caller: Caller = Depends(auth), x_user_groups: str | None = Header(default=None)) -> list[str]:
        requested = list(dict.fromkeys(g.strip() for g in (x_user_groups or "").split(",") if g.strip()))[:20]
        if caller.allowed_groups is not None:
            denied = [g for g in requested if g not in caller.allowed_groups]
            if denied:  # 可能是配置错误，也可能是越权尝试：记下来
                log.warning("user groups not allowed for this api client, dropped",
                            extra={"client": caller.name, "denied": denied})
            requested = [g for g in requested if g in caller.allowed_groups]
        return requested or ["public"]

    def guarded(route: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """在工作线程里执行重请求：先拿并发名额（拿不到 → 503），再调用。"""
        if slots is not None and not slots.acquire(timeout=s.overload_wait_s):
            metrics.OVERLOAD_REJECTED.labels(route).inc()
            raise HTTPException(status_code=503, detail="server busy, please retry",
                                headers={"Retry-After": str(max(1, round(s.overload_wait_s)))})
        metrics.INFLIGHT.inc()
        try:
            return fn(*args, **kwargs)
        finally:
            metrics.INFLIGHT.dec()
            if slots is not None:
                slots.release()

    # ------------------------------------------------------------------ 探针 / 监控
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """存活探针：进程活着就返回 ok。"""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(k: ServiceKB = Depends(get_kb)) -> dict[str, Any]:
        """就绪探针：数据库可用才返回 ok（K8s / 负载均衡用它决定是否转发流量）。

        大模型熔断**不影响**就绪：熔断期间仍能给出抽取式回答。状态放在 ``llm`` 字段里供排查。
        """
        try:
            st = k.store.stats()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"store unavailable: {e!r}") from e
        return {"status": "ok", "chunks": st.get("chunks"), "embedding_model": st.get("embedding_model"),
                "llm": k.llm_status()}

    if s.metrics_enabled:
        @app.get("/metrics", include_in_schema=False)
        def metrics_endpoint() -> Response:
            body, content_type = metrics.render()
            return Response(body, media_type=content_type)

    # ------------------------------------------------------------------ 业务接口
    @app.post("/v1/ask", response_model=AskResponse)
    def ask(req: AskRequest, caller: Caller = Depends(auth), groups: list[str] = Depends(user_groups),
            k: ServiceKB = Depends(get_kb)) -> AskResponse:
        r = guarded("/v1/ask", k.ask, req.question, groups, [d.value for d in req.doc_types or []] or None,
                    req.product_models, client=caller.name)
        return _to_ask_response(r)

    @app.post("/v1/search", response_model=list[SearchHit])
    def search(req: AskRequest, groups: list[str] = Depends(user_groups),
               k: ServiceKB = Depends(get_kb)) -> list[SearchHit]:
        r = guarded("/v1/search", k.search, req.question, groups, [d.value for d in req.doc_types or []] or None,
                    req.product_models, top_k=10)
        return [
            SearchHit(rank=i, score=sc.score, rerank_score=sc.rerank_score, channel_ranks=sc.channel_ranks,
                      doc_ref=sc.chunk.doc_ref, title=sc.chunk.title, section=" > ".join(sc.chunk.heading_path[1:]),
                      text=sc.chunk.text, duplicates=sc.duplicates)
            for i, sc in enumerate(r.results, 1)
        ]

    max_upload = 15 * 1024 * 1024  # 15 MB：手机原图一般 3~8 MB

    async def read_upload(image: UploadFile) -> bytes:
        data = await image.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(status_code=413, detail="image too large (max 15 MB)")
        if not data:
            raise HTTPException(status_code=422, detail="empty file")
        return data

    @app.post("/v1/photo/identify", response_model=IdentifyResponse)
    async def photo_identify(image: UploadFile = File(...), groups: list[str] = Depends(user_groups),
                             k: ServiceKB = Depends(get_kb)) -> IdentifyResponse:
        """上传现场照片，返回候选料号、证据和"长得像"提醒。"""
        data = await read_upload(image)
        try:  # OCR / 图像模型是阻塞调用：放进线程池，不卡事件循环
            res = await run_in_threadpool(guarded, "/v1/photo/identify", k.identify_photo, data, groups)
        except OSError as e:  # PIL 打不开：不是图片 / 文件损坏
            raise HTTPException(status_code=422, detail=f"cannot read image: {e}") from e
        return _to_identify_response(res)

    @app.post("/v1/photo/ask", response_model=PhotoAskResponse)
    async def photo_ask(image: UploadFile = File(...), question: str = Form("", max_length=2000),
                        caller: Caller = Depends(auth), groups: list[str] = Depends(user_groups),
                        k: ServiceKB = Depends(get_kb)) -> PhotoAskResponse:
        """上传照片 + 问题：先识别零件，再用知识库回答。"""
        data = await read_upload(image)
        try:
            ident, r = await run_in_threadpool(guarded, "/v1/photo/ask", k.ask_with_photo, data, question or None,
                                               groups, client=caller.name)
        except OSError as e:
            raise HTTPException(status_code=422, detail=f"cannot read image: {e}") from e
        return PhotoAskResponse(identification=_to_identify_response(ident), answer=_to_ask_response(r))

    @app.post("/v1/feedback", dependencies=[Depends(auth)])
    def feedback(req: FeedbackRequest, k: ServiceKB = Depends(get_kb)) -> dict[str, bool]:
        if not k.feedback(str(req.query_id), req.rating, req.comment, req.corrected_answer):
            raise HTTPException(status_code=404, detail="unknown query_id")
        return {"ok": True}

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats(k: ServiceKB = Depends(get_kb)) -> dict[str, Any]:
        return k.store.stats()

    # 把中间件装在最外层：FastAPI 的异常处理、路由都在它里面，所以能拿到最终状态码和匹配到的路由
    inner_app = app.build_middleware_stack()
    app.middleware_stack = observe
    return app
