"""上线前自检：``servicekb doctor``。

部署出问题，大多不是代码 bug，而是**配置和依赖**：数据库连不上、向量维度和建库时不一致、
TEI 地址写错、API Key 没配、连接池比并发数还小……这些问题在第一个真实用户请求进来时才暴露，就太晚了。

``doctor`` 把每个依赖实际调用一次，并检查常见的危险配置：

- ✓ ok    ：正常
- ⚠ warn  ：能跑，但不适合生产（例如没开鉴权、用的是非语义的哈希向量）
- ✗ fail  ：会导致功能不可用（退出码 1，可以放进部署流水线 / K8s initContainer）

默认不调用大模型（要花钱）；加 ``--llm`` 才会发一条测试请求。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from servicekb.config import Settings


@dataclass
class Check:
    name: str
    status: str  # ok / warn / fail
    detail: str


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def config_checks(s: Settings) -> list[Check]:
    """只看配置、不连接任何服务的检查。"""
    out: list[Check] = []

    def warn(name: str, detail: str) -> None:
        out.append(Check(name, "warn", detail))

    if not s.api_keys and not s.api_clients:
        warn("config.auth", "API 未开启鉴权（SERVICEKB_API_KEYS / SERVICEKB_API_CLIENTS 为空），仅限本地开发")
    if s.store_backend == "memory":
        warn("config.store", "memory 存储：重启即丢数据，仅用于 demo / 测试")
    if s.embedding_backend == "hashing":
        warn("config.embedding", "哈希向量没有语义理解能力（不懂同义词、不跨语言），生产请换 bge-m3")
    if s.reranker_backend == "none":
        warn("config.reranker", "没有重排模型：排序精度低，拒答只能靠启发式规则")
    if s.llm_provider == "extractive":
        warn("config.llm", "未配置大模型（extractive）：答案是证据原文摘录")
    if not s.log_json:
        warn("config.logging", "日志不是 JSON 格式（SERVICEKB_LOG_JSON=true），日志平台不好检索")
    if not s.query_log_redact_pii:
        warn("config.privacy", "查询日志不脱敏（SERVICEKB_QUERY_LOG_REDACT_PII=false）")
    n = s.max_concurrent_requests
    if s.store_backend == "postgres" and n and s.db_pool_max_size < n:
        warn("config.db_pool", f"数据库连接池（{s.db_pool_max_size}）小于并发上限（{n}）：高峰时请求会排队等连接")
    if n and s.retrieval_max_workers < 3 * n:
        warn("config.retrieval_pool", f"召回线程池（{s.retrieval_max_workers}）< 3 × 并发上限（{3 * n}）：请求之间会互相排队")
    if not out:
        out.append(Check("config", "ok", "未发现危险配置"))
    return out


def run_checks(s: Settings, call_llm: bool = False, vision: bool = False) -> list[Check]:
    from servicekb.embedding import build_embedder
    from servicekb.generation.llm import LLMError, build_chat_model
    from servicekb.retrieval.rerank import build_reranker
    from servicekb.store import build_store
    from servicekb.vision.ocr import MetadataOCR, build_ocr

    checks = config_checks(s)

    # ---- 向量模型
    embedder = None
    try:
        embedder = build_embedder(s)
        vec, ms = _timed(lambda: embedder.embed_query("连通性测试 E037"))
        if len(vec) != s.embedding_dim:
            checks.append(Check("embedding", "fail", f"模型输出 {len(vec)} 维，但 SERVICEKB_EMBEDDING_DIM={s.embedding_dim}"))
        else:
            checks.append(Check("embedding", "ok", f"{embedder.model_id}，{ms:.0f} ms"))
    except Exception as e:  # noqa: BLE001  自检要把所有问题都列出来，不能遇到一个就停
        checks.append(Check("embedding", "fail", f"{type(e).__name__}: {e}"[:300]))

    # ---- 存储（Postgres：连接 + 表 + 建库时的向量模型是否一致）
    try:
        store = build_store(s)
        try:
            st, ms = _timed(store.stats)
            built_with = st.get("embedding_model")
            if s.store_backend == "postgres" and embedder and built_with and built_with != embedder.model_id:
                checks.append(Check("store", "fail", f"索引用 {built_with} 建的，当前配置是 {embedder.model_id}：需要重建索引"))
            elif s.store_backend == "postgres" and not st.get("chunks"):
                checks.append(Check("store", "warn", "数据库连通，但还没有数据：运行 servicekb ingest"))
            else:
                checks.append(Check("store", "ok", f"{s.store_backend}，{st.get('chunks', 0)} 个块，{ms:.0f} ms"))
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001
        hint = "（表不存在？先运行 servicekb db init）" if "does not exist" in str(e) else ""
        checks.append(Check("store", "fail", f"{type(e).__name__}: {e}{hint}"[:300]))

    # ---- 重排模型
    if s.reranker_backend != "none":
        try:
            rr = build_reranker(s)
            scores, ms = _timed(lambda: rr.score("E037 怎么处理", ["E037 绝缘阻抗低：检查直流侧绝缘", "无关内容"]))
            checks.append(Check("reranker", "ok", f"{rr.name}，{ms:.0f} ms，分数 {[round(x, 3) for x in scores]}"))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("reranker", "fail", f"{type(e).__name__}: {e}（请求时会降级为融合排序）"[:300]))

    # ---- 大模型
    try:
        m = build_chat_model(s)
        if m is None:
            pass  # config_checks 已经提示过 extractive
        elif not call_llm:
            checks.append(Check("llm", "ok", f"{m.name}：配置有效（未实际调用，加 --llm 测试连通性）"))
        else:
            out, ms = _timed(lambda: m.complete("你是连通性测试助手。", "请只回复：收到"))
            checks.append(Check("llm", "ok", f"{m.name}，{ms:.0f} ms，回复：{out.strip()[:40]}"))
    except (ValueError, ImportError, LLMError) as e:
        checks.append(Check("llm", "fail", str(e)[:300]))

    # ---- OCR / 图像模型（拍照识别）
    try:
        ocr = build_ocr(s.ocr_backend)
        if isinstance(ocr, MetadataOCR) and s.ocr_backend != "metadata":
            checks.append(Check("ocr", "warn", 'RapidOCR 未安装：真实照片读不出标签（pip install -e ".[ocr]"）'))
        else:
            checks.append(Check("ocr", "ok", type(ocr).__name__))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("ocr", "fail", f"{type(e).__name__}: {e}"[:300]))
    if vision:
        try:
            from PIL import Image

            from servicekb.vision.embedders import build_image_embedder

            ie = build_image_embedder(s)
            vecs, ms = _timed(lambda: ie.embed_images([Image.new("RGB", (64, 64), "white")]))
            checks.append(Check("image_embedding", "ok", f"{ie.model_id}，{len(vecs[0])} 维，{ms:.0f} ms"))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("image_embedding", "fail", f"{type(e).__name__}: {e}"[:300]))
    return checks
