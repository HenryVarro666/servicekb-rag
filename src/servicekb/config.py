"""全局配置。

设计原则
--------
1. **所有可调参数都集中在这里**，通过环境变量（前缀 ``SERVICEKB_``）或 ``.env`` 文件覆盖。
   这样开发 / 测试 / 生产只需要换一份 ``.env``，代码不用改。
2. 默认值选择"**离线可跑通**"的组合：内存存储 + 哈希向量 + 抽取式回答。
   新人 clone 下来，不需要 GPU、不需要 API Key、不需要数据库，就能跑完整条链路。
3. 生产环境推荐组合见 ``.env.example`` 和 ``docs/06_deployment.md``。

用法::

    from servicekb.config import get_settings
    s = get_settings()
    print(s.embedding_backend)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SERVICEKB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ 存储
    # memory  : 进程内存储，重启即丢。只用于 demo / 单元测试。
    # postgres: PostgreSQL + pgvector，生产用。
    store_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://servicekb:servicekb@localhost:5432/servicekb"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    # HNSW 查询时的候选队列长度。越大召回越高、越慢。pgvector 默认 40。
    hnsw_ef_search: int = 100

    # ------------------------------------------------------------------ 向量模型（Embedding）
    # hashing              : 字符 n-gram 哈希，零依赖、非语义，只用于 demo / CI。
    # sentence_transformers: 本地 GPU 跑开源模型（推荐 BAAI/bge-m3，中英混合效果好）。
    # openai               : OpenAI 或任何 OpenAI-compatible 的 /v1/embeddings 服务（如 TEI、vLLM）。
    embedding_backend: Literal["hashing", "sentence_transformers", "openai"] = "hashing"
    embedding_model: str = "BAAI/bge-m3"
    # 向量维度必须和模型一致：bge-m3=1024，text-embedding-3-small=1536，hashing 任意。
    # 注意：数据库表的 vector(维度) 在建表时就固定了，换模型 = 重建索引。
    embedding_dim: int = 512
    embedding_batch_size: int = 32
    embedding_device: str | None = None  # 例如 "cuda:0"；None 表示自动选择
    embedding_base_url: str | None = None  # openai 后端：None=官方；TEI 例: http://gpu-box:8080/v1
    embedding_api_key: SecretStr | None = None
    # 有些模型（e5 / bge-v1.5 英文版）要求给 query 加指令前缀；bge-m3 不需要，留空即可。
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    # ------------------------------------------------------------------ 重排模型（Reranker）
    # none         : 不重排，直接用融合分数。
    # cross_encoder: 本地 GPU 跑 cross-encoder（推荐 BAAI/bge-reranker-v2-m3）。
    # tei          : 调用 HuggingFace text-embeddings-inference 的 /rerank 接口。
    reranker_backend: Literal["none", "cross_encoder", "tei"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_url: str | None = None  # tei 后端用，例如 http://gpu-box:8081
    reranker_candidates: int = 40  # 送进 reranker 的候选数（越多越准、越慢）
    # 重排分数（sigmoid 后 0~1）低于此值的证据视为"不相关"。需要在你们自己的评估集上校准！
    reranker_min_score: float = 0.15

    # ------------------------------------------------------------------ 大模型（LLM）
    # extractive: 不调用任何大模型，从证据里抽句子拼答案。离线 demo 用。
    # openai    : OpenAI 或任何 OpenAI-compatible 的 /v1/chat/completions（vLLM、Ollama、Azure 代理等）。
    llm_backend: Literal["extractive", "openai"] = "extractive"
    llm_base_url: str | None = None  # None=OpenAI 官方；vLLM 例: http://gpu-box:8000/v1
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"  # 换成你们账号 / 本地服务实际可用的模型名
    llm_temperature: float = 0.1  # 问答类任务要"稳"，温度低一些
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # ------------------------------------------------------------------ 解析（Parsing）
    # pypdf  : 只抽文本层，快、零依赖；扫描件会报错。
    # docling: 版面分析 + 表格结构 + OCR，慢但准；需要 pip install -e ".[docling]"
    pdf_parser: Literal["pypdf", "docling"] = "pypdf"

    # ------------------------------------------------------------------ 切块（Chunking）
    # token 数是估算值（中文 1 字≈1 token，英文 1 词≈1.3 token），见 processing/text.py
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 15  # 太短的块（例如孤立标题）会被丢弃或合并

    # ------------------------------------------------------------------ 检索（Retrieval）
    top_k_dense: int = 40
    top_k_keyword: int = 40
    top_k_entity: int = 30
    rrf_k: int = 60  # RRF 平滑常数，业界惯例 60（Haystack 用 61，R2R 用 50）
    weight_dense: float = 1.0
    weight_keyword: float = 1.0
    weight_entity: float = 1.0
    final_top_k: int = 6  # 最终送给 LLM 的证据块数量
    # 没有 reranker 时的拒答门槛：最佳证据块覆盖问题关键词（按 IDF 加权）的比例
    min_keyword_coverage: float = 0.30
    # 近似重复折叠：两个块 SimHash 汉明距离 <= 该值视为重复，只保留一个
    near_dup_hamming: int = 3

    # ------------------------------------------------------------------ 数据 & 权限
    sources_file: Path = Path("configs/sources.yaml")
    # 料号 / 故障码 / 型号的正则。不同公司格式不同，务必按实际改！
    part_number_pattern: str = r"\b\d{3}-\d{5}-[A-Z]\b"
    fault_code_pattern: str = r"\b[EWF]\d{3}\b"
    # 未显式传入用户组时使用的默认组（CLI 场景）。API 场景必须由网关 / 身份系统传入。
    default_user_groups: list[str] = Field(default_factory=lambda: ["public", "service"])
    # 全量同步时，如果"本次没看到的文档"占比超过该值，拒绝删除（防止网盘挂载失败导致清库）
    max_delete_ratio: float = 0.5

    # ------------------------------------------------------------------ API 服务
    # 为空 = 不鉴权（仅限本地开发！启动时会打印警告）
    api_keys: list[SecretStr] = Field(default_factory=list)
    log_level: str = "INFO"
    log_json: bool = False  # 生产建议 True，方便 ELK / Loki 采集


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里如需改配置，直接构造 ``Settings(...)`` 传入即可。"""
    return Settings()
