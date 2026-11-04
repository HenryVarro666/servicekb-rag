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

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSpec(BaseModel):
    """一个"厂家 + 模型"的完整配置。主模型和每个备用模型都是一个 LLMSpec。"""

    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_key_env: str | None = None  # 指定从哪个环境变量读 key（默认用厂家约定的变量名）
    temperature: float | None = 0.1
    max_tokens: int | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    azure_api_version: str = "2024-10-21"
    aws_region: str | None = None
    gcp_project_id: str | None = None
    gcp_region: str = "global"


class ApiClient(BaseModel):
    """一个 API 调用方：名字（写进日志 / 指标）、密钥、允许访问的用户组（None = 不限制，等同可信网关）。"""

    name: str
    key: SecretStr
    groups: list[str] | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SERVICEKB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_parse_none_str="none",  # 环境变量写 none 表示"不设置"，例如 SERVICEKB_LLM_TEMPERATURE=none
    )

    # ------------------------------------------------------------------ 存储
    # memory  : 进程内存储，重启即丢。只用于 demo / 单元测试。
    # postgres: PostgreSQL + pgvector，生产用。
    store_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://servicekb:servicekb@localhost:5432/servicekb"
    db_pool_min_size: int = 1
    # 每个问答请求最多同时占用 3 个连接（三路检索并行）+ 1 个（写日志）。
    # 经验值：max_size ≈ 3 × 单进程并发请求数；多进程部署时总连接数 = 进程数 × max_size，别超过数据库的 max_connections
    db_pool_max_size: int = 20
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
    # 远程向量服务（openai 后端）的超时 / 重试。SDK 默认超时是 600 秒——服务卡住时请求会挂 10 分钟，必须改小
    embedding_timeout_s: float = 30.0
    embedding_max_retries: int = 2
    # 问题向量的 LRU 缓存条数（0 = 关闭）。售后问题重复率高（同一个故障码被反复问），命中后省掉一次模型调用
    query_embedding_cache_size: int = 4096

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
    reranker_timeout_s: float = 10.0  # tei 后端：重排超时就降级为融合排序，不让整个请求卡住

    # ------------------------------------------------------------------ 大模型（LLM），见 docs/11
    # 厂家（provider）：
    #   extractive        不调用任何大模型，从证据里抽句子拼答案（离线 demo / 降级）
    #   openai            OpenAI 官方
    #   azure_openai      Azure OpenAI（model 填部署名）
    #   anthropic         Anthropic Claude 官方 API（原生 SDK，pip install -e ".[anthropic]"）
    #   anthropic_bedrock Claude on Amazon Bedrock      anthropic_vertex  Claude on Google Vertex AI
    #   gemini            Google Gemini（OpenAI 兼容接口）
    #   deepseek / qwen / moonshot / zhipu    国内厂商（OpenAI 兼容接口）
    #   vllm / ollama     本地部署（OpenAI 兼容接口）
    #   openai_compatible 其他任何 OpenAI 兼容服务（必须填 llm_base_url）
    # 运行 `servicekb llm providers` 查看每家的默认地址和 API Key 环境变量。
    llm_provider: str = "extractive"
    llm_backend: Literal["extractive", "openai"] | None = None  # 旧配置项，仅为兼容保留；请改用 llm_provider
    llm_model: str | None = None  # 必须按厂家填写实际可用的模型名（anthropic 默认 claude-opus-5-5）
    llm_base_url: str | None = None  # 覆盖厂家默认地址（例如自建网关、vLLM 地址）
    llm_api_key: SecretStr | None = None  # 不填时读取厂家自己的环境变量，例如 DEEPSEEK_API_KEY
    # 采样温度：none = 不发送。Claude 当前模型不支持 temperature（adapter 自动不发）；
    # OpenAI 的推理模型（o 系列等）也只接受默认温度，用这类模型时设 SERVICEKB_LLM_TEMPERATURE=none
    llm_temperature: float | None = 0.1
    llm_max_tokens: int | None = None  # None = 厂家默认（Claude 16000，因为思考也占输出 token；其他 1024）
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2  # SDK 内置重试（429 / 5xx / 网络错误）
    # 熔断：某个厂家连续失败 N 次后"断开" cooldown 秒，期间直接跳过它（不再每个请求都等一次超时），
    # 冷却后放一个请求试探，成功就恢复。0 = 不熔断
    llm_circuit_failures: int = 3
    llm_circuit_cooldown_s: float = 30.0
    # Claude 专用：思考深度（low / medium / high / xhigh / max）；RAG 问答 medium 足够
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    # Claude 官方 API 专用：被安全分类器拒答时，服务端自动换模型重跑（server-side fallback）
    anthropic_server_fallbacks: bool = True
    azure_api_version: str = "2024-10-21"
    aws_region: str | None = None  # anthropic_bedrock
    gcp_project_id: str | None = None  # anthropic_vertex
    gcp_region: str = "global"  # anthropic_vertex
    # 备用厂家（按顺序）：主模型报错 / 超时 / 拒答时依次尝试。JSON 数组，每项字段同上（不带 llm_ 前缀）：
    # SERVICEKB_LLM_FALLBACKS='[{"provider":"vllm","model":"Qwen/Qwen3-32B","base_url":"http://gpu:8000/v1"}]'
    llm_fallbacks: list[LLMSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compat_llm_backend(self) -> Settings:
        # 旧版本只有 llm_backend=openai（含 vLLM 等兼容服务），映射到新的 provider
        if self.llm_backend == "openai" and self.llm_provider == "extractive":
            self.llm_provider = "openai"
        return self

    def primary_llm_spec(self) -> LLMSpec:
        return LLMSpec(
            provider=self.llm_provider, model=self.llm_model, base_url=self.llm_base_url, api_key=self.llm_api_key,
            temperature=self.llm_temperature, max_tokens=self.llm_max_tokens, effort=self.llm_effort,
            azure_api_version=self.azure_api_version, aws_region=self.aws_region,
            gcp_project_id=self.gcp_project_id, gcp_region=self.gcp_region,
        )

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
    # 三路召回的线程池大小（**整个进程共享**）。旧版本固定为 3：并发请求时所有请求排队等这 3 个线程。
    # 经验值：≥ 3 × 单进程并发请求数。真正的并发上限还受 db_pool_max_size 约束
    retrieval_max_workers: int = 48  # = 3 × max_concurrent_requests 的默认值
    retrieval_channel_timeout_s: float = 10.0  # 单路召回超时：超时 / 报错的那一路被跳过（降级），其他路照常

    # ------------------------------------------------------------------ 数据 & 权限
    sources_file: Path = Path("configs/sources.yaml")
    # 料号 / 故障码 / 型号的正则。不同公司格式不同，务必按实际改！
    part_number_pattern: str = r"\b\d{3}-\d{5}-[A-Z]\b"
    fault_code_pattern: str = r"\b[EWF]\d{3}\b"
    # 未显式传入用户组时使用的默认组（CLI 场景）。API 场景必须由网关 / 身份系统传入。
    default_user_groups: list[str] = Field(default_factory=lambda: ["public", "service"])
    # 全量同步时，如果"本次没看到的文档"占比超过该值，拒绝删除（防止网盘挂载失败导致清库）
    max_delete_ratio: float = 0.5

    # ------------------------------------------------------------------ 图片识别（拍照识别备件，见 docs/10）
    # handcrafted: 颜色 / 纹理 / 形状手工特征。零依赖、无语义，只用于 demo / CI（示例图片是程序画的）。
    # siglip     : 本地加载 SigLIP / SigLIP 2（需要 pip install -e ".[vision]"，建议 GPU）。
    # http       : 调用已有的图像编码服务（POST 图片，返回向量），适合团队已部署 SigLIP worker 的情况。
    image_embedding_backend: Literal["handcrafted", "siglip", "http"] = "handcrafted"
    image_embedding_model: str = "google/siglip-so400m-patch14-384"
    image_embedding_url: str | None = None  # http 后端用
    image_embedding_dim: int = 1152  # 只有 http 后端需要手动指定；其他后端由模型自己报告
    # 高分辨率照片切块：最长边 >= tile_min_side 时，除整图外再切 grid×grid 块分别编码（小零件不会被缩没）
    image_tile_grid: int = 2
    image_tile_min_side: int = 1200
    image_top_k_refs: int = 20  # 每个切块检索多少张参考图
    # auto: 装了 rapidocr 就用 rapidocr，否则退回 metadata（只读样例图片里预置的标签文字）
    ocr_backend: Literal["auto", "rapidocr", "metadata", "none"] = "auto"
    # 序列号格式（示例：AS + 2 位机型码 + YYMM + 6 位流水号）。year / month 两个命名组用于解析出厂年月
    serial_number_pattern: str = r"\bAS\d{2}(?P<year>\d{2})(?P<month>0[1-9]|1[0-2])\d{6}\b"
    # 视觉最高分低于该值且 OCR 也没认出料号 → 判定为"未识别"（需要在你们自己的照片上校准！）
    photo_min_visual_score: float = 0.80
    # 视觉分数与第一名相差小于该值的候选视为"长得像"，没有 OCR 确认时提示用户拍铭牌
    photo_lookalike_margin: float = 0.03

    # ------------------------------------------------------------------ API 服务
    # 为空 = 不鉴权（仅限本地开发！启动时会打印警告）。这些 Key 可以访问任意用户组（等同可信网关）
    api_keys: list[SecretStr] = Field(default_factory=list)
    # 按调用方分配 Key 和**允许的用户组**（最小权限）：X-User-Groups 里超出范围的组会被丢弃。
    # 例：SERVICEKB_API_CLIENTS='[{"name":"teams-bot","key":"...","groups":["public","service"]}]'
    api_clients: list[ApiClient] = Field(default_factory=list)
    # 同时处理的重请求（ask / search / photo）上限（每个进程）。超过时排队最多 overload_wait_s 秒，仍拿不到就返回 503 + Retry-After，
    # 保护 GPU / 大模型配额，避免突发流量把所有请求一起拖慢。0 = 不限制
    max_concurrent_requests: int = 16
    overload_wait_s: float = 5.0
    metrics_enabled: bool = True  # GET /metrics（Prometheus 格式）。不要暴露到公网
    # 网页问答 /ui（静态页面，调用上面这些 API）。生产环境放在网关后面，由网关负责登录和设置用户组
    ui_enabled: bool = True
    # 查询日志里的问题 / 答案先脱敏（手机号、邮箱、身份证号）再落库；料号、序列号不受影响
    query_log_redact_pii: bool = True
    query_log_retention_days: int = 180  # servicekb db prune-logs 的默认保留天数（有反馈的记录默认保留）
    log_level: str = "INFO"
    log_json: bool = False  # 生产建议 True，方便 ELK / Loki 采集


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里如需改配置，直接构造 ``Settings(...)`` 传入即可。"""
    return Settings()
