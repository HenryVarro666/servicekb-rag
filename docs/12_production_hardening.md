# 12 生产加固：工程评审结论与改进

> 前面的文档讲的是"RAG 怎么做得准"，这篇讲的是"**系统怎么在真实环境里稳定地跑**"：
> 并发、依赖故障、超时、过载、安全、监控、运维。
> 方法：逐个模块读代码 → 压测 → 故障注入（让向量服务 / 某一路召回 / 大模型故意失败），用数据确认问题、用测试锁住修复。
> 对应测试：[`tests/test_hardening.py`](../tests/test_hardening.py)；压测脚本：[`scripts/loadtest.py`](../scripts/loadtest.py)。

## 1. 结论一览

| # | 问题 | 后果（工业场景） | 严重度 | 状态 |
| --- | --- | --- | --- | --- |
| H1 | 三路召回共用一个**固定 3 线程**的进程级线程池 | 并发请求互相排队，延迟随并发线性上涨（实测见 §3） | 高 | ✅ 已修复 |
| H2 | 单进程部署（`serve` / Docker 默认 1 个 worker） | Python GIL：CPU 密集时多核用不上 | 中 | ✅ 支持 `WEB_CONCURRENCY`，文档化 |
| H3 | 问题向量算不出来（向量服务挂了）→ 整个请求 500 | 一个依赖故障 = 整个知识库不可用 | 高 | ✅ 降级为关键词 + 实体两路 |
| H4 | 任何一路召回报错 / 卡住 → 整个请求失败 / 挂住 | 同上 | 高 | ✅ 单路超时 + 跳过 |
| H5 | 远程向量服务**超时默认 600 秒**（SDK 默认值） | 服务卡住时请求挂 10 分钟，线程被占满 | 高 | ✅ 默认 30 秒，可配 |
| H6 | 大模型挂了时每个请求都先等一次超时（60 秒 × 重试） | 故障期间整体延迟爆炸；备用厂家形同虚设 | 高 | ✅ 熔断器 |
| H7 | 没有过载保护 | 突发流量把 GPU / 大模型配额打满，所有请求一起变慢、一起超时 | 高 | ✅ 并发上限 + 503 + Retry-After |
| H8 | 拍照接口是 `async def` 里直接调用 OCR / 图像模型 | **阻塞事件循环**：识别一张图期间，连 `/healthz` 都没响应，K8s 可能误杀 Pod | 高 | ✅ 放进线程池 |
| H9 | 非法输入（`doc_types` 写错、反馈的 `query_id` 不是 UUID）→ 500 | 调用方以为服务坏了；告警被噪音淹没 | 中 | ✅ 422 |
| H10 | 所有 API Key 权限相同，`X-User-Groups` 完全信任调用方 | 任何一个调用方（例如机器人）的 Key 泄露 = 能看所有内部资料 | 高 | ✅ 按调用方限制可访问的用户组 |
| H11 | 没有监控指标、没有请求 ID | 出问题只能翻日志猜；用户报障无法定位到具体请求 | 高 | ✅ `/metrics` + `X-Request-ID` |
| H12 | 查询日志存原文（可能含手机号）、永久保存、缺拒答原因 / token 用量 | 隐私合规风险；无法分析"为什么答不了""花了多少钱" | 中 | ✅ 脱敏 + 保留期清理 + 新字段 |
| H13 | 同一数据源可以同时跑两个导入任务 | 定时任务重叠时重复计算向量，"删除源头已不存在的文档"可能互相干扰 | 中 | ✅ 数据库 advisory lock |
| H14 | 配置错误要等第一个真实请求才暴露 | 上线即故障 | 中 | ✅ `servicekb doctor` |
| H15 | 数据库不可用时返回 500 | 调用方不知道该不该重试 | 低 | ✅ 503 + Retry-After |
| — | 流式输出、换模型零停机重建、OIDC、链路追踪、语义缓存…… | 见 §6 | — | ⏳ 待做 |

## 2. 降级矩阵：每个依赖挂了会怎样

工业系统的依赖一定会出故障。设计目标：**任何单个非核心依赖故障，用户仍然能拿到（可能质量差一点的）答案，并且监控能看到。**

| 依赖 | 故障时的行为 | 用户看到的 | 监控 |
| --- | --- | --- | --- |
| 向量服务（TEI / OpenAI embeddings） | 跳过向量路，关键词 + 实体两路照常 | 正常答案（同义表达的召回变差） | `degraded_total{component="dense"}` |
| 某一路召回（SQL 报错 / 超时） | 跳过这一路 | 正常答案 | `degraded_total{component=…}` |
| **所有**召回都失败（数据库挂了） | 503 + `Retry-After` | "暂时不可用，请重试" | 5xx 比例、`/readyz` 失败 |
| 重排服务 | 用 RRF 融合排序 | 正常答案（排序变差；拒答回到启发式） | `degraded_total{component="rerank"}` |
| 主大模型 | 熔断 → 备用厂家（见 [docs/11](11_llm_providers.md)） | 正常答案 | `llm_requests_total`、`llm_circuit_open` |
| 所有大模型 | 抽取式回答（证据原文摘录） | 带引用的原文摘录 | `ask_total{outcome="llm_fallback"}` |
| OCR / 图像模型 | 照片识别报错 | 422 / 500（文字问答不受影响） | HTTP 指标 |
| 查询日志写入 | 记录异常日志，回答照常返回 | 无感 | 应用日志 |

每次降级都写在返回结果的 `degraded` 字段和查询日志里（例如 `["dense:embedding_failed"]`）。

## 3. 压测：数据说话

### 3.1 压测方法

```bash
servicekb serve --port 8000 &
python scripts/loadtest.py --endpoint /v1/search --concurrency 16 --duration 20
python scripts/loadtest.py --endpoint /v1/ask    --concurrency 8  --duration 20   # 接真实大模型时会花钱
```

脚本从评估集里轮流取问题，输出吞吐、p50 / p95 / p99 延迟和状态码分布。**上线前、改了部署配置后都要压一遍。**

### 3.2 共享线程池（H1）：I/O 密集时的瓶颈

真实部署里，三路召回的时间主要花在**等待**：等数据库、等网络。下面用"每一路召回等待 30 ms"模拟
（`tests/test_hardening.py` 里的 `SlowStore`，内存存储 + 示例数据）：

| 并发 | 线程池 3（旧） | 线程池 32（新） |
| --- | --- | --- |
| 4 | 31 req/s，p50 127 ms | 104 req/s，p50 37 ms |
| 16 | 31 req/s，**p50 508 ms** | **290 req/s**，p50 49 ms |

旧版本的吞吐被死死卡在 31 req/s（= 3 线程 ÷ 3 路 ÷ 30 ms），并发越高延迟越长；新版本吞吐随并发增长（**9.3 倍**）。
这个瓶颈在"示例数据 + 进程内哈希向量"的 demo 里看不出来（那时是 CPU 密集），换成真实的远程服务后才会暴露——
这正是为什么要按真实部署形态去压测。

### 3.3 单进程与多进程（H2）：CPU 密集时的瓶颈

真实 Postgres + 示例数据（65 个块）+ 哈希向量，4 核机器（压测客户端也在同一台机器上），`/v1/search`：

| 部署 | 并发 1 | 并发 16 |
| --- | --- | --- |
| 1 个 worker | 98.6 req/s，p50 9.8 ms | 85.7 req/s，p50 184 ms |
| 4 个 worker | — | **244.8 req/s**，p50 62 ms |

单进程时吞吐不随并发增长——Python 的 GIL 让 CPU 密集的部分（分词、融合、去重、行转换）只能用一个核。
做法：单机部署 `WEB_CONCURRENCY=<核数>`；K8s 部署每个 Pod 1 个 worker、靠副本数扩容（指标也更简单）。

**新代码在这个场景下没有性能回退**：同一台机器上新旧版本交替各压两轮（并发 16），旧版本 95~96 req/s，
新版本 93~95 req/s，差异在测量噪声内（同一配置两次运行的差异约 5%）。为此把访问日志 / 指标中间件写成了纯 ASGI 形式
（`@app.middleware("http")` 每个请求多建一个任务，开销肉眼可见）。

> ⚠️ 这些数字只说明"瓶颈在哪、改动有没有用"，不代表生产性能。生产数据量大得多，
> 时间主要花在 GPU 服务和大模型上——请在真实部署上用 `loadtest.py` 重新测。

## 4. 各项改进说明

### 4.1 并发：线程池 / 连接池 / 进程数（H1、H2）

一个问答请求同时占用：1 个召回线程池名额 × 3（三路并行）+ 最多 3 个数据库连接 + 1 个（写日志）。三个参数要一起配：

| 配置 | 默认 | 经验值 |
| --- | --- | --- |
| `SERVICEKB_MAX_CONCURRENT_REQUESTS` | 16 | 按 GPU / 大模型能承受的并发定 |
| `SERVICEKB_RETRIEVAL_MAX_WORKERS` | 48 | ≥ 3 × 并发上限 |
| `SERVICEKB_DB_POOL_MAX_SIZE` | 20 | ≥ 并发上限；**进程数 × 该值** ≤ 数据库 `max_connections`（Postgres 默认 100） |
| `WEB_CONCURRENCY` | 1 | 单机：CPU 核数；K8s：1 |

`servicekb doctor` 会检查这几个值是否互相矛盾。

### 4.2 超时与降级（H3、H4、H5、H15）

- 远程向量服务：`SERVICEKB_EMBEDDING_TIMEOUT_S=30`（以前是 SDK 默认的 600 秒）、`EMBEDDING_MAX_RETRIES=2`；
- 重排服务：`SERVICEKB_RERANKER_TIMEOUT_S=10`，只重试"连接失败"（服务重启瞬间），不重试超时——超时说明服务过载，重试只会更糟；
- 单路召回：`SERVICEKB_RETRIEVAL_CHANNEL_TIMEOUT_S=10`；
- 关键词路和实体路不依赖向量，现在**先提交、和"算问题向量"并行**，顺带省一点延迟。

**超时怎么定**：比该依赖正常 p99 大几倍，但远小于调用方（网关 / 前端）的超时。否则调用方先超时，服务端还在白干活。

### 4.3 问题向量缓存

售后问题重复率很高（同一个故障码一天被问几十次）。`CachedQueryEmbedder` 给问题向量加了线程安全的 LRU 缓存
（`SERVICEKB_QUERY_EMBEDDING_CACHE_SIZE=4096`，0 = 关闭）。上面的压测里命中率 99.5%（问题集只有 18 个，偏高；
真实命中率看 `servicekb_query_embedding_cache_total`）。只缓存问题向量，不缓存答案——答案缓存要处理权限和数据更新，见 §6。

### 4.4 大模型熔断 + token 用量（H6）

熔断器三种状态：正常 →（连续失败 3 次）→ 断开 30 秒（直接跳过，立即走备用厂家）→ 试探一个请求 → 成功则恢复。
**拒答不算失败**：那说明服务是好的，只是这条内容不行；否则几个敏感问题就能让所有人都用不了这家。

每次回答的 token 用量记在 `AnswerResult.usage`、API 响应、查询日志（`usage` 字段）和指标（`servicekb_llm_tokens_total`）里，
**包括失败后切换备用厂家的那几次调用**——它们也要花钱。按厂家单价乘一下就是成本。

### 4.5 过载保护（H7）

同时处理的 ask / search / photo 请求超过 `MAX_CONCURRENT_REQUESTS` 时，新请求最多等 `OVERLOAD_WAIT_S` 秒，
仍然没有名额就返回 **503 + Retry-After**。为什么"拒绝一部分"比"全部接下来"好：大模型 / GPU 的处理能力是固定的，
接下超出能力的请求只会让**所有**请求都排队、都超时；快速失败的请求，客户端可以退避重试。

这是进程内的保护。跨实例的限流（按调用方、按用户的 QPS 限制）应该在网关上做。

### 4.6 不阻塞事件循环（H8）

FastAPI 里 `def` 接口自动在线程池执行，`async def` 接口直接在事件循环里执行。拍照接口需要 `await` 读上传文件，
所以写成了 `async def`，但随后同步调用 OCR / 图像模型（几百毫秒到几秒）——这期间整个进程不能处理任何其他请求。
修复：`await run_in_threadpool(...)`。**规则：`async def` 里不能有任何阻塞调用**（同步数据库、同步 HTTP、模型推理）。

### 4.7 调用方最小权限（H10）

```bash
# 可信网关：可以传任意用户组（它负责登录和鉴权）
SERVICEKB_API_KEYS='["gateway-key"]'
# 其他调用方：只能访问指定的组，X-User-Groups 里超出范围的组会被丢弃（并记录告警日志）
SERVICEKB_API_CLIENTS='[{"name":"teams-bot","key":"bot-key","groups":["public","service"]}]'
```

调用方名字会写进查询日志的 `client` 字段，方便按调用方审计。Key 比较是常数时间的，并且遍历所有 Key（耗时与匹配到哪个无关）。

### 4.8 可观测性（H11）

- **请求 ID**：调用方可以传 `X-Request-ID`（只接受安全字符，防日志注入），否则自动生成；写进这次请求的所有日志、
  查询日志的 `request_id` 字段，并在响应头返回。用户报障时让对方提供这个 ID。
- **访问日志**：每个请求一行（方法、路由、状态码、耗时），探针请求除外。建议生产开启 `SERVICEKB_LOG_JSON=true`。
- **指标**：`GET /metrics`，指标清单和建议的告警规则见 [`metrics.py`](../src/servicekb/metrics.py) 文件头。最少要配的告警：

| 告警 | 条件（示例） | 通常意味着 |
| --- | --- | --- |
| 错误率 | 5xx 比例 > 1%（5 分钟） | 数据库 / 代码问题 |
| 延迟 | `/v1/ask` p95 > 10 s | 大模型慢 / 过载 |
| 拒答率突增 | `ask_total{outcome="refused"}` 占比是上周同期的 2 倍 | 数据没导进来 / reranker 阈值 / 新问题类型 |
| 熔断 | `llm_circuit_open == 1` 持续 5 分钟 | 主大模型故障 / 配额用完 |
| 降级 | `degraded_total` 增速突增 | 向量 / 重排服务故障 |
| 过载 | `overload_rejected_total` 持续增长 | 需要扩容或调整并发上限 |

多进程部署时注意：每个进程的计数器是独立的。要么每个容器 1 个进程（推荐），要么设置 `PROMETHEUS_MULTIPROC_DIR`。

### 4.9 查询日志：隐私与保留期（H12）

- 落库前脱敏问题和答案（手机号、邮箱、身份证号 → 占位符），**料号、序列号、故障码不受影响**（`SERVICEKB_QUERY_LOG_REDACT_PII`）；
- 新字段（迁移 `003_query_log_ops.sql`）：`reason`（拒答 / 降级原因）、`degraded`、`usage`、`request_id`、`client`；
- 保留期：`servicekb db prune-logs`（默认删除 180 天前的日志，**有用户反馈的保留**——它们是评估集和精选问答的来源）。
  建议每天定时跑一次（cron / K8s CronJob）。

有了 `reason` 字段，"为什么答不了"可以直接统计：

```sql
SELECT split_part(reason, ':', 1) AS why, count(*) FROM query_logs
WHERE escalate AND created_at > now() - interval '7 days' GROUP BY 1 ORDER BY 2 DESC;
```

### 4.10 导入并发锁（H13）

`pipeline.run` 先拿"数据源锁"：Postgres 用 session 级 advisory lock（独立连接，不占连接池；进程崩溃时数据库自动释放）。
拿不到锁 → 本次跳过，状态 `skipped_locked`。K8s CronJob 也建议同时设置 `concurrencyPolicy: Forbid`（双保险）。

### 4.11 上线前自检（H14）

```bash
servicekb doctor          # 依赖连通性 + 危险配置检查；有 ✗ 时退出码 1
servicekb doctor --llm    # 额外实际调用一次大模型（会产生少量费用）
servicekb doctor --vision # 额外加载图像模型（SigLIP 会下载 / 加载权重）
```

检查项：向量模型（维度是否和配置一致）、数据库（连通、有没有数据、建库时的向量模型是否和当前一致）、
重排服务、大模型配置、OCR、危险配置（没开鉴权、memory 存储、哈希向量、连接池 / 线程池和并发上限矛盾、日志不脱敏……）。
可以放进部署流水线或 K8s 的 initContainer。

## 5. 上线检查清单

- [ ] `servicekb doctor` 没有 ✗，⚠ 都确认过
- [ ] 鉴权：网关用 `API_KEYS`，其他调用方用 `API_CLIENTS` 并限制用户组；`X-User-Groups` 只由网关设置
- [ ] `SERVICEKB_LOG_JSON=true`，日志接入日志平台；`/metrics` 接入 Prometheus，配好 §4.8 的告警；`/metrics` 不暴露公网
- [ ] 并发三件套（并发上限 / 召回线程池 / 连接池）和进程数按 §4.1 配好，数据库 `max_connections` 够用
- [ ] 用 `scripts/loadtest.py` 在真实部署上压测，记录 p95 和最大吞吐
- [ ] 故障演练：分别停掉向量服务、重排服务、主大模型，确认降级行为和告警（§2）
- [ ] `servicekb db prune-logs` 定时任务；数据库备份（PITR）
- [ ] 导入定时任务设置 `concurrencyPolicy: Forbid`，监控导入退出码和 `ingestion_runs.status`

## 6. 还没做的（按优先级）

| 优先级 | 事项 | 为什么重要 | 设计要点 |
| --- | --- | --- | --- |
| P0 | **用户身份直接校验**（Entra ID / OIDC JWT） | 现在依赖网关传 `X-User-Groups` | API 校验 JWT 签名，从 token 的 groups claim 映射用户组；网关模式保留 |
| P1 | **流式输出**（SSE） | 接大模型后一个答案要 5~20 秒，用户盯着空白屏幕 | `ChatModel.complete_stream()`；`/v1/ask/stream` 先推送文字增量，最后一个事件带引用校验后的完整结果 |
| P1 | **换向量模型零停机** | 现在换模型 = `db reset` + 全量重建，期间不可用 | 新建 `chunks_v2` 表 / 新的 index_meta 版本后台重建，评估通过后切换读路径，旧表保留回滚 |
| P1 | **链路追踪**（OpenTelemetry） | 指标告诉你"慢了"，追踪告诉你"慢在哪一次调用" | FastAPI / httpx / psycopg 都有自动埋点；span 带上 request_id |
| P1 | **导入任务队列** | 单进程同步导入，几万份文档首次导入慢、无法暂停 / 优先级 | Arq / Celery + Redis；按文档分发；`source_lock` 已经解决重复执行 |
| P1 | **查询日志分区** | 日志表持续增长 | 按月分区（`PARTITION BY RANGE (created_at)`），清理变成 `DROP PARTITION` |
| P2 | **语义答案缓存** | 高频问题省掉大模型调用 | 键必须包含用户组、过滤条件、索引版本；导入后失效；只缓存非拒答结果 |
| P2 | **提示词注入加固** | 证据（工单、聊天记录）里可能夹带指令 | 证据用不可伪造的分隔标记包裹、过滤分隔符；输出侧检查是否泄露系统提示 |
| P2 | **数据删除请求** | 隐私合规：某人要求删除其数据 | 聊天记录来源的问答对带 `source_url` / 会话 ID，按人反查并删除 |

## 7. 故障处理手册（Runbook）

| 现象 | 先看 | 处理 |
| --- | --- | --- |
| 大量 503 `server busy` | `overload_rejected_total`、`inflight_requests`、大模型延迟 | 大模型变慢 → 看厂家状态 / 熔断；流量真的大了 → 扩容副本 |
| 大量 503 `knowledge base temporarily unavailable` | `/readyz`、数据库连接数 / 负载 | 数据库故障或连接池耗尽 |
| 答案都是"原文摘录"（`model=extractive-fallback`） | `llm_requests_total{outcome}`、`llm_circuit_open` | 所有大模型都失败：Key / 配额 / 网络；`servicekb llm test` |
| `degraded` 里出现 `dense:embedding_failed` | 向量服务健康状况 | 检查 TEI；恢复前同义问题召回变差 |
| 拒答率突增 | 查询日志的 `reason` 统计（§4.9 SQL） | `low_rerank_score` 多 → 阈值 / reranker；`no_results` 多 → 数据没导入 |
| 用户报"刚才那个回答不对" | 让用户提供 `X-Request-ID` | 日志平台按 request_id 查整条链路；查询日志里看召回了什么 |
| 导入状态 `skipped_locked` | `ingestion_runs` 最近一次运行 | 上一轮还没结束：正常；一直跳过 → 查是否有卡住的导入进程 |
