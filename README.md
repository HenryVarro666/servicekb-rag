# servicekb — 面向工业售后 / 现场服务的 RAG 知识库框架

> 把**产品手册、服务通告、历史工单、物料主数据、Teams 聊天记录**这些又多又脏的数据，变成一个能回答现场技术问题、**带出处、懂权限、不知道就说不知道**的知识库。

- 🧱 **一个数据库搞定**：PostgreSQL + pgvector，同时做关系库、向量库、全文库。3~4 人的小团队也运维得动
- 🔀 **三路混合检索**：向量（语义）+ 关键词（中英混合分词）+ **实体**（型号 / 料号 / 故障码），RRF 融合，可选 cross-encoder 重排
- 🧩 **对象层**：把"按记录存"的数据整理成"按对象存"：型号别名归一、料号替代关系、停产状态，直接用于检索和回答
- ♻️ **工业级数据 pipeline**：增量、幂等、向量复用、故障隔离、安全删除、每次运行可审计
- 🔐 **权限与合规**：块级 ACL、PII 脱敏、证据不足自动拒答并标记升级
- 💬 **聊天记录清洗**：Teams 导出 → 线程重建 → 问答抽取 → 人工审核分流（[详细指南](docs/09_chat_log_cleaning.md)）
- 📏 **自带评估**：检索 hit@k / MRR / nDCG + 回答关键事实召回 + 拒答准确率，CI 回归门禁
- 🎓 **为新人而写**：每个模块开头都解释"这一步在 RAG 里是干什么的、为什么这么做"

**零依赖体验**：不需要 GPU、不需要 API Key、不需要数据库，5 分钟跑通全链路。

> ⚠️ `data/sample/` 下的所有数据（ACME Solar 品牌、型号、料号、故障码、阈值）都是**虚构的**，仅用于演示，不代表任何真实产品，不能用于真实设备维修。

---

## 目录

- [5 分钟上手](#5-分钟上手)
- [它是怎么工作的](#它是怎么工作的)
- [接入 PostgreSQL](#接入-postgresql)
- [接入真实模型（GPU / OpenAI）](#接入真实模型gpu--openai)
- [项目结构](#项目结构)
- [核心设计决策](#核心设计决策)
- [当前效果](#当前效果)
- [常见任务](#常见任务)
- [文档导航（推荐阅读顺序）](#文档导航推荐阅读顺序)
- [不足与路线图](#不足与路线图)
- [FAQ](#faq)
- [参考与致谢](#参考与致谢)

---

## 5 分钟上手

需要 Python 3.10+。

```bash
git clone <this-repo> servicekb-rag && cd servicekb-rag
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
servicekb demo
```

你会看到它导入示例数据（16 个料号、3 本手册、2 份服务通告、21 张工单，其中 2 张没有结论被质量门槛过滤），然后回答 4 个问题（以下为输出节选）：

```text
──────────────── 310-04521-A 还能订到吗？ ────────────────
- 料号 310-04521-A（散热风扇组件）状态：停产；已被 310-04521-B 替代；适用机型：AS-110K-HV。   ← 对象层注入的结构化事实
- 所有新的风扇更换工单一律使用 310-04521-B，不再发 310-04521-A 的库存。 [3]
- 自 2025 年 6 月 1 日起，AS-110K-HV 散热风扇组件旧料号 310-04521-A 停产，由新料号 310-04521-B 替代。 [4]
  [3] bulletins:SB-2025-014_fan_module_change.md  3 服务要求
  [4] bulletins:SB-2025-014_fan_module_change.md  1 变更内容
──────────────── 逆变器支持接入风力发电机吗？ ────────────────
知识库中没有找到足够依据回答这个问题。建议转交技术支持工程师处理（已标记为需升级）。
```

> 默认是**离线抽取模式**（不调用大模型，直接摘录证据原文），所以答案读起来像"摘抄"。接上 LLM 之后就是正常的自然语言回答，见[接入真实模型](#接入真实模型gpu--openai)。

再试试：

```bash
servicekb ask "AS-250K-HV 的直流开关坏了，仓库里的 220-11008-A 能用吗？"
servicekb search "as110k 早上报 E037"          # 看检索细节：每个结果在三路检索里各排第几
servicekb ask "E051 换哪个料号" --groups public  # 换成外部用户权限：内部服务通告和工单就搜不到了
servicekb eval                                  # 跑评估集
servicekb chat clean                            # 清洗示例 Teams 聊天记录
servicekb serve                                 # 启动 HTTP API → http://localhost:8000/docs
pytest                                          # 跑测试
```

---

## 它是怎么工作的

```mermaid
flowchart TB
    subgraph Sources[数据源]
        D[领域词典 YAML<br/>型号别名 / 故障码]
        P[物料主数据 CSV/XLSX<br/>料号 / 替代件]
        M[手册 / 通告<br/>MD / PDF / DOCX / XLSX]
        Q[历史工单 JSONL/CSV]
        T[Teams 聊天记录]
        S[SharePoint<br/>Graph delta]
    end

    T -->|chat clean<br/>清洗 + 抽取问答| Q2[teams_qa.jsonl]

    subgraph Ingest[导入 pipeline]
        direction LR
        I1[解析<br/>→ Block] --> I2[清洗<br/>去噪 / 脱敏] --> I3[结构化切块] --> I4[实体链接] --> I5[分词] --> I6[向量化<br/>复用旧向量]
    end

    D & P -->|对象层先行| E[(entities<br/>entity_relations)]
    M & Q & Q2 & S --> Ingest
    Ingest --> C[(chunks<br/>vector + tsvector + entity_ids + acl)]

    subgraph Query[问答]
        direction LR
        A1[查询分析<br/>实体 / 替代件扩展 / 分词] --> A2[三路召回<br/>向量 · 关键词 · 实体]
        A2 --> A3[RRF 融合<br/>重复折叠] --> A4[重排<br/>可选] --> A5{证据够吗?}
        A5 -->|够| A6[LLM 生成<br/>+ 结构化事实<br/>+ 引用校验]
        A5 -->|不够| A7[拒答 + 标记升级]
    end

    C --> A2
    E --> A1
    A6 --> L[(query_logs / feedback)]
    A7 --> L
```

一句话版本：**先建对象层（型号、料号、故障码是什么、谁替代谁），再把文档切块并挂到对象上；查询时三路一起找、融合排序、证据不足就拒答，回答时把对象层的事实和证据一起交给 LLM，并强制引用。**

完整解释见 [docs/01_architecture.md](docs/01_architecture.md)；如果你还不熟悉 RAG 的基本概念，先读 [docs/00_rag_primer.md](docs/00_rag_primer.md)。

---

## 接入 PostgreSQL

内存模式每次启动都要重新导入，只适合体验。真正使用请接 Postgres：

```bash
docker compose up -d db                   # 启动 PostgreSQL 16 + pgvector
cp .env.example .env                      # 然后编辑 .env：
#   SERVICEKB_STORE_BACKEND=postgres
#   SERVICEKB_DATABASE_URL=postgresql://servicekb:servicekb@localhost:5432/servicekb
servicekb db init                         # 建表（自动执行 migrations）
servicekb ingest                          # 导入；再跑一次会发现全部 unchanged（增量）
servicekb ask "E037 早上出现中午消失"
```

没有 Docker？任何 PostgreSQL 14+ 装上 [pgvector](https://github.com/pgvector/pgvector) 扩展即可（Ubuntu：`apt install postgresql-16-pgvector`）。表结构和设计说明见 [docs/03_database.md](docs/03_database.md)。

---

## 接入真实模型（GPU / OpenAI）

离线模式用的"哈希向量"**没有语义理解能力**（例如英文问题搜不到中文文档）。生产环境的推荐组合：

| 组件 | 推荐 | 部署位置 | 配置 |
| --- | --- | --- | --- |
| 向量模型 | `BAAI/bge-m3`（中英混合、1024 维、8192 长度） | 48GB 卡，TEI 容器 | `EMBEDDING_BACKEND=openai` + `EMBEDDING_BASE_URL=http://gpu:8080/v1` + `EMBEDDING_DIM=1024` |
| 重排模型 | `BAAI/bge-reranker-v2-m3` | 48GB 卡，TEI 容器 | `RERANKER_BACKEND=tei` + `RERANKER_URL=http://gpu:8081` |
| 大模型 | OpenAI API，或本地 vLLM（96GB 卡可跑 30B 级模型） | 96GB 卡 / 云 | `LLM_BACKEND=openai` + `LLM_MODEL=...`（+ `LLM_BASE_URL` 指向 vLLM）|

```bash
docker compose --profile gpu up -d        # 启动 embeddings / reranker / llm 三个 GPU 服务
# 按 .env.example 里"方案 B"修改 .env
servicekb db reset --yes && servicekb db init && servicekb ingest   # 换了向量模型必须重建索引
```

以上配置项都带 `SERVICEKB_` 前缀，完整列表见 [`src/servicekb/config.py`](src/servicekb/config.py)，部署细节、GPU 规划、安全清单见 [docs/06_deployment.md](docs/06_deployment.md)。

---

## 项目结构

```text
servicekb-rag/
├── configs/sources.yaml            # 数据源配置：有哪些数据、在哪、谁能看
├── data/sample/                    # 虚构示例数据（词典、料号、手册、通告、工单、Teams 导出）
├── eval/golden_qa.jsonl            # 评估集：问题 + 相关文档 + 必须出现的关键事实
├── src/servicekb/
│   ├── config.py                   # 所有可调参数（环境变量覆盖）
│   ├── models.py                   # 核心数据结构：SourceRecord → ParsedDocument → Chunk → AnswerResult
│   ├── connectors/                 # 【读数据】每种数据源一个 connector
│   │   ├── filesystem.py           #   本地 / 共享盘文件
│   │   ├── qa_archive.py           #   历史工单（含质量门槛）
│   │   ├── parts_master.py         #   物料主数据 → 实体 + 关系 + 每料号一条记录
│   │   ├── dictionary.py           #   领域词典 YAML → 型号 / 故障码实体
│   │   └── sharepoint.py           #   SharePoint（Graph delta 增量 + 限流重试）
│   ├── parsing/                    # 【解析】各种格式 → 统一的 Block 列表
│   ├── processing/
│   │   ├── text.py                 #   规范化、token 估算、中英分句、jieba 分词
│   │   ├── cleaning.py             #   邮件去噪、PII 脱敏
│   │   ├── chunking.py             #   结构化切块（标题路径 / 表格重复表头 / 问答原子块）
│   │   ├── entities.py             #   实体链接、替代件扩展、结构化事实
│   │   └── dedup.py                #   SimHash 近似去重
│   ├── embedding.py                # 【向量化】hashing / sentence-transformers / OpenAI-compatible
│   ├── store/
│   │   ├── base.py                 #   存储接口
│   │   ├── memory.py               #   内存实现（含标准 BM25，适合对照学习）
│   │   ├── postgres.py             #   Postgres 实现（HNSW、tsvector、原子替换）
│   │   └── migrations/001_init.sql #   表结构（每个字段都有注释）
│   ├── pipeline/ingest.py          # 【导入】增量 / 幂等 / 向量复用 / 故障隔离 / 安全删除
│   ├── retrieval/
│   │   ├── retriever.py            #   查询分析 + 三路并行召回
│   │   ├── fusion.py               #   RRF 融合 + 重复折叠
│   │   └── rerank.py               #   cross-encoder / TEI 重排
│   ├── generation/
│   │   ├── prompts.py              #   prompt（带版本号）
│   │   └── generator.py            #   LLM / 抽取式降级 + 引用校验
│   ├── chat/                       # 【聊天记录清洗】Teams HTML、线程、名册、问答抽取、分流
│   ├── service.py                  # 应用容器：ingest / search / ask（证据门槛、拒答、日志）
│   ├── evaluation.py               # 评估指标与报告
│   ├── api.py                      # FastAPI：/v1/ask /v1/search /v1/feedback /healthz /readyz
│   └── cli.py                      # 命令行
├── tests/                          # 58 个测试，含 Postgres 集成测试
├── docs/                           # 详细文档（见下文导航）
├── Dockerfile / docker-compose.yml # 部署（含 GPU profile：TEI ×2 + vLLM）
└── .github/workflows/ci.yml        # lint → 测试 → Postgres 集成测试 → 评估门禁
```

---

## 核心设计决策

每一条都来自对开源项目源码的分析（详见 [docs/07_repo_survey.md](docs/07_repo_survey.md)）或工业场景的真实约束。

| 决策 | 为什么 | 借鉴 / 对比 |
| --- | --- | --- |
| 只用 PostgreSQL + pgvector | 小团队只运维一套系统；事务保证"文档和块原子替换"；ACL / 实体 / 日志都是关系数据 | R2R、LightRAG 都支持全 Postgres；RAGFlow 需要 MySQL+Redis+MinIO+ES，太重 |
| 中文分词在 Python 侧做，直接写 tsvector 字面量 | 托管 Postgres 往往装不了 zhparser / pg_jieba；分词逻辑与内存实现完全一致 | R2R / LlamaIndex / Haystack 的 pgvector 集成都默认 `english` 配置，对中文无效 |
| **实体检索作为第三路** | 向量检索分不清 "310-04521-A" 和 "310-04521-B"；型号写法五花八门 | 各大框架都没有内置，是工业场景特有需求 |
| 物料主数据 → 结构化事实注入 prompt | 历史工单可能推荐已停产料号；主数据才是权威 | 对应"record-oriented → object-oriented"的数据现状 |
| RRF，k=60，三路等权 | 不同检索分数量纲不同，只融合排名最稳；不照搬 R2R 的 5:1 语义偏置（对 "E037" 这类查询不利） | Haystack（k=61）、R2R（k=50） |
| 块 ID = hash(文档 ID, 块内容) | 文档改一段，只有那一块重新算向量 | R2R 用位置做 ID（插入一段后面全变），LlamaIndex/RAGFlow 用内容哈希 |
| pipeline_version = 代码版本 + 切块配置 + 模型 + 词典指纹 | 任何处理逻辑变化都会自动触发重建，向量尽量复用 | RAGFlow 的 task fingerprint |
| 增量源只认显式墓碑；全量源删除有比例保护 | 防止"网盘没挂上 → 知识库被清空" | LlamaIndex `UPSERTS_AND_DELETE` 在增量场景会误删 |
| 解析失败保留旧版本 | 新版 PDF 坏了，用户仍能查到旧版 | Dify 的分阶段状态机 |
| 入库不删重复，检索时折叠 | 保留数据血缘（不同机型 / 时间 / 权限）；top-k 不被重复内容占满 | — |
| 标题路径只进向量文本，不进正文 | 上下文化提升召回，又不污染引用片段 | docling `contextualize`；Chatchat 把标题写进正文（不推荐） |
| 问答对是原子块；表格按行拆并重复表头 | 一问一答拆开就没法用；第二块表格没有表头看不懂 | RAGFlow qa / table 模板 |
| 证据不足 → 不调 LLM 直接拒答并标记升级 | 省钱、防幻觉；对应"未回复请求的跟踪升级" | QAnything 的重排分数阈值 + 相对跌幅截断 |
| 权限只来自身份系统，绝不让 LLM 决定 | 防止越权 | Dify 的自动元数据过滤只用于业务字段 |

---

## 当前效果

示例数据 + 离线哈希向量（无 GPU、无 LLM）下的评估结果（`servicekb eval`，18 个问题，k=5）：

| 指标 | 内存存储 | PostgreSQL |
| --- | --- | --- |
| hit@5 | 0.938 | 0.938 |
| MRR@5 | 0.823 | 0.833 |
| nDCG@5 | 0.765 | 0.760 |
| 关键事实召回（must_include） | 0.857 | 0.857 |
| 拒答准确率 | 1.000 | 1.000 |

两个后端的差异来自关键词打分方式：内存实现是标准 BM25，Postgres 是 `ts_rank`（没有 IDF），见 [docs/03_database.md](docs/03_database.md#5-中文全文检索)。"关键事实召回"低于检索指标，主要是离线**抽取式**回答的局限（只能摘原句，摘不到就丢了）；接上 LLM 后由 LLM 组织答案。

唯一失败的 q13（"How to upgrade the firmware of AS-110K-HV without network?"）是**跨语言**问题：英文提问、答案在中文手册里。哈希向量不懂跨语言，换成 bge-m3 即可解决。**这个失败用例被刻意保留在评估集里**，用来说明为什么生产必须用真正的语义向量模型。

> 示例数据很小，这些数字只证明"链路是通的、逻辑是对的"，**不代表真实数据上的效果**。上线前必须用真实工单构建评估集（[docs/05_evaluation.md](docs/05_evaluation.md)）。

---

## 常见任务

**新增一个型号别名 / 故障码**：编辑 `data/sample/dictionaries/entity_aliases.yaml`，重新 `servicekb ingest`。所有文档的实体链接会自动刷新（向量复用，很快）。

**接入新的文件夹**：在 `configs/sources.yaml` 加一段 `type: filesystem`，指定 `path`、`doc_type`、`acl_groups`。

**接入新的系统（Jira、ServiceNow……）**：参考 `connectors/qa_archive.py` 写一个 connector，在 `connectors/__init__.py` 注册。只需要实现 `iter_events()`。

**检索效果不好，怎么排查**：先用 `servicekb search "问题"` 看三路各自的排名：
- 向量路排名靠后 → 换更好的向量模型 / 检查 query prefix
- 关键词路没命中 → 分词问题，把领域词加进词典
- 实体路没命中 → 别名没登记，或料号正则 `SERVICEKB_PART_NUMBER_PATTERN` 不匹配你们的格式

完整的"症状 → 调哪个参数"表见 [docs/04_retrieval_generation.md](docs/04_retrieval_generation.md#调参指南)。

**清洗 Teams 聊天记录**：`servicekb chat clean`，详见 [docs/09_chat_log_cleaning.md](docs/09_chat_log_cleaning.md)。

**物料主数据质量检查**：`servicekb audit parts`（重复料号、相似名称、引用了不存在的料号）。

---

## 文档导航（推荐阅读顺序）

| 顺序 | 文档 | 适合 |
| --- | --- | --- |
| 1 | [00 RAG 入门](docs/00_rag_primer.md) | 完全没接触过 RAG：概念、类比、常见坑 |
| 2 | [01 架构总览](docs/01_architecture.md) | 想知道各模块怎么配合 |
| 3 | [02 数据 pipeline](docs/02_data_pipeline.md) | 要接新数据源、调切块 |
| 4 | [03 数据库设计](docs/03_database.md) | 要运维 / 扩容数据库 |
| 5 | [04 检索与生成](docs/04_retrieval_generation.md) | 要调效果、改 prompt |
| 6 | [05 评估](docs/05_evaluation.md) | 要建评估集、做回归测试 |
| 7 | [06 部署与运维](docs/06_deployment.md) | 要上线：GPU 规划、安全、监控 |
| 8 | [07 开源项目调研](docs/07_repo_survey.md) | 想知道设计决策的来源 |
| 9 | [08 不足与路线图](docs/08_limitations_roadmap.md) | 接下来做什么 |
| 10 | [09 聊天记录清洗](docs/09_chat_log_cleaning.md) | 要处理 Teams / Slack / 企业微信记录 |
| — | [术语表](docs/glossary.md) | 随时查 |

---

## 不足与路线图

当前版本是一个**可以上线试点的骨架**，而不是终点。最重要的几个不足（完整清单和改进方案见 [docs/08_limitations_roadmap.md](docs/08_limitations_roadmap.md)）：

1. **Postgres 关键词检索不是真 BM25**（`ts_rank` 没有 IDF）→ 计划：BGE-M3 稀疏向量（pgvector `sparsevec`）或 ParadeDB `pg_search`
2. **PDF 解析只支持文本层**，扫描件、复杂表格要靠 docling / MinerU（已留接口）
3. **没有图片能力**：现场照片、铭牌、故障屏幕截图还用不上 → 与拍照识别备件（OCR + 图像向量）打通
4. **结构化问题不适合 RAG**（"某型号去年 E037 报了多少次"）→ 需要 Text-to-SQL 路由
5. **离线模式的拒答是启发式**，生产必须开 reranker 并在真实数据上校准阈值
6. **导入是同步单进程**，大规模时需要任务队列（按页拆分 PDF、重试、进度）
7. **SharePoint 权限没有自动同步**，目前按数据源统一配置 ACL
8. **引用只到"块"级别**，还不能高亮到具体句子

---

## FAQ

**Q：为什么不直接用 Dify / RAGFlow / FastGPT？**
A：它们是很好的平台，适合"快速搭一个通用知识库"。但这个场景的核心难点——料号精确匹配、替代件关系、型号别名、工单质量门槛、聊天记录清洗、和物料主数据联动——都需要深度定制，而在平台上做深度定制往往比自己写一个小而清晰的框架更难维护。另外注意许可证：Dify 是修改版 Apache-2.0（限制多租户 SaaS），QAnything 是 AGPL-3.0。详见 [docs/07_repo_survey.md](docs/07_repo_survey.md)。

**Q：为什么不用 LangChain / LlamaIndex？**
A：可以用，但这里刻意没用：RAG 的每一步只有几十到几百行代码，直接写出来，新人能一眼看懂数据怎么流动，出了问题也好排查。框架的抽象层在调试时反而是负担。

**Q：知识库多大会遇到瓶颈？**
A：Postgres + pgvector（HNSW）在百万级块以内表现很好。1024 维 float32 向量每块约 4KB，100 万块 ≈ 4GB 向量 + 约 2 倍的 HNSW 索引。再往上可以用 `halfvec` 减半，或拆到 Milvus / Elasticsearch。见 [docs/03_database.md](docs/03_database.md#容量规划)。

**Q：数据能发给 OpenAI 吗？**
A：这是合规问题，不是技术问题。框架支持完全本地部署（TEI + vLLM），也支持 OpenAI。无论哪种，入库前都会做 PII 脱敏。具体请与法务确认。

---

## 参考与致谢

本项目的设计参考了以下开源项目的源码（2026-10 版本），在此致谢：
[RAGFlow](https://github.com/infiniflow/ragflow) ·
[Dify](https://github.com/langgenius/dify) ·
[LlamaIndex](https://github.com/run-llama/llama_index) ·
[Haystack](https://github.com/deepset-ai/haystack) ·
[R2R](https://github.com/SciPhi-AI/R2R) ·
[QAnything](https://github.com/netease-youdao/QAnything) ·
[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) ·
[kotaemon](https://github.com/Cinnamon/kotaemon) ·
[LightRAG](https://github.com/HKUDS/LightRAG) ·
[GraphRAG](https://github.com/microsoft/graphrag) ·
[docling](https://github.com/docling-project/docling) ·
[MinerU](https://github.com/opendatalab/MinerU) ·
[Unstructured](https://github.com/Unstructured-IO/unstructured) ·
[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) ·
[ragas](https://github.com/explodinggradients/ragas) ·
[pgvector](https://github.com/pgvector/pgvector)

本项目没有复制以上项目的代码。

License: MIT
