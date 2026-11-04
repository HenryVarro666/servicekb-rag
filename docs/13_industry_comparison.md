# 13 行业横向对比与改进路线（2026 年 11 月）

> **一句话结论：先补评估与解析地基，再追平 RAG 行业默认。**
>
> 读完这篇你会知道：成熟的开源 RAG 项目和科技大厂的托管服务现在怎么做、哪些已经是"行业默认"、
> 本项目和它们比领先在哪里 / 落后在哪里，以及接下来按什么优先级改进（P0 / P1 / P2）、哪些事现在**不**该做。
>
> - **怎么做的**：2026-11-04 对 7 个方向做了调研——开源 RAG 平台、检索引擎与混合检索、文档解析与多模态、
>   图谱与 Agentic RAG、大厂托管服务、工业售后 / 现场服务 AI、评估 / 可观测性 / 治理——再逐项对照本仓库的代码和文档。
> - **这是一份快照**：版本号、Star 数、榜单名次变化很快，引用前请到原链接确认最新情况。
> - **数据可信度**：大部分效果数字是厂商 / 作者自报，只有少数学术评测做过独立验证，文中逐项标注；
>   调研时 arXiv、Hugging Face 和多数厂商文档站无法直接访问，部分数字来自搜索引擎摘要（见第五节"可信度"）。
>   文中标注"本报告估算"的数字（置信区间、显存、人周等）是推算，不是测量。
> - **和其他文档的关系**：[docs/07](07_repo_survey.md) 是项目起步时对开源项目的源码调研（设计决策的来源）；
>   本文是更新、更宽的横向对比；[docs/08](08_limitations_roadmap.md) 的路线图已按本文的优先级更新。

servicekb-rag 不需要换架构，需要补"默认值"和"测量"。对照 2026 年 11 月的开源平台、检索引擎和大厂托管服务，它的骨架（PostgreSQL + pgvector 单库，稠密、关键词、实体三路 RRF，对象层，证据门槛拒答，块级 ACL 早绑定）已经覆盖了行业收敛出的大部分"地板"。在标识符精确检索、替代 / 停产结构化事实、拒答并升级这三点上，它还领先多数通用平台，RAGFlow 直到 1.0-rc3 才加入"证据充分性"判定。真正拖后腿的是三件事。第一，评估集只有 18 道虚构题，答对 15 道时 95% 置信区间约为 61%–94%，除了灾难性回归，什么都测不出来。第二，PDF 只读文本层，而 0.9–1.2B 的开源 VLM 解析器在中英文 OmniDocBench 上已经拿到 95–96 分、表格 TEDS 93–95，超过 Gemini 3 Pro。第三，公开评测里收益最大的单项 cross-encoder 重排，代码已经有了，却默认关闭。因此建议：**P0（2026 年底前）**建 200–500 题的真实分层评估集，默认开启重排并校准阈值、让标识符精确命中"置顶"，接入 PaddleOCR-VL / MinerU 版式解析，补上 OIDC 身份校验和高压 / 储能危险作业的回答策略；**P1（2027 年一季度）**在评估门禁下逐项补齐父子块、真 BM25 或稀疏向量、上下文增强、多轮改写与"由门槛触发的一次纠正式重检索"、OTel 追踪、SharePoint 权限同步、KCS 知识闭环和故障码优先入口；**P2** 再按数据就绪程度做 Text-to-SQL、工单 / 序列号上下文、视觉页检索和关系图。现在不该做的是迁移到 RAGFlow / Dify 等平台、更换检索引擎、上完整的 GraphRAG / LightRAG、ColBERT 首阶段索引、默认开启多智能体深度研究和语义答案缓存：独立评测显示图谱方法在单跳事实题上持平或更差，查询成本却高出一到两个数量级。需要提醒的是，支撑这些结论的效果数字大多是厂商自报，只有少数学术评测做过独立验证，而且研究过程中 arXiv、Hugging Face 和多数厂商文档无法直接访问，部分数据来自搜索摘要。

## 一、横向对比：行业已收敛到同一条流水线，差异化落在领域结构

到 2026 年底，开源平台和大厂托管服务的 RAG 流水线已经高度同构，依次是：版式感知解析，结构化切块（常见父子块），稠密 + 关键词 / 稀疏混合检索（用 RRF 或权重融合），带分数阈值的 cross-encoder 重排，元数据过滤，块级或页级引用，外加多轮查询改写。2025–2026 年又出现了两个新的"准默认"：LLM 规划式多跳检索，以及通过 MCP 暴露知识库。仍属差异化的能力有：源系统权限同步、生成后的 grounding 检查、GraphRAG、NL→SQL 和多模态检索。下表把这一格局与 servicekb 的现状并列，出处见后面各小节。

| 能力 | 行业状态 | 代表做法 | servicekb 现状 |
|---|---|---|---|
| 稠密 + 关键词 / 稀疏混合，RRF 或加权融合 | 默认 | Azure RRF k=60；OpenAI 加权 RRF；百炼稠密 + 稀疏；WeKnora pgvector + ParadeDB | 已有三路 RRF k=60；关键词路是 `ts_rank`，三路等权 |
| cross-encoder 重排 + 分数阈值 | 默认 | Azure 语义排序器（前 50 条）；百炼 `RerankMinScore`；百度阈值从 0.2 起调 | 代码已有，默认关闭（`reranker_backend=none`） |
| 版式感知解析、表格结构 | 默认 | Google 版式解析器；AWS BDA；RAGFlow 统一 12 种解析器 | 只读 PDF 文本层 |
| 父子块 / 相邻块扩展 | 默认 | Dify、RAGFlow、WeKnora、Haystack；AWS hierarchical；百度相邻块 | 无 |
| 多轮查询改写 | 普遍提供，多数默认关闭 | 百炼 `conv-rewrite-qwen-1.8b`；火山 3 轮；OpenAI `rewrite_query` | 无 |
| 块级 / 页级引用 | 默认 | Gemini File Search 页级；Claude `search_result` 块级 | 块级 + 引用校验 |
| 链路追踪（Langfuse / OTel） | 开源平台默认 | RAGFlow、Dify、WeKnora 均接入 Langfuse | Prometheus 指标 + 请求 ID，没有 span 树 |
| 规划式多跳检索；知识库经 MCP 暴露 | 2026 年的新默认 | Azure 2026-04 GA；AWS 2026-06 GA；百炼 RAG 4.0；RAGFlow / Onyx / WeKnora 提供 MCP | 无 |
| 证据不足时拒答 | 少数 | RAGFlow 1.0-rc3 证据充分性判定；Google 不可回答查询分类 | **已有，领先** |
| 标识符专用通道 + 对象层事实 | 几乎无人内置 | Qdrant 1.20 新增 `code` 分词器；Milvus 中文分析器会丢弃标识符 | **已有，核心差异化** |
| 源系统文档级 ACL 同步 | 差异化 | Onyx 企业版覆盖 19 个源；AWS 查询时实时复核；Azure SharePoint ACL 预览 | 块级 ACL 有，同步无 |
| 生成后 grounding 检查 / LLM 裁判评估 | 差异化 | Google check grounding；AWS RAG Evaluation | 有生成前门槛和引用校验，没有裁判 |
| LLM 生成的上下文块头 | 被广泛引用，但没有托管默认 | Anthropic Contextual Retrieval | 只有标题路径前缀 |
| GraphRAG / NL→SQL / 多模态检索 | 差异化 | AWS Neptune GraphRAG 与 NL→SQL；百炼 NL2SQL；Gemini 多模态 File Search | 只有对象层；无 SQL；多模态只有拍照找件 |

### 开源平台：父子块和可插拔解析已成标配，代价是不可逆升级与许可证条款

在解析栈上，RAGFlow 是参照实现。它把 DeepDoc、MinerU、Docling、PaddleOCR-VL、Mistral OCR 等后端统一到同一种结构化中间格式。v0.23.0（2025 年 12 月）上线了**父子块**，v0.25.3 给表格列加上"内容 / 元数据 / 主键"角色，v0.25.1 起对 50 页以上的 PDF 分段懒加载解析（[RAGFlow release notes](https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md)）。1.0-rc3 又在 agentic 工具循环里加了一个"证据充分性判定模型"（[RAGFlow releases](https://github.com/infiniflow/ragflow/releases)）。

父子块已经是标配。Dify 建议首次建库就用父子模式（[Dify 文档](https://docs.dify.ai/guides/knowledge-base/create-knowledge-and-upload-documents/2.-choose-a-chunk-mode)），Haystack 的 AutoMergingRetriever 直接支持 pgvector（[Haystack 文档](https://docs.haystack.deepset.ai/docs/automergingretriever)）。

与 servicekb 最接近的开源项目是腾讯 **WeKnora**（MIT，36.4k 星）。它用 pgvector + ParadeDB 做真 BM25，提供父子块、可编辑且带修订历史的切块、内置 MCP Server、Langfuse 追踪，以及召回 + BLEU/ROUGE 评估（[WeKnora](https://github.com/Tencent/WeKnora)）。

FastGPT 有几个小而值钱的做法：记录**重排前的候选**，用来区分"没召回"和"排错了"；喂给重排的是同义词归一后的文本，展示给用户的仍是原文（[FastGPT releases](https://github.com/labring/FastGPT/releases)）。Dify 的 annotation reply 则在相似度超过阈值时直接返回人工审定的答案（[Dify 文档](https://docs.dify.ai/en/cloud/use-dify/monitor/annotation-reply)）。

文档级权限同步方面 Onyx 遥遥领先。它的企业版为 SharePoint、Teams、Confluence 等 19 个源写了权限同步模块，还有 `post_query_censoring`（[Onyx 源码目录](https://github.com/onyx-dot-app/onyx/tree/main/backend/ee/onyx/external_permissions)）。不过这部分代码位于 `ee/` 目录，属于商业版。

代价同样清楚。RAGFlow 1.0 用 Go 全量重写，从 v0.27.2 升级**不可回退**（[RAGFlow release notes](https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md)）。Dify v1.18.0 要求 API、worker、Agent 后端同步升级（[Dify releases](https://github.com/langgenius/dify/releases)）。Onyx 从 Vespa 迁到 OpenSearch 只是"尽力而为"，官方建议重建索引（[Onyx 文档](https://docs.onyx.app/admins/advanced_configs/opensearch_document_index_migration)，仅见搜索摘要）。Haystack 3.4.0 更换默认向量模型，旧向量必须全部重算（[Haystack releases](https://github.com/deepset-ai/haystack/releases)）。

许可证方面：Dify 未经书面授权不得运营多租户环境（[Dify LICENSE](https://github.com/langgenius/dify/blob/main/LICENSE)）；FastGPT 不允许作为 SaaS 提供（[FastGPT](https://github.com/labring/FastGPT)）；QAnything 和 Khoj 是 AGPL-3.0，MaxKB 是 GPL-3.0。维护状态方面：R2R 的最后一次发布很可能在 2025 年 6 月；QAnything 最后一个大版本发布于 2024 年 8 月（[QAnything](https://github.com/netease-youdao/QAnything)）。

RAGFlow 在 v0.27.0 把 GraphRAG 和 RAPTOR 从界面撤下，换成"Knowledge Compilation"。这是对这两项技术用得最重的平台发出的信号（本报告根据更新日志推断，官方没有说明原因）。对一个 3–4 人的团队来说，结论是**借思路，不借依赖**。

### 检索引擎：融合方式已无差别，分水岭在中文 BM25、过滤召回和标识符

所有主流引擎都同时提供基于排名的 RRF 和基于分数的加权融合。Weaviate 自 1.24 起默认 relativeScoreFusion；Qdrant 提供 RRF 和 DBSF，并直言两者"没有普遍的赢家"；Elasticsearch 在 8.18/9.0 加入了 linear retriever；Milvus 3.0.1 加入了加权 RRF（[Qdrant 混合检索文档](https://qdrant.tech/documentation/search-tuning/hybrid-search/)；[Milvus releases](https://github.com/milvus-io/milvus/releases)）。因此，"支持混合检索"本身已经不构成换引擎的理由。

至于融合方式，Bruch 等人发现，调过权重的凸组合在域内、域外都优于 RRF，而且只需要少量标注样本（[arXiv 2210.11934](https://arxiv.org/abs/2210.11934)）。2026 年的研究还提示，把弱 BM25 与强稠密检索等权融合，可能反而拖累结果（[arXiv 2604.03676](https://arxiv.org/pdf/2604.03676)，仅见摘要）。servicekb 用 RRF k=60 是稳健的默认值，与 Azure AI Search 的建议一致（[MS Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)）。但三路等权意味着精确命中料号的实体路在融合里只有"一票"。

pgvector 真正的短板不在规模，而在过滤。最新版本是 0.8.7（2026-10-01），其中 0.8.3 修复了"HNSW vacuum 可能导致索引损坏"的问题（[pgvector CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md)）。近似索引的过滤发生在扫描之后：AWS 的示例中，ef_search=40、过滤只放行 10% 的行时，一次查询只返回约 4 行（[AWS Database Blog](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql)，搜索摘要）。servicekb 已经开启了 `iterative_scan = relaxed_order`，但 ACL 这类与语义无关的过滤条件仍是召回风险最大的地方。

规模方面，EDB 发布的 VectorChord 对比显示，在 1 亿向量上它的 QPS 是 pgvector 的 2–14 倍；但 VectorChord 自己的文档显示，500 万向量时优势只有 +33%–45%（两者都是厂商数据）（[EDB 基准页](https://enterprisedb.com/pgvector-vs-vectorchord-benchmark)；[VectorChord 文档](https://docs.vectorchord.ai/vectorchord/benchmark/pgvector.html)）。售后知识库的体量大约在 10^5–10^7 块，用 `halfvec` 时 64–128 GB 内存就放得下（本报告估算）。

在库内做中文 BM25 已经有可用方案。

| 方案 | 许可证 | 关键能力 | 限制 |
|---|---|---|---|
| **pg_textsearch**（Tiger Data） | PostgreSQL License | 2026-03-27 GA，最新 v1.5.1；v1.1.0 起可直接索引 `text[]`；v1.4.0 起支持 zhparser 中文 | 只支持 PG 17/18；语料统计会把 RLS 隐藏的行算进去 |
| ParadeDB pg_search | 社区版 AGPL-3.0 | 功能最全，内置 jieba | 许可证需法务确认 |
| VectorChord-BM25 | AGPL / ELv2 | — | 官方称"只在英文上测过" |

出处：[pg_textsearch README](https://github.com/timescale/pg_textsearch)；[pg_textsearch releases](https://github.com/timescale/pg_textsearch/releases)；[ParadeDB](https://github.com/paradedb/paradedb)；[VectorChord-bm25](https://github.com/tensorchord/VectorChord-bm25)。

值得注意的是，BGE-M3 论文里 BM25 在 MIRACL 中文集上的 nDCG@10 只有 17.5；在长文档 MLDR 上，学习型稀疏（62.2）反超稠密（52.5）（[arXiv 2402.03216](https://arxiv.org/pdf/2402.03216.pdf)）。也就是说，在中文场景里学习型稀疏向量可能比 BM25 更值得投入。

标识符是另一回事。Milvus 内置的 `chinese` 分析器会**过滤掉含非中文字符的词元**（[Zilliz 文档](https://docs.zilliz.com/docs/chinese-analyzer)），E037、310-04521-B 这类 token 会直接消失。Qdrant 1.20 为此专门新增了 `code` 分词器（[Qdrant releases](https://github.com/qdrant/qdrant/releases)）。通用框架里没有一个内置"标识符通道"，这正是 servicekb 实体路存在的理由。

模型层面，以下数字都来自 Qwen 团队的自报（[Qwen3-Embedding README](https://raw.githubusercontent.com/QwenLM/Qwen3-Embedding/main/README.md)）：

| 模型 | 基准 | 得分 |
|---|---|---|
| Qwen3-Embedding-8B / 4B | C-MTEB | 73.84 / 72.27 |
| BGE-M3 | 多语言 MTEB | 59.56 |
| Qwen3-Reranker-8B / 4B | CMTEB-R（中文重排） | 77.45 / 75.94 |
| bge-reranker-v2-m3（servicekb 当前所用） | CMTEB-R | 72.16 |
| Qwen3-Reranker-0.6B | CMTEB-R | 71.31（中文上并不更好） |

此外，微软的 harrier-oss-v1-27b（MIT）以约 74.3 分领跑 MMTEB v2（[HF 模型卡](https://huggingface.co/microsoft/harrier-oss-v1-27b)）。

选型有两条硬约束。其一，pgvector 的 `vector` 索引上限 2,000 维、`halfvec` 上限 4,000 维（[pgvector README](https://github.com/pgvector/pgvector)），Qwen3-Embedding-8B 的 4,096 维必须做 Matryoshka 截断，4B 的 2,560 维也要用 `halfvec`。其二，jina-embeddings-v5-text-small 是 CC BY-NC 非商用许可（[Jina](https://jina.ai/en-US/models/jina-embeddings-v5-text-small/)）。按 BF16 每参数 2 字节估算，4B 向量模型 + 4B 重排约 16 GB，8B + 8B 约 32 GB，都放得进 48 GB 的卡。

### 解析与多模态：0.9B 的 VLM 解析器在中英文表格上已超过 Gemini 3 Pro

OmniDocBench 是中英文文档解析的主要基准。2026 年 11 月的端到端榜单上，前几名如下（[OmniDocBench](https://github.com/opendatalab/OmniDocBench)）：

| 解析器 | 总分 | 表格 TEDS |
|---|---|---|
| PaddleOCR-VL-1.6（0.9B） | 96.34 | 94.76 |
| MinerU2.5-Pro（1.2B） | 95.75 | 93.42 |
| GLM-OCR（0.9B） | 95.22 | 92.83 |
| Gemini 3 Pro | 92.91 | 89.15 |
| GPT-5.2 | 86.59 | 82.95 |
| Marker | 78.44 | 65.77 |

Docling 不在端到端榜上。

这份榜单有两个盲区需要自己补。一是它由 OpenDataLab 维护，而 MinerU 也出自这家机构；不过排第一的是百度的 PaddleOCR-VL，偏向的嫌疑因此小了很多。二是它没有"技术手册 / 数据手册"这一文档类型，而且从 v1.5 起不再分中英文统计。在偏英文的 olmOCR-Bench 上，Datalab 自己跑出的 Docling 得分只有 50.3（[Marker README](https://github.com/datalab-to/marker)，由竞品自跑）。

现场拍照或扫描的页面方面，PaddleOCR-VL-1.5 在物理重拍的 Real5-OmniDocBench 上保持 92.05%，干净页面上是 94.5%（[arXiv 2601.21957](https://arxiv.org/abs/2601.21957v1)，搜索摘要）。吞吐量约为每块 A100 每秒 1–1.6 页（[arXiv 2603.24326](https://arxiv.org/pdf/2603.24326)，搜索摘要），一万页手册大约需要 2–3 GPU 小时（本报告估算），全库重新解析的成本很低。

许可证格局在 2026 年变了。MinerU 自 3.1.0（2026-04-18）起**不再是 AGPL**，改为以 Apache-2.0 为基础、附加条件的自定义许可：月活超过 1 亿或月收入超过 2,000 万美元时触发授权要求，在线服务须署名；同时移除了 AGPL 许可的 YOLO 组件（[MinerU](https://github.com/opendatalab/MinerU)）。PaddleOCR 的代码是 Apache-2.0（[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)），docling 是 MIT（[docling](https://github.com/docling-project/docling)）。Marker 的代码是 Apache-2.0，但 Surya/Chandra 权重用的是按营收设限的修改版 OpenRAIL-M（[Marker](https://github.com/datalab-to/marker)）。PaddleOCR-VL 权重的许可原文本次未能核实。

切块方面，业界的做法是从解析器输出的 JSON 块树切，而不是从扁平化的 Markdown 切。表格要么作为整体的原子块，要么按行切并重复表头，每块都带章节面包屑（[Marker README](https://github.com/datalab-to/marker)；[MinerU](https://github.com/opendatalab/MinerU)）。servicekb 的切块器已经做到按行拆表、重复表头、加标题路径前缀，缺口在上游：pypdf 给不出表格结构，扫描件没有 OCR，也没有"页码 + 坐标"来支撑点击跳转式引用。

视觉检索方面，ViDoRe V3 显示，在难题上"视觉 top-5 + 文本 top-5"的混合方案达到 54.7%，纯文本是 52.1%；而给文本流水线加上强重排，可提升约 13.2 nDCG（[Emergent Mind 对 arXiv 2601.08620 的摘要](https://www.emergentmind.com/papers/2601.08620)，二手）。对故障码表、规程这类以文本为主的内容，先把解析和重排做好收益更大；视觉检索留给接线图、爆炸图和劣质扫描件。许可证上，ColQwen2/2.5/3 系权重是 Apache-2.0，jina-embeddings-v4 是非商用的 Qwen Research License（[ColPali 仓库](https://github.com/illuin-tech/colpali)；[Jina v4](https://jina.ai/en-US/models/jina-embeddings-v4/)）。

拍照找件的经验同样指向"模型之外的工程"。nyris 把生产环境里 90% 以上的准确率归功于重排、OCR 和 BOM 约束，而它的通用模型 Acc@1 只有 57.6%（[nyris](https://www.nyris.io/blog-posts/beating-google-meta-and-cohere-at-visual-product-search-and-spare-parts-identification)，厂商数据）。在 ILIAS 实例匹配上，SigLIP 中间层特征的 mAP 是 0.50，全局特征只有 0.20（[arXiv 2512.12610](https://arxiv.org/pdf/2512.12610)，搜索摘要）。欧盟 CORDIS 项目的一项 55 人用户研究发现，用视觉搜索找备件比关键词搜索快 3.3 倍、成功率高 51%（[CORDIS](https://cordis.europa.eu/project/id/190121231/reporting)），这是本次找到的最接近独立的证据。读铭牌时，所有模型的中文关键信息抽取得分都只有约 50–63（最高的 Gemini 3.1 Pro 为 63.4）（[Qianfan-OCR, arXiv 2603.13398](https://arxiv.org/pdf/2603.13398)），所以保留专用 OCR、再与 VLM 交叉核对的做法是对的。

### 图谱与 Agentic：图谱方法只在多跳题上领先约 10 分，单跳持平或更差

独立评测的结论相当一致。ICLR 2026 的 GraphRAG-Bench 上（[FalkorDB 复现页](https://falkordb.github.io/GraphRAG-SDK/benchmark/)；[arXiv 2506.05690](https://arxiv.org/abs/2506.05690v3)，数字为二手转引）：

| 方法 | 事实检索题 | 复杂推理题 |
|---|---|---|
| 带重排的普通 RAG | 60.92 | 42.93 |
| HippoRAG2 | 60.14 | 53.38 |
| LightRAG | 58.62 | — |
| 微软 GraphRAG（local） | 49.29 | — |

也就是说，HippoRAG2 只在复杂推理题上领先带重排的 RAG 约 10 分。

Han 等人的系统评测同样发现，RAG 在单跳题和细节题上更好，GraphRAG 只在多跳题上占优，社区摘要式的 global search 总体不如 RAG（[arXiv 2502.11371](https://arxiv.org/abs/2502.11371)）。更关键的是 Zeng 等人的去偏实验：在 GraphRAG 常用的成对 LLM 裁判协议下，LightRAG 跟**它自己**比都能打出 90% 对 10% 的胜负；去掉位置偏差和长度偏差后，普通 RAG 反而略胜 LightRAG（[arXiv 2506.06331](https://arxiv.org/html/2506.06331v1)）。

成本上，GraphRAG-Bench 测得 LightRAG 每题消耗约 10 万 prompt token，GraphRAG global search 约 33 万（[arXiv 2506.05690](https://arxiv.org/html/2506.05690v3)）。售后问答大多是查规格、查规程、查单个故障码含义这类单跳题，图谱方法在这里带不来提升。至于"故障码 → 零件 → 是否停产 → 替代件"这类多跳链条，servicekb 的对象层已经用确定性关系覆盖了相当一部分。

Agentic 方面，可靠的是"自适应"，而不是"一律多步"。CRAG 式的"检索质量评估 + 纠正步骤"在 2026 年一项开源独立复现中提升了 3 个点（54.4% 对 51.4%）（[arXiv 2603.16169](https://arxiv.org/pdf/2603.16169)）。Adaptive-RAG 用一个小分类器决定要不要多步，单题相对耗时 3.60，而一律多步是 8.81（[arXiv 2403.14403](https://arxiv.org/html/2403.14403v2)，2024 年）。Anthropic 的多智能体研究系统，token 用量约为普通对话的 15 倍（[Anthropic Engineering](https://www.anthropic.com/engineering/multi-agent-research-system)）。

查询理解方面，IBM 的 MTRAG-UN（2026 年 2 月）显示：把多轮对话改写成独立问题后，各类检索器的 Recall@5 普遍提升 7–9 个点（例如 0.40 → 0.49），而非独立问题占了数据集的近一半（[arXiv 2602.23184](https://arxiv.org/html/2602.23184v1)）。但在带重排的图文表混合金融文档上，HyDE 反而不如普通稠密检索，对数值类精确查询有害（[arXiv 2604.01733](https://arxiv.org/html/2604.01733v1)）。所以对以料号、故障码为主的查询，改写必须加门控，并且原样保留标识符。

结构化问题上，Text-to-SQL 开源工具在 2026 年洗了一轮牌。Vanna 于 2026-03-29 归档（[Vanna](https://github.com/vanna-ai/vanna)）；Wren AI 转向语义层，行列级安全是商业功能（[WrenAI](https://github.com/Canner/WrenAI)）。VLDB 2026 的一篇论文发现，BIRD Mini-Dev 的标注错误率高达 52.8%（[arXiv 2601.08778](https://arxiv.org/abs/2601.08778)），排行榜本身就不可靠。一项独立配对实验中，加入语义层让准确率提高了 17–23 个百分点（[arXiv 2604.25149](https://arxiv.org/abs/2604.25149)）。方向很清楚：先做参数化、预先验证过的统计模板，再谈自由生成 SQL。

### 评估与治理：18 道题的置信区间宽达 33 个点，统计功效比选什么工具更要紧

评估工具在更替。DeepEval（4.3.2，Apache-2.0）、promptfoo（MIT，自带红队扫描）、Opik、MLflow 都在 2026 年 10–11 月有新版本；RAGAS 自 2026-01-13 以来没有发布新版本，OpenAI Evals 停在 2024 年（[PyPI deepeval](https://pypi.org/project/deepeval/)；[PyPI ragas](https://pypi.org/project/ragas/)；[npm promptfoo](https://registry.npmjs.org/promptfoo)）。

比工具更要紧的是样本量。ICML 2025 的一篇立场论文指出，少于"几百条"的评估集如果用常规误差条，会严重低估不确定性（[ICML 2025](https://icml.cc/virtual/2025/poster/40132)）。功效分析工具给出的量级是：基线通过率 75% 时，要检出 3 个点的配对差异，约需 1,568 题（[abeval](https://pypi.org/project/abeval/)）。servicekb 的 18 道题如果答对 15 道，Wilson 95% 区间约为 61%–94%；即使扩到 300 题、通过率 80%，单个系统的区间仍有约 ±4.5 个点（均为本报告计算）。

LLM 裁判在定义清晰的任务上能达到与人类相当的一致性，GPT-4 与人类偏好的一致率超过 80%（[Zheng et al., NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73434)）。但它有位置偏差、长度偏差和自我偏好，必须先用人工标注校准。

大厂的评估指标可以直接拿来当 schema。AWS Bedrock RAG Evaluation 评估 context relevance、context coverage、citation precision、citation coverage 四项，并支持评估自建流水线的输出（[AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-bedrock-rag-evaluation-generally-available/)）。Google check grounding 按"一句话一个论断"给支持分，引用阈值默认 0.6，延迟低于 500 ms（[Google Cloud](https://cloud.google.com/generative-ai-app-builder/docs/check-grounding)）。

可观测性方面，OTel GenAI 语义约定在 2026 年年中迁到了独立仓库 `semantic-conventions-genai`。它定义了 `retrieval`、`embeddings`、`chat` 等操作，消息内容采集默认关闭，但所有 gen_ai.* 属性仍处于 Development 状态（[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)；[gen-ai-spans.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)）。自托管后端里 Langfuse 规模最大（37k 星，核心代码 MIT），Opik 是 Apache-2.0，Phoenix 是 Elastic-2.0；Helicone 被 Mintlify 收购后已进入维护模式（[GitHub llm-observability](https://github.com/topics/llm-observability)；[DEV](https://dev.to/torrixai/helicone-is-now-in-maintenance-mode-here-is-how-to-switch-to-a-self-hosted-alternative-in-5-4li0)）。

安全与权限方面，没有哪种提示注入防御扛得住自适应攻击：Nasr、Carlini、Tramèr 等人以 71%–100% 的成功率绕过了 12 种已发表的防御（[arXiv 2510.09023](https://arxiv.org/abs/2510.09023)）。但分层防御对非自适应攻击仍然有效：spotlighting（分隔符 + datamarking）在 GPT 系模型上把攻击成功率从 50% 以上压到 2% 以下（[arXiv 2403.14720](https://arxiv.org/pdf/2403.14720)，2024 年）。Meta 自报 Prompt Guard 2 在 AgentDojo 上使攻击成功率降低 57%，而 2026 年的一篇独立论文测得它对触发式注入的零样本召回只有 0.136（[LlamaFirewall](https://arxiv.org/pdf/2505.03574)；[arXiv 2609.22510](https://arxiv.org/pdf/2609.22510)）。

权限的主流做法是"早绑定"：入库时把源系统 ACL 挂到块上，在查询内部过滤。它的主要弱点是同步滞后（[tianpan.co](https://tianpan.co/blog/2026-05-04-permission-aware-retrieval-enterprise-rag-access-control)）。语义答案缓存则是已知的越权泄露通道，缓存键必须带上 ACL 范围（[Redis 文档](https://redis.io/docs/latest/develop/use-cases/semantic-cache/)）；早期的参考库 GPTCache 自 2024 年 8 月以来没有再发布（[PyPI gptcache](https://pypi.org/project/gptcache/)）。

### 大厂托管：重排、改写、规划式多跳都成了"带开关的默认件"

**微软。** Azure AI Search 用 RRF 融合关键词与向量结果，k 建议取 60，可以给向量查询加权；语义排序器是对前 50 条结果重排的 cross-encoder（[MS Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)；[微软白皮书](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/ai-search-query-engine-performance-whitepaper.pdf)）。白皮书自报新排序器比纯混合检索高 22 NDCG@3，而查询改写单独只贡献 +4，且主要在召回低时有效。agentic retrieval 于 2026-04-01 在 REST API 上 GA，知识库可以通过 MCP 端点查询（[MS Learn](https://learn.microsoft.com/en-sg/Azure/search/agentic-retrieval-overview)）。微软自报复杂问题平均提升 16 分，简单问题"持平"（[Tech Community](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/up-to-40-better-relevance-for-complex-queries-with-new-agentic-retrieval-engine/4413832)）；它还专门推出了"auto"推理力度档位，少用 46.9% 的规划 token，仍保住 97.6% 的证据召回（[Tech Community](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/let-the-query-decide-introducing-auto-retrieval-reasoning-effort/4558943)）。

**AWS。** 2026-06-17 GA 的 Bedrock Managed Knowledge Base 自带 SharePoint 等六个连接器、混合检索、重排和 agentic 检索（[AWS What's New](https://aws.amazon.com/about-aws/whats-new/2026/06/amazon-bedrock-managed-knowledge-base/)）。它的 ACL 设计值得照搬：查询时向 SharePoint 复核当前权限，出错或超时就**失败即关闭、不回填**；并且明确声明，在调用方身份未经认证时，这"只是 ACL 感知过滤，不是安全边界"（[AWS 文档](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-acl.html)）。AWS 还宣布 Amazon Q Business 自 2026-07-31 起不再接受新客户（[AWS 文档](https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/qbusiness-availability-change.html)），可见大厂自己的产品线也在震荡。

**Google。** RAG Engine 默认按 1,024 token 切块、重叠 256；Document AI 版式解析器保证块不跨越标题或列表实体；排序服务延迟低于 100 ms，LLM 重排要 1–2 s（[Google 切块](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/fine-tune-rag-transformations)；[版式解析](https://cloud.google.com/vertex-ai/generative-ai/docs/layout-parser-integration)；[重排](https://cloud.google.com/vertex-ai/generative-ai/docs/retrieval-and-ranking)）。

**Anthropic。** 它不卖向量库，但贡献了被引用最多的两项技术。一是 Contextual Retrieval：先为每个块生成 50–100 token 的定位说明，再建向量和 BM25 索引。top-20 检索失败率从 5.7% 降到 3.7%（−35%），加上 BM25 降到 2.9%（−49%），再加重排降到 1.9%（−67%）。Anthropic 还建议，知识库小于 20 万 token 时直接整库放进 prompt（[Anthropic](https://www.anthropic.com/news/contextual-retrieval)）。二是 `search_result` 内容块引用，其中"文本块是最小可引用单元"，块切得越细，引用就越细（[Claude 文档](https://platform.claude.com/docs/en/build-with-claude/search-results)）。本次调研没有发现任何厂商把"LLM 生成上下文块头"做成托管服务的默认功能。

**OpenAI。** file search 用带权重的 RRF 做混合检索，`rewrite_query` 默认关闭，可以设置 `score_threshold`（[OpenAI API 参考](https://developers.openai.com/api/reference/resources/vector_stores/methods/search)）。

国内厂商也在向同一方向收敛。

**阿里百炼。** Retrieve API 做稠密 + 稀疏双路召回，合计 TopK 不超过 200；推荐 `qwen3-rerank-hybrid`，`RerankMinScore` 可以覆盖相似度阈值；多轮改写开关 `EnableRewrite` 默认关闭，用的是 `conv-rewrite-qwen-1.8b`（[阿里云](https://help.aliyun.com/zh/model-studio/api-bailian-2023-12-29-retrieve)）。2026 年 7 月的"RAG 4.0"加入了 Agentic Search、跨库统一重排、NL2SQL，以及 MCP / Skill / CLI 调用（[阿里云创新中心](https://startup.aliyun.com/info/1093745.html)）。

**火山引擎。** 知识库借助多功能向量模型实现混合检索（`dense_weight` 取 0.2–1）；新接口支持基于最多 3 轮历史的改写；`doc_filter` 用到的字段必须在建库时声明（[火山引擎知识检索](https://www.volcengine.com/docs/84313/1254449)；[search_knowledge](https://docs.volcengine.com/docs/vector_database_vikingdb/search_knowledgeNew?lang=zh)）。

**腾讯。** 知识引擎原子能力把解析、切分模型（自报答案完整度 +20%）、多轮改写和重排拆成独立接口（[腾讯云](https://cloud.tencent.com/document/product/1772/115338)）。

**百度。** 千帆给出了场景预设：法务合规场景"开重排、少召回、高阈值"，客服场景"关重排、多召回、极低阈值"；阈值从 0.2 起，按分数分布调整（[百度 AppBuilder](https://ai.baidu.com/ai-doc/AppBuilder/cm8717ef8)）。

**Glean。** 它的连接器同时抓取内容和权限，检索结果"镜像各源系统的访问控制"（[Glean 文档](https://docs.glean.com/connectors/connectors-power-glean)）。

国内厂商的官方文档里都没有找到按用户同步的文档级 ACL，只有标签过滤和工作空间 RBAC。另外，几乎所有厂商的查询改写都默认关闭，微软还做了"auto"推理力度。可见行业把这些能力当作按需触发的升级路径，而不是每次请求都要付的固定开销。

### 工业售后：竞争点转向工单内嵌和"以故障码为键"，效果数字几乎全是自报

领先产品早已不止"和手册聊天"：

| 厂商 / 产品 | 能力 | 出处 |
|---|---|---|
| ABB My Measurement Assistant+ | 为现场技术员提供设备手册、按 NAMUR NE 107 状态信号的**故障码翻译**，以及无需切换系统的备件下单 | [ABB](https://new.abb.com/news/detail/124533/abb-simplifies-industrial-device-maintenance-with-generative-ai) |
| Salesforce Agentforce（Fisher & Paykel 案例） | 在工单内告诉技术员"先查什么"、分步诊断、可直接跳到某一步，只引用"经批准的品牌专属来源" | [Salesforce](https://www.salesforce.com/customer-stories/fisher-and-paykel/agentic-field-service-support/) |
| PTC ServiceMax | 资产服务历史智能体回答某台设备过去出过的问题、解决方案和换过的件 | [PTC](https://www.ptc.com/en/news/2025/ptc-delivers-new-slm-ai-solutions) |
| ServiceNow Now Assist | 只从已解决 / 已关闭的工单，按 KCS 模板起草知识文章，发布前必须审核 | [ServiceNow 文档](https://www.servicenow.com/docs/r/x7YYxCVARboMBfN2zKAZsA/_CSz4yfJG8Svh8O92KusJg) |
| SAP Joule | 用技术员在移动端写的活动小结构建知识库 | [SAPinsider](https://sapinsider.org/blogs/accrete-sap-field-service-asset-management/) |
| Dynamics 365 | 排程智能体只给建议，由调度员确认后应用 | [MS release plan](https://learn.microsoft.com/en-us/dynamics365/release-plan/2024wave2/service/dynamics365-field-service/optimize-schedules-scheduling-operations-agent) |

光伏和储能厂商则以监控云里的遥测诊断为主。Enphase 的 AI 助手在用户登录后读取系统数据和质保状态，并强调自己"不是通用聊天机器人"（[Enphase 新闻稿](https://finance.yahoo.com/energy/articles/enphase-energy-launches-ai-assistant-120000657.html)）。华为 FusionSolar AI agent 能否做服务知识问答尚不清楚。没有一家逆变器厂商公开过达到 FSM 平台深度的技术员知识 copilot。

效果数字几乎都是厂商自报，而且没有基线。西门子称试点平均节省 25% 的被动维护时间，但没有披露试点规模（[Automation.com](https://www.automation.com/en-us/products/march-2025/siemens-industrial-copilot-generative-ai)）。Unisys 报告了 95% 的首次上门修复率，但没有前后对比（[Unisys](https://www.unisys.com/siteassets/collateral/news/nr-042210048-unisys-salesforce-dws.pdf)）。ABB 称首次修复率"最高提升 50%"（[Control.com](https://control.com/news/abb-enhances-device-maintenance-with-updated-digital-solution/)）。本次没有找到任何第三方对照研究，评估 copilot 对首次修复率或 MTTR 的影响。

独立证据反而指向"检索质量是瓶颈"。在航空维修基准 CAMB 上，LLM 做选择题的准确率只有 60%–70%，嵌入模型"偏重语义相似，领域事实的检索准确率有限"（[arXiv 2508.20420](https://arxiv.org/html/2508.20420v1)）。KCS 的收益分三个阶段兑现：3–9 个月、9–18 个月、18–36 个月，被形容为"延迟满足"（[KCS v6 Adoption Guide](https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Adoption_Guide/000_Introduction/020_KCS_Benefits)）。

架构上，这些产品共享三层：一是上下文解析器，按序列号、型号或故障码取出设备信息和服务历史；二是受控的知识检索；三是只读的、或"系统建议、人工确认"的工具。厂商公开的问责机制以流程控制为主，包括限定来源、发布前审核、标识 AI 生成内容、可追溯。**没有一家公开过针对直流高压、储能热失控、上锁挂牌这类危险作业的回答策略**，这部分只能自己设计。

## 二、差距清单：领先在领域结构，落后在默认值、解析和测量

先说该保住的东西。

**实体路和对象层。** 这在行业里是空白。通用框架都没有内置标识符检索通道，部分引擎的默认中文分析器还会主动丢掉料号。把物料主数据里的停产 / 替代关系作为结构化事实注入，比任何平台的"知识图谱"都便宜、可审计。

**证据门槛拒答并标记升级。** 这一点领先于大多数只靠相似度阈值的平台。RAGFlow 到 1.0-rc3 才在 agentic 循环里加入同类判定（[RAGFlow releases](https://github.com/infiniflow/ragflow/releases)），工业厂商的公开材料则几乎不谈拒答。

**其他工程选择。** 在 SQL 里按块级 ACL 过滤，是教科书式的早绑定。增量导入用显式墓碑和删除比例保护，RAGFlow 到 v0.25 才补上对应的删除同步（[RAGFlow release notes](https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md)）。拍照识别把"自信答错 = 0"设为 CI 门禁，与 nyris 的经验一致。在各平台频繁做不可逆升级的背景下，单 Postgres 的选择显得更加合理。

差距集中在下表中，括号里是 [docs/08](08_limitations_roadmap.md) 的既有编号，便于追踪。

| # | 差距 | servicekb 现状 | 行业参照 | 判断 | 路线 |
|---|---|---|---|---|---|
| 1 | 评估规模与方法（E5、G3） | 18 道虚构题；没有忠实度、引用精确度、注入测试；专家修正不会自动沉淀为评测题 | AWS 四项 RAG 指标；Google 论断级 grounding；评估集需要几百条起步 | 严重落后，阻塞其他一切 | P0-1 |
| 2 | 重排默认关闭（G1） | 后端默认 `none`，阈值 0.15 未在真实数据上校准 | 所有大厂默认带重排 + 阈值 | 落后，但代码已有 | P0-2 |
| 3 | 标识符在融合里只有"一票" | 三路等权 RRF | 加权 RRF / 凸组合；没有平台内置标识符置顶 | 优势没用足 | P0-2 |
| 4 | PDF 解析（D1） | pypdf 只读文本层；扫描件无 OCR；表格结构和页内坐标丢失；手册图示没有利用 | VLM 解析器 OmniDocBench 95–96 分 | 落后一代 | P0-3 |
| 5 | 身份校验（E1） | 依赖网关传入 `X-User-Groups` | AWS：身份未经认证时，ACL 过滤不是安全边界 | 安全短板 | P0-4 |
| 6 | 危险作业回答策略 | 没有专门策略 | 厂商都没有公开 | 行业空白，必须自建 | P0-4 |
| 7 | 父子块（R4） | 无 | 平台和大厂普遍提供 | 落后 | P1-1 |
| 8 | 关键词路（R1） | `ts_rank`，没有 IDF | pg_textsearch / ParadeDB / 学习型稀疏 | 落后 | P1-2 |
| 9 | 上下文增强 | 标题路径已进向量文本和 tsvector（B 权重） | Anthropic −35% / −49%（自报） | 与平台持平，落后于前沿 | P1-3 |
| 10 | 多轮、改写、纠正（R3、G4） | 都没有；门槛不通过就直接拒答 | 大厂普遍提供改写和 agentic 检索 | 落后 | P1-4 |
| 11 | 链路追踪（E2） | 有指标、请求 ID，查询日志记录了逐路排名 | Langfuse / OTel span 树 | 部分具备 | P1-5 |
| 12 | 源系统 ACL 同步（D3） | 块级 ACL 有；SharePoint 权限没有同步 | Onyx EE；Azure 预览；AWS 实时复核 | 存储侧持平，同步侧落后 | P1-6 |
| 13 | 知识闭环（D7、§3.6） | 有 `corrected_answer` 字段，但没有沉淀流程 | ServiceNow KCS；Dify annotation reply | 落后 | P1-7 |
| 14 | 故障码优先入口 | 只有自由文本问答 | ABB NE 107；Fincantieri 触发条件字段 | 落后 | P1-8 |
| 15 | 导入、流式、换模型（D4、G5、E6） | 同步单进程导入；没有 SSE；换向量模型要停机 | RAGFlow 大 PDF 分段懒解析；FastGPT 逐条状态 | 落后 | P1-9 |
| 16 | 注入防护 | 只有"证据是数据，不是指令"这一条提示 | datamarking、入库扫描、CI 注入测试集 | 落后（系统低 agency，风险有所缓解） | P1-9 |
| 17 | 拍照识别（D2） | SigLIP-1 全局特征 + 切块，没在真实照片上验证；没有目标检测和条码识别 | 检测裁剪 + 重排 + OCR + BOM 约束 | 未经验证 | P0-1、P1-10 |
| 18 | 结构化统计（D5） | 无 | AWS、百炼 NL→SQL | 落后，依赖数据接入 | P2-1 |
| 19 | 设备 / 工单上下文 | 只能从序列号解码出厂日期 | ServiceMax、Enphase | 落后，依赖 CRM | P2-2 |
| 20 | 图谱、MCP、视觉页检索 | 有对象层；无 MCP；无页图索引 | AWS、百炼、RAGFlow 等 | 暂时不必追 | P2 / 不做 |

后面的路线贯穿两条横切原则。

第一，**标识符保留是每个新组件的验收条件**。分词器、重排器、改写器、解析器都可能悄悄破坏料号。zhparser、pg_jieba 和 Tantivy jieba 怎么切 `310-04521-B` 这类带连字符的标识符，都没有文档说明，必须用 `ts_debug` 或分析器预览实测。而稠密模型无论多大，都很难可靠区分只差一位的料号（本报告推断）。

第二，**证据门槛应该从"终点"变成"路由器"**。行业已经把改写和规划式多跳都做成按需触发，servicekb 已有的门槛判定恰好是最好的触发信号：通过就直接回答；不通过，先做一次纠正式重检索；仍不通过，再拒答升级，同时生成内容缺口工单。

## 三、P0（2026 年底前）：可测量、读得懂文档、默认就准

P0 的入选标准有三条，满足其一即可：后续所有改进都依赖它；公开评测里收益最大且代码已经有了；属于上线前的安全底线。四项合计约 6–10 人周的工程量，外加 50–75 小时专家标注（本报告估算），与仓库现有 Gantt 图里的 P0 窗口吻合。

### P0-1 建 200–500 题真实分层评估集：后面每一项改进都靠它验收

现在的 18 道虚构题只能证明"链路是通的"。按 15/18 估算，Wilson 95% 区间约为 61%–94%。ICML 2025 的立场论文指出，少于几百条的评估集会严重低估不确定性（[ICML 2025](https://icml.cc/virtual/2025/poster/40132)）；要检出 3 个点的配对差异，约需 1,568 题（[abeval](https://pypi.org/project/abeval/)）。

现实的目标是 200–500 题，按问题类型分层：故障码诊断，料号 / 替代 / 停产，规程步骤，参数表查询，跨语言（中文问英文答、英文问中文答），多轮追问，应当拒答，权限隔离（不同用户组），危险作业，以及真实现场照片（沿用"自信答错"指标）。题目优先来自专家修正（`feedback.corrected_answer`）、已关闭工单、Teams 问答对和查询日志里被拒答的问题；合成题只用来补覆盖。每道题都标注金标准答案、金标准证据块和必含事实。

指标方面，在现有的 hit@k、MRR、nDCG、必含事实召回、拒答准确率之外，再增加三类：一是可以和 Anthropic 对比的 1−recall@20 失败率；二是逐路召回；三是沿用 AWS 命名的 context relevance、context coverage、citation precision、citation coverage 以及忠实度（[AWS](https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-bedrock-rag-evaluation-generally-available/)）。

LLM 裁判要用与生成模型不同厂家的模型，做二元判定而不是 1–5 分打分，先在几百条人工标注上测出一致性，再拿来当门禁。检索类指标每个 PR 都跑；裁判类指标每晚跑，与上一版做配对比较，只有置信区间排除零时才判定失败。

同时，查询日志里已有逐路排名（`channel_ranks`），要扩展到重排前的完整候选、重排分数和门槛判定原因（FastGPT 的做法，[FastGPT releases](https://github.com/labring/FastGPT/releases)）。这样出了问题，才分得清是没召回还是排错了。

工具可以用 DeepEval（Apache-2.0）或 promptfoo（MIT，还能顺带跑注入红队），也可以继续用自研的评估模块。不建议把 RAGAS 作为硬依赖，它自 2026-01-13 以来没有发布过新版本（[PyPI ragas](https://pypi.org/project/ragas/)）。

成本主要是专家时间：按每题 10–15 分钟估算，300 题约需 50–75 专家小时，工程约 1–2 人周（本报告估算）。验证的对象是评估集本身的质量：抽 10%–20% 的题做双人标注、看一致性；报告每个指标的置信区间宽度；确认裁判与人工的一致性达到可用水平。实践中常说的 κ 0.6–0.8 只是厂商页面给出的经验值（[Koji](https://www.koji.so/docs/llm-as-a-judge-vs-human-evaluation)）。

### P0-2 默认开启重排，并让标识符命中"置顶"：测得收益最大，代码已有

在所有公开评测中，重排都是收益最大的单步。Anthropic 在"上下文增强 + BM25"的基础上再加重排，失败率从 2.9% 降到 1.9%（[Anthropic](https://www.anthropic.com/news/contextual-retrieval)）；微软自报新排序器比混合检索高 22 NDCG@3（[微软白皮书](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/ai-search-query-engine-performance-whitepaper.pdf)）；所有大厂托管服务都默认带重排和分数阈值。servicekb 已经有 cross-encoder 和 TEI 两个后端，也有"绝对阈值 + 相对跌幅"的截断逻辑，只是默认关闭，而离线启发式门槛又被它自己的文档标为脆弱。

具体做法有四步。第一，在 P0-1 的评估集上对比 bge-reranker-v2-m3 和 Qwen3-Reranker-4B / 8B（0.6B 在中文上不占优，不考虑）。第二，候选数从 40 逐步试到 50–100。第三，按可回答问题与不可回答问题的最高分分布校准 `RERANKER_MIN_SCORE`；百度的建议是从 0.2 起，按分布调整（[百度 AppBuilder](https://ai.baidu.com/ai-doc/AppBuilder/cm8717ef8)）。第四，在同一轮评估里定下向量模型：bge-m3，对比截断到 1,024 或 2,048 维的 Qwen3-Embedding-4B，以及 harrier-oss-v1-0.6b；用 Qwen3 系模型时要加查询指令前缀。

与之配套的是**标识符置顶**：当查询里有已登记的料号或故障码、且实体路精确命中时，把这条命中钉在融合结果前列，并禁止重排把它压到门槛以下。重排的输入用别名归一后的文本，展示仍用原文（FastGPT v4.17.3 的做法）。

成本与风险：4B 重排约占 8 GB 显存，8B 约 16 GB（估算），48 GB 卡可以和向量模型同时驻留。主要风险是 p95 延迟。本次没有找到自托管重排的实测延迟；Google 托管排序服务自报低于 100 ms，可以作为量级参照（[Google](https://cloud.google.com/vertex-ai/generative-ai/docs/retrieval-and-ranking)），实际数值需用 `scripts/loadtest.py` 测。重排服务故障时降级为 RRF 的路径已经存在。

许可证：二手来源把 Qwen3 系标为 Apache-2.0（[supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)），需要到模型卡上确认；harrier-oss-v1-27b 是 MIT；避开 CC BY-NC 的 jina v5。

验证：标识符切片的 hit@1 不得下降，整体 nDCG@5 要提升，画出拒答的精确率 / 召回率曲线，p95 延迟要在预算内。

### P0-3 接入 VLM 版式解析：把故障码表和参数表还原成结构

逆变器手册里被问得最多的就是故障码表、参数表和备件清单，而 pypdf 的文本层恰恰在这里丢掉结构，遇到扫描件则完全失效。

建议以 PaddleOCR-VL-1.5 / 1.6 为主。它在 OmniDocBench 上排第一，支持跨页表格合并和标题层级，对重拍页面最稳，代码是 Apache-2.0。MinerU2.5-Pro 作对照（95.75 分，自定义许可）。docling（MIT）作兜底，负责 DOCX / PPTX 和低风险场景。

三者的输出统一归一到内部的块模式，字段包括：块类型、标题层级、页码、bbox、保留 rowspan / colspan 的表格 HTML、图片题注和裁剪指针。现有切块器在此基础上生成三种东西：一是"章节路径 › 表名 › 表头: 值"式的行级块；二是整表父块；三是把故障码、参数 ID、料号所在的行写进实体层，并关联页码和 bbox。这样就能支持点击跳转到原文区域的引用，与 RAGFlow 的 Position API、kotaemon 的 PDF 高亮做法相同（[RAGFlow releases](https://github.com/infiniflow/ragflow/releases)；[kotaemon](https://github.com/Cinnamon/kotaemon)）。

上线前，先用 50–100 页有代表性的手册页做内部对比，包括合并单元格的故障码表、跨页参数表、带标注的接线图和扫描件。原因是公开基准里没有技术手册这一类型，而且维护方与 MinerU 同属一家机构。

成本：一万页约 2–3 GPU 小时；工程约 2–4 人周（估算）。有一点要注意：vLLM 默认按 `gpu_memory_utilization` 预占显存，论文里 42 GB 级别的显存占用大概率来自这里，而不是模型本身（推断）。所以和向量 / 重排模型同卡时要设上限，或者把解析放到 96 GB 卡上低峰批处理。

许可证：MinerU 只用 3.1.0 及以上版本（更早的版本含 AGPL 组件），提供在线服务时按其许可署名；PaddleOCR-VL 权重的许可需要核实；不用 Marker / Chandra 的权重。

验证：看内部页集上的表格单元格准确率、故障码行召回率，以及评估集"参数表 / 故障码表"分层上的检索指标和答案指标。

### P0-4 安全底线：OIDC、pgvector ≥ 0.8.3 和危险作业回答策略

这三件事成本都不高，但都是上线底线。

第一，API 直接校验 Entra ID / OIDC 的 JWT，从 groups claim 映射用户组。AWS 明确表示，在身份未经认证时，ACL 过滤"不是安全边界"（[AWS 文档](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-acl.html)），而 servicekb 目前直接信任网关传来的 `X-User-Groups`。

第二，把 pgvector 固定在 0.8.3 及以上（推荐 0.8.7），因为 0.8.3 修复了 HNSW vacuum 可能导致的索引损坏（[pgvector CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md)），并在 `servicekb doctor` 里加入版本检查。

第三，为直流侧高压、储能热失控、上锁挂牌等危险作业设计回答策略。做法是：给相关规程打标签；这类问题只允许引用官方手册和服务通告，来源可信度按"服务通告 > 手册 > 工单 > 聊天"排序；如果只有工单或聊天证据，就拒答并升级；回答前置安全提示；必要时用 ACL 限定只有"授权技术员"组能看到。厂商没有公开过可参照的策略，这部分只能自建。验证方法是在评估集里加入危险作业分层，要求 100% 的回答都带官方来源引用，仅凭聊天证据作答的回答数为 0。成本约 1–2 人周（估算）。

| 编号 | 预期收益 | 主要成本 / 风险 | 许可证 | 上线门槛 |
|---|---|---|---|---|
| P0-1 | 5–10 个点以上的变化可以检出；每个失败都能归因到召回或排序 | 50–75 专家小时；裁判偏差 | DeepEval Apache-2.0 / promptfoo MIT | 分层齐全，能报告置信区间，裁判已校准 |
| P0-2 | 行业公认的最大单步收益；标识符题不再被语义相近的料号压下去 | 显存 8–16 GB；p95 延迟 | Qwen3 待核实（二手来源称 Apache-2.0）；harrier MIT | nDCG@5 提升，标识符 hit@1 不降，p95 达标 |
| P0-3 | 表格和扫描件可用；支持页级 / 区域级引用 | GPU 时间分配；内部页集标注 | PaddleOCR Apache-2.0；MinerU ≥ 3.1 为自定义许可；docling MIT | 内部页集表格准确率；表格分层指标 |
| P0-4 | 关闭越权风险和危险作业风险 | 低 | — | JWT 校验上线；pgvector ≥ 0.8.3；危险作业分层 100% 引用官方来源 |

## 四、P1 与 P2：一次只改一处，再按数据就绪程度扩展

### P1（2027 年一季度）：十项改进分三波上线，每项单独过门禁

按前面的功效估算，几百题的评估集只能可靠检出 5–10 个点以上的变化。同时改好几处，回归会互相掩盖。所以 P1 的纪律是：一次只合入一项，对照 P0 基线做配对比较，通过了才合下一项。

| # | 事项 | 依据（证据性质） | 预期收益 | 成本 / 风险 | 许可证 | 验证 |
|---|---|---|---|---|---|---|
| P1-1 | 父子块：用 200–400 token 的子块检索，返回所属的 800–1,000 token 父块或整个小节；`chunks` 表加 `parent_id` | Dify、RAGFlow v0.23、WeKnora、Haystack（支持 pgvector）、AWS hierarchical、百度相邻块（只证明功能普及，没有公开的效果数字） | 答案更完整，对多步骤规程尤其重要 | 低（约 1–2 人周）；上下文 token 增加；AWS 提示父块替换后返回条数可能少于请求数（[AWS](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)） | 自研 | 必含事实召回、忠实度、每个回答的 token 数，配对对比 |
| P1-2 | 关键词路升级 + 加权融合。A 方案：BGE-M3 稀疏向量存入 `sparsevec`（最多 1,000 个非零元素）。B 方案：用 pg_textsearch 索引现有的 jieba 词元 `text[]`。融合权重在评估集上调 | `ts_rank` 没有 IDF；BM25 在 MIRACL 中文上只有 17.5（论文数据）；凸组合优于 RRF（独立研究） | 提升关键词路自身质量，减少弱 BM25 对融合的拖累 | A 方案取决于向量模型选型；B 方案要升级到 PG 17/18、配置 `shared_preload_libraries`，且语料统计包含 RLS 隐藏的行 | pg_textsearch 为 PostgreSQL License；ParadeDB 为 AGPL-3.0，仅在法务同意时考虑；不用 VectorChord-BM25 | 单路 recall@20、融合后 nDCG@5、标识符切片 |
| P1-3 | 上下文增强：为每块生成 50–100 token 的定位说明，只进向量文本和关键词索引，按"文档版本 + 块哈希"缓存，在本地 96 GB 卡上生成 | Anthropic −35% / −49%（自报，2024 年，指标较窄）；没有托管厂商提供默认实现 | 召回失败率下降（方向可信，幅度需自测） | 每块一次 LLM 调用；`pipeline_version` 变化会触发向量重算 | 取决于所用 LLM | 1−recall@20 失败率 A/B |
| P1-4 | 多轮改写 + 门槛触发的纠正重检索。非首轮用小模型改写，继承实体槽位，原问题并行检索、结果取并集，缺关键槽位就反问用户。门槛不通过、或查询里的实体没被证据覆盖时，做一次改写 / 分解后并行重检索 | MTRAG-UN +7–9（独立）；CRAG 复现 +3（独立）；大厂普遍提供但默认关闭；HyDE 对精确查询有害（独立） | 提升追问时的召回；救回一部分错误拒答 | 只在非首轮或失败路径上增加一次小模型调用；风险是把本该拒答的问题也"救"了回来 | 自研 + 本地小模型 | 多轮切片上对比"只用最后一轮 / 改写后 / 取并集"的 R@k；标识符保留率 100%；拒答精确率不下降 |
| P1-5 | 链路追踪 + 在线评估：OTel GenAI span 覆盖 embeddings / retrieval / rerank / chat / 门槛结果，带 prompt、索引、检索配置的版本标签，接入 Langfuse 或 Opik；抽样生产 trace 跑裁判，专家修正自动进入评估集 | RAGFlow、Dify、WeKnora 都接入 Langfuse；规范仍是 Development 状态 | 能定位慢在哪里；A/B 结果可归因；评估集持续增长 | 1–2 人周 + 一套自托管服务；属性名可能变动，要固定 semconv 版本；内容采集保持默认关闭 | Langfuse 核心 MIT；Opik Apache-2.0；Phoenix ELv2 | 100% 请求都有完整 span 树；每周新增的金标准题数 |
| P1-6 | SharePoint 权限同步：用 Graph delta 独立同步 ACL，展开"用户 → 组"，为每个文档记录 ACL 新鲜度时间戳并告警，启用 Postgres RLS，失败即关闭 | Onyx EE 的设计；Azure 增量 ACL（预览）；AWS 实时复核、不回填 | 消除权限陈旧的风险 | 2–4 人周；需要用证书认证的 Entra 应用（[Onyx 文档](https://docs.onyx.app/admins/connectors)）；高选择性 ACL 下 HNSW 召回会下降 | Onyx `ee/` 代码不能复用，只借鉴设计 | 权限变更到生效的时延；按 ACL 类别与精确检索对比召回 |
| P1-7 | KCS 闭环：专家修正和已关闭工单按 KCS 模板（问题 / 环境 / 原因 / 解决）起草，查重、审批后入库；审定答案作为高阈值的精选通道；被拒答的问题生成内容缺口工单；未能链接的新型号写法进入词典审核队列 | ServiceNow 只从已解决工单生成且需审核；Dify annotation reply；KCS 联盟 | 提升知识质量（行业公认的瓶颈） | 主要是流程和专家时间；收益滞后 3–36 个月；ServiceNow 2024 年时曾不做查重（[社区帖](https://www.servicenow.com/community/servicenow-ai-platform-forum/how-do-we-generate-a-knowledge-article-using-ai-with-the/m-p/3207898)） | 自研 | 拒答转化为文章的比例；精选通道的命中率和误命中率 |
| P1-8 | 故障码优先入口：故障码 → 含义 → 按工单频次排序的原因 → 规程 → 备件；支持把事件日志或监控平台导出直接粘贴为查询上下文 | ABB NE 107；Fincantieri 把触发条件与描述分开存（[arXiv 2601.08706](https://arxiv.org/html/2601.08706v1)）；CAMB | 最高频的场景走确定性路径 | 依赖 P0-3 的表格结构化；成本中等 | 自研 | 故障码分层准确率、首次回答耗时 |
| P1-9 | 工程与安全：异步导入队列 + 大 PDF 分页解析 + 逐条状态；换向量模型零停机（`chunks_v2` 双写后切换）；SSE 流式输出（引用最后统一校验）；证据加 datamarking + 入库时扫描注入 + CI 注入测试集 | RAGFlow 对 50 页以上 PDF 分段懒解析；FastGPT 逐条状态；spotlighting 从 50% 以上压到 2% 以下（2024 年）；自适应攻击可绕过所有防御 | 首次全量导入可控；可以换模型；缩短首字时间；注入回归可检出 | 队列可优先基于 Postgres 实现，维持单库运维（本报告建议）；注入分类器的独立召回低，只用来打标记 | promptfoo MIT；garak | 导入吞吐和中断后可恢复性；TTFT p95；注入集通过率 |
| P1-10 | 拍照识别补全：先检测并裁剪零件；条码 / 二维码先做确定性解码；按机型 BOM 约束候选；用中间层特征对 top-k 重排 | nyris 把生产环境 90% 以上的准确率归功于重排 + OCR + BOM（自报）；ILIAS 中间层 0.50 对全局 0.20；CORDIS 用户研究 | 提升找件准确率和速度 | 需要真实照片图库（多角度）；SigLIP 2 / DINOv3 许可证未核实 | RapidOCR Apache-2.0（[PyPI](https://pypi.org/project/rapidocr/)）；PP-OCRv6 Apache-2.0 | 真实照片 top1 / top3；"自信答错"保持为 0 |

P1 内部分三波。

**1 月：P1-5、P1-1、P1-2。** 追踪是之后所有 A/B 归因的前提；父子块和关键词路是对检索地基的直接修补。

**2 月：P1-4、P1-3、P1-8。** 多轮改写要等 Teams 机器人或 Web 前端（[docs/08](08_limitations_roadmap.md) 的 E4）上线后才能体现价值。上下文增强每块都要调用一次 LLM，应在父子块定型之后再做，免得重复计算。故障码入口依赖 P0-3 产出的结构化表格。

**3 月：P1-6、P1-7、P1-9、P1-10。** 权限同步是例外：如果试点范围内 SharePoint 内容的权限粒度细于"数据源级"，就应提前到 P0。

关键词路的两个方案各有隐性成本。pg_textsearch 的许可证最宽松，能直接索引现有的 Python 侧 jieba 词元，保住自定义词典和料号处理逻辑；但它只支持 PG 17/18，而仓库的 [docker-compose.yml](../docker-compose.yml) 用的是 `pgvector/pgvector:pg16` 镜像，选它就意味着先做一次数据库大版本升级。BGE-M3 稀疏向量不需要升级 PG，也不依赖分词器，而且 BGE-M3 论文显示中文 BM25 本来就弱；但如果 P0-2 选了 Qwen3 系向量模型，就要额外跑一遍 bge-m3 编码。建议先做 A 方案（如果 bge-m3 在 P0-2 中胜出，这几乎是免费的），B 方案与 PG 升级一起排期。无论选哪种，精确标识符仍由实体路负责。

### P2（2027 年二季度起）：等工单数据接入、或评估出现对应失败时再启动

| # | 事项 | 启动条件 | 依据 | 要点 / 风险 |
|---|---|---|---|---|
| P2-1 | Text-to-SQL，模板先行 | 维修、收货、现场记录已经可以按对象建模接入 | 语义层带来 +17–23 个百分点（独立）；BIRD 标注错误率 52.8%；Vanna 已归档 | 参数化统计模板作为工具；规则路由（"多少 / 统计 / 趋势 / 比例 / how many" + 已链接实体 → 数据通道）；只读角色 + 视图 + statement timeout + LIMIT + sqlglot 白名单；身份过滤由服务端注入（[sql-guardrail](https://pypi.org/project/sql-guardrail/0.5.2/)） |
| P2-2 | 设备 / 工单上下文 + 嵌入工单流 | 有工单或 CRM 数据接入 | ServiceMax、Enphase、Fisher & Paykel | 按序列号、批次、固件版本取服务历史和适用通告；入口是 Teams 机器人或移动端；同时上线行业指标：升级到总部技术支持的比例、同一序列号的重复上门率、首次回答耗时 |
| P2-3 | 基于对象层的关系增强 | 评估集的多跳切片显著失败 | HippoRAG2 复杂推理 +10（独立）；LinearRAG、LazyGraphRAG 不做关系抽取（自报，[arXiv 2510.10114](https://arxiv.org/abs/2510.10114v1)） | 规范节点只来自对象层；实体共现边不消耗 token；从查询实体出发做 PPR 或 1–2 跳扩展；LLM 只在服务通告、故障码表上做封闭 schema 抽取，每条边带来源块、可信度和有效期（[Graphiti](https://github.com/getzep/graphiti)） |
| P2-4 | 视觉页检索 | 图示类问题失败 | ViDoRe V3 混合方案 54.7 对纯文本 52.1（二手） | ColQwen 系（Apache-2.0）页图索引，与文本索引做 RRF；池化 + int8 后 5 万页约 2.2 GB（估算）；查询端需要常驻 GPU 的 VLM |
| P2-5 | 引导式排障 | 故障码入口和 KCS 内容积累到一定程度 | Fisher & Paykel 的分步诊断；CAMB 的故障树问答 | 由手册 + 已解决工单生成决策树，技术员每一步录入测量值，逐步收窄分支；先离线人工审核 |
| P2-6 | 只读 MCP 服务 | 出现 Copilot / Claude / Teams 智能体等消费方 | RAGFlow、Onyx、WeKnora、Azure、百炼都提供 | 只暴露 search 和 fetch 两个工具，与 OpenAI deep research 对 MCP 的要求一致（[OpenAI](https://developers.openai.com/api/docs/guides/deep-research.md)）；做好鉴权和 SSRF 加固，MaxKB、FastGPT 都修过 MCP 的 SSRF 漏洞（[MaxKB releases](https://github.com/1Panel-dev/MaxKB/releases)） |
| P2-7 | 成本优化 | 调用量上来之后 | Anthropic prompt caching 自报成本 −90%（[Anthropic](https://www.anthropic.com/news/prompt-caching)） | 按任务或难度路由模型；固定前缀做 prompt caching，它缓存的是前缀的计算结果而不是答案，所以不涉及 ACL 问题 |
| P2-8 | 视觉骨干升级 | P1-10 之后，真实照片上仍有系统性错误 | DINOv3 / SigLIP 2；CALIPER 冻结骨干 + 轻量头（[arXiv 2609.17820](https://arxiv.org/pdf/2609.17820)） | 每个零件只有 5–20 张图时，只训练线性头 / 原型头或 LoRA，避免全量微调 |

P2 有一个共同前提：数据就绪。Text-to-SQL 和设备上下文都依赖 servicekb 目前还没有的工单 / CRM 数据接入。而这条数据通道一旦打通，还能顺带拿到工业界真正衡量效果的指标，也就是首次修复率、升级率、重复上门率。图谱和视觉检索则要等评估集拿出对应的失败证据才启动，而不是因为"别人都有"。

## 五、不做清单与证据可信度

### 现在不做的十件事：多数在单跳事实题上没有收益，成本却高一个数量级

| 不做 | 理由（证据） | 何时重新考虑 |
|---|---|---|
| 迁移到 RAGFlow / Dify / FastGPT / Onyx | 不可逆升级、多服务联动升级；许可证限制（Dify 多租户、FastGPT SaaS、Onyx 权限同步属 EE）；对象层和标识符通道在平台上很难深度定制 | 需求转向"面向多部门的通用知识库平台"时；需要可视化编排时，让平台调用本项目的 API（与 [docs/08](08_limitations_roadmap.md) 的既有立场一致） |
| 更换或新增检索引擎（Milvus / ES / OpenSearch / Vespa / Qdrant），或上 VectorChord / pgvectorscale | 规模在 10^5–10^7 块，pgvector + halfvec 足够；VectorChord 在 500 万向量时自报优势只有 +33%–45%，且是 AGPL / ELv2；Onyx 迁移的教训 | 单节点超过约 1,000–2,000 万向量、索引放不进内存；或者高选择性的 ACL 组合下，召回无法靠精确扫描兜底 |
| 完整的 GraphRAG / LightRAG（LLM 全量抽取 + 社区摘要） | 单跳事实题上持平或更差；LightRAG 每题约 10 万 token；成对裁判胜率存在 90/10 的"自己比自己"假象；RAGFlow 已从界面撤下 | 评估集多跳切片显著失败，且 P2-3 的轻量方案不够用 |
| ColBERT / 多向量首阶段索引 | 存储约为稠密向量的 10–15 倍；Postgres 里只有 AGPL 的 VectorChord 支持；jina-colbert-v2 可能是非商用许可；cross-encoder 重排已经拿到大部分收益 | 超长文档或页图检索（P2-4）证明确有必要时 |
| 默认开启深度研究 / 多智能体 / RL 搜索智能体 | 多智能体的 token 用量约为 15 倍，延迟达到分钟级；RL 训练需要专门的基础设施，且面向开放网页；连微软都做了 auto 推理力度 | 作为离线的"深度排查"模式，用于对工单和聊天记录做根因分析 |
| 对标识符查询做 HyDE 或无条件 multi-query | 在带重排的精确数值查询上 HyDE 有害（独立研究）；改写可能篡改料号 | 只对症状描述类、多实体问题加门控开启 |
| 语义答案缓存 | 是已知的越权泄露通道；相似但型号或固件不同的问题会命中错误答案；当前 QPS 不需要 | 调用量和成本成为瓶颈时，先做按 ACL 组隔离的精确匹配缓存，键里包含索引版本和 prompt 版本 |
| 自由形式 Text-to-SQL | 基准标注错误率超过 50%，排行榜不可靠；Vanna 已归档；Wren 的行列级安全是商业功能 | 出现模板化统计（P2-1）覆盖不了的长尾分析需求时 |
| 用成对胜率式 LLM 裁判做验收，或把 RAGAS 当硬依赖 | 存在 90/10 的自比假象；RAGAS 自 2026 年 1 月起没有新版本 | 不适用，应改用经人工校准的二元判定 |
| 引入受限许可证的组件 | Onyx `ee/` 代码、QAnything / Khoj（AGPL）、MaxKB（GPL）；MinerU 3.1.0 之前的版本（含 AGPL 的 YOLO）；营收超过门槛时的 Marker / Chandra 权重；jina v5 small、jina-embeddings-v4（非商用）；Nougat、GOT-OCR（仅限研究用途） | 法务书面确认之后 |

### 可信度：独立验证只覆盖少数学术评测，工业效果数字全部自报

本报告里起决定作用的数字，可以按证据性质分为三类。

| 结论 | 关键数字 | 证据性质 |
|---|---|---|
| 重排是收益最大的单项 | Anthropic 失败率 −49% → −67%；Azure +22 NDCG@3 | 厂商自报，2024 年，各家指标不同，不可横向比较 |
| 上下文增强有效 | 失败率 −35% / −49% | 厂商自报（Anthropic），只衡量 top-20 失败率 |
| 规划式多跳只在复杂题上赢 | 复杂题 +16，简单题持平；auto 档少用 46.9% token | 厂商自报（微软） |
| Qwen3 模型优于 BGE 系 | C-MTEB 72–74；CMTEB-R 76–77 对 72 | 模型方自己跑的 |
| VLM 解析器领先 | OmniDocBench 95–96，TEDS 93–95 | 公开基准，但维护方与 MinerU 同属一家机构，且没有技术手册类型 |
| 图谱方法只在多跳题上赢，单跳持平 | GraphRAG-Bench；Han 等；Zeng 等的 90/10 | 独立学术评测（部分数字经复现页或搜索摘要转引） |
| 多轮改写有效 | Recall@5 +7–9 | 独立基准（MTRAG-UN） |
| 纠正式检索有效，但幅度小 | +3 个点 | 独立复现（原论文自报 +4.4 至 +7） |
| 注入防御都能被绕过 | 12 种防御被以 71%–100% 的成功率绕过 | 独立 |
| 评估集需要几百题起步 | 检出 3 个点差异约需 1,568 道配对题 | 独立（立场论文 + 功效分析工具） |
| 语义层能改善 Text-to-SQL | +17–23 个百分点 | 独立；dbt 的 11 题实验属于厂商数据 |
| 工业 copilot 的效果 | 西门子 25%、Unisys 95%、ABB 最高 50%、SAP 50% / 75% | 全部厂商自报，没有基线或对照 |
| 视觉找件有效 | 快 3.3 倍，成功率高 51% | 欧盟项目的用户研究，最接近独立 |

研究笔记本身还受几项数据来源限制。**网页访问**上，arXiv、Hugging Face、learn.microsoft.com、docs.aws.amazon.com、help.aliyun.com、火山引擎、腾讯云、glean.com、docs.onyx.app、ragflow.io 等站点无法直接抓取，相关内容来自搜索引擎摘要；工业售后部分的页面抓取全部失败，内容全部来自搜索摘要，引用其中的具体数字前应先复核。**GitHub 数据**上，API 受限，星标数来自渲染后的页面，多数发布日期的年份是推断出来的。**许可证**上，Qwen3-Embedding、PaddleOCR-VL 权重、DeepSeek-OCR 2、SigLIP 2 / DINOv3 的许可都没能核实。**基准覆盖**上，没有任何基准覆盖售后 / 逆变器语料、料号密集的中英混合查询、中文 GraphRAG 或中文多轮 RAG，父子块、表格行级切块、标题面包屑也都没有找到受控实验的效果数字。**工业效果**上，没有找到任何独立研究衡量现场服务 copilot 对首次修复率或 MTTR 的影响。因此，本报告里所有的"预期收益"都只是方向判断，具体幅度必须在 P0-1 的评估集上自己测。

## 结论

servicekb 的护城河不是任何一项通用 RAG 技术，而是对象层和标识符通道。偏偏是那些"行业默认"组件，比如中文分析器、HyDE 式改写、语义缓存、纯语义融合，在默认配置下最容易侵蚀这条护城河。所以引入行业默认能力时，真正的工作量不在接入本身，而在"保住标识符"的测试和适配上。这也意味着，评估集里的标识符切片应当和 nDCG 一样，是每次合入的一票否决项。

2026 年，行业整体从"一律多步"转向"按需升级"：改写默认关闭，推理力度有 auto 档，Adaptive-RAG 和 CRAG 得到复现。这让 servicekb 的证据门槛不再只是防幻觉的刹车，而成为整个系统的调度中枢：由它决定何时直接回答、何时纠正后重检索、何时转人工，以及哪些问题应该变成知识缺口工单。工业界的独立证据一致指向"知识质量才是瓶颈"，而 KCS 的收益要 3–36 个月才能兑现。因此最值得尽早埋点的不是模型效果，而是两条业务曲线：被拒答的问题转化为新知识的速度，以及升级到总部技术支持的比例。这两条曲线才能证明这个系统对售后业务的真正价值。
