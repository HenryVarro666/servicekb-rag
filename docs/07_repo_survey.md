# 07 开源 RAG 项目调研

> 2026-10-31 对以下项目做了浅克隆并**阅读源码**（不只是 README）。星数来自当天的 GitHub 页面（约数）。文件路径相对于各仓库根目录，代码会持续变化，以最新版本为准。
>
> 本项目没有复制以下任何项目的代码，只借鉴设计思路。

## 总览

| 项目 | Stars | 许可证 | 定位 | 本项目借鉴 | 不采用 |
| --- | --- | --- | --- | --- | --- |
| [langgenius/dify](https://github.com/langgenius/dify) | ~160k | **修改版 Apache-2.0**（限制多租户 SaaS、保留 logo）| LLM 应用平台 | 分阶段导入状态机、SQL 预过滤、精选问答优先、引用元数据 | 平台本身、许可证限制 |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | ~93k | Apache-2.0 | 深度文档理解 RAG 引擎 | qa / table / manual 切块模板、任务指纹复用、表格拆分重复表头、乱码字体检测 | 5 组件的部署栈；**HEAD 已改写为 Go**，Python 代码在 v0.27.3 |
| [opendatalab/MinerU](https://github.com/opendatalab/MinerU) | ~83k | Apache-2.0 + **附加条款** | PDF 解析 | 作为独立解析服务、跨页表格合并、LLM 标题分级 | 作为库直接引入 |
| [docling-project/docling](https://github.com/docling-project/docling) | ~70k | MIT | 文档转换 | 统一文档模型、HybridChunker、contextualize（标题只进向量文本）| 默认英文 tokenizer |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | ~53k | MIT | RAG 框架 | 文档哈希 + docstore 策略、变换缓存 | `UPSERTS_AND_DELETE` 的批次删除语义 |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | ~40k | MIT | 图 RAG | 实体抽取 prompt 格式、实体溯源到块、全 Postgres 后端 | 以名字作为实体身份 |
| [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | ~39k | Apache-2.0 | 中文本地知识库 | 中文分句符、`jieba.lcut_for_search`、标题上下文 | 把标题写进正文、每次查询重建 BM25 |
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | ~37k | MIT | 图 RAG | "什么时候值得用图"的判断 | 对本场景成本过高 |
| [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | ~27k | Apache-2.0 | RAG 框架 | 小接口 DocumentStore、DuplicatePolicy、RRF（k=61）| 元数据进入文档 ID |
| [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) | ~26k | Apache-2.0 | RAG 应用 | 引用片段校验 + 页面高亮思路 | 无状态导入、未融合的混合检索 |
| [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | ~16k | Apache-2.0 | RAG 评估 | ID 型指标、`SingleTurnSample` 记录结构 | — |
| [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | ~16k | Apache-2.0 | 文档解析 | 元素类型、`chunk_by_title`、子表检测 | 中文场景的 hi_res 模式 |
| [netease-youdao/QAnything](https://github.com/netease-youdao/QAnything) | ~14k | **AGPL-3.0** | 本地知识库 | 父子块、重排阈值 + 相对跌幅截断 | **代码（AGPL）** |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | ~12k | MIT | BGE 模型 | bge-m3 + bge-reranker-v2-m3 选型 | colbert 多向量存进 Postgres |
| [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) | ~8k | MIT | 生产 RAG 服务 | 单表 chunks + 数组 ACL + GIN、导入状态枚举 | 位置型块 ID、英文 tsvector、5:1 语义偏置 |

> ⚠️ **许可证提醒**
> - **MinerU**：Apache-2.0 之外有附加条款——企业及其关联方月活超过 1 亿或月收入超过 2000 万美元时需要单独的商业授权。中大型设备制造商可能超过收入线，**使用前请法务确认**。
> - **QAnything**：AGPL-3.0，只能借鉴思路，不能复制代码到闭源系统。
> - **Dify**：修改版 Apache-2.0，不能未经许可做多租户 SaaS。
> - **ParadeDB pg_search**（真 BM25 扩展）：AGPL-3.0。

---

## 1. 解析与切块

### RAGFlow（v0.27.3 Python 代码）

- **问答模板** `rag/app/qa.py`：每个问答对一个块，存储为 `问题：q\t回答：a`；**全文索引只用问题**；`rmPrefix` 去掉 "问/Q:/Question:" 前缀；Markdown / DOCX 里用父标题栈作为问题。
- **表格模板** `rag/app/table.py`：一行一块，渲染成 `- 列名: 值`；`column_data_type` 推断列类型并写入带类型后缀的字段，可用于过滤和 SQL；大表按 3000 行拆任务。
- **手册模板** `rag/app/manual.py`：标题层级优先来自 PDF 书签，否则用 `rag/nlp/__init__.py` 的 `BULLET_PATTERN`（第X章、1.2.3、（一）、Chapter）；按 1024 token 合并章节；表格单独成块并附带上下文。
- **PDF** `deepdoc/parser/pdf_parser.py`：OCR → 版面分析 → 表格结构识别；`_is_garbled_by_font_encoding` 检测中文字体编码损坏并回退到 OCR；每个块带页码和坐标，用于引用高亮。
- **增量** `api/db/services/task_service.py`：任务指纹 = 解析配置 + 文件哈希 + 向量模型 + 页范围，相同则复用旧块；块 ID = 文本哈希 + 文档 ID；PDF 按 12 页拆任务。
- **HEAD（Go）**：`internal/ingestion/component/idempotency.go` 按（任务、pipeline 版本、步骤名、步骤版本、输入指纹）做幂等；`chunker/table_split.go` 按整行拆表、不切合并单元格、每块重复表头。
- **注意**：`rag/app/email.py` 保留了所有引用回复，邮件线程会产生大量重复文本——本项目在入库前就去掉了。

→ 本项目：`qa` / `record` / `table` 块类型、表格重复表头、标题编号规则、pipeline_version 指纹、内容哈希块 ID。

### docling

- 文档模型 `docling_core/types/doc/document.py`：带类型的元素（标题、章节标题、表格、列表项、题注、页眉页脚……），每个元素有页码、坐标、字符范围；表格单元格有行列跨度和表头标记。
- `HybridChunker`：用真实 tokenizer 拆分超长元素，合并同一标题下的小元素（`merge_peers`），重复表头；**`contextualize()` 把标题加进向量文本，原文保留用于展示**。
- ⚠️ 默认 tokenizer 是英文的 `all-MiniLM-L6-v2`，中文必须换成向量模型自己的 tokenizer。
- 版面模型把所有标题都标为一级，真实层级靠书签 → 编号 → 字体样式重建。

→ 本项目：`Chunk.embed_text()` = 标题路径 + 正文，`text` 只存正文；PDF 可选 docling 解析。

### MinerU

- 质量分级：flash（只用文本层）→ basic（小 ONNX 模型，CPU 可跑）→ standard / advanced（加 1.2B 视觉语言模型，需 vLLM）。自动判断 PDF 是否需要 OCR。
- 输出 `markdown.md` + content list（文本带 `text_level`、表格带 HTML、图片、公式；每块有 `page_idx` 和 `bbox`）。
- 后处理：跨页表格合并、LLM 标题分级（prompt 要求忽略标题里的指令）。

→ 建议：作为独立服务部署在 48GB 卡上处理扫描件 / 复杂 PDF，输出 Markdown 后走本项目的 Markdown 解析器。**注意许可证附加条款。**

### Unstructured

- 约 40 种元素类型；`id_to_hash` 用文件名 + 文本 + 页码 + 位置生成稳定 ID。
- `chunk_by_title`：遇到 Title 开新块，默认 500 字符上限，表格单独成块。
- ⚠️ 默认分隔符只有 `("\n", " ")`，没有中文标点，中文长段落会被从句子中间切断。

### QAnything（只借鉴思路，AGPL）

- 父子块：父块 800 token（含表格的章节不拆），子块 400 token、25% 重叠；检索子块、返回父块。
- FAQ：只对问题做向量，答案在检索后拼回。
- 两阶段：向量 + 关键词各取 top 30 → 重排 → 丢弃低于 0.28 的结果 → 在分数比最高分下降超过 50% 处截断。

→ 本项目：重排后的绝对阈值 + 相对跌幅截断（`service.py::_drop_weak_tail`）；父子块进入路线图。

### Langchain-Chatchat

- `ChineseRecursiveTextSplitter` 分隔符：`["\n\n", "\n", "。|！|？", "\.\s|\!\s|\?\s", "；|;\s", "，|,\s"]`——英文句号后必须有空白，所以不会切坏 "V1.2"、"PN-40321-A"。
- `zh_title_enhance`：识别短标题，在后续块前加"下文与(标题)有关。"
- 混合检索只在 FAISS 后端有，而且**每次查询都在内存里重建 BM25**；PG 后端只有向量检索；文件变化靠 mtime 判断。

→ 本项目：分句规则（`split_sentences`）、jieba 搜索模式分词；不把标题写进正文。

---

## 2. 存储、检索、增量

### R2R

- chunks 单表：`id, document_id, owner_id, collection_ids UUID[], vec vector(d), text, metadata JSONB, fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text))`，`collection_ids` 上有 GIN 索引。
- 混合检索：两次查询，Python 里加权 RRF：`(w_s/(k+rank_s) + w_ft/(k+rank_ft)) / (w_s+w_ft)`，默认 `rrf_k=50`、**语义权重 5.0、全文权重 1.0**。
- ACL：非超级用户的每个查询都包一层 `owner_id = 我 OR collection_ids && 允许的集合`。
- 导入状态枚举：pending / parsing / chunking / embedding / storing / success / failed，带重试次数。
- 发现的问题：新建表时 GIN 索引建在表达式上，而查询用的是 `fts` 列（只有旧迁移建了列索引）；`ef_search` 声明了但从未 `SET`；块 ID 依赖位置（插入一段后面全变，必须重算向量）；英文 tsvector 对中文无效。

→ 本项目：单表 chunks + 数组 ACL + GIN、所有通道同一 SQL 过滤、显式 `set_config('hnsw.ef_search')`、内容哈希块 ID、三路等权。

### Haystack

- `DocumentStore` 协议只有 4 个方法（count / filter / write / delete）——接口越小越好替换。
- `DuplicatePolicy {NONE, SKIP, OVERWRITE, FAIL}`；pgvector 集成里 OVERWRITE = `ON CONFLICT DO UPDATE`。
- RRF 实现 `k = 61`（60 + 1，因为排名从 0 开始），支持每路权重；另有 `distribution_based_rank_fusion`。
- 内存 BM25 的 tokenizer 把每个汉字单独切成一个 token——简单粗暴的中英混合兜底。
- 文档 ID = `sha256(内容 + 元数据 + …)`：改一个标签就变成新文档（不推荐）。

### LlamaIndex

- `IngestionPipeline` + `DocstoreStrategy {UPSERTS, DUPLICATES_ONLY, UPSERTS_AND_DELETE}`：按文档哈希判断跳过 / 删除重建。
- **`UPSERTS_AND_DELETE` 会删除"本批次没出现"的所有文档**——只在全量快照下正确，增量同步会误删。
- 变换缓存键 = 输入节点内容哈希 + 变换配置，改切块参数只让这一步及之后失效。
- docstore 和向量库分开写，不是原子的。

→ 本项目：增量源只认显式墓碑、全量源有删除比例保护、Postgres 单事务原子替换。

### FlagEmbedding（BGE）

- `M3Embedder`：一次编码输出 dense（1024 维）、lexical_weights（`{token_id: 权重}`，稀疏）、colbert 多向量；默认最大长度 512（支持 8192）；**不需要查询指令前缀**。
- 推荐的混合打分权重 `[0.4, 0.2, 0.4]`（dense / sparse / colbert）。
- `bge-reranker-v2-m3`：默认 `max_length=512`，`normalize=False` 时输出原始 logits（例如 -8.2、5.3），**做阈值前要先 sigmoid**。

→ 本项目：`_sigmoid_if_logits`；稀疏向量进入路线图（pgvector `sparsevec`）。

### Postgres 中文全文检索

- 没有内置中文配置。zhparser（SCWS）、pg_jieba（`jiebacfg` / `jiebaqry`）需要 C 扩展。
- ParadeDB `pg_search`：Tantivy 引擎真 BM25，`USING bm25` 索引，支持 jieba 分词器，AGPL-3.0。
- pgvector：`sparsevec` 最多 16000 个非零元素、HNSW 索引最多 1000；`halfvec` HNSW 最多 4000 维；0.8 起支持 `hnsw.iterative_scan`。

---

## 3. 平台、图、评估

### Dify

- 导入分阶段（解析 → 切分 → 索引），阶段之间检查暂停 / 删除，状态：waiting / parsing / cleaning / splitting / indexing / paused / completed / error，每阶段有时间戳。
- 检索：向量和全文并行，按文档去重保留高分，再用重排模型或加权打分；代码注释（issue #35233）指出**用户的分数阈值不能在融合前作用于原始向量分数**。
- 元数据过滤在 SQL 里先算出 document_ids，再传给向量库；"自动模式"让 LLM 从问题里抽取过滤条件（限于声明过的字段）。
- 权限只到数据集级别（只有我 / 全团队 / 部分成员）——对本场景太粗。
- **Annotation reply**：人工整理的问答按向量匹配，命中阈值时直接返回，不调用 LLM。

→ 本项目：阈值只在重排之后使用；权限只来自身份系统；精选问答进入路线图。

### kotaemon

- 引用：要求 LLM 输出 `CITATION【n】` 和逐字的起止短语，再用 difflib 模糊匹配回原文字符范围，生成带页码和高亮的 PDF 链接。
- 混合检索没有分数融合（全文结果打 -1 分）；去重代码把文档对象和 ID 字符串比较，实际没有去重。

→ 路线图：句子级引用校验。

### LightRAG

- 实体 / 关系抽取 prompt：分隔符行格式、数量上限、完成标记、"gleaning"（补漏抽取）、输出语言参数、可自定义实体类型。
- 查询模式：local（实体）/ global（关系）/ hybrid / naive / mix。
- 每个节点和边都记录来源块 ID；删除文档时从缓存的抽取结果重建受影响的实体。
- Postgres 可以承担全部 4 类存储（KV、向量、图（Apache AGE）、文档状态）。
- 实体按名字精确合并——相似名字会分裂成多个实体（正是本场景的数据问题）。

### GraphRAG

- 标准流程：切块 → LLM 抽图 → Leiden 社区发现 → LLM 生成社区报告 → 向量化。文档说图抽取约占索引成本的 75%。
- 增量更新只追加，不重新聚类，也没有删除——需要定期全量重建。
- 适合"整个语料的主题性问题"（例如"过去一年工单里反复出现的故障模式"），不适合"X 的替代件是什么"这种查找。

→ 结论：先用对象层（实体表 + 关系表，零 LLM 成本）解决 80% 的问题；图抽取只在评估证明有多跳失败时再加，见路线图。

### ragas

| 指标 | 需要 LLM 裁判 |
| --- | --- |
| Faithfulness、ResponseRelevancy、LLMContextPrecision/Recall、FactualCorrectness | 是 |
| IDBasedContextPrecision / IDBasedContextRecall | **否** |
| 非 LLM 的 context precision/recall（字符串相似度）、BLEU / ROUGE / chrF | 否 |

- 测试集生成会先建一个小知识图谱再出单跳 / 多跳题；**100 token 以下的文档会报错**（料号记录、短工单要用 `default_transforms_for_prechunked`）。

→ 本项目：先实现不需要 LLM 的 ID 型指标 + 关键事实召回 + 拒答准确率。

---

## 4. 综合结论

1. **一个 Postgres 足够**（R2R、LightRAG 都验证过），小团队不要上 5 个组件的平台。
2. **中文是一等公民**：几乎所有框架的 Postgres 集成都默认英文全文检索，必须自己处理分词。
3. **标识符需要独立的检索通道**：没有一个通用框架内置了这一点，而这是工业场景最核心的需求。
4. **增量的删除语义是最大的坑**：全量和增量必须区分。
5. **对象层先于知识图谱**：用确定性的实体表 + 关系表解决名称归一和替代关系，比 LLM 抽图便宜、可控、可审计。
6. **阈值只在重排之后使用**（Dify issue #35233、QAnything）。
7. **评估先做不需要 LLM 的指标**，放进 CI。
