# 术语表

| 术语 | 英文 | 解释 | 代码位置 |
| --- | --- | --- | --- |
| 检索增强生成 | RAG, Retrieval-Augmented Generation | 先从知识库检索相关资料，再让大模型依据资料回答 | 整个项目 |
| 块 | Chunk | 文档切分后的小段，检索的最小单位 | `processing/chunking.py` |
| 标题路径 | Heading path | 块所在章节的层级标题，如"手册 > 4 故障排除 > 4.1 E037" | `Chunk.heading_path` |
| 向量 / 嵌入 | Embedding | 把文本映射成一串数字，语义相近的文本向量相近 | `embedding.py` |
| 余弦相似度 | Cosine similarity | 两个向量夹角的余弦，1 表示方向相同；归一化后等于点积 | `store/memory.py` |
| 稠密检索 | Dense retrieval | 用向量相似度检索 | `dense_search` |
| 稀疏检索 | Sparse / lexical retrieval | 按词匹配检索，如 BM25 | `keyword_search` |
| BM25 | Best Matching 25 | 经典关键词打分算法：词频（有饱和）× 逆文档频率 × 长度惩罚 | `store/memory.py` |
| IDF | Inverse Document Frequency | 逆文档频率：越少见的词越重要 | `service.py::keyword_coverage` |
| 分词 | Tokenization | 把句子切成词。中文需要 jieba 等工具 | `processing/text.py::tokenize` |
| tsvector / tsquery | — | Postgres 全文检索的文档表示 / 查询表示 | `store/postgres.py` |
| HNSW | Hierarchical Navigable Small World | 近似最近邻向量索引，多层图结构 | `migrations/001_init.sql` |
| ef_search | — | HNSW 查询时的候选队列长度，越大越准越慢 | `config.py` |
| 混合检索 | Hybrid search | 同时用多种检索方式再融合 | `retrieval/retriever.py` |
| RRF | Reciprocal Rank Fusion | 倒数排名融合：Σ 1/(k + 名次) | `retrieval/fusion.py` |
| 双塔 / 双编码器 | Bi-encoder | 问题和文档分别编码，适合召回 | `embedding.py` |
| 交叉编码器 | Cross-encoder | 问题和文档拼在一起编码，适合重排 | `retrieval/rerank.py` |
| 重排 | Rerank | 用更准的模型对候选重新打分 | `retrieval/rerank.py` |
| 召回 | Recall（阶段） | 检索的第一阶段：快速找出候选 | — |
| 实体 | Entity | 业务对象：型号、料号、故障码，有唯一 ID 和别名 | `processing/entities.py` |
| 实体链接 | Entity linking | 把文本里的提及映射到实体 ID | `EntityLinker.extract` |
| 对象层 | Object layer | 实体 + 关系组成的结构化数据层 | `entities` / `entity_relations` 表 |
| 替代件 | Substitute part | 取代停产料号的新料号 | `substituted_by` 关系 |
| 查询扩展 | Query expansion | 在查询中加入相关词 / 实体（如替代件） | `EntityLinker.expand` |
| 结构化事实 | Structured facts | 从对象层生成、注入 prompt 的事实句 | `EntityLinker.facts_for` |
| 访问控制列表 | ACL | 哪些用户组能看到某个块 | `chunks.acl_groups` |
| 幂等 | Idempotent | 重复执行结果不变 | `pipeline/ingest.py` |
| 增量 | Incremental | 只处理变化的部分 | 同上 |
| 墓碑 | Tombstone | 数据源显式告知"这条已删除"的记录 | `models.Tombstone` |
| 全量快照 | Full snapshot | 每次返回数据源全部内容的同步方式 | `Connector.full_snapshot` |
| Delta 查询 | Delta query | Microsoft Graph 的增量变化查询 | `connectors/sharepoint.py` |
| SimHash | — | 64 位文本指纹，相似文本只差几位，用于近似去重 | `processing/dedup.py` |
| 汉明距离 | Hamming distance | 两个指纹不同位的个数 | `dedup.hamming` |
| MinHash / LSH | — | 另一种近似去重方法，适合大规模 | 见 docs/09 |
| PII | Personally Identifiable Information | 个人身份信息：手机号、邮箱、证件号、姓名 | `processing/cleaning.py` |
| 化名 | Pseudonymization | 用稳定的假名替换真名 | `chat/cleaner.py::Roster` |
| 幻觉 | Hallucination | 模型生成了证据中没有的内容 | — |
| 拒答 / 升级 | Abstain / Escalate | 证据不足时不回答并转人工 | `service.py::_evidence_ok` |
| 引用 | Citation | 答案中的 [n]，指向证据块 | `generation/generator.py` |
| Prompt 注入 | Prompt injection | 数据里的文字试图改变模型的指令 | `generation/prompts.py` 规则 7 |
| 上下文窗口 | Context window | 模型一次能读入的最大长度 | — |
| Hit@k / MRR / nDCG | — | 检索评估指标 | `evaluation.py` |
| 忠实度 | Faithfulness | 答案是否完全由证据支持 | 路线图 |
| 评估集 | Golden set | 带标注的问题集合 | `eval/golden_qa.jsonl` |
| TEI | Text Embeddings Inference | HuggingFace 的向量 / 重排模型推理服务 | `docker-compose.yml` |
| vLLM | — | 高吞吐 LLM 推理服务，提供 OpenAI 兼容接口 | `docker-compose.yml` |
| RSC | Resource-Specific Consent | Teams 资源级授权，由 Team owner 授予应用读取某个 Team 的权限 | docs/09 |
| 会话切分 | Sessionization | 把无线程的聊天流按时间间隔切成会话 | `chat/cleaner.py::build_threads` |
| 对话解缠 | Conversation disentanglement | 把交错的多话题聊天分开 | docs/09 |
