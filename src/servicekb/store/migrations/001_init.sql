-- =============================================================================
-- servicekb 初始表结构
--
-- 设计思路：一个 PostgreSQL 同时承担三种角色
--   1. 关系库：文档、实体、关系、任务、日志 —— 事务、外键、备份都是现成的
--   2. 向量库：pgvector 的 vector 列 + HNSW 索引
--   3. 全文库：tsvector 列 + GIN 索引（中文分词在 Python 侧完成，见 store/postgres.py）
-- 对 3~4 人的小团队，"一个数据库"意味着只需要运维一套东西。
-- 数据量到千万级块、或需要专业 BM25 时再考虑拆分（见 docs/03_database.md）。
--
-- {{EMBEDDING_DIM}} 会在执行迁移时替换成配置里的向量维度。
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 建库元信息：记录向量模型和维度，防止"换了模型却没重建索引"
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 文档层
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,             -- sha256(source_id, external_id)，确定性 ID
    source_id         TEXT NOT NULL,                -- 数据源（sources.yaml 里的 name）
    external_id       TEXT NOT NULL,                -- 源系统里的 ID（文件路径 / 工单号 / item id）
    title             TEXT NOT NULL,
    doc_type          TEXT NOT NULL,
    uri               TEXT NOT NULL DEFAULT '',
    acl_groups        TEXT[] NOT NULL DEFAULT '{public}',
    metadata          JSONB NOT NULL DEFAULT '{}',
    content_hash      TEXT,                         -- 规范化内容的哈希：没变就跳过
    pipeline_version  TEXT,                         -- 切块配置 + 模型 + 代码版本：变了就重建
    status            TEXT NOT NULL CHECK (status IN ('indexed', 'failed')),
    attempts          INT NOT NULL DEFAULT 0,       -- 连续失败次数
    error             TEXT,
    source_updated_at TIMESTAMPTZ,
    indexed_at        TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, external_id)
);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source_id);

-- -----------------------------------------------------------------------------
-- 检索单元：块
-- 注意 doc_type / acl_groups / product_models 是从 documents 冗余过来的（反范式），
-- 这样检索时所有过滤条件都在一张表上完成，不需要 join，HNSW / GIN 索引能直接用。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,               -- sha256(doc_id, 内容哈希, 序号)：内容不变 ID 不变
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    ordinal         INT NOT NULL,
    text            TEXT NOT NULL,
    heading_path    TEXT[] NOT NULL DEFAULT '{}',
    page            INT,
    doc_type        TEXT NOT NULL,
    acl_groups      TEXT[] NOT NULL,
    product_models  TEXT[] NOT NULL DEFAULT '{}',
    entity_ids      TEXT[] NOT NULL DEFAULT '{}',   -- 关联对象：part:/model:/fault:
    content_hash    TEXT NOT NULL,
    simhash         BIGINT NOT NULL DEFAULT 0,
    token_count     INT NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}',
    embedding       vector({{EMBEDDING_DIM}}) NOT NULL,
    search_tsv      TSVECTOR NOT NULL,              -- Python 侧分词后构造，正文权重 A，标题权重 B
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 向量索引 HNSW：
--   m               每个节点的邻居数，越大召回越高、内存越大（默认 16）
--   ef_construction 建索引时的候选队列，越大索引质量越高、建得越慢（默认 64）
--   查询时还有 hnsw.ef_search（默认 40，本项目设 100），见 store/postgres.py
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin       ON chunks USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS chunks_acl_gin       ON chunks USING gin (acl_groups);
CREATE INDEX IF NOT EXISTS chunks_entities_gin  ON chunks USING gin (entity_ids);
CREATE INDEX IF NOT EXISTS chunks_models_gin    ON chunks USING gin (product_models);
CREATE INDEX IF NOT EXISTS chunks_doc_idx       ON chunks (doc_id);

-- -----------------------------------------------------------------------------
-- 对象层：实体 + 别名 + 关系（知识图谱的雏形）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    entity_id       TEXT PRIMARY KEY,               -- part:310-04521-B / model:AS-110K-HV / fault:E037
    entity_type     TEXT NOT NULL,
    canonical_name  TEXT NOT NULL,
    display_name    TEXT NOT NULL DEFAULT '',
    aliases         TEXT[] NOT NULL DEFAULT '{}',
    attributes      JSONB NOT NULL DEFAULT '{}',
    source_id       TEXT NOT NULL,                  -- 由哪个数据源维护（同步时按源整体替换）
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (entity_type);

CREATE TABLE IF NOT EXISTS entity_relations (
    src_entity_id   TEXT NOT NULL,
    relation        TEXT NOT NULL,                  -- substituted_by / interchangeable_with / fits / related_fault
    dst_entity_id   TEXT NOT NULL,
    attributes      JSONB NOT NULL DEFAULT '{}',
    source_id       TEXT NOT NULL,
    PRIMARY KEY (src_entity_id, relation, dst_entity_id)
);
CREATE INDEX IF NOT EXISTS entity_relations_dst_idx ON entity_relations (dst_entity_id);

-- -----------------------------------------------------------------------------
-- 运维：connector 游标、导入任务
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS connector_state (
    source_id   TEXT PRIMARY KEY,
    state       JSONB NOT NULL,                     -- 例如 SharePoint 的 deltaLink
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id       BIGSERIAL PRIMARY KEY,
    source_id    TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'partial', 'failed')),
    stats        JSONB NOT NULL DEFAULT '{}',
    error        TEXT,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

-- -----------------------------------------------------------------------------
-- 线上闭环：查询日志 + 用户反馈（评估集和持续改进的数据来源）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    query_id     UUID PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    question     TEXT NOT NULL,
    user_groups  TEXT[] NOT NULL DEFAULT '{}',
    filters      JSONB NOT NULL DEFAULT '{}',
    retrieved    JSONB NOT NULL DEFAULT '[]',       -- [{chunk_id, doc_ref, score, channel_ranks}]
    answer       TEXT,
    citations    JSONB NOT NULL DEFAULT '[]',
    escalate     BOOLEAN NOT NULL DEFAULT false,
    model        TEXT,
    prompt_version TEXT,
    timings_ms   JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS query_logs_created_idx ON query_logs (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id       BIGSERIAL PRIMARY KEY,
    query_id          UUID NOT NULL REFERENCES query_logs (query_id) ON DELETE CASCADE,
    rating            SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment           TEXT,
    corrected_answer  TEXT,                         -- 专家修正后的答案：可以沉淀为"精选问答"
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
