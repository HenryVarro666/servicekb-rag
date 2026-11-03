-- =============================================================================
-- 002：参考图库的图像向量（拍照识别备件，见 docs/10_part_photo_recognition.md）
--
-- 和文本块放在同一个库里：图片 → entity_id（part:xxx）→ 对象层 → 知识库，一条 SQL 就能串起来。
-- 向量维度由图像模型决定（SigLIP so400m = 1152），{{IMAGE_EMBEDDING_DIM}} 在执行迁移时替换。
-- pgvector 的 HNSW 索引最多支持 2000 维的 vector（halfvec 4000 维），主流图像模型都在范围内。
-- =============================================================================

CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id      TEXT PRIMARY KEY,                 -- sha256(source_id, 相对路径)
    source_id     TEXT NOT NULL,
    entity_id     TEXT NOT NULL,                    -- 这张参考图是哪个对象：part:310-04521-B
    uri           TEXT NOT NULL,
    view          TEXT NOT NULL DEFAULT '',         -- front / label / render_rotated ……
    content_hash  TEXT NOT NULL,                    -- 图片字节 + 模型 ID 的哈希：没变就不重算向量
    acl_groups    TEXT[] NOT NULL DEFAULT '{public}',
    metadata      JSONB NOT NULL DEFAULT '{}',
    embedding     vector({{IMAGE_EMBEDDING_DIM}}) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS image_embeddings_hnsw
    ON image_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS image_embeddings_source_idx ON image_embeddings (source_id);
CREATE INDEX IF NOT EXISTS image_embeddings_entity_idx ON image_embeddings (entity_id);
CREATE INDEX IF NOT EXISTS image_embeddings_acl_gin ON image_embeddings USING gin (acl_groups);
