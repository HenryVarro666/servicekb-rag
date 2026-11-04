-- =============================================================================
-- 003：查询日志补充运维字段（见 docs/12_production_hardening.md）
--
-- reason     ：拒答 / 降级原因（low_rerank_score、llm_error…）——统计"为什么答不了"
-- degraded   ：哪些依赖失败了但请求仍成功（dense:embedding_failed、rerank:failed…）
-- usage      ：大模型 token 用量 {"input_tokens","output_tokens","calls","models"}——算成本
-- request_id ：和应用日志、响应头 X-Request-ID 对应，串起一次请求的全部日志
-- client     ：哪个调用方（API Key 对应的名字）
-- =============================================================================

ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS reason     TEXT NOT NULL DEFAULT '';
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS degraded   TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS usage      JSONB NOT NULL DEFAULT '{}';
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS request_id TEXT;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS client     TEXT;

-- 日志清理（servicekb db prune-logs）要判断"这条日志有没有反馈"，需要这个索引
CREATE INDEX IF NOT EXISTS feedback_query_id_idx ON feedback (query_id);
-- 按拒答原因 / 调用方统计
CREATE INDEX IF NOT EXISTS query_logs_escalate_idx ON query_logs (created_at) WHERE escalate;
