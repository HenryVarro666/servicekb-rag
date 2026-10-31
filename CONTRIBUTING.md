# 贡献指南

## 开发流程

1. 从 `main` 拉分支：`git checkout -b feat/xxx`
2. 改代码 + 写测试。**任何影响检索 / 生成效果的改动，都要在 PR 描述里贴 `servicekb eval` 前后对比。**
3. 本地通过：`make lint && make test`（有数据库的话再跑 `make test-pg`）
4. 提 PR，CI 会跑 lint、测试、评估门禁。

## 改动类型 → 需要同步做的事

| 你改了什么 | 还需要做什么 |
| --- | --- |
| 清洗 / 切块 / 实体链接的逻辑 | `pipeline/ingest.py` 里的 `PIPELINE_CODE_VERSION` +1（触发重建，向量会复用） |
| prompt | `generation/prompts.py` 里的 `PROMPT_VERSION` +1，跑评估 |
| 向量模型 | 必须重建索引：`servicekb db reset --yes && servicekb ingest`，并更新 `EMBEDDING_DIM` |
| 数据库表结构 | 新增 `store/migrations/00X_xxx.sql`，**不要修改已执行过的迁移文件** |
| 新数据源 | 新建 connector + 在 `connectors/__init__.py` 注册 + 在 `configs/sources.yaml` 加配置 + 测试 |
| 评估集 | 新问题追加到 `eval/golden_qa.jsonl`，不要删除旧问题（防止"刷分"） |

## 代码风格

- `ruff check` 通过；行宽 110。
- 注释写"为什么"，不写"做了什么"。面向的读者是刚接手项目的新人。
- 不要提交真实客户数据、聊天记录、密钥。示例数据只能是虚构的。
