# 常用命令。Windows 用户可以直接复制每条命令到终端执行。
.PHONY: install test test-pg lint demo db-up ingest eval serve chat

install:            ## 安装（含开发依赖）
	pip install -e ".[dev]" fpdf2

test:               ## 单元测试（不需要数据库）
	pytest

test-pg:            ## 含 Postgres 集成测试（需要先 make db-up）
	SERVICEKB_TEST_DATABASE_URL=postgresql://servicekb:servicekb@localhost:5432/servicekb pytest

lint:
	ruff check src tests

demo:               ## 离线一键演示
	servicekb demo

db-up:              ## 启动 Postgres + pgvector
	docker compose up -d db

ingest:             ## 导入数据（用 .env 里的配置）
	servicekb ingest

eval:               ## 跑评估集
	servicekb eval

serve:              ## 启动 API，文档 http://localhost:8000/docs
	servicekb serve

chat:               ## 清洗示例 Teams 聊天记录
	servicekb chat clean
