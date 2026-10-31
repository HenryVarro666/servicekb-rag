# servicekb API / ingestion 镜像
#   docker build -t servicekb .
#   docker run --rm -p 8000:8000 servicekb              # memory 模式体验
# 生产用 docker compose（见 docker-compose.yml）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先只拷依赖声明，利用 Docker 层缓存：代码改动不会触发重新安装依赖
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY data ./data
COPY eval ./eval

# 不要用 root 跑服务
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["servicekb", "serve", "--host", "0.0.0.0", "--port", "8000"]
