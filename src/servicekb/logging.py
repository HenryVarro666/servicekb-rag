"""日志：开发环境人类可读，生产环境输出 JSON（一行一条，方便 ELK / Loki / CloudWatch 采集）。

用法::

    from servicekb.logging import get_logger
    log = get_logger(__name__)
    log.info("ingested", extra={"source": "manuals", "docs": 12})
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

# 当前 HTTP 请求的 ID（API 中间件设置）。同一个请求的所有日志都带上它，排查时 grep 一个 ID 就能串起整条链路；
# 也会写进 query_logs，并通过响应头 X-Request-ID 返回给调用方（用户报障时让对方提供这个 ID）。
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get()
        if rid and not hasattr(record, "request_id"):
            record.request_id = rid
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        return f"{base} {extras}" if extras else base


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        _JsonFormatter() if json_format else _TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger("servicekb")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("servicekb") else f"servicekb.{name}")
