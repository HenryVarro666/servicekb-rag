from __future__ import annotations

from pathlib import Path

import pytest

from servicekb.config import Settings
from servicekb.logging import setup_logging

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "configs" / "sources.yaml"

setup_logging("WARNING")


def memory_settings(**kw) -> Settings:
    return Settings(store_backend="memory", sources_file=SOURCES, _env_file=None, **kw)


@pytest.fixture(scope="session")
def kb():
    """导入了全部示例数据的 ServiceKB（内存存储），整个测试会话共享一份，节省时间。"""
    from servicekb.service import ServiceKB

    k = ServiceKB(memory_settings())
    k.ingest(SOURCES)
    return k


@pytest.fixture()
def fresh_kb():
    from servicekb.service import ServiceKB

    return ServiceKB(memory_settings())
