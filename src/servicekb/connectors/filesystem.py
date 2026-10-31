"""本地文件夹 / 网络共享盘（NAS、挂载的 SharePoint 同步目录）。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from servicekb.connectors.base import Connector
from servicekb.models import DocType, SourceEvent, SourceRecord
from servicekb.parsing import EXTENSION_MIME

_TEXT_MIME = {"text/markdown", "text/plain"}


class FilesystemConnector(Connector):
    full_snapshot = True  # 每次扫描整个目录 = 全量快照

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        patterns: list[str] | None = None,
        doc_type: str = "manual",
        acl_groups: list[str] | None = None,
        uri_prefix: str = "",
    ) -> None:
        self.source_id = source_id
        self.root = Path(path)
        self.patterns = patterns or ["**/*.md", "**/*.pdf", "**/*.docx", "**/*.xlsx", "**/*.txt"]
        self.doc_type = DocType(doc_type)
        self.acl_groups = acl_groups or ["public"]
        self.uri_prefix = uri_prefix
        self._skipped = 0

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        if not self.root.exists():
            # 目录不存在时直接报错，而不是返回空——否则全量同步会把整个数据源"删光"
            raise FileNotFoundError(f"source {self.source_id}: path not found: {self.root}")
        seen: set[Path] = set()
        for pattern in self.patterns:
            for f in sorted(self.root.glob(pattern)):
                if f in seen or not f.is_file() or f.name.startswith((".", "~$")):
                    continue
                seen.add(f)
                mime = EXTENSION_MIME.get(f.suffix.lower())
                if mime is None:
                    self._skipped += 1
                    continue
                rel = f.relative_to(self.root).as_posix()
                content: bytes | str = f.read_text("utf-8") if mime in _TEXT_MIME else f.read_bytes()
                yield SourceRecord(
                    source_id=self.source_id,
                    external_id=rel,
                    title=f.stem,
                    doc_type=self.doc_type,
                    content=content,
                    mime_type=mime,
                    uri=(self.uri_prefix + rel) if self.uri_prefix else f.resolve().as_uri(),
                    acl_groups=list(self.acl_groups),
                    updated_at=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc),
                )

    def stats(self) -> dict[str, int]:
        return {"skipped_unsupported_files": self._skipped}
