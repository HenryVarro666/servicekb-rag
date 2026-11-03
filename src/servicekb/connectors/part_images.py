"""参考图库：每个料号一个文件夹，里面放这个零件的目录照片 / CAD 渲染图 / 标准现场照片。

    data/sample/part_images/
    ├── 310-04521-B/
    │   ├── front.png
    │   └── render_rotated.png
    └── 220-11009-A/
        └── front.png

为什么用"文件夹 = 料号"这种约定？业务人员（备件管理员）不需要懂任何格式，
往对应文件夹里拖图片就行；文件名会作为 view（视角）记录下来。

参考图怎么准备（效果好坏的关键，见 docs/10 第 5 节）：
- 每个料号至少 3~5 张：正面、侧面、**标签特写**、装在机器里的样子；
- 能拿到 CAD 的结构件，可以批量渲染多角度图；
- 长得像的零件（新旧版本、不同额定电压）一定要各自有图，并且标签清晰。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from servicekb.connectors.base import Connector, ImageSource
from servicekb.connectors.parts_master import canonical_part_number
from servicekb.models import SourceEvent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class PartImageConnector(Connector):
    full_snapshot = True

    def __init__(self, source_id: str, path: str | Path, acl_groups: list[str] | None = None,
                 entity_prefix: str = "part") -> None:
        self.source_id = source_id
        self.root = Path(path)
        self.acl_groups = acl_groups or ["public", "service"]
        self.entity_prefix = entity_prefix

    def iter_events(self, state: dict[str, Any] | None) -> Iterator[SourceEvent]:
        return iter(())  # 只产出图片，不产出文档

    def images(self) -> list[ImageSource]:
        if not self.root.exists():
            raise FileNotFoundError(f"source {self.source_id}: path not found: {self.root}")
        out: list[ImageSource] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            entity_id = f"{self.entity_prefix}:{canonical_part_number(folder.name)}"
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS or f.name.startswith("."):
                    continue
                rel = f.relative_to(self.root).as_posix()
                out.append(ImageSource(self.source_id, rel, entity_id, f.read_bytes(), f.resolve().as_uri(),
                                       f.stem, list(self.acl_groups)))
        return out
