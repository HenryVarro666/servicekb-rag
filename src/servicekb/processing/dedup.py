"""去重：精确重复 + 近似重复。

为什么要去重？历史问答里同一个问题会被反复问（"早上报 E037"可能有几十张工单）。
如果不处理，检索 top-6 里可能有 5 条是同一件事，**挤掉了其他有用的证据**。

本项目的策略（重要设计决策）：
    **入库时不删重复，检索时折叠重复。**
    原因：两张内容相同的工单可能权限不同、机型不同、时间不同；入库时删掉会丢失这些信息（数据血缘）。
    检索时再把相似的结果折叠成一条，并在结果里记录"还有 N 条类似工单"。

近似重复用 SimHash（Google 网页去重用的经典算法）：
    1. 把文本切成特征（中文字符 3-gram + 英文单词）
    2. 每个特征哈希成 64 位，按位投票（该位是 1 就 +1，是 0 就 -1）
    3. 投票结果 > 0 的位记为 1，得到 64 位指纹
    相似文本的指纹只有少数位不同，用汉明距离（不同位的个数）衡量。一般 <= 3 视为近似重复。

更大规模（百万级）时用 MinHash + LSH（ekzhu/datasketch、ChenghaoMou/text-dedup）。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from servicekb.processing.text import normalize_text

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[一-鿿]+")

_MASK64 = (1 << 64) - 1


def _features(text: str) -> Iterable[str]:
    text = normalize_text(text).lower()
    yield from _WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) < 3:
            yield run
        for i in range(len(run) - 2):
            yield run[i : i + 3]


def simhash(text: str) -> int:
    """返回 64 位 SimHash（为了能存进 Postgres BIGINT，转成有符号整数）。"""
    votes = [0] * 64
    n = 0
    for feat in _features(text):
        n += 1
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
        for i in range(64):
            votes[i] += 1 if (h >> i) & 1 else -1
    if n == 0:
        return 0
    value = 0
    for i in range(64):
        if votes[i] > 0:
            value |= 1 << i
    # 转成有符号 64 位整数（Postgres BIGINT 范围）
    return value - (1 << 64) if value >= (1 << 63) else value


def hamming(a: int, b: int) -> int:
    return ((a & _MASK64) ^ (b & _MASK64)).bit_count()


def cluster_near_duplicates(items: list[tuple[str, int]], max_distance: int = 3) -> list[list[str]]:
    """把 (id, simhash) 列表聚成近似重复簇，返回 size>=2 的簇。

    O(n²) 的朴素实现，几万条以内没问题；更大规模请用 LSH 分桶。
    用于 ``servicekb audit qa`` 生成数据质量报告。
    """
    parent = {i: i for i, _ in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hamming(items[i][1], items[j][1]) <= max_distance:
                parent[find(items[i][0])] = find(items[j][0])

    clusters: dict[str, list[str]] = {}
    for i, _ in items:
        clusters.setdefault(find(i), []).append(i)
    return [c for c in clusters.values() if len(c) >= 2]
