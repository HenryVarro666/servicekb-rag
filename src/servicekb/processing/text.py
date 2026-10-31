"""文本基础工具：规范化、token 估算、分句、中英混合分词。

这些函数被清洗、切块、关键词检索共用。它们看起来琐碎，但**检索质量的很多问题都出在这里**：
全角/半角不统一、零宽字符、"V1.2" 被当成句号切开、料号被分词器切碎……
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

import jieba

# jieba 默认会在首次分词时打印加载日志，这里静默掉
jieba.setLogLevel(60)

# --------------------------------------------------------------------------- 规范化

_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t　\xa0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


# 中文标点在 NFKC 下会被转成半角（"，" → ","、"；" → ";"），既不好看，也会破坏中文分句。
# 这些字符保持原样，其余全角字符照常转换。
_KEEP_FULLWIDTH = frozenset("，！？；：（）")
_NEEDS_NFKC = re.compile(r"[^\x00-\x7f㐀-䶿一-鿿。、“”‘’《》【】—…·]")


def _nfkc_keep_cjk_punct(text: str) -> str:
    return _NEEDS_NFKC.sub(
        lambda m: m.group(0) if m.group(0) in _KEEP_FULLWIDTH else unicodedata.normalize("NFKC", m.group(0)), text
    )


def normalize_text(text: str) -> str:
    """文本规范化，入库前**必须**做。

    - NFKC：全角字母数字转半角（"ＡＳ－１１０Ｋ" → "AS-110K"）、兼容字符统一（"①" → "1"）。
      中文文档里全角字母数字非常常见，不统一的话同一个型号会被当成两个词。
      中文标点（，；：等）保留全角。
    - 去掉零宽字符、控制字符（从 Word / 网页复制来的文本里经常有）。
    - 合并多余空白，但保留换行（换行是段落结构信息）。
    """
    text = _nfkc_keep_cjk_punct(text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def normalize_key(s: str) -> str:
    """生成用于"对象匹配"的键：大写 + 去掉空格 / 连字符 / 下划线 / 点。

    "as-110k hv" / "AS110KHV" / "AS_110K-HV" → "AS110KHV"。
    料号、型号、别名查表都用这个键，解决"名字不统一"的问题。
    """
    s = unicodedata.normalize("NFKC", s).upper()
    return re.sub(r"[\s\-_./]+", "", s)


# --------------------------------------------------------------------------- token 估算

_CJK = r"㐀-䶿一-鿿豈-﫿"
CJK_CHAR = re.compile(f"[{_CJK}]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（不依赖具体模型的 tokenizer）。

    经验值：中文 1 个汉字 ≈ 1 token（bge-m3 / GPT 系 tokenizer 大致如此），
    英文 1 个单词 ≈ 1.3 token。误差 ±20%，对切块足够用了。
    如果需要精确值（例如严格控制 LLM 上下文长度），换成模型自带的 tokenizer。
    """
    cjk = len(CJK_CHAR.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    return cjk + math.ceil(latin * 1.3)


# --------------------------------------------------------------------------- 分句

# 中文句末标点直接切；英文的 . ! ? 后面必须跟空白才切——否则 "V1.2"、"310-04521-B." 会被切坏。
# （借鉴 Langchain-Chatchat 的 ChineseRecursiveTextSplitter）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?])|(?<=[.;])(?=\s)|\n")


def split_sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENT_SPLIT.split(text))
    return [p for p in parts if p]


# --------------------------------------------------------------------------- 分词（关键词检索用）

_STOPWORDS = frozenset(
    """的 了 和 与 或 及 是 在 有 为 对 把 被 从 到 就 也 都 而 并 这 那 个 吗 呢 吧 啊 呀 哪 怎么 怎样 如何 什么
    请问 一下 需要 可以 是否 我们 你们 他们 现在 时候 因为 所以 如果 但是 还是 已经 进行 出现 问题 情况 一个
    the a an of to in on for and or is are was were be been it this that with as at by from how what why
    when which can could should do does did i we you they my our your please hi hello thanks regards""".split()
)
_WORD_OR_CJK = re.compile(rf"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*|[{_CJK}]+")


def add_user_words(words: Iterable[str]) -> None:
    """把领域词（料号名称、故障名称、专业术语）加进 jieba 词典，避免被切碎。

    例如不加的话 "绝缘阻抗" 可能被切成 "绝缘/阻抗"，"直流开关" 切成 "直流/开关"——
    分开也能匹配，但整体词的权重更准确。
    """
    for w in words:
        w = w.strip()
        if w and CJK_CHAR.search(w):
            jieba.add_word(w, freq=20000)


def tokenize(text: str, extra_tokens: Iterable[str] = ()) -> list[str]:
    """中英混合分词，用于关键词检索（BM25 / Postgres 全文检索）。

    - 英文 / 数字 / 标识符：整体保留并转小写，"310-04521-b"、"v2.3.0" 不会被拆开。
    - 中文：用 jieba 的**搜索引擎模式**（lcut_for_search），会同时输出长词和其中的短词，
      "绝缘阻抗低" → ["绝缘", "阻抗", "绝缘阻抗", "低"]，提高召回。
    - 去停用词、去单字符英文。
    - ``extra_tokens``：调用方额外提供的规范化 token（例如实体 ID），直接追加。
    """
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for m in _WORD_OR_CJK.finditer(text):
        piece = m.group(0)
        if CJK_CHAR.match(piece):
            for w in jieba.lcut_for_search(piece):
                if w not in _STOPWORDS:
                    out.append(w)
        else:
            w = piece.lower()
            if len(w) > 1 and w not in _STOPWORDS:
                out.append(w)
    out.extend(t.lower() for t in extra_tokens)
    return out
