"""生成：证据 → 答案（带引用）。

两个实现：
- ``LLMGenerator``：调用大模型。厂家无关——OpenAI / Azure / Claude / Gemini / DeepSeek / 通义 /
  本地 vLLM……都通过 ``generation/llm.py`` 的统一接口 ``ChatModel`` 调用，可配置主备切换（见 docs/11）。
- ``ExtractiveGenerator``：不调用大模型，从证据里挑最相关的句子拼成答案。
  用途：离线 demo、CI、以及**大模型服务故障时的降级方案**（至少能给出相关原文）。

引用后处理 ``extract_citations``：LLM 偶尔会引用不存在的编号（例如只给了 5 条证据却写 [7]），
这里统一解析、校验、删除非法编号——引用错误比没有引用更糟糕。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from servicekb.config import LLMSpec, Settings
from servicekb.generation.llm import ChatModel, build_chat_model
from servicekb.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from servicekb.models import Citation, ScoredChunk
from servicekb.processing.text import split_sentences, tokenize


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, contexts: list[ScoredChunk], facts: list[str]) -> str: ...


class LLMGenerator(Generator):
    """把 RAG 的提示词交给任意厂家的 ``ChatModel``。重试、主备切换都在 ChatModel 里完成。"""

    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    @property
    def name(self) -> str:  # 主备切换时，记录的是实际回答的那一家
        return self.chat_model.name

    def generate(self, question: str, contexts: list[ScoredChunk], facts: list[str]) -> str:
        return self.chat_model.complete(SYSTEM_PROMPT, build_user_prompt(question, contexts, facts)).strip()


class ExtractiveGenerator(Generator):
    name = "extractive"

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[ScoredChunk], facts: list[str]) -> str:
        q = set(tokenize(question))
        sents: list[tuple[int, str, set[str]]] = []
        seen: set[str] = set()
        for i, sc in enumerate(contexts, 1):
            for s in split_sentences(sc.chunk.text):
                key = s.lstrip("-> ").strip()
                if len(key) < 6 or key in seen:  # 太短的、重复的句子不要
                    continue
                seen.add(key)
                sents.append((i, s, set(tokenize(s))))
        # 局部 IDF：在候选句子里到处出现的词（例如型号名）权重低，少见的词（例如 E077）权重高
        n = len(sents)
        df: dict[str, int] = {}
        for _, _, toks in sents:
            for t in toks & q:
                df[t] = df.get(t, 0) + 1
        scored: list[tuple[float, int, str]] = []
        for i, s, toks in sents:
            overlap = sum(math.log((n + 1) / (df[t] + 1)) + 1 for t in toks & q)
            scored.append((overlap + 0.5 / i, i, s))  # 排名靠前的证据略加分
        scored.sort(key=lambda x: -x[0])
        picked = sorted(scored[: self.max_sentences], key=lambda x: x[1])
        lines = ["（离线抽取模式：未调用大模型，以下为证据原文摘录）"]
        lines += [f"- {f}" for f in facts]
        lines += [f"- {s.lstrip('- ')} [{i}]" for _, i, s in picked]
        return "\n".join(lines)


_CITE = re.compile(r"[\[【]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】]")


def extract_citations(answer: str, contexts: list[ScoredChunk]) -> tuple[str, list[Citation]]:
    """解析答案中的 [n] / 【n】 / [1, 2]；删掉越界编号；返回 (清理后的答案, 引用列表)。"""
    used: list[int] = []

    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"[,，、\s]+", m.group(1)) if x]
        valid = [n for n in nums if 1 <= n <= len(contexts)]
        for n in valid:
            if n not in used:
                used.append(n)
        return "".join(f"[{n}]" for n in valid)

    cleaned = _CITE.sub(repl, answer)
    cites = []
    for n in sorted(used):
        c = contexts[n - 1].chunk
        cites.append(
            Citation(index=n, chunk_id=c.chunk_id, doc_ref=c.doc_ref, title=c.title, uri=c.uri,
                     heading_path=c.heading_path, page=c.page, snippet=c.text[:200])
        )
    return cleaned, cites


def build_generator(s: Settings, spec: LLMSpec | None = None) -> Generator:
    """``llm_provider=extractive``（默认）→ 抽取式；否则按配置构造对应厂家的大模型。"""
    chat_model = build_chat_model(s, spec)
    return ExtractiveGenerator() if chat_model is None else LLMGenerator(chat_model)
