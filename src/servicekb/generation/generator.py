"""生成：证据 → 答案（带引用）。

两个实现：
- ``LLMGenerator``：调用 OpenAI 或任何 OpenAI-compatible 接口（vLLM / Ollama / 企业网关）。
- ``ExtractiveGenerator``：不调用大模型，从证据里挑最相关的句子拼成答案。
  用途：离线 demo、CI、以及**大模型服务故障时的降级方案**（至少能给出相关原文）。

引用后处理 ``extract_citations``：LLM 偶尔会引用不存在的编号（例如只给了 5 条证据却写 [7]），
这里统一解析、校验、删除非法编号——引用错误比没有引用更糟糕。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from servicekb.config import Settings
from servicekb.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from servicekb.models import Citation, ScoredChunk
from servicekb.processing.text import split_sentences, tokenize


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, contexts: list[ScoredChunk], facts: list[str]) -> str: ...


class LLMGenerator(Generator):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI

        # max_retries：SDK 内置指数退避重试（429 / 5xx / 超时）
        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=timeout_s, max_retries=max_retries)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai:{base_url or 'openai'}:{model}"

    def generate(self, question: str, contexts: list[ScoredChunk], facts: list[str]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts, facts)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


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


def build_generator(s: Settings) -> Generator:
    if s.llm_backend == "extractive":
        return ExtractiveGenerator()
    return LLMGenerator(
        s.llm_model,
        api_key=s.llm_api_key.get_secret_value() if s.llm_api_key else None,
        base_url=s.llm_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout_s=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )
