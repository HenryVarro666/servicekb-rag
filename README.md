# servicekb — a RAG knowledge-base framework for industrial after-sales & field service

**English** | [简体中文](README.zh-CN.md)

> Turn large, messy service data — **product manuals, service bulletins, ticket archives, the parts master, Teams chat logs** — into a knowledge base that answers field technicians' questions **with citations, respects access control, and says "I don't know" when the evidence isn't there.**

- 🧱 **One database**: PostgreSQL + pgvector is the relational store, vector store and full-text index at once. A 3–4 person team can run it.
- 🔀 **Three-lane hybrid retrieval**: dense vectors (semantics) + keywords (mixed Chinese/English tokenization) + **entities** (model / part number / fault code), fused with RRF, optional cross-encoder reranking.
- 🧩 **Object layer**: turns record-oriented data into object-oriented data — model-name aliases unified, part substitutions and obsolete status modeled — and uses it directly in retrieval and answers.
- ♻️ **Production-grade ingestion**: incremental, idempotent, embedding reuse, fault isolation, safe deletes, every run auditable.
- 🔐 **Access & compliance**: chunk-level ACLs, PII redaction, automatic refusal + escalation when evidence is insufficient.
- 💬 **Chat-log cleaning**: Teams export → thread reconstruction → Q&A extraction → human-review routing ([guide](docs/09_chat_log_cleaning.md), Chinese).
- 📏 **Built-in evaluation**: retrieval hit@k / MRR / nDCG + key-fact recall + refusal accuracy, wired into CI as a regression gate.
- 🎓 **Written for newcomers**: every module opens with an explanation of what the step does in a RAG system and why.

**Zero-dependency trial**: no GPU, no API key, no database — the full pipeline runs in 5 minutes.

> ⚠️ Everything under `data/sample/` (the ACME Solar brand, models, part numbers, fault codes, thresholds) is **fictional**, for demonstration only. It does not describe any real product and must not be used to service real equipment.
>
> 📖 The in-depth docs under `docs/` and the code comments are written in **Chinese**; this README gives an English overview.

---

## Contents

- [Quick start (5 minutes)](#quick-start-5-minutes)
- [How it works](#how-it-works)
- [Using PostgreSQL](#using-postgresql)
- [Plugging in real models (GPU / OpenAI)](#plugging-in-real-models-gpu--openai)
- [Project layout](#project-layout)
- [Key design decisions](#key-design-decisions)
- [Current results](#current-results)
- [Common tasks](#common-tasks)
- [Documentation map](#documentation-map)
- [Limitations & roadmap](#limitations--roadmap)
- [FAQ](#faq)
- [References](#references)

---

## Quick start (5 minutes)

Requires Python 3.10+.

```bash
git clone https://github.com/HenryVarro666/servicekb-rag.git && cd servicekb-rag
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
servicekb demo
```

It ingests the sample data (16 part numbers, 3 manuals, 2 service bulletins, 21 tickets — 2 of which have no resolution and are dropped by the quality gate) and answers four questions. Excerpt (the sample data is mostly Chinese):

```text
──────────────── 310-04521-A 还能订到吗？ (Can I still order 310-04521-A?) ────────────────
- 料号 310-04521-A（散热风扇组件）状态：停产；已被 310-04521-B 替代 …   ← structured fact from the object layer
                                                                         ("obsolete; replaced by 310-04521-B")
- 所有新的风扇更换工单一律使用 310-04521-B，不再发 310-04521-A 的库存。 [3]
- 自 2025 年 6 月 1 日起 … 旧料号 310-04521-A 停产，由新料号 310-04521-B 替代。 [4]
  [3] bulletins:SB-2025-014_fan_module_change.md  3 服务要求
  [4] bulletins:SB-2025-014_fan_module_change.md  1 变更内容
──────────────── 逆变器支持接入风力发电机吗？ (Does the inverter support wind turbines?) ────────────────
知识库中没有找到足够依据回答这个问题。… (refused: not enough evidence → flagged for escalation)
```

> The default is **offline extractive mode** (no LLM; it quotes evidence sentences), so answers read like excerpts. With an LLM configured you get normal prose answers — see [Plugging in real models](#plugging-in-real-models-gpu--openai).

More to try:

```bash
servicekb ask "AS-250K-HV 的直流开关坏了，仓库里的 220-11008-A 能用吗？"
servicekb search "as110k 早上报 E037"          # retrieval debug view: rank of each hit in each lane
servicekb ask "E051 换哪个料号" --groups public  # as an external user: internal bulletins/tickets are invisible
servicekb eval                                  # run the evaluation set
servicekb chat clean                            # clean the sample Teams export
servicekb serve                                 # HTTP API → http://localhost:8000/docs
pytest                                          # tests
```

---

## How it works

![servicekb architecture](docs/images/architecture.en.svg)

In one sentence: **build the object layer first (what each model / part / fault code is, what replaces what), then chunk documents and attach them to those objects; at query time search three lanes, fuse, refuse when evidence is weak, and give the LLM both the object-layer facts and the evidence, with mandatory citations.**

Details: [docs/01_architecture.md](docs/01_architecture.md). New to RAG? Start with the primer: [docs/00_rag_primer.md](docs/00_rag_primer.md).

### What happens inside one question

Example: "Can I still order 310-04521-A?" (numbers are from a real run on the sample data):

![What happens inside one question](docs/images/query_flow.en.svg)

- The **dense lane** understands meaning but can't tell apart part numbers that differ by one character; the **keyword lane** matches literally; the **entity lane** matches object IDs exactly and also pulls in substitutes.
- **RRF** only looks at ranks — results found by several lanes rise to the top.
- **Structured facts** come from the parts master, which is more authoritative than old tickets: even if an old ticket recommends part A, the answer warns "A is obsolete, use B".

### The object layer: from records to objects

Typical field data: the same model is spelled five different ways, the same part number is entered twice with different names, and parts recommended in old tickets have since been discontinued. The object layer gives each model / part / fault code a single ID plus aliases, and attaches document chunks to those objects:

![Object layer](docs/images/object_layer.en.svg)

### Ingestion: incremental, idempotent, fault-isolated

![Ingestion pipeline](docs/images/ingest_pipeline.en.svg)

- Unchanged documents are skipped; edit one paragraph and only that chunk is re-embedded.
- A broken PDF doesn't affect other documents; its previous version stays searchable and it is retried on the next run.
- If more than half of a full-snapshot source "disappears", deletion is refused — so an unmounted network drive can't wipe the knowledge base.

---

## Using PostgreSQL

Memory mode re-ingests on every start and is only for trying things out. For real use, connect Postgres:

```bash
docker compose up -d db                   # PostgreSQL 16 + pgvector
cp .env.example .env                      # then edit .env:
#   SERVICEKB_STORE_BACKEND=postgres
#   SERVICEKB_DATABASE_URL=postgresql://servicekb:servicekb@localhost:5432/servicekb
servicekb db init                         # create tables (runs migrations)
servicekb ingest                          # ingest; run it again and everything is "unchanged"
servicekb ask "E037 早上出现中午消失"
```

No Docker? Any PostgreSQL 14+ with the [pgvector](https://github.com/pgvector/pgvector) extension works (Ubuntu: `apt install postgresql-16-pgvector`; macOS: `brew install postgresql@16 pgvector`). Schema and design notes: [docs/03_database.md](docs/03_database.md).

---

## Plugging in real models (GPU / OpenAI)

The offline "hashing embedder" has **no semantic understanding** (for example, an English question won't find a Chinese document). Recommended production setup:

| Component | Recommendation | Where | Config |
| --- | --- | --- | --- |
| Embeddings | `BAAI/bge-m3` (multilingual zh/en, 1024-d, 8192 tokens) | 48 GB GPU, TEI container | `EMBEDDING_BACKEND=openai` + `EMBEDDING_BASE_URL=http://gpu:8080/v1` + `EMBEDDING_DIM=1024` |
| Reranker | `BAAI/bge-reranker-v2-m3` | 48 GB GPU, TEI container | `RERANKER_BACKEND=tei` + `RERANKER_URL=http://gpu:8081` |
| LLM | OpenAI API, or local vLLM (a 96 GB GPU fits ~30B-class models) | 96 GB GPU / cloud | `LLM_BACKEND=openai` + `LLM_MODEL=...` (+ `LLM_BASE_URL` pointing at vLLM) |

```bash
docker compose --profile gpu up -d        # embeddings / reranker / llm GPU services
# edit .env following "方案 B" (option B) in .env.example
servicekb db reset --yes && servicekb db init && servicekb ingest   # changing the embedding model requires a rebuild
```

All settings carry the `SERVICEKB_` prefix; the full list is in [`src/servicekb/config.py`](src/servicekb/config.py). Deployment, GPU planning and the security checklist: [docs/06_deployment.md](docs/06_deployment.md).

---

## Project layout

```text
servicekb-rag/
├── configs/sources.yaml            # data sources: what data, where, who may see it
├── data/sample/                    # fictional sample data (dictionary, parts, manuals, bulletins, tickets, Teams export)
├── eval/golden_qa.jsonl            # evaluation set: question + relevant docs + must-include facts
├── src/servicekb/
│   ├── config.py                   # every tunable (overridable via env vars)
│   ├── models.py                   # core types: SourceRecord → ParsedDocument → Chunk → AnswerResult
│   ├── connectors/                 # [read] one connector per data source
│   │   ├── filesystem.py           #   local / network-share files
│   │   ├── qa_archive.py           #   ticket archive (with a quality gate)
│   │   ├── parts_master.py         #   parts master → entities + relations + one record per part
│   │   ├── dictionary.py           #   domain dictionary YAML → model / fault-code entities
│   │   └── sharepoint.py           #   SharePoint (Graph delta sync + throttling retries)
│   ├── parsing/                    # [parse] every format → one unified list of Blocks
│   ├── processing/
│   │   ├── text.py                 #   normalization, token estimation, zh/en sentence split, jieba tokenization
│   │   ├── cleaning.py             #   email-noise removal, PII redaction
│   │   ├── chunking.py             #   structure-aware chunking (heading paths / table headers / atomic Q&A)
│   │   ├── entities.py             #   entity linking, substitute expansion, structured facts
│   │   └── dedup.py                #   SimHash near-duplicate detection
│   ├── embedding.py                # [embed] hashing / sentence-transformers / OpenAI-compatible
│   ├── store/
│   │   ├── base.py                 #   storage interface
│   │   ├── memory.py               #   in-memory store (includes textbook BM25 — good for learning)
│   │   ├── postgres.py             #   Postgres store (HNSW, tsvector, atomic replace)
│   │   └── migrations/001_init.sql #   schema (every column commented)
│   ├── pipeline/ingest.py          # [ingest] incremental / idempotent / embedding reuse / fault isolation / safe deletes
│   ├── retrieval/
│   │   ├── retriever.py            #   query analysis + three lanes in parallel
│   │   ├── fusion.py               #   RRF + duplicate collapsing
│   │   └── rerank.py               #   cross-encoder / TEI reranking
│   ├── generation/
│   │   ├── prompts.py              #   prompts (versioned)
│   │   └── generator.py            #   LLM / extractive fallback + citation validation
│   ├── chat/                       # [chat cleaning] Teams HTML, threads, roster, Q&A extraction, routing
│   ├── service.py                  # app container: ingest / search / ask (evidence gate, refusal, logging)
│   ├── evaluation.py               # metrics & reports
│   ├── api.py                      # FastAPI: /v1/ask /v1/search /v1/feedback /healthz /readyz
│   └── cli.py                      # command line
├── tests/                          # 58 tests, including Postgres integration tests
├── docs/                           # in-depth docs (Chinese) + diagrams (docs/images, generated by make_diagrams.py)
├── Dockerfile / docker-compose.yml # deployment (GPU profile: 2× TEI + vLLM)
└── .github/workflows/ci.yml        # lint → tests → Postgres integration → eval gate
```

---

## Key design decisions

Each one comes from reading the source of open-source RAG projects ([docs/07_repo_survey.md](docs/07_repo_survey.md)) or from real constraints of the field-service setting.

| Decision | Why | Borrowed from / compared with |
| --- | --- | --- |
| PostgreSQL + pgvector only | A small team runs one system; transactions make "document + chunks" replacement atomic; ACLs, entities and logs are relational anyway | R2R and LightRAG both support all-Postgres; RAGFlow needs MySQL + Redis + MinIO + ES — too heavy |
| Chinese tokenized in Python, written as tsvector literals | Managed Postgres often can't install zhparser / pg_jieba; tokenization is identical to the in-memory store | The pgvector integrations of R2R / LlamaIndex / Haystack default to the `english` config, which doesn't work for Chinese |
| **Entity retrieval as a third lane** | Vectors can't tell "310-04521-A" from "310-04521-B"; model names are spelled every possible way | Not built into any general framework — specific to industrial data |
| Parts master → structured facts in the prompt | Old tickets may recommend obsolete parts; the master data is authoritative | Addresses record-oriented (not object-oriented) source data |
| RRF, k=60, equal lane weights | Scores have different scales, ranks fuse robustly; no 5:1 semantic bias (bad for "E037"-style queries) | Haystack (k=61), R2R (k=50, 5:1) |
| chunk id = hash(doc id, chunk content) | Edit one paragraph → only that chunk is re-embedded | R2R uses positional IDs (one insertion shifts everything); LlamaIndex / RAGFlow hash content |
| pipeline_version = code version + chunking params + model + dictionary fingerprint | Any logic change triggers reprocessing automatically, with embeddings reused | RAGFlow task fingerprints |
| Incremental sources delete only on explicit tombstones; full snapshots have a delete-ratio guard | Prevents "drive not mounted → knowledge base wiped" | LlamaIndex `UPSERTS_AND_DELETE` deletes wrongly in incremental syncs |
| Parse failure keeps the previous version | A corrupted new PDF doesn't remove the old one from search | Dify's phased status machine |
| Keep duplicates at ingest, collapse at query time | Keeps lineage (different models / dates / ACLs) without flooding top-k | — |
| Heading path goes into the embedded text, not the stored text | Contextualized chunks retrieve better, citations stay clean | docling `contextualize`; Chatchat writes titles into the text (not recommended) |
| Q&A pairs are atomic chunks; tables split by row with the header repeated | A split Q&A is useless; a table fragment without a header is unreadable | RAGFlow qa / table templates |
| Weak evidence → refuse and escalate without calling the LLM | Saves cost, prevents hallucination; feeds the "unanswered request" escalation flow | QAnything's rerank threshold + relative-drop cut |
| Permissions come only from the identity system, never from the LLM | Prevents privilege escalation | Dify's automatic metadata filters are limited to business fields |

---

## Current results

Sample data + offline hashing embedder (no GPU, no LLM), `servicekb eval`, 18 questions, k=5:

| Metric | In-memory store | PostgreSQL |
| --- | --- | --- |
| hit@5 | 0.938 | 0.938 |
| MRR@5 | 0.823 | 0.833 |
| nDCG@5 | 0.765 | 0.760 |
| Key-fact recall (must_include) | 0.857 | 0.857 |
| Refusal accuracy | 1.000 | 1.000 |

The two backends differ in keyword scoring: the in-memory store implements textbook BM25, Postgres uses `ts_rank` (no IDF) — see [docs/03_database.md](docs/03_database.md#5-中文全文检索). Key-fact recall is lower than retrieval hit rate mainly because offline **extractive** answers can only quote whole sentences; with an LLM the answer is composed from the evidence.

The one retrieval miss, q13 ("How to upgrade the firmware of AS-110K-HV without network?"), is **cross-lingual**: English question, answer in a Chinese manual. The hashing embedder can't bridge languages; bge-m3 can. **This failing case is deliberately kept in the evaluation set** to show why production needs a real multilingual embedding model.

> The sample data is tiny; these numbers only show that the pipeline works and the logic is correct — **not** how it performs on real data. Build an evaluation set from real tickets before going live ([docs/05_evaluation.md](docs/05_evaluation.md)).

---

## Common tasks

**Add a model alias / fault code**: edit `data/sample/dictionaries/entity_aliases.yaml` and re-run `servicekb ingest`. Entity links in every document are refreshed automatically (embeddings are reused, so it's fast).

**Add a folder of documents**: add a `type: filesystem` entry to `configs/sources.yaml` with `path`, `doc_type` and `acl_groups`.

**Connect another system (Jira, ServiceNow, …)**: write a connector modeled on `connectors/qa_archive.py` and register it in `connectors/__init__.py`. Only `iter_events()` is required.

**Debug poor retrieval**: run `servicekb search "question"` and look at each lane's ranks:
- dense lane ranks low → better embedding model / check the query prefix
- keyword lane misses → tokenization; add domain terms to the dictionary
- entity lane misses → alias not registered, or `SERVICEKB_PART_NUMBER_PATTERN` doesn't match your part-number format

The full "symptom → which knob" table: [docs/04_retrieval_generation.md](docs/04_retrieval_generation.md#11-调参指南).

**Clean Teams chat logs**: `servicekb chat clean` — see [docs/09_chat_log_cleaning.md](docs/09_chat_log_cleaning.md).

![Teams chat cleaning funnel](docs/images/chat_cleaning.en.svg)

**Check parts-master data quality**: `servicekb audit parts` (duplicate part numbers, near-identical names, references to non-existent parts).

**Edit the diagrams**: change labels in `docs/images/make_diagrams.py` and run `python docs/images/make_diagrams.py` (standard library only; regenerates both languages).

---

## Documentation map

The docs are in Chinese; titles translated here.

| # | Doc | For |
| --- | --- | --- |
| 1 | [00 RAG primer](docs/00_rag_primer.md) | Never touched RAG: concepts, analogies, common pitfalls |
| 2 | [01 Architecture](docs/01_architecture.md) | How the modules fit together |
| 3 | [02 Data pipeline](docs/02_data_pipeline.md) | Adding sources, tuning chunking |
| 4 | [03 Database design](docs/03_database.md) | Operating / scaling the database |
| 5 | [04 Retrieval & generation](docs/04_retrieval_generation.md) | Tuning quality, editing prompts |
| 6 | [05 Evaluation](docs/05_evaluation.md) | Building an eval set, regression testing |
| 7 | [06 Deployment & operations](docs/06_deployment.md) | Going live: GPU planning, security, monitoring |
| 8 | [07 Open-source survey](docs/07_repo_survey.md) | Where the design decisions come from |
| 9 | [08 Limitations & roadmap](docs/08_limitations_roadmap.md) | What to build next |
| 10 | [09 Chat-log cleaning](docs/09_chat_log_cleaning.md) | Teams / Slack / WeCom exports |
| — | [Glossary](docs/glossary.md) | Look things up |

---

## Limitations & roadmap

This version is a **skeleton ready for a pilot**, not the finish line. The most important gaps (full list and improvement designs in [docs/08_limitations_roadmap.md](docs/08_limitations_roadmap.md)):

1. **Postgres keyword search isn't true BM25** (`ts_rank` has no IDF) → BGE-M3 sparse vectors (pgvector `sparsevec`) or ParadeDB `pg_search`
2. **PDF parsing reads the text layer only**; scans and complex tables need docling / MinerU (hook already in place)
3. **No image capability**: site photos, nameplates and screen captures aren't used yet → connect to photo-based part recognition (OCR + image embeddings)
4. **Statistical questions don't fit RAG** ("how many E037 alarms did this model have last year?") → Text-to-SQL routing
5. **Refusal in offline mode is heuristic**; production must enable the reranker and calibrate its threshold on real data
6. **Ingestion is synchronous and single-process**; at scale it needs a job queue (page-range PDF tasks, retries, progress)
7. **SharePoint permissions aren't synced**; ACLs are configured per source
8. **Citations point to chunks**, not to individual sentences

---

## FAQ

**Why not just use Dify / RAGFlow / FastGPT?**
They're good platforms for a general-purpose knowledge base. But the hard parts here — exact part-number matching, substitute relations, model aliases, ticket quality gates, chat-log cleaning, linking to the parts master — all need deep customization, and deep customization inside a platform is often harder to maintain than a small, clear framework. Mind the licenses too: Dify uses a modified Apache-2.0 (restricts multi-tenant SaaS), QAnything is AGPL-3.0. See [docs/07_repo_survey.md](docs/07_repo_survey.md).

**Why not LangChain / LlamaIndex?**
You could; it was avoided on purpose. Each RAG step is tens to hundreds of lines of code — written out directly, a newcomer can see exactly how data flows and debug it. Framework abstractions get in the way when debugging.

**How big can the knowledge base get?**
Postgres + pgvector (HNSW) works well up to millions of chunks. A 1024-d float32 vector is ~4 KB, so 1M chunks ≈ 4 GB of vectors plus roughly 2× for the HNSW index. Beyond that, `halfvec` halves the size, or move vectors to Milvus / Elasticsearch. See [docs/03_database.md](docs/03_database.md#7-容量规划).

**Can the data be sent to OpenAI?**
That's a compliance question, not a technical one. The framework supports fully local deployment (TEI + vLLM) as well as OpenAI. Either way, PII is redacted before ingestion. Check with your legal team.

---

## References

The design draws on reading the source code (October 2026 versions) of:
[RAGFlow](https://github.com/infiniflow/ragflow) ·
[Dify](https://github.com/langgenius/dify) ·
[LlamaIndex](https://github.com/run-llama/llama_index) ·
[Haystack](https://github.com/deepset-ai/haystack) ·
[R2R](https://github.com/SciPhi-AI/R2R) ·
[QAnything](https://github.com/netease-youdao/QAnything) ·
[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) ·
[kotaemon](https://github.com/Cinnamon/kotaemon) ·
[LightRAG](https://github.com/HKUDS/LightRAG) ·
[GraphRAG](https://github.com/microsoft/graphrag) ·
[docling](https://github.com/docling-project/docling) ·
[MinerU](https://github.com/opendatalab/MinerU) ·
[Unstructured](https://github.com/Unstructured-IO/unstructured) ·
[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) ·
[ragas](https://github.com/explodinggradients/ragas) ·
[pgvector](https://github.com/pgvector/pgvector)

No code from these projects was copied.

License: MIT
