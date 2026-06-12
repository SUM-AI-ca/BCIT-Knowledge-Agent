# BCIT AI Advisor — RAG Chatbot

**Live: [https://bcitai.ca](https://bcitai.ca)** — tech blog landing page at `/`, chatbot at [`/chat`](https://bcitai.ca/chat).

A retrieval-augmented generation (RAG) chatbot that answers questions about BCIT
programs, courses, admissions, and campus life, grounded in **11,129 documents
crawled from the live BCIT website** (September 2024 intake onwards). The entire
pipeline is **CPU-only**: embeddings, vector search, reranking, and the LLM all
run on GCP managed services — no GPU, no GCP API keys (Application Default
Credentials end to end). Every query is traced in LangSmith.

| Component | Service |
|---|---|
| Corpus | Own crawler (`crawl_bcit.py`) — sitemaps + BCIT outlines API |
| Embeddings | Vertex AI `gemini-embedding-001` (1536-dim) |
| Vector store | Cloud SQL PostgreSQL + pgvector (HNSW) |
| Sparse retrieval | In-process BM25 (`rank_bm25`) |
| Rank fusion | Reciprocal Rank Fusion (alpha 0.48, k=60) |
| Reranker | Vertex AI Ranking API (`semantic-ranker-default-004`) |
| LLM | `gemini-3.1-flash-lite` (generation) + `gemini-3.5-flash` (query rewriter) via LangChain `ChatVertexAI` |
| Backend | FastAPI + uvicorn (Python 3.10, managed via uv) |
| Frontend | React 19 + Vite 7 (no router — pathname-based pages) |
| Hosting | GCE `e2-standard-2` VM + Cloudflare (DNS, TLS, port routing) |
| Observability | LangSmith tracing (project `bcit-chatbot`) |

---

## How a query flows

1. **Rewrite + decompose** — one schema-constrained JSON call per turn
   (`REWRITER_MODEL=gemini-3.5-flash`, temperature 0, thinking 0, ~430
   tokens) returns a standalone question (pronouns resolved from history)
   plus 1–4 self-contained sub-queries. The rewriter deliberately runs on a
   stronger model than generation: sub-query quality drives retrieval
   (benchmark: hit-rate 0.929 with a lite rewriter vs 0.963 with 3.5-flash). Single questions yield one sub-query;
   "admission requirements AND tuition AND housing" yields three. Parse
   failure falls back to the raw question (counted in `query_usage` logs).
2. **Embed** — each query is embedded with Vertex AI `gemini-embedding-001`
   (1536 dimensions).
3. **Hybrid retrieval** — dense (pgvector HNSW, MMR, `fetch_k=50`,
   `ef_search=100`) and sparse (in-process BM25 — exact identifiers like
   "COMP 1510" that embeddings miss) merge via **Reciprocal Rank Fusion**:
   `1 / (60 + rank)` per list, weighted by `HYBRID_ALPHA=0.48`. RRF compares
   ranks, not raw scores, so incompatible similarity scales never need
   normalizing.
   - *Single-topic turns* retrieve once at full width (23+23 → RRF top 25).
   - *Multipart turns* fan out per sub-query (12+12 → top 15 each) on a
     shared thread pool, then rank-interleave into one deduped pool
     (cap 40). Retrieval returns metadata copies — BM25 documents alias the
     shared corpus pickle, and annotating them in place would corrupt it.
4. **Rerank** — ONE Vertex AI Ranking API call per turn regardless of
   sub-query count (the API bills per query): the pooled candidates are
   scored against the standalone question (`semantic-ranker-default-004`),
   the top `RERANKER_TOP_K=10` are kept, and a coverage quota guarantees
   every sub-query keeps ≥ 2 of its own chunks (its best leftovers swap in,
   evicting only above-quota docs).
5. **Assemble context from chunks** — the selected chunks themselves go into
   the prompt (not their whole source files, which is what the pre-June-2026
   pipeline did at ~44k input tokens/query). Each chunk pulls in ±2
   neighbors from an in-process ordinal index built off the BM25 pickle at
   startup, consecutive runs merge with their 130-char overlaps deduped, and
   the result is capped at 24k chars (neighbor expansion drops from the
   lowest-ranked sources first). Citation headers
   (`Document N: [URL: …]`) are identical to the legacy format — the
   Sources-section prompt instructions depend on them.
6. **Generate** — `GEMINI_MODEL=gemini-3.1-flash-lite`, static instructions
   first / variable inputs last (implicit-cache-friendly),
   `{question_parts}` enumerating multipart sub-questions,
   `max_output_tokens=2048` with `thinking_budget=0` (see gotchas — thinking
   counts against the cap).

Conversation state: each browser session holds a 5-turn
`ConversationBufferWindowMemory`; sessions expire after 30 minutes. Saved
answers are stripped of their Sources section and capped at 1,500 chars —
history is re-sent every turn, so it pays to keep it lean. Chat requests run
in a thread pool so the FastAPI event loop never blocks.

Every query produces one LangSmith trace (project `bcit-chatbot`) rooted at
`bcit_query`, with a dedicated `vertex_rerank` retriever span; one
structured `query_usage` JSON log line per query records tokens
(input/output/reasoning/cache-read), per-stage latency, sub-query count,
context size, and fallback flags.

---

## Cost & accuracy optimization (June 2026)

Measured on a 40-case golden set (`backend/eval/golden_set.jsonl`: outline
facts, program/admission, student life, multipart, follow-up pairs) against
the same corpus and models — only the pipeline changed:

| Metric | Before (full-doc context) | After (chunks + decompose) |
|---|---|---|
| Input tokens / query (mean) | 44,322 | **5,884 (−87%)** |
| Input tokens / query (p50) | 36,471 | 5,936 (−84%) |
| Output tokens (incl. thinking) | 210 + 896 thinking | **175 + 0** |
| Retrieval URL hit-rate | 0.892 | **0.963** |
| Key-fact recall | 0.848 | **0.885** |
| Citation precision | 1.000 | 1.000 |
| Latency p50 / p95 | 7.6 s / 10.8 s | **4.1 s / 5.2 s** |
| Truncated answers / decompose fallbacks | 0 / 0 | 0 / 0 |

The legacy pipeline stays one env away:
`CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false RERANKER_TOP_K=13`.
Sweep notes: `NEIGHBOR_RADIUS=2` beat 1 (outline recall +0.04 for +5.6%
tokens); `RERANKER_TOP_K=13`, `RERANK_SCORE_THRESHOLD=0.2`, and model-default
thinking all measured worse (see `config.py` comments for why).

A follow-up model benchmark (`eval/benchmarks/202606_model_comparison/`)
then cut cost a further 69%: generation moved to `gemini-3.1-flash-lite`
with the rewriter kept on `gemini-3.5-flash` — $0.0126 → $0.0039/query
($12.61 → $3.87 per 1k) at hit-rate parity (0.963) and the best fact recall
of any run (0.890). At lite-class generation prices the Ranking API's fixed
$0.001/query is now the largest cost component.

Round 2 (`eval/benchmarks/202606_retrieval_cost_experiments/`, 8 experiments
/ 26 gated runs) attacked exactly that and the section-flooding weakness:

- **BM25 title-aware indexing** (`BM25_INDEX_AUG`) — the BM25 vectorizer now
  indexes `title + category + URL-slug + chunk text` while serving unchanged
  documents (no re-embed, no pickle rebuild). Deep program-page chunks carry
  their page identity, so "X program entrance requirements" no longer floods
  with sibling programs: multipart hit 0.917 → **1.000**.
- **Consensus rerank-skip** (`RERANK_SKIP_CONSENSUS=0.6`) — when ≥60% of the
  fusion top slice was surfaced by both retrieval arms, fusion order stands
  and the Ranking API call is skipped (~half of all queries). Guard: a turn
  that skipped the rewriter always reranks — every query keeps at least one
  semantic stage.
- **Simple-query rewrite skip** (`REWRITE_SKIP_SIMPLE`) — implemented and
  initially adopted, then **reverted by a second 25-case eval set**
  (`eval/golden_set_v2.jsonl`) built around messy real-world queries:
  acronyms, typos, and a Korean-language question. Skipping is free on clean
  English prose but messy recall collapsed 1.000 → 0.50-0.58 without the
  rewriter (which normalizes, corrects, and translates — and raises the
  rerank-skip rate enough to refund part of its own cost). The flag stays
  available for clean-traffic cost pressure.

Shipped result: **v1 set hit 0.975 / recall 0.931 / $0.00315 (−18%)**;
v2 messy set hit 0.940 / recall 0.937 (reproduced twice), p50 −0.3 s.
Rejected with evidence: bigger
rerank pools (sibling-chunk dilution), cheaper rewriters (2.5-flash hit 0.929
twice — rewrite quality is load-bearing), keyword/HyDE rewriter extensions
(no gain once BM25 is title-aware, +$0.0005 in 3.5-flash output tokens), and
all fusion-parameter moves (alpha 0.48 / rrf_k 60 / MMR λ 0.87 confirmed
optimal).

### The eval harness

`backend/eval/golden_set.jsonl` — 40 cases written against the actual corpus
(every expected URL and key fact verified in the source documents): 12
course-outline facts, 10 program/admission, 6 student life, 6 multipart, 6
follow-up pairs (scored on the second turn, exercising the rewrite).
`expected_urls` and `key_facts` are lists of alternative-groups — a group
counts when ANY alternative matches, so "$500 to $800" / "$500–$800" phrasing
differences don't punish correct answers. Facts match with word-boundary
normalization ("75" ≠ "175", "$6,000" = "$6000").

```bash
cd backend
export PG_CONNECTION=$(grep '^PG_CONNECTION=' .env | cut -d= -f2- | sed 's/:5432/:5433/')  # WSL

# measure a configuration (env flags BEFORE launch — config reads env at import)
.venv/bin/python eval/run_eval.py --label after
CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false RERANKER_TOP_K=13 \
  .venv/bin/python eval/run_eval.py --label before

# compare two runs: aggregate deltas + per-case regressions
.venv/bin/python eval/run_eval.py --compare eval/results/before.json eval/results/after.json

# subsets while iterating
.venv/bin/python eval/run_eval.py --label quick --category multipart --limit 3
```

Each run writes per-case records (retrieved URLs, missed facts, token usage,
stage timings, answer excerpt) plus aggregates to `eval/results/<label>.json`
(gitignored — measurements are per-machine). Metrics: URL hit-rate (groups of
acceptable sources), key-fact recall, citation precision (cited URLs ⊆
retrieved context), input/output/reasoning/cache-read tokens, p50/p95
latency, MAX_TOKENS truncations, decompose fallbacks.

---

## The corpus

Everything is crawled from the live site, so catalog content is current by
construction. The one place a date policy applies is course outlines, which are
inherently term-bound:

| Category | Docs | Freshness policy |
|---|---|---|
| Course outlines | 3,262 | **Terms ≥ 202430 (Sept 2024), latest term per course only.** Multiple terms of the same course would surface contradictory schedules/instructors at retrieval time. |
| Course catalog pages | 7,060 | Live page = current truth; no date filter. |
| Program pages | 529 | Same. |
| Admission / international / student-services pages | 236 | Same, scoped to 10 path prefixes (see `PAGE_CATEGORIES`). |
| BCIT Student Association (bcitsa.ca) | 42 | Separate WordPress site, own sitemap. |

### Crawling / refreshing

```bash
cd backend
.venv/bin/python crawl_bcit.py                  # full crawl into data_new/ (~85 min)
.venv/bin/python crawl_bcit.py --phase outlines # phases: worklist|outlines|courses|programs|pages|bcitsa
CRAWL_OUT=./data .venv/bin/python crawl_bcit.py --phase bcitsa   # custom output dir
```

Polite by design (3 concurrent requests, jittered delay, retries) and resumable —
existing non-empty files are skipped. Failures land in `data_new/_state/errors.log`.

---

## Indexing

```bash
cd backend
PG_COLLECTION=bcit_docs_<version> \
DOCUMENTS_PICKLE=./vectorstore/documents_<version>.pkl \
.venv/bin/python build_pgvector.py
```

Chunks `data/` (1,024 chars, 130 overlap) into ~100k chunks, writes the BM25
pickle, embeds via Vertex AI (~1.5 h at ~20 chunks/s), inserts into pgvector,
and ensures the HNSW index. Three properties worth knowing:

- **Resumable / incremental** — chunk ids are deterministic
  (`uuid5(collection:source#index)`); re-running skips chunks already in the
  database. Adding 42 documents later embedded only the 365 new chunks (17 s).
- **Collection-salted ids** — `langchain_pg_embedding` upserts on id across
  *all* collections. Without the collection salt, rebuilding from same-named
  source files would hijack rows out of the live collection mid-build.
- **Blue-green cutover** — build into a fresh versioned collection while
  production serves the old one, flip the `PG_COLLECTION` **and**
  `DOCUMENTS_PICKLE` defaults in `config.py` **together** (the dense and BM25
  sides must serve the same crawl — June 2026 the pickle default was left
  behind and local BM25 quietly served the retired corpus), deploy config +
  pickle, restart. Zero downtime; the old collection doubles as instant
  rollback until you drop it (`drop_old_collection.sh` pattern).

---

## Design decisions (and why)

A decision record for future development — each of these was deliberate:

1. **Managed services over local models.** The first version ran FAISS + a
   local cross-encoder; it was migrated to Vertex AI + Cloud SQL pgvector so a
   2-vCPU VM can serve everything. Embeddings, reranking, and generation are
   all API calls; the VM only does BM25 + orchestration.
2. **1536-dim MRL truncation.** pgvector's HNSW index supports ≤ 2000 dims, so
   `gemini-embedding-001` (3072 native) is truncated to 1536 via MRL.
3. **RRF over score normalization.** Dense cosine scores and BM25 scores are
   incomparable; fusing by rank position avoids calibration entirely.
4. **`ef_search=100`.** pgvector's default of 40 silently caps how many
   candidates an HNSW scan returns — below MMR's `fetch_k=50`. Set per session
   in `query_rag.py`.
5. **Latest-term-per-course outlines.** The old corpus carried up to 6 terms of
   the same course; retrieval surfaced stale instructors/dates alongside
   current ones. One outline per course removes the contradiction class.
6. **Sitemap `lastmod` rejected as freshness filter.** 78% of course pages
   carry a 2022 CMS-migration timestamp while content is current. Filtering on
   `lastmod ≥ 2024-09` would have dropped 89% of courses. Live-crawl instead.
7. **Outline discovery via REST API, not page scraping.**
   `GET /wp-json/bcit/outlines/v1/load_subjects_term/{term}` returns the full
   term catalog (subject → course → CRN) in one call. Outline URL =
   `/outlines/{term}{crn}/`. Term codes: `10`=Jan, `20`=Apr, `30`=Sept.
8. **No frontend router.** Two pages (`/` blog, `/chat` chat) split on
   `window.location.pathname`; FastAPI's SPA fallback serves both. `GET /chat`
   (page) and `POST /chat` (API) coexist because FastAPI routes by method.
9. **No auth.** The login gate was removed on purpose — it's a public demo.
   Rate limiting, if ever needed, should go in Cloudflare (WAF rules), not code.
10. **Cloudflare Origin Rule instead of nginx.** The GCP firewall opens only
    tcp:8000; Cloudflare rewrites destination port 80→8000 ("Change port"
    template) and terminates TLS (Flexible mode). No reverse proxy on the VM.
11. **Filename conventions are load-bearing.** `build_pgvector.py` parses
    outline metadata from filenames (`DEPT_NUM_TERM.txt`, regex
    `([A-Z]{2,4})_(\d[A-Z0-9]{3,4})_(\d{6})` — apprenticeship codes like
    `AATE 1GAP` exist). The `URL:` header line in each txt feeds source
    citation metadata.
12. **Chunks in the prompt, whole files never.** The original pipeline loaded
    each cited source file from disk into the prompt (~9k tokens × ~3–5 files
    = 44k input tokens/query) to preserve list/table completeness across
    chunk boundaries. Neighbor expansion (±2 chunks from the in-process
    ordinal index) buys that completeness for a few hundred tokens instead;
    eval showed accuracy went UP (fact recall 0.848 → 0.885) because precise
    chunks beat 9k-token haystacks.
13. **One Ranking API call per turn, not per sub-query.** The Ranking API
    bills per query; pooled rerank with `top_n=len(pool)` + a per-sub-query
    coverage quota keeps multipart quality without multiplying the fixed
    cost (`RERANK_MODE=per_subquery` exists as the comparison flag).
14. **Decompose on every turn.** The old history-only rewrite skipped turn 1,
    missing first-turn multipart questions. The merged rewrite+decompose
    call is ~430 tokens with thinking 0 — cheap enough to always run, and
    single questions short-circuit back to full-width legacy retrieval.
15. **Thinking off for generation.** This is an extraction-and-summarize
    workload over provided context: eval measured model-default thinking as
    strictly worse (recall 0.852 vs 0.877, p95 2.2×, ~900 thinking tokens
    billed per query).

---

## Operational gotchas (hard-won)

Environment quirks that cost time once — don't rediscover them:

- **WSL + port 5432**: a local PostgreSQL 14 owns 5432 in WSL. Run the Cloud
  SQL proxy on **5433** and rewrite `PG_CONNECTION` accordingly (see
  `drop_old_collection.sh` for the pattern).
- **ADC reauth is separate from gcloud reauth.** `gcloud compute ssh` working
  does not mean ADC works — `invalid_rapt` errors need
  `gcloud auth application-default login` (interactive, user-run).
- **Terminal paste mangles long commands**: one-liners lose spaces at wrap
  points and heredocs gain indentation (breaking the `EOF` terminator and
  Python). For anything non-trivial, write a script file and hand over a
  single short command.
- **Deploying to the VM**: `/opt/bcit-rag` is root-owned — scp to `/tmp`, then
  `sudo cp/mv`. Upload and install in quick succession (a staged file in `/tmp`
  vanished once between commands). After unit-file edits systemd warns until
  `daemon-reload`.
- **VM facts**: SSH user `park`, app at `/opt/bcit-rag/`, services
  `bcit-chatbot.service` (uvicorn :8000, `User=park`,
  `ExecStart=/opt/bcit-rag/backend/.venv/bin/uvicorn server:app`) and
  `cloud-sql-proxy.service`. `system_config.txt` paths (`/home/jpark440`) are
  from a retired VM — ignore.
- **BCIT site quirks**: robots.txt disallows `/outlines/` (crawl politely —
  3 concurrent, jittered delay was never throttled); one program page
  (marine-engineering 2935dipma) resets every connection server-side; CRNs are
  reused across years for the same section; next-fall outlines start appearing
  ~9 months early (202630 had 40 courses in June 2026).
- **langchain-postgres schema**: all collections share one
  `langchain_pg_embedding` table with a single HNSW index; collection delete =
  SQL `DELETE` by `collection_id` + `VACUUM ANALYZE` (no API).
- **Embedding throughput**: `gemini-embedding-001` takes one text per request;
  the build parallelizes across threads (~20 chunks/s sustained, 100k chunks
  ≈ 1.5 h, no quota errors at that rate).
- **Thinking tokens count against `max_output_tokens`.** With the model
  default thinking budget and a 2048 cap, gemini-3.5-flash burned ~1,950
  tokens thinking and truncated every answer at ~80 visible tokens
  (`finish_reason=MAX_TOKENS`, Sources lost). Pair the cap with
  `GEMINI_THINKING_BUDGET=0`, or raise the cap to ≥ 4096 if re-enabling
  thinking. The `query_usage` log warns on MAX_TOKENS.
- **`RERANK_SCORE_THRESHOLD` undermines the multipart quota.** The threshold
  filter runs during context assembly — after pooled-rerank selection — so it
  strips exactly the lower-scored chunks the coverage quota swapped in
  (eval: multipart recall 0.819 → 0.611 at threshold 0.2). Leave it at 0
  unless re-evaluating multipart cases alongside.
- **`config.py` reads env at import.** Anything that should affect a run
  (eval flags, `PG_CONNECTION` port rewrite) must be exported before Python
  starts; `load_dotenv()` does not override already-exported variables.

---

## System state reference

Everything deployed/configured outside this repo, in one place:

| Thing | Value |
|---|---|
| Production URL | https://bcitai.ca (chat at `/chat`) |
| GitHub | `SUM-AI-ca/bcit-RAG-chatbot` (private; moved from `jp-ml/`) |
| GCP project | `wine-agent-jh-2026` |
| VM | `bcit-rag-vm`, us-west1-b, e2-standard-2, 34.19.21.7, firewall: tcp:8000 only (tag `bcit-rag`) |
| Cloud SQL | `bcit-rag-pg` (us-west1), db `ragdb`, user `raguser`, pgvector |
| Live collection | `bcit_docs_202606` — 100,515 chunks from 11,129 docs (June 2026 crawl) |
| Cloudflare | zone `bcitai.ca`: A `@` → 34.19.21.7 (proxied), Origin Rule port→8000, SSL Flexible |
| LangSmith | project `bcit-chatbot` ([smith.langchain.com](https://smith.langchain.com)) — tracing on since 2026-06-10, enabled purely via env vars |
| Secrets | `backend/.env` (`PG_CONNECTION`, `LANGSMITH_API_KEY`) — local + VM copies, never committed |
| Rollback assets | `backend/data_old_202409/` (old corpus, gitignored), VM `vectorstore/documents_old.pkl` |

`www` CNAME added 2026-06-11 (`www` → `bcitai.ca`, proxied) — verified:
`https://www.bcitai.ca/health` and `/chat` both 200 through the same Origin
Rule; `server.py` CORS already allowlists the `www` origin.

Pending niceties: "Always Use HTTPS" is off (http://bcitai.ca serves without
redirect).

---

## Future work — where to start

Ideas that fit the current architecture, with their natural hook points:

- **Per-source incremental refresh** — `build_pgvector.py` currently
  appends-or-skips; add a "delete rows for source X, reinsert" mode so a
  changed page can be re-embedded without a new collection. Hook:
  `existing_ids()` / `deterministic_ids()`.
- **Scheduled term refresh** — the whole refresh is two commands (runbook
  below); wire into cron/GitHub Actions once credentials story is decided.
- **Program-page section flooding** — the one retrieval weakness the eval
  left open: section headings like "Entrance requirements" exist on all 529
  program pages, so BM25 floods the pool with sibling programs' chunks and
  the asked-about program's section chunk sometimes loses (multipart CST
  cases in `eval/results/`). Hook: section-aware metadata at build time
  (`chunk_index` is already stamped), or a BM25 field boost on the
  program/course code.
- **LLM-judge eval** — `run_eval.py --judge` is reserved but unimplemented;
  key-fact recall is substring-based today. Hook: `score_case()`.
- **Term-aware retrieval** — outline chunks carry `term_code` metadata already;
  boosting newer terms (or filtering expired ones at query time) is a metadata
  filter away.
- **Request timeout in the chat executor** — `/chat` has no deadline: a
  Vertex call that wedges (observed once locally — a gRPC call sat ~8 min
  before completing) parks one of the two worker threads the whole time.
  Two wedged requests would freeze the service. Hook: `asyncio.wait_for`
  around `query_chatbot_async` + a 504 response.
- **Streaming responses** — `ChatVertexAI` supports streaming; `/chat` returns
  a single JSON blob today. Needs SSE endpoint + frontend incremental render.
- **Citations UI** — answers already end with a `Sources` section (prompt-enforced);
  the frontend renders it as plain text. Parse and render as cards/links.
- **Observability** — LangSmith traces (with a dedicated rerank span) and a
  structured `query_usage` JSON log line per query exist; remaining: ship
  the log line somewhere queryable and add a `/metrics` endpoint.
- **Frontend dist in CI** — `frontend/dist` is gitignored and deployed by scp;
  a GitHub Action building + deploying on push would remove the manual step.

## Repository layout

```
backend/
  server.py                FastAPI app: /chat API + serves the built frontend
  query_rag.py             BCITChatbot — retrieval pipeline + LLM chain
  hybrid_retriever.py      BM25 + pgvector with RRF fusion
  reranker.py              Vertex AI Ranking API client
  embeddings.py            Vertex AI embedding wrapper
  config.py                All tunables (models, top-k, chunking, collection)
  crawl_bcit.py            Corpus crawler (sitemaps + outlines API)
  build_pgvector.py        Indexing job (resumable, collection-versioned)
  drop_old_collection.sh   One-off: drop a retired collection via proxy
  eval/golden_set.jsonl    40-case eval set (facts verified against corpus)
  eval/run_eval.py         Offline eval harness (--label runs, --compare)
  eval/results/            Per-run metrics JSON (gitignored)
  data/                    Crawled corpus (11,129 txt docs, 16 categories)
  vectorstore/             documents_<version>.pkl — BM25 source (generated, not committed)
frontend/
  src/App.jsx              Pathname routing: "/" → Blog, "/chat" → Chat
  src/Blog.jsx             Tech blog landing page (architecture deep dive)
  vite.config.js           Dev proxy: POST /chat → backend; GET /chat → SPA
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Blog landing page (also serves any non-API path as SPA fallback) |
| `GET /chat` | Chatbot UI |
| `POST /chat` | `{message, session_id?}` → `{reply, session_id, stats}` — `stats` = per-reply `{input_tokens, output_tokens, total_tokens, cost_usd, latency_s, model}` (rendered as the footer under each answer; cost = role-priced LLM tokens + Ranking API call + embedding, prices in `config.py`) |
| `POST /reset` | Clear a session's conversation history |
| `GET /health` | Status + active session count |

No authentication — the chatbot is public.

---

## Local development

### Requirements

- Python 3.10 via [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- GCP project with `aiplatform`, `discoveryengine`, `sqladmin` APIs enabled
- Cloud SQL Auth Proxy v2

### Setup

```bash
# 1. Google Cloud auth (no API keys anywhere)
gcloud auth application-default login
gcloud auth application-default set-quota-project wine-agent-jh-2026

# 2. Backend deps + secrets
cd backend
uv sync
# .env (not committed):
#   PG_CONNECTION=postgresql+psycopg://raguser:<pw>@127.0.0.1:5432/ragdb
#   LANGSMITH_TRACING=true                              # optional: tracing
#   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
#   LANGSMITH_API_KEY=<key>
#   LANGSMITH_PROJECT=bcit-chatbot

# 3. Database tunnel (separate terminal; 5433 + matching PG_CONNECTION on WSL)
cloud-sql-proxy --port 5432 wine-agent-jh-2026:us-west1:bcit-rag-pg

# 4. Run
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# Frontend dev server (hot reload, proxies API to :8000)
cd frontend && npm install && npm run dev    # http://localhost:5173
# or production build, served by FastAPI:    # http://localhost:8000
npm run build
```

---

## Production

### Topology

```
Browser ──HTTPS──▶ Cloudflare (proxied DNS, TLS termination, edge cache)
                      │  Origin Rule: destination port → 8000
                      ▼ HTTP :8000
              GCE bcit-rag-vm (us-west1-b, e2-standard-2)
                      │  systemd: bcit-chatbot.service (uvicorn :8000)
                      │  systemd: cloud-sql-proxy.service
                      ▼
              Cloud SQL PostgreSQL + pgvector  ·  Vertex AI APIs
```

### Deploying a code update

```bash
cd frontend && npm run build

gcloud compute scp backend/server.py bcit-rag-vm:/tmp/server.py --zone=us-west1-b
gcloud compute scp --recurse frontend/dist bcit-rag-vm:/tmp/dist-new --zone=us-west1-b

gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  sudo cp /tmp/server.py /opt/bcit-rag/backend/server.py &&
  sudo rm -rf /opt/bcit-rag/frontend/dist &&
  sudo mv /tmp/dist-new /opt/bcit-rag/frontend/dist &&
  sudo systemctl daemon-reload && sudo systemctl restart bcit-chatbot &&
  sleep 30 && curl -s localhost:8000/health'
```

### Refreshing the corpus (per-term runbook)

```bash
# 1. Crawl (local, ~85 min) — picks up new terms/courses automatically
cd backend && .venv/bin/python crawl_bcit.py
mv data data_old_<date> && mv data_new data

# 2. Build a NEW collection (production keeps serving the old one)
cloud-sql-proxy --port 5433 wine-agent-jh-2026:us-west1:bcit-rag-pg &
PG_CONNECTION=<...5433...> PG_COLLECTION=bcit_docs_<version> \
  DOCUMENTS_PICKLE=./vectorstore/documents_<version>.pkl \
  .venv/bin/python build_pgvector.py

# 3. Smoke-test against the new collection (same env vars + query_rag), then
#    flip BOTH defaults in config.py: PG_COLLECTION and DOCUMENTS_PICKLE
#    (they version together — dense and BM25 must serve the same crawl)

# 4. Deploy config + pickle (keep the versioned filename — it matches the
#    config default and leaves the previous pickle in place for rollback)
gcloud compute scp backend/config.py bcit-rag-vm:/tmp/config.py --zone=us-west1-b
gcloud compute scp vectorstore/documents_<version>.pkl bcit-rag-vm:/tmp/documents_<version>.pkl --zone=us-west1-b
gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  sudo cp /tmp/documents_<version>.pkl /opt/bcit-rag/backend/vectorstore/ &&
  sudo cp /tmp/config.py /opt/bcit-rag/backend/config.py &&
  sudo systemctl restart bcit-chatbot && sleep 30 && curl -s localhost:8000/health'

# 5. After verifying production, drop the old collection
#    (adapt backend/drop_old_collection.sh)
```

### Tracing (LangSmith)

Enabled 2026-06-10 with **zero code changes** — `langsmith` is already a
transitive dependency of `langchain-core`, so four env vars in `backend/.env`
(present in both the local and VM copies) switch it on:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<key — in .env, never committed>
LANGSMITH_PROJECT=bcit-chatbot
```

- Traces land in project `bcit-chatbot` at https://smith.langchain.com — one
  root run per query (see "How a query flows" for what each trace contains).
- Uploads are batched in a background thread; a LangSmith outage slows nothing
  and blocks no replies.
- To disable: set `LANGSMITH_TRACING=false` in the VM `.env`, then
  `sudo systemctl restart bcit-chatbot`.

### Operations

```bash
gcloud compute ssh bcit-rag-vm --zone=us-west1-b
sudo systemctl status bcit-chatbot
sudo journalctl -u bcit-chatbot -f
curl -s https://bcitai.ca/health
```
