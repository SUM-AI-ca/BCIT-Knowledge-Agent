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
| LLM | `gemini-3.5-flash` via LangChain `ChatVertexAI` |
| Backend | FastAPI + uvicorn (Python 3.10, managed via uv) |
| Frontend | React 19 + Vite 7 (no router — pathname-based pages) |
| Hosting | GCE `e2-standard-2` VM + Cloudflare (DNS, TLS, port routing) |
| Observability | LangSmith tracing (project `bcit-chatbot`) |

---

## How a query flows

1. **Rewrite** — follow-up questions are rewritten into standalone queries using
   the conversation history (`QUERY_REWRITE_TEMPLATE` in `config.py`).
2. **Embed** — the query is embedded with Vertex AI `gemini-embedding-001`
   (1536 dimensions).
3. **Hybrid retrieval** — two retrievers run in parallel:
   - *Dense*: pgvector HNSW search in Cloud SQL (MMR, `fetch_k=50`,
     `ef_search=100`).
   - *Sparse*: in-process BM25 over the same chunks (exact identifiers like
     "COMP 1510" that embeddings miss).

   Results merge via **Reciprocal Rank Fusion**: each document scores
   `1 / (60 + rank)` per list, weighted by `HYBRID_ALPHA=0.48` between
   retrievers. RRF compares ranks, not raw scores, so incompatible similarity
   scales never need normalizing.
4. **Rerank** — the candidate pool (13 chunks) goes to the Vertex AI Ranking
   API (`semantic-ranker-default-004`), a managed cross-encoder that re-scores
   each candidate against the query and keeps the top 10.
5. **Generate** — `gemini-3.5-flash` answers with the reranked chunks as
   grounding context, citing source URLs.

Conversation state: each browser session holds a 5-turn
`ConversationBufferWindowMemory`; sessions expire after 30 minutes. Chat
requests run in a thread pool so the FastAPI event loop never blocks.

Every query produces one LangSmith trace (project `bcit-chatbot`): the root
`RunnableSequence` nests the query-rewrite LLM call, the hybrid retriever run
(with the returned chunks), and the generation call. The Vertex reranker is a
plain HTTP client rather than a Runnable, so its latency is included in the
context-assembly step instead of getting its own span.

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
  production serves the old one, flip the `PG_COLLECTION` default in
  `config.py`, deploy config + pickle, restart. Zero downtime; the old
  collection doubles as instant rollback until you drop it
  (`drop_old_collection.sh` pattern).

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

Pending niceties: `www` CNAME is not set up; "Always Use HTTPS" is off
(http://bcitai.ca serves without redirect).

---

## Future work — where to start

Ideas that fit the current architecture, with their natural hook points:

- **Per-source incremental refresh** — `build_pgvector.py` currently
  appends-or-skips; add a "delete rows for source X, reinsert" mode so a
  changed page can be re-embedded without a new collection. Hook:
  `existing_ids()` / `deterministic_ids()`.
- **Scheduled term refresh** — the whole refresh is two commands (runbook
  below); wire into cron/GitHub Actions once credentials story is decided.
- **Retrieval evaluation harness** — no golden set exists. Build ~50 Q/A pairs
  (mix: outline facts, program requirements, BCITSA), measure retrieval
  hit-rate and answer faithfulness before touching retrieval params
  (`HYBRID_ALPHA`, `RERANKER_*` in `config.py`).
- **Term-aware retrieval** — outline chunks carry `term_code` metadata already;
  boosting newer terms (or filtering expired ones at query time) is a metadata
  filter away.
- **Streaming responses** — `ChatVertexAI` supports streaming; `/chat` returns
  a single JSON blob today. Needs SSE endpoint + frontend incremental render.
- **Citations UI** — answers already end with a `Sources` section (prompt-enforced);
  the frontend renders it as plain text. Parse and render as cards/links.
- **Observability** — LangSmith tracing (June 2026) gives per-query traces
  with stage latency and token counts; logs are still print statements.
  Remaining: wrap `VertexRanker.rerank` in `@traceable` so rerank latency gets
  its own span, structured logging, and a `/metrics` endpoint.
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
  data/                    Crawled corpus (11,129 txt docs, 16 categories)
  vectorstore/             documents.pkl — BM25 source (generated, not committed)
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
| `POST /chat` | `{message, session_id?}` → `{reply, session_id}` |
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

# 3. Smoke-test against the new collection (same env vars + query_rag),
#    then flip the PG_COLLECTION default in config.py

# 4. Deploy config + pickle, restart
gcloud compute scp backend/config.py bcit-rag-vm:/tmp/config.py --zone=us-west1-b
gcloud compute scp vectorstore/documents_<version>.pkl bcit-rag-vm:/tmp/documents.pkl --zone=us-west1-b
gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  cd /opt/bcit-rag/backend/vectorstore && sudo cp documents.pkl documents_old.pkl &&
  sudo cp /tmp/config.py /opt/bcit-rag/backend/config.py &&
  sudo cp /tmp/documents.pkl documents.pkl &&
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
