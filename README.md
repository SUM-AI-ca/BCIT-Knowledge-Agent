# BCIT AI Advisor — RAG Chatbot

**Live: [https://bcitai.ca](https://bcitai.ca)** — tech blog landing page at `/`, chatbot at [`/chat`](https://bcitai.ca/chat).

A retrieval-augmented generation (RAG) chatbot that answers questions about BCIT
programs, courses, admissions, and campus life, grounded in **11,129 documents
crawled from the live BCIT website** (September 2024 intake onwards). The entire
pipeline is **CPU-only**: embeddings, vector search, reranking, and the LLM all
run on GCP managed services — no GPU, no API keys (Application Default
Credentials end to end).

| Component | Service |
|---|---|
| Corpus | Own crawler (`crawl_bcit.py`) — sitemaps + BCIT outlines API |
| Embeddings | Vertex AI `gemini-embedding-001` (1536-dim) |
| Vector store | Cloud SQL PostgreSQL + pgvector (HNSW) |
| Sparse retrieval | In-process BM25 (`rank_bm25`) |
| Rank fusion | Reciprocal Rank Fusion (alpha 0.5, k=60) |
| Reranker | Vertex AI Ranking API (`semantic-ranker-default-004`) |
| LLM | `gemini-3.5-flash` via LangChain `ChatVertexAI` |
| Backend | FastAPI + uvicorn (Python 3.10, managed via uv) |
| Frontend | React 19 + Vite 7 (no router — pathname-based pages) |
| Hosting | GCE `e2-standard-2` VM + Cloudflare (DNS, TLS, port routing) |

## The corpus

Everything is crawled from the live site, so catalog content is current by
construction. The one place a date policy applies is course outlines, which are
inherently term-bound:

| Category | Docs | Freshness policy |
|---|---|---|
| Course outlines | 3,262 | **Terms ≥ 202430 (Sept 2024), latest term per course only.** Multiple terms of the same course would surface contradictory schedules/instructors at retrieval time. |
| Course catalog pages | 7,060 | Live page = current truth; no date filter. |
| Program pages | 529 | Same. |
| Admission / international / student-services pages | 236 | Same, scoped to 10 path prefixes. |
| BCIT Student Association (bcitsa.ca) | 42 | Separate WordPress site, own sitemap. |

Two findings that shaped the crawler — written down so nobody re-learns them:

- **Sitemap `lastmod` is not a freshness signal here.** 78% of course pages
  carry a 2022 CMS-migration timestamp while the live content is current.
  Filtering on `lastmod ≥ 2024-09` would have silently dropped 89% of courses.
- **Outline discovery is one API call per term**:
  `GET /wp-json/bcit/outlines/v1/load_subjects_term/{term}` returns the full
  term catalog (subject → course → CRN). Outline URL is `/outlines/{term}{crn}/`.
  Term codes: `10`=Jan, `20`=Apr, `30`=Sept (so `202430` = Fall 2024).

### Crawling / refreshing the corpus

```bash
cd backend
.venv/bin/python crawl_bcit.py                  # full crawl into data_new/ (~85 min)
.venv/bin/python crawl_bcit.py --phase outlines # or a single phase:
                                                # worklist|outlines|courses|programs|pages|bcitsa
CRAWL_OUT=./data .venv/bin/python crawl_bcit.py --phase bcitsa   # write into a custom dir
```

Polite by design (3 concurrent requests, jittered delay, retries) and resumable —
existing non-empty files are skipped, so a crashed crawl continues where it left
off. Failures land in `data_new/_state/errors.log`.

## How a query flows

1. **Embed** — the question is embedded with Vertex AI `gemini-embedding-001`
   (1536 dimensions).
2. **Hybrid retrieval** — two retrievers run in parallel:
   - *Dense*: pgvector HNSW search in Cloud SQL. `ef_search` is raised from
     pgvector's default 40 to 100 per session — the default silently caps the
     candidate pool below MMR's `fetch_k=50`.
   - *Sparse*: in-process BM25 over the same chunks (exact identifiers like
     "COMP 1510" that embeddings miss).

   Results merge via **Reciprocal Rank Fusion**: each document scores
   `1 / (60 + rank)` per list, weighted 50/50 between retrievers. RRF compares
   ranks, not raw scores, so incompatible similarity scales never need
   normalizing.
3. **Rerank** — the candidate pool (13 chunks) goes to the Vertex AI Ranking
   API (`semantic-ranker-default-004`), a managed cross-encoder that re-scores
   each candidate against the query and keeps the top 10.
4. **Generate** — `gemini-3.5-flash` answers with the reranked chunks as
   grounding context.

Conversation state: each browser session holds a 5-turn
`ConversationBufferWindowMemory`, so follow-up questions resolve correctly.
Sessions expire after 30 minutes of inactivity. Chat requests run in a thread
pool so the FastAPI event loop never blocks on a RAG query.

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
  database, so adding 42 documents later embedded only the 365 new chunks.
- **Collection-salted ids** — `langchain_pg_embedding` upserts on id across
  *all* collections. Without the collection salt, rebuilding from same-named
  source files would hijack rows out of the live collection mid-build.
- **Blue-green cutover** — build into a fresh versioned collection while
  production serves the old one, flip the `PG_COLLECTION` default in
  `config.py`, deploy config + pickle, restart. Zero downtime; the old
  collection doubles as instant rollback until you drop it.

## Repository layout

```
backend/
  server.py            FastAPI app: /chat API + serves the built frontend
  query_rag.py         BCITChatbot — retrieval pipeline + LLM chain
  hybrid_retriever.py  BM25 + pgvector with RRF fusion
  reranker.py          Vertex AI Ranking API client
  embeddings.py        Vertex AI embedding wrapper
  config.py            All tunables (models, top-k, chunking, collection)
  crawl_bcit.py        Corpus crawler (sitemaps + outlines API)
  build_pgvector.py    Indexing job (resumable, collection-versioned)
  data/                Crawled corpus (11,129 txt docs, 16 categories)
  vectorstore/         documents.pkl — BM25 source (generated, not committed)
frontend/
  src/App.jsx          Pathname routing: "/" → Blog, "/chat" → Chat
  src/Blog.jsx         Tech blog landing page (architecture deep dive)
  vite.config.js       Dev proxy: POST /chat → backend; GET /chat → SPA
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
# .env (not committed):  PG_CONNECTION=postgresql+psycopg://raguser:<pw>@127.0.0.1:5432/ragdb

# 3. Database tunnel (separate terminal; use --port 5433 + matching
#    PG_CONNECTION if a local postgres already owns 5432)
cloud-sql-proxy --port 5432 wine-agent-jh-2026:us-west1:bcit-rag-pg

# 4. Run
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# Frontend dev server (hot reload, proxies API to :8000)
cd frontend && npm install && npm run dev    # http://localhost:5173
# or production build, served by FastAPI:    # http://localhost:8000
npm run build
```

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

- App lives at `/opt/bcit-rag/` on the VM (`backend/` + `frontend/dist`).
- GCP firewall opens **only tcp:8000** to the internet (tag `bcit-rag`);
  Cloudflare connects to the origin on 8000 directly via an Origin Rule
  ("Change port" template), so no reverse proxy is needed on the VM.
- Cloudflare SSL mode: Flexible (HTTPS at the edge, HTTP to origin).

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

# 5. After verifying production, drop the old collection (frees ~130k rows)
```

### Operations

```bash
gcloud compute ssh bcit-rag-vm --zone=us-west1-b
sudo systemctl status bcit-chatbot
sudo journalctl -u bcit-chatbot -f
curl -s https://bcitai.ca/health
```
