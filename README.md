# BCIT AI Advisor — RAG Chatbot

**Live: [https://bcitai.ca](https://bcitai.ca)** — tech blog landing page at `/`, chatbot at [`/chat`](https://bcitai.ca/chat).

A retrieval-augmented generation (RAG) chatbot that answers questions about BCIT
programs, courses, admissions, and campus life, grounded in 5,000+ pages of
institutional documentation. The entire pipeline is **CPU-only**: embeddings,
vector search, reranking, and the LLM all run on GCP managed services — no GPU,
no API keys (Application Default Credentials end to end).

| Component | Service |
|---|---|
| Embeddings | Vertex AI `gemini-embedding-001` (1536-dim) |
| Vector store | Cloud SQL PostgreSQL + pgvector (HNSW) |
| Sparse retrieval | In-process BM25 (`rank_bm25`) |
| Rank fusion | Reciprocal Rank Fusion (alpha 0.5, k=60) |
| Reranker | Vertex AI Ranking API (`semantic-ranker-default-004`) |
| LLM | `gemini-3.5-flash` via LangChain `ChatVertexAI` |
| Backend | FastAPI + uvicorn (Python 3.10, managed via uv) |
| Frontend | React 19 + Vite 7 (no router — pathname-based pages) |
| Hosting | GCE `e2-standard-2` VM + Cloudflare (DNS, TLS, port routing) |

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

## Repository layout

```
backend/
  server.py            FastAPI app: /chat API + serves the built frontend
  query_rag.py         BCITChatbot — retrieval pipeline + LLM chain
  hybrid_retriever.py  BM25 + pgvector with RRF fusion
  reranker.py          Vertex AI Ranking API client
  embeddings.py        Vertex AI embedding wrapper
  config.py            All tunables (models, top-k, chunking, HNSW params)
  build_pgvector.py    One-time indexing job (resumable)
  data/                Source corpus (program + course descriptions)
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

### 1. Google Cloud auth (no API keys)

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project wine-agent-jh-2026
# On the GCE VM, the attached service account (bcit-rag-sa) is used automatically.
```

### 2. Backend

```bash
cd backend
uv sync                      # creates .venv with Python 3.10

# .env (not committed):
#   PG_CONNECTION=postgresql+psycopg://raguser:<password>@127.0.0.1:5432/ragdb
```

### 3. Database connection

```bash
# Separate terminal — all DB access goes through the proxy:
cloud-sql-proxy --port 5432 wine-agent-jh-2026:us-west1:bcit-rag-pg
```

### 4. Build the index (one-time, ~1 hour)

```bash
cd backend
uv run python build_pgvector.py
```

Chunks `data/` (1,024 chars, 130 overlap), writes `vectorstore/documents.pkl`
(BM25 source), embeds via Vertex AI, inserts into pgvector, and creates the
HNSW index. **Resumable**: re-running skips chunks already in the database.

### 5. Run

```bash
cd backend
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# Frontend dev server (hot reload, proxies API to :8000):
cd frontend && npm install && npm run dev

# Or serve the production build from FastAPI directly:
cd frontend && npm run build      # output: frontend/dist
```

Open http://localhost:8000 (built frontend) or http://localhost:5173 (dev).

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
- `bcit-chatbot.service` runs uvicorn from `/opt/bcit-rag/backend/.venv` and
  depends on `cloud-sql-proxy.service` (IAM-authenticated tunnel to Cloud SQL).

### Deploying an update

```bash
# Build locally
cd frontend && npm run build

# Upload (paths: app lives at /opt/bcit-rag, owned by root — stage via /tmp)
gcloud compute scp backend/server.py bcit-rag-vm:/tmp/server.py --zone=us-west1-b
gcloud compute scp --recurse frontend/dist bcit-rag-vm:/tmp/dist-new --zone=us-west1-b

# Install + restart + verify
gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  sudo cp /tmp/server.py /opt/bcit-rag/backend/server.py &&
  sudo rm -rf /opt/bcit-rag/frontend/dist &&
  sudo mv /tmp/dist-new /opt/bcit-rag/frontend/dist &&
  sudo systemctl daemon-reload && sudo systemctl restart bcit-chatbot &&
  sleep 30 && curl -s localhost:8000/health'
```

### Operations

```bash
gcloud compute ssh bcit-rag-vm --zone=us-west1-b
sudo systemctl status bcit-chatbot
sudo journalctl -u bcit-chatbot -f
curl -s https://bcitai.ca/health
```
