# Backend

FastAPI service and retrieval pipeline. Full architecture, tuning history, and
runbooks live in the [repository README](../README.md).

| File | Role |
|---|---|
| `server.py` | FastAPI app — `/chat`, `/chat/stream` (SSE), `/metrics`, `/feedback`; serves the built frontend |
| `query_rag.py` | `BCITChatbot` — rewrite/decompose, retrieval, rerank, context assembly, generation |
| `hybrid_retriever.py` | BM25 + pgvector with RRF fusion, entity-scoped arm |
| `reranker.py` | Vertex AI Ranking API client |
| `embeddings.py` | Vertex AI embedding wrapper |
| `config.py` | Every tunable, with the measurement that set it |
| `response_cache.py` | First-turn exact-match answer cache |
| `crawl_bcit.py` | Corpus crawler (sitemaps + outlines API) |
| `build_pgvector.py` | Indexing job (resumable, collection-versioned) |
| `eval/` | Golden sets, offline harness, re-scorer, archived benchmarks |

Setup, credentials, and the Cloud SQL proxy are covered under
[Local development](../README.md#local-development). Dependencies are pinned in
`pyproject.toml` / `requirements.txt` and installed with `uv`; there is nothing
to install outside that file — embeddings, reranking, and generation are all
managed-service calls.
