# Backend

FastAPI service and retrieval pipeline. Full architecture, tuning history, and
runbooks live in the [repository README](../README.md).

| File | Role |
|---|---|
| `server.py` | FastAPI app — `/chat`, `/chat/stream` (SSE), `/metrics`, `/feedback`; serves the built frontend |
| `query_rag.py` | `BCITChatbot` — rewrite/decompose, retrieval, rerank, context assembly, generation |
| `graph.py` | LangGraph controller (`USE_GRAPH`) — one LLM node serving route, coverage gate and orchestration, plus the two guards that must be code rather than prompt |
| `session_memory.py` | Windowed conversation memory; replaces `langchain.memory`, removed in langchain 1.x. Window is k **turns**, not k messages |
| `hybrid_retriever.py` | BM25 + pgvector with RRF fusion, entity-scoped arm |
| `reranker.py` | Vertex AI Ranking API client |
| `embeddings.py` | Vertex AI embedding wrapper |
| `config.py` | Every tunable, with the measurement that set it |
| `response_cache.py` | First-turn exact-match answer cache |
| `crawl_bcit.py` | Corpus crawler (sitemaps + outlines API) |
| `build_pgvector.py` | Indexing job (resumable, collection-versioned) |
| `deploy.sh` | Deploy backend modules to the production VM — use this rather than ad-hoc `gcloud` commands. `--deps` also rebuilds the VM venv, and is required whenever `requirements.txt` or `.python-version` changed |
| `.python-version` | Interpreter version (3.13). Read by uv locally and by `deploy.sh --deps` on the VM, so the two cannot drift |
| `drop_old_collection.sh` | One-off: drop a retired pgvector collection through the proxy |
| `eval/` | Golden sets, offline harness, re-scorer, archived benchmarks |
| `eval/person_lookup.py` | "what courses does X teach?" harness — set comparison, because substring matching cannot see role misattribution |
| `eval/golden_set_graph.jsonl` | 16 cases in untidy phrasing, each declaring `expected_route` so mis-routes are scored rather than inferred |

Setup, credentials, and the Cloud SQL proxy are covered under
[Local development](../README.md#local-development). Dependencies are pinned in
`pyproject.toml` / `requirements.txt` and installed with `uv`. Embeddings,
reranking and generation are all managed-service calls, so almost everything in
there is a client library; the exception is **langgraph**, which runs the
controller graph in-process. A change to `requirements.txt` or `.python-version`
means the VM venv must be rebuilt — deploy with `bash deploy.sh --deps ...`,
never a plain deploy.
