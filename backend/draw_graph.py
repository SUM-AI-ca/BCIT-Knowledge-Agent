"""Draw the BCIT Knowledge Agent architecture: edge, controller graph, retrieval.

The diagram is generated from this file, so it is the one place to edit when the
architecture changes. Two artifacts are written to the repo root and must not
drift apart:

    graph_mermaid.md   the Mermaid source (paste into mermaid.live)
    graph.png          the rendered diagram the README embeds

Every number below is a live default from `config.py`. `main()` re-reads config
and refuses to write if any of them has moved, so the diagram cannot quietly
fall behind the code it describes.

Usage:   .venv/bin/python draw_graph.py
"""
import base64
import json
import os
import urllib.request
import zlib

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── The numbers the diagram prints, checked against config.py at build time ──
# name -> (config attribute, value as it appears in the diagram text)
_ASSERTED = {
    "GEMINI_MODEL": "gemini-3.5-flash-lite",
    "REWRITER_MODEL": "gemini-3.7-flash",
    "GRAPH_MODEL": "gemini-3.5-flash-lite",
    "EMBEDDING_MODEL": "gemini-embedding-001",
    "EMBEDDING_DIMENSIONS": 1536,
    "EMBEDDING_LOCATION": "us-central1",
    "GEMINI_TEMPERATURE": 0.05,
    "GEMINI_MAX_OUTPUT_TOKENS": 2048,
    "GEMINI_THINKING_LEVEL": "minimal",
    "REWRITE_THINKING_LEVEL": "low",
    "REWRITE_MAX_OUTPUT_TOKENS": 2048,
    "GRAPH_TEMPERATURE": 0.0,
    "GRAPH_MAX_OUTPUT_TOKENS": 1024,
    "GRAPH_THINKING_LEVEL": "minimal",
    "GRAPH_MAX_HOPS": 3,
    "GRAPH_HOP_TOP_K": 20,
    "GRAPH_DIGEST_CHARS": 400,
    "GRAPH_HOP_CONTEXT_MAX_CHARS": 48000,
    "HYBRID_ALPHA": 0.48,
    "RRF_K": 60,
    "MMR_LAMBDA": 0.87,
    "RETRIEVAL_FETCH_K": 50,
    "RETRIEVAL_TOP_K": 10,
    "RETRIEVAL_DENSE_K": 23,
    "RETRIEVAL_BM25_K": 23,
    "MAX_SUB_QUERIES": 4,
    "MQ_DENSE_K": 12,
    "MQ_BM25_K": 12,
    "MQ_CANDIDATES_PER_SUBQUERY": 15,
    "MQ_POOL_CAP": 40,
    "MQ_MIN_CHUNKS_PER_SUBQUERY": 2,
    "RERANKER_MODEL": "semantic-ranker-default-004",
    "RERANKER_CANDIDATES": 25,
    "RERANKER_TOP_K": 10,
    "ENTITY_SCOPED_K": 8,
    "ENTITY_SCOPED_MAX_ENTITIES": 3,
    "PERSON_SCOPED_K": 2,
    "PERSON_SCOPED_MAX_SOURCES": 12,
    "NEIGHBOR_RADIUS": 2,
    "CONTEXT_MAX_CHARS": 24000,
    "MIN_CONTEXT_CHUNKS": 3,
    "MEMORY_WINDOW_K": 5,
    "CHAT_TIMEOUT_S": 90,
    "WORKER_THREADS": 4,
    "MAX_MESSAGE_CHARS": 2000,
    "RESPONSE_CACHE_MAX": 1000,
    "RESPONSE_CACHE_TTL_S": 86400,
    "HISTORY_MAX_ANSWER_CHARS": 1500,
    "PG_COLLECTION": "bcit_docs_202606da",
}


MERMAID = """\
---
config:
  flowchart:
    curve: basis
    nodeSpacing: 70
    rankSpacing: 85
    padding: 16
  themeVariables:
    fontSize: 15px
---
graph TD
    %% ── Edge: browser → Cloudflare → GCE → FastAPI ──────────────────────
    USER(["Student · bcitai.ca"])
    FE["Frontend · React 19 + Vite 7<br/>pathname-based pages, no router<br/>blog at / · chat at /chat<br/>served as static files by FastAPI"]
    CF["Cloudflare · zone bcitai.ca<br/>proxied DNS · TLS termination (Flexible)<br/>Origin Rule: destination port → 8000"]
    VM["GCE bcit-rag-vm · us-west1-b · e2-medium<br/>firewall tcp:8000 from Cloudflare IPv4 only<br/>systemd: bcit-chatbot (uvicorn :8000)<br/>systemd: cloud-sql-proxy"]
    API["FastAPI · server.py<br/>POST /chat · POST /chat/stream (SSE)<br/>GET /health · /metrics · POST /feedback · /reset"]
    USER --> FE --> CF --> VM --> API

    %% ── Per-request guards, in the order server.py applies them ─────────
    GV{"validate_chat_request<br/>503 not initialized · 400 empty<br/>422 over 2,000 chars"}
    POOL["ThreadPoolExecutor · 4 workers<br/>the pipeline is API-bound, so threads &gt; vCPUs<br/>a wedged Vertex call holds its thread to completion"]
    TMO{"asyncio.wait_for · 90 s<br/>504 releases the REQUEST only<br/>the worker thread is uncancellable"}
    API --> GV --> POOL --> TMO

    MEM[("SessionMemory · window k=5<br/>in-process, 30 min idle expiry<br/>history stores answers Sources-stripped, ≤1,500 chars")]
    CACHE{"Response cache · first turn only<br/>exact normalized match, never semantic<br/>LRU 1,000 · TTL 24 h · cleared on restart"}
    HIT(["cached answer<br/>&lt;100 ms · $0 · no LLM call"])
    TMO --> CACHE
    CACHE -- "hit" --> HIT
    CACHE -- "miss" --> CTRL
    TMO -. "follow-up turns bypass the cache" .-> MEM
    MEM -. "chat_history" .-> CTRL

    %% ── Controller · graph.py · USE_GRAPH=true ──────────────────────────
    %% One LLM node, then two guards that graph.py enforces in CODE rather
    %% than asking of the prompt. That distinction is the whole reason the
    %% gate can be trusted, so the drawing keeps the guards separate.
    CTRL["controller_llm · graph.py · gemini-3.5-flash-lite<br/>temp 0 · thinking minimal · max 1,024 out<br/>schema-constrained JSON: action · reason · queries · missing<br/>———<br/>ONE node, called once per iteration:<br/>iteration 0 has no evidence → the decision IS the route<br/>iteration 1+ sees the digest → the decision IS the coverage gate"]

    CONC{"is_concrete(missing)<br/>enforced in graph.py, not in the prompt:<br/>a hop must name a course code, program<br/>or person — a vague 'missing' answers instead"}
    CAP{"hop cap ≤ 3<br/>enforced on the graph edge"}

    DIRECT["direct_prompt · no retrieval, no Sources<br/>ONE prompt serves both 'answer' and 'refuse';<br/>only the logged route label differs. The scope<br/>policy lives here, so skipping retrieval never<br/>becomes 'answer BCIT questions from memory'"]
    RW["Rewrite + decompose · gemini-3.7-flash<br/>temp 0 · thinking low · max 2,048 out<br/>schema-constrained JSON<br/>→ standalone question + 1–4 sub-queries<br/>direct/refuse turns never pay for it"]
    HOP["Hop retrieve · top_k 20<br/>context budget 48,000 chars<br/>digest 400 chars/source feeds the next gate"]

    CTRL -- "answer / refuse" --> DIRECT
    CTRL -- "retrieve · hop 0" --> RW
    CTRL -- "retrieve · hop 1+" --> CONC
    CONC -- "concrete" --> CAP
    CONC -- "vague" --> GEN
    CAP -- "under cap" --> HOP
    CAP -- "at cap" --> GEN
    DIRECT --> GEN
    RW --> FAN
    HOP --> FAN

    %% ── Retrieval: per-sub-query fan-out, four arms into one pool ───────
    subgraph RET ["Retrieval · one thread per sub-query · four arms merged and deduped"]
        FAN{"multipart?"}
        DENSE["Dense · pgvector HNSW<br/>MMR λ 0.87 · fetch 50<br/>k=12 fan-out / 23 full-width"]
        BM["Sparse · in-process BM25 (rank_bm25)<br/>index fit on title + category<br/>+ filename keywords + URL slug<br/>k=12 fan-out / 23 full-width"]
        ENT["Entity-scoped arm<br/>course code or program named → k=8 per source<br/>≤3 entities · merged round-robin"]
        PER["Person-scoped arm<br/>instructor named → k=2 per source, ≤12 sources<br/>indexes the INSTRUCTOR relation only,<br/>not approval signatures"]
        FAN -- "yes · fan out" --> DENSE
        FAN -- "no · full width" --> DENSE
        FAN --> BM
        FAN -.-> ENT
        FAN -.-> PER
        RRF["RRF fusion · α 0.48 · k 60<br/>rank position, not score — dense cosine<br/>and BM25 scores are incomparable"]
        DENSE --> RRF
        BM --> RRF
        ENT --> RRF
        PER --> RRF
        POOLM["Pooled candidates · cap 40<br/>≥2 chunks kept per sub-query (coverage quota)<br/>15 candidates per sub-query"]
        RRF --> POOLM
    end

    %% ── Rerank ──────────────────────────────────────────────────────────
    RR["Vertex AI Ranking API · semantic-ranker-default-004<br/>pooled: ONE billed query per turn (≤100 records)<br/>25 candidates → top 10<br/>page identity rides in the title field"]
    HEDGE["Hedged request · RERANK_HEDGE_AFTER_S=0.4 in the VM .env<br/>(code default 0 = off) · duplicate call after 0.4 s,<br/>first to land wins — the loser is a duplicate, not a fallback"]
    POOLM --> RR
    RR -. "slow tail" .-> HEDGE

    %% ── Context assembly ────────────────────────────────────────────────
    CTX["Context · small-to-big<br/>neighbor expansion ±2 from the ordinal index<br/>render-and-shrink cap 24,000 chars<br/>floor 3 chunks"]
    RR --> CTX
    CTX -. "gate: is this enough?" .-> CTRL

    %% ── Generation ──────────────────────────────────────────────────────
    GEN["Generation · gemini-3.5-flash-lite<br/>temp 0.05 · thinking minimal · max 2,048 out<br/>RESPONSE_LANGUAGE=match: answers in the student's<br/>language, retrieval + rewrite stay English"]
    CTX --> GEN

    OUT["Answer + Sources<br/>/chat → JSON · /chat/stream → SSE<br/>events: session · delta · done · error"]
    GEN --> OUT
    OUT --> FE
    OUT -. "write turn back" .-> MEM

    %% ── Stores and managed services ─────────────────────────────────────
    PG[("Cloud SQL PostgreSQL + pgvector<br/>bcit-rag-pg · us-west1 · db ragdb<br/>collection bcit_docs_202606da<br/>100,515 chunks from 11,129 docs · HNSW")]
    PKL[("documents_202606.pkl · in-process<br/>feeds BM25 and the neighbor ordinal index<br/>byte-identical to the stored chunks")]
    EIDX[("Entity index · built at startup, ~0.07 s<br/>7,623 course codes · 498 programs<br/>1,291 instructors + 76 partial-name aliases")]
    DENSE --> PG
    BM --> PKL
    ENT --> EIDX
    PER --> EIDX

    EMB["gemini-embedding-001 · 1536-dim MRL<br/>us-central1 — embeddings are regional,<br/>chat runs at location global"]
    DENSE -- "query embedding" --> EMB

    %% ── Ops ─────────────────────────────────────────────────────────────
    LS["LangSmith tracing<br/>project bcit-chatbot"]
    LOG["journald · query_usage JSON per turn<br/>tokens · cost · latency · route · hops<br/>feedback_log on thumbs up/down"]
    API -. "traces" .-> LS
    OUT -. "cost + latency" .-> LOG
    LOG -. "thumbs-down cases seed the golden set" .-> EVAL
    EVAL["eval/run_eval.py · offline<br/>golden sets v1/v2/v3 + people<br/>--judge on gemini-3.7-flash"]

    %% ── Palette ─────────────────────────────────────────────────────────
    classDef gate fill:#fff3cd,stroke:#856404,color:#856404
    classDef llm fill:#d4edda,stroke:#155724,color:#155724
    classDef retr fill:#cce5ff,stroke:#004085,color:#004085
    classDef store fill:#e2e3e5,stroke:#383d41,color:#383d41
    classDef infra fill:#f8d7da,stroke:#721c24,color:#721c24
    classDef edge fill:#d1ecf1,stroke:#0c5460,color:#0c5460
    class GV,TMO,CACHE,CONC,CAP,FAN gate
    class HIT edge
    class CTRL,RW,GEN,DIRECT,EMB llm
    class DENSE,BM,ENT,PER,RRF,POOLM,RR,CTX,HOP retr
    class PG,PKL,EIDX,MEM store
    class LS,LOG,EVAL,HEDGE,POOL infra
    class FE,CF,VM,API,OUT edge
"""


def check_against_config() -> list:
    """Return a list of diagram numbers that no longer match config.py."""
    import config

    drift = []
    for name, expected in _ASSERTED.items():
        actual = getattr(config, name, "<missing>")
        if actual != expected:
            drift.append(f"{name}: diagram says {expected!r}, config.py says {actual!r}")
    return drift


def render_png(mermaid_text: str) -> bytes:
    """Render via mermaid.ink's `pako:` form (deflate + base64url).

    Same approach as the wine project's draw_graph.py: the plain-base64 form
    puts the whole diagram in the URL and overruns the server's request-URI
    limit (HTTP 414). Deflating first cuts it to roughly a third.
    """
    payload = json.dumps(
        {"code": mermaid_text, "mermaid": {"theme": "default"}}
    ).encode("utf-8")
    compressor = zlib.compressobj(9, zlib.DEFLATED, 15)
    deflated = compressor.compress(payload) + compressor.flush()
    encoded = base64.urlsafe_b64encode(deflated).decode("ascii").rstrip("=")
    url = f"https://mermaid.ink/img/pako:{encoded}?type=png&bgColor=!white"
    req = urllib.request.Request(url, headers={"User-Agent": "draw_graph/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main():
    drift = check_against_config()
    if drift:
        print("REFUSING to write — the diagram no longer matches config.py:")
        for line in drift:
            print(f"  {line}")
        print("\nFix the label (or _ASSERTED) and re-run.")
        raise SystemExit(1)
    print(f"config check: all {len(_ASSERTED)} asserted values match config.py")

    md_path = os.path.join(_REPO_ROOT, "graph_mermaid.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(MERMAID)
    print(f"Mermaid saved -> graph_mermaid.md ({len(MERMAID):,} chars)")

    try:
        png = render_png(MERMAID)
    except Exception as e:
        print(f"PNG render failed ({type(e).__name__}: {e})")
        print("Paste graph_mermaid.md into https://mermaid.live to view")
        return

    png_path = os.path.join(_REPO_ROOT, "graph.png")
    with open(png_path, "wb") as f:
        f.write(png)
    print(f"PNG saved -> graph.png ({len(png):,} bytes)")


if __name__ == "__main__":
    main()
