import os
import json
import logging
import threading
import time
import warnings
import uuid
from collections import deque
from typing import Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("bcit.server")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from query_rag import BCITChatbot
from config import (
    MEMORY_WINDOW_K,
    CHAT_TIMEOUT_S,
    WORKER_THREADS,
    MAX_MESSAGE_CHARS,
)
from session_memory import SessionMemory

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatStats(BaseModel):
    """Per-reply transparency footer: what this answer actually consumed."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    currency: str = "USD"  # Google Cloud bills in USD
    latency_s: float
    model: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    stats: Optional[ChatStats] = None


chatbot: Optional[BCITChatbot] = None
sessions: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)

SESSION_TIMEOUT_MINUTES = 30
# Backstop between the 5-minute cleanup ticks: a burst of fresh session ids
# (each ~a few KB of memory) triggers an inline sweep instead of growing
# unbounded until the next tick.
MAX_SESSIONS_BEFORE_SWEEP = 5000


def get_or_create_session(session_id: Optional[str]) -> tuple[str, SessionMemory]:
    if not session_id:
        session_id = str(uuid.uuid4())

    now = datetime.now()

    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS_BEFORE_SWEEP:
            cleanup_expired_sessions()
        sessions[session_id] = {
            "memory": SessionMemory(
                k=MEMORY_WINDOW_K,
                memory_key="chat_history",
                return_messages=True
            ),
            "last_access": now
        }
        logger.info("[Session] Created new session: %s...", session_id[:8])
    else:
        sessions[session_id]["last_access"] = now
    
    return session_id, sessions[session_id]["memory"]


def cleanup_expired_sessions():
    now = datetime.now()
    timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    expired_sessions = [
        sid for sid, data in sessions.items()
        if now - data["last_access"] > timeout
    ]
    
    for sid in expired_sessions:
        del sessions[sid]

    return len(expired_sessions)


def get_session_stats():
    return {
        "active_sessions": len(sessions),
        "timeout_minutes": SESSION_TIMEOUT_MINUTES
    }


# Lightweight in-process metrics for GET /metrics: totals since boot plus a
# ring buffer for last-hour percentiles. Deliberately not persisted — the
# durable signals live in the query_usage log lines and LangSmith traces.
SERVER_STARTED = datetime.now(timezone.utc)
_metrics_lock = threading.Lock()
_recent_queries = deque(maxlen=2000)
_metrics_totals = {
    "queries": 0, "errors": 0, "timeouts": 0,
    "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}


def record_query_metric(meta: Optional[dict] = None, *, error: bool = False, timeout: bool = False):
    usage = (meta or {}).get("usage") or {}
    entry = {
        "ts": time.time(),
        "latency_s": ((meta or {}).get("timings") or {}).get("total_s"),
        "cost_usd": (meta or {}).get("est_cost_usd") or 0.0,
        "error": error,
    }
    with _metrics_lock:
        _recent_queries.append(entry)
        _metrics_totals["queries"] += 1
        if error:
            _metrics_totals["errors"] += 1
        if timeout:
            _metrics_totals["timeouts"] += 1
        _metrics_totals["input_tokens"] += (
            (usage.get("input_tokens") or 0) + (usage.get("rewrite_input_tokens") or 0)
        )
        _metrics_totals["output_tokens"] += (
            (usage.get("output_tokens") or 0)
            + (usage.get("reasoning_tokens") or 0)
            + (usage.get("rewrite_output_tokens") or 0)
        )
        _metrics_totals["cost_usd"] += entry["cost_usd"]


def query_chatbot_sync(question: str, memory: SessionMemory) -> dict:
    # Memory is passed per request — never swap chatbot.memory globally,
    # concurrent requests would leak history across sessions.
    return chatbot.query_with_meta(question, memory=memory)


async def query_chatbot_async(question: str, memory: SessionMemory) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, query_chatbot_sync, question, memory)


def build_stats(meta: dict) -> ChatStats:
    usage = meta.get("usage") or {}
    tokens_in = (usage.get("input_tokens", 0) or 0) + (usage.get("rewrite_input_tokens", 0) or 0)
    tokens_out = (
        (usage.get("output_tokens", 0) or 0)
        + (usage.get("reasoning_tokens", 0) or 0)
        + (usage.get("rewrite_output_tokens", 0) or 0)
    )
    return ChatStats(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=meta.get("est_cost_usd") or 0.0,
        latency_s=(meta.get("timings") or {}).get("total_s", 0.0),
        model=(meta.get("models") or {}).get("generation", ""),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    
    print("\n" + "=" * 60)
    print("Starting BCIT Chatbot Server...")
    print("=" * 60 + "\n")
    
    try:
        chatbot = BCITChatbot()
        print("\n" + "=" * 60)
        print("Chatbot loaded successfully!")
        print(f"Session timeout: {SESSION_TIMEOUT_MINUTES} minutes")
        print("Server ready at http://localhost:8000")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Failed to load chatbot: {e}")
        raise
    
    async def cleanup_task():
        while True:
            await asyncio.sleep(300)
            cleanup_expired_sessions()
    
    task = asyncio.create_task(cleanup_task())
    
    yield
    
    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    print("\nShutting down")
    executor.shutdown(wait=True)
    sessions.clear()


app = FastAPI(
    title="BCIT Academic Advisor Chatbot",
    description="RAG-based chatbot for BCIT students",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "https://bcitai.ca", "https://www.bcitai.ca"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    stats = get_session_stats()
    return {
        "status": "healthy",
        "chatbot_loaded": chatbot is not None,
        **stats
    }


@app.get("/metrics")
async def metrics():
    """Aggregate usage since boot + last-hour latency percentiles. Public on
    purpose: nothing here is sensitive (no questions, no session contents)."""
    cutoff = time.time() - 3600
    with _metrics_lock:
        totals = dict(_metrics_totals)
        recent = [e for e in _recent_queries if e["ts"] >= cutoff]

    latencies = sorted(e["latency_s"] for e in recent if e.get("latency_s") is not None)

    def pct(q):
        if not latencies:
            return None
        return round(latencies[int(round(q * (len(latencies) - 1)))], 3)

    return {
        "started_at": SERVER_STARTED.isoformat(),
        "totals": {**totals, "cost_usd": round(totals["cost_usd"], 4)},
        "last_hour": {
            "queries": len(recent),
            "errors": sum(1 for e in recent if e["error"]),
            "latency_p50_s": pct(0.5),
            "latency_p95_s": pct(0.95),
            "cost_usd": round(sum(e["cost_usd"] for e in recent), 4),
        },
        "response_cache": (
            chatbot.response_cache.stats()
            if chatbot and getattr(chatbot, "response_cache", None) else None
        ),
        **get_session_stats(),
    }


def validate_chat_request(request: ChatRequest):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(request.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Message too long (max {MAX_MESSAGE_CHARS} characters)",
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    validate_chat_request(request)

    # Get or create session
    session_id, memory = get_or_create_session(request.session_id)

    try:
        logger.info("[Query] Session %s...: %s", session_id[:8], request.message[:80])

        # Query chatbot (async to avoid blocking). The timeout only releases
        # the REQUEST — the worker thread is uncancellable and runs on, which
        # is why WORKER_THREADS keeps headroom over expected concurrency.
        meta = await asyncio.wait_for(
            query_chatbot_async(request.message, memory),
            timeout=CHAT_TIMEOUT_S,
        )

        logger.debug("[Reply] %s...", meta["answer"][:100])
        record_query_metric(meta)

        return ChatResponse(
            reply=meta["answer"],
            session_id=session_id,
            stats=build_stats(meta),
        )

    except asyncio.TimeoutError:
        logger.error("[Timeout] chat request exceeded %ss", CHAT_TIMEOUT_S)
        record_query_metric(timeout=True)
        raise HTTPException(
            status_code=504,
            detail="The advisor took too long to answer — please try again.",
        )
    except Exception as e:
        logger.exception("[Error] chat request failed")
        record_query_metric(error=True)
        raise HTTPException(status_code=500, detail=str(e))


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stats_dict(meta: dict) -> dict:
    stats = build_stats(meta)
    return stats.model_dump() if hasattr(stats, "model_dump") else stats.dict()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE variant of POST /chat: `session` (id) → `delta` (text fragments)
    → `done` (the same stats footer /chat returns), or `error`. The plain
    /chat endpoint stays for the eval harness and as the frontend fallback."""
    validate_chat_request(request)

    session_id, memory = get_or_create_session(request.session_id)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def produce():
        # Runs query_stream to completion regardless of the consumer: memory
        # and the query_usage log are written even if the client disconnects
        # or the request times out (the queue items just go unread).
        try:
            for item in chatbot.query_stream(request.message, memory=memory):
                if item[0] == "done":
                    record_query_metric(item[1])
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:
            logger.exception("[Error] stream request failed")
            record_query_metric(error=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

    async def event_source():
        logger.info("[Query] Session %s... (stream): %s", session_id[:8], request.message[:80])
        yield sse_event("session", {"session_id": session_id})
        loop.run_in_executor(executor, produce)
        deadline = loop.time() + CHAT_TIMEOUT_S
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                kind, payload = await asyncio.wait_for(queue.get(), timeout=remaining)
                if kind == "delta":
                    yield sse_event("delta", {"text": payload})
                elif kind == "done":
                    yield sse_event("done", {"stats": _stats_dict(payload)})
                elif kind == "error":
                    yield sse_event("error", {"detail": payload})
                elif kind == "end":
                    break
        except asyncio.TimeoutError:
            logger.error("[Timeout] stream request exceeded %ss", CHAT_TIMEOUT_S)
            # Only bump the timeout counter — the producer thread still runs
            # to completion and records the query itself when it finishes.
            with _metrics_lock:
                _metrics_totals["timeouts"] += 1
            yield sse_event(
                "error",
                {"detail": "The advisor took too long to answer — please try again."},
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        # no-cache + no-transform keeps Cloudflare from buffering the stream;
        # X-Accel-Buffering covers any nginx-style proxy in between.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    verdict: str
    question: Optional[str] = None
    answer_excerpt: Optional[str] = None


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Thumbs up/down from the chat UI. Sink is a structured log line
    (journald) — eval/mine_langsmith.py picks `down` verdicts up as golden-set
    candidates. No DB on purpose: feedback volume is tiny at demo scale."""
    if request.verdict not in ("up", "down"):
        raise HTTPException(status_code=422, detail="verdict must be 'up' or 'down'")
    logger.info("feedback_log %s", json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": (request.session_id or "")[:36],
        "verdict": request.verdict,
        "question": (request.question or "")[:500],
        "answer_excerpt": (request.answer_excerpt or "")[:1000],
    }, ensure_ascii=False))
    return {"status": "ok"}


@app.post("/reset")
async def reset_session(session_id: Optional[str] = None):
    """Reset a session's conversation history"""
    if session_id and session_id in sessions:
        sessions[session_id]["memory"].clear()
        return {"status": "reset", "session_id": session_id}
    return {"status": "session_not_found"}

FRONTEND_BUILD_DIR = "../frontend/dist"

if os.path.exists(FRONTEND_BUILD_DIR):
    # Serve static files
    app.mount("/assets", StaticFiles(directory=f"{FRONTEND_BUILD_DIR}/assets"), name="assets")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(f"{FRONTEND_BUILD_DIR}/index.html")
    
    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = f"{FRONTEND_BUILD_DIR}/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{FRONTEND_BUILD_DIR}/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
