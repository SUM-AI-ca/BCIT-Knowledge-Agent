import os
import logging
import warnings
import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

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
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from query_rag import BCITChatbot
from config import MEMORY_WINDOW_K
from langchain.memory import ConversationBufferWindowMemory

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatStats(BaseModel):
    """Per-reply transparency footer: what this answer actually consumed."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    model: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    stats: Optional[ChatStats] = None


chatbot: Optional[BCITChatbot] = None
sessions: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)

SESSION_TIMEOUT_MINUTES = 30


def get_or_create_session(session_id: Optional[str]) -> tuple[str, ConversationBufferWindowMemory]:
    if not session_id:
        session_id = str(uuid.uuid4())
    
    now = datetime.now()
    
    if session_id not in sessions:
        sessions[session_id] = {
            "memory": ConversationBufferWindowMemory(
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


def query_chatbot_sync(question: str, memory: ConversationBufferWindowMemory) -> dict:
    # Memory is passed per request — never swap chatbot.memory globally,
    # concurrent requests would leak history across sessions.
    return chatbot.query_with_meta(question, memory=memory)


async def query_chatbot_async(question: str, memory: ConversationBufferWindowMemory) -> dict:
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Get or create session
    session_id, memory = get_or_create_session(request.session_id)
    
    try:
        logger.info("[Query] Session %s...: %s", session_id[:8], request.message[:80])

        # Query chatbot (async to avoid blocking)
        meta = await query_chatbot_async(request.message, memory)

        logger.debug("[Reply] %s...", meta["answer"][:100])

        return ChatResponse(
            reply=meta["answer"],
            session_id=session_id,
            stats=build_stats(meta),
        )

    except Exception as e:
        logger.exception("[Error] chat request failed")
        raise HTTPException(status_code=500, detail=str(e))


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
