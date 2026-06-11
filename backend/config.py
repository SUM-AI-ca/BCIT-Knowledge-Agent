import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Tunables are env-overridable so eval runs and prod rollbacks can flip
# behavior without code changes (config is read once at import time).
def _env_str(name, default):
    return os.getenv(name, default)


def _env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name, default):
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_opt_int(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    if raw.strip().lower() in ("none", "null"):
        return None
    return int(raw)

# Paths (env-overridable so a fresh corpus can be indexed side by side)
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
# Versioned with PG_COLLECTION (blue-green): bcit_docs_202606 pairs with
# documents_202606.pkl. Flip BOTH defaults together on corpus rebuilds —
# the dense and BM25 sides must serve the same crawl.
DOCUMENTS_PICKLE = Path(os.getenv("DOCUMENTS_PICKLE", "./vectorstore/documents_202606.pkl"))

# Embeddings (Vertex AI / Gemini Enterprise Agent Platform, ADC auth)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 1536  # MRL truncation; pgvector HNSW supports <= 2000 dims
EMBEDDING_LOCATION = "us-central1"  # embeddings are regional, chat uses global

# Vector store (Cloud SQL PostgreSQL + pgvector, via Cloud SQL Auth Proxy)
PG_CONNECTION = os.getenv(
    "PG_CONNECTION",
    "postgresql+psycopg://raguser:raguser@127.0.0.1:5432/ragdb"
)
# Collection is versioned: build a new one, then flip this default and redeploy
# (blue-green — the live server keeps serving the old collection during a build)
PG_COLLECTION = os.getenv("PG_COLLECTION", "bcit_docs_202606")

# Mixed-model setup (June 2026 benchmark, eval/benchmarks/202606_model_comparison):
# generation on flash-lite cut cost 69% at equal-or-better quality, but ONLY
# with the rewriter kept on 3.5-flash — sub-query quality drives retrieval
# (pure lite: hit 0.929; mix: hit 0.963 / recall 0.890, best of all runs).
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-3.1-flash-lite")
REWRITER_MODEL = _env_str("REWRITER_MODEL", "gemini-3.5-flash")
GEMINI_PROJECT = "wine-agent-jh-2026"
GEMINI_LOCATION = "global"
GEMINI_TEMPERATURE = 0.05
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 2048)

# Single source of truth for conversation window (was k=5 in server.py and
# k=3 in query_rag.py, neither actually applied due to the window bug).
MEMORY_WINDOW_K = _env_int("MEMORY_WINDOW_K", 5)

# What gets SAVED into history (and therefore re-sent every later turn):
# Sources lists add no value to follow-up resolution, and unbounded answers
# compound across the window.
STRIP_SOURCES_FROM_HISTORY = _env_bool("STRIP_SOURCES_FROM_HISTORY", True)
HISTORY_MAX_ANSWER_CHARS = _env_int("HISTORY_MAX_ANSWER_CHARS", 1500)

# Per-call prices for the cost estimate shown to users and logged per query.
# List prices as of 2026-06 (ai.google.dev/gemini-api/docs/pricing and the
# Ranking API pricing page) — UPDATE THESE when models or prices change; all
# env-overridable. Output prices include thinking tokens.
PRICE_GEN_INPUT_PER_M = _env_float("PRICE_GEN_INPUT_PER_M", 0.25)      # gemini-3.1-flash-lite
PRICE_GEN_OUTPUT_PER_M = _env_float("PRICE_GEN_OUTPUT_PER_M", 1.50)
PRICE_REWRITE_INPUT_PER_M = _env_float("PRICE_REWRITE_INPUT_PER_M", 1.50)   # gemini-3.5-flash
PRICE_REWRITE_OUTPUT_PER_M = _env_float("PRICE_REWRITE_OUTPUT_PER_M", 9.00)
PRICE_RERANK_PER_CALL = _env_float("PRICE_RERANK_PER_CALL", 0.001)     # Ranking API $1/1k queries
PRICE_EMBED_PER_QUERY = _env_float("PRICE_EMBED_PER_QUERY", 0.00001)   # $0.15/M x ~60 tokens

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 130

USE_HYBRID_SEARCH = _env_bool("USE_HYBRID_SEARCH", True)

HYBRID_ALPHA = 0.48

RETRIEVAL_TOP_K = 10
RETRIEVAL_DENSE_K = 23
RETRIEVAL_BM25_K = 23

RETRIEVAL_FETCH_K = 50
MMR_LAMBDA = 0.87
HNSW_EF_SEARCH = 100  # pgvector default 40 would silently cap MMR fetch_k=50

USE_RERANKING = _env_bool("USE_RERANKING", True)
RERANKER_MODEL = "semantic-ranker-default-004"  # Vertex AI Ranking API
RANKING_LOCATION = "global"
RANKING_CONFIG = "default_ranking_config"
RERANKER_CANDIDATES = _env_int("RERANKER_CANDIDATES", 25)
RERANKER_TOP_K = _env_int("RERANKER_TOP_K", 10)

# Context construction. "chunks" injects the reranked chunks themselves with
# small-to-big neighbor expansion; "full_doc" restores the legacy behavior of
# loading each cited source file whole (~9k tokens/file). Legacy baseline =
# CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false RERANKER_TOP_K=13.
CONTEXT_MODE = _env_str("CONTEXT_MODE", "chunks")
# How many adjacent chunks to pull in on each side of a selected chunk. The
# corpus pickle stores chunks per source in document order, so neighbors can
# be resolved in-process without a DB migration. Eval: 2 beats 1 on outline
# fact recall (+0.04) and entrance-requirement chunks for +5.6% tokens.
NEIGHBOR_RADIUS = _env_int("NEIGHBOR_RADIUS", 2)
# Hard cap for the assembled context; neighbor expansion is dropped from the
# lowest-ranked sources first, then whole trailing sources.
CONTEXT_MAX_CHARS = _env_int("CONTEXT_MAX_CHARS", 24000)
# Chunks scoring below this after reranking are dropped (0 disables). The
# reranker fallback path returns unscored docs — the filter is skipped then.
RERANK_SCORE_THRESHOLD = _env_float("RERANK_SCORE_THRESHOLD", 0.0)
MIN_CONTEXT_CHUNKS = _env_int("MIN_CONTEXT_CHUNKS", 3)

# Multi-query decomposition. One LLM call per turn (replacing the old
# history-only rewrite) returns the standalone question plus 1..MAX
# self-contained sub-queries; multipart questions get one query per part.
MULTI_QUERY_ENABLED = _env_bool("MULTI_QUERY_ENABLED", True)
MAX_SUB_QUERIES = _env_int("MAX_SUB_QUERIES", 4)
# Per-sub-query retrieval breadth (single-question turns keep the legacy
# full-width retrieval; these only apply when a turn decomposes into 2+).
MQ_DENSE_K = _env_int("MQ_DENSE_K", 12)
MQ_BM25_K = _env_int("MQ_BM25_K", 12)
MQ_CANDIDATES_PER_SUBQUERY = _env_int("MQ_CANDIDATES_PER_SUBQUERY", 15)
MQ_POOL_CAP = _env_int("MQ_POOL_CAP", 40)
# After the pooled rerank, every sub-query keeps at least this many of its
# own candidates in the final context (swap-in by that sub-query's best).
MQ_MIN_CHUNKS_PER_SUBQUERY = _env_int("MQ_MIN_CHUNKS_PER_SUBQUERY", 2)
# "pooled": ONE Ranking API call per turn over the merged candidate pool
# (the API bills per query). "per_subquery": one call per sub-query.
RERANK_MODE = _env_str("RERANK_MODE", "pooled")
REWRITE_MAX_OUTPUT_TOKENS = _env_int("REWRITE_MAX_OUTPUT_TOKENS", 512)
REWRITE_THINKING_BUDGET = _env_int("REWRITE_THINKING_BUDGET", 0)
# Thinking tokens are billed as output AND count against max_output_tokens —
# with the model default (~2k thinking) a 2048 cap truncates every answer
# (observed: finish=MAX_TOKENS with 80 visible tokens). 0 pairs with the 2048
# cap; the eval sweep compares None (+4096 cap) for answer quality.
GEMINI_THINKING_BUDGET = _env_opt_int("GEMINI_THINKING_BUDGET", 0)

# Static instructions come FIRST and the variable inputs (context, history,
# question) LAST: Gemini's implicit caching can only reuse a shared prompt
# prefix, and anything after the first changed byte is a cache miss.
RAG_PROMPT_TEMPLATE = """You are a BCIT (British Columbia Institute of Technology) academic advisor chatbot.

Your role:
- Answer the student's question using ONLY the provided BCIT documents and recent conversation history for any BCIT specific facts.
- You may use your general world knowledge only for non BCIT background explanations.
- Always respond in ENGLISH.

INSTRUCTIONS:

1. LANGUAGE
   - Always respond in English only.
   - Do not use hedging phrases such as "I think", "I would say", or "maybe",
     unless you are explicitly describing uncertainty in the documents.

2. CONVERSATION CONTINUITY
   - Use the conversation history to resolve references like "it", "that course",
     "this program", "the prerequisite", and similar wording.
   - If something remains ambiguous, briefly state your assumption and then answer.

3. USE OF BCIT DOCUMENTS
   - For any BCIT specific facts (dates, URLs, admission requirements, program
     details, course outlines, policies, schedules, tuition, and similar content), rely ONLY on
     information that appears explicitly in the retrieved BCIT context below.
   - Treat any BCIT information that is not present in that context as unknown.
   - Do not invent or guess BCIT specific facts.

4. PROGRAM PRIORITY
   - If both full time and flexible learning information appear in the retrieved text, assume the student is asking about the full time program unless they clearly specify flex or part time.
   - If only flexible learning information is present, answer using that and briefly clarify that only flexible learning details were available.

5. MISSING OR INCOMPLETE INFORMATION
   - If the student asks for BCIT specific information that is not present in the retrieved text:
       "This specific information is not in the available documents."
   - You may then suggest how the student could find the information, for example
     by checking the official BCIT website, but do not fabricate URLs, dates,
     or numeric values.

6. ANALYSIS AND ADVISING
   - When comparing programs or courses, or making a recommendation, base your
     reasoning only on the retrieved text, including prerequisites, credits, course level,
     workload hints, and descriptions.
   - Be practical and student focused. Explain your reasoning briefly and clearly.

7. ANSWER FORMAT
   - Start with a short, direct summary that answers the question.
   - Then provide a concise explanation using short paragraphs or bullet points.
   - Keep the whole answer under roughly 350 words, unless the question has
     multiple distinct parts that each need their own answer.
   - Do not copy long paragraphs verbatim from the documents. Paraphrase in
     your own words.
   - Do not mention the words "context", "documents", or "prompt" in your answer.

8. SOURCES (BCIT URLs)
   - At the end of your answer, add a section titled "Sources".
   - Under "Sources", list only the BCIT URLs that appear in the retrieved text and that
     you actually used.
   - Copy each URL exactly as it appears, for example in lines like
     "Document 1 [URL: https://...]".
   - If you used information that has no URL in the provided documents, write:
       Sources: No BCIT URL available in the provided documents.

Inputs:

- Retrieved BCIT context:
{context}

- Conversation history:
{chat_history}

- Student's question:
{question}{question_parts}

Answer:"""

# Combined rewrite + decompose (one JSON call per turn, replaces the legacy
# rewrite when MULTI_QUERY_ENABLED). The schema is enforced server-side via
# response_schema; the prompt still spells out the rules for quality.
REWRITE_DECOMPOSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "standalone_question": {"type": "STRING"},
        "sub_queries": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["standalone_question", "sub_queries"],
}

REWRITE_DECOMPOSE_TEMPLATE = """You prepare search queries for a BCIT (British Columbia Institute of Technology) academic advisor chatbot.

Given the conversation history and the student's latest message, return JSON with:
1. "standalone_question": the latest message rewritten as ONE self-contained question. Resolve references like "it", "that course", "this program" using the history. If the message is already self-contained, return it unchanged.
2. "sub_queries": 1 to {max_sub_queries} short, self-contained search queries.
   - If the message asks a single thing, return exactly one sub-query (the standalone question itself).
   - If it asks several distinct things (e.g. admission requirements AND tuition AND housing), return one sub-query per distinct thing.
   - Every sub-query must name the concrete program/course/topic (no pronouns).

Rules:
- English only.
- Do not invent topics the student did not ask about.
- At most {max_sub_queries} sub-queries.

Conversation history:
{chat_history}

Student's latest message:
{question}"""

# Legacy query rewriting prompt (used when MULTI_QUERY_ENABLED=false)
QUERY_REWRITE_TEMPLATE = """You are helping a BCIT academic advisor chatbot with retrieval.

Task:
- Rewrite the follow-up question into a standalone question that can be used
  for vector search over BCIT documents.

Rules:
- If the question already contains all necessary context (full program names,
  course codes, campus names, etc.), return it unchanged.
- If the question uses references like "it", "this course", "that program",
  or "the prerequisite", replace them with the concrete entities from the
  conversation history.
- Keep the language in English.
- Return ONLY the rewritten question text, with no explanations.

Conversation history:
{chat_history}

Follow-up question:
{question}

Standalone question:"""
