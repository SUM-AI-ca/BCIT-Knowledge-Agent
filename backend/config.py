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


# Same None-means-omit contract as _env_opt_int, for the thinking_level enums:
# None leaves the parameter off the request entirely, so the model applies its
# own default. "default" is accepted as a spelling of that for readability in
# eval sweeps (THINKING sweeps used to pass GEMINI_THINKING_BUDGET=none).
#
# The values here are passed to ChatGoogleGenerativeAI as `thinking_level`.
# In langchain-google-genai 4.3.x that is a pydantic ALIAS: the field is really
# named `reasoning_effort` (LangChain's cross-provider name) and `thinking_level`
# is kept because it is Gemini's own name for the setting. Both work as
# constructor kwargs and both read the value back, so `thinking_level` is used
# throughout this repo to match Google's documentation. Do not be surprised when
# `model_fields` shows `reasoning_effort` and no `thinking_level`.
def _env_opt_str(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    if raw.strip().lower() in ("none", "null", "default"):
        return None
    return raw.strip().lower()

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
# (blue-green — the live server keeps serving the old collection during a build).
# 202606da = same chunks as 202606 with identity-prefixed EMBEDDINGS
# (build_pgvector.py EMBED_IDENTITY_PREFIX): documented exception to the
# "flip PG_COLLECTION and DOCUMENTS_PICKLE together" rule — the pickle is
# shared because the stored chunks are byte-identical (verified corpus-wide).
PG_COLLECTION = os.getenv("PG_COLLECTION", "bcit_docs_202606da")

# Mixed-model setup (June 2026 benchmark, eval/benchmarks/202606_model_comparison):
# generation on flash-lite cut cost 69% at equal-or-better quality, but ONLY
# with the rewriter kept on 3.5-flash — sub-query quality drives retrieval
# (pure lite: hit 0.929; mix: hit 0.963 / recall 0.890, best of all runs).
# August 2026: both models moved to their successors (3.1-flash-lite →
# 3.5-flash-lite, 3.5-flash → 3.6-flash). The *shape* of the finding is what
# carries over — a lite generator paired with a full-size rewriter — not the
# specific scores above, which were measured on the older pair and are left
# as-is in eval/benchmarks/ rather than being restated for models never run.
#
# 2026-08-14: the rewriter and judge moved 3.6-flash -> 3.7-flash (GA
# 2026-08-13), and the whole chat layer moved off ChatVertexAI onto
# ChatGoogleGenerativeAI. The two changes are one change: 3.7-flash configures
# reasoning with a thinking_level enum instead of thinking_budget, and
# langchain-google-vertexai 3.2.4 (the latest release, and deprecated since
# 3.2.0 in favour of ChatGoogleGenerativeAI) exposes no thinking_level at all.
# See THINKING LEVELS below for what that costs. Generation and the controller
# stay on 3.5-flash-lite; only the two full-size calls moved.
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-3.5-flash-lite")
REWRITER_MODEL = _env_str("REWRITER_MODEL", "gemini-3.7-flash")
# Offline eval only (run_eval.py --judge): scores answer faithfulness and
# completeness against the retrieved context. Same class as the rewriter —
# judging is a reading task that lite models measurably do worse.
JUDGE_MODEL = _env_str("JUDGE_MODEL", "gemini-3.7-flash")
GEMINI_PROJECT = "wine-agent-jh-2026"
GEMINI_LOCATION = "global"
GEMINI_TEMPERATURE = 0.05
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 2048)

# Single source of truth for conversation window (was k=5 in server.py and
# k=3 in query_rag.py, neither actually applied due to the window bug).
MEMORY_WINDOW_K = _env_int("MEMORY_WINDOW_K", 5)

# Server guardrails. The chat executor has no other deadline — a wedged
# Vertex gRPC call was once observed holding a worker thread ~8 minutes.
# The timeout 504s the client; the underlying thread cannot be cancelled
# and runs to completion, which is why WORKER_THREADS stays above 2 (the
# pipeline is API-bound, so threads > vCPUs is fine on the e2-medium).
CHAT_TIMEOUT_S = _env_int("CHAT_TIMEOUT_S", 90)
WORKER_THREADS = _env_int("WORKER_THREADS", 4)
# Longest real question in the eval sets is ~200 chars; 2000 leaves room for
# pasted context while keeping a flood from inflating rewriter input tokens.
MAX_MESSAGE_CHARS = _env_int("MAX_MESSAGE_CHARS", 2000)

# What gets SAVED into history (and therefore re-sent every later turn):
# Sources lists add no value to follow-up resolution, and unbounded answers
# compound across the window.
STRIP_SOURCES_FROM_HISTORY = _env_bool("STRIP_SOURCES_FROM_HISTORY", True)
HISTORY_MAX_ANSWER_CHARS = _env_int("HISTORY_MAX_ANSWER_CHARS", 1500)

# First-turn response cache (exact-match, in-process). Only first-turn (no
# history) questions are cached — follow-ups depend on session history, and
# the answer to a no-history question is a pure function of the question +
# corpus. Exact normalized match (no semantic similarity) so a near-miss can
# never serve a wrong answer. In-process: a restart (every deploy / blue-green
# corpus cutover) clears it, so a cached answer never outlives its corpus.
RESPONSE_CACHE_ENABLED = _env_bool("RESPONSE_CACHE_ENABLED", True)
RESPONSE_CACHE_MAX = _env_int("RESPONSE_CACHE_MAX", 1000)
RESPONSE_CACHE_TTL_S = _env_int("RESPONSE_CACHE_TTL_S", 86400)

# Per-call prices for the cost estimate shown to users and logged per query.
# List prices as of 2026-08 (ai.google.dev/gemini-api/docs/pricing and the
# Ranking API pricing page) — UPDATE THESE when models or prices change; all
# env-overridable. Output prices include thinking tokens.
#
# These MUST be changed together with GEMINI_MODEL / REWRITER_MODEL above: the
# number they produce is rendered under every answer, so a model swap without a
# price swap quotes the user a figure for a model that did not run. The 2026-08
# move to 3.5-flash-lite / 3.6-flash pushes generation up and rewriting down.
#
# CALENDAR ITEM: the rewriter figures below are 3.7-flash's INTRODUCTORY price,
# which Google has published only through 2026-12-31. On 2027-01-01 it reverts
# to 1.50 / 7.50 (the same numbers 3.6-flash carried, i.e. what these two lines
# held before 2026-08-14). Put those values back then, or every answer will
# quote roughly half what the turn actually cost.
PRICE_GEN_INPUT_PER_M = _env_float("PRICE_GEN_INPUT_PER_M", 0.30)      # gemini-3.5-flash-lite
PRICE_GEN_OUTPUT_PER_M = _env_float("PRICE_GEN_OUTPUT_PER_M", 2.50)
PRICE_REWRITE_INPUT_PER_M = _env_float("PRICE_REWRITE_INPUT_PER_M", 0.75)   # gemini-3.7-flash (intro, to 2026-12-31)
PRICE_REWRITE_OUTPUT_PER_M = _env_float("PRICE_REWRITE_OUTPUT_PER_M", 3.75)
PRICE_RERANK_PER_CALL = _env_float("PRICE_RERANK_PER_CALL", 0.001)     # Ranking API $1/1k queries
PRICE_EMBED_PER_QUERY = _env_float("PRICE_EMBED_PER_QUERY", 0.00001)   # $0.15/M x ~60 tokens

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 130

USE_HYBRID_SEARCH = _env_bool("USE_HYBRID_SEARCH", True)

HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.48)
RRF_K = _env_int("RRF_K", 60)

# Fit the BM25 index on title/category/URL-slug-augmented text (the served
# documents are untouched). Deep chunks of the 529 program pages never mention
# their own program, so section queries ("entrance requirements") flood the
# pool with sibling programs; this gives every chunk its page identity.
# Default since round 2 (eval/benchmarks/202606_retrieval_cost_experiments):
# multipart hit 0.917 -> 1.000, recall +0.048, outline untouched.
BM25_INDEX_AUG = _env_bool("BM25_INDEX_AUG", True)

RETRIEVAL_TOP_K = 10
RETRIEVAL_DENSE_K = 23
RETRIEVAL_BM25_K = 23

# Drop query tokens this common from the BM25 arm (document frequency in the
# augmented index, 0 disables). BM25_INDEX_AUG stamps title/category/filename
# keywords onto every chunk, so a rewritten sub-query carrying generic corpus
# vocabulary matches 40-60% of the corpus with a small positive weight and
# reorders the tail. Measured on mh3-02 ("BCIT courses with ACIT 1515 as a
# prerequisite"): leave-one-out says `bcit` (df 55.9%), `a` (60.1%), `with`
# (45.1%), `courses` (41.8%) are the pollutants, and the target chunk moves
# from BM25 rank 161 to 22 without them — in vs out of RETRIEVAL_BM25_K=23.
# A df threshold rather than a stopword list: it is derived from this corpus
# and needs no upkeep when the crawl changes. 0.35 drops exactly those four
# and leaves ol-02/ol-04/pa-01/sf3-05/ex3-01 ranks byte-identical.
# REJECTED 2026-08: 0 cases changed on any of the three sets. The offline
# ranking effect is real (the target chunk moves from BM25 rank 161 to 22) but
# the pooled rerank re-orders the candidates anyway, so a better candidate set
# did not become a better context. Kept for the fan-in query shape, where the
# candidate set is the binding constraint.
BM25_STOPWORD_DF = _env_float("BM25_STOPWORD_DF", 0.0)

# Chunk identity for pool dedup and RRF fusion. The historical key is
# source + the chunk's first 200 chars; measured on this corpus it collides on
# 418 keys covering 715 chunks (0.71%) — two DIFFERENT chunks of the same page
# that open identically (repeated tables in course pages, the English-language
# proficiency assessment table that pa-07/pa-08 depend on) merge, and the
# loser's content never reaches the context. `true` keys on an md5 of the whole
# chunk. Off by default: it slightly enlarges the candidate pool, and round 2
# (exp1_pool56) showed undifferentiated pool growth dilutes the pooled rerank —
# so it is an experiment with a gate, not a free correctness win.
DEDUP_FULL_CONTENT = _env_bool("DEDUP_FULL_CONTENT", False)

# Entity-scoped retrieval: when a sub-query names a concrete course code or
# program, add a retrieval arm restricted to that entity's OWN chunks.
# Targets the "boilerplate section" failure class — a chunk whose body carries
# no entity identity and whose wording is shared corpus-wide (1,772 chunks say
# "Final Exam |", 3,038 say "Total Hours |"). Such a chunk cannot be ranked
# globally by any phrasing: ol-04's answer chunk is outside the BM25 top-25
# under all three phrasings tried and is rank 1 once scoped; same for ol-11,
# sf3-05 and mp-01's entrance-requirements/costs sections.
# Candidates join the existing pool, so the Ranking API call count is unchanged
# (it bills per query for up to 100 records).
#
# ADOPTED 2026-08 together with RERANK_IDENTITY — neither is worth adopting
# alone. Measured on v1 (n=39, ×2 identical runs): each flag ALONE fixes the
# same single case (mp-01) and nothing else; together they also fix ol-04 and
# ol-12, because the two halves of the failure need each other — without the
# scoped arm the course's chunks never enter the pool, and without identity
# the ranker drops them for sibling outlines whose identical section is what
# the question literally describes.
ENTITY_SCOPED_RETRIEVAL = _env_bool("ENTITY_SCOPED_RETRIEVAL", True)
# Per SOURCE, not per entity: a course resolves to its outline AND its
# catalogue page, and the shorter page wins BM25 length normalisation on every
# chunk, so a shared budget let it evict the outline rows that hold the answer.
ENTITY_SCOPED_K = _env_int("ENTITY_SCOPED_K", 8)
# Cap per turn. Each entity is scored over its own chunks only (~0.1-0.3 ms,
# not the 128 ms a full-index get_scores() pass costs), so the cap is about
# bounding how much of the rerank pool one turn's entities may claim, not CPU.
ENTITY_SCOPED_MAX_ENTITIES = _env_int("ENTITY_SCOPED_MAX_ENTITIES", 3)

# Drop the outline approval-signature lines from the text the BM25 index is
# fit on (documents are served untouched, as with BM25_INDEX_AUG — no
# re-embed). The block is 3,279 chunks / 3.3% of the corpus and its only
# non-boilerplate content is a staff name in a role that is NOT the course's
# instructor, so it hijacks person queries: "Chi En Huang" occurs in 62
# signature chunks against 4 Instructor Details chunks, and a "what does X
# teach" query is outvoted 15:1 by chunks about a different relation.
SIGNATURE_DEMOTE = _env_bool("SIGNATURE_DEMOTE", False)

# Fan-in retention: guarantee up to N DISTINCT sources whose text contains the
# literal the question names (course code, or a person's name). Targets the
# query shape whose answer is spread over N sibling pages, where the pooled
# rerank otherwise spends the budget globally — mh3-02 ("which courses require
# ACIT 1515?", 3 correct outlines, 1 kept) and person lookups.
FANIN_RETAIN = _env_bool("FANIN_RETAIN", False)
FANIN_RETAIN_N = _env_int("FANIN_RETAIN_N", 6)

# Person-scoped retrieval: index instructor names -> the sources that name
# them AS THE INSTRUCTOR, and scope a retrieval arm to those when a question
# names one. The corpus states this relation structurally (outline
# "Instructor Details / Name |", course page "### Instructor"), the same kind
# of convention the course-code index is built on. Only the instructor
# relation is indexed — approval signatures and program-page coordinator
# lists name the same people in relations that answer a different question.
# ADOPTED 2026-08. Guard set (15 cases incl. held-out instructors and
# adversarial "who is the Program Head" lookups): F1 0.576 -> 0.968, two
# held-out complete failures fixed, 0 inventions, adversarial cases intact.
# v3 (24 cases) url 0.676 -> 0.867 x2 with the 21 older cases byte-identical;
# v1 (n=39) and the v2 band unchanged. Fires only when a question names one of
# the indexed instructors - 0 of 86 regression questions - so it is a no-op on
# everything else.
PERSON_SCOPED_RETRIEVAL = _env_bool("PERSON_SCOPED_RETRIEVAL", True)
# Breadth over depth, the mirror of ENTITY_SCOPED_K: a person is one entity
# spread over many pages, so take few chunks from each of many sources.
PERSON_SCOPED_K = _env_int("PERSON_SCOPED_K", 2)
PERSON_SCOPED_MAX_SOURCES = _env_int("PERSON_SCOPED_MAX_SOURCES", 12)

RETRIEVAL_FETCH_K = _env_int("RETRIEVAL_FETCH_K", 50)
MMR_LAMBDA = _env_float("MMR_LAMBDA", 0.87)
HNSW_EF_SEARCH = 100  # pgvector default 40 would silently cap MMR fetch_k=50

USE_RERANKING = _env_bool("USE_RERANKING", True)
# Vertex AI Ranking API. `semantic-ranker-fast-004` is the same 004 generation
# (1024 tokens/record, so RERANK_IDENTITY's `title` still applies) and Google
# advertises 3x lower latency for it, against `default-004` leading nDCG@5 on
# BEIR. No absolute latency figures and no records-vs-latency scaling are
# published for either, so whether the fast tier moves the tail measured below
# is an open question — env-overridable so the golden sets can answer it
# instead of a redeploy. Older tiers (`semantic-ranker-default-003`/`-002`)
# take 512 tokens per record and would silently truncate our 1024-char
# records.
RERANKER_MODEL = _env_str("RERANKER_MODEL", "semantic-ranker-default-004")
# Only `global` is documented for rankingConfigs — the us/eu multi-regions are
# listed for Agent Search generally but the ranking path is not among them, so
# this is not a latency lever, whatever the VM's region.
RANKING_LOCATION = "global"
RANKING_CONFIG = "default_ranking_config"
# Deadline for one Ranking API call, and how many times to issue it. 0 / 1 is
# today's behaviour: the GAPIC transport ships `default_timeout=None` for
# `rank`, so the call currently has NO deadline at all and the graceful
# degrade-to-retrieval-order path below it can only fire on an exception,
# never on slowness.
#
# Measured on production LangSmith traces (real user traffic, not the eval
# harness), 2026-08-09..11: vertex_rerank p50 0.136s — the cheapest stage in
# the pipeline — against p90 8.19s, p95 27.8s, max 42.8s. The tail is one
# window: 2026-08-09 05:17-06:32 PDT ran 36/36 calls at <=0.29s, while
# 2026-08-10 17:45-18:10 PDT ran 6/6 at 6.6-30.1s. Server-side, and none of
# ours:
#   - other spans in the SAME request stayed normal (controller LLM 1.6s,
#     rewriter 1.3s, BM25 0.52s, dense 0.47s, generation 2.79s, against a
#     30.1s rerank) — so not VM contention;
#   - slowness repeats WITHIN a request ([30.1, 12.7], [9.4, 6.6, 11.7]), so
#     not a cold channel or a token refresh;
#   - 3-hop turns on the fast day scored [0.108, 0.117, 0.137], same code and
#     same call shape as the 3-hop turn that scored [9.4, 6.6, 11.7];
#   - latency is flat against candidate count over 1,962 archived eval calls
#     (25 records p50 0.191s, 40 records p50 0.209s), so pool size is a
#     quality lever only — shrinking it buys no latency;
#   - the API's published design target is under 100 ms, and the quota (500
#     requests/min/project) is ~2,000x our rate and fails rather than delays.
# No public incident covered the window; a project-scoped one would only show
# in Personalized Service Health.
#
# So the fix is not to rerank less, it is to stop paying an unbounded wait for
# it. At 2.5s the deadline sits 25x above the published target and ~18x above
# the observed warm p95, and a second attempt usually lands on a healthy
# backend; if both miss, `_select_from_pool` keeps its existing fallback and
# the turn answers on fusion order. Bounds the damage at ~5s instead of 43s.
# Note the trade: a fired deadline costs that turn its semantic ranking, which
# RERANK_SKIP_CONSENSUS's comment already records as measurably lossy on
# multi-page questions — which is why the retry exists rather than a bare
# timeout. Note also that `n_rerank_calls` counts turns, not attempts, so a
# retried turn may be billed twice while the cost line reports one call.
RERANK_TIMEOUT_S = _env_float("RERANK_TIMEOUT_S", 0.0)
RERANK_ATTEMPTS = _env_int("RERANK_ATTEMPTS", 1)

# Hedged request: if the call has not answered within RERANK_HEDGE_AFTER_S,
# issue an identical second one and take whichever lands first. 0 disables.
#
# This is the mitigation the deadline above cannot be: it never degrades a
# turn's ranking. A deadline trades ranking quality for latency on every slow
# call; a hedge just takes the faster of two identical answers.
#
# It works because the stall is substantially per-request, not a uniformly
# slow service. Probed live during the 2026-08-11 degradation (synthetic
# records, no corpus), 10 pairs of identical requests issued concurrently:
#     single request    p50 2.367s   under 1s:  7/20   max 10.932s
#     best-of-2         p50 0.615s   under 1s:  6/10   max  4.952s
# i.e. ~3.8x on the median and ~2.2x on the worst case. It is a real
# improvement, NOT a cure: 3 of the 10 pairs had both twins slow (faster leg
# still 2.8-5.0s), so a degraded window stays degraded, just less. An earlier
# 3-pair probe suggested a fast twin every time; 10 pairs did not hold that up.
#
# The same probing killed both alternatives, which is why this is the lever:
# `semantic-ranker-fast-004` was no better under the degradation (p50 17.6s
# against default's 10.3s, plus a ServiceUnavailable), and record count was
# irrelevant live (5 -> 0.135s, 25 -> 8.733s, 40 -> 0.194s) — matching the
# archive and ruling out any candidate-count remedy for good.
#
# Cost is bounded and demand-shaped: at 0.4s the trigger sits ~3x above the
# measured warm p50 (0.136s), so healthy turns never hedge and never pay. A
# degraded turn pays one extra billed query (~$0.001). At most two requests
# are ever in flight — the hedge fires once, not per stall.
RERANK_HEDGE_AFTER_S = _env_float("RERANK_HEDGE_AFTER_S", 0.0)
# Threads for the hedge. Each concurrent turn can hold two, and a hung request
# holds its thread until HEDGE_ABANDON_S releases it.
RERANK_HEDGE_WORKERS = _env_int("RERANK_HEDGE_WORKERS", 8)
# Hygiene, not policy: the deadline on BOTH hedged attempts, so a hung one
# eventually gives its thread back instead of pinning it forever (the GAPIC
# transport would otherwise wait without limit). Set far above anything
# observed — the worst measured call was 30.1s — because by the time this
# fires the winner has long since been returned and nothing about the answer
# depends on it. It is not RERANK_TIMEOUT_S and does not degrade a turn.
RERANK_HEDGE_ABANDON_S = _env_float("RERANK_HEDGE_ABANDON_S", 60.0)
# Send each candidate's page identity (title + filename keywords + category)
# in the Ranking API's `title` field. Without it the ranker scores raw chunk
# text, so a boilerplate section is indistinguishable from the same section of
# any sibling page — the identical blindness BM25_INDEX_AUG and the
# identity-prefixed embeddings already fixed for the two retrieval arms.
# No extra call and no extra billed query; `title` rides on the same record.
# ADOPTED 2026-08 (see ENTITY_SCOPED_RETRIEVAL — they ship as a pair).
RERANK_IDENTITY = _env_bool("RERANK_IDENTITY", True)
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
# Skip the (pooled/single) Ranking API call when the retrieval arms already
# agree: if at least this fraction of the fusion-ordered top slice was
# surfaced by BOTH dense and BM25 (or by 2+ sub-queries), trust fusion order
# and save the per-call fee. 0 disables (always rerank).
# Default back to 0 since the identity-embedded collection (202606da): the
# prefix concentrates dense results so arm consensus inflates (skip rate
# 60% -> 65%+) and fires on multi-page questions where fusion order alone
# measurably loses facts (dense_id_v1 vs _noskip in the round-2 benchmark).
# The 0.6 skip was calibrated for — and only earns its keep with —
# identity-blind vectors; the flag stays for cost-pressure rollback.
RERANK_SKIP_CONSENSUS = _env_float("RERANK_SKIP_CONSENSUS", 0.0)
# Ask the rewriter for extra retrieval signals in the SAME JSON call (a few
# output tokens, no extra request): exact keyword terms for the BM25 arm,
# and/or a hypothetical answer sentence (HyDE) for the dense arm.
# HYDE_MODE: "off" | "extra" (additional dense arm) | "replace" (single-turn
# dense query becomes the hypothetical sentence).
MQ_BM25_KEYWORDS = _env_bool("MQ_BM25_KEYWORDS", False)
HYDE_MODE = _env_str("HYDE_MODE", "off")
# Skip the rewrite/decompose LLM call for first-turn questions that are short
# and single-clause. Default OFF: safe on clean English questions, but the
# v2 messy-query eval (acronyms, typos, non-English — golden_set_v2.jsonl)
# showed raw queries lose hard without the rewrite (messy recall 1.000 ->
# 0.50-0.58), and the rewriter partly pays for itself by raising the
# rerank-skip consensus rate. Net saving when on is only ~$0.0002/query.
REWRITE_SKIP_SIMPLE = _env_bool("REWRITE_SKIP_SIMPLE", False)
REWRITE_SKIP_MAX_WORDS = _env_int("REWRITE_SKIP_MAX_WORDS", 12)
# Raised 512 -> 2048 on 2026-08-14, forced by the rewriter's move to
# 3.7-flash. Thinking tokens count against max_output_tokens, and 3.7-flash
# cannot be told not to think (see REWRITE_THINKING_LEVEL), so the old 512 cap
# had no room left for the ~430-token JSON body once "low" thinking is charged
# against it. 2048 is headroom chosen to make truncation impossible, NOT a
# measured figure: log finish_reason and the reasoning token count on a real
# run, then tighten it.
REWRITE_MAX_OUTPUT_TOKENS = _env_int("REWRITE_MAX_OUTPUT_TOKENS", 2048)
# "low" is the FLOOR on 3.7-flash, not a choice. 3.7-flash accepts only
# low/medium/high and returns an error for "minimal", so the thinking-0 tuning
# this call carried from 3.5-flash through 3.6-flash cannot be reproduced on
# it. This is the one place the 2026-08-14 migration changes runtime behavior:
# every turn now pays for some rewriter thinking, billed at output rate.
REWRITE_THINKING_LEVEL = _env_str("REWRITE_THINKING_LEVEL", "low")
# Thinking tokens are billed as output AND count against max_output_tokens —
# with the model default (~2k thinking) a 2048 cap truncates every answer
# (observed: finish=MAX_TOKENS with 80 visible tokens). "minimal" pairs with
# the 2048 cap; the eval sweep compares None (model default, +4096 cap) for
# answer quality. GEMINI_MODEL is 3.5-flash-lite, whose own default IS
# "minimal", so this is exactly the thinking_budget=0 behavior it replaced.
GEMINI_THINKING_LEVEL = _env_opt_str("GEMINI_THINKING_LEVEL", "minimal")

# --- Controller graph (LangGraph) -------------------------------------------
# One LLM node decides, each iteration, whether the BCIT corpus is needed at
# all (route), whether what has been retrieved is enough (gate), and what to
# fetch next (orchestration). It runs entirely BEFORE generation, so the
# answer is still produced by a single uninterrupted llm.stream() call and
# query_stream/_finalize_turn/server.py are untouched — the reason Self-RAG
# was rejected in 202608_adaptive_rag_prep does not apply here.
# ADOPTED 2026-08 (eval/benchmarks/202608_graph_controller). Measured against
# the same dependency stack with the flag off, controller on the lite tier:
#
#   set          url_hit            fact_recall        mis-routes  $/query
#   v1 (40)      0.9667 -> 0.9750   0.9917 -> 1.0000   0           0.0040 -> 0.0058
#   v2 (25)      1.0000 -> 1.0000   1.0000 -> 0.9800   0           0.0041 -> 0.0062
#   v3 (24)      0.8667 -> 0.9778   0.9148 -> 0.9426   0           0.0040 -> 0.0045
#   rough (16)   0.3333 -> 0.8333   0.6944 -> 0.9444   0           0.0039 -> 0.0047
#
# v3 multi_hop, the class this was built for: url 0.600 -> 0.933, fact
# 0.733 -> 0.833. out_of_scope avoidance 0.750 -> 1.000 on rough phrasings
# (production was explaining CSS flexbox to people who asked). unanswerable
# recall and the person guard set are unchanged.
#
# TWO GATES FAILED and were accepted on the evidence, not met:
#   - cost +13% (v3) to +51% (v2) against a +10% gate; ~$0.0053/query mean,
#     i.e. +$1.30 per 1,000 queries
#   - v3 multi_hop fact 0.833 against 0.90. The shortfall is mh3-02 (single-pass
#     ranking) and mh3-04's credits (generation) — neither is control flow, and
#     no controller change moves them.
#
# Turning it off is the rollback and needs no redeploy: with the flag off
# nothing imports langgraph and the turn takes exactly the pre-graph path.
USE_GRAPH = _env_bool("USE_GRAPH", True)
# Controller model. The dominant cost variable in the design: the reasoning
# tier is ~5x the lite tier per call and the controller fires on every turn.
# Both were measured on the same sets. 3.6-flash bought url 0.9917 on v1 (vs
# 0.9750) for +133% cost, and LOST to lite on the rough set (0.633 vs 0.733) —
# so the cheap tier is not a compromise here, it is also the better router.
GRAPH_MODEL = _env_str("GRAPH_MODEL", "gemini-3.5-flash-lite")
GRAPH_TEMPERATURE = _env_float("GRAPH_TEMPERATURE", 0.0)
# 512 truncated a hop-2 decision mid-JSON (unterminated string at char 1452),
# which the parse then rejected and the graph fail-opened on. A controller that
# silently stops controlling is the worst failure mode available to it, so the
# cap has headroom and the prompt bounds `reason` explicitly.
GRAPH_MAX_OUTPUT_TOKENS = _env_int("GRAPH_MAX_OUTPUT_TOKENS", 1024)
# "minimal" is the thinking_level spelling of the thinking_budget=0 this call
# has always run at. GRAPH_MODEL stays on 3.5-flash-lite, which supports
# "minimal" and in fact defaults to it, so the controller is untouched by the
# 2026-08-14 model move. Keep it that way: the controller fires on every turn
# and its per-call cost is the design's dominant cost variable.
GRAPH_THINKING_LEVEL = _env_str("GRAPH_THINKING_LEVEL", "minimal")
# Hard iteration cap, enforced in code regardless of what the controller asks
# for. The deterministic prototype of this gate could not fire on a question
# the corpus cannot answer (no relation in the index -> nothing to ask for);
# an LLM gate has no such structural guarantee, so the cap plus the
# concrete-entity requirement in graph.py are what keep `unanswerable` from
# looping. 3 leaves room for prereq-of-prereq chains; observed use is 1.
GRAPH_MAX_HOPS = _env_int("GRAPH_MAX_HOPS", 3)
# Per-document budget for the evidence digest the controller reads instead of
# the assembled context. Showing it the real context (~5.7k tokens) would cost
# more per gate call than the entire query does today; the digest is ~1.6k at
# this setting and is a better input for the judgement anyway — it is a
# coverage question, not a reading-comprehension one.
GRAPH_DIGEST_CHARS = _env_int("GRAPH_DIGEST_CHARS", 400)
# Context budget on turns where the controller actually took a hop. Turns that
# route straight through pay nothing extra.
#
# GRAPH_HOP_TOP_K without GRAPH_HOP_CONTEXT_MAX_CHARS would be a no-op:
# measured on the 24-case v3 run, context_chars is already mean 20,944 /
# p50 22,276 / max 23,900 against a 24,000 cap, so the assembler is dropping
# neighbor expansion and trailing sources to fit. The char cap is the binding
# constraint, not the chunk count — both have to move together.
GRAPH_HOP_TOP_K = _env_int("GRAPH_HOP_TOP_K", 20)
GRAPH_HOP_CONTEXT_MAX_CHARS = _env_int("GRAPH_HOP_CONTEXT_MAX_CHARS", 48000)
# "node": the controller's first call is the router (an extra request).
# "inline": the route decision rides on the rewriter's existing JSON call,
# saving one request and ~0.5 s at the cost of sharing a prompt with a
# differently-tuned task. Both are LLM routing; measured against each other.
GRAPH_ROUTER_MODE = _env_str("GRAPH_ROUTER_MODE", "node")
# Priced separately because the controller can run on a different model than
# either generation or rewriting. Same rule as the pair above: these MUST move
# with GRAPH_MODEL or the per-query cost shown to users quotes a model that
# did not run. Defaults are gemini-3.5-flash-lite list prices, matching the
# adopted GRAPH_MODEL above; the 3.6-flash pair is 1.50 / 7.50.
PRICE_GRAPH_INPUT_PER_M = _env_float("PRICE_GRAPH_INPUT_PER_M", 0.30)
PRICE_GRAPH_OUTPUT_PER_M = _env_float("PRICE_GRAPH_OUTPUT_PER_M", 2.50)

# Response language. "match": answer in the language of the student's latest
# question (retrieval + rewriter stay English — the corpus is English, and the
# v2 eval showed the rewriter's translation is load-bearing); "en": always
# English (legacy). Env-overridable for instant rollback.
RESPONSE_LANGUAGE = _env_str("RESPONSE_LANGUAGE", "match")
_LANGUAGE_RULES = {
    "match": (
        "   - Reply in the SAME language the student used in their latest question\n"
        "     below. If the question mixes languages, is only a course/program code,\n"
        "     or is otherwise ambiguous, default to English.\n"
        "   - The retrieved BCIT documents are in English: translate the facts you\n"
        "     use into the student's language, but keep program/course names, codes,\n"
        "     and all URLs exactly as written."
    ),
    "en": "   - Always respond in English only.",
}
_LANGUAGE_RULE = _LANGUAGE_RULES.get(RESPONSE_LANGUAGE, _LANGUAGE_RULES["match"])

# Static instructions come FIRST and the variable inputs (context, history,
# question) LAST: Gemini's implicit caching can only reuse a shared prompt
# prefix, and anything after the first changed byte is a cache miss.
# NOTE: this static prefix is only ~1k tokens, below the 2,048-token implicit-
# cache minimum, and it shows: 47/47 production query_usage lines (2026-06-18)
# and every v2/v3 eval run report cache_read_tokens=0. It is not dead, though —
# three cases in the August 2026 v1 runs (pa-05, sl-01, fu-04) each read 4,073
# cached tokens, i.e. the prefix PLUS the head of a preceding request's
# context, which only happens when consecutive queries share a long prefix.
# So the ordering is kept and the cache is treated as an occasional windfall,
# never a planned saving. Cost accounting prices every input token at full
# rate, which errs high on those cases. Per-query cost is instead cut by the
# response cache, which is deterministic.
RAG_PROMPT_TEMPLATE = """You are a BCIT (British Columbia Institute of Technology) academic advisor chatbot.

Your role:
- Answer the student's question using ONLY the provided BCIT documents and recent conversation history for any BCIT specific facts.
- You may use your general world knowledge only for non BCIT background explanations.
- Answer in the language set by the LANGUAGE rule in the instructions below.

INSTRUCTIONS:

1. LANGUAGE
__LANGUAGE_RULE__
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
   - If the student asks for BCIT specific information that is not present in the
     retrieved text, tell them that this specific information is not in the
     available documents (phrased in the same language as the rest of your answer).
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
   - At the end of your answer, add a section whose heading is the exact word
     "Sources" — keep that heading in English even when the rest of the answer
     is written in another language.
   - Under "Sources", list only the BCIT URLs that appear in the retrieved text and that
     you actually used, one per line, each formatted exactly as:
       - https://...
   - Output ONLY the URL itself on each line. Never copy the "Document N" labels,
     square brackets, or "[URL: ...]" wrappers that appear in the retrieved text.
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

# Inject the configured language rule. A non-brace sentinel keeps it out of the
# ChatPromptTemplate {var} namespace (which owns {context}/{chat_history}/etc).
RAG_PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE.replace("__LANGUAGE_RULE__", _LANGUAGE_RULE)

# Combined rewrite + decompose (one JSON call per turn, replaces the legacy
# rewrite when MULTI_QUERY_ENABLED). The schema is enforced server-side via
# response_schema; the prompt still spells out the rules for quality.
# Schema dialect note (2026-08-14): these are JSON Schema (lowercase types),
# not the uppercase proto spelling ("OBJECT"/"STRING"/"ARRAY") they carried
# under ChatVertexAI. ChatGoogleGenerativeAI sends response_schema through as
# the API's `response_json_schema` field, which is standard JSON Schema, so the
# uppercase spelling is no longer valid input. Same constraint, other dialect:
# the fields, nesting and required lists are unchanged.
REWRITE_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
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

# Optional rewriter outputs (experiment-gated). Conditional so that with the
# flags off the schema and template are byte-identical to the originals —
# zero drift in the rewriter's behavior or its prompt-cache prefix.
_extra_field_idx = 3
if MQ_BM25_KEYWORDS:
    REWRITE_DECOMPOSE_SCHEMA["properties"]["bm25_keywords"] = {
        "type": "array", "items": {"type": "string"},
    }
    REWRITE_DECOMPOSE_TEMPLATE = REWRITE_DECOMPOSE_TEMPLATE.replace(
        "\nRules:",
        f'\n{_extra_field_idx}. "bm25_keywords": 3 to 8 exact keyword-search terms for the question:'
        "\n   program/course codes, acronym expansions (e.g. CST -> computer systems technology),"
        "\n   and close synonyms. Return [] if none apply.\n\nRules:",
        1,
    )
    _extra_field_idx += 1
if HYDE_MODE != "off":
    REWRITE_DECOMPOSE_SCHEMA["properties"]["hyde_passage"] = {"type": "string"}
    REWRITE_DECOMPOSE_TEMPLATE = REWRITE_DECOMPOSE_TEMPLATE.replace(
        "\nRules:",
        f'\n{_extra_field_idx}. "hyde_passage": one short factual sentence that a BCIT web page answering'
        "\n   the question would plausibly contain. Name the concrete program or course.\n\nRules:",
        1,
    )

# --- Controller graph prompts (USE_GRAPH) -----------------------------------
# One schema serves all three jobs. On the first iteration the controller has
# no evidence and the decision IS the route; on later iterations it sees the
# digest and the decision IS the coverage gate. Same contract, different state.
GRAPH_CONTROLLER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["answer", "refuse", "retrieve"]},
        "reason": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "reason"],
}

# Rules 4 and 5 are the load-bearing ones. The deterministic version of this
# gate could not fire on an unanswerable question because the entity index had
# no relation to follow; this prompt has to earn that property instead, and
# graph.py enforces it a second time by rejecting a `retrieve` whose `missing`
# names nothing concrete.
GRAPH_CONTROLLER_TEMPLATE = """You control retrieval for a BCIT (British Columbia Institute of Technology) academic advisor chatbot.

Decide the SINGLE next action. Return JSON with:
1. "action": one of
   - "retrieve" - the BCIT corpus is needed and something specific is still missing.
   - "answer"   - respond now. Either no BCIT lookup was needed (greetings, thanks,
                  questions about what you can do), or the evidence below already
                  covers the question, or the corpus plainly does not contain it.
   - "refuse"   - the question is outside BCIT's scope (other institutions, general
                  programming help, weather, news, personal advice unrelated to BCIT).
2. "reason": ONE sentence, at most 20 words. Never a list.
3. "queries": when action is "retrieve", 1 to 3 specific search queries. Each must
   name a concrete course code, program, person, or topic. Never a pronoun.
4. "missing": when action is "retrieve", the concrete things still unaccounted for
   (e.g. "COMP 2510", "ACIT 2520 credits"). Leave empty otherwise.

Rules:
- ON ITERATION 0 THERE IS NO EVIDENCE YET. Every question about BCIT is
  "retrieve" at this point - including ones the corpus may well not answer,
  such as graduation rates, class sizes, failure rates or salary data. You
  cannot know what the corpus holds without looking, and telling a student the
  information is unavailable is only honest after a search has been run.
  "answer" on iteration 0 is for greetings, thanks, and questions about what
  you can do. "refuse" on iteration 0 is for questions that are not about BCIT
  at all.
- THE CONVERSATION HISTORY IS NOT A SOURCE. If the history already contains the
  fact the student is asking about, the answer is still "retrieve". Every BCIT
  fact in an answer has to be backed by a cited BCIT page, and a turn that
  skipped retrieval has nothing to cite. Follow-up questions - "what are its
  entrance requirements?", "how much does it cost?", "what about the
  prerequisites?" - are retrieve, always.
- FROM ITERATION 1 ONWARD, if the evidence shows the corpus does not hold the
  answer, choose "answer". The answer will say the information is unavailable.
  Searching again will not produce something the corpus does not contain.
- Do NOT choose "retrieve" unless you can name what is missing. "more detail",
  "additional information" and similar are not valid entries in "missing".
- A question answered in full by the evidence is "answer", even if more related
  material could exist.
- English only in this JSON, whatever language the student used.

Student's question:
{question}

Conversation history:
{chat_history}

Retrieval evidence so far:
{evidence}

Iteration {hop} of at most {max_hops}."""

# Generation prompt for turns the controller routed away from retrieval. It
# carries the same scope policy and the same language rule as the RAG prompt —
# "no retrieval" must not become "answer BCIT questions from memory", which is
# what would quietly break the out_of_scope avoidance the eval holds at 1.000.
# No Sources section: nothing was retrieved, so there is nothing to cite.
GRAPH_DIRECT_TEMPLATE = """You are a BCIT (British Columbia Institute of Technology) academic advisor chatbot.

This turn needs no document lookup. Reply directly.

INSTRUCTIONS:

1. LANGUAGE
__LANGUAGE_RULE__

2. WHAT TO DO
   - For greetings, thanks, and small talk: reply warmly in one or two sentences.
   - If asked what you can do: say you answer questions about BCIT programs,
     courses, admission requirements, tuition, campuses and student services,
     using official BCIT web pages. Keep it under four sentences.
   - If the question is outside BCIT's scope (another institution, general
     programming or homework help, weather, news, or anything unrelated to
     BCIT): say briefly that you can only help with BCIT topics, then name one
     or two BCIT things you could help with instead. Do not answer the
     off-topic question, even partially, and do not refer the student to
     another institution's website.

3. NEVER INVENT BCIT FACTS
   - No BCIT documents were retrieved for this turn, so you have no BCIT facts
     available. Do not state any BCIT specific detail - no dates, tuition
     figures, course codes, admission requirements, or URLs.
   - If the student is in fact asking for such a detail, say you need to look
     it up and invite them to ask directly about the program or course.

4. FORMAT
   - Plain prose, under 100 words. No headings, no bullet lists.
   - Do not add a "Sources" section and do not mention documents or context.

Conversation history:
{chat_history}

Student's question:
{question}

Answer:"""

GRAPH_DIRECT_TEMPLATE = GRAPH_DIRECT_TEMPLATE.replace("__LANGUAGE_RULE__", _LANGUAGE_RULE)

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
