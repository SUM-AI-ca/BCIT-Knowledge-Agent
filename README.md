# BCIT AI Advisor — RAG Chatbot

**Live: [https://bcitai.ca](https://bcitai.ca)** — tech blog landing page at `/`, chatbot at [`/chat`](https://bcitai.ca/chat).

A retrieval-augmented generation (RAG) chatbot that answers questions about BCIT
programs, courses, admissions, and campus life, grounded in **11,129 documents
crawled from the live BCIT website** (September 2024 intake onwards).
Embeddings, vector search, reranking, and generation are all managed services
(Vertex AI + Cloud SQL). Every Google Cloud call authenticates through
Application Default Credentials — the VM's attached service account in
production — so no Google Cloud API key exists to leak, rotate, or commit.
Every query emits a LangSmith trace and a structured cost/latency log line.

The retrieval configuration is not a generic RAG recipe: every value below was
selected against BCIT's specific information architecture and re-measured when
that architecture pushed back. See
[Why these hyperparameters are BCIT-specific](#why-these-hyperparameters-are-bcit-specific).

| Component | Service |
|---|---|
| Corpus | Own crawler (`crawl_bcit.py`) — sitemaps + BCIT outlines API |
| Embeddings | Vertex AI `gemini-embedding-001` (1536-dim) |
| Vector store | Cloud SQL PostgreSQL + pgvector (HNSW) |
| Sparse retrieval | In-process BM25 (`rank_bm25`) |
| Rank fusion | Reciprocal Rank Fusion (alpha 0.48, k=60) |
| Reranker | Vertex AI Ranking API (`semantic-ranker-default-004`) |
| LLM | `gemini-3.5-flash-lite` (generation) + `gemini-3.6-flash` (query rewriter) via LangChain `ChatVertexAI` |
| Backend | FastAPI + uvicorn (Python 3.10, managed via uv) |
| Frontend | React 19 + Vite 7 (no router — pathname-based pages) |
| Hosting | GCE `e2-medium` VM + Cloudflare (DNS, TLS, port routing) |
| Observability | LangSmith tracing (project `bcit-chatbot`) |

---

## How a query flows

1. **Rewrite + decompose** — one schema-constrained JSON call per turn
   (`REWRITER_MODEL=gemini-3.6-flash`, temperature 0, thinking 0, ~430
   tokens) returns a standalone question (pronouns resolved from history)
   plus 1–4 self-contained sub-queries. The rewriter deliberately runs on a
   stronger model than generation: sub-query quality drives retrieval
   (benchmark: hit-rate 0.929 with a lite rewriter vs 0.963 with 3.5-flash). Single questions yield one sub-query;
   "admission requirements AND tuition AND housing" yields three. Parse
   failure falls back to the raw question (counted in `query_usage` logs).
2. **Embed** — each query is embedded with Vertex AI `gemini-embedding-001`
   (1536 dimensions).
3. **Hybrid retrieval** — dense (pgvector HNSW, MMR, `fetch_k=50`,
   `ef_search=100`) and sparse (in-process BM25 — exact identifiers like
   "COMP 1510" that embeddings miss) merge via **Reciprocal Rank Fusion**:
   `1 / (60 + rank)` per list, weighted by `HYBRID_ALPHA=0.48`. RRF compares
   ranks, not raw scores, so incompatible similarity scales never need
   normalizing.
   - *Single-topic turns* retrieve once at full width (23+23 → RRF top 25).
   - *Multipart turns* fan out per sub-query (12+12 → top 15 each) on a
     shared thread pool, then rank-interleave into one deduped pool
     (cap 40). Retrieval returns metadata copies — BM25 documents alias the
     shared corpus pickle, and annotating them in place would corrupt it.
   - *Entity-scoped arm* (`ENTITY_SCOPED_RETRIEVAL`): when a sub-query names a
     concrete course code or program, a third arm scores only that entity's own
     chunks and joins the pool. This is the only path that can surface a
     templated section — an outline's evaluation table names no course and
     3,000-odd sibling chunks share its wording, so global ranking cannot
     separate them. Sub-millisecond (it scores the entity's chunks directly,
     not the whole index) and costs no extra billed call.
4. **Rerank** — ONE Vertex AI Ranking API call per turn regardless of
   sub-query count (the API bills per query): the pooled candidates are
   scored against the standalone question (`semantic-ranker-default-004`),
   the top `RERANKER_TOP_K=10` are kept, and a coverage quota guarantees
   every sub-query keeps ≥ 2 of its own chunks (its best leftovers swap in,
   evicting only above-quota docs). Each record carries its page identity in
   the API's `title` field (`RERANK_IDENTITY`); without it the ranker scores
   identity-blind text and prefers a sibling page's identical section. It
   ships as a pair with the entity-scoped arm — neither fixes on its own what
   the two fix together.
5. **Assemble context from chunks** — the selected chunks themselves go into
   the prompt (not their whole source files, which is what the pre-June-2026
   pipeline did at ~44k input tokens/query). Each chunk pulls in ±2
   neighbors from an in-process ordinal index built off the BM25 pickle at
   startup, consecutive runs merge with their 130-char overlaps deduped, and
   the result is capped at 24k chars (neighbor expansion drops from the
   lowest-ranked sources first). Citation headers
   (`Document N: [URL: …]`) are identical to the legacy format — the
   Sources-section prompt instructions depend on them.
6. **Generate** — `GEMINI_MODEL=gemini-3.5-flash-lite`, static instructions
   first / variable inputs last (cache-friendly ordering, though the ~1k-token
   prefix sits below Gemini's 2,048-token implicit-cache minimum, so the cache
   stays inert today — verified `cache_read_tokens=0`; per-query cost is cut by
   the first-turn response cache instead),
   `{question_parts}` enumerating multipart sub-questions,
   `max_output_tokens=2048` with `thinking_budget=0` (see gotchas — thinking
   counts against the cap).

Answer language: retrieval and the rewrite always run in **English** (the
corpus is English), but generation replies in the **language of the student's
question** (`RESPONSE_LANGUAGE=match`; set `en` to force English). The original
question — not the English rewrite — is what the generation prompt carries, so
the model knows which language to use; facts are translated from the English
context while program/course names, codes, URLs, and the literal "Sources"
heading stay verbatim. Verified live in English, Korean, Spanish, French, and
Japanese, including follow-ups and multipart questions.

First-turn cache: before step 1, a question with **no conversation history**
is looked up in an in-process TTL/LRU keyed on its normalized text — an exact
repeat returns the stored answer in <100 ms at zero API cost and skips the
whole pipeline (rewrite → retrieve → rerank → generate). Only no-history turns
are cached (follow-ups depend on session state); the cache clears on restart,
so it never serves an answer from a retired corpus. Exact-match only — no
semantic similarity, so a near-miss can never return the wrong answer.

Conversation state: each browser session holds a 5-turn
`ConversationBufferWindowMemory`; sessions expire after 30 minutes. Saved
answers are stripped of their Sources section and capped at 1,500 chars —
history is re-sent every turn, so it pays to keep it lean. Chat requests run
in a thread pool so the FastAPI event loop never blocks.

Answers stream: the UI talks to `POST /chat/stream` (SSE), which sends
tokens as generation produces them — first token lands right after the
rewrite/retrieve/rerank stages (~1.5–2 s) instead of after the full answer
(3.6–6.7 s). The blocking `POST /chat` stays for the eval harness and as the
UI's automatic fallback; both paths share the exact same pipeline
(`_prepare_turn`/`_finalize_turn` in `query_rag.py`), so memory, cost
accounting, and the `query_usage` log line are identical. Guardrails:
`CHAT_TIMEOUT_S=90` (504 — the underlying worker thread is uncancellable,
which is why `WORKER_THREADS=4` keeps headroom over expected concurrency)
and `MAX_MESSAGE_CHARS=2000` (422).

Every query produces one LangSmith trace (project `bcit-chatbot`) rooted at
`bcit_query`, with a dedicated `vertex_rerank` retriever span; one
structured `query_usage` JSON log line per query records tokens
(input/output/reasoning/cache-read), per-stage latency, sub-query count,
context size, and fallback flags.

---

## Cost & accuracy optimization (June – August 2026)

Measured on a 40-case golden set (`backend/eval/golden_set.jsonl`: outline
facts, program/admission, student life, multipart, follow-up pairs) against
the same corpus. The "before" column is the original implementation-focused
build: whole documents in the prompt, no decomposition, `gemini-3.1-pro`, and
no cost instrumentation of any kind. Cost optimization begins *after* this
table — the columns below compare pipeline shape, not price:

| Metric | Before (full-doc context) | After (chunks + decompose) |
|---|---|---|
| Input tokens / query (mean) | 44,322 | **5,884 (−87%)** |
| Input tokens / query (p50) | 36,471 | 5,936 (−84%) |
| Output tokens (incl. thinking) | 210 + 896 thinking | **175 + 0** |
| Retrieval URL hit-rate | 0.892 | **0.963** |
| Key-fact recall | 0.848 | **0.885** |
| Citation precision | 1.000 | 1.000 |
| Latency p50 / p95 | 7.6 s / 10.8 s | **4.1 s / 5.2 s** |
| Truncated answers / decompose fallbacks | 0 / 0 | 0 / 0 |

The legacy pipeline stays one env away:
`CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false RERANKER_TOP_K=13`.
Sweep notes: `NEIGHBOR_RADIUS=2` beat 1 (outline recall +0.04 for +5.6%
tokens); `RERANKER_TOP_K=13`, `RERANK_SCORE_THRESHOLD=0.2`, and model-default
thinking all measured worse (see `config.py` comments for why).

A follow-up model benchmark (`eval/benchmarks/202606_model_comparison/`)
then cut cost a further 69%: generation moved to `gemini-3.1-flash-lite`
with the rewriter kept on `gemini-3.5-flash` — $0.0126 → $0.0039/query
($12.61 → $3.87 per 1k) at hit-rate parity (0.963) and the best fact recall
of any run (0.890). At lite-class generation prices the Ranking API's fixed
$0.001/query is now the largest cost component.

Round 2 (`eval/benchmarks/202606_retrieval_cost_experiments/`, 8 experiments
/ 26 gated runs) attacked exactly that and the section-flooding weakness:

- **BM25 title-aware indexing** (`BM25_INDEX_AUG`) — the BM25 vectorizer now
  indexes `title + category + URL-slug + chunk text` while serving unchanged
  documents (no re-embed, no pickle rebuild). Deep program-page chunks carry
  their page identity, so "X program entrance requirements" no longer floods
  with sibling programs: multipart hit 0.917 → **1.000**.
- **Consensus rerank-skip** (`RERANK_SKIP_CONSENSUS=0.6`) — when ≥60% of the
  fusion top slice was surfaced by both retrieval arms, fusion order stands
  and the Ranking API call is skipped (~half of all queries). Guard: a turn
  that skipped the rewriter always reranks — every query keeps at least one
  semantic stage.
- **Simple-query rewrite skip** (`REWRITE_SKIP_SIMPLE`) — implemented and
  initially adopted, then **reverted by a second 25-case eval set**
  (`eval/golden_set_v2.jsonl`) built around messy real-world queries:
  acronyms, typos, and a Korean-language question. Skipping is free on clean
  English prose but messy recall collapsed 1.000 → 0.50-0.58 without the
  rewriter (which normalizes, corrects, and translates — and raises the
  rerank-skip rate enough to refund part of its own cost). The flag stays
  available for clean-traffic cost pressure.

Round-2 shipped result: **v1 set hit 0.975 / recall 0.931 / $0.00315 (−18%)**;
v2 messy set hit 0.940 / recall 0.937 (reproduced twice), p50 −0.3 s.
Rejected with evidence: bigger
rerank pools (sibling-chunk dilution), cheaper rewriters (2.5-flash hit 0.929
twice — rewrite quality is load-bearing), keyword/HyDE rewriter extensions
(no gain once BM25 is title-aware, +$0.0005 in 3.5-flash output tokens), and
all fusion-parameter moves (alpha 0.48 / rrf_k 60 / MMR λ 0.87 confirmed
optimal).

A Tier-2 follow-up then gave the **dense arm** the same page identity the
BM25 arm got in round 2: the corpus was re-embedded with
`"title (category). "` prefixed to the *embedded* text only
(`build_pgvector.py` `EMBED_IDENTITY_PREFIX` + `REUSE_PICKLE` — stored
chunks verified byte-identical corpus-wide, new blue-green collection
`bcit_docs_202606da`, ~$4 / 85 min). A dense-only probe improved 5/5
identity queries (a BMET query's own-page chunks in top-20: 6 → 20/20);
the full pipeline took the **v2 messy set to hit 1.000 / recall 0.950**
(×2 identical) and fixed the long-stuck v2 follow_up category
(0.750 → 1.000), with the clean set at parity. One interaction surfaced:
identity vectors concentrate dense results, inflating arm consensus enough
that the round-2 rerank-skip began firing on multi-page questions and
losing facts — so the skip was retired (`RERANK_SKIP_CONSENSUS=0.0`; its
saving was real only for identity-blind vectors). Net cost ~$0.0039/query.

### The full metric history (June 2026, every stage archived)

How the production numbers moved across the four optimization passes — all
rows are archived eval runs on the same 40-case clean set
(`eval/benchmarks/202606_model_comparison/` and
`…/202606_retrieval_cost_experiments/`):

| Stage | What changed | hit | recall | in_tok | $/query | p50 / p95 |
|---|---|---|---|---|---|---|
| 0. Baseline | whole-document context, no decomposition, `gemini-3.1-pro` + default thinking (896 tok/answer) | 0.892 | 0.848 | 44,410 | not comparable¹ | 7.6 / 10.8 s |
| 1. Chunk pipeline | decompose → parallel hybrid fan-out → pooled rerank + quota → neighbor expansion ±2, thinking 0 | 0.963 | 0.885 | 6,263 (−86%) | $0.0126 | 4.1 / 5.2 s |
| 2. Model mix | generation → `3.1-flash-lite`, rewriter kept on `3.5-flash` | 0.963 | 0.890 | 6,274 | $0.0039 (−69%) | 3.6 / 4.8 s |
| 3. Round 2 | BM25 title-aware index + consensus rerank-skip @0.6 | 0.975 | 0.931 | 5,877 | $0.00315 | 3.4 / 6.4 s |
| 4. Round 2 end | identity-prefixed embeddings; rerank-skip retired (quality over the $0.0007) | 0.975 | 0.925 | 6,259 | $0.00385 | 3.6 / 6.7 s |
| 5. **Current** (2026-08) | entity-scoped retrieval + reranker identity, on the 3.5-flash-lite / 3.6-flash pair | **0.991** | **0.991** | 5,526 | $0.00405 | see note² |

¹ The baseline predates any cost work. It ran `gemini-3.1-pro` because the
goal at that stage was to get the pipeline correct, not cheap — there was no
per-query cost instrumentation, no model comparison, and no token budget. A
figure of ~$0.078/query has been quoted for this row in the past; that was an
after-the-fact estimate computed at *flash* prices from the measured 44,410
input tokens, so it understates what a Pro model at that token volume actually
cost. Rather than restate a number that was never measured, the row is left
without one. What the row is evidence for is the token and quality baseline
(44,410 input tokens, hit 0.892, recall 0.848), all of which were measured.

² Row 5 is the first benchmark on the current model pair and the first run of
the corrected scorer; rows 0-4 are the older pair. Its `fu-04` is excluded
(n=39): that case scores 1.000 or 0.000 depending on whether the rewriter
resolves "the newest residence" to "Tall Timber", independent of
configuration, and it flipped between two otherwise identical runs. Latency is
not quoted — the runs were measured from WSL through a cold Cloud SQL proxy to
us-west1, which is environment-dominated. Reproduced twice with identical
quality numbers (`eval/benchmarks/202608_adaptive_rag_prep/`).

> **The `recall` column above is under-reported by ≈0.036.** The key-fact
> matcher had three systematic bugs (2026-08): a fact ending in a digit never
> matched its ordinal form, so every `"July 2"` answer written as "July 2nd"
> scored as a miss; hyphenated compounds (`"4.0-credit"`) missed their spaced
> alternative; and `\w` is Unicode-aware, so a Korean particle glued to a token
> (`"english studies 12에서"`) read as a word continuation — which silently
> penalised every non-English answer, i.e. exactly what `RESPONSE_LANGUAGE=match`
> produces. `eval/rescore.py` re-scored all 49 archived runs from their stored
> `answer_excerpt` (no DB, no API): **every run moved up by +0.025 to +0.070
> (mean +0.036) and 0 of 816 pairwise config rankings inverted**, so no past
> ADOPT/REJECT decision changes. Corrected current figures: **v1 recall 0.925 →
> 0.956, v2 recall 0.950 → 1.000**. `eval/test_matcher.py` locks all three bugs
> and their guards.

> **These rows are archived runs, not the shipping configuration.** Every row
> above — including the one labelled *Current* — was measured on the previous
> model pair (`gemini-3.1-flash-lite` generation + `gemini-3.5-flash` rewriter).
> In August 2026 both moved to their successors (`gemini-3.5-flash-lite` /
> `gemini-3.6-flash`), which has **not** been re-benchmarked, so the numbers are
> left as measured rather than restated for models that never ran. The quality
> columns should carry over closely; the `$/query` column will not — flash-lite
> output went $1.50 → $2.50/M while the rewriter went $9.00 → $7.50/M. The cost
> shown live under each answer is computed from `config.py`'s current prices and
> is accurate; these figures are historical. Re-run `eval/run_eval.py` to refresh.

The v2 messy-query set (acronyms, typos, Korean — created in round 2) tells
the second half of the story:

| Stage | hit | recall | messy cat. | follow_up cat. |
|---|---|---|---|---|
| Round-2 config | 0.940 | 0.937 | 1.000/1.000 | 0.750 (stuck in every config) |
| + rewrite-skip on | 0.900 | 0.817 | **0.583 — reverted** | 0.750 |
| **Current** | **1.000** | **0.950** | 1.000/1.000 | **1.000 — fixed** |

Net: cost $0.0126 → $0.0039 (−69% from the first instrumented figure; the
`gemini-3.1-pro` baseline before it was never cost-instrumented, so the true
reduction from the starting point is larger and is not quoted), hit 0.892 →
0.975 clean / 1.000 messy, recall 0.848 → 0.925–0.950,
input tokens −86%, thinking tokens 896 → 0, p50 −53%, citation precision
1.000 throughout. The path was not monotonic: 45+ gated runs rejected ~15
configurations outright and **reversed two adopted ones** (the simple-query
rewrite skip, caught by the v2 set; the consensus rerank-skip, invalidated
by identity embeddings) — and the final step deliberately spent +$0.0007
per query to buy quality back.

### Current production configuration (August 2026, after entity-scoped retrieval + reranker identity)

The complete live setup, with every value env-overridable for rollback:

| Stage | Setting | Value |
|---|---|---|
| Rewrite + decompose | `REWRITER_MODEL` | `gemini-3.6-flash`, temp 0, JSON schema, thinking 0 — runs on **every** turn (`REWRITE_SKIP_SIMPLE=false`: the v2 eval showed raw acronym/typo/non-English queries collapse without it) |
| Embeddings | `PG_COLLECTION=bcit_docs_202606da` | `gemini-embedding-001`, 1536-dim MRL, corpus embedded as `"title (category). chunk"` — stored text untouched; shares `documents_202606.pkl` with BM25 (chunks byte-identical) |
| Retrieval (per sub-query) | dense / sparse | pgvector HNSW MMR (λ 0.87, fetch 50) + in-process BM25, RRF α 0.48 / k 60 |
| BM25 index | `BM25_INDEX_AUG=true` | vectorizer fit on `title + category + filename keywords + URL slug + text`; served documents untouched |
| Entity-scoped retrieval | `ENTITY_SCOPED_RETRIEVAL=true`, `ENTITY_SCOPED_K=8` | when a sub-query names a concrete course code or program, a BM25 arm restricted to that entity's own chunks joins the pool (k is per **source** — a course resolves to its outline plus its catalogue page). Closes the boilerplate-section class: a chunk whose body carries no entity identity and whose wording is shared by 1,772-3,038 siblings cannot be ranked globally by any phrasing, and is rank 1 once scoped. Sub-millisecond; no extra billed call |
| Rerank identity | `RERANK_IDENTITY=true` | page identity (title + filename keywords + category) rides in the Ranking API's `title` field. Without it the ranker scores identity-blind text and prefers a *sibling* course's identical section — it ranked COMP 4870's evaluation table 0.859 against 0.258 for the asked-about course's own chunks. Ships as a pair with the row above: **neither flag alone fixes anything the other does not** |
| Rerank | `RERANK_MODE=pooled`, `RERANK_SKIP_CONSENSUS=0.0` | one `semantic-ranker-default-004` call on **every** turn over the merged pool (≤100 records = 1 billed query); per-sub-query coverage quota ≥2. The round-2 consensus skip is retired: identity embeddings inflate arm agreement and fusion-only selection loses facts on multi-page questions |
| Context | `NEIGHBOR_RADIUS=2`, `CONTEXT_MAX_CHARS=24000` | small-to-big neighbor expansion from the in-process ordinal index, render-and-shrink cap |
| Generation | `GEMINI_MODEL` | `gemini-3.5-flash-lite`, temp 0.05, max 2048, thinking 0 |
| Response language | `RESPONSE_LANGUAGE=match` | generation replies in the **student's** language while retrieval + rewrite stay English (the corpus is English; the rewriter's translation is load-bearing). Facts are translated from the English context, but program/course names, codes, URLs, and the literal "Sources" heading are kept as-is. `en` forces English (legacy) |
| Memory | `MEMORY_WINDOW_K=5` | per-session window; history stores answers Sources-stripped, capped 1500 chars |
| Server | `CHAT_TIMEOUT_S=90`, `WORKER_THREADS=4`, `MAX_MESSAGE_CHARS=2000` | request deadline → 504 (the timeout frees the request, not the uncancellable worker thread — the extra workers are the backstop), 4 IO-bound chat workers, input cap → 422 |
| Response cache | `RESPONSE_CACHE_ENABLED=true` | first-turn (no-history) questions key an in-process TTL/LRU on the normalized text (`RESPONSE_CACHE_MAX=1000`, `RESPONSE_CACHE_TTL_S=86400`); an exact repeat returns in <100 ms at $0. Exact-match only (no semantic similarity → never a wrong answer); follow-ups never cached; cleared on restart so it never outlives a corpus rebuild |

Measured quality (both benchmark sets in `eval/`, all runs archived in
`eval/benchmarks/202606_retrieval_cost_experiments/`; every number below
reproduced in two identical runs):

| Metric | v1 set (40 clean cases) | v2 set (25 incl. messy) |
|---|---|---|
| URL hit rate | 0.975 | **1.000** |
| Key-fact recall | 0.925 | 0.950 |
| Citation precision | 1.000 | 1.000 |
| Cost / query | $0.00385 | $0.00392 |
| Latency p50 | ~3.6 s | ~4.2 s |

Cost anatomy at these settings: generation input ≈ $0.0015, generation
output ≈ $0.0009, rewriter ≈ $0.0006, reranker $0.001 (every turn),
embedding ≈ $0.00001 — still −69% vs the pre-optimization $0.0126.

### The eval harness

`backend/eval/golden_set.jsonl` — 40 cases written against the actual corpus
(every expected URL and key fact verified in the source documents): 12
course-outline facts, 10 program/admission, 6 student life, 6 multipart, 6
follow-up pairs (scored on the second turn, exercising the rewrite).
`expected_urls` and `key_facts` are lists of alternative-groups — a group
counts when ANY alternative matches, so "$500 to $800" / "$500–$800" phrasing
differences don't punish correct answers. Facts match with word-boundary
normalization ("75" ≠ "175", "$6,000" = "$6000").

```bash
cd backend
export PG_CONNECTION=$(grep '^PG_CONNECTION=' .env | cut -d= -f2- | sed 's/:5432/:5433/')  # WSL

# measure a configuration (env flags BEFORE launch — config reads env at import)
.venv/bin/python eval/run_eval.py --label after
CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false RERANKER_TOP_K=13 \
  .venv/bin/python eval/run_eval.py --label before

# compare two runs: aggregate deltas + per-case regressions
.venv/bin/python eval/run_eval.py --compare eval/results/before.json eval/results/after.json

# subsets while iterating
.venv/bin/python eval/run_eval.py --label quick --category multipart --limit 3

# LLM judge: faithfulness + completeness graded against the retrieved
# passages (one extra JUDGE_MODEL call per case, ~$0.002 each)
.venv/bin/python eval/run_eval.py --label after_judged --judge

# golden-set candidates mined from production LangSmith traces; thumbs-down
# feedback (journalctl dump) floats problem cases to the top
.venv/bin/python eval/mine_langsmith.py --days 30 --feedback-log /tmp/feedback.log
```

`--judge` adds `judge_faithfulness` / `judge_completeness` /
`judge_unsupported_claims` per case and the corresponding means to the
aggregate (`JUDGE_MODEL=gemini-3.6-flash`, schema-constrained JSON, graded
strictly against the retrieved passages). Substring fact recall stays the
headline metric — the judge complements it by catching paraphrases the
substring match misses and by flagging claims the context never contained.
A judge failure never sinks a case (`judge_error` + count instead).

`eval/golden_set_v3.jsonl` — 21 cases covering what v1/v2 do not measure, built
2026-08 to make an adaptive/corrective-retrieval decision testable. Every fact
verified against the corpus, every URL taken from the crawled documents:

| Category | n | What it isolates |
|---|---|---|
| `scoped_fact` | 5 | a boilerplate outline section (exam weights, hours/weeks, term dates) of a **named** course — the `ol-04`/`ol-11` failure class, on five different courses |
| `multi_hop` | 5 | a second retrieval target only knowable after reading the first result (prerequisites-of-prerequisites, reverse "what requires X") — the parallel rewrite+decompose call structurally cannot name it up front |
| `unanswerable` | 3 | facts verified **absent** from the corpus; correct behaviour is to say so, and the failure mode is a confident invented number |
| `out_of_scope` | 3 | not BCIT, or not advising at all — today they pay for the full pipeline |
| `chitchat` | 3 | greetings/thanks — same |
| `exploratory` | 2 | the shape real traffic actually takes (topic search, person lookup), taken verbatim from LangSmith |

New scoring field `must_not_contain` (same alternative-group shape as
`key_facts`) reports `avoidance` — the only way to score a case whose correct
answer is a refusal, since a recall-shaped metric returns `None` there.

```bash
# re-score archived runs after a scorer or golden-set change (no DB, no API,
# no regeneration — every run stores answer_excerpt + retrieved_urls)
.venv/bin/python eval/rescore.py eval/benchmarks/*/*.json --verbose

# matcher regression tests (three real bugs + their guards)
.venv/bin/python eval/test_matcher.py
```

`rescore.py` recomputes the URL metrics too, purely as a self-check: they do not
depend on the fact matcher, so if `url_hit_rate` ever moves the *tool* is wrong,
not the run. It prints `MISMATCH` if that happens.

`eval/mine_langsmith.py` closes the data loop: it pulls root `bcit_query`
runs, drops questions already in the golden sets (eval traffic), dedupes,
and writes `eval/candidates_<date>.jsonl` with answer excerpts and
sub-queries for triage. Curation stays manual on purpose — golden-set facts
must be verified against the corpus before a candidate is promoted.

Each run writes per-case records (retrieved URLs, missed facts, token usage,
stage timings, answer excerpt) plus aggregates to `eval/results/<label>.json`
(gitignored — measurements are per-machine). Metrics: URL hit-rate (groups of
acceptable sources), key-fact recall, citation precision (cited URLs ⊆
retrieved context), input/output/reasoning/cache-read tokens, p50/p95
latency, MAX_TOKENS truncations, decompose fallbacks.

---

## Why these hyperparameters are BCIT-specific

None of the values above are defaults, and none of them generalise for free.
Each one was chosen because a measurable property of BCIT's site broke the
generic behaviour, and each was re-measured against the golden sets rather than
argued from first principles. The corpus is templated: 7,060 course pages,
3,262 course outlines and 529 program pages, all machine-generated from the
same CMS, which means the discriminating signal is almost never in the prose.

| BCIT structural property | Measured | Parameter it forced |
|---|---|---|
| The outline template repeats verbatim across every course | **3,258** chunks contain `Prerequisite(s) \|`, **3,038** contain `Total Hours \|`, **2,980** contain `Minimum Passing Grade \|`, **1,772** contain `Final Exam \|` — and the chunk carrying those rows names no course | `ENTITY_SCOPED_RETRIEVAL` + `RERANK_IDENTITY` (2026-08). Global ranking cannot separate 3,000 near-duplicates; restricted to the named course's own chunks the answer is rank 1 |
| Program pages share section headings | `Entrance requirements` on **370** of 529 pages, `Costs & Supplies` on **435** | `BM25_INDEX_AUG=true` — the sparse index is fit on `title + category + filename keywords + URL slug + text` so deep chunks carry page identity. Multipart hit 0.917 → 1.000 |
| Deep program chunks are near-identical *in vector space* too | BMET entrance-requirement probe: own-page chunks in the dense top-20 went **6/20 → 20/20** after prefixing the embedded text with page identity | `EMBED_IDENTITY_PREFIX`, collection `bcit_docs_202606da`. v2 URL hit 0.940 → 1.000, and it retired the round-2 rerank-skip, which had been calibrated for identity-blind vectors |
| Students ask with exact identifiers (`COMP 1510`, `ACIT 2515`) that embeddings blur | `HYBRID_ALPHA=0.56` (more dense weight) collapses hit-rate to 0.854; 0.40 and 0.48 hold ~0.95+ | `HYBRID_ALPHA=0.48` — BM25's 0.52 share is load-bearing, the opposite of the dense-heavy default most RAG stacks ship |
| Course codes are a closed, regex-matchable vocabulary | **7,623** codes indexed; the filename convention parses **3,259 / 3,262** outlines and **7,049 / 7,060** course pages | the entity index behind `ENTITY_SCOPED_RETRIEVAL`, and `build_pgvector.py` deriving outline metadata from filenames rather than page scraping |
| Facts live in tables that straddle chunk boundaries | `NEIGHBOR_RADIUS=2` beat 1 on outline fact recall (+0.04) for +5.6% tokens | `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=130`, `NEIGHBOR_RADIUS=2` with overlap-aware merging |
| The same course is published for up to 6 terms | retrieval surfaced stale instructors and dates alongside current ones | latest-term-per-course outline policy at crawl time — removes the contradiction class instead of ranking around it |
| Sitemap `lastmod` is unusable as a freshness signal | **78%** of course pages carry a 2022 CMS-migration timestamp; filtering on `lastmod ≥ 2024-09` would have dropped **89%** of courses | live crawl every refresh, no `lastmod` filter |
| Answers must cite BCIT URLs and nothing else | citation precision **1.000** across every archived run | corpus-only prompt invariant; a web-search fallback is rejected on these grounds, not on cost |

Two further parameters were confirmed rather than chosen: `RRF_K=60` and
`MMR_LAMBDA=0.87` were swept (40/80 and 0.80/0.95 respectively) and both
alternatives scored at or below centre, so the defaults stand *because they
were tested here*, not because they are the library defaults.

**How much testing sits behind this.** Across the June 2026 rounds and the
August 2026 round, **60+ gated evaluation runs** on three golden sets (40
clean, 25 messy/multilingual, 21 targeted) rejected roughly eighteen
configurations, and **reversed two that had already been adopted** — the
simple-query rewrite skip (caught by the messy set, where raw acronym/typo/
non-English queries collapsed 1.000 → 0.583) and the consensus rerank-skip
(invalidated once identity-prefixed embeddings changed what "the arms agree"
means). Two more adopted values were later found to be *interacting* rather
than independent: entity-scoped retrieval and reranker identity each fix
exactly one case alone and three together.

**The honest caveat.** This configuration is fitted to one institution's site.
Porting it to another catalogue would keep the *method* — measure the corpus's
duplication structure, then decide where identity has to be injected — but not
the numbers. `HYBRID_ALPHA`, `NEIGHBOR_RADIUS`, the identity prefixes and the
entity index all assume templated pages with stable code conventions and a
sparse arm that carries real weight. On a corpus of prose without repeated
scaffolding, most of these knobs would be doing nothing, and the ones that
matter here (identity injection at all three stages) would be unnecessary.

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
  production serves the old one, flip the `PG_COLLECTION` **and**
  `DOCUMENTS_PICKLE` defaults in `config.py` **together** (the dense and BM25
  sides must serve the same crawl — June 2026 the pickle default was left
  behind and local BM25 quietly served the retired corpus), deploy config +
  pickle, restart. Zero downtime; the old collection doubles as instant
  rollback until you drop it (`drop_old_collection.sh` pattern).

---

## Design decisions (and why)

A decision record for future development — each of these was deliberate:

1. **Managed services over self-hosted models.** The first version ran FAISS
   with a locally hosted cross-encoder. It was migrated to Vertex AI + Cloud SQL
   pgvector: embeddings, reranking, and generation became API calls, leaving the
   application host responsible only for BM25 and orchestration. That removed
   model-serving from the deployment surface (no weights to ship, no serving
   runtime to patch, no capacity planning per model swap) and made the
   `3.1-flash-lite → 3.5-flash-lite` and `3.5-flash → 3.6-flash` moves
   configuration changes rather than migrations.
2. **`gemini-embedding-001` at 1536-dim MRL truncation.** pgvector's HNSW index
   supports ≤ 2000 dims, so `gemini-embedding-001` (3072 native) is truncated to
   1536 via MRL. `gemini-embedding-2` (multimodal, GA 2026-04) was A/B-tested
   2026-06 and **regressed** retrieval on both golden sets (v1 URL-hit
   0.975→0.950, fact recall 0.912→0.900; v2 URL full-hit 1.000→0.960, fact
   recall 0.930→0.897) at no cost win — its value is multimodal, irrelevant to
   this text-only corpus, and it needs the `google-genai` SDK at
   `location="global"` (the legacy `VertexAIEmbeddings` path rejects it). 001 stays.
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
12. **Chunks in the prompt, whole files never.** The original pipeline loaded
    each cited source file from disk into the prompt (~9k tokens × ~3–5 files
    = 44k input tokens/query) to preserve list/table completeness across
    chunk boundaries. Neighbor expansion (±2 chunks from the in-process
    ordinal index) buys that completeness for a few hundred tokens instead;
    eval showed accuracy went UP (fact recall 0.848 → 0.885) because precise
    chunks beat 9k-token haystacks.
13. **One Ranking API call per turn, not per sub-query.** The Ranking API
    bills per query; pooled rerank with `top_n=len(pool)` + a per-sub-query
    coverage quota keeps multipart quality without multiplying the fixed
    cost (`RERANK_MODE=per_subquery` exists as the comparison flag).
14. **Decompose on every turn.** The old history-only rewrite skipped turn 1,
    missing first-turn multipart questions. The merged rewrite+decompose
    call is ~430 tokens with thinking 0 — cheap enough to always run, and
    single questions short-circuit back to full-width legacy retrieval.
15. **Page identity has to be injected at all three ranking stages.** The
    corpus is templated, so a chunk's body is often not enough to tell which
    page it belongs to. Each stage needed the identity added separately and
    each was found the same way — by a case that failed for no other reason:
    the sparse arm in round 2 (`BM25_INDEX_AUG`, fit on title + category +
    filename keywords + URL slug), the dense arm in the identity-prefixed
    rebuild (`EMBED_IDENTITY_PREFIX`, collection `bcit_docs_202606da`), and the
    reranker in August 2026 (`RERANK_IDENTITY`, page identity in the Ranking
    API's `title` field). Fixing two of the three is not enough: asked about
    one course's exam weights the ranker scored two *other* courses' evaluation
    tables 0.859 / 0.709 against 0.258 for the right course, because those
    tables are exactly what the question describes and only the course name
    separates them.
16. **Thinking off for generation.** This is an extraction-and-summarize
    workload over provided context: eval measured model-default thinking as
    strictly worse (recall 0.852 vs 0.877, p95 2.2×, ~900 thinking tokens
    billed per query).

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
- **Thinking tokens count against `max_output_tokens`.** With the model
  default thinking budget and a 2048 cap, gemini-3.5-flash burned ~1,950
  tokens thinking and truncated every answer at ~80 visible tokens
  (`finish_reason=MAX_TOKENS`, Sources lost). Pair the cap with
  `GEMINI_THINKING_BUDGET=0`, or raise the cap to ≥ 4096 if re-enabling
  thinking. The `query_usage` log warns on MAX_TOKENS.
- **`RERANK_SCORE_THRESHOLD` undermines the multipart quota.** The threshold
  filter runs during context assembly — after pooled-rerank selection — so it
  strips exactly the lower-scored chunks the coverage quota swapped in
  (eval: multipart recall 0.819 → 0.611 at threshold 0.2). Leave it at 0
  unless re-evaluating multipart cases alongside.
- **`config.py` reads env at import.** Anything that should affect a run
  (eval flags, `PG_CONNECTION` port rewrite) must be exported before Python
  starts; `load_dotenv()` does not override already-exported variables.
- **Local uvicorn + LangSmith cold init can stall queries for minutes.** On
  the WSL box, the first traced calls after a local server start sometimes
  block while the LangSmith client initializes over a slow network path
  (observed 2026-06-12: an easter-egg query — zero GCP calls — took 6 s in a
  plain script and stalled minutes under uvicorn while requests piled up;
  identical code with `LANGSMITH_TRACING=false` answered instantly, and a
  later tracing-on run was also fine). Production on GCE is unaffected.
  Run local smoke tests with `LANGSMITH_TRACING=false` first; only add
  tracing once the no-tracing path is green.
- **A restarted Cloud SQL proxy must postdate ADC reauth.** The proxy caches
  credentials from launch: after `gcloud auth application-default login`,
  an already-running proxy keeps failing (`server closed the connection
  unexpectedly` on 5433) until you kill and restart it. And mind `pkill -f
  cloud-sql-proxy` from a `bash -c` one-liner — the pattern matches the
  shell's own command line and kills it (exit 15); use the saved pidfile.

---

## System state reference

Everything deployed/configured outside this repo, in one place:

| Thing | Value |
|---|---|
| Production URL | https://bcitai.ca (chat at `/chat`) |
| GitHub | `SUM-AI-ca/bcit-RAG-chatbot` (private; moved from `jp-ml/`) |
| GCP project | `wine-agent-jh-2026` |
| VM | `bcit-rag-vm`, us-west1-b, **e2-medium** (resized from e2-standard-2 2026-06-18), **34.187.152.133 (static, reservation `bcit-rag-ip`)**, firewall: tcp:8000 from **Cloudflare IPv4 ranges only** (rule `allow-bcit-chat`, tag `bcit-rag`, locked down 2026-06-12 — debug via `gcloud compute ssh` + `curl localhost:8000`, direct IP times out) |
| Cloud SQL | `bcit-rag-pg` (us-west1), db `ragdb`, user `raguser`, pgvector |
| Live collection | `bcit_docs_202606da` — 100,515 chunks from 11,129 docs (June 2026 crawl, identity-prefixed embeddings). `bcit_docs_202606` is the identity-blind predecessor, retained for rollback; both hold byte-identical chunk text and share `documents_202606.pkl` |
| Cloudflare | zone `bcitai.ca`: A `@` → 34.187.152.133 (proxied), Origin Rule port→8000, SSL Flexible |
| LangSmith | project `bcit-chatbot` ([smith.langchain.com](https://smith.langchain.com)) — tracing on since 2026-06-10, enabled purely via env vars |
| Secrets | `backend/.env` (`PG_CONNECTION`, `LANGSMITH_API_KEY`) — local + VM copies, never committed. Google Cloud access is ADC only (VM service account in production), so there is no Google Cloud key in this file |
| Rollback assets | `backend/data_old_202409/` (old corpus, gitignored), VM `vectorstore/documents_old.pkl` |

`www` CNAME added 2026-06-11 (`www` → `bcitai.ca`, proxied) — verified:
`https://www.bcitai.ca/health` and `/chat` both 200 through the same Origin
Rule; `server.py` CORS already allowlists the `www` origin.

### Edge/origin hardening state (2026-06-12)

- ✅ **"Always Use HTTPS" on** — `http://bcitai.ca` 301s to https (verified).
- ✅ **Origin locked to Cloudflare** — `allow-bcit-chat` (tcp:8000) source
  ranges replaced with Cloudflare's published IPv4 list
  (www.cloudflare.com/ips-v4, 15 CIDRs); direct `34.187.152.133:8000` access
  verified blocked. Debugging goes through `gcloud compute ssh bcit-rag-vm`
  + `curl localhost:8000`. If Cloudflare expands its ranges, rerun:

  ```bash
  gcloud compute firewall-rules update allow-bcit-chat \
    --source-ranges="$(curl -s https://www.cloudflare.com/ips-v4 | tr '\n' ',' | sed 's/,$//')"
  ```

- ⏸ **Rate limiting deferred** — Cloudflare rate-limiting rules require a
  paid add-on on this plan. In-app guardrails (input cap, 90 s deadline,
  worker pool) bound the damage but not the bill (~$4/1k queries); accepted
  for now — revisit if `/metrics` shows abnormal volume.

---

## Future work — where to start

Ideas that fit the current architecture, with their natural hook points:

- **Per-source incremental refresh** — `build_pgvector.py` currently
  appends-or-skips; add a "delete rows for source X, reinsert" mode so a
  changed page can be re-embedded without a new collection. Hook:
  `existing_ids()` / `deterministic_ids()`.
- **Scheduled term refresh** — the whole refresh is two commands (runbook
  below); wire into cron/GitHub Actions once credentials story is decided.
- **Program-page section flooding** — the one retrieval weakness the eval
  left open: section headings like "Entrance requirements" exist on all 529
  program pages, so BM25 floods the pool with sibling programs' chunks and
  the asked-about program's section chunk sometimes loses (multipart CST
  cases in `eval/results/`). Hook: section-aware metadata at build time
  (`chunk_index` is already stamped), or a BM25 field boost on the
  program/course code.
- **Fan-in retention** — "which courses require ACIT 1515 as a prerequisite?"
  has three correct answers on three sibling outlines, and `RERANKER_TOP_K=10`
  keeps one of them. The candidates are reachable (all three sit in the BM25
  top-25 for a terse query) but the pooled rerank spreads the context budget
  across unrelated sources. Needs a retention rule for candidates sharing a
  relation to the named entity — not another retrieval hop. `BM25_STOPWORD_DF`
  is implemented and rejected for the candidate-set half of it.
- **Second-hop retrieval** — "what are the prerequisites of COMP 4537's
  prerequisites?" The model reads hop 1, names COMP 1537 and COMP 3522 itself,
  then says it cannot reach their details: the decomposer emits sub-queries in
  parallel, so it cannot name a target that is only knowable after the first
  result. This is the one place a cycle is warranted (`eval/benchmarks/
  202608_adaptive_rag_prep/REPORT.md` §6). Guard: the loop must never fire on a
  question the corpus genuinely cannot answer — the in-process entity index
  answers "does this entity exist at all" for free.
- **Follow-up referent resolution** — `fu-04` scores 1.000 or 0.000 depending
  on whether the rewriter turns "the newest residence" into "Tall Timber
  Student Housing" or leaves it generic (then it retrieves the 1978 Maquinna
  residence). Roughly 50/50 across four runs, independent of retrieval config;
  `mp-05` misses the same Tall Timber content in every configuration.
- **Term-aware retrieval** — outline chunks carry `term_code` metadata already;
  boosting newer terms (or filtering expired ones at query time) is a metadata
  filter away.
- **Citations UI cards** — the Sources section renders as clickable links
  today; richer cards (page titles, categories) need a structured
  `sources: [{url, title}]` array in the chat response (metadata is already
  on the retrieved docs server-side).
- **Ship logs somewhere queryable** — `/metrics` (in-process aggregates),
  LangSmith traces, and `query_usage`/`feedback_log` JSON lines exist;
  remaining: a Cloud Logging sink or BigQuery export so log lines survive
  restarts and can be queried historically.
- **Frontend dist in CI** — `frontend/dist` is gitignored and deployed by scp;
  a GitHub Action building + deploying on push would remove the manual step.
- **Migrate the embedding SDK off the deprecated path** — `embeddings.py` uses
  `langchain_google_vertexai.VertexAIEmbeddings` (the legacy
  `vertexai.language_models` path, removal scheduled ~2026-06-24). The 2026-06
  emb2 trial confirmed the already-installed `google-genai` SDK
  (`genai.Client(vertexai=True).models.embed_content`) is retrieval-equivalent
  for 001 — swap to it (also unlocks newer models, e.g. emb2 at
  `location="global"`). Hook: `backend/embeddings.py`.

Shipped from this list in August 2026: **entity-scoped retrieval + reranker
identity**, which closed the long-open `ol-04`/`ol-11` failure (they were one
bug, not two — the answer chunk carries no course identity and 1,772 chunks
share its wording, so no global ranking reaches it); `golden_set_v3.jsonl`
(scoped facts, multi-hop, unanswerable, out-of-scope, chit-chat, exploratory);
`eval/rescore.py` + `eval/test_matcher.py` after three matcher bugs were found
to be under-reporting every archived run by ~0.036.

Shipped from this list in June 2026: SSE streaming (+ `/chat` fallback),
request timeout + worker headroom, input caps, `run_eval.py --judge`,
`/metrics`, per-answer feedback buttons + `/feedback`, trace mining,
first-turn response cache, route-split frontend bundle, VM right-sized to
`e2-medium` (≈ half the prior monthly cost at the same quality).

## Repository layout

```
backend/
  server.py                FastAPI app: /chat API + serves the built frontend
  query_rag.py             BCITChatbot — retrieval pipeline + LLM chain
  hybrid_retriever.py      BM25 + pgvector with RRF fusion
  reranker.py              Vertex AI Ranking API client
  embeddings.py            Vertex AI embedding wrapper
  config.py                All tunables (models, top-k, chunking, collection)
  response_cache.py        First-turn exact-match answer cache (in-process TTL/LRU)
  crawl_bcit.py            Corpus crawler (sitemaps + outlines API)
  build_pgvector.py        Indexing job (resumable, collection-versioned)
  drop_old_collection.sh   One-off: drop a retired collection via proxy
  eval/golden_set.jsonl    40-case eval set (facts verified against corpus)
  eval/golden_set_v2.jsonl 25-case messy-query set (acronyms, typos, Korean)
  eval/golden_set_v3.jsonl 21-case set: scoped facts, multi-hop, unanswerable,
                           out-of-scope, chitchat, exploratory
  eval/run_eval.py         Offline eval harness (--label, --compare, --judge)
  eval/rescore.py          Re-score archived runs offline after a scorer change
  eval/test_matcher.py     Key-fact matcher regression tests
  eval/mine_langsmith.py   Production traces -> golden-set candidate JSONL
  eval/results/            Per-run metrics JSON (gitignored)
  data/                    Crawled corpus (11,129 txt docs, 16 categories)
  vectorstore/             documents_<version>.pkl — BM25 source (generated, not committed)
frontend/
  src/App.jsx              Pathname routing; lazy-loads Blog/Chat as separate chunks
  src/Chat.jsx             Chat UI (own lazy chunk: SSE streaming + stats footer)
  src/Blog.jsx             Tech blog landing page (architecture deep dive)
  vite.config.js           Dev proxy + manualChunks vendor split (react, lucide)
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Blog landing page (also serves any non-API path as SPA fallback) |
| `GET /chat` | Chatbot UI |
| `POST /chat` | `{message, session_id?}` → `{reply, session_id, stats}` — `stats` = per-reply `{input_tokens, output_tokens, total_tokens, cost_usd, latency_s, model}` (rendered as the footer under each answer; cost = role-priced LLM tokens + Ranking API call + embedding, prices in `config.py`) |
| `POST /chat/stream` | Same request body, SSE response: `session` ({session_id}) → `delta` ({text} fragments) → `done` ({stats}), or `error` ({detail}). The UI uses this and falls back to `POST /chat` if unavailable |
| `POST /feedback` | `{session_id?, verdict: "up"/"down", question?, answer_excerpt?}` from the per-answer thumbs buttons → one structured `feedback_log` JSON log line (journald); `eval/mine_langsmith.py --feedback-log` floats the downs to the top of the golden-set candidates |
| `POST /reset` | Clear a session's conversation history |
| `GET /health` | Status + active session count |
| `GET /metrics` | Aggregates since boot (queries, errors, timeouts, tokens, cost) + last-hour p50/p95 latency + response-cache hit/miss. Public — contains no question/session contents |

No authentication — the chatbot is public. Input is capped at
`MAX_MESSAGE_CHARS` (2000 → 422) and every request carries a
`CHAT_TIMEOUT_S` (90 s → 504) deadline.

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
              GCE bcit-rag-vm (us-west1-b, e2-medium)
                      │  systemd: bcit-chatbot.service (uvicorn :8000)
                      │  systemd: cloud-sql-proxy.service
                      ▼
              Cloud SQL PostgreSQL + pgvector  ·  Vertex AI APIs
```

### Deploying a code update

```bash
gcloud auth login                        # separate from ADC — see gotchas
bash backend/deploy.sh                   # deploys what the last commit changed
bash backend/deploy.sh query_rag.py ...  # or name the modules explicitly
```

`backend/deploy.sh` exists because every way this deploy goes wrong is a
documented gotcha that a pasted one-liner walks straight into: long commands
lose spaces at terminal wrap points (bash then tries to *execute* `config.py`);
`gcloud compute ssh` needs its own `gcloud auth login`, which is not the same
session as the ADC credentials `run_eval.py` uses; and that login leaves the
active project wherever the account defaults to, which is not
`wine-agent-jh-2026`. The script passes `--project` explicitly on every call and
confirms the VM is visible *before* uploading anything, so a wrong-project
session fails in a second instead of half-way through.

The script prints the file list and the local HEAD, asks for confirmation, then
stages through `/tmp` (`/opt/bcit-rag` is root-owned), restarts the service,
polls `/health` until `chatbot_loaded: true`, prints the startup log lines that
prove which retrieval flags are live, and finally sends one real question to
production.

**It refuses to deploy an empty file list.** The pipeline lives in
`query_rag.py`, `hybrid_retriever.py`, `reranker.py` and `config.py`, so a
change can touch none of `server.py` — the August 2026 retrieval work changed
exactly those four and left `server.py` alone. The older runbook copied only
`server.py`, which would have restarted the service on old code and reported
success. `backend/eval/` is not used in production and is never deployed.

The frontend is deployed separately and only when `frontend/src` changed:

```bash
cd frontend && npm run build
gcloud compute scp --recurse frontend/dist bcit-rag-vm:/tmp/dist-new --zone=us-west1-b
gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  sudo rm -rf /opt/bcit-rag/frontend/dist &&
  sudo mv /tmp/dist-new /opt/bcit-rag/frontend/dist'
```

Startup takes ~30-45 s (pickle load, BM25 fit, entity index), which is why the
script polls rather than sleeping a fixed interval. Expect these lines:

```
Loaded 100,515 documents
Entity index: 7,623 course codes, 498 programs
Reranker loaded: semantic-ranker-default-004 (identity titles)
```

and the COMP 1510 question to answer **40%** — that case is answered wrongly by
the pre-August configuration, so it distinguishes a real deploy from a restart
on old code.

Two rollback paths: `git checkout <previous-sha> -- backend/` then redeploy, or
flip the retrieval flags off with no deploy at all —
`ENTITY_SCOPED_RETRIEVAL=false RERANK_IDENTITY=false` in the VM's
`/opt/bcit-rag/backend/.env`, then `sudo systemctl restart bcit-chatbot`.

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

# 3. Smoke-test against the new collection (same env vars + query_rag), then
#    flip BOTH defaults in config.py: PG_COLLECTION and DOCUMENTS_PICKLE
#    (they version together — dense and BM25 must serve the same crawl).
#    Documented exception: an embeddings-only rebuild (REUSE_PICKLE=true
#    EMBED_IDENTITY_PREFIX=true) reuses the serving pickle by design — the
#    chunks are byte-identical, so only PG_COLLECTION flips (e.g. 202606da)

# 4. Deploy config + pickle (keep the versioned filename — it matches the
#    config default and leaves the previous pickle in place for rollback)
gcloud compute scp backend/config.py bcit-rag-vm:/tmp/config.py --zone=us-west1-b
gcloud compute scp vectorstore/documents_<version>.pkl bcit-rag-vm:/tmp/documents_<version>.pkl --zone=us-west1-b
gcloud compute ssh bcit-rag-vm --zone=us-west1-b --command='
  sudo cp /tmp/documents_<version>.pkl /opt/bcit-rag/backend/vectorstore/ &&
  sudo cp /tmp/config.py /opt/bcit-rag/backend/config.py &&
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

---

## License & attribution

Full statement in [`NOTICE`](NOTICE); the short version:

**Corpus.** The 11,129 text files in `backend/data/` are copies of pages from
BCIT's public website — **© British Columbia Institute of Technology**. BCIT's
website terms allow copying and distributing that content for informational,
non-commercial purposes, unmodified, and only if the copyright notice travels
with it. That notice is carried in [`NOTICE`](NOTICE), in
[`backend/data/README.md`](backend/data/README.md) next to the files, and in the
footer of both web pages. BCIT may revoke the permission at any time; on written
notice this content comes down.

**Not official.** This project is **not affiliated with, endorsed by, or
sponsored by** BCIT. "BCIT" and "British Columbia Institute of Technology" are
BCIT's registered trade marks, used here only to identify what the chatbot
answers questions about. No BCIT logo is reproduced. **bcit.ca is always the
authoritative source** — the corpus is a point-in-time snapshot and goes stale.

**Answers are generated, not quoted.** The model paraphrases retrieved page text
(`config.py`: *"Do not copy long paragraphs verbatim from the documents"*), so
answers are summaries rather than BCIT content, and can be wrong or out of date.
They are not official BCIT information or advice.

**Non-commercial.** A personal engineering project. Nothing is sold, no ads, no
referral or affiliate revenue.

**No warranty.** Provided "as is", without warranty of any kind, express or
implied, including the implied warranties of merchantability, fitness for a
particular purpose, and non-infringement.

The project's own source code carries no license grant — all rights reserved
unless a `LICENSE` file is added.
