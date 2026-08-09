# LLM controller graph (LangGraph) — August 2026

Question on the table: replace the deterministic route/gate prototype with an
LLM controller that makes every control decision — whether the corpus is
needed, whether what was retrieved is enough, and what to fetch next.

Status: **measurement in progress.** Sections 1-4 record findings that are
settled and reproduced; section 5 onward is filled from the gate runs.

## 0. Why this round exists

`202608_adaptive_rag_prep` closed with a roadmap whose remaining items were
"coverage gate + one hop-2 re-retrieval" and "route field in the rewriter".
Its own measurement had shrunk the LangGraph mandate to **two cases** in v3's
`multi_hop`. Before building, the open v3 failures were re-counted:

| case | url | recall | what is missing |
|---|---|---|---|
| `mh3-01` | 0.333 | 0.500 | comp 2510, comp 2522 |
| `mh3-04` | 0.000 | 0.500 | 4 credits, 5 credits |
| `mh3-02` | 0.667 | 0.667 | comp 2501 |
| `pl3-01` | 1.000 | 0.800 | elex 2610 |

Identical across four runs — deterministic failures, not noise. `multi_hop`
is 100% of v3's URL deficit and 87% of its recall deficit.

A cheaper alternative was proposed first and rejected by the user in favour of
the LLM design: BCIT outlines carry `Prerequisite(s) | ...` on 3,258 of 3,262
files and `Course Credits | ...` on 3,171, and a reverse grep of ACIT 1515
returns exactly `mh3-02`'s three expected courses. That structured field did
not go to waste — it became the digest's evidence source (§3).

## 1. Shape

One LLM node, one JSON contract, called once per iteration:

    iteration 0 (no evidence)  -> the decision IS the route
    iteration 1+ (digest)      -> the decision IS the coverage gate
    the sequence               -> the orchestration

The graph runs entirely inside `_prepare_turn`. Generation stays outside it,
so the answer is still one uninterrupted `llm.stream()` call and
`query_stream` / `_finalize_turn` / `server.py` are untouched. This is why a
loop is compatible with streaming here when the previous round rejected
Self-RAG for not being.

Retrieval was lifted out of `_prepare_turn` verbatim into
`_retrieve_and_rerank`, so the existing pipeline serves as the first hop
unchanged and `USE_GRAPH=false` runs the same statements as before.

## 2. What replaced the structural safety guarantee

The deterministic prototype could not loop on an unanswerable question: with
no relation in the entity index there was nothing to ask for. An LLM gate has
no such property, so it is enforced in code instead:

- a `retrieve` whose `missing` names nothing concrete is downgraded to
  `answer` (`graph.is_concrete`)
- the hop cap is enforced on the graph edge, not in the prompt
- a controller exception fails open to today's behaviour, never to a refusal

Ten stubbed unit checks pin these (`_tmp_graph_unit.py`).

## 3. Five defects, all found by measurement

None of these were reachable by the unit checks; each needed a live run.

**3.1 Unanswerable questions mis-routed to `direct`.** The prompt listed
graduation rates and class sizes as things the corpus does not hold, and the
controller applied that on iteration 0, where there is no evidence at all.
What the corpus holds is not knowable without looking, and telling a student
the information is unavailable is only honest after a search. The rule is now
bound to iteration >= 1. Routes 4/5 -> 5/5.

**3.2 The digest failed at its only job.** On the 2-hop case the controller
asked for COMP 1537 and COMP 3522 prerequisites, got them, could not see that
they had arrived, asked again, and hit the cap. The digest showed the head of
each source's top chunk, which did not happen to contain the answering line.
It now extracts the structured `Label | value` fields — verified:
`Prerequisite(s) | COMP 1537 and COMP 3522` appears in COMP 4537's entry.

**3.3 Follow-up turns lost their citations.** Four v1 `follow_up` cases routed
`direct` because the conversation history already contained the fact. Some
answers were factually right (`"COMP 1510 is worth 6 credits."`) but carried
no Sources section, so url_hit went 1.000 -> 0.000. The conversation history
is not a source; a turn that skips retrieval has nothing to cite. Fixed in the
prompt, and — the deeper fix — v1 and v2 now carry `expected_route` on every
case, because `route_mismatches=0` had been computed over zero eligible cases.

**3.4 `GRAPH_MAX_OUTPUT_TOKENS=512` truncated a hop-2 decision mid-JSON**, the
parse rejected it, and the graph fail-opened. A controller that silently stops
controlling is the worst failure available to it. Raised to 1024.

**3.5 The graph triples embedding calls on hop turns**, and this project
already sits at the `online_prediction_requests_per_base_model` ceiling for
gemini-embedding. Not a defect in the graph — a consequence of it, and the
difference between an occasional 429 and a user-visible 500. `embeddings.py`
now waits out quota exhaustion with jittered backoff; only 429 is retried.

## 4. Measurement hazards hit in this round

Four separate infrastructure faults produced numbers that looked like results:

- **The embedding retry contaminated every latency figure.** `retrieve_s` max
  80.7s and 84.0s are the backoff sleeping, not the pipeline working. The
  first latency read (p50 4.29s -> 9.74s) was discarded.
- **ADC expired mid-run** (`RefreshError`), killing v1 at 13/40.
- **Re-login attached the wrong quota project** (`sumai-web-2026`, not
  `wine-agent-jh-2026` where the resources live) — the documented gotcha in
  this repo's deploy notes, and a plausible contributor to the 429 storms.
- **The Cloud SQL proxy held a stale token** after the re-login: the port
  stayed open and every connection was closed by the server, which killed a
  v2 run and all eight ph-03 repeats before it was diagnosed.

The lesson worth carrying: on this project a latency or error-rate number is
only meaningful after checking `retrieve_s` distribution and `n_errors` first.
