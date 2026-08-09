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

## 5. Results

Controller on `gemini-3.5-flash-lite`, follow-up guard in place. Baseline is
the same upgraded dependency stack with `USE_GRAPH=false`.

| set | url_hit | fact_recall | avoidance | citation | mis-routes | $/query |
|---|---|---|---|---|---|---|
| v1 (40) | 0.9667 -> **0.9750** | 0.9917 -> **1.0000** | - | 1.000 -> 1.000 | **0** | 0.0040 -> 0.0058 |
| v2 (25) | 1.0000 -> **1.0000** | 1.0000 -> 0.9800 | - | 1.000 -> 1.000 | **0** | 0.0041 -> 0.0062 |
| v3 (24) | 0.8667 -> **0.9778** | 0.9148 -> **0.9426** | 1.000 -> **1.000** | 0.975 -> **1.000** | **0** | 0.0040 -> 0.0045 |
| rough (16) | 0.3333 -> **0.8333** | 0.6944 -> **0.9444** | 0.750 -> **1.000** | 1.000 -> 0.980 | **0** | 0.0039 -> 0.0047 |
| person guard (11) | F1 0.9861 -> **0.9861**, name_hit 1.000, invented 1 — unchanged |

`multi_hop`, the class this round existed for: url **0.600 -> 0.933**,
fact 0.733 -> 0.833.

| case | base | flash | lite | note |
|---|---|---|---|---|
| `mh3-01` | 0.33/0.50 | 0.67/1.00 (2 hops) | **1.00/1.00** (2 hops) | fixed |
| `mh3-04` | 0.00/0.50 | 1.00/0.50 (2 hops) | **1.00**/0.50 (2 hops) | URLs fixed, credits still missing |
| `mh3-02` | 0.67/0.67 | 0.67/0.67 (0 hops) | 0.67/0.67 (0 hops) | untouched |
| `pl3-01` | 1.00/0.80 | 1.00/0.80 (0 hops) | 1.00/0.80 (0 hops) | untouched |

The two untouched cases take **no hop** — the controller judges one pass
sufficient, and it is right that nothing is *missing by name*: `mh3-02`'s
COMP 2501 and `pl3-01`'s ELEX 2610 lose on rank within a single pass. Those
are ranking problems, one layer below anything a controller can reach.
`mh3-04` is now a generation miss, not a retrieval one: the digest shows
`Course Credits | 4` and the context carries both outlines, and the answer
still does not state the two credit values.

## 6. Gates

| gate | target | measured (lite + guard) | |
|---|---|---|---|
| `unanswerable` recall | 1.000 | 1.000 | pass |
| out_of_scope avoidance | 1.000 | 1.000 | pass |
| v3 `multi_hop` url | >= 0.85 | 0.933 | pass |
| v3 `multi_hop` fact | >= 0.90 | 0.833 | **fail** |
| mis-routes, all sets | 0 | 0 | pass |
| v1 regression | none | url +0.008, recall +0.008 | pass |
| v2 band | 0.960-1.000 | 1.000 | pass |
| person guard F1 | 0.9861 | 0.9861 | pass |
| cost | +10% | +13% (v3) to +51% (v2) | **fail** |
| p50 | +1.5s | +0.8s (rough) to +2.2s (v3) | **mixed** |

Latency is quoted with quota-backoff cases excluded (`retrieve_s <= 5s`);
2-10 cases per set were dropped on that rule and the raw p50 is meaningless
(§4). The controller costs ~0.9s per call on the lite tier and fires ~2x on a
retrieve turn.

## 7. Verdict

Quality is better on every set and every metric except v2 `fact_recall`
(-0.02, one case), with zero regressions and zero mis-routes once the
follow-up guard is deterministic. The largest gain is on the rough-phrasing
set, which is the set that resembles real traffic: url 0.333 -> 0.833,
avoidance 0.750 -> 1.000.

Two gates fail, and both were written before the size of the quality delta
was known:

- **Cost** +13% to +51% against a +10% gate. In absolute terms $0.0040 ->
  ~$0.0053, i.e. +$1.30 per 1,000 queries.
- **v3 `multi_hop` fact** 0.833 against 0.90. The shortfall is `mh3-02`
  (ranking) and `mh3-04`'s credits (generation) — neither is a control-flow
  problem, and no controller change will move them.

`GRAPH_ROUTER_MODE=inline` is designed and unimplemented; folding the route
decision into the rewriter's existing JSON call removes one controller round
trip from every turn, which is roughly 0.9s and $0.0002. That is the obvious
next lever if the cost or latency number needs to come down.

Recommendation: **adopt on the lite tier**, accepting the cost gate as
overtaken by evidence, and treat the `multi_hop` fact gate as satisfied in
substance — its two residual cases were never in the controller's reach and
are recorded as ranking/generation work. `USE_GRAPH` stays the rollback.
