# Adaptive/Corrective RAG — prerequisites and decision baseline (2026-08)

Question on the table: would Adaptive-RAG (LangGraph) meaningfully improve this
system? It could not be answered, because v1/v2 are saturated (corrected recall
0.956 / 1.000) and neither set contains a case the proposal targets. This round
builds the measurement instead of the feature.

Three things came out of it: the scorer was wrong, the two long-open retrieval
failures are one bug with a known fix, and the v3 baseline says which of the
three published techniques is actually needed.

## 1. The scorer was under-reporting every run by ~0.036

`_fact_hit` failed three ways at once. Found by re-scoring archived runs, not by
reading code:

| bug | example | cases |
|---|---|---|
| digit-final fact never matched its ordinal form | `"july 2"` vs "July 2**nd**" | pa-02, fu2-01, mp2-01, mp-01 |
| hyphenated compound | `"4.0 credit"` vs "4.0**-**credit" | ol-07 |
| `\w` is Unicode-aware, so a Korean particle read as a word continuation | `"english studies 12"` vs "…12**에서**" | ms2-05 |

The third one penalised *every* non-English answer — which is precisely what
`RESPONSE_LANGUAGE=match` exists to produce.

`eval/rescore.py` re-scores archived runs from their stored `answer_excerpt` +
`retrieved_urls` (no DB, no API, no regeneration), so fixing the scorer does not
strand 49 archived runs:

```
49 runs re-scored: fact_recall +0.025 … +0.070 (mean +0.036)
816 pairwise config comparisons: 0 ranking inversions outside the noise band
URL self-check: 49/49 runs reproduced their stored url_hit_rate exactly
```

No past ADOPT/REJECT decision changes. Corrected: **v1 recall 0.925 → 0.956,
v2 recall 0.950 → 1.000**. `eval/test_matcher.py` locks all three bugs plus the
guards that must survive them (`"75"` ≠ "175", `"15"` ≠ "1510", `"july 2"` ≠
"July 20").

## 2. `ol-04` and `ol-11` are one bug, and query rewriting cannot fix it

Both were filed separately — "probably a corpus gap" and "flaky". The fact is in
the corpus (`COMP_1510_202610.txt:52`, `Final Exam | 40`).

The answer-bearing chunk begins `"Midterm Exam | 25 / Final Exam | 40 /
Participation | 5"`. Its body names no course, and **1,772 chunks share the
`"Final Exam |"` wording** (3,038 share `"Total Hours |"`). `BM25_INDEX_AUG`
prepends page identity but lifts all 3,262 sibling outlines equally.

Measured offline against the corpus pickle (sparse arm isolated, no DB/API):

| query | rank of the answer chunk |
|---|---|
| "COMP 1510 final exam grade percentage weight" | not in top 25 |
| "In COMP 1510, what percentage of the final grade is the final exam worth?" | not in top 25 |
| "COMP 1510 evaluation criteria" | not in top 25 |
| **same question, restricted to COMP 1510's own 13 chunks** | **1** |

The target is not out-ranked — it is *indistinguishable* from ~3,000
near-duplicates. No reformulation of the query changes that, which rules out the
standard CRAG "rewrite and retry" payload for this class. What fixes it is a
metadata filter on the entity the rewriter already extracted.

## 3. v3 baseline (21 cases, `golden_set_v3.jsonl`)

`v3_baseline.json` — production config, 2026-08 model pair
(`gemini-3.5-flash-lite` generation / `gemini-3.6-flash` rewriter).

| category | n | url_hit | fact_recall | avoidance | in_tok | $/query |
|---|---|---|---|---|---|---|
| scoped_fact | 5 | 1.000 | 0.800 | – | 5,900 | 0.0041 |
| **multi_hop** | 5 | **0.600** | **0.733** | – | 5,981 | 0.0044 |
| unanswerable | 3 | – | 1.000 | – | 5,354 | 0.0037 |
| exploratory | 2 | 1.000 | 1.000 | – | 6,168 | 0.0042 |
| out_of_scope | 3 | – | – | 1.000 | 5,934 | 0.0038 |
| chitchat | 3 | – | – | 1.000 | 6,037 | 0.0039 |
| **overall** | 21 | 0.833 | 0.844 | 1.000 | 5,891 | 0.0040 |

Citation precision 1.000; zero `must_not_contain` violations; zero fabricated
claims across all 21.

> **Do not read latency from this run.** It was measured from WSL through a cold
> Cloud SQL proxy to us-west1; `retrieve_s` spiked to 31.9 s and `rewrite_s` p50
> was 3.97 s against 1.22 s in the archived runs. The environment, not the
> pipeline. Quality columns are unaffected.

### What each category settled

**`unanswerable` 3/3, `avoidance` 1.000, citation precision 1.000, zero
fabricated claims.** The generator is already honest — on every failure it says
so rather than inventing. Self-RAG exists to fix a dishonest generator, so it
has nothing here to fix. Its post-hoc critique is also incompatible with the
shipped SSE streaming, and the offline `run_eval.py --judge` already covers the
grading use case.

**`multi_hop` is the only collapse**, and the failure shape is ideal for a
corrective loop. `mh3-01` answered:

> "The prerequisite courses for COMP 4537 are COMP 1537 and COMP 3522. The
> available documents do not contain information regarding the prerequisites for
> COMP 1537 or COMP 3522."

The model reads hop 1, **names the hop-2 entities itself**, then reports it
cannot reach them. `mh3-04` does the same in Korean. Both the retry trigger and
the retry target are already in the output, for free. The decomposer did emit a
second sub-query on mh3-01 — "the prerequisites for the prerequisite courses of
COMP 4537" — which names no concrete entity and therefore cannot retrieve
anything. That is the structural limit of parallel decomposition.

**`out_of_scope` + `chitchat` are correct but expensive**: 6 cases at a mean
**$0.0038, 5,985 input tokens and 1.0 Ranking API calls each.** "hello" pays for
a 6k-token context and a billed rerank. Accuracy is already perfect, so a
no-retrieval branch here is pure cost/latency saving (~83% on those queries) at
approximately zero quality risk.

**`scoped_fact` 4/5** — better than predicted. `sf3-05` (ACIT 2515 midterm/final
weights) retrieved the outline URL but not the evaluation chunk: the section-2
bug again, at ~40% incidence across the seven cases of this class in v1+v3.

## Verdict

| technique | verdict | why |
|---|---|---|
| **Self-RAG** | reject | nothing to fix (measured); incompatible with streaming; already offline as `--judge` |
| **Adaptive-RAG** | A-branch only | B/C routing already exists via `sub_queries` length at zero extra API cost; A is measured pure waste |
| **CRAG** | skeleton only | detect→switch is right; its standard payloads are not: query-rewrite retry is disproved above, web-search fallback would break the corpus-only invariant and citation precision 1.000 |

The payload that belongs in CRAG's slot is **entity-scoped re-retrieval**, and
one implementation covers both open failures: `sf3-05`'s entity is in the
question, `mh3-01`'s is in the first answer.

## Next, in order

1. **Entity-scoped retrieval behind a flag, no graph, no loop.** Gate on v3
   `scoped_fact` = 1.000 and v1 `ol-04`/`ol-11`, with v1/v2 not regressing.
2. **Coverage gate + one hop-2 re-retrieval.** This is where LangGraph earns its
   place — the first cycle in the pipeline. Gate on v3 `multi_hop`
   fact ≥ 0.90 / url ≥ 0.85, cost +10% max, p95 (measured on the VM) +1 s max.
3. **`route` field in the existing rewriter JSON schema** (A-branch). No extra
   API call — the same conditional-schema pattern as `MQ_BM25_KEYWORDS`.
   Gate on zero mis-routes across all three sets.

Traffic mix, needed to size step 3, is currently **unmeasurable**: LangSmith
retention holds only 12 root runs (2026-07-27 → 08-08), one of them chitchat.
The open README item "ship logs somewhere queryable" is the blocker. Those 12
questions do say something, though — they are person lookups ("who is michal?"),
topic searches ("which course can I learn how to use terraform?") and browsing
("ai related programs?"), a shape v1/v2 contain **zero** of. Two were promoted
into v3 as `exploratory`; both pass.

## Files

`v3_baseline.json` — the run above, re-scored through `eval/rescore.py` after
`multi_hop` groups gained their course-page URL alternatives (a course page is a
legitimate source for prerequisite facts; v1/v2 already treat the two as
alternatives, v3 initially did not, which scored `mh3-03` as a URL miss for an
answer that was correct and correctly cited).
