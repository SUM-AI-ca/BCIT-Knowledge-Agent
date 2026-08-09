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

## 4. What actually fixed the class — measured

Two flags, adopted together. `ENTITY_SCOPED_RETRIEVAL` adds a BM25 arm
restricted to the chunks of the course or program a sub-query names.
`RERANK_IDENTITY` sends that page identity in the Ranking API's `title` field.
Every other run below defaults to the previous behaviour; the flag-off run
reproduced the pre-change baseline on 21/21 cases.

### v1 (40 cases, `fu-04` excluded — see the noise note)

| | base | scoped only | identity only | **both** | **both, run 2** |
|---|---|---|---|---|---|
| URL hit | 0.966 | 0.966 | 0.966 | **0.991** | **0.991** |
| Fact recall | 0.940 | 0.953 | 0.953 | **0.991** | **0.991** |
| Citation precision | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| $/query | 0.00415 | 0.00419 | 0.00407 | **0.00405** | 0.00407 |
| Input tokens (generation only) | 5,902 | 5,892 | 5,663 | **5,526** | 5,561 |

| case | base | scoped | identity | both |
|---|---|---|---|---|
| `ol-04` COMP 1510 final-exam weight | 0.000 | 0.000 | 0.000 | **1.000** |
| `ol-12` COMP 3522 passing condition | 0.500 | 0.500 | 0.500 | **1.000** |
| `mp-01` CST entrance requirements | 0.500 | 1.000 | 1.000 | 1.000 |

**The two flags interact.** Each one alone fixes exactly the same single case
(`mp-01`, a program page that was already being retrieved) and nothing more.
`ol-04` and `ol-12` need both, and the mechanism says why:

- Without the scoped arm, COMP 1510's outline never enters the pool at all —
  the final context is ten *other* courses' outlines.
- With it but without identity, the scoped chunks enter the pool (12 of them)
  and the ranker still drops every one: it scored COMP 4870's and COMP 7402's
  evaluation tables 0.859 / 0.709 against 0.258 for ACIT 2515's own chunks on
  the equivalent v3 case. Those tables *are* what the question describes.
  Only the course name separates them, and the ranker had never been shown it.
- With both, the asked-about course lands at context position 1 and the
  context *concentrates*: sources 10 → 8 on `ol-04`, 10 → 6 on `ol-12`.

That concentration is why the adopted config is also the cheapest measured:
identity stops the context budget being spent on lookalikes, so input tokens
fall 6.4% and cost 2.4% while quality rises.

### v2 (25 cases) — no regression

| | base | scoped | both |
|---|---|---|---|
| URL hit / fact recall | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Input tokens | 5,547 | 5,628 | **5,389** |

25/25 cases scored identically to baseline in both configs. v2 has no headroom
on the current models, so this set only had to show the scoped arm does not
displace correct results — it injects 4.6 candidates per case and changes
nothing.

### v3 (21 cases)

| | base | stopword-df | both |
|---|---|---|---|
| URL hit | 0.833 | 0.818 | 0.833 |
| Fact recall | 0.844 | 0.833 | **0.911** |
| Avoidance | 1.000 | 1.000 | 1.000 |
| $/query | 0.00404 | 0.00403 | **0.00392** |
| Input tokens | 5,989 | 5,828 | **5,502** |

One case changed: `sf3-05` 0.000 → 1.000. `scoped_fact` as a category goes
0.800 → 1.000. `multi_hop` is untouched at 0.600 / 0.733 — as expected, it is
a different failure and needs the second retrieval round, not this.

> **Corrected 2026-08 (post-deploy audit).** This table originally read
> 0.818/0.833/0.905 across the row. Those figures predate the final
> `eval/rescore.py` pass — the one recorded under *Files* below, which gave the
> `multi_hop` groups their course-page URL alternatives — and the "base" column
> had additionally been filled from `v3_l2.json` rather than `v3_base.json`. The
> numbers above are what the archived runs hold today and what a fresh
> `rescore.py` reproduces (`0 moved`, URL self-check exact). The finding is
> unchanged: one case flips, `sf3-05`, worth +0.067 on a 15-case fact
> denominator. §3's baseline table was always correct.

### Adopted

`ENTITY_SCOPED_RETRIEVAL=true` + `RERANK_IDENTITY=true`, as a pair. Quality up
on every set that had headroom, no regression on any, citation precision 1.000
throughout, and cheaper on all three. Both remain env-overridable.

### Rejected

- **`BM25_STOPWORD_DF=0.35`** — 0 cases changed on any set. The offline effect
  is real (leave-one-out names `bcit` df 55.9%, `a` 60.1%, `with` 45.1%,
  `courses` 41.8% as pollutants; the `mh3-02` target moves from BM25 rank 161
  to 22, i.e. outside vs inside `RETRIEVAL_BM25_K=23`) but the pooled rerank
  re-orders the candidates afterwards, so a better candidate set did not become
  a better context. The flag stays for the fan-in query shape, where the
  candidate set is the binding constraint.
- **`RERANKER_TOP_K=14`** — 0 cases, +2% cost.
- **Either adopted flag on its own** — see the interaction above.
- **Applying the stopword filter inside the scoped arm** — regressed `sf3-05`
  from 1.000 to 0.000. Within one document the "generic" words are the
  discriminating ones, and the augmented text repeats the course code on every
  chunk, so filtering made all 13 chunks tie on the entity's own name.

## 5. Measurement notes that cost time

- **`fu-04` is a coin flip, not a signal.** It scored 1.000 / 0.000 / 1.000 /
  0.000 across four runs *independent of config*: the rewriter resolves "the
  newest residence" to "Tall Timber Student Housing" about half the time, and
  retrieves the 1978 Maquinna residence when it does not. It is excluded from
  the v1 aggregate above and is the only case that moves between the two
  identical `both` runs. `mp-05` misses the same Tall Timber content in every
  configuration — one corpus-side weak spot, two cases.
- **Single-case A/B is not usable here.** The rewriter emits different
  sub-queries for the same question across runs (`'ACIT 2515 midterm exam
  percentage weight'` vs `'…grade percentage'`), which changes retrieval. Full
  sets are stable: two identical v3 runs matched on 21/21, and the two `both`
  runs on v1 matched on all 39 stable cases.
- **A 429 used to delete a case from the run**, silently changing which cases
  the aggregate averages over; in one sweep 5 of 21 cases errored, each in a
  different config, and every apparent delta was that artifact.
  `run_eval.py --sleep/--retries` now paces and retries instead.
- **Three golden-set paraphrase gaps surfaced while comparing configs**
  (`ol-07` "optional", `ol2-01` "does not require any prerequisites",
  and the ordinal/hyphen/Unicode matcher bugs in §1). Each was verified by
  reading the answer, then applied to *all* arms via `eval/rescore.py` — a
  golden-set edit that lands on one arm of a comparison is how a config gets
  credited for a scoring change.

## 6. What this changes about the original question

The `scoped_fact` class is closed without a graph, a loop, or an extra LLM
call. That shrinks what a corrective/adaptive layer would be for:

| failure class | status |
|---|---|
| boilerplate section of a named entity (`ol-04`, `ol-12`, `mp-01`, `sf3-05`) | **fixed here** |
| fan-in breadth (`mh3-02`) | open; needs candidate-set retention, not a hop |
| true 2-hop (`mh3-01`, `mh3-04`) | open; the only case for a cycle |
| chit-chat / out-of-scope cost | open; the A-branch, worth ~83% on those queries |
| generator dishonesty | does not exist (avoidance 1.000, citation 1.000) |

So the LangGraph case now rests on **two cases** in v3's `multi_hop`, not on
the retrieval quality story.

> **Measured afterwards (2026-08, `../202608_graph_controller/`): the two cases
> were real, and they were not the point.** The controller fixed `mh3-01` and
> `mh3-04`'s URLs as predicted, and it also took out_of_scope avoidance from
> 0.750 to 1.000 and citation precision from 0.975 to 1.000 — neither of which
> this table anticipated, because both failures only surface under phrasings
> none of these sets contain. The `chit-chat / out-of-scope cost` row turned
> out to be a *quality* row, not a cost row. That is a much smaller mandate than it looked
like before this round, and it should be sized against real traffic — which
remains unmeasurable (12 root runs in LangSmith retention).

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

1. ~~Entity-scoped retrieval behind a flag, no graph, no loop.~~ **DONE,
   adopted and deployed** (§4 and *Shipped* below), together with reranker
   identity, which the gate did not anticipate needing. v1 0.966/0.940 →
   0.991/0.991 (×2), v2 unchanged at 1.000/1.000, v3 fact 0.844 → 0.911,
   cheaper on all three.
2. ~~**Fan-in retention**~~ — **built and rejected**; see
   `../202608_person_lookup/REPORT.md` §5. The premise stated here, that the
   pooled rerank "spreads the context budget across unrelated sources", was
   already false when this was written: §4's own adoptions made the context
   fully on-entity, and measurement shows all ten context chunks mention the
   entity for `mh3-02` itself. `FANIN_RETAIN` fired 0 times in 12 runs because
   its quota is satisfied by the very chunks it was meant to evict — they name
   the entity in the wrong *relation*. A working retry must be relation-aware.
   `BM25_STOPWORD_DF` remains implemented and rejected for the candidate-set
   half of it.
3. ~~**Coverage gate + one hop-2 re-retrieval**~~ — **DONE, adopted and
   deployed** as an LLM controller graph; see
   `../202608_graph_controller/REPORT.md`. `mh3-01` 0.33/0.50 -> **1.00/1.00**
   in two hops, `mh3-04` recovers both prerequisite URLs, v3 `multi_hop` url
   0.600 -> 0.933. Every gate below was met except cost.

   Two things this analysis got wrong, both worth keeping:

   - **It sized the mandate at two cases, and the two cases were not where
     most of the value was.** The *route* half — which this section treated as
     a separate, lower-priority cost optimization (step 4) — closed an
     out-of-scope leak worth avoidance 0.750 -> 1.000 and cut retrieval on 25%
     of turns. That leak was invisible to every set here because all of them
     phrase out-of-scope questions tidily; it only appears under
     `"how do i center a div in css"`.
   - **The guard proposed here ("the in-process entity index answers 'does
     this exist at all' for free") is not the guard that was needed.** With an
     LLM gate the risk is not a missing entity, it is a controller that keeps
     asking for something vague, or one that answers a follow-up from
     conversation history and drops the citation. Both needed deterministic
     backstops in code (`graph.is_concrete`, and the has-history rule); two
     rounds of prompt revision got the second from five cases to two, not to
     zero. A prompt states a preference; only code states a requirement.

   Original text of this item, for the record: *Still the
   only place a cycle is warranted, and now the whole case for LangGraph:
   two cases.* Gate on v3 `multi_hop` fact ≥ 0.90 / url ≥ 0.85, `unanswerable`
   holding at 1.000 (the loop must never fire on a question the corpus cannot
   answer — the in-process entity index answers "does this exist at all" for
   free), cost +10% max, p95 measured **on the VM** +1 s max.
4. ~~**`route` field in the existing rewriter JSON schema**~~ (A-branch) —
   **DONE in substance, differently in form.** Routing shipped as the
   controller's first call (`GRAPH_ROUTER_MODE=node`), not as a rewriter field,
   because the same node also serves the coverage gate and one contract for
   both was simpler than two. The gate this item asked for was met exactly:
   **zero mis-routes across all four sets** (v1 40, v2 25, v3 24, rough 16).

   The cheap form proposed here — folding the route onto the rewriter's
   existing JSON call — was costed out afterwards and **rejected**. It saves
   one 815-token controller call ($0.00035) on retrieve turns but makes
   `direct`/`refuse` turns pay for a rewriter call they currently skip
   ($0.00088), nets 0-6% depending on the set, and would move routing to the
   model tier that measured *worse* at it. The "no extra API call" framing was
   right about the call and wrong about the cost.

Two findings that fall out of this round and belong to neither step:

- **Follow-up referent resolution is ~50/50 on `fu-04`.** The rewriter turns
  "the newest residence" into "Tall Timber Student Housing" in half the runs
  and into a generic phrase in the other half, which retrieves the 1978
  Maquinna residence instead. `mp-05` misses the same content in every
  configuration. One corpus/rewriter weak spot, two cases, config-independent.
- **The substring fact matcher under-counts paraphrase often enough to matter
  in a config comparison.** Three gaps surfaced in this round alone. Consider
  running `--judge` alongside the substring metric for adoption decisions.

Traffic mix, needed to size step 4, is currently **unmeasurable**: LangSmith
retention holds only 12 root runs (2026-07-27 → 08-08), one of them chitchat.
The open README item "ship logs somewhere queryable" is the blocker. Those 12
questions do say something, though — they are person lookups ("who is michal?"),
topic searches ("which course can I learn how to use terraform?") and browsing
("ai related programs?"), a shape v1/v2 contain **zero** of. Two were promoted
into v3 as `exploratory`; both pass.

## Shipped — production verification (2026-08-08)

Deployed to `bcit-rag-vm` and verified live at https://bcitai.ca. The startup
log confirms which code is actually serving:

```
Loaded 100,515 documents
Entity index: 7,623 course codes, 498 programs
Reranker loaded: semantic-ranker-default-004 (identity titles)
```

Four questions were put to production, chosen because the pre-August
configuration answers each one wrongly — a restart on old code cannot fake them:

| case | production answer |
|---|---|
| `ol-04` COMP 1510 final-exam weight | "worth 40% of the final grade" (previously: not in the documents) |
| `ol-12` COMP 3522 passing condition | 50%, **and** a weighted average of ≥50% across midterm and final (previously: second condition dropped) |
| `sf3-05` ACIT 2515 exam weights | midterm 23%, final 34% (previously: not in the documents) |
| follow-up "How many credits is it?" | resolved to COMP 3522, 5.0 credits — session memory intact |

SSE streaming re-checked on the deployed build (`session` → 5×`delta` → `done`);
both request paths share `_prepare_turn`/`_finalize_turn`, so this also covers
the blocking `/chat` the eval harness uses.

Rollback needs no deploy: `ENTITY_SCOPED_RETRIEVAL=false RERANK_IDENTITY=false`
in the VM's `.env` plus a restart returns the pre-August behaviour, and the
flag-off run in §4 is the evidence that it returns to it exactly.

The deploy itself took four attempts, each failing on a *different* documented
gotcha (runbook copied only `server.py`; a wrapped one-liner made bash try to
execute `config.py`; `gcloud auth login` left the active project pointing away
from the VM's; a `read` prompt killed the script under `set -e` when stdin was
not a terminal). All four are now handled by `backend/deploy.sh`, which
pre-checks VM reachability before uploading anything.

## Files

`v1_{base,l1,id,l1id,l1id_b}.json`, `v2_{base,l1,l1id}.json`,
`v3_{base,l2,l1l2k,l1id}.json` — the runs in §4. `l1` = entity-scoped only,
`id` = reranker identity only, `l1id` = both (the adopted config), `l1id_b` =
its reproduction. All re-scored through `eval/rescore.py` so every run is
measured against the same golden sets.

`v3_baseline.json` — the §3 baseline, re-scored through `eval/rescore.py` after
`multi_hop` groups gained their course-page URL alternatives (a course page is a
legitimate source for prerequisite facts; v1/v2 already treat the two as
alternatives, v3 initially did not, which scored `mh3-03` as a URL miss for an
answer that was correct and correctly cited).
