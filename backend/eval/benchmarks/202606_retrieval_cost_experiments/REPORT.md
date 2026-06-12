# Round 2 — Retrieval Quality + Cost Experiments (2026-06)

Follow-up to `202606_model_comparison` (mixed models, $0.0126 → $0.0039/query).
Goal: push cost below $0.0030/query while raising retrieval quality, with the
Ranking API's fixed $0.001/query (the largest single cost item, ~26%) as the
primary target and the documented program-page section-flooding weakness as
the primary quality target.

Every experiment is a config flag that defaults to the previous behavior;
adoption required passing per-experiment gates against a fresh baseline.

## Method

- Harness: `eval/run_eval.py`, 40-case golden set, env set before launch.
- Fresh baseline ×2 re-measured the noise band: url_hit ±0.025 between
  identical runs (one flaky case, `ol-11`, flips; `ol-04` fails in every run
  of every config — golden-set investigation candidate). p95 latency on 40
  cases is the 2nd-slowest case and swings up to +28% between identical runs.
- The second baseline run executed the new flag-gated code with all flags
  off and matched run 1 within noise — flag-off behavioral identity check.
- `cache_read_tokens_mean = 0` in every run: Gemini implicit caching never
  fires for this workload (static prefix ~800 tokens < 1,024 minimum), so
  prompt-cache engineering was dropped as a lever.
- Ranking API billing verified: $1/1k queries, 1 query = up to 100 records —
  pool size changes below 100 records do not change rerank cost.

## Results

| label | hit | recall | cite | $/query | p95 s | notes |
|---|---|---|---|---|---|---|
| exp0_baseline | 0.963 | 0.877 | 1.00 | 0.00384 | 5.07 | fresh baseline (prod config) |
| exp0_baseline_b | 0.938 | 0.865 | 1.00 | 0.00384 | 6.50 | rerun = noise band + flag-off identity |
| exp1_pool56 | 0.963 | 0.846 | 1.00 | 0.00383 | 4.35 | REJECT |
| exp2_rw25flash | 0.929 | 0.902 | 1.00 | 0.00296 | 12.44 | REJECT (confirmed) |
| exp2_rw25flash_b | 0.929 | 0.877 | 1.00 | 0.00296 | 7.71 | confirm rerun — same hit |
| exp2_rw25lite | 0.917 | 0.831 | 1.00 | 0.00279 | 5.87 | REJECT |
| **exp3_bm25aug** | **0.975** | **0.925** | 1.00 | 0.00382 | 6.57 | **ADOPT** |
| **exp4_skip60_aug** | **0.975** | **0.931** | 1.00 | **0.00315** | 6.39 | **ADOPT @0.6** (60% skip) |
| exp4_skip70_aug | 0.950 | 0.875 | 1.00 | 0.00320 | 4.80 | noise dip; fewer skips |
| exp4_skip80_aug | 0.975 | 0.925 | 1.00 | 0.00324 | 4.58 | quality equal, less saving |
| exp5_kw_aug | 0.967 | 0.879 | 1.00 | 0.00443 | 5.22 | REJECT |
| exp5_hyde_extra_aug | 0.975 | 0.912 | 1.00 | 0.00430 | 5.54 | REJECT |
| exp5_hyde_replace_aug | 0.975 | 0.925 | 1.00 | 0.00428 | 6.39 | REJECT |
| exp6_alpha40 / 56 | 0.954 / 0.854 | 0.873 / 0.827 | 1.00 | ~0.00385 | – | keep alpha 0.48 |
| exp6_rrf40 / 80 | 0.963 / 0.963 | 0.835 / 0.877 | 0.99 / 1.00 | ~0.00385 | – | keep rrf_k 60 |
| exp6_mmr80 / 95 | 0.963 / 0.963 | 0.877 / 0.852 | 1.00 | ~0.00384 | – | keep lambda 0.87 |
| **exp7_skipsimple** | **0.963** | **0.902** | 1.00 | **0.00360** | 5.05 | **ADOPT** (25% rewrite skip) |
| exp9_combined | 0.950 | 0.906 | 1.00 | 0.00292 | 5.78 | full stack, run 1 |
| exp9_combined_b | 0.950 | 0.894 | 1.00 | 0.00293 | 4.55 | full stack, run 2 |
| **exp9_combined_c** | **0.975** | **0.919** | 1.00 | **0.00305** | 6.84 | **SHIPPED** + guard, run 1 |
| **exp9_combined_d** | **0.975** | **0.919** | 1.00 | **0.00304** | 4.91 | **SHIPPED** + guard, run 2 |

Per-category highlights (url_hit / fact_recall):

| label | outline | multipart | follow_up |
|---|---|---|---|
| exp0_baseline | 0.917/0.875 | 0.917/0.819 | 1.000/0.833 |
| exp3_bm25aug | 0.917/0.875 | **1.000/0.917** | 1.000/1.000 |
| exp4_skip60_aug | 0.917/0.875 | **1.000/0.958** | 1.000/1.000 |

## Adopted

1. **`BM25_INDEX_AUG=true` (E3)** — fit the runtime BM25 vectorizer on
   `title + category + filename_keywords + URL-slug + page_content` while
   serving the ORIGINAL documents (page_content untouched: neighbor-index md5
   keys, dedup keys, display all unchanged). No re-embed, no pickle rebuild,
   no DB change. This closes the documented section-flooding weakness: deep
   chunks of the 529 program pages now carry their page identity in the
   keyword index. Multipart hit 0.917 → 1.000 (the failing CST
   entrance-requirement cases flip), recall +0.048 overall, outline
   untouched (course-code matching not diluted).
2. **`RERANK_SKIP_CONSENSUS=0.6` (E4)** — skip the pooled/single Ranking API
   call when ≥60% of the fusion-ordered top slice was surfaced by BOTH
   retrieval arms (or 2+ sub-queries). With aug-BM25 the arms agree often:
   57-60% of queries skip, rerank cost $0.001 → ~$0.0004/query, quality flat
   (0.975/0.931 at threshold 0.6 — the best run of the round). Skipped turns
   also shave the rerank round-trip from latency. Guard: a turn that skipped
   the rewriter never also skips the rerank (defense in depth — the golden
   set showed no double-skip regression, but raw production queries keep at
   least one semantic stage this way at ~0.1% query cost).
3. **`REWRITE_SKIP_SIMPLE=true` (E7)** — first-turn, short, single-clause
   questions (<12 words, no separators) bypass the rewrite/decompose LLM
   call: the rewriter returns them unchanged, so the call is pure cost.
   Exactly the predicted 25% of the golden set skips; follow-ups and
   multipart questions never skip (history/separator guards); −$0.00024
   mean, −0.3s p50. Known limitation: bare acronyms on skipped turns are not
   expanded (BM25 still matches course codes exactly).

## Rejected (with evidence)

- **E1 bigger pool (40→56)**: outline recall 0.875 → 0.708 — extra
  candidates are sibling-program chunks that dilute the pooled rerank.
  Multipart gain +0.042 < +0.05 gate.
- **E2 cheaper rewriter**: 2.5-flash hit 0.929 twice (not noise), follow_up
  1.000 → 0.833 both runs; 2.5-flash-lite multipart 0.778. Confirms round 1:
  sub-query/rewrite quality drives retrieval; the 3.5-flash rewriter is
  load-bearing. The −23% cost is not worth −0.034 hit.
- **E5 keywords / HyDE**: with aug-BM25 already disambiguating programs, the
  extra rewriter fields add no quality (kw: multipart recall 0.917 → 0.778
  from turn-level keyword cross-pollution; hyde_replace: byte-equal quality)
  while their output tokens bill at the 3.5-flash $9/M rate (+$0.0005).
- **E6 fusion sweeps**: alpha 0.40/0.56, rrf_k 40/80, mmr 0.80/0.95 all ≤
  center; alpha 0.56 collapses to 0.854 (BM25's 0.52 weight is
  load-bearing). Current center (0.48 / 60 / 0.87) confirmed optimal.

Carried forward from round 1: RERANKER_TOP_K=13, RERANK_SCORE_THRESHOLD=0.2,
thinking budget >0, pure 2.5-flash-lite — all still rejected.

## V2 messy-query eval — E7 adoption REVERSED

The original golden set is clean English prose, which the round-1 report
flagged as under-measuring the rewrite-skip risk. A second, disjoint golden
set (`eval/golden_set_v2.jsonl`, 25 cases, every URL corpus-verified) was
built from sources the v1 set never touches (CIT, BMET, BSN Nursing, EE
degree, transfer credit, refunds, ACIT outlines) plus a new **messy**
category: acronyms ("cit entrance reqs?"), typos ("biomedicl enginering"),
informal phrasing, a Korean-language question, and a cross-program
comparison.

| config (×2 each) | hit | recall | $/query | messy hit/recall |
|---|---|---|---|---|
| stack with REWRITE_SKIP_SIMPLE=true | 0.900 / 0.900 | 0.837 / 0.817 | 0.00322 | 0.833/0.583 · 0.833/0.500 |
| stack with REWRITE_SKIP_SIMPLE=false | **0.940 / 0.940** | **0.937 / 0.937** | 0.00345 | **1.000/1.000 · 1.000/1.000** |

Every other category was byte-identical between the configs; the messy
category was the entire difference. Skipping the rewrite is safe on clean
English questions (outline/program_admission stayed 1.000/1.000 while
skipped) but collapses on raw real-world queries, where the rewriter
normalizes acronyms, fixes typos, and translates. The rewriter also raises
the rerank-skip rate (0.36 → 0.44 — rewritten queries produce stronger
fusion consensus), refunding part of its own cost: the true saving from
skip-on was only ~$0.0002/query. **E7 default reverted to false**; the code
path stays for cost-pressure scenarios with clean traffic.

## Production recommendation (adopted)

Two defaults flipped in `config.py` (env overrides remain as rollback):
`BM25_INDEX_AUG=true`, `RERANK_SKIP_CONSENSUS=0.6` (with the double-skip
guard; `REWRITE_SKIP_SIMPLE` stays false after the v2 eval above).

Measured for this exact stack: **v1 set hit 0.975 / recall 0.931 /
$0.00315 (exp4_skip60_aug)**; **v2 set hit 0.940 / recall 0.937 / $0.00345
(twice, identical)**. Vs the round-1 production config: hit +0.012, recall
+0.054, multipart 1.000, cost −18% ($3.87 → $3.15 per 1k on the v1 mix).

Notes: exp9 a/b's outline dip traced to `ol-11` (the known flaky case,
failed with NO skips active, consensus 0.1) — noise, not a double-skip
regression. The v2 set is harder by construction (its follow_up category
holds at 0.750 in every config — corpus-side, config-independent).

## Dense-side page identity (Tier-2 follow-up) — ADOPTED, and it retired the rerank-skip

E3 gave the BM25 arm page identity; the dense arm still embedded raw chunk
text, so deep chunks of the 529 program pages stayed near-identical vectors
across programs. Follow-up experiment: re-embed the SAME chunks with an
identity prefix on the *embedded* text only (`build_pgvector.py
EMBED_IDENTITY_PREFIX` + `REUSE_PICKLE`, new blue-green collection
`bcit_docs_202606da`, ~$4 / 85 min). Stored text verified byte-identical
corpus-wide (md5 multiset diff = 0 across 100,515 rows); vectors verified
changed.

Dense-only probe (similarity top-20, expected-page rank/hits): improved on
5/5 identity queries — e.g. BMET entrance requirements went from 6/20
own-page chunks to **20/20**, nursing chemistry from rank 2 with 1 hit to
rank 1 with 8.

Full-pipeline evals (all reproduced ×2 with identical numbers):

| config | v1 hit/recall | v2 hit/recall | $/query |
|---|---|---|---|
| prod (identity-blind vectors, skip 0.6) | 0.975 / 0.931 | 0.940 / 0.937 | 0.00315 / 0.00346 |
| dense-id, skip 0.6 | 0.967 / 0.906 | 1.000 / 0.950 | 0.00309 / 0.00329 |
| dense-id, skip 0.9 | 0.950 / 0.881 | 1.000 / 0.950 | mid — dominated |
| **dense-id, skip OFF (adopted)** | **0.975 / 0.925** | **1.000 / 0.950** | 0.00385 / 0.00392 |

Key interaction finding: identity vectors concentrate dense results, which
**inflates arm consensus** (skip rate 60% → 65%+) — the rerank-skip then
fires on multi-page questions where pure fusion order loses secondary pages
(every v1 regression under skip 0.6 was a `rerank_skipped` case; mp-05
skipped at consensus 0.8 that used to rerank at 0.3). The 0.6 threshold was
calibrated for identity-blind vectors. With the reranker always on, v1
returns to parity and v2 reaches **hit 1.000 with the long-stuck follow_up
category fixed (0.750 → 1.000)** and messy still perfect.

Adopted defaults: `PG_COLLECTION=bcit_docs_202606da`,
`RERANK_SKIP_CONSENSUS=0.0` (flag retained for cost-pressure rollback;
`DOCUMENTS_PICKLE` unchanged — chunks byte-identical, documented exception
to the flip-together rule). Net vs pre-experiment prod: v2 hit +0.060,
recall +0.013, v1 flat, cost +~$0.0006 (the E4 saving was real but only for
identity-blind vectors — better retrieval made the reranker worth paying
for again on the cases that matter).

## Future work

- E8 (local cross-encoder) is NOT warranted: the consensus skip already
  removes 57-60% of Ranking API calls; remaining spend ~$0.0004/query.
- `/chat` request timeout (carried from round 1, still open).
- `ol-04` fails in every configuration — likely a golden-set or corpus gap,
  not a retrieval bug; investigate separately.
- BM25 aug at corpus-build time (bake augmented text into the pickle) would
  shave server startup BM25Okapi fit; runtime fit chosen for zero-rebuild.

## Files

`exp0_baseline(.b)` `exp1_pool56` `exp2_rw25flash(_b)` `exp2_rw25lite`
`exp3_bm25aug` `exp4_skip{60,70,80}_aug` `exp5_{kw,hyde_extra,hyde_replace}_aug`
`exp6_{alpha40,alpha56,rrf40,rrf80,mmr80,mmr95}` `exp7_skipsimple`
`exp9_combined(_b,_c,_d)` — raw result JSONs alongside this report.
