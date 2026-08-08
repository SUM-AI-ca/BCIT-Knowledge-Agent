# Model comparison — June 2026

Motivation: at gemini-3.5-flash prices ($1.50/M in, $9.00/M out) the optimized
pipeline still costs ≈ $0.0126/query — too expensive for sustained production
use. This benchmark compares generation models on the 40-case golden set with
identical settings (temperature 0.05 generation / 0.0 rewriter,
thinking_budget 0, NEIGHBOR_RADIUS 2, RERANKER_TOP_K 10, chunks context).


> **Retro-note (2026-08).** The `recall` column below was measured with a
> key-fact matcher since found to have three bugs (ordinal dates, hyphenated
> compounds, and a Unicode `\w` that read a Korean particle as a word
> continuation); it under-reports by ~0.036. `eval/rescore.py` re-scores these
> archived runs offline — re-scoring all 49 moved every run up by
> +0.025…+0.070 and inverted 0 of 816 pairwise comparisons, so the model
> ranking and the ADOPT decision here are unchanged. `hit` and `cite` are
> unaffected. The models compared here have also since been succeeded
> (`3.1-flash-lite` → `3.5-flash-lite`, `3.5-flash` → `3.6-flash`); what
> carries over is the *shape* of the finding — a lite generator paired with a
> full-size rewriter — not the absolute numbers.

## Method

- `eval/run_eval.py --label model_<name>` with `GEMINI_MODEL=<model>`;
  the mixed run sets `REWRITER_MODEL=gemini-3.5-flash` and
  `GEMINI_MODEL=gemini-3.1-flash-lite` (both knobs env-overridable since
  this change).
- Cost/query = measured tokens × official per-M prices + $0.001 Ranking API
  (one pooled call/turn, $1.00/1k queries) + ~$0.00001 embedding
  ($0.15/M × ~60 tokens). Output prices include thinking tokens.
- Prices (paid tier, standard, 2026-06, ai.google.dev/gemini-api/docs/pricing):
  3.5-flash $1.50/$9.00 · 3.1-flash-lite $0.25/$1.50 · 2.5-flash $0.30/$2.50 ·
  2.5-flash-lite $0.10/$0.40.
- Run-to-run noise: identical configs measured hit-rate 0.938↔0.963 across
  runs (the rewriter's sub-query phrasing varies slightly even at temp 0, and
  retrieval follows). Treat deltas ≤ ~0.03 as noise.

## Results (40 cases each, zero truncations/fallbacks, citation precision 1.0)

| model                             | hit | recall | cite | in_tok | p95 | $/query | $/1k | vs 3.5-flash |
|---|---|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite-preview     | 0.917 | 0.802 | 1.000 | 5,925 | 3.6s | $0.0017 | $1.74 | −86% |
| gemini-3.1-flash-lite             | 0.929 | 0.881 | 1.000 | 5,815 | 4.1s | $0.0029 | $2.94 | −77% |
| gemini-2.5-flash                  | 0.929 | 0.865 | 1.000 | 5,891 | 6.9s | $0.0034 | $3.41 | −73% |
| MIX: 3.5 rewriter + 3.1-lite gen  | 0.963 | 0.890 | 1.000 | 5,896 | 4.8s | $0.0039 | $3.87 | −69% |
| gemini-3.5-flash                  | 0.963 | 0.885 | 1.000 | 5,884 | 5.2s | $0.0126 | $12.61 | — |

## Findings

1. **The hit-rate gap of pure 3.1-flash-lite came from the rewriter, not the
   generator.** Keeping the rewriter on 3.5-flash restored hit-rate to 0.963
   and produced the best fact recall of any run (0.890). Sub-query quality
   drives retrieval; generation quality only drives the final wording.
2. **2.5-flash-lite-preview fails the quality gate** (recall 0.802 < the
   0.848 pre-optimization baseline). 2.5-flash is dominated by 3.1-flash-lite
   on every axis.
3. At lite-class generation prices the **Ranking API's fixed $0.001/query is
   the largest single cost component** — future optimization would target
   rerank skipping (e.g., bypass rerank when RRF consensus is strong).
4. The rewriter premium costs ~$0.0010/query (3.5 vs 3.1-lite rewrite
   tokens) and buys +0.034 hit-rate — cheap insurance.

## Recommendation

`REWRITER_MODEL=gemini-3.5-flash` + `GEMINI_MODEL=gemini-3.1-flash-lite`
(the MIX row): quality indistinguishable-or-better vs 3.5-flash everywhere,
−69% cost ($12.61 → $3.87 per 1k queries), p95 latency also improves.
If cost pressure outweighs the last 0.03 of hit-rate, pure 3.1-flash-lite at
$2.94/1k also passes the original quality gates (0.848 recall / 0.892 hit).

## Files

Raw eval outputs for every run in this directory (copied from the gitignored
`eval/results/` scratch dir). The optimization-era baselines (`before*`,
`after`, `sweep_*`) are included for the full June 2026 story: full-doc
pipeline → chunk pipeline → model selection.

Note on what came before this benchmark: the original full-document pipeline
ran `gemini-3.1-pro`. That build was about getting retrieval and grounding
correct — there was no per-query cost instrumentation, no model comparison and
no token budget, so its cost was never measured and is not quoted anywhere.
Cost work starts with the chunk pipeline and this document.
