# Person lookup / fan-in — "what courses does X teach?" (2026-08)

Opened by a production conversation. Asked `what courses chi en teach?`, the
system answered that Chi En Huang teaches COMP 3725 and, as Program Head,
*reviewed* COMP 3981 / 4989 / 4983. Asked again with a leading question ("I
heard that Chi En teaches AI courses"), it answered that he **instructs** those
three. The two answers contradict each other; the corpus says the second is
right, and the complete answer — 4 courses — was never produced by either turn.

## 1. The corpus fact

`### Instructor Details / Name |` names the instructor. The approval block at
the bottom of every outline names a Program Head, who is often a *different*
person, and sometimes the same one.

| | Chi En Huang's role |
|---|---|
| COMP 3725, 3981, 4983, 4989 | **instructor** |
| COMP 2522, 3940, 3948, 3975, 4800, 4870 | Program Head signature only |

Corpus-wide the second relation drowns the first:

```
chunks containing "Chi En Huang"                73
  Instructor Details                             4
  approval signature                            62      15.5 : 1
approval-block chunks corpus-wide            3,279      3.3% of the corpus
distinct Program-Head signers               ~2,136 chunks; top signer 81
```

## 2. It is a selection failure, not a retrieval-depth failure

Offline against the BM25 arm (no DB, no API), for the raw question the four
instructor chunks rank **9, 10, 11, 12** — inside `RETRIEVAL_BM25_K=23`. They
reach the candidate pool. What they lose is the `RERANKER_TOP_K=10` selection,
to the **15 approval chunks that also sit in the top 25**. Raising top-k admits
roughly four wrong-role chunks per right-role chunk at that ratio, and the
observed error was misattribution rather than omission, so depth is the wrong
lever. (`RERANKER_TOP_K=14` was already measured in the previous round: 0 cases,
+2% cost.)

## 3. What was tried

| flag | what it does |
|---|---|
| `SIGNATURE_DEMOTE` | strips the approval signatures from the text the BM25 index is fit on; documents served untouched, no re-embed (the `BM25_INDEX_AUG` contract) |
| `FANIN_RETAIN` | roadmap step 2 — guarantee N distinct sources whose text contains the literal the question names |
| `PERSON_SCOPED_RETRIEVAL` | index instructor names → the sources that name them **as the instructor**; scope a retrieval arm to those. 1,291 instructors, 0 malformed entries |

## 4. Guard set — built because the dev set cannot judge itself

`golden_set_people.jsonl` (6 cases) was written *after* seeing the failure, so
it is an overfitting instrument, not a verdict. `golden_set_people_guard.jsonl`
adds nine cases chosen to break the fixes:

- `ho-*` 4 **held-out** instructors, unused during development
- `nt-*` 2 people who sign outlines but instruct nothing — invention check
- `ph-*` 3 **adversarial**: "who is the Program Head that reviewed X?", which
  asks for exactly the text `SIGNATURE_DEMOTE` removes from the index

Answers are graded by extracting the set of courses the answer claims the
person **teaches** and comparing sets. Substring fact matching cannot score this
class: a wrong answer names the same course codes, in the wrong role.

### Results (15 cases)

| | base | sig | person | sig+person |
|---|---|---|---|---|
| F1 | 0.576 | 0.884 | **0.968** | **0.968** |
| recall | 0.439 | 0.856 | 0.949 | 0.949 |
| precision | 0.729 | 0.950 | 0.983 | 0.983 |
| exact sets (of 12) | 5 | 7 | 8 | 8 |
| **adversarial name_hit** (of 3) | 1.000 | 1.000 | 0.667¹ | 1.000 |
| named the instructor instead | 0 | 0 | 1¹ | 0 |
| ctx instructor chunks | 1.92 | 3.83 | **5.00** | **5.00** |
| ctx signature chunks | 7.17 | 2.17 | **2.00** | **2.00** |
| $/query | 0.00520 | 0.00600 | 0.00590 | 0.00510 |

¹ `ph-03` passed in 5 of the 6 arm-runs recorded here and failed once, in a
person-only arm — i.e. in a configuration that does not touch the signature
text at all. Treated as noise, not as an effect of either flag.

Held-out generalisation is the load-bearing result: `ho-01` and `ho-02` were
**complete baseline failures** (0.00, with zero instructor chunks in context)
and both are fixed. The no-teach cases never invented a course in any arm.

**The adversarial case did not break.** `SIGNATURE_DEMOTE` hides the approval
text from the *sparse index*, but those questions name a course code, which
triggers the course-scoped arm, and the stored document is served untouched —
so the text is still in the context when the chunk arrives. Designed for, now
measured.

## 5. `FANIN_RETAIN` — rejected, and its premise is wrong

**0 swaps in all 12 runs.** The retrieval counters are byte-identical between
`base`/`fanin` and between `sig`/`both`, so those arms are same-config
replicates.

The roadmap says the pooled rerank "spreads the context budget across
*unrelated* sources". Measured, that is false on this pipeline:

```
"Which BCIT courses require ACIT 1515...?"   ctx 10 chunks / 7 sources, 10 mention the key
"What courses does Victor Mendez teach?"     ctx 10 chunks / 10 sources, 10 mention the key
```

There are no unrelated sources left to displace — `RERANK_IDENTITY` and
entity-scoped retrieval already made the context fully on-entity in the August
round. The slots are taken by chunks naming the entity in the **wrong
relation**, which a rule keyed on "mentions the entity" cannot distinguish; the
quota is satisfied by exactly the chunks it was meant to evict. Retention has to
be relation-aware. Roadmap step 2 should be rewritten before anyone builds it.

Useful side effect: the two inert arms put the noise band on this metric at
**±0.14 aggregate F1 and up to 0.40 per case**, which is why the deterministic
`ctx_instr` / `ctx_sig` counters carry the argument above rather than F1 alone.

## 6. Regression — 86 cases the fixes were not designed for

| set | baseline (production) | person only | sig+person |
|---|---|---|---|
| v1 (n=39, `fu-04` excluded) | 0.9915 / 0.9915 | **0.9915 / 0.9915** | **0.9915 / 0.9915** |
| v3 (21) | 0.8333 / 0.9111 | **0.8333 / 0.9111** (21/21 identical) | **0.8333 / 0.9111** |
| v2 (25) | 1.0000 / 1.0000 (×2 runs) | 0.9600 / 1.0000 | 1.0000 / 0.9867 · 1.0000 / 0.9600 |

Citation precision 1.000 everywhere, 0 errors everywhere, cost within $0.00006.

v2 is the one set where every candidate run lost exactly one case while both
baseline runs were clean. A different case each time, and each traced to a
mechanism upstream of both flags:

| run | case | cause |
|---|---|---|
| sig+person r1 | `mp2-04` | answered an English question **in French**; the missed fact is `physics` ("physique") |
| sig+person r2 | `ms2-05` | Korean answer; **identical sub-query, `n_scoped_candidates=0`, `n_candidates=25` as baseline** — same retrieval input, different generation |
| person only | `fu2-02` | the rewriter failed to resolve "What are its prerequisites?" to ACIT 2515, so the *course-code* arm lost its entity (`n_scoped` 4 → 0) |

None involves the person index firing. Across all three regression sets the
person arm fired on **0 of 86** questions — it is a literal no-op unless a
question names an indexed instructor.

## 7. Verdict

**Adopt `PERSON_SCOPED_RETRIEVAL`. Do not adopt `SIGNATURE_DEMOTE` yet. Reject
`FANIN_RETAIN`.**

- The person index carries the whole effect on its own (0.576 → 0.968) and is
  the only change that fixes `pl-02`, where the flood source is program-page
  coordinator listings rather than approval signatures.
- `SIGNATURE_DEMOTE` adds **nothing** on top of it (0.968 → 0.968) while
  rewriting the sparse index for all 100,515 chunks. When two changes tie, the
  one whose blast radius is "questions naming a known instructor" beats the one
  whose blast radius is the whole corpus. It stays implemented and off.
- `FANIN_RETAIN` never fired; see §5.

### Before shipping

1. One more v2 pair (baseline + candidate) to settle §6's wobble. Every mover
   has an explanation, but the baseline is 2/2 clean and the candidate 3/3 not.
2. `pl-05` sits at 0.89 in both person arms against 1.00 in sig-only — one
   residual case, worth a look but not a blocker.
3. Promote 2-3 of these cases into `golden_set_v3.jsonl` so the class stays
   measured. v1/v2/v3 contain **zero** person lookups today, which is why 60+
   gated runs never caught this.

## Files

`people_{base,sig,fanin,both}.json` — §5, the 6-case dev set.
`people_g_{base,sig,person,sigperson}.json` and `_b` reruns — §4, dev+guard.
The `_b` person arms postdate the fix for hyphenated names ("Julia
Alards-Tomalin" truncated to "Julia Alards" and missed the index; `pl-04` scored
0.00 on the person-only arm for that reason alone).
`reg_{v1,v2,v3}_{person,sigperson}.json`, `reg_v2_baseline_b.json` — §6.
