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

v2 looked like a problem at first: every candidate run lost exactly one case
while both baseline runs were clean. A different case each time, and each
traced to a mechanism upstream of both flags:

| run | case | cause |
|---|---|---|
| sig+person r1 | `mp2-04` | answered an English question **in French**; the missed fact is `physics` ("physique") |
| sig+person r2 | `ms2-05` | Korean answer; **identical sub-query, `n_scoped_candidates=0`, `n_candidates=25` as baseline** — same retrieval input, different generation |
| person only | `fu2-02` | the rewriter failed to resolve "What are its prerequisites?" to ACIT 2515, so the *course-code* arm lost its entity (`n_scoped` 4 → 0) |

None involves the person index firing. Across all three regression sets the
person arm fired on **0 of 86** questions — it is a literal no-op unless a
question names an indexed instructor.

### The v2 tiebreak — v2 is not a stable 1.000 set

The candidate looked worse on v2 only because the baseline had been sampled
twice and both samples were clean. Run six times, the baseline wobbles too:

| | runs | recall | url | runs with an imperfect case |
|---|---|---|---|---|
| baseline | 6 | 1.000, 1.000, 1.000, 0.980, 0.960, 1.000 (mean **0.990**) | 1.000 ×6 | **2 / 6** |
| person only | 3 | 1.000, 0.960, 1.000 (mean **0.987**) | 0.960, 1.000, 1.000 | **2 / 3** |

The candidate's range sits inside the baseline's. Six different cases are
involved across the nine runs — `ol2-01`, `ol2-03`, `ms2-05`, `fu2-02`,
`pa2-06`, and `ms2-05` again — and `ms2-05` fails on **both** sides, which is
the direct proof that the instability is not config-borne. Two further checks
close it: the person index matched **0 times** across every standalone question
and every sub-query of all three person runs, and with zero matches
`_detect_entities` returns the same list and `_scoped_candidates` takes the same
branch, so the two configurations are behaviourally identical on this set.

`pa2-06` was not a failure at all but a golden-set paraphrase gap: the answer
said "does **not reduce** your program tuition" against an alternative list that
only had "not be reduced". Fixed in `golden_set_v2.jsonl` and re-scored across
**all ten** archived v2 runs (1 moved, 0 other case-deltas, URL self-check
exact), per the rule that a golden-set edit lands on every arm or none.

**This retires a claim the previous round made.** v2 was recorded as
"1.000/1.000, reproduced twice" and treated as saturated. On the current model
pair its true run-to-run band is **recall 0.960-1.000**, so a single v2 run
cannot gate anything at the 0.04 level.

## 7. Promoted into v3, and re-measured on the enlarged set

Three cases went into `golden_set_v3.jsonl` (21 → 24), chosen from the ten
teaching cases as the only ones where a **plain substring matcher still
separates a right answer from a wrong one** — for the other seven the baseline
names the correct course codes while attributing them to the reviewing role,
which `_fact_hit` scores as a hit. They cover three different flood sources:
`pl3-01` Victor Mendez (signature block), `pl3-02` Esti Jacobs (held-out name),
`pl3-03` Juan Azmitia (program-page coordinator listings).

Both arms were re-run on the 24-case set, twice each, because the archived v3
runs are scored on the 21-case version and are not comparable:

| run | url | recall | citation |
|---|---|---|---|
| base r1 / r2 | 0.6762 / 0.6762 | 0.7672 / 0.7593 | 1.000 / 1.000 |
| person r1 / r2 | **0.8667 / 0.8667** | **0.9148 / 0.9069** | 0.914¹ / 1.000 |

| new case | base r1 / r2 | person r1 / r2 |
|---|---|---|
| `pl3-01` Victor Mendez (5 courses) | 0.00 / 0.00 | 0.80 / 0.80 |
| `pl3-02` Esti Jacobs (3) | 0.00 / 0.00 | 1.00 / 1.00 |
| `pl3-03` Juan Azmitia (7) | 0.14 / 0.00 | 1.00 / 0.86 |

The **21 pre-existing cases score 0.8333 / 0.9111 in all four runs** — identical
across both configurations and both reproductions, and identical to the archived
production baseline. The set-level gain is entirely the new class.

¹ The one sub-1.000 citation score did not reproduce. Two cases caused it:
`os3-02`, a refusal that cited `https://www.bcit.ca` (the site root, not a
retrieved URL — and in the baseline run that answer cited *nothing*, so the case
was excluded from the mean entirely rather than scored), and `ex3-02`, where the
model rewrote outline URLs to an `aws.bcit.ca` subdomain that **does not exist
anywhere in the corpus**. Neither involved the person arm — `n_scoped_candidates`
was 0 in both configurations for both cases, and the runs differ only in the
rewriter's sub-query. The URL mangling is a real generation defect worth its own
issue; it is not a property of this change.

## 8. The deploy caught what the guard set had missed

Deployed, the held-out case answered perfectly — and **the question that opened
this round still failed in production**. The reason is the wording. Every case
in the dev and guard sets asks `What courses does <Full Name> teach at BCIT?`;
the user typed `what courses chi en teach?` — lowercase, partial name,
ungrammatical. The index is keyed on full names and detection needs a
capitalised run, so it depended on the rewriter completing "chi en" to "Chi En
Huang". Measured five times on that exact string, the rewriter completes it
**4 times in 5**; the fifth is the reported failure.

A guard set built from tidy phrasings of the right *class* is still an
overfitting instrument. Fixed by indexing unambiguous partial-name aliases
(first+last and first-two-words, 2 words minimum, only where the alias resolves
to one instructor and is not itself somebody's full name — 76 aliases over
1,291 instructors, 0 ambiguous, 0 clashes). Detection then fires **5/5** on the
user's exact string, and `lc-01` / `lc-02` were added to the guard set so the
phrasing itself stays measured.

Re-verified with aliases on: guard F1 **0.972** (was 0.968), all three
adversarial program-head cases still pass, no-teach still 1.000, and v3
unchanged at url 0.8667 / recall 0.9148 with **no case worse**.

## 9. Verdict

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

1. ~~One more v2 pair to settle §6's wobble.~~ **Done** — see the tiebreak
   above. v2's own band is recall 0.960-1.000 over six baseline runs and the
   candidate sits inside it.
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
