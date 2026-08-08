"""Regression tests for the key-fact matcher.  `python eval/test_matcher.py`

The matcher is the whole eval's ground truth, and it failed silently three
different ways at once (found 2026-08 by re-scoring 49 archived runs): every
"July 2nd" answer, every hyphenated "4.0-credit", and every non-English answer
whose language glues a particle onto the token. Each case below is one of those
bugs or one of the guards that must survive fixing them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _fact_hit, _normalize

HIT = [
    # --- must match ---
    ("july 2", "The deadline is July 2nd.", "ordinal date (pa-02, fu2-01, mp2-01)"),
    ("september 1", "The domestic deadline is September 1st.", "ordinal date (mp-01)"),
    ("august 1", "New rates take effect on August 1st.", "ordinal date (pa-09)"),
    ("july 2", "The deadline is July 2.", "plain date still matches"),
    ("4.0 credit", "COMP 2714 is a 4.0-credit course.", "hyphenated compound (ol-07)"),
    ("english studies 12", "english studies 12에서 73% 이상", "Korean particle (ms2-05)"),
    ("english studies 12", "English Studies 12(73%) 또는", "Korean, punctuation follows"),
    ("40", "the final exam is worth 40% of the grade", "percent sign is a boundary"),
    ("$6000", "the fee is $6,000 per term", "comma stripped"),
    ("75", "75 hours of instruction", "plain number"),
    ("45", "총 45시간 동안 진행됩니다", "Korean unit glued to a number"),
    ("no prerequisites", "There are no prerequisites for this course.", "phrase"),
]

MISS = [
    # --- must NOT match: these are the guards ---
    ("75", "the course runs 175 hours", "digit prefix (the original boundary rule)"),
    ("15", "COMP 1510 is a prerequisite", "15 inside 1510 — the real ol-11 miss"),
    ("90", "the fee is $1900", "digit suffix"),
    ("40", "COMP 4010 covers this", "embedded in a course number"),
    ("comp 2510", "COMP 25100 does not exist", "code prefix"),
    ("july 2", "The deadline is July 20.", "ordinal widening must not eat a longer number"),
    ("free", "freedom of information", "word prefix"),
]


def main():
    failures = []
    for alt, answer, why in HIT:
        if not _fact_hit([alt], _normalize(answer)):
            failures.append(f"  MISSED  {alt!r} in {answer!r}  ({why})")
    for alt, answer, why in MISS:
        if _fact_hit([alt], _normalize(answer)):
            failures.append(f"  MATCHED {alt!r} in {answer!r} — should not  ({why})")

    total = len(HIT) + len(MISS)
    if failures:
        print(f"FAIL — {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok — {total} matcher cases ({len(HIT)} hit, {len(MISS)} guard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
