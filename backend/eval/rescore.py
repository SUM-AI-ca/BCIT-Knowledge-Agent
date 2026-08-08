"""Re-score archived eval runs offline — no DB, no API, no re-generation.

Every result file stores `answer_excerpt` and `retrieved_urls` per case, which
is everything the scorer needs. So when the SCORER or the GOLDEN SET changes,
the 100+ archived runs can be re-measured on the new rules instead of being
stranded on the old ones (or re-run at ~$0.15 and 5 minutes each).

    python eval/rescore.py eval/benchmarks/*/*.json
    python eval/rescore.py eval/results/after.json --verbose
    python eval/rescore.py eval/results/after.json --write eval/results/after_rescored.json

Two caveats, both reported rather than hidden:

- `answer_excerpt` is capped at ANSWER_EXCERPT_CHARS. On a case that hit the
  cap, a fact "miss" may just be text that was cut off, so those cases are
  counted separately and never claimed as a scorer fix.
- URL metrics do not depend on the fact matcher at all. They are recomputed
  anyway as a self-check: if `url_hit_rate` moves, this tool is wrong, not the
  run. Any such case is printed as a MISMATCH.
"""

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from run_eval import (  # noqa: E402
    ANSWER_EXCERPT_CHARS,
    _cited_urls,
    _fact_hit,
    _groups,
    _norm_url,
    _normalize,
    aggregate,
    load_golden,
)


def rescore_case(golden, case):
    """Re-derive the scored fields from the stored answer + retrieved URLs."""
    answer = case.get("answer_excerpt") or ""
    normalized = _normalize(answer)
    retrieved = case.get("retrieved_urls")

    out = {"truncated": len(answer) >= ANSWER_EXCERPT_CHARS}

    fact_groups = _groups(golden.get("key_facts", []))
    if fact_groups:
        hits = [_fact_hit(g, normalized) for g in fact_groups]
        out["fact_recall"] = sum(hits) / len(hits)
        out["facts_missed"] = [g[0] for g, h in zip(fact_groups, hits) if not h]
    else:
        out["fact_recall"] = None
        out["facts_missed"] = []

    forbidden_groups = _groups(golden.get("must_not_contain", []))
    if forbidden_groups:
        hits = [_fact_hit(g, normalized) for g in forbidden_groups]
        out["avoidance"] = 1.0 - sum(hits) / len(hits)
        out["forbidden_present"] = [g[0] for g, h in zip(forbidden_groups, hits) if h]

    # URL side: recomputed only when the run stored the retrieved list.
    if retrieved is not None:
        url_groups = _groups(golden.get("expected_urls", []))
        if url_groups:
            hits = [any(_norm_url(a) in retrieved for a in g) for g in url_groups]
            out["url_hit_fraction"] = sum(hits) / len(hits)
            out["url_full_hit"] = all(hits)
            out["urls_missed"] = [g[0] for g, h in zip(url_groups, hits) if not h]
        cited = _cited_urls(answer)
        out["n_cited"] = len(cited)
        out["citation_precision"] = (
            len([u for u in cited if u in retrieved]) / len(cited) if cited else None
        )
    return out


# Fields the scorer owns; everything else in a case record is measurement
# (tokens, timings, cost) and is carried through untouched.
SCORED_FIELDS = (
    "fact_recall", "facts_missed",
    "avoidance", "forbidden_present",
    "url_hit_fraction", "url_full_hit", "urls_missed",
    "n_cited", "citation_precision",
)


def rescore_run(path, golden_dir, verbose=False):
    with open(path, "r", encoding="utf-8") as f:
        run = json.load(f)
    if "cases" not in run:
        return None

    golden_name = run.get("golden_set", "golden_set.jsonl")
    golden_path = Path(golden_dir) / golden_name
    if not golden_path.exists():
        print(f"  ! {path}: golden set {golden_name} not found, skipped")
        return None
    golden_by_id = {c["id"]: c for c in load_golden(golden_path)}

    before = json.loads(json.dumps(run["cases"]))
    changed, truncated_changed, mismatches, missing = [], [], [], 0

    for case in run["cases"]:
        golden = golden_by_id.get(case.get("id"))
        if golden is None or case.get("error"):
            missing += golden is None
            continue
        new = rescore_case(golden, case)
        old_facts = case.get("fact_recall")
        old_url = case.get("url_hit_fraction")

        for key in SCORED_FIELDS:
            if key in new:
                case[key] = new[key]

        if new.get("url_hit_fraction") is not None and old_url is not None:
            if abs(new["url_hit_fraction"] - old_url) > 1e-9:
                mismatches.append((case["id"], old_url, new["url_hit_fraction"]))
        if new["fact_recall"] is not None and old_facts is not None:
            delta = new["fact_recall"] - old_facts
            if abs(delta) > 1e-9:
                record = (case["id"], case.get("category"), old_facts, new["fact_recall"])
                (truncated_changed if new["truncated"] else changed).append(record)

    result = {
        "label": run.get("label"),
        "path": str(path),
        "golden_set": golden_name,
        "before": aggregate(before),
        "after": aggregate(run["cases"]),
        "changed": changed,
        "truncated_changed": truncated_changed,
        "mismatches": mismatches,
        "missing_from_golden": missing,
        "run": run,
    }
    if verbose:
        for cid, cat, old, new in changed:
            print(f"    {cid:9s} {cat or '':18s} fact_recall {old:.3f} -> {new:.3f}")
        for cid, cat, old, new in truncated_changed:
            print(f"    {cid:9s} {cat or '':18s} fact_recall {old:.3f} -> {new:.3f}  (TRUNCATED excerpt)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="archived result JSON files")
    parser.add_argument("--golden-dir", default=str(EVAL_DIR))
    parser.add_argument("--verbose", action="store_true", help="per-case deltas")
    parser.add_argument("--write", help="write the rescored run (single input only)")
    args = parser.parse_args()

    rows, all_mismatches = [], []
    for path in args.results:
        res = rescore_run(path, args.golden_dir, verbose=args.verbose)
        if res is None:
            continue
        rows.append(res)
        all_mismatches.extend((Path(path).name, *m) for m in res["mismatches"])

    if args.write:
        if len(rows) != 1:
            parser.error("--write takes exactly one input file")
        rows[0]["run"]["rescored_from"] = rows[0]["path"]
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(rows[0]["run"], f, indent=2, ensure_ascii=False)
        print(f"wrote {args.write}")

    print(f"\n{'run':46s} {'facts':>15s} {'url_hit':>9s} {'±cases':>7s}")
    print("-" * 82)
    moved = 0
    for res in sorted(rows, key=lambda r: r["path"]):
        b, a = res["before"], res["after"]
        fb, fa = b["fact_recall"], a["fact_recall"]
        n = len(res["changed"]) + len(res["truncated_changed"])
        moved += n > 0
        flag = "" if n == 0 else " *"
        name = f"{Path(res['path']).parent.name}/{Path(res['path']).name}"[-45:]
        print(f"{name:46s} {fb:.3f} -> {fa:.3f} {a['url_hit_rate']:>9.3f} {n:>7d}{flag}")

    n_trunc = sum(len(r["truncated_changed"]) for r in rows)
    print("-" * 82)
    print(f"{len(rows)} runs re-scored, {moved} moved, {n_trunc} case-deltas on truncated excerpts")
    if all_mismatches:
        print("\n!! URL METRIC MISMATCH — the re-scorer disagrees with the stored run:")
        for name, cid, old, new in all_mismatches:
            print(f"   {name} {cid}: {old} -> {new}")
    else:
        print("URL self-check: every run reproduced its stored url_hit_rate exactly.")


if __name__ == "__main__":
    main()
