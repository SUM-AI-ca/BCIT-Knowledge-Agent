"""Mine production LangSmith traces into golden-set candidates.

Pulls root `bcit_query` runs from the LangSmith project, dedupes real user
questions (eval-harness questions are excluded by matching against the golden
sets), and writes a candidates JSONL a human can curate into golden_set_v3:
verify the facts against the corpus, add expected_urls + key_facts, drop the
rest. Candidates carry the answer excerpt and sub-queries so triage doesn't
require re-running anything.

Thumbs-down feedback (the frontend's POST /feedback writes `feedback_log`
JSON lines to journald) can be joined in to float problem cases to the top:

    # on the VM
    journalctl -u bcit-chatbot --since "30 days ago" | grep feedback_log > /tmp/feedback.log
    # locally
    python eval/mine_langsmith.py --days 30 --feedback-log /tmp/feedback.log

Needs LANGSMITH_API_KEY in backend/.env (already there if tracing is on).
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
GOLDEN_FILES = ["golden_set.jsonl", "golden_set_v2.jsonl", "golden_set_v3.jsonl"]


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def golden_questions():
    """Questions already covered by (or queued for) the golden sets — these
    show up in traces whenever an eval ran with tracing on."""
    known = set()
    for name in GOLDEN_FILES:
        path = EVAL_DIR / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                known.add(_norm(case.get("question")))
                if case.get("follow_up"):
                    known.add(_norm(case["follow_up"]))
    return known


def load_feedback_downs(path):
    """Normalized questions with a thumbs-down, from a journalctl dump that
    contains `feedback_log {...}` lines."""
    downs = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            idx = line.find("feedback_log ")
            if idx == -1:
                continue
            try:
                entry = json.loads(line[idx + len("feedback_log "):].strip())
            except json.JSONDecodeError:
                continue
            if entry.get("verdict") == "down" and entry.get("question"):
                downs.add(_norm(entry["question"]))
    return downs


def fetch_runs(project, since, limit):
    from langsmith import Client

    client = Client()
    # No `limit` kwarg: the API caps it at 100 per request; the SDK paginates
    # an unbounded iterator, so the scan budget is enforced client-side.
    runs = client.list_runs(project_name=project, start_time=since, is_root=True)
    for i, run in enumerate(runs):
        if i >= limit:
            break
        yield run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    parser.add_argument("--limit", type=int, default=1000, help="max runs to scan (default 1000)")
    parser.add_argument("--project", default=None, help="LangSmith project (default: env LANGSMITH_PROJECT or bcit-chatbot)")
    parser.add_argument("--out", default=None, help="output path (default eval/candidates_<date>.jsonl)")
    parser.add_argument("--feedback-log", help="journalctl dump containing feedback_log lines; thumbs-down questions sort first")
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")
    import os
    project = args.project or os.getenv("LANGSMITH_PROJECT", "bcit-chatbot")
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY not set (expected in backend/.env)")

    known = golden_questions()
    downs = load_feedback_downs(args.feedback_log) if args.feedback_log else set()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    candidates = {}  # normalized question -> candidate dict
    scanned = 0
    for run in fetch_runs(project, since, args.limit):
        scanned += 1
        if run.name != "bcit_query":
            continue
        question = (run.inputs or {}).get("question")
        key = _norm(question)
        if not key or key in known:
            continue

        outputs = run.outputs or {}
        latency = None
        if run.start_time and run.end_time:
            latency = round((run.end_time - run.start_time).total_seconds(), 2)

        if key in candidates:
            cand = candidates[key]
            cand["n_occurrences"] += 1
            cand["last_seen"] = max(cand["last_seen"], str(run.start_time))
        else:
            candidates[key] = {
                "question": question,
                "standalone_question": outputs.get("standalone_question"),
                "sub_queries": outputs.get("sub_queries"),
                "answer_excerpt": (outputs.get("answer") or "")[:400],
                "n_occurrences": 1,
                "first_seen": str(run.start_time),
                "last_seen": str(run.start_time),
                "latency_s": latency,
                "error": run.error,
                "feedback": "down" if key in downs else None,
                "note": "TODO: verify facts against the corpus, add expected_urls + key_facts, then move into golden_set_v3.jsonl",
            }

    ordered = sorted(
        candidates.values(),
        key=lambda c: (
            c["feedback"] != "down",      # thumbs-down first
            c["error"] is None,           # then errored runs
            -c["n_occurrences"],
            c["last_seen"],
        ),
    )
    for i, cand in enumerate(ordered, 1):
        cand_id = f"cand-{i:03d}"
        cand_with_id = {"id": cand_id, **cand}
        ordered[i - 1] = cand_with_id

    out_path = Path(args.out) if args.out else EVAL_DIR / f"candidates_{datetime.now():%Y%m%d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for cand in ordered:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")

    n_down = sum(1 for c in ordered if c["feedback"] == "down")
    n_err = sum(1 for c in ordered if c["error"])
    print(
        f"Scanned {scanned} runs ({project}, last {args.days}d): "
        f"{len(ordered)} unique candidate questions "
        f"({n_down} thumbs-down, {n_err} errored, golden-set questions excluded)"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
