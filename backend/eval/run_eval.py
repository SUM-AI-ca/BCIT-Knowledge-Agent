"""Offline eval harness for the BCIT RAG chatbot.

Runs the golden set through BCITChatbot.query_with_meta() directly (no HTTP
server) and writes per-case + aggregate metrics to eval/results/<label>.json.

config.py reads tunables from env at import time, so flags must be set BEFORE
launching, e.g.:

    # BEFORE baseline (legacy full-document context, no decomposition)
    CONTEXT_MODE=full_doc MULTI_QUERY_ENABLED=false \\
        .venv/bin/python eval/run_eval.py --label before

    # AFTER (current defaults)
    .venv/bin/python eval/run_eval.py --label after

    # Compare two result files (no DB/API access needed)
    .venv/bin/python eval/run_eval.py --compare results/before.json results/after.json

Local WSL note (README gotchas): a local postgres owns 5432, so rewrite
PG_CONNECTION to the 5433 Cloud SQL proxy first:

    export PG_CONNECTION=$(grep '^PG_CONNECTION=' .env | cut -d= -f2- | sed 's/:5432/:5433/')

Golden set schema (golden_set.jsonl, one JSON object per line):
    id, category, question, follow_up?, expected_urls, key_facts
expected_urls and key_facts are lists of groups; each group is a list of
alternatives (a plain string is treated as a single-alternative group). A
group counts as hit/recalled when ANY alternative matches. For cases with a
follow_up, the follow-up turn is the one scored (it exercises the rewrite).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"

# Tunables snapshotted into the results file. Names that config.py does not
# define (yet) are skipped, so this list can stay ahead of the code.
SNAPSHOT_KEYS = [
    "GEMINI_MODEL", "REWRITER_MODEL", "JUDGE_MODEL", "GEMINI_TEMPERATURE", "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_THINKING_BUDGET", "MEMORY_WINDOW_K",
    "USE_HYBRID_SEARCH", "HYBRID_ALPHA", "USE_RERANKING",
    "RERANKER_CANDIDATES", "RERANKER_TOP_K", "RERANK_SCORE_THRESHOLD",
    "MIN_CONTEXT_CHUNKS", "CONTEXT_MODE", "NEIGHBOR_RADIUS",
    "CONTEXT_MAX_CHARS", "MULTI_QUERY_ENABLED", "MAX_SUB_QUERIES",
    "MQ_DENSE_K", "MQ_BM25_K", "MQ_CANDIDATES_PER_SUBQUERY", "MQ_POOL_CAP",
    "MQ_MIN_CHUNKS_PER_SUBQUERY", "RERANK_MODE",
    "REWRITE_MAX_OUTPUT_TOKENS", "REWRITE_THINKING_BUDGET",
    "HISTORY_MAX_ANSWER_CHARS", "STRIP_SOURCES_FROM_HISTORY",
    "RETRIEVAL_TOP_K", "RETRIEVAL_DENSE_K", "RETRIEVAL_BM25_K",
    "PG_COLLECTION",
    "PRICE_GEN_INPUT_PER_M", "PRICE_GEN_OUTPUT_PER_M",
    "PRICE_REWRITE_INPUT_PER_M", "PRICE_REWRITE_OUTPUT_PER_M",
    "PRICE_RERANK_PER_CALL", "PRICE_EMBED_PER_QUERY",
    "RRF_K", "MMR_LAMBDA", "RETRIEVAL_FETCH_K",
    "BM25_INDEX_AUG", "RERANK_SKIP_CONSENSUS",
    "MQ_BM25_KEYWORDS", "HYDE_MODE",
    "REWRITE_SKIP_SIMPLE", "REWRITE_SKIP_MAX_WORDS",
]

# Optional per-query meta keys added by later optimization steps; copied into
# per-case results when present so before/after files stay comparable.
EXTRA_META_KEYS = [
    "n_subqueries", "sub_queries", "decompose_fallback",
    "context_mode", "context_chars", "n_context_sources",
    "n_chunks_kept", "neighbor_misses",
    "n_rerank_calls", "rerank_skipped", "pool_consensus",
    "rewrite_skipped", "retrieval_mode", "n_candidates",
]

ANSWER_EXCERPT_CHARS = 1500


def _normalize(text):
    text = text.lower()
    for dash in ("–", "—", "‑"):
        text = text.replace(dash, "-")
    text = text.replace(",", "")  # "6,000" -> "6000"
    # "4.0-credit course" -> "4.0 credit course": models hyphenate compound
    # modifiers freely, and enumerating both forms in every key_facts group
    # is the kind of upkeep that silently rots. Applied to BOTH sides, so it
    # can never make a group match something it did not already say.
    text = re.sub(r"(?<=\d)-(?=[a-z])", " ", text)
    return re.sub(r"\s+", " ", text)


def _groups(raw):
    """[[a, b], c] -> [[a, b], [c]]"""
    return [[g] if isinstance(g, str) else list(g) for g in raw]


def _fact_hit(alternatives, normalized_answer):
    for alt in alternatives:
        needle = _normalize(alt)
        pattern = re.escape(needle)
        # A date/number fact also counts in its ordinal form: the corpus says
        # "July 2" and every model writes "July 2nd". Only for digit-final
        # alternatives, so this can never widen a word-shaped fact.
        if needle and needle[-1].isdigit():
            pattern += r"(?:st|nd|rd|th)?"
        # ASCII-only boundaries so "75" still does not match "175" or "1510".
        # NOT \w: it is Unicode-aware, so a Korean particle glued to the token
        # ("english studies 12에서") read as a word continuation and scored a
        # correct non-English answer as a miss — systematically, since
        # RESPONSE_LANGUAGE=match means answers come back in the asker's
        # language while key_facts stay English.
        if re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", normalized_answer):
            return True
    return False


def _norm_url(url):
    return (url or "").strip().lower().rstrip("/")


def _cited_urls(answer):
    # Prefer the Sources section if present; fall back to the whole answer.
    match = re.search(r"sources\s*:?", answer, re.IGNORECASE)
    scope = answer[match.start():] if match else answer
    urls = re.findall(r"https?://[^\s)\]>\"']+", scope)
    return [_norm_url(u.rstrip(".,;")) for u in urls]


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round(q * (len(ordered) - 1)))
    return ordered[idx]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def score_case(case, result):
    answer = result.get("answer") or ""
    normalized_answer = _normalize(answer)

    retrieved = []
    for doc in result.get("docs") or []:
        url = _norm_url(doc.metadata.get("url"))
        if url and url not in retrieved:
            retrieved.append(url)

    url_groups = _groups(case.get("expected_urls", []))
    url_hits = [
        any(_norm_url(alt) in retrieved for alt in group)
        for group in url_groups
    ]

    fact_groups = _groups(case.get("key_facts", []))
    fact_hits = [_fact_hit(group, normalized_answer) for group in fact_groups]

    # `must_not_contain`: same alternative-group shape as key_facts, but the
    # answer is supposed to say NONE of it. This is the only way to score a
    # case whose correct answer is a refusal — an out-of-scope or
    # not-in-the-corpus question has no key fact to recall, so every
    # recall-shaped metric returns None and the case measures nothing.
    forbidden_groups = _groups(case.get("must_not_contain", []))
    forbidden_hits = [_fact_hit(group, normalized_answer) for group in forbidden_groups]

    cited = _cited_urls(answer)
    cited_in_context = [u for u in cited if u in retrieved]

    return {
        "retrieved_urls": retrieved,
        "url_hit_fraction": (sum(url_hits) / len(url_hits)) if url_hits else None,
        "url_full_hit": all(url_hits) if url_hits else None,
        "urls_missed": [g[0] for g, hit in zip(url_groups, url_hits) if not hit],
        "fact_recall": (sum(fact_hits) / len(fact_hits)) if fact_hits else None,
        "facts_missed": [g[0] for g, hit in zip(fact_groups, fact_hits) if not hit],
        "avoidance": (
            1.0 - sum(forbidden_hits) / len(forbidden_hits) if forbidden_groups else None
        ),
        "forbidden_present": [
            g[0] for g, hit in zip(forbidden_groups, forbidden_hits) if hit
        ],
        "n_cited": len(cited),
        "citation_precision": (len(cited_in_context) / len(cited)) if cited else None,
    }


# --- LLM judge (--judge) ---------------------------------------------------
# One extra schema-constrained call per case, grading the answer against the
# retrieved passages only. Complements (does not replace) the substring
# metrics: fact_recall undercounts paraphrases, and nothing else measures
# whether the answer invented claims the context never contained.

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "faithfulness": {"type": "NUMBER"},
        "completeness": {"type": "NUMBER"},
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
        "notes": {"type": "STRING"},
    },
    "required": ["faithfulness", "completeness", "unsupported_claims"],
}

JUDGE_TEMPLATE = """You are grading an answer from a BCIT academic-advisor RAG chatbot.

You get the student's question, the retrieved source passages the bot was
given, and the bot's answer. Grade ONLY against the provided passages —
your own knowledge must never rescue an unsupported claim.

Return JSON:
1. "faithfulness" (0.0-1.0): the fraction of the answer's factual claims that
   are directly supported by the passages. URLs in the Sources section are
   not claims. 1.0 = every claim supported.
2. "unsupported_claims": every factual claim NOT supported by the passages,
   quoted or closely paraphrased. Empty list if none.
3. "completeness" (0.0-1.0): how fully the answer addresses every part of the
   question using information that IS present in the passages. An answer that
   correctly says the information is unavailable counts as complete.
4. "notes": one short sentence, only if something needs explaining.

Question:
{question}

Retrieved passages:
{context}

Answer:
{answer}"""


def _judge_context(docs, per_doc_chars=1500, max_total_chars=30000):
    blocks = []
    total = 0
    for i, doc in enumerate(docs or [], 1):
        label = doc.metadata.get("url") or doc.metadata.get("title") or ""
        block = f"[Passage {i}] {label}\n{doc.page_content[:per_doc_chars]}"
        total += len(block)
        if total > max_total_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks) or "(no passages retrieved)"


def judge_case(judge_llm, question, result):
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context=_judge_context(result.get("docs")),
        answer=(result.get("answer") or "")[:6000],
    )
    msg = judge_llm.invoke(prompt)
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    data = json.loads(content)
    faith = data.get("faithfulness")
    comp = data.get("completeness")
    return {
        "judge_faithfulness": float(faith) if faith is not None else None,
        "judge_completeness": float(comp) if comp is not None else None,
        "judge_unsupported_claims": [
            c for c in (data.get("unsupported_claims") or []) if isinstance(c, str)
        ][:10],
        "judge_notes": (data.get("notes") or "")[:300],
    }


def make_judge(config):
    from langchain_google_vertexai import ChatVertexAI
    return ChatVertexAI(
        model=config.JUDGE_MODEL,
        project=config.GEMINI_PROJECT,
        location=config.GEMINI_LOCATION,
        temperature=0.0,
        max_output_tokens=1024,
        thinking_budget=0,
        response_mime_type="application/json",
        response_schema=JUDGE_SCHEMA,
    )


def run_case(bot, case, make_memory, judge_llm=None):
    memory = make_memory()
    turns = []

    final_question = case["question"]
    if case.get("follow_up"):
        first = bot.query_with_meta(case["question"], memory=memory)
        turns.append({
            "question": case["question"],
            "usage": first["usage"],
            "est_cost_usd": first.get("est_cost_usd"),
            "timings": first["timings"],
        })
        final_question = case["follow_up"]

    result = bot.query_with_meta(final_question, memory=memory)

    record = {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "question": case["question"],
        "follow_up": case.get("follow_up"),
        "standalone_question": result.get("standalone_question"),
        "finish_reason": result.get("finish_reason"),
        "n_context_docs": result.get("n_context_docs"),
        "usage": result.get("usage", {}),
        "est_cost_usd": result.get("est_cost_usd"),
        "timings": result.get("timings", {}),
        "prior_turns": turns,
        "answer_excerpt": (result.get("answer") or "")[:ANSWER_EXCERPT_CHARS],
        "error": None,
    }
    for key in EXTRA_META_KEYS:
        if key in result:
            record[key] = result[key]
    record.update(score_case(case, result))
    if judge_llm is not None:
        try:
            record.update(judge_case(judge_llm, final_question, result))
        except Exception as exc:  # a judge failure must not sink the case
            record["judge_error"] = f"{type(exc).__name__}: {exc}"
    return record


def aggregate(cases):
    scored = [c for c in cases if c.get("error") is None]
    usage = lambda key: [c["usage"].get(key) for c in scored if c.get("usage")]
    total = lambda c: (c["usage"].get("input_tokens", 0) or 0) + (c["usage"].get("rewrite_input_tokens", 0) or 0)

    latencies = [c["timings"].get("total_s") for c in scored if c.get("timings")]
    citation = [c["citation_precision"] for c in scored if c.get("citation_precision") is not None]
    costs = [c.get("est_cost_usd") for c in scored if c.get("est_cost_usd") is not None]

    return {
        "n_cases": len(cases),
        "n_errors": sum(1 for c in cases if c.get("error")),
        "url_hit_rate": _mean([c["url_hit_fraction"] for c in scored]),
        "url_full_hit_rate": _mean([1.0 if c["url_full_hit"] else 0.0 for c in scored if c["url_full_hit"] is not None]),
        "fact_recall": _mean([c["fact_recall"] for c in scored]),
        "avoidance": _mean([c.get("avoidance") for c in scored]),
        "n_cases_with_forbidden": sum(1 for c in scored if c.get("forbidden_present")),
        "citation_precision": _mean(citation),
        "n_answers_without_citation": sum(1 for c in scored if c.get("n_cited") == 0),
        "input_tokens_mean": _mean(usage("input_tokens")),
        "input_tokens_p50": _percentile([v for v in usage("input_tokens") if v is not None], 0.5),
        "input_tokens_with_rewrite_mean": _mean([total(c) for c in scored]),
        "output_tokens_mean": _mean(usage("output_tokens")),
        "reasoning_tokens_mean": _mean(usage("reasoning_tokens")),
        "cache_read_tokens_mean": _mean(usage("cache_read_tokens")),
        "est_cost_usd_mean": _mean(costs),
        "est_cost_usd_total": sum(costs) if costs else None,
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "max_tokens_truncations": sum(1 for c in scored if c.get("finish_reason") == "MAX_TOKENS"),
        "decompose_fallbacks": sum(1 for c in scored if c.get("decompose_fallback")),
        "judge_faithfulness_mean": _mean([c.get("judge_faithfulness") for c in scored]),
        "judge_completeness_mean": _mean([c.get("judge_completeness") for c in scored]),
        "judge_flagged_cases": sum(1 for c in scored if c.get("judge_unsupported_claims")),
        "judge_errors": sum(1 for c in scored if c.get("judge_error")),
        "n_context_docs_mean": _mean([c.get("n_context_docs") for c in scored]),
        "rerank_skip_rate": _mean([1.0 if c.get("rerank_skipped") else 0.0 for c in scored]),
        "rewrite_skip_rate": _mean([1.0 if c.get("rewrite_skipped") else 0.0 for c in scored]),
        "rerank_calls_mean": _mean([c.get("n_rerank_calls") for c in scored]),
    }


def load_golden(path, only=None, category=None, limit=None):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if only:
        cases = [c for c in cases if only in c["id"]]
    if category:
        cases = [c for c in cases if c.get("category") == category]
    if limit:
        cases = cases[:limit]
    return cases


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def print_aggregate(label, agg, per_category=None):
    print(f"\n=== {label} ===")
    for key, value in agg.items():
        print(f"  {key:32s} {_fmt(value)}")
    if per_category:
        print(f"  {'-' * 50}")
        print(f"  {'category':20s} {'url_hit':>8s} {'facts':>8s} {'in_tok':>10s}")
        for name, cat_agg in sorted(per_category.items()):
            print(
                f"  {name:20s} {_fmt(cat_agg['url_hit_rate']):>8s} "
                f"{_fmt(cat_agg['fact_recall']):>8s} "
                f"{_fmt(cat_agg['input_tokens_mean']):>10s}"
            )


def compare(path_a, path_b):
    with open(path_a, "r", encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        b = json.load(f)

    agg_a, agg_b = a["aggregate"], b["aggregate"]
    label_a, label_b = a.get("label", "A"), b.get("label", "B")

    print(f"\n{'metric':34s} {label_a:>14s} {label_b:>14s} {'delta':>12s}")
    print("-" * 78)
    for key in agg_a:
        va, vb = agg_a.get(key), agg_b.get(key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            diff = vb - va
            delta = f"{diff:+,.3f}" if abs(diff) < 10 else f"{diff:+,.1f}"
            if va not in (0, None) and key.endswith(("_mean", "_p50", "_p95_s", "_p50_s")):
                delta += f" ({diff / va * 100:+.0f}%)"
        print(f"{key:34s} {_fmt(va):>14s} {_fmt(vb):>14s} {delta:>12s}")

    # Per-case regressions on retrieval/facts
    cases_a = {c["id"]: c for c in a.get("cases", [])}
    regressions = []
    for case in b.get("cases", []):
        prev = cases_a.get(case["id"])
        if not prev:
            continue
        for metric in ("url_hit_fraction", "fact_recall"):
            va, vb = prev.get(metric), case.get(metric)
            if va is not None and vb is not None and vb < va:
                regressions.append((case["id"], metric, va, vb))
    if regressions:
        print(f"\nPer-case regressions ({label_b} < {label_a}):")
        for case_id, metric, va, vb in regressions:
            print(f"  {case_id:8s} {metric:18s} {va:.2f} -> {vb:.2f}")
    else:
        print(f"\nNo per-case regressions in url_hit_fraction / fact_recall.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", help="name for this run; results land in eval/results/<label>.json")
    parser.add_argument("--golden", default=str(EVAL_DIR / "golden_set.jsonl"))
    parser.add_argument("--only", help="substring filter on case id (e.g. 'mp-')")
    parser.add_argument("--category", help="exact category filter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="override output path")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    parser.add_argument(
        "--judge", action="store_true",
        help="LLM judge: grade faithfulness/completeness against the retrieved "
             "passages (one extra JUDGE_MODEL call per case, ~$0.002 each)",
    )
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if not args.label:
        parser.error("--label is required for an eval run (or use --compare)")

    cases = load_golden(args.golden, only=args.only, category=args.category, limit=args.limit)
    if not cases:
        parser.error("no cases selected")

    # config reads env at import; .env paths are relative to backend/.
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, str(BACKEND_DIR))
    import config
    from query_rag import BCITChatbot
    from langchain.memory import ConversationBufferWindowMemory

    snapshot = {}
    for key in SNAPSHOT_KEYS:
        if hasattr(config, key):
            value = getattr(config, key)
            snapshot[key] = value if isinstance(value, (int, float, bool, str, type(None))) else str(value)

    make_memory = lambda: ConversationBufferWindowMemory(
        k=config.MEMORY_WINDOW_K, memory_key="chat_history", return_messages=True
    )

    judge_llm = make_judge(config) if args.judge else None

    bot = BCITChatbot()
    print(f"\nRunning {len(cases)} cases (label={args.label}"
          f"{', judge=' + config.JUDGE_MODEL if judge_llm else ''})\n")

    records = []
    started = time.perf_counter()
    for i, case in enumerate(cases, 1):
        try:
            record = run_case(bot, case, make_memory, judge_llm)
        except Exception as exc:  # keep going; the failure is data too
            record = {
                "id": case["id"],
                "category": case.get("category", "uncategorized"),
                "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        if record.get("error"):
            print(f"[{i}/{len(cases)}] {case['id']:8s} ERROR {record['error']}")
        else:
            usage = record.get("usage", {})
            judge_part = ""
            if record.get("judge_faithfulness") is not None:
                judge_part = (
                    f" judge={record['judge_faithfulness']:.2f}"
                    f"/{record.get('judge_completeness', 0) or 0:.2f}"
                )
            elif record.get("judge_error"):
                judge_part = " judge=ERR"
            print(
                f"[{i}/{len(cases)}] {case['id']:8s} ok "
                f"urls={_fmt(record.get('url_hit_fraction'))} "
                f"facts={_fmt(record.get('fact_recall'))} "
                f"in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
                f"t={record.get('timings', {}).get('total_s')}s{judge_part}"
            )
        sys.stdout.flush()

    per_category = {}
    for name in sorted({r["category"] for r in records}):
        per_category[name] = aggregate([r for r in records if r["category"] == name])

    payload = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set": str(Path(args.golden).name),
        "wall_time_s": round(time.perf_counter() - started, 1),
        "config_snapshot": snapshot,
        "aggregate": aggregate(records),
        "per_category": per_category,
        "cases": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print_aggregate(args.label, payload["aggregate"], per_category)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
