"""Eval for the person-lookup / fan-in class: "what courses does X teach?"

Why this is not a run_eval.py golden set: the failure here is ROLE
MISATTRIBUTION, and substring fact matching cannot see it. Every course code
in the ground truth appears in a wrong answer too — turn 1 of the reported
conversation named COMP 3981 while calling it "reviewed as Program Head", and
`_fact_hit` would score that a hit. So the answer is graded by extracting the
set of courses the answer CLAIMS THE PERSON TEACHES and comparing sets.

Retrieval-side counters are recorded alongside, because they are deterministic
and they say which stage a change acted on:
  ctx_instr  - context chunks that carry an Instructor Details / ### Instructor
               block naming the person (the evidence the answer needs)
  ctx_sig    - context chunks that are the outline approval signature block
               (the same name in a role that is NOT the instructor)

Usage (env before launch — config reads env at import):
    .venv/bin/python eval/person_lookup.py --label base
    SIGNATURE_DEMOTE=true .venv/bin/python eval/person_lookup.py --label sig
    FANIN_RETAIN=true     .venv/bin/python eval/person_lookup.py --label fanin
    SIGNATURE_DEMOTE=true FANIN_RETAIN=true \
        .venv/bin/python eval/person_lookup.py --label both
    .venv/bin/python eval/person_lookup.py --compare base sig fanin both
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

GOLDEN = EVAL_DIR / "golden_set_people.jsonl"
RESULTS = EVAL_DIR / "results"

CODE_RE = re.compile(r"\b([A-Z]{2,4})\s*-?\s*(\d[A-Z0-9]{3,4})\b")
SIG_RE = re.compile(r"I verify that .*?(?:Program Head|Associate Dean|Faculty|Dean)")


def load_cases():
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def classify_chunk(text, person):
    """What role does this chunk give the person? Mirrors the corpus layout."""
    has_person = person.lower() in text.lower()
    if not has_person:
        return None
    if re.search(r"### Instructor Details", text) or re.search(r"^Name \| ", text, re.M) \
            or re.search(r"### Instructor\s*\n", text):
        return "instr"
    if SIG_RE.search(text) or "Program Head" in text:
        return "sig"
    return "other"


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "taught": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Course codes the answer states this person TEACHES or "
                           "INSTRUCTS. Exclude any course the answer only says they "
                           "reviewed, approved, or signed off as Program Head.",
        },
        "reviewed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Course codes the answer attributes to a reviewing/"
                           "approving/Program Head role rather than teaching.",
        },
    },
    "required": ["taught", "reviewed"],
}

EXTRACT_PROMPT = """Read the answer below and classify every BCIT course code it mentions.

Put a code in "taught" ONLY if the answer says {person} teaches / instructs / is the
instructor for it. Put it in "reviewed" if the answer attributes it to a reviewing,
approving, or Program Head role. A code mentioned in both ways goes in "taught".
Do not infer anything the answer does not say. Use the exact code format "DEPT 1234".

ANSWER:
{answer}"""


def make_extractor():
    from langchain_google_vertexai import ChatVertexAI
    from config import JUDGE_MODEL, GEMINI_PROJECT, GEMINI_LOCATION
    llm = ChatVertexAI(
        model_name=JUDGE_MODEL, project=GEMINI_PROJECT, location=GEMINI_LOCATION,
        temperature=0, max_output_tokens=1024,
        response_mime_type="application/json", response_schema=EXTRACT_SCHEMA,
        thinking_budget=0,
    )

    def extract(person, answer):
        for attempt in range(3):
            try:
                raw = llm.invoke(EXTRACT_PROMPT.format(person=person, answer=answer))
                d = json.loads(raw.content if isinstance(raw.content, str)
                               else raw.content[0]["text"])
                norm = lambda xs: {f"{m.group(1)} {m.group(2)}"
                                   for x in xs for m in [CODE_RE.search(x.upper())] if m}
                return norm(d.get("taught", [])), norm(d.get("reviewed", []))
            except Exception as e:
                if attempt == 2:
                    return None, str(e)
                time.sleep(3)
    return extract


def run(label, sleep):
    from query_rag import BCITChatbot
    import config

    cases = load_cases()
    bot = BCITChatbot()
    extract = make_extractor()
    flags = {
        "SIGNATURE_DEMOTE": config.SIGNATURE_DEMOTE,
        "FANIN_RETAIN": config.FANIN_RETAIN,
        "FANIN_RETAIN_N": config.FANIN_RETAIN_N,
        "RERANKER_TOP_K": config.RERANKER_TOP_K,
    }
    print(f"\n=== {label}  {flags}\n")

    out = []
    for c in cases:
        t0 = time.time()
        # memory=None -> every case is a first turn, as in the reported
        # conversation. The response cache is disabled by the runner env so a
        # repeat inside an arm cannot short-circuit the pipeline.
        meta = bot.query_with_meta(c["question"])
        answer = meta["answer"]
        docs = meta.get("docs") or []
        roles = [classify_chunk(d.page_content, c["person"]) for d in docs]
        said_taught, said_reviewed = extract(c["person"], answer)
        truth = set(c["teaches"])
        if said_taught is None:
            rec = {"id": c["id"], "error": said_reviewed}
        else:
            tp = len(said_taught & truth)
            rec = {
                "id": c["id"], "person": c["person"], "question": c["question"],
                "truth": sorted(truth),
                "said_taught": sorted(said_taught),
                "said_reviewed": sorted(said_reviewed or []),
                "correct": sorted(said_taught & truth),
                "missed": sorted(truth - said_taught),
                "wrong": sorted(said_taught - truth),
                "recall": tp / len(truth) if truth else None,
                "precision": tp / len(said_taught) if said_taught else 0.0,
                "ctx_instr": roles.count("instr"),
                "ctx_sig": roles.count("sig"),
                "ctx_docs": len(docs),
                "ctx_sources": len({d.metadata.get("source") for d in docs}),
                "fanin_swaps": meta.get("fanin_swaps"),
                "n_subqueries": meta.get("n_subqueries"),
                "input_tokens": (meta.get("usage") or {}).get("input_tokens"),
                "cost_usd": meta.get("est_cost_usd"),
                "latency_s": round(time.time() - t0, 2),
                "answer_excerpt": answer[:1200],
            }
            f1 = (2 * rec["precision"] * rec["recall"] / (rec["precision"] + rec["recall"])
                  if rec["precision"] + rec["recall"] else 0.0)
            rec["f1"] = round(f1, 4)
            print(f"  {c['id']} {c['person']:<22} R={rec['recall']:.2f} P={rec['precision']:.2f} "
                  f"F1={f1:.2f}  ctx instr={rec['ctx_instr']} sig={rec['ctx_sig']}"
                  f"  wrong={rec['wrong'] or '-'}")
        out.append(rec)
        if sleep:
            time.sleep(sleep)

    ok = [r for r in out if "error" not in r]
    mean = lambda k: round(sum(r[k] for r in ok) / len(ok), 4) if ok else None
    agg = {
        "label": label, "flags": flags, "n": len(ok), "n_errors": len(out) - len(ok),
        "recall": mean("recall"), "precision": mean("precision"), "f1": mean("f1"),
        "ctx_instr": mean("ctx_instr"), "ctx_sig": mean("ctx_sig"),
        "ctx_sources": mean("ctx_sources"),
        "input_tokens": mean("input_tokens"), "cost_usd": mean("cost_usd"),
        "exact_sets": sum(1 for r in ok if set(r["said_taught"]) == set(r["truth"])),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"people_{label}.json"
    path.write_text(json.dumps({"aggregate": agg, "cases": out}, indent=2, ensure_ascii=False))
    print(f"\n{json.dumps(agg, indent=2)}\nwrote {path}")
    return agg


def compare(labels):
    runs = [json.loads((RESULTS / f"people_{l}.json").read_text()) for l in labels]
    print(f"\n{'metric':<26}" + "".join(f"{l:>12}" for l in labels))
    print("-" * (26 + 12 * len(labels)))
    for k, fmt in [("recall", "{:.3f}"), ("precision", "{:.3f}"), ("f1", "{:.3f}"),
                   ("exact_sets", "{:.0f}"), ("ctx_instr", "{:.2f}"), ("ctx_sig", "{:.2f}"),
                   ("ctx_sources", "{:.2f}"), ("input_tokens", "{:.0f}"), ("cost_usd", "{:.5f}")]:
        row = "".join(f"{fmt.format(r['aggregate'][k]):>12}"
                      if r["aggregate"].get(k) is not None else f"{'-':>12}" for r in runs)
        print(f"{k:<26}{row}")
    print(f"\n{'per-case F1':<26}" + "".join(f"{l:>12}" for l in labels))
    print("-" * (26 + 12 * len(labels)))
    ids = [c["id"] for c in runs[0]["cases"]]
    for i, cid in enumerate(ids):
        person = runs[0]["cases"][i].get("person", "")
        row = "".join(f"{r['cases'][i].get('f1', float('nan')):>12.2f}" for r in runs)
        print(f"{cid + ' ' + person:<26}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.label:
        ap.error("--label or --compare required")
    run(a.label, a.sleep)


if __name__ == "__main__":
    main()
