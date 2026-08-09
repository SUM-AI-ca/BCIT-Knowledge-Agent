"""LLM controller graph (USE_GRAPH).

One LLM node makes every control decision, once per iteration:

  iteration 0 (no evidence yet)  -> the decision IS the route
  iteration 1+ (digest in hand)  -> the decision IS the coverage gate
  across iterations              -> the sequence IS the orchestration

Three jobs, one JSON contract (`GRAPH_CONTROLLER_SCHEMA`), one prompt. Splitting
them into three prompts would cost two extra requests per turn to reach the
same decisions.

The graph produces retrieved documents and a routing verdict, and nothing else.
Generation stays outside it, so the answer is still one uninterrupted
`llm.stream()` call — `query_stream`, `_finalize_turn` and `server.py` never
learn this module exists. That is what makes a loop compatible with streaming
here when Self-RAG was rejected for not being.

This module knows nothing about retrieval. `BCITChatbot` injects two callables
and gets back a verdict; see `ControllerGraph.__init__`.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from config import (
    GRAPH_CONTROLLER_SCHEMA,
    GRAPH_DIGEST_CHARS,
    GRAPH_MAX_HOPS,
)

logger = logging.getLogger(__name__)

# Placeholders a model reaches for when it wants to keep searching but has
# nothing concrete to search for. A `retrieve` whose `missing` is only these is
# downgraded to `answer`: it is the signature of a question the corpus cannot
# answer, and honouring it is how `unanswerable` would start looping.
_VAGUE_MISSING = {
    "more information", "more info", "more detail", "more details",
    "additional information", "additional details", "further information",
    "details", "information", "data", "specifics", "context", "everything",
    "other courses", "other programs", "anything else", "the rest",
}

ACTIONS = ("answer", "refuse", "retrieve")


class TurnState(TypedDict, total=False):
    """State threaded through the graph. `docs` is the accumulator."""
    question: str
    chat_history: str
    hop: int
    action: str
    reason: str
    queries: List[str]
    missing: List[str]
    docs: List[Document]
    sub_queries: List[str]
    retrieval: Dict[str, Any]      # stats from the last retrieve node
    trace: List[Dict[str, Any]]    # one record per controller call, for eval
    usage: Dict[str, int]          # accumulated controller tokens


def is_concrete(missing: List[str]) -> bool:
    """Does `missing` name something a search could actually find?

    The deterministic prototype of this gate got this property from its data
    structure — no relation in the entity index meant nothing to ask for, so an
    unanswerable question could not produce a hop. An LLM gate has to be held
    to it explicitly, in code, because the prompt asking nicely is not a
    guarantee.
    """
    for item in missing or []:
        text = (item or "").strip().lower()
        if len(text) < 3 or text in _VAGUE_MISSING:
            continue
        # A concrete target names something: a code (COMP 2510), a number
        # (credits, year), or at least two words that are not filler.
        if any(ch.isdigit() for ch in text) or len(text.split()) >= 2:
            return True
    return False


# Structured fields BCIT outlines state as "Label | value". They are where the
# answers to the hop-shaped questions actually live (3,258 of 3,262 outlines
# carry Prerequisite(s), 3,171 carry Course Credits), and they are one short
# line each. Measured need: without them the controller re-requested COMP 1537
# and COMP 3522 prerequisites on consecutive hops because the digest's
# head-of-chunk excerpt did not happen to include the line that had arrived —
# it kept searching for something it already had, and hit the cap.
_FIELD_RE = re.compile(
    r"^(Prerequisite\(s\)|Corequisite\(s\)|Course Credits|Course Level|Name|Credits)\s*\|\s*(.+)$",
    re.M,
)
_MAX_DIGEST_SOURCES = 12
_MAX_FIELDS_PER_SOURCE = 4


def build_digest(docs: List[Document], sub_queries: List[str], max_chars: int = None) -> str:
    """Compact evidence summary for the controller.

    Deliberately NOT the assembled context. At 10 chunks the real context is
    ~5.7k tokens; a gate call on that would cost more per turn than the whole
    query does today. It is also the wrong input — the gate answers "what is
    covered?", not "what does this say?", so page identity plus the head of the
    text beats the full prose.

    Grouped by source page because that is the unit the controller reasons
    about; the ~10 chunks land on ~6 pages.
    """
    if max_chars is None:
        max_chars = GRAPH_DIGEST_CHARS
    if not docs:
        return "(nothing retrieved yet)"

    by_source: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        md = doc.metadata or {}
        source = md.get("source") or md.get("url") or "unknown"
        entry = by_source.setdefault(source, {
            "title": (md.get("title") or "").strip(),
            "url": (md.get("url") or "").strip(),
            "chunks": 0,
            "head": "",
            "fields": [],
        })
        entry["chunks"] += 1
        if not entry["head"]:
            entry["head"] = " ".join(doc.page_content.split())[:max_chars]
        for label, value in _FIELD_RE.findall(doc.page_content):
            field = f"{label} | {' '.join(value.split())[:160]}"
            if field not in entry["fields"]:
                entry["fields"].append(field)

    lines = []
    shown = list(by_source.items())[:_MAX_DIGEST_SOURCES]
    for i, (source, entry) in enumerate(shown, 1):
        label = entry["title"] or source.rsplit("/", 1)[-1]
        lines.append(f"[{i}] {label}" + (f" - {entry['url']}" if entry["url"] else ""))
        for field in entry["fields"][:_MAX_FIELDS_PER_SOURCE]:
            lines.append(f"    {field}")
        lines.append(f"    ({entry['chunks']} chunk(s)) {entry['head']}")

    header = (
        f"{len(docs)} chunks across {len(by_source)} BCIT pages"
        + (f" (showing {len(shown)})" if len(shown) < len(by_source) else "")
        + (f"; sub-queries issued: {sub_queries}" if sub_queries else "")
    )
    return header + "\n" + "\n".join(lines)


class ControllerGraph:
    """LangGraph wrapper around the controller/retrieve cycle.

    Injected by `BCITChatbot`:
      controller_fn(question, chat_history, evidence, hop, max_hops)
          -> (decision_dict, usage_dict)
      initial_retrieve_fn()
          -> {"docs": [...], "sub_queries": [...], ...stats}
      hop_retrieve_fn(queries, docs_so_far)
          -> {"docs": [...], ...stats}

    Keeping retrieval behind callables is what lets the whole existing pipeline
    (rewrite, decompose, fan-out, scoped arms, rerank) serve as the first hop
    unchanged, rather than being reimplemented as nodes.
    """

    def __init__(
            self,
            controller_fn: Callable[..., tuple],
            initial_retrieve_fn: Callable[[], dict],
            hop_retrieve_fn: Callable[[List[str], List[Document]], dict],
            max_hops: int = None,
    ):
        self._controller_fn = controller_fn
        self._initial_retrieve = initial_retrieve_fn
        self._hop_retrieve = hop_retrieve_fn
        self.max_hops = GRAPH_MAX_HOPS if max_hops is None else max_hops
        self._compiled = self._build()

    # -- nodes ---------------------------------------------------------------

    def _controller_node(self, state: TurnState) -> dict:
        hop = state.get("hop", 0)
        docs = state.get("docs") or []
        evidence = build_digest(docs, state.get("sub_queries") or [])

        try:
            decision, usage = self._controller_fn(
                question=state["question"],
                chat_history=state.get("chat_history", ""),
                evidence=evidence,
                hop=hop,
                max_hops=self.max_hops,
            )
        except Exception:
            # Fail open to today's behaviour, never to a refusal: on the first
            # iteration that means "retrieve" (the current pipeline), and after
            # that it means "answer" with what we already have. A controller
            # outage must not turn into the bot declining to help.
            logger.warning("controller call failed at hop %d", hop, exc_info=True)
            decision, usage = ({"action": "retrieve" if hop == 0 else "answer",
                                "reason": "controller error; fell back"}, {})

        action = str(decision.get("action") or "").strip().lower()
        if action not in ACTIONS:
            logger.warning("controller returned unknown action %r at hop %d", action, hop)
            action = "retrieve" if hop == 0 else "answer"

        queries = [q.strip() for q in (decision.get("queries") or [])
                   if isinstance(q, str) and q.strip()][:3]
        missing = [m.strip() for m in (decision.get("missing") or [])
                   if isinstance(m, str) and m.strip()]

        # Guard 1: a hop must name a target. Guard 2: a hop must have somewhere
        # to search. Either failing means the honest move is to answer with
        # what is already in hand.
        if action == "retrieve" and hop > 0:
            if not is_concrete(missing):
                logger.info("hop %d downgraded to answer: missing=%r not concrete", hop, missing)
                action, queries = "answer", []
            elif not queries:
                logger.info("hop %d downgraded to answer: no queries supplied", hop)
                action = "answer"

        acc = dict(state.get("usage") or {})
        for key in ("input_tokens", "output_tokens"):
            acc[key] = acc.get(key, 0) + int(usage.get(key, 0) or 0)
        acc["calls"] = acc.get("calls", 0) + 1

        record = {
            "hop": hop,
            "action": action,
            "reason": str(decision.get("reason") or "")[:200],
            "queries": queries,
            "missing": missing,
            "n_docs_seen": len(docs),
        }
        return {
            "action": action,
            "reason": record["reason"],
            "queries": queries,
            "missing": missing,
            "usage": acc,
            "trace": (state.get("trace") or []) + [record],
        }

    def _retrieve_node(self, state: TurnState) -> dict:
        hop = state.get("hop", 0)
        if hop == 0:
            result = self._initial_retrieve()
        else:
            result = self._hop_retrieve(state.get("queries") or [], state.get("docs") or [])

        return {
            "docs": result.get("docs") or [],
            "sub_queries": result.get("sub_queries") or state.get("sub_queries") or [],
            "retrieval": result,
            "hop": hop + 1,
        }

    # -- edges ---------------------------------------------------------------

    def _route(self, state: TurnState) -> str:
        action = state.get("action")
        if action in ("answer", "refuse"):
            return END
        # The cap is enforced here, on the edge, so no prompt wording and no
        # malformed response can produce a fourth pass.
        if state.get("hop", 0) >= self.max_hops:
            logger.info("hop cap %d reached; answering with what is retrieved", self.max_hops)
            return END
        return "retrieve"

    def _build(self):
        graph = StateGraph(TurnState)
        graph.add_node("controller", self._controller_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_edge(START, "controller")
        graph.add_conditional_edges("controller", self._route,
                                    {"retrieve": "retrieve", END: END})
        graph.add_edge("retrieve", "controller")
        return graph.compile()

    # -- entry point ---------------------------------------------------------

    def run(self, question: str, chat_history: str) -> TurnState:
        """Returns the settled state. `action` is the verdict the caller acts
        on: "retrieve" can never survive to here — the edge turns an exhausted
        loop into END, and the caller reads `docs` for whatever was gathered.
        """
        final: TurnState = self._compiled.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "hop": 0,
                "docs": [],
                "sub_queries": [],
                "trace": [],
                "usage": {},
            },
            # One controller call and one retrieval per hop, plus the initial
            # pair; generous headroom over max_hops so the cap that fires is
            # ours (logged, with partial results) and not LangGraph's.
            {"recursion_limit": 2 * (GRAPH_MAX_HOPS + 2) + 4},
        )
        return final
