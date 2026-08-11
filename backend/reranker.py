from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List

from langchain_core.documents import Document
from google.api_core import exceptions as gapi_exceptions
from google.cloud import discoveryengine_v1 as discoveryengine

# Worth a second attempt on a fresh RPC: a deadline we imposed ourselves, and
# the transient server-side classes. ResourceExhausted is deliberately absent —
# a 429 is quota, and retrying it immediately only deepens the hole.
RETRYABLE = (
    gapi_exceptions.DeadlineExceeded,
    gapi_exceptions.ServiceUnavailable,
    gapi_exceptions.InternalServerError,
    gapi_exceptions.GatewayTimeout,
    gapi_exceptions.Aborted,
)

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def record_title(doc: Document) -> str:
    """Page identity for the Ranking API's `title` field.

    The reranker is the one stage that still scores identity-blind text. BM25
    got page identity in round 2 (BM25_INDEX_AUG) and the dense arm got it in
    the identity-prefixed rebuild (bcit_docs_202606da), but records here are
    built from `page_content` alone — and the chunks that matter most carry no
    identity in their body. Measured on sf3-05: asked about ACIT 2515's exam
    weights, the ranker scored COMP 4870's and COMP 7402's evaluation tables
    0.859 / 0.709 against 0.258 for ACIT 2515's own chunks. Those tables ARE
    what the question describes; only the course name distinguishes them, and
    the ranker never saw it.
    """
    md = doc.metadata
    parts = [md.get("title") or "", md.get("filename_keywords") or "", md.get("category") or ""]
    return " ".join(p for p in parts if p and p != "None")[:200]


class VertexRanker:

    def __init__(
            self,
            project: str,
            model: str = "semantic-ranker-default-004",
            location: str = "global",
            ranking_config: str = "default_ranking_config",
            identity: bool = False,
            timeout_s: float = 0.0,
            attempts: int = 1,
            hedge_after_s: float = 0.0,
            hedge_workers: int = 8,
            hedge_abandon_s: float = 60.0,
    ):
        self.model = model
        self.identity = identity
        # 0 leaves the call unbounded, which is what the GAPIC transport does
        # on its own (`default_timeout=None` for rank).
        self.timeout_s = timeout_s if timeout_s and timeout_s > 0 else None
        self.attempts = max(1, attempts)
        self.hedge_after_s = hedge_after_s if hedge_after_s and hedge_after_s > 0 else None
        self.hedge_abandon_s = hedge_abandon_s if hedge_abandon_s and hedge_abandon_s > 0 else None
        # Only pay for the pool when hedging is actually on.
        self._pool = (
            ThreadPoolExecutor(max_workers=max(2, hedge_workers),
                               thread_name_prefix="rerank-hedge")
            if self.hedge_after_s else None
        )
        self.client = discoveryengine.RankServiceClient()
        self.ranking_config = self.client.ranking_config_path(
            project=project,
            location=location,
            ranking_config=ranking_config,
        )
        deadline = f", deadline {self.timeout_s}s x{self.attempts}" if self.timeout_s else ""
        hedge = f", hedge after {self.hedge_after_s}s" if self.hedge_after_s else ""
        print(f"Reranker loaded: {model}"
              f"{' (identity titles)' if identity else ''}{deadline}{hedge}")

    def _issue(self, request):
        kwargs = {"timeout": self.timeout_s} if self.timeout_s else {}
        return self.client.rank(request=request, **kwargs)

    def _issue_abandonable(self, request):
        """Hedged attempt. Carries its own generous deadline purely so a hung
        request gives its thread back — the winner has already been returned
        by then, so this bound never decides what the turn answers with."""
        kwargs = {"timeout": self.hedge_abandon_s} if self.hedge_abandon_s else {}
        return self.client.rank(request=request, **kwargs)

    def _rank_hedged(self, request):
        """Issue the call; if it has not answered within hedge_after_s, issue
        an identical second one and return whichever lands first.

        Measured against the live API: a stalled request's concurrent twin
        answered in ~0.2s in 3 of 3 trials, because the stall is which backend
        the request landed on, not how busy the service is. Unlike a deadline
        this cannot cost the turn its ranking — both attempts return the same
        kind of answer, so the loser is discarded, not fallen back to.
        """
        inflight = [self._pool.submit(self._issue_abandonable, request)]
        hedged = False
        last_exc = None
        while True:
            if inflight:
                done, not_done = wait(
                    inflight,
                    # Once hedged, there is nothing left to escalate to, so
                    # wait out whichever of the two answers first.
                    timeout=None if hedged else self.hedge_after_s,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    try:
                        return future.result()
                    except Exception as e:
                        # A failed attempt does not settle the call while its
                        # twin is still in flight.
                        last_exc = e
                inflight = list(not_done)

            # Nothing has answered yet. Slow and failed-outright are the same
            # situation from here — no answer in hand, one hedge left to spend.
            if not hedged:
                why = (f"slow past {self.hedge_after_s}s" if inflight
                       else f"failed with {type(last_exc).__name__}")
                print(f"Ranking API {why}, hedging with a second request")
                inflight.append(self._pool.submit(self._issue_abandonable, request))
                hedged = True
                continue
            if not inflight:
                raise last_exc

    @traceable(run_type="retriever", name="vertex_rerank")
    def rerank(
            self,
            query: str,
            documents: List[Document],
            top_k: int = 10,
            return_all: bool = False
    ) -> List[Document]:
        """return_all=True scores and returns every candidate (rank order);
        the caller slices after applying its own selection logic."""

        if not documents:
            return []

        records = [
            discoveryengine.RankingRecord(
                id=str(i),
                content=doc.page_content[:1024],
                **({"title": record_title(doc)} if self.identity else {}),
            )
            for i, doc in enumerate(documents)
        ]

        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model=self.model,
            top_n=len(records) if return_all else top_k,
            query=query,
            records=records,
        )
        issue = self._rank_hedged if self._pool else self._issue

        response = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = issue(request)
                break
            except RETRYABLE as e:
                # A stalled backend is the failure this guards: it repeats
                # within a single turn, so the retry is worth issuing, but it
                # is not worth issuing forever.
                if attempt < self.attempts:
                    print(f"Ranking API {type(e).__name__} "
                          f"(attempt {attempt}/{self.attempts}), retrying: {e}")
                    continue
                print(f"Ranking API {type(e).__name__} after {attempt} attempt(s), "
                      f"using retrieval order: {e}")
                return documents if return_all else documents[:top_k]
            except Exception as e:
                # degrade to retrieval order instead of failing the request
                print(f"Ranking API error, using retrieval order: {e}")
                return documents if return_all else documents[:top_k]

        reranked_docs = []
        for record in response.records:
            doc = documents[int(record.id)]
            doc.metadata["rerank_score"] = float(record.score)
            reranked_docs.append(doc)

        return reranked_docs
