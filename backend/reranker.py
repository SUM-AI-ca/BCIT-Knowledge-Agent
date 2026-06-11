from typing import List

from langchain_core.documents import Document
from google.cloud import discoveryengine_v1 as discoveryengine

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


class VertexRanker:

    def __init__(
            self,
            project: str,
            model: str = "semantic-ranker-default-004",
            location: str = "global",
            ranking_config: str = "default_ranking_config",
    ):
        self.model = model
        self.client = discoveryengine.RankServiceClient()
        self.ranking_config = self.client.ranking_config_path(
            project=project,
            location=location,
            ranking_config=ranking_config,
        )
        print(f"Reranker loaded: {model}")

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
                content=doc.page_content[:1024]
            )
            for i, doc in enumerate(documents)
        ]

        try:
            response = self.client.rank(
                request=discoveryengine.RankRequest(
                    ranking_config=self.ranking_config,
                    model=self.model,
                    top_n=len(records) if return_all else top_k,
                    query=query,
                    records=records,
                )
            )
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
