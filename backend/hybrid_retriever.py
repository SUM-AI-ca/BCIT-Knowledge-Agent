import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


def preprocess_func(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [token for token in text.split() if token]
    return tokens


class HybridRetriever(BaseRetriever):
    dense_retriever: Any
    bm25_retriever: Any
    alpha: float = 0.5
    top_k: int = 5
    rrf_k: int = 60

    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self,
            dense_retriever: BaseRetriever,
            bm25_retriever: BM25Retriever,
            alpha: float = 0.5,
            top_k: int = 5,
            rrf_k: int = 60,
            **kwargs
    ):
        super().__init__(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            alpha=alpha,
            top_k=top_k,
            rrf_k=rrf_k,
            **kwargs
        )

    def _create_doc_id(self, doc: Document) -> str:
        source = doc.metadata.get("source", "unknown")
        content_snippet = doc.page_content[:200]
        return f"{source}::{content_snippet}"

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            doc_id = self._create_doc_id(doc)
            rrf_score = self._rrf_score(rank)
            weighted_score = self.alpha * rrf_score

            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weighted_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = self._create_doc_id(doc)
            rrf_score = self._rrf_score(rank)
            weighted_score = (1.0 - self.alpha) * rrf_score

            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weighted_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_doc_ids = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        for doc_id, score in sorted_doc_ids[:self.top_k]:
            doc = doc_map[doc_id]
            doc.metadata["fusion_score"] = score
            result.append(doc)

        return result


def create_hybrid_retriever(
        vectorstore,
        documents: List[Document],
        alpha: float = 0.5,
        top_k: int = 5,
        dense_k: int = 10,
        bm25_k: int = 10,
        dense_search_type: str = "mmr",
        dense_fetch_k: int = 50,
        dense_lambda: float = 0.75
) -> HybridRetriever:
    if dense_search_type == "mmr":
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": dense_k,
                "fetch_k": dense_fetch_k,
                "lambda_mult": dense_lambda
            }
        )
    else:
        dense_retriever = vectorstore.as_retriever(
            search_type=dense_search_type,
            search_kwargs={"k": dense_k}
        )

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=preprocess_func
    )
    bm25_retriever.k = bm25_k

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=alpha,
        top_k=top_k
    )

    return hybrid_retriever