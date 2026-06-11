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

    def _dense_search(self, query: str, dense_k: int = None) -> List[Document]:
        if dense_k is None:
            return self.dense_retriever.invoke(query)
        # Per-call k: go through the underlying vectorstore with the same MMR
        # parameters the configured retriever uses.
        kwargs = dict(self.dense_retriever.search_kwargs)
        kwargs["k"] = dense_k
        return self.dense_retriever.vectorstore.max_marginal_relevance_search(
            query, **kwargs
        )

    def _bm25_search(self, query: str, bm25_k: int = None) -> List[Document]:
        if bm25_k is None:
            return self.bm25_retriever.invoke(query)
        # Per-call k: score directly against the (read-only, thread-safe)
        # BM25 index instead of mutating the shared retriever's k.
        tokens = self.bm25_retriever.preprocess_func(query)
        scores = self.bm25_retriever.vectorizer.get_scores(tokens)
        if len(scores) <= bm25_k:
            top = range(len(scores))
        else:
            import numpy as np
            part = np.argpartition(scores, -bm25_k)[-bm25_k:]
            top = part[np.argsort(scores[part])[::-1]]
        docs = self.bm25_retriever.docs
        return [docs[i] for i in top]

    def search(
            self,
            query: str,
            dense_k: int = None,
            bm25_k: int = None,
            top_k: int = None,
    ) -> List[Document]:
        """RRF-fused hybrid search with per-call k overrides.

        Returns COPIES (fresh metadata dicts) — BM25 hands out Documents that
        alias the shared pickle corpus, and downstream steps annotate
        metadata, which must never leak back into the corpus (or race when
        sub-queries run in parallel).
        """
        dense_docs = self._dense_search(query, dense_k)
        bm25_docs = self._bm25_search(query, bm25_k)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for weight, ranked_docs in (
            (self.alpha, dense_docs),
            (1.0 - self.alpha, bm25_docs),
        ):
            for rank, doc in enumerate(ranked_docs, start=1):
                doc_id = self._create_doc_id(doc)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight * self._rrf_score(rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        sorted_doc_ids = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        for doc_id, score in sorted_doc_ids[:(top_k or self.top_k)]:
            doc = doc_map[doc_id]
            copy = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
            )
            copy.metadata["fusion_score"] = score
            result.append(copy)

        return result

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self.search(query)


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