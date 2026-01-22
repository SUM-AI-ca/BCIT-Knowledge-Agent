from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            device: str = "cuda"
    ):
        self.model = CrossEncoder(model_name, device=device)
        print(f"Reranker loaded: {model_name}")

    def rerank(
            self,
            query: str,
            documents: List[Document],
            top_k: int = 10
    ) -> List[Document]:

        if not documents:
            return []

        pairs = [[query, doc.page_content[:1024]] for doc in documents]

        scores = self.model.predict(pairs)

        doc_score_pairs = list(zip(documents, scores))

        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)

        return reranked_docs