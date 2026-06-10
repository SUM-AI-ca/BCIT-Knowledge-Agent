from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import VertexAIEmbeddings


class VertexGeminiEmbeddings(Embeddings):
    def __init__(
            self,
            model_name: str = "gemini-embedding-001",
            project: str = None,
            location: str = "us-central1",
            dimensions: int = 1536,
    ):
        self.model_name = model_name
        self.dimensions = dimensions

        self._client = VertexAIEmbeddings(
            model_name=model_name,
            project=project,
            location=location,
        )

    def _normalize(self, vectors: List[List[float]]) -> np.ndarray:
        # API returns unnormalized vectors when dimensions != 3072
        arr = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._client.embed(
            texts,
            batch_size=1,  # gemini-embedding-001 allows 1 instance per request
            embeddings_task_type="RETRIEVAL_DOCUMENT",
            dimensions=self.dimensions,
        )
        return self._normalize(vectors).tolist()

    def embed_query(self, text: str) -> List[float]:
        vectors = self._client.embed(
            [text],
            batch_size=1,
            embeddings_task_type="RETRIEVAL_QUERY",
            dimensions=self.dimensions,
        )
        return self._normalize(vectors)[0].tolist()
