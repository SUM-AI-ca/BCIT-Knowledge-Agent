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

    def _embed_one(self, text: str, task_type: str) -> List[float]:
        # gemini-embedding-001 accepts exactly 1 instance per request. This
        # used to be expressed as batch_size=1, which langchain-google-vertexai
        # 3.x dropped — its embed() now sends the whole list in one request, so
        # the limit has to be enforced here or a multi-text call 400s.
        return self._client.embed(
            [text],
            embeddings_task_type=task_type,
            dimensions=self.dimensions,
        )[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = [self._embed_one(t, "RETRIEVAL_DOCUMENT") for t in texts]
        return self._normalize(vectors).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._normalize([self._embed_one(text, "RETRIEVAL_QUERY")])[0].tolist()
