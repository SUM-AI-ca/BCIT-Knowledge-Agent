import os
import sys
from typing import List
from langchain_core.embeddings import Embeddings
from FlagEmbedding import BGEM3FlagModel
import numpy as np


class BGEM3Embeddings(Embeddings):
    def __init__(
            self,
            model_name: str = "BAAI/bge-m3",
            device: str = "cuda",
            use_fp16: bool = True,
            normalize_embeddings: bool = True
    ):

        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.normalize_embeddings = normalize_embeddings

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=self.use_fp16,
                device=device
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=6,
                max_length=1512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        dense_vecs = embeddings['dense_vecs']

        if self.normalize_embeddings:
            dense_vecs = dense_vecs / np.linalg.norm(dense_vecs, axis=1, keepdims=True)

        return dense_vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            embedding = self.model.encode(
                [text],
                batch_size=1,
                max_length=1024,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        dense_vec = embedding['dense_vecs'][0]

        if self.normalize_embeddings:
            dense_vec = dense_vec / np.linalg.norm(dense_vec)

        return dense_vec.tolist()