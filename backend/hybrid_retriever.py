import hashlib
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi


def preprocess_func(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [token for token in text.split() if token]
    return tokens


def doc_key(doc: Document, full: bool = False) -> str:
    """THE identity of a chunk for deduplication and RRF fusion.

    One helper because three call sites used to build this string
    independently (the fusion map, the sub-query pool, and now the scoped
    arm); if they ever disagree, a chunk gets merged by one and kept by
    another, and which copy survives depends on call order.

    `full=False` is the historical key: source + the first 200 chars. It
    collides on this corpus for 418 keys covering 715 chunks (0.71%) — two
    DIFFERENT chunks of the same page that open identically (repeated tables in
    course pages, the English-proficiency assessment table) merge, and the
    loser's content never reaches the context. `full=True` keys on the whole
    chunk and cannot do that.
    """
    source = doc.metadata.get("source", "unknown")
    if full:
        return f"{source}::{hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()}"
    return f"{source}::{doc.page_content[:200]}"


def bm25_augment_text(doc: Document) -> str:
    """Index-time text for BM25_INDEX_AUG: prepend the parent page's identity
    (title, category, filename keywords, URL slug) so deep chunks — section
    bodies that never mention their own program — stay findable by
    program-qualified keyword queries. page_content itself is NEVER modified
    (neighbor-index md5 keys and pool dedup keys depend on it)."""
    md = doc.metadata
    url = (md.get("url") or "").lower()
    slug = re.sub(r"[^a-z0-9]+", " ", url.split("bcit.ca", 1)[-1]) if url else ""
    parts = [
        md.get("title") or "",
        md.get("category") or "",
        md.get("filename_keywords") or "",
        slug,
        doc.page_content,
    ]
    return " ".join(p for p in parts if p)


class HybridRetriever(BaseRetriever):
    dense_retriever: Any
    bm25_retriever: Any
    alpha: float = 0.5
    top_k: int = 5
    rrf_k: int = 60
    # Document frequency per token in the (augmented) BM25 index, used to drop
    # corpus-generic tokens from a query. Empty dict = feature off.
    token_df: dict = {}
    n_docs: int = 0
    stopword_df: float = 0.0
    # source path -> the indices its chunks occupy in bm25_retriever.docs,
    # for entity-scoped scoring.
    source_indices: dict = {}
    dedup_full: bool = False

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

    def _filter_query_tokens(self, tokens: List[str]) -> List[str]:
        """Drop tokens this corpus says carry no discrimination.

        BM25_INDEX_AUG stamps title/category/filename keywords onto every
        chunk, so a rewritten sub-query carrying generic corpus vocabulary
        ("BCIT courses with ...") matches 40-60% of the index with a small
        positive weight and reorders the tail. Never returns empty — an
        all-generic query keeps its original tokens.
        """
        if not (self.stopword_df > 0 and self.token_df and self.n_docs):
            return tokens
        cutoff = self.stopword_df * self.n_docs
        kept = [t for t in tokens if self.token_df.get(t, 0) < cutoff]
        return kept or tokens

    def scoped_search(self, query: str, sources, k: int) -> List[Document]:
        """BM25 over ONE entity's own chunks, using the global IDF.

        For a chunk whose body carries no entity identity and whose wording is
        shared corpus-wide (outline evaluation tables, program page sections),
        no global ranking can surface it — it is not out-ranked, it is
        indistinguishable from thousands of siblings. Restricted to the named
        entity's own chunks it is trivially top-ranked.

        Scores the subset directly rather than calling get_scores() over the
        whole index (128 ms per call here, and this runs once per entity).
        """
        bm = self.bm25_retriever.vectorizer
        docs = self.bm25_retriever.docs
        per_source = [
            (s, self.source_indices.get(s, ()))
            for s in sources if self.source_indices.get(s)
        ]
        if not per_source:
            return []
        # Deliberately NOT _filter_query_tokens: that filter exists to stop a
        # query from matching 40-60% of the INDEX, which cannot happen when the
        # candidate set is one document. Inside those chunks the "generic"
        # words are the discriminating ones — filtering them made every chunk
        # of the entity tie on its own name (the augmented text repeats the
        # code on every chunk) and sf3-05 regressed from 1.00 to 0.00.
        tokens = self.bm25_retriever.preprocess_func(query)

        def score_one(i):
            freqs = bm.doc_freqs[i]
            norm = bm.k1 * (1 - bm.b + bm.b * bm.doc_len[i] / bm.avgdl)
            total = 0.0
            for t in tokens:
                f = freqs.get(t, 0)
                if f:
                    total += bm.idf.get(t, 0.0) * f * (bm.k1 + 1) / (f + norm)
            return total

        # k is per SOURCE, not per entity. A course entity resolves to two
        # sources (its outline and its catalogue page) and the short catalogue
        # page wins BM25 length normalisation on every chunk, so a shared
        # budget spent all of it there: ACIT 2515's evaluation table sits at
        # combined rank 12 (cut at k=8) but rank 5 within its own outline.
        ranked_per_source = []
        for _, indices in per_source:
            scored = sorted(((score_one(i), i) for i in indices), key=lambda x: -x[0])
            ranked_per_source.append(scored[:k])

        # Round-robin so every source of the entity contributes its best first.
        out = []
        for depth in range(k):
            for scored in ranked_per_source:
                if depth >= len(scored):
                    continue
                score, i = scored[depth]
                doc = docs[i]
                copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                copy.metadata["scoped_rank"] = len(out) + 1
                copy.metadata["scoped_score"] = score
                # Same RRF scale as the fused arms, so the pool can order it.
                copy.metadata["fusion_score"] = self._rrf_score(len(out) + 1)
                out.append(copy)
        return out

    def _create_doc_id(self, doc: Document) -> str:
        return doc_key(doc, self.dedup_full)

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
        if bm25_k is None and not self.stopword_df:
            return self.bm25_retriever.invoke(query)
        bm25_k = bm25_k if bm25_k is not None else self.bm25_retriever.k
        # Per-call k: score directly against the (read-only, thread-safe)
        # BM25 index instead of mutating the shared retriever's k.
        tokens = self._filter_query_tokens(self.bm25_retriever.preprocess_func(query))
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
            bm25_query: str = None,
            dense_query: str = None,
    ) -> List[Document]:
        """RRF-fused hybrid search with per-call k and per-arm query overrides
        (bm25_query: keyword-expanded terms; dense_query: e.g. HyDE passage).

        Returns COPIES (fresh metadata dicts) — BM25 hands out Documents that
        alias the shared pickle corpus, and downstream steps annotate
        metadata, which must never leak back into the corpus (or race when
        sub-queries run in parallel).
        """
        dense_docs = self._dense_search(dense_query or query, dense_k)
        bm25_docs = self._bm25_search(bm25_query or query, bm25_k)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        arm_ranks: Dict[str, dict] = {}

        for arm, weight, ranked_docs in (
            ("dense_rank", self.alpha, dense_docs),
            ("bm25_rank", 1.0 - self.alpha, bm25_docs),
        ):
            for rank, doc in enumerate(ranked_docs, start=1):
                doc_id = self._create_doc_id(doc)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight * self._rrf_score(rank)
                arm_ranks.setdefault(doc_id, {}).setdefault(arm, rank)
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
            # Which arm(s) surfaced the doc — consensus signal for the
            # rerank-skip experiment; inert metadata otherwise.
            copy.metadata.update(arm_ranks.get(doc_id, {}))
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
        dense_lambda: float = 0.75,
        rrf_k: int = 60,
        bm25_index_aug: bool = False,
        stopword_df: float = 0.0,
        scoped: bool = False,
        dedup_full: bool = False,
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

    corpus = None
    if bm25_index_aug:
        # Fit the index on augmented text while serving the ORIGINAL
        # documents: scores come from title-aware tokens, but everything
        # downstream (dedup keys, neighbor lookups, context) sees the same
        # page_content as before. BM25Retriever's own from_texts ends with
        # exactly this constructor call.
        corpus = [preprocess_func(bm25_augment_text(d)) for d in documents]
        bm25_retriever = BM25Retriever(
            vectorizer=BM25Okapi(corpus),
            docs=list(documents),
            preprocess_func=preprocess_func
        )
    else:
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=preprocess_func
        )
    bm25_retriever.k = bm25_k

    # Document frequency over the SAME text the index was fit on (augmented if
    # BM25_INDEX_AUG, raw otherwise) — a stopword list derived from the corpus
    # instead of maintained by hand.
    token_df = {}
    if stopword_df > 0:
        if corpus is None:
            corpus = [preprocess_func(d.page_content) for d in documents]
        for tokens in corpus:
            for token in set(tokens):
                token_df[token] = token_df.get(token, 0) + 1

    # source -> chunk positions, for entity-scoped scoring. Positions index
    # bm25_retriever.docs, which is `documents` in order.
    source_indices = {}
    if scoped:
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source")
            if source:
                source_indices.setdefault(source, []).append(i)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=alpha,
        top_k=top_k,
        rrf_k=rrf_k,
        token_df=token_df,
        n_docs=len(documents),
        stopword_df=stopword_df,
        source_indices=source_indices,
        dedup_full=dedup_full,
    )

    return hybrid_retriever