import warnings
import pickle

warnings.filterwarnings('ignore')

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Set

from langchain_postgres import PGVector
from langchain_google_vertexai import ChatVertexAI
from sqlalchemy import create_engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from embeddings import VertexGeminiEmbeddings
from reranker import VertexRanker
from hybrid_retriever import create_hybrid_retriever
from config import (
    DOCUMENTS_PICKLE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_LOCATION,
    PG_CONNECTION,
    PG_COLLECTION,
    HNSW_EF_SEARCH,
    GEMINI_MODEL,
    REWRITER_MODEL,
    GEMINI_PROJECT,
    GEMINI_LOCATION,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    USE_HYBRID_SEARCH,
    HYBRID_ALPHA,
    RETRIEVAL_TOP_K,
    RETRIEVAL_DENSE_K,
    RETRIEVAL_BM25_K,
    RETRIEVAL_FETCH_K,
    MMR_LAMBDA,
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    USE_RERANKING,
    RERANKER_MODEL,
    RANKING_LOCATION,
    RANKING_CONFIG,
    RERANKER_CANDIDATES,
    RERANKER_TOP_K,
    MEMORY_WINDOW_K,
    PRICE_GEN_INPUT_PER_M,
    PRICE_GEN_OUTPUT_PER_M,
    PRICE_REWRITE_INPUT_PER_M,
    PRICE_REWRITE_OUTPUT_PER_M,
    PRICE_RERANK_PER_CALL,
    PRICE_EMBED_PER_QUERY,
    CONTEXT_MODE,
    NEIGHBOR_RADIUS,
    CONTEXT_MAX_CHARS,
    RERANK_SCORE_THRESHOLD,
    MIN_CONTEXT_CHUNKS,
    CHUNK_OVERLAP,
    MULTI_QUERY_ENABLED,
    MAX_SUB_QUERIES,
    REWRITE_MAX_OUTPUT_TOKENS,
    REWRITE_THINKING_BUDGET,
    REWRITE_DECOMPOSE_TEMPLATE,
    REWRITE_DECOMPOSE_SCHEMA,
    GEMINI_THINKING_BUDGET,
    MQ_DENSE_K,
    MQ_BM25_K,
    MQ_CANDIDATES_PER_SUBQUERY,
    MQ_POOL_CAP,
    MQ_MIN_CHUNKS_PER_SUBQUERY,
    RERANK_MODE,
    STRIP_SOURCES_FROM_HISTORY,
    HISTORY_MAX_ANSWER_CHARS,
)

logger = logging.getLogger("bcit.rag")

EASTER_EGGS = {
    "WHO IS THE BEST INSTRUCTOR AT BCIT": "Chi En Huang",
    "WHO IS THE BEST INSTRUCTOR AT BCIT?": "Chi En Huang"
}


def _message_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return str(content)


class BCITChatbot:

    def __init__(self):
        print("BCIT ADVISOR CHATBOT")

        self._validate_requirements()
        self._load_embeddings()
        self._load_vectorstore()
        self._load_documents()
        self._initialize_llm()
        self._initialize_memory()
        self._initialize_reranker()
        self._setup_retriever()
        self._create_prompts()

        print("\nCommands: 'quit', 'exit', 'q' to exit\n")

    def _validate_requirements(self):

        if USE_HYBRID_SEARCH and not DOCUMENTS_PICKLE.exists():
            raise FileNotFoundError(f"Documents pickle not found: {DOCUMENTS_PICKLE}")

        print("Requirements validated\n")

    def _load_embeddings(self):
        self.embeddings = VertexGeminiEmbeddings(
            model_name=EMBEDDING_MODEL,
            project=GEMINI_PROJECT,
            location=EMBEDDING_LOCATION,
            dimensions=EMBEDDING_DIMENSIONS
        )
        print("Embeddings loaded\n")

    def _load_vectorstore(self):
        engine = create_engine(
            PG_CONNECTION,
            pool_pre_ping=True,
            connect_args={"options": f"-c hnsw.ef_search={HNSW_EF_SEARCH}"}
        )
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=PG_COLLECTION,
            connection=engine,
            use_jsonb=True,
            embedding_length=EMBEDDING_DIMENSIONS,
            create_extension=False
        )
        print(f"Vectorstore connected (collection: {PG_COLLECTION})\n")

    def _load_documents(self):
        if USE_HYBRID_SEARCH:
            with open(DOCUMENTS_PICKLE, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents):,} documents\n")
        else:
            self.documents = None
        self._neighbor_index = (
            self._build_neighbor_index(self.documents) if self.documents else None
        )

    @staticmethod
    def _chunk_key(source: str, page_content: str) -> tuple:
        return source, hashlib.md5(page_content.encode("utf-8")).hexdigest()

    def _build_neighbor_index(self, documents) -> dict:
        # The pickle stores chunks per source in document order (the splitter
        # emits them sequentially in build_pgvector.py), so a chunk's position
        # in its source doubles as its ordinal for neighbor lookups. Dense
        # results are distinct objects deserialized from pgvector, so they are
        # matched by content hash, not identity.
        chunks_by_source = {}
        ordinals = {}
        for doc in documents:
            source = doc.metadata.get("source")
            if not source:
                continue
            chunks = chunks_by_source.setdefault(source, [])
            key = self._chunk_key(source, doc.page_content)
            if key not in ordinals:
                ordinals[key] = len(chunks)
            chunks.append(doc.page_content)
        return {"chunks": chunks_by_source, "ordinals": ordinals}

    def _initialize_llm(self):
        llm_kwargs = dict(
            model=GEMINI_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        )
        if GEMINI_THINKING_BUDGET is not None:
            llm_kwargs["thinking_budget"] = GEMINI_THINKING_BUDGET
        self.llm = ChatVertexAI(**llm_kwargs)

        # Dedicated rewriter: deterministic, schema-constrained JSON, no
        # thinking budget — a cheap fixed-shape call that runs every turn.
        self.rewriter_llm = ChatVertexAI(
            model=REWRITER_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            temperature=0.0,
            max_output_tokens=REWRITE_MAX_OUTPUT_TOKENS,
            thinking_budget=REWRITE_THINKING_BUDGET,
            response_mime_type="application/json",
            response_schema=REWRITE_DECOMPOSE_SCHEMA,
        )
        print("LLM initialized\n")

    def _initialize_memory(self):
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_K,
            memory_key="chat_history",
            return_messages=True
        )
        print("Memory initialized")

    def _initialize_reranker(self):
        if USE_RERANKING:
            self.reranker = VertexRanker(
                project=GEMINI_PROJECT,
                model=RERANKER_MODEL,
                location=RANKING_LOCATION,
                ranking_config=RANKING_CONFIG
            )
            print("Reranker initialized")
        else:
            self.reranker = None

    def _setup_retriever(self):
        if USE_HYBRID_SEARCH and self.documents:
            if USE_RERANKING:
                print("hybrid search + reranking")
            else:
                print("hybrid search")

            if USE_RERANKING:
                retriever_top_k = RERANKER_CANDIDATES
            else:
                retriever_top_k = RETRIEVAL_TOP_K

            self.base_retriever = create_hybrid_retriever(
                vectorstore=self.vectorstore,
                documents=self.documents,
                alpha=HYBRID_ALPHA,
                top_k=retriever_top_k,
                dense_k=RETRIEVAL_DENSE_K,
                bm25_k=RETRIEVAL_BM25_K,
                dense_search_type="mmr",
                dense_fetch_k=RETRIEVAL_FETCH_K,
                dense_lambda=MMR_LAMBDA
            )
            # Shared fan-out pool for sub-query retrieval: 2 concurrent
            # requests x 4 sub-queries fits SQLAlchemy's default pool (5+10).
            self._retrieval_pool = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="subq"
            )
        else:
            print("dense only")
            self.base_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVAL_TOP_K,
                    "fetch_k": RETRIEVAL_FETCH_K,
                    "lambda_mult": MMR_LAMBDA
                }
            )
            self._retrieval_pool = None

    def _create_prompts(self):
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)
        self.rewrite_decompose_prompt = ChatPromptTemplate.from_template(
            REWRITE_DECOMPOSE_TEMPLATE
        ).partial(max_sub_queries=str(MAX_SUB_QUERIES))

    def _history_view(self, answer: str) -> str:
        """What gets saved to memory — and re-sent in every later prompt.

        The Sources section is citation plumbing the follow-up rewriter never
        needs, and uncapped answers compound across the k-turn window.
        """
        text = answer
        if STRIP_SOURCES_FROM_HISTORY:
            text = re.sub(
                r"\n[ \t*#]*Sources[ \t*#]*:?.*\Z", "", text,
                flags=re.IGNORECASE | re.DOTALL,
            ).rstrip()
        if len(text) > HISTORY_MAX_ANSWER_CHARS:
            text = text[:HISTORY_MAX_ANSWER_CHARS].rstrip() + " ..."
        return text or answer[:HISTORY_MAX_ANSWER_CHARS]

    def _format_chat_history(self, memory: ConversationBufferWindowMemory) -> str:
        # buffer_as_messages applies the k-window; chat_memory.messages is the
        # raw unbounded list and must not be used here.
        messages = memory.buffer_as_messages
        if not messages:
            return "No previous conversation."

        formatted = []
        for msg in messages:
            role = "Student" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def _rewrite_query(self, question: str, chat_history: str) -> tuple:
        if chat_history == "No previous conversation.":
            return question, {}

        try:
            prompt_value = self.rewrite_prompt.invoke({
                "chat_history": chat_history,
                "question": question
            })
            msg = self.llm.invoke(prompt_value)
            rewritten = _message_text(msg).strip()
            if not rewritten:
                return question, {}

            if rewritten != question.strip():
                logger.info("query rewritten: %r -> %r", question, rewritten)

            return rewritten, dict(msg.usage_metadata or {})
        except Exception:
            return question, {}

    def _rewrite_and_decompose(self, question: str, chat_history: str) -> tuple:
        """One JSON call: standalone rewrite + sub-query decomposition.

        Runs every turn (a first-turn multipart question still needs
        decomposing). Returns (standalone, sub_queries, usage, fallback).
        """
        try:
            prompt_value = self.rewrite_decompose_prompt.invoke({
                "chat_history": chat_history,
                "question": question
            })
            msg = self.rewriter_llm.invoke(prompt_value)
            data = json.loads(_message_text(msg))

            standalone = (data.get("standalone_question") or "").strip() or question
            sub_queries = [
                q.strip() for q in (data.get("sub_queries") or [])
                if isinstance(q, str) and q.strip()
            ][:MAX_SUB_QUERIES] or [standalone]

            if standalone != question.strip():
                logger.info("query rewritten: %r -> %r", question, standalone)
            if len(sub_queries) > 1:
                logger.info("decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

            return standalone, sub_queries, dict(msg.usage_metadata or {}), False
        except Exception:
            logger.warning("rewrite+decompose failed, using raw question", exc_info=True)
            return question, [question], {}, True

    def _load_full_document(self, source_path):
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""

    def _format_docs_full(self, docs: List[Document]):
        unique_sources: Set[str] = set()
        source_to_metadata = {}

        for doc in docs:
            source = doc.metadata.get("source")
            if source:
                unique_sources.add(source)
                if source not in source_to_metadata:
                    source_to_metadata[source] = doc.metadata

        formatted = []
        filenames = []
        for i, source_path in enumerate(sorted(unique_sources), 1):
            metadata = source_to_metadata.get(source_path, {})
            full_content = self._load_full_document(source_path)

            if not full_content:
                continue

            filename = metadata.get('filename', 'N/A')
            filenames.append(filename)

            source_info = ""
            if metadata.get("url"):
                source_info = f" [URL: {metadata['url']}]"
            elif metadata.get("title"):
                source_info = f" [Title: {metadata['title']}]"

            formatted.append(
                f"Document {i}:{source_info}\n"
                f"Filename: {filename}\n"
                f"Category: {metadata.get('category', 'N/A')}\n\n"
                f"{full_content}"
            )

        logger.debug("full-doc context sources: %s", ", ".join(filenames))

        return "\n\n" + "-" * 60 + "\n\n".join(formatted)

    def _apply_score_threshold(self, docs: List[Document]) -> List[Document]:
        if RERANK_SCORE_THRESHOLD <= 0 or not docs:
            return docs
        if any("rerank_score" not in doc.metadata for doc in docs):
            # reranker fallback path returns unscored docs in retrieval order
            return docs
        kept = [d for d in docs if d.metadata["rerank_score"] >= RERANK_SCORE_THRESHOLD]
        if len(kept) < MIN_CONTEXT_CHUNKS:
            kept = docs[:MIN_CONTEXT_CHUNKS]  # docs arrive in rerank order
        return kept

    def _merge_chunk_run(self, texts: List[str]) -> str:
        # Consecutive chunks overlap by up to CHUNK_OVERLAP chars (less at
        # split boundaries); drop the duplicated span when it matches exactly,
        # otherwise just join.
        merged = texts[0]
        for text in texts[1:]:
            joined = False
            for k in range(min(len(merged), len(text), CHUNK_OVERLAP + 70), 9, -1):
                if merged[-k:] == text[:k]:
                    merged += text[k:]
                    joined = True
                    break
            if not joined:
                merged += "\n" + text
        return merged

    def _source_text(self, source: str, entry: dict, radius: int) -> str:
        index = self._neighbor_index or {"chunks": {}, "ordinals": {}}
        chunks = index["chunks"].get(source, [])
        texts = []
        if entry["ordinals"]:
            wanted = set()
            for ordinal in entry["ordinals"]:
                for j in range(ordinal - radius, ordinal + radius + 1):
                    if 0 <= j < len(chunks):
                        wanted.add(j)
            run = []
            for j in sorted(wanted):
                if run and j != run[-1] + 1:
                    texts.append(self._merge_chunk_run([chunks[i] for i in run]))
                    run = []
                run.append(j)
            if run:
                texts.append(self._merge_chunk_run([chunks[i] for i in run]))
        texts.extend(entry["loose"])
        return "\n[...]\n".join(texts)

    def _expand_and_format_chunks(self, docs: List[Document]) -> tuple:
        docs = self._apply_score_threshold(docs)
        index = self._neighbor_index or {"chunks": {}, "ordinals": {}}

        order = []
        grouped = {}
        neighbor_misses = 0
        for doc in docs:
            source = doc.metadata.get("source")
            if not source:
                continue
            if source not in grouped:
                grouped[source] = {"metadata": doc.metadata, "ordinals": [], "loose": []}
                order.append(source)
            key = self._chunk_key(source, doc.page_content)
            ordinal = index["ordinals"].get(key)
            if ordinal is None:
                neighbor_misses += 1
                grouped[source]["loose"].append(doc.page_content)
            elif ordinal not in grouped[source]["ordinals"]:
                grouped[source]["ordinals"].append(ordinal)

        def render(radii, sources):
            blocks = []
            for i, source in enumerate(sources, 1):
                metadata = grouped[source]["metadata"]
                source_info = ""
                if metadata.get("url"):
                    source_info = f" [URL: {metadata['url']}]"
                elif metadata.get("title"):
                    source_info = f" [Title: {metadata['title']}]"
                blocks.append(
                    f"Document {i}:{source_info}\n"
                    f"Filename: {metadata.get('filename', 'N/A')}\n"
                    f"Category: {metadata.get('category', 'N/A')}\n\n"
                    f"{self._source_text(source, grouped[source], radii[source])}"
                )
            return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(blocks)

        radii = {source: NEIGHBOR_RADIUS for source in order}
        sources = list(order)
        context = render(radii, sources)
        # Over budget: first strip neighbor expansion from the lowest-ranked
        # sources, then drop trailing sources entirely.
        while len(context) > CONTEXT_MAX_CHARS:
            shrinkable = [s for s in reversed(sources) if radii[s] > 0]
            if shrinkable:
                radii[shrinkable[0]] = 0
            elif len(sources) > 1:
                sources.pop()
            else:
                context = context[:CONTEXT_MAX_CHARS]
                break
            context = render(radii, sources)

        stats = {
            "n_chunks_kept": len(docs),
            "n_context_sources": len(sources),
            "neighbor_misses": neighbor_misses,
            "context_chars": len(context),
        }
        return context, stats

    def _multi_query_retrieve(self, standalone_question: str, sub_queries: List[str]) -> tuple:
        """Fan out retrieval per sub-query, merge into one deduped pool, then
        rerank once. Returns (docs, stats)."""
        t0 = time.perf_counter()
        futures = [
            self._retrieval_pool.submit(
                self.base_retriever.search,
                q,
                dense_k=MQ_DENSE_K,
                bm25_k=MQ_BM25_K,
                top_k=MQ_CANDIDATES_PER_SUBQUERY,
            )
            for q in sub_queries
        ]
        per_sub = [f.result() for f in futures]

        # Interleave by rank so the pool cap cuts fairly across sub-queries.
        # Dedupe keeps the first copy and records every sub-query that
        # surfaced the chunk (for the coverage quota below).
        pooled: List[Document] = []
        by_key = {}
        origins = {}  # id(doc) -> set of sub-query indices
        for rank in range(max((len(d) for d in per_sub), default=0)):
            for si, docs_i in enumerate(per_sub):
                if rank >= len(docs_i):
                    continue
                doc = docs_i[rank]
                key = f"{doc.metadata.get('source', 'unknown')}::{doc.page_content[:200]}"
                if key in by_key:
                    origins[id(by_key[key])].add(si)
                elif len(pooled) < MQ_POOL_CAP:
                    by_key[key] = doc
                    origins[id(doc)] = {si}
                    pooled.append(doc)
        t_retrieve = time.perf_counter() - t0

        t0 = time.perf_counter()
        docs = self._select_from_pool(standalone_question, sub_queries, pooled, origins)
        t_rerank = time.perf_counter() - t0

        return docs, {
            "retrieve_s": t_retrieve,
            "rerank_s": t_rerank,
            "n_candidates": len(pooled),
        }

    def _select_from_pool(
            self,
            question: str,
            sub_queries: List[str],
            pooled: List[Document],
            origins: dict,
    ) -> List[Document]:
        if not (USE_RERANKING and self.reranker):
            return pooled[:RERANKER_TOP_K]  # interleaved order ≈ even coverage

        if RERANK_MODE == "per_subquery":
            # One Ranking API call per sub-query (costs n_subqueries x the
            # pooled mode's fixed per-query fee).
            per_share = max(MQ_MIN_CHUNKS_PER_SUBQUERY, -(-RERANKER_TOP_K // len(sub_queries)))
            selected, seen = [], set()
            for si, sub_query in enumerate(sub_queries):
                candidates = [d for d in pooled if si in origins[id(d)]]
                for doc in self.reranker.rerank(sub_query, candidates, top_k=per_share):
                    if id(doc) not in seen:
                        seen.add(id(doc))
                        selected.append(doc)
            selected.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
            return selected[:max(RERANKER_TOP_K, MQ_MIN_CHUNKS_PER_SUBQUERY * len(sub_queries))]

        # Default "pooled": ONE Ranking API call scoring the whole pool.
        scored = self.reranker.rerank(question, pooled, top_k=RERANKER_TOP_K, return_all=True)
        if any("rerank_score" not in d.metadata for d in scored):
            return scored[:RERANKER_TOP_K]  # API fallback: interleaved order

        selected = scored[:RERANKER_TOP_K]
        rest = scored[RERANKER_TOP_K:]

        def covered(si, sel):
            return sum(1 for d in sel if si in origins[id(d)])

        # Coverage quota: every sub-query keeps >= MQ_MIN_CHUNKS_PER_SUBQUERY
        # of its own candidates; swap in its best leftovers, evicting the
        # lowest-scored doc whose sub-queries all stay above quota.
        for si in range(len(sub_queries)):
            deficit = MQ_MIN_CHUNKS_PER_SUBQUERY - covered(si, selected)
            if deficit <= 0:
                continue
            for candidate in [d for d in rest if si in origins[id(d)]][:deficit]:
                evict = next(
                    (d for d in reversed(selected)
                     if all(covered(sj, selected) > MQ_MIN_CHUNKS_PER_SUBQUERY
                            for sj in origins[id(d)])),
                    None,
                )
                if evict is not None:
                    selected.remove(evict)
                selected.append(candidate)
                rest.remove(candidate)

        selected.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
        return selected

    @traceable(name="bcit_query")
    def query_with_meta(self, question: str, memory: Optional[ConversationBufferWindowMemory] = None) -> dict:
        if memory is None:
            memory = self.memory

        normalized = question.strip().upper()
        if normalized in EASTER_EGGS:
            answer = EASTER_EGGS[normalized]
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return {
                "answer": answer,
                "docs": [],
                "standalone_question": question,
                "finish_reason": "",
                "usage": {},
                "est_cost_usd": 0.0,
                "timings": {},
                "n_context_docs": 0,
            }

        t_start = time.perf_counter()
        chat_history = self._format_chat_history(memory)

        t0 = time.perf_counter()
        if MULTI_QUERY_ENABLED:
            standalone_question, sub_queries, rewrite_usage, decompose_fallback = (
                self._rewrite_and_decompose(question, chat_history)
            )
        else:
            standalone_question, rewrite_usage = self._rewrite_query(question, chat_history)
            sub_queries, decompose_fallback = [standalone_question], False
        t_rewrite = time.perf_counter() - t0

        # Single-question turns keep the full-width legacy retrieval; only
        # genuinely multipart turns fan out per sub-query.
        use_fanout = (
            len(sub_queries) > 1
            and self._retrieval_pool is not None
            and hasattr(self.base_retriever, "search")
        )
        if use_fanout:
            docs, mq_stats = self._multi_query_retrieve(standalone_question, sub_queries)
            t_retrieve = mq_stats["retrieve_s"]
            t_rerank = mq_stats["rerank_s"]
            n_candidates = mq_stats["n_candidates"]
            n_rerank_calls = 0
            if USE_RERANKING and self.reranker:
                n_rerank_calls = len(sub_queries) if RERANK_MODE == "per_subquery" else 1
        else:
            t0 = time.perf_counter()
            docs = self.base_retriever.invoke(standalone_question)
            t_retrieve = time.perf_counter() - t0

            t0 = time.perf_counter()
            n_candidates = len(docs)
            n_rerank_calls = 0
            if USE_RERANKING and self.reranker:
                docs = self.reranker.rerank(
                    query=standalone_question,
                    documents=docs,
                    top_k=RERANKER_TOP_K
                )
                n_rerank_calls = 1
            t_rerank = time.perf_counter() - t0

        if CONTEXT_MODE == "chunks":
            context, context_stats = self._expand_and_format_chunks(docs)
        else:
            context = self._format_docs_full(docs)
            context_stats = {"context_chars": len(context)}

        question_parts = ""
        if len(sub_queries) > 1:
            question_parts = (
                "\n\nThe question has multiple parts; answer each one:\n"
                + "\n".join(f"  {i}) {q}" for i, q in enumerate(sub_queries, 1))
            )

        t0 = time.perf_counter()
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "question_parts": question_parts,
            "chat_history": chat_history
        })
        msg = self.llm.invoke(prompt_value)
        t_generate = time.perf_counter() - t0

        answer = _message_text(msg)
        usage = dict(msg.usage_metadata or {})
        finish_reason = str((msg.response_metadata or {}).get("finish_reason", ""))
        if finish_reason == "MAX_TOKENS":
            logger.warning("answer truncated (finish_reason=MAX_TOKENS) — Sources section likely lost")

        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(self._history_view(answer))

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        details_in = usage.get("input_token_details") or {}
        details_out = usage.get("output_token_details") or {}
        reasoning_tokens = details_out.get("reasoning", 0)
        rw_in = rewrite_usage.get("input_tokens", 0)
        rw_out = rewrite_usage.get("output_tokens", 0)

        # Generation and rewrite run on different models (and prices); thinking
        # tokens bill as output. Rerank is per-call, embedding ~flat.
        est_cost = round(
            input_tokens / 1e6 * PRICE_GEN_INPUT_PER_M
            + (output_tokens + reasoning_tokens) / 1e6 * PRICE_GEN_OUTPUT_PER_M
            + rw_in / 1e6 * PRICE_REWRITE_INPUT_PER_M
            + rw_out / 1e6 * PRICE_REWRITE_OUTPUT_PER_M
            + n_rerank_calls * PRICE_RERANK_PER_CALL
            + PRICE_EMBED_PER_QUERY,
            6
        )

        meta = {
            "answer": answer,
            "docs": docs,
            "standalone_question": standalone_question,
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cache_read_tokens": details_in.get("cache_read", 0),
                "rewrite_input_tokens": rw_in,
                "rewrite_output_tokens": rw_out,
            },
            "n_rerank_calls": n_rerank_calls,
            "models": {"generation": GEMINI_MODEL, "rewriter": REWRITER_MODEL},
            "est_cost_usd": est_cost,
            "timings": {
                "rewrite_s": round(t_rewrite, 3),
                "retrieve_s": round(t_retrieve, 3),
                "rerank_s": round(t_rerank, 3),
                "generate_s": round(t_generate, 3),
                "total_s": round(time.perf_counter() - t_start, 3),
            },
            "n_context_docs": len(docs),
            "context_mode": CONTEXT_MODE,
            "n_subqueries": len(sub_queries),
            "sub_queries": sub_queries,
            "decompose_fallback": decompose_fallback,
            "retrieval_mode": "fanout" if use_fanout else "single",
            "n_candidates": n_candidates,
            **context_stats,
        }

        log_entry = {k: v for k, v in meta.items() if k not in ("answer", "docs")}
        logger.info("query_usage %s", json.dumps(log_entry, ensure_ascii=False))

        return meta

    def query(self, question: str, memory: Optional[ConversationBufferWindowMemory] = None) -> str:
        return self.query_with_meta(question, memory=memory)["answer"]

    def chat(self):
        while True:
            try:
                question = input("You: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye")
                    break

                print("\nThinking...")
                answer = self.query(question)

                print("\n" + "=" * 80)
                print("ANSWER:")
                print("=" * 80)
                print(answer)
                print("=" * 80 + "\n")

            except KeyboardInterrupt:
                print("\nGoodbye")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    try:
        chatbot = BCITChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("1. Run: python build_pgvector.py")
        print("2. Check ADC: gcloud auth application-default login (or VM service account)")
        print("3. Check PG_CONNECTION in .env and that the Cloud SQL Auth Proxy is running")
        print("4. Check: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
