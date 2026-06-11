import warnings
import pickle

warnings.filterwarnings('ignore')

import json
import logging
import time
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
    PRICE_INPUT_PER_M,
    PRICE_OUTPUT_PER_M,
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

    def _initialize_llm(self):
        self.llm = ChatVertexAI(
            model=GEMINI_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
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

    def _create_prompts(self):
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)

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
                "est_cost_usd": None,
                "timings": {},
                "n_context_docs": 0,
            }

        t_start = time.perf_counter()
        chat_history = self._format_chat_history(memory)

        t0 = time.perf_counter()
        standalone_question, rewrite_usage = self._rewrite_query(question, chat_history)
        t_rewrite = time.perf_counter() - t0

        t0 = time.perf_counter()
        docs = self.base_retriever.invoke(standalone_question)
        t_retrieve = time.perf_counter() - t0

        t0 = time.perf_counter()
        if USE_RERANKING and self.reranker:
            docs = self.reranker.rerank(
                query=standalone_question,
                documents=docs,
                top_k=RERANKER_TOP_K
            )
        t_rerank = time.perf_counter() - t0

        context = self._format_docs_full(docs)

        t0 = time.perf_counter()
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
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
        memory.chat_memory.add_ai_message(answer)

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        details_in = usage.get("input_token_details") or {}
        details_out = usage.get("output_token_details") or {}
        rw_in = rewrite_usage.get("input_tokens", 0)
        rw_out = rewrite_usage.get("output_tokens", 0)

        est_cost = None
        if PRICE_INPUT_PER_M > 0 or PRICE_OUTPUT_PER_M > 0:
            est_cost = round(
                (input_tokens + rw_in) / 1e6 * PRICE_INPUT_PER_M
                + (output_tokens + rw_out) / 1e6 * PRICE_OUTPUT_PER_M,
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
                "reasoning_tokens": details_out.get("reasoning", 0),
                "cache_read_tokens": details_in.get("cache_read", 0),
                "rewrite_input_tokens": rw_in,
                "rewrite_output_tokens": rw_out,
            },
            "est_cost_usd": est_cost,
            "timings": {
                "rewrite_s": round(t_rewrite, 3),
                "retrieve_s": round(t_retrieve, 3),
                "rerank_s": round(t_rerank, 3),
                "generate_s": round(t_generate, 3),
                "total_s": round(time.perf_counter() - t_start, 3),
            },
            "n_context_docs": len(docs),
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
