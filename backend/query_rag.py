import warnings
import pickle

warnings.filterwarnings('ignore')

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import List, Optional, Set

from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages.ai import add_usage
from session_memory import SessionMemory

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from embeddings import VertexGeminiEmbeddings
from reranker import VertexRanker
from hybrid_retriever import create_hybrid_retriever, doc_key
from response_cache import ResponseCache, normalize_question
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
    RRF_K,
    BM25_INDEX_AUG,
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
    RERANK_IDENTITY,
    RERANK_TIMEOUT_S,
    RERANK_ATTEMPTS,
    RERANK_HEDGE_AFTER_S,
    RERANK_HEDGE_WORKERS,
    RERANK_HEDGE_ABANDON_S,
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
    REWRITE_THINKING_LEVEL,
    REWRITE_DECOMPOSE_TEMPLATE,
    REWRITE_DECOMPOSE_SCHEMA,
    GEMINI_THINKING_LEVEL,
    MQ_DENSE_K,
    MQ_BM25_K,
    MQ_CANDIDATES_PER_SUBQUERY,
    MQ_POOL_CAP,
    MQ_MIN_CHUNKS_PER_SUBQUERY,
    USE_GRAPH,
    GRAPH_MODEL,
    GRAPH_TEMPERATURE,
    GRAPH_MAX_OUTPUT_TOKENS,
    GRAPH_THINKING_LEVEL,
    GRAPH_MAX_HOPS,
    GRAPH_HOP_TOP_K,
    GRAPH_HOP_CONTEXT_MAX_CHARS,
    GRAPH_CONTROLLER_SCHEMA,
    GRAPH_CONTROLLER_TEMPLATE,
    GRAPH_DIRECT_TEMPLATE,
    PRICE_GRAPH_INPUT_PER_M,
    PRICE_GRAPH_OUTPUT_PER_M,
    RERANK_MODE,
    RERANK_SKIP_CONSENSUS,
    MQ_BM25_KEYWORDS,
    HYDE_MODE,
    REWRITE_SKIP_SIMPLE,
    REWRITE_SKIP_MAX_WORDS,
    STRIP_SOURCES_FROM_HISTORY,
    HISTORY_MAX_ANSWER_CHARS,
    RESPONSE_CACHE_ENABLED,
    RESPONSE_CACHE_MAX,
    RESPONSE_CACHE_TTL_S,
    BM25_STOPWORD_DF,
    DEDUP_FULL_CONTENT,
    SIGNATURE_DEMOTE,
    FANIN_RETAIN,
    FANIN_RETAIN_N,
    PERSON_SCOPED_RETRIEVAL,
    PERSON_SCOPED_K,
    PERSON_SCOPED_MAX_SOURCES,
    ENTITY_SCOPED_RETRIEVAL,
    ENTITY_SCOPED_K,
    ENTITY_SCOPED_MAX_ENTITIES,
)

logger = logging.getLogger("bcit.rag")


def _is_simple_query(question: str) -> bool:
    """First-turn questions the rewriter would return unchanged: short,
    single-clause, no multipart separators. Conservative on purpose — a miss
    just means one ordinary rewrite call."""
    q = question.strip().lower()
    if len(q.split()) >= REWRITE_SKIP_MAX_WORDS:
        return False
    if q.count("?") > 1:
        return False
    return not any(sep in q for sep in (",", ";", " and ", " or ", " vs ", " versus "))


# A BCIT course/program code as it appears in questions: 2-4 letters then a
# 4-5 char number-led code (apprenticeship codes like "AATE 1GAP" exist — the
# same shape build_pgvector.py parses out of outline filenames).
COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})\s*-?\s*(\d[A-Za-z0-9]{3,4})\b")

# Capitalised runs that are not the thing a question is fanning out over.
_FANIN_NOISE = frozenset(
    "bcit what which who whose when where how does do did is are the a an "
    "and or of for in at to i he she they teach teaches taught course courses "
    "class classes instructor instructors program programs".split()
)
# Hyphens and apostrophes are part of the word: without them "Julia
# Alards-Tomalin" truncates to "Julia Alards" and "Sean O'Brien" to "Sean O",
# neither of which is in any index — measured as a silent miss on pl-04.
_NAME_WORD = r"[A-Z][a-z]*\.?(?:[-'’][A-Za-z]+)*"
_PROPER_RUN_RE = re.compile(rf"\b([A-Z][a-z]+(?:[-'’][A-Za-z]+)*(?:\s+{_NAME_WORD}){{0,3}})")

# The two ways the corpus states "this person instructs this course": the
# outline template's Instructor Details table and the course page's Instructor
# heading. Approval signatures and program-page coordinator lists deliberately
# do NOT match — they are other relations about the same person.
INSTRUCTOR_NAME_RE = re.compile(
    r"(?:### Instructor Details\s*\nName \|\s*([^\n|]+)"
    r"|### Instructor\s*\n([^\n]+))"
)


def _instructor_names(text: str):
    for m in INSTRUCTOR_NAME_RE.finditer(text):
        yield (m.group(1) or m.group(2) or "").strip()


def _fanin_key(question: str) -> Optional[str]:
    """The literal a fan-in question is spread over: a course code if the
    question names one (mh3-02's "ACIT 1515"), else the longest capitalised
    proper-noun run that is not a question word (a person's name).

    Deliberately literal: the retention rule swaps a chunk in only when the
    chunk *contains* this string, so a loose key would retain noise. A
    question with neither shape returns None and retention is a no-op — which
    is most traffic.
    """
    m = COURSE_CODE_RE.search(question)
    if m:
        return f"{m.group(1).upper()} {m.group(2).upper()}"
    runs = []
    for r in _PROPER_RUN_RE.findall(question):
        words = [w for w in r.split() if w.lower().strip(".") not in _FANIN_NOISE]
        if len(words) >= 2:
            runs.append(" ".join(words))
    return max(runs, key=len) if runs else None


# Words too generic to identify a program on their own.
_PROGRAM_STOPWORDS = frozenset(
    "the a an and or of for in at to bcit program programs diploma degree "
    "certificate full time part flexible learning studies option options".split()
)


def _norm_tokens(text: str) -> Set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


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


def _reduce_stream_outputs(outputs):
    """LangSmith trace output for query_stream: collapse the yielded
    ("delta", ...) / ("done", meta) tuples into the final meta (sans docs)
    so the trace looks like query_with_meta's instead of a token list."""
    for kind, payload in reversed(outputs or []):
        if kind == "done":
            return {k: v for k, v in payload.items() if k != "docs"}
    return {"answer": "".join(p for k, p in (outputs or []) if k == "delta")}


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
        self._initialize_cache()
        self._initialize_graph()

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
        self._entity_index = (
            self._build_entity_index(self.documents)
            if (self.documents and (ENTITY_SCOPED_RETRIEVAL or PERSON_SCOPED_RETRIEVAL)) else None
        )
        if self._entity_index:
            people = self._entity_index.get("people") or {}
            # people holds partial-name aliases too; count the canonical keys.
            n_people = sum(1 for k, v in people.items() if k == v["name"].lower())
            print(f"Entity index: {len(self._entity_index['codes']):,} course codes, "
                  f"{len(self._entity_index['programs']):,} programs, "
                  f"{n_people:,} instructors ({len(people) - n_people} aliases)\n")

    @staticmethod
    def _build_entity_index(documents) -> dict:
        """course code / program title -> the source files that ARE that entity.

        Built from filenames and titles, which build_pgvector.py already
        guarantees the shape of (`DEPT_NUM_TERM.txt` for outlines, a
        `..._dept_num.txt` suffix for course pages). Used two ways: to scope a
        retrieval arm to one entity's own chunks, and — just as important — to
        answer "does this entity exist in the corpus at all", so a future
        corrective loop never retries a question the corpus cannot answer.
        """
        codes = {}
        programs = []
        seen_sources = set()
        for doc in documents:
            source = doc.metadata.get("source")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            metadata = doc.metadata
            filename = metadata.get("filename") or ""
            category = metadata.get("category")

            match = None
            if category == "course_outline":
                match = re.match(r"^([A-Za-z]{2,4})_(\d[A-Za-z0-9]{3,4})_\d{6}\.txt$", filename)
            elif category == "course":
                match = re.search(r"_([a-z]{2,4})_(\d[a-z0-9]{3,4})\.txt$", filename)
            if match:
                key = f"{match.group(1).upper()} {match.group(2).upper()}"
                codes.setdefault(key, []).append(source)
                continue

            if category == "program":
                title = metadata.get("title") or ""
                tokens = _norm_tokens(title) - _PROGRAM_STOPWORDS
                # 2+ distinctive tokens, else the match would be a coin flip.
                if len(tokens) >= 2:
                    programs.append((frozenset(tokens), source))

        people = BCITChatbot._build_person_index(documents) if PERSON_SCOPED_RETRIEVAL else {}
        return {"codes": codes, "programs": programs, "people": people}

    @staticmethod
    def _build_person_index(documents) -> dict:
        """instructor name -> the sources that name them AS THE INSTRUCTOR.

        The same trick as `codes`, on the one relation the corpus states
        structurally: outlines carry `### Instructor Details / Name | X` and
        course pages carry `### Instructor\\nX`. Nothing else counts — a person
        also appears in approval signatures (62 chunks for one name against 4
        instructor chunks) and in program-page coordinator lists, and those are
        *different relations* about the same person. Indexing only the
        instructor relation is what makes "what does X teach" answerable
        without the ranker having to infer role from boilerplate.

        Names come from the corpus, so a query naming someone who instructs
        nothing simply misses the index and the arm stays off.
        """
        people = {}
        for doc in documents:
            source = doc.metadata.get("source")
            if not source:
                continue
            for name in _instructor_names(doc.page_content):
                # Two-to-four capitalised words: a name, not "Instructor to
                # provide" or an email line the template also puts here.
                words = name.split()
                if not 2 <= len(words) <= 4 or not all(w[:1].isupper() for w in words):
                    continue
                people.setdefault(name.lower(), {"name": name, "sources": []})
                if source not in people[name.lower()]["sources"]:
                    people[name.lower()]["sources"].append(source)

        # Partial-name aliases. Students type "chi en", not "Chi En Huang", and
        # the rewriter completes the name only ~4 times in 5 — measured on the
        # question that opened this round, where the one run in five that left
        # it partial is exactly the failure that was reported. Two words
        # minimum (a lone first name is not evidence of anything) and only when
        # the alias resolves to ONE instructor and is not itself somebody's
        # full name: 76 aliases over 1,291 instructors, 0 ambiguous, 0 clashes.
        owners = {}
        for key, entry in people.items():
            w = entry["name"].split()
            if len(w) < 3:
                continue
            for alias in {f"{w[0]} {w[-1]}".lower(), " ".join(w[:2]).lower()}:
                if alias != key:
                    owners.setdefault(alias, set()).add(key)
        for alias, keys in owners.items():
            if len(keys) == 1 and alias not in people:
                people[alias] = people[next(iter(keys))]
        return people

    def _detect_entities(self, text: str) -> List[tuple]:
        """(label, [sources]) for every corpus entity this text names.

        A regex hit that is not in the index is simply dropped, so false
        positives ("Category 1", "top 1000") cost nothing.
        """
        index = self._entity_index
        if not index:
            return []
        found, seen = [], set()

        for dept, num in COURSE_CODE_RE.findall(text):
            key = f"{dept.upper()} {num.upper()}"
            if key in index["codes"] and key not in seen:
                seen.add(key)
                found.append((key, index["codes"][key]))

        # A person named in the question, matched against instructors the
        # corpus actually has. Checked before programs: "what does Lynn
        # Erickson teach" should scope to her courses, not to a program page
        # whose title happens to share two tokens with the question.
        for name_run in _PROPER_RUN_RE.findall(text):
            words = [w for w in name_run.split() if w.lower().strip(".") not in _FANIN_NOISE]
            for n in range(len(words), 1, -1):
                cand = " ".join(words[:n]).lower()
                entry = index.get("people", {}).get(cand)
                if entry and cand not in seen:
                    seen.add(cand)
                    found.append((entry["name"], entry["sources"]))
                    break

        tokens = _norm_tokens(text)
        for title_tokens, source in index["programs"]:
            if title_tokens <= tokens:
                label = " ".join(sorted(title_tokens))
                if label not in seen:
                    seen.add(label)
                    found.append((label, [source]))
        return found[:ENTITY_SCOPED_MAX_ENTITIES]

    def _scoped_candidates(self, query: str) -> List[Document]:
        """Entity-scoped hits for one query, in entity order."""
        if not (self._entity_index and hasattr(self.base_retriever, "scoped_search")):
            return []
        people = self._entity_index.get("people") or {}
        out = []
        for label, sources in self._detect_entities(query):
            if label.lower() in people:
                # A person is one entity spread over many sources, the mirror
                # of a course (one entity, two sources). Breadth over depth:
                # take the best PERSON_SCOPED_K chunks from each of up to
                # PERSON_SCOPED_MAX_SOURCES sources, so a 7-course instructor
                # contributes 7 pages instead of 8 chunks of one page.
                sources = sources[:PERSON_SCOPED_MAX_SOURCES]
                k = PERSON_SCOPED_K
            else:
                k = ENTITY_SCOPED_K
            hits = self.base_retriever.scoped_search(query, sources, k)
            for doc in hits:
                doc.metadata["scoped_entity"] = label
            out.extend(hits)
        return out

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
        # vertexai=True is explicit on purpose. ChatGoogleGenerativeAI picks its
        # backend by inference when the flag is left off (env var, then
        # credentials, then project, else the public Gemini Developer API), so a
        # blank or mistyped project would silently route this app off Vertex and
        # away from the VM's service-account ADC. Naming the backend removes the
        # inference. No API key is set anywhere; auth stays ADC-only.
        llm_kwargs = dict(
            model=GEMINI_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            vertexai=True,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        )
        if GEMINI_THINKING_LEVEL is not None:
            llm_kwargs["thinking_level"] = GEMINI_THINKING_LEVEL
        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)

        # Dedicated rewriter: deterministic, schema-constrained JSON, the lowest
        # thinking the model allows — a cheap fixed-shape call that runs every
        # turn. REWRITER_MODEL is 3.7-flash, whose floor is "low"; the
        # thinking-off setting this call used to run at does not exist on it.
        self.rewriter_llm = ChatGoogleGenerativeAI(
            model=REWRITER_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            vertexai=True,
            temperature=0.0,
            max_output_tokens=REWRITE_MAX_OUTPUT_TOKENS,
            thinking_level=REWRITE_THINKING_LEVEL,
            response_mime_type="application/json",
            response_schema=REWRITE_DECOMPOSE_SCHEMA,
        )
        print("LLM initialized\n")

    def _initialize_memory(self):
        self.memory = SessionMemory(
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
                ranking_config=RANKING_CONFIG,
                identity=RERANK_IDENTITY,
                timeout_s=RERANK_TIMEOUT_S,
                attempts=RERANK_ATTEMPTS,
                hedge_after_s=RERANK_HEDGE_AFTER_S,
                hedge_workers=RERANK_HEDGE_WORKERS,
                hedge_abandon_s=RERANK_HEDGE_ABANDON_S,
            )
            print("Reranker initialized")
        else:
            self.reranker = None

    def _initialize_cache(self):
        self.response_cache = (
            ResponseCache(maxsize=RESPONSE_CACHE_MAX, ttl_s=RESPONSE_CACHE_TTL_S)
            if RESPONSE_CACHE_ENABLED else None
        )
        print(f"Response cache: {'on' if self.response_cache else 'off'}")

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
                dense_lambda=MMR_LAMBDA,
                rrf_k=RRF_K,
                bm25_index_aug=BM25_INDEX_AUG,
                stopword_df=BM25_STOPWORD_DF,
                scoped=ENTITY_SCOPED_RETRIEVAL,
                dedup_full=DEDUP_FULL_CONTENT,
                signature_demote=SIGNATURE_DEMOTE,
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

    def _format_chat_history(self, memory: SessionMemory) -> str:
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
        decomposing). Returns (standalone, sub_queries, usage, fallback,
        extras) — extras carries the optional experiment-gated fields
        (bm25_keywords, hyde_passage) from the same call.
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

            extras = {}
            if MQ_BM25_KEYWORDS:
                extras["bm25_keywords"] = [
                    k.strip() for k in (data.get("bm25_keywords") or [])
                    if isinstance(k, str) and k.strip()
                ][:8]
            if HYDE_MODE != "off":
                extras["hyde_passage"] = (data.get("hyde_passage") or "").strip()

            if standalone != question.strip():
                logger.info("query rewritten: %r -> %r", question, standalone)
            if len(sub_queries) > 1:
                logger.info("decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

            return standalone, sub_queries, dict(msg.usage_metadata or {}), False, extras
        except Exception:
            logger.warning("rewrite+decompose failed, using raw question", exc_info=True)
            return question, [question], {}, True, {}

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

    def _expand_and_format_chunks(self, docs: List[Document], max_chars: int = None) -> tuple:
        # max_chars overrides CONTEXT_MAX_CHARS for a single call. Only the
        # controller graph passes it, on turns that actually took a hop:
        # the cap is what binds (measured context_chars p50 22,276 against
        # a 24,000 cap), so raising the chunk count without it does nothing.
        if max_chars is None:
            max_chars = CONTEXT_MAX_CHARS
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
        while len(context) > max_chars:
            shrinkable = [s for s in reversed(sources) if radii[s] > 0]
            if shrinkable:
                radii[shrinkable[0]] = 0
            elif len(sources) > 1:
                sources.pop()
            else:
                context = context[:max_chars]
                break
            context = render(radii, sources)

        stats = {
            "n_chunks_kept": len(docs),
            "n_context_sources": len(sources),
            "neighbor_misses": neighbor_misses,
            "context_chars": len(context),
        }
        return context, stats

    def _multi_query_retrieve(self, standalone_question: str, sub_queries: List[str], extras: dict = None) -> tuple:
        """Fan out retrieval per sub-query, merge into one deduped pool, then
        rerank once. Returns (docs, stats)."""
        extras = extras or {}
        keywords = extras.get("bm25_keywords") or [] if MQ_BM25_KEYWORDS else []
        kw_suffix = (" " + " ".join(keywords)) if keywords else ""

        # A genuinely single-question turn only lands here for the HyDE-extra
        # arm — keep its real query at full single-turn width.
        single = len(sub_queries) == 1
        dense_k = RETRIEVAL_DENSE_K if single else MQ_DENSE_K
        bm25_k = RETRIEVAL_BM25_K if single else MQ_BM25_K
        per_query_cap = RERANKER_CANDIDATES if single else MQ_CANDIDATES_PER_SUBQUERY

        t0 = time.perf_counter()
        futures = [
            self._retrieval_pool.submit(
                self.base_retriever.search,
                q,
                dense_k=dense_k,
                bm25_k=bm25_k,
                top_k=per_query_cap,
                bm25_query=(q + kw_suffix) if kw_suffix else None,
            )
            for q in sub_queries
        ]
        per_sub = [f.result() for f in futures]

        # Entity-scoped arm: prepend each sub-query's scoped hits to its OWN
        # candidate list, so the rank-interleave below picks them first and the
        # coverage quota protects them. Global search cannot rank a chunk whose
        # body has no entity identity; this is the only path that can.
        n_scoped = 0
        if self._entity_index:
            for si, sub_query in enumerate(sub_queries):
                scoped = self._scoped_candidates(sub_query)
                if not scoped:
                    continue
                have = {doc_key(d, DEDUP_FULL_CONTENT) for d in per_sub[si]}
                fresh = [d for d in scoped if doc_key(d, DEDUP_FULL_CONTENT) not in have]
                per_sub[si] = fresh + per_sub[si]
                n_scoped += len(fresh)

        # HyDE "extra": one additional dense-only arm queried with the
        # hypothetical answer sentence. Its origin index (len(sub_queries))
        # is outside every quota loop, so it contributes candidates without
        # claiming coverage.
        hyde_passage = (extras.get("hyde_passage") or "") if HYDE_MODE == "extra" else ""
        if hyde_passage:
            try:
                hyde_hits = self.base_retriever._dense_search(hyde_passage, dense_k=MQ_DENSE_K)
                per_sub.append([
                    Document(page_content=d.page_content, metadata=dict(d.metadata))
                    for d in hyde_hits[:MQ_CANDIDATES_PER_SUBQUERY]
                ])
            except Exception:
                logger.warning("hyde retrieval failed, continuing without it", exc_info=True)

        # Interleave by rank so the pool cap cuts fairly across sub-queries.
        # Dedupe keeps the first copy and records every sub-query that
        # surfaced the chunk (for the coverage quota below); duplicates also
        # accumulate a cross-sub-query fusion score (RRF scores share scale).
        pooled: List[Document] = []
        by_key = {}
        origins = {}  # id(doc) -> set of sub-query indices
        for rank in range(max((len(d) for d in per_sub), default=0)):
            for si, docs_i in enumerate(per_sub):
                if rank >= len(docs_i):
                    continue
                doc = docs_i[rank]
                key = doc_key(doc, DEDUP_FULL_CONTENT)
                if key in by_key:
                    kept = by_key[key]
                    origins[id(kept)].add(si)
                    kept.metadata["pool_fusion_score"] = (
                        kept.metadata.get("pool_fusion_score",
                                          kept.metadata.get("fusion_score", 0.0))
                        + doc.metadata.get("fusion_score", 0.0)
                    )
                elif len(pooled) < MQ_POOL_CAP:
                    by_key[key] = doc
                    origins[id(doc)] = {si}
                    pooled.append(doc)
        t_retrieve = time.perf_counter() - t0

        t0 = time.perf_counter()
        docs, sel_info = self._select_from_pool(standalone_question, sub_queries, pooled, origins)
        t_rerank = time.perf_counter() - t0

        return docs, {
            "retrieve_s": t_retrieve,
            "rerank_s": t_rerank,
            "n_candidates": len(pooled),
            "n_scoped_candidates": n_scoped,
            **sel_info,
        }

    @staticmethod
    def _consensus(docs: List[Document], origins: dict = None) -> float:
        """Fraction of docs both retrieval arms agree on (surfaced by dense
        AND BM25, or by 2+ sub-queries)."""
        if not docs:
            return 0.0
        agreed = 0
        for d in docs:
            both_arms = (
                d.metadata.get("dense_rank") is not None
                and d.metadata.get("bm25_rank") is not None
            )
            multi_origin = origins is not None and len(origins.get(id(d), ())) >= 2
            if both_arms or multi_origin:
                agreed += 1
        return agreed / len(docs)

    def _apply_coverage_quota(self, selected, rest, sub_queries, origins, sort_key):
        """Coverage quota: every sub-query keeps >= MQ_MIN_CHUNKS_PER_SUBQUERY
        of its own candidates; swap in its best leftovers, evicting the
        lowest-scored doc whose sub-queries all stay above quota."""
        def covered(si, sel):
            return sum(1 for d in sel if si in origins[id(d)])

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

        selected.sort(key=sort_key, reverse=True)
        return selected

    def _apply_fanin_retention(self, question, selected, rest, sort_key):
        """Fan-in retention: a question whose answer is spread across N sibling
        pages ("which courses require ACIT 1515?", "what does X teach?") has N
        correct sources, and the pooled rerank hands the context budget to
        whichever chunks score best globally — often several from one page, or
        pages unrelated to the named entity.

        Rule: for the literal the question names, guarantee up to
        FANIN_RETAIN_N *distinct sources* whose chunk text actually contains
        that literal. Swap each in against the lowest-scored selected chunk
        that neither mentions the literal nor is its source's only
        representative, so this can only redistribute slots, never shrink
        coverage of a source already held.

        Returns the number of swaps, for the meta line.
        """
        key = _fanin_key(question)
        if not key:
            return selected, 0
        held = {d.metadata.get("source") for d in selected}
        mentions = lambda d: key.lower() in d.page_content.lower()
        n_key_sources = len({d.metadata.get("source") for d in selected if mentions(d)})
        swaps = 0
        for cand in [d for d in rest if mentions(d)]:
            if n_key_sources >= FANIN_RETAIN_N:
                break
            src = cand.metadata.get("source")
            if src in held:
                continue
            counts = Counter(d.metadata.get("source") for d in selected)
            evict = next(
                (d for d in sorted(selected, key=sort_key)
                 if not mentions(d) and counts[d.metadata.get("source")] > 1),
                None,
            )
            if evict is None:
                evict = next((d for d in sorted(selected, key=sort_key) if not mentions(d)), None)
            if evict is None:
                break
            selected.remove(evict)
            selected.append(cand)
            held.discard(evict.metadata.get("source"))
            held.add(src)
            n_key_sources += 1
            swaps += 1
        selected.sort(key=sort_key, reverse=True)
        return selected, swaps

    def _select_from_pool(
            self,
            question: str,
            sub_queries: List[str],
            pooled: List[Document],
            origins: dict,
    ) -> tuple:
        """Returns (docs, sel_info) — sel_info reports whether the Ranking
        API call was consensus-skipped and the measured consensus."""
        sel_info = {"rerank_skipped": False, "pool_consensus": None}

        if not (USE_RERANKING and self.reranker):
            return pooled[:RERANKER_TOP_K], sel_info  # interleaved order ≈ even coverage

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
            return selected[:max(RERANKER_TOP_K, MQ_MIN_CHUNKS_PER_SUBQUERY * len(sub_queries))], sel_info

        # Rerank-skip: when both retrieval arms already agree on the
        # fusion-ordered top slice, trust that order and save the API call.
        if RERANK_SKIP_CONSENSUS > 0:
            fusion_key = lambda d: d.metadata.get(
                "pool_fusion_score", d.metadata.get("fusion_score", 0.0))
            tentative = sorted(pooled, key=fusion_key, reverse=True)
            sel = self._apply_coverage_quota(
                tentative[:RERANKER_TOP_K], tentative[RERANKER_TOP_K:],
                sub_queries, origins, fusion_key,
            )
            consensus = self._consensus(sel, origins)
            sel_info["pool_consensus"] = round(consensus, 3)
            if consensus >= RERANK_SKIP_CONSENSUS:
                sel_info["rerank_skipped"] = True
                return sel, sel_info

        # Default "pooled": ONE Ranking API call scoring the whole pool.
        scored = self.reranker.rerank(question, pooled, top_k=RERANKER_TOP_K, return_all=True)
        if any("rerank_score" not in d.metadata for d in scored):
            return scored[:RERANKER_TOP_K], sel_info  # API fallback: interleaved order

        score_key = lambda d: d.metadata.get("rerank_score", 0.0)
        rest = scored[RERANKER_TOP_K:]
        selected = self._apply_coverage_quota(
            scored[:RERANKER_TOP_K], rest, sub_queries, origins, score_key,
        )
        if FANIN_RETAIN:
            selected, swaps = self._apply_fanin_retention(question, selected, rest, score_key)
            sel_info["fanin_swaps"] = swaps
        return selected, sel_info

    def _cached_turn(self, question, memory, answer, turn_start) -> dict:
        """Build the meta for a first-turn cache hit. Mirrors _finalize_turn's
        shape (so server stats/metrics and the query_usage log line are
        identical) but zeroes usage/cost — a cache hit spends no API tokens —
        and writes the same memory + log a generated turn would."""
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(self._history_view(answer))

        meta = {
            "answer": answer,
            "docs": [],
            "standalone_question": question,
            "finish_reason": "",
            "usage": {
                "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "rewrite_input_tokens": 0, "rewrite_output_tokens": 0,
            },
            "n_rerank_calls": 0,
            "models": {"generation": GEMINI_MODEL, "rewriter": REWRITER_MODEL},
            "est_cost_usd": 0.0,
            "timings": {"total_s": round(time.perf_counter() - turn_start, 3)},
            "n_context_docs": 0,
            "context_mode": CONTEXT_MODE,
            "n_subqueries": 0,
            "sub_queries": [],
            "decompose_fallback": False,
            "rewrite_skipped": False,
            "rerank_skipped": False,
            "pool_consensus": None,
            "retrieval_mode": "cached",
            "n_candidates": 0,
            "cached": True,
        }
        log_entry = {k: v for k, v in meta.items() if k not in ("answer", "docs")}
        logger.info("query_usage %s", json.dumps(log_entry, ensure_ascii=False))
        return meta

    def _retrieve_and_rerank(self, standalone_question: str, sub_queries: List[str],
                             extras: dict, rewrite_skipped: bool) -> dict:
        """Retrieval + rerank for one pass: fan-out or full-width, scoped
        arms, then the Ranking API call. Lifted verbatim out of
        _prepare_turn so the controller graph can use it as its first hop
        without reimplementing the pipeline — the statements and their
        order are unchanged, only the surrounding function is new.
        """
        # Single-question turns keep the full-width legacy retrieval; only
        # genuinely multipart turns fan out per sub-query (or single turns
        # carrying a HyDE-extra arm, which need the pooled path too).
        hyde_extra = bool(extras.get("hyde_passage")) and HYDE_MODE == "extra"
        use_fanout = (
            (len(sub_queries) > 1 or hyde_extra)
            and self._retrieval_pool is not None
            and hasattr(self.base_retriever, "search")
        )
        rerank_skipped = False
        pool_consensus = None
        n_scoped_candidates = 0
        fanin_swaps = 0
        if use_fanout:
            docs, mq_stats = self._multi_query_retrieve(standalone_question, sub_queries, extras)
            t_retrieve = mq_stats["retrieve_s"]
            t_rerank = mq_stats["rerank_s"]
            n_candidates = mq_stats["n_candidates"]
            n_scoped_candidates = mq_stats.get("n_scoped_candidates", 0)
            rerank_skipped = mq_stats.get("rerank_skipped", False)
            pool_consensus = mq_stats.get("pool_consensus")
            fanin_swaps = mq_stats.get("fanin_swaps", 0)
            n_rerank_calls = 0
            if USE_RERANKING and self.reranker and not rerank_skipped:
                n_rerank_calls = len(sub_queries) if RERANK_MODE == "per_subquery" else 1
        else:
            t0 = time.perf_counter()
            keywords = extras.get("bm25_keywords") or [] if MQ_BM25_KEYWORDS else []
            hyde_replace = (extras.get("hyde_passage") or "") if HYDE_MODE == "replace" else ""
            if (keywords or hyde_replace) and hasattr(self.base_retriever, "search"):
                docs = self.base_retriever.search(
                    standalone_question,
                    bm25_query=(standalone_question + " " + " ".join(keywords)) if keywords else None,
                    dense_query=hyde_replace or None,
                )
            else:
                docs = self.base_retriever.invoke(standalone_question)
            if self._entity_index:
                have = {doc_key(d, DEDUP_FULL_CONTENT) for d in docs}
                fresh = [
                    d for d in self._scoped_candidates(standalone_question)
                    if doc_key(d, DEDUP_FULL_CONTENT) not in have
                ]
                n_scoped_candidates = len(fresh)
                docs = fresh + docs  # ahead of fusion order; the rerank decides
            t_retrieve = time.perf_counter() - t0

            t0 = time.perf_counter()
            n_candidates = len(docs)
            n_rerank_calls = 0
            if USE_RERANKING and self.reranker:
                # Defense in depth: a turn that already skipped the rewriter
                # keeps the rerank, so every query passes at least one
                # semantic stage. The round-2 eval showed no regression from
                # double-skips on the golden set, but raw user queries (typos,
                # bare acronyms) have no such safety net at ~0.1% query cost.
                if RERANK_SKIP_CONSENSUS > 0 and not rewrite_skipped:
                    head = docs[:RERANKER_TOP_K]
                    pool_consensus = round(self._consensus(head), 3)
                    rerank_skipped = pool_consensus >= RERANK_SKIP_CONSENSUS
                if rerank_skipped:
                    docs = docs[:RERANKER_TOP_K]  # fusion order stands
                else:
                    docs = self.reranker.rerank(
                        query=standalone_question,
                        documents=docs,
                        top_k=RERANKER_TOP_K
                    )
                    n_rerank_calls = 1
            t_rerank = time.perf_counter() - t0

        return {
            "docs": docs,
            "use_fanout": use_fanout,
            "n_candidates": n_candidates,
            "n_scoped_candidates": n_scoped_candidates,
            "rerank_skipped": rerank_skipped,
            "pool_consensus": pool_consensus,
            "fanin_swaps": fanin_swaps,
            "n_rerank_calls": n_rerank_calls,
            "t_retrieve": t_retrieve,
            "t_rerank": t_rerank,
            "sub_queries": sub_queries,
        }

    def _initialize_graph(self):
        """Controller graph wiring. Nothing here runs — and langgraph is not
        even imported — unless USE_GRAPH is on."""
        self.controller_llm = None
        self.controller_prompt = None
        self.direct_prompt = None
        self._controller_graph_cls = None
        if not USE_GRAPH:
            return

        from graph import ControllerGraph

        self._controller_graph_cls = ControllerGraph
        # Same shape as the rewriter: deterministic, schema-constrained JSON,
        # minimal thinking. It runs at least once on every turn, so its
        # per-call cost is the design's dominant cost variable — which is why
        # GRAPH_MODEL stayed on the lite tier through the 2026-08-14 move and
        # can still be told not to think at all.
        self.controller_llm = ChatGoogleGenerativeAI(
            model=GRAPH_MODEL,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
            vertexai=True,
            temperature=GRAPH_TEMPERATURE,
            max_output_tokens=GRAPH_MAX_OUTPUT_TOKENS,
            thinking_level=GRAPH_THINKING_LEVEL,
            response_mime_type="application/json",
            response_schema=GRAPH_CONTROLLER_SCHEMA,
        )
        self.controller_prompt = ChatPromptTemplate.from_template(GRAPH_CONTROLLER_TEMPLATE)
        self.direct_prompt = ChatPromptTemplate.from_template(GRAPH_DIRECT_TEMPLATE)
        print(f"Controller graph: on ({GRAPH_MODEL}, <={GRAPH_MAX_HOPS} hops)")

    def _controller_call(self, question, chat_history, evidence, hop, max_hops):
        """One controller decision. Raises on failure — graph.py owns the
        fail-open policy so there is exactly one place that decides what a
        controller outage degrades to."""
        prompt_value = self.controller_prompt.invoke({
            "question": question,
            "chat_history": chat_history,
            "evidence": evidence,
            "hop": hop,
            "max_hops": max_hops,
        })
        msg = self.controller_llm.invoke(prompt_value)
        return json.loads(_message_text(msg)), dict(msg.usage_metadata or {})

    def _graph_hop_retrieve(self, queries, docs_so_far, standalone_question):
        """A follow-up retrieval pass, merged into what the earlier passes found.

        Merging rather than replacing is the point: hop-1 documents answer the
        first half of a 2-hop question, so evicting them to make room for hop-2
        would trade one missing fact for another. The August fan-in experiment
        failed precisely because it could not tell those apart; here the
        controller has named what is missing, so the merge is relation-aware.
        """
        # With one query, retrieve on it directly — passing the original
        # standalone question would re-run the pass that already happened and
        # return the same documents.
        hop_question = queries[0] if len(queries) == 1 else standalone_question
        result = self._retrieve_and_rerank(hop_question, queries, {}, False)

        seen = {doc_key(d, DEDUP_FULL_CONTENT) for d in docs_so_far}
        merged = list(docs_so_far)
        for doc in result["docs"]:
            key = doc_key(doc, DEDUP_FULL_CONTENT)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        result["docs"] = merged[:GRAPH_HOP_TOP_K]
        return result

    def _prepare_turn_graph(self, question: str, chat_history: str,
                            first_turn: bool, t_start: float) -> dict:
        """USE_GRAPH twin of _prepare_turn's body, from the rewrite onward.

        A separate method rather than branches threaded through _prepare_turn:
        with the flag off, the shipped path is then not merely equivalent to
        today's but literally the same statements, which is the only way the
        "no regression when off" claim needs no measurement to believe.

        Returns the same prep dict _finalize_turn consumes, plus route/hop
        bookkeeping.
        """
        # The rewrite lives inside the first retrieval so a directly-routed
        # turn skips it — that saving is most of what the route buys.
        acc = {
            "standalone": question,
            "sub_queries": [question],
            "rewrite_usage": {},
            "decompose_fallback": False,
            "rewrite_skipped": False,
            "t_rewrite": 0.0,
            "t_retrieve": 0.0,
            "t_rerank": 0.0,
            "n_candidates": 0,
            "n_scoped_candidates": 0,
            "n_rerank_calls": 0,
            "fanin_swaps": 0,
            "use_fanout": False,
            "rerank_skipped": False,
            "pool_consensus": None,
            "passes": 0,
        }

        def absorb(result):
            acc["t_retrieve"] += result["t_retrieve"]
            acc["t_rerank"] += result["t_rerank"]
            acc["n_candidates"] += result["n_candidates"]
            acc["n_scoped_candidates"] += result["n_scoped_candidates"]
            acc["n_rerank_calls"] += result["n_rerank_calls"]
            acc["fanin_swaps"] += result["fanin_swaps"]
            acc["use_fanout"] = result["use_fanout"]
            acc["rerank_skipped"] = result["rerank_skipped"]
            acc["pool_consensus"] = result["pool_consensus"]
            acc["passes"] += 1
            return result

        def initial_retrieve():
            t0 = time.perf_counter()
            extras = {}
            if (REWRITE_SKIP_SIMPLE
                    and chat_history == "No previous conversation."
                    and _is_simple_query(question)):
                standalone, sub_queries = question, [question]
                acc["rewrite_skipped"] = True
            elif MULTI_QUERY_ENABLED:
                (standalone, sub_queries, acc["rewrite_usage"],
                 acc["decompose_fallback"], extras) = self._rewrite_and_decompose(
                    question, chat_history)
            else:
                standalone, acc["rewrite_usage"] = self._rewrite_query(question, chat_history)
                sub_queries = [standalone]
            acc["t_rewrite"] = time.perf_counter() - t0
            acc["standalone"], acc["sub_queries"] = standalone, sub_queries
            return absorb(self._retrieve_and_rerank(
                standalone, sub_queries, extras, acc["rewrite_skipped"]))

        def hop_retrieve(queries, docs_so_far):
            return absorb(self._graph_hop_retrieve(queries, docs_so_far, acc["standalone"]))

        graph = self._controller_graph_cls(
            self._controller_call, initial_retrieve, hop_retrieve)
        final = graph.run(question, chat_history,
                          has_history=chat_history != "No previous conversation.")

        docs = final.get("docs") or []
        action = final.get("action", "answer")
        retrieved = acc["passes"] > 0
        route = "retrieve" if retrieved else ("refuse" if action == "refuse" else "direct")

        if not retrieved:
            # No lookup happened, so there is no context and no Sources
            # section. The scope policy lives in the direct prompt, which is
            # why "no retrieval" does not become "answer BCIT questions from
            # memory" — the behaviour the out_of_scope cases measure.
            prompt_value = self.direct_prompt.invoke({
                "question": question,
                "chat_history": chat_history,
            })
            # context_chars is 0, not the prompt length: no BCIT context was
            # assembled, and reporting the prompt here would make a directly
            # routed turn look like it had retrieved something.
            context_stats = {"context_chars": 0, "n_chunks_kept": 0,
                             "n_context_sources": 0, "neighbor_misses": 0}
        else:
            # Hop turns get the wider budget; single-pass turns are billed
            # exactly as they are today.
            hop_budget = acc["passes"] > 1
            if CONTEXT_MODE == "chunks":
                context, context_stats = self._expand_and_format_chunks(
                    docs,
                    max_chars=GRAPH_HOP_CONTEXT_MAX_CHARS if hop_budget else None)
            else:
                context = self._format_docs_full(docs)
                context_stats = {"context_chars": len(context)}

            question_parts = ""
            if len(acc["sub_queries"]) > 1:
                question_parts = (
                    "\n\nThe question has multiple parts; answer each one:\n"
                    + "\n".join(f"  {i}) {q}"
                                for i, q in enumerate(acc["sub_queries"], 1))
                )
            prompt_value = self.prompt.invoke({
                "context": context,
                "question": question,
                "question_parts": question_parts,
                "chat_history": chat_history,
            })

        logger.info("graph route=%s passes=%d docs=%d trace=%s",
                    route, acc["passes"], len(docs), final.get("trace"))

        return {
            "t_start": t_start,
            "question": question,
            "cacheable": first_turn,
            "prompt_value": prompt_value,
            "docs": docs,
            "standalone_question": acc["standalone"],
            "sub_queries": acc["sub_queries"],
            "rewrite_usage": acc["rewrite_usage"],
            "decompose_fallback": acc["decompose_fallback"],
            "rewrite_skipped": acc["rewrite_skipped"],
            "rerank_skipped": acc["rerank_skipped"],
            "pool_consensus": acc["pool_consensus"],
            "use_fanout": acc["use_fanout"],
            "n_candidates": acc["n_candidates"],
            "n_scoped_candidates": acc["n_scoped_candidates"],
            "fanin_swaps": acc["fanin_swaps"],
            "n_rerank_calls": acc["n_rerank_calls"],
            "context_stats": context_stats,
            "t_rewrite": acc["t_rewrite"],
            "t_retrieve": acc["t_retrieve"],
            "t_rerank": acc["t_rerank"],
            "route": route,
            "graph_hops": max(acc["passes"] - 1, 0),
            "graph_trace": final.get("trace"),
            "graph_usage": final.get("usage"),
        }

    def _prepare_turn(self, question: str, memory: SessionMemory) -> dict:
        """Everything before generation: easter-egg short-circuit, history
        formatting, rewrite+decompose, (fan-out) retrieval, rerank, context
        assembly, prompt construction. Shared by query_with_meta (blocking)
        and query_stream (token streaming) so the two paths cannot drift.

        Returns {"short_circuit": <complete meta>} for the easter-egg or
        response-cache fast paths (memory already written), else the state
        _finalize_turn consumes."""
        turn_start = time.perf_counter()
        normalized = question.strip().upper()
        if normalized in EASTER_EGGS:
            answer = EASTER_EGGS[normalized]
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return {"short_circuit": {
                "answer": answer,
                "docs": [],
                "standalone_question": question,
                "finish_reason": "",
                "usage": {},
                "est_cost_usd": 0.0,
                "timings": {},
                "n_context_docs": 0,
            }}

        t_start = time.perf_counter()
        chat_history = self._format_chat_history(memory)

        # First-turn exact-match cache: a no-history question's answer is a
        # pure function of the question + corpus, so identical first questions
        # share an answer and skip the whole pipeline. Follow-ups are never
        # cached (they depend on session history). _finalize_turn stores on a
        # miss; `cacheable` flows through prep so it knows whether to.
        first_turn = (
            self.response_cache is not None
            and chat_history == "No previous conversation."
        )
        if first_turn:
            cached_answer = self.response_cache.get(normalize_question(question))
            if cached_answer is not None:
                return {"short_circuit": self._cached_turn(
                    question, memory, cached_answer, turn_start)}

        if USE_GRAPH:
            return self._prepare_turn_graph(question, chat_history, first_turn, t_start)

        t0 = time.perf_counter()
        rewrite_skipped = False
        extras = {}
        if (
            REWRITE_SKIP_SIMPLE
            and chat_history == "No previous conversation."
            and _is_simple_query(question)
        ):
            # The rewriter would return this unchanged — skip its cost/latency.
            standalone_question, sub_queries, rewrite_usage, decompose_fallback = (
                question, [question], {}, False
            )
            rewrite_skipped = True
        elif MULTI_QUERY_ENABLED:
            standalone_question, sub_queries, rewrite_usage, decompose_fallback, extras = (
                self._rewrite_and_decompose(question, chat_history)
            )
        else:
            standalone_question, rewrite_usage = self._rewrite_query(question, chat_history)
            sub_queries, decompose_fallback = [standalone_question], False
        t_rewrite = time.perf_counter() - t0

        retrieval = self._retrieve_and_rerank(
            standalone_question, sub_queries, extras, rewrite_skipped)
        docs = retrieval["docs"]
        use_fanout = retrieval["use_fanout"]
        n_candidates = retrieval["n_candidates"]
        n_scoped_candidates = retrieval["n_scoped_candidates"]
        rerank_skipped = retrieval["rerank_skipped"]
        pool_consensus = retrieval["pool_consensus"]
        fanin_swaps = retrieval["fanin_swaps"]
        n_rerank_calls = retrieval["n_rerank_calls"]
        t_retrieve = retrieval["t_retrieve"]
        t_rerank = retrieval["t_rerank"]

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

        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "question_parts": question_parts,
            "chat_history": chat_history
        })

        return {
            "t_start": t_start,
            "question": question,
            "cacheable": first_turn,
            "prompt_value": prompt_value,
            "docs": docs,
            "standalone_question": standalone_question,
            "sub_queries": sub_queries,
            "rewrite_usage": rewrite_usage,
            "decompose_fallback": decompose_fallback,
            "rewrite_skipped": rewrite_skipped,
            "rerank_skipped": rerank_skipped,
            "pool_consensus": pool_consensus,
            "use_fanout": use_fanout,
            "n_candidates": n_candidates,
            "n_scoped_candidates": n_scoped_candidates,
            "fanin_swaps": fanin_swaps,
            "n_rerank_calls": n_rerank_calls,
            "context_stats": context_stats,
            "t_rewrite": t_rewrite,
            "t_retrieve": t_retrieve,
            "t_rerank": t_rerank,
        }

    def _finalize_turn(
            self,
            prep: dict,
            memory: SessionMemory,
            answer: str,
            usage: dict,
            finish_reason: str,
            t_generate: float,
    ) -> dict:
        """Post-generation bookkeeping shared by both generation paths:
        memory write, cost estimate, meta assembly, query_usage log."""
        if finish_reason == "MAX_TOKENS":
            logger.warning("answer truncated (finish_reason=MAX_TOKENS) — Sources section likely lost")

        memory.chat_memory.add_user_message(prep["question"])
        memory.chat_memory.add_ai_message(self._history_view(answer))

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        details_in = usage.get("input_token_details") or {}
        details_out = usage.get("output_token_details") or {}
        reasoning_tokens = details_out.get("reasoning", 0)
        rw_in = prep["rewrite_usage"].get("input_tokens", 0)
        rw_out = prep["rewrite_usage"].get("output_tokens", 0)
        graph_usage = prep.get("graph_usage") or {}
        g_in = graph_usage.get("input_tokens", 0)
        g_out = graph_usage.get("output_tokens", 0)

        # Generation and rewrite run on different models (and prices); thinking
        # tokens bill as output. Rerank is per-call, embedding ~flat.
        est_cost = round(
            input_tokens / 1e6 * PRICE_GEN_INPUT_PER_M
            + (output_tokens + reasoning_tokens) / 1e6 * PRICE_GEN_OUTPUT_PER_M
            + rw_in / 1e6 * PRICE_REWRITE_INPUT_PER_M
            + rw_out / 1e6 * PRICE_REWRITE_OUTPUT_PER_M
            + g_in / 1e6 * PRICE_GRAPH_INPUT_PER_M
            + g_out / 1e6 * PRICE_GRAPH_OUTPUT_PER_M
            + prep["n_rerank_calls"] * PRICE_RERANK_PER_CALL
            + PRICE_EMBED_PER_QUERY,
            6
        )

        meta = {
            "answer": answer,
            "docs": prep["docs"],
            "standalone_question": prep["standalone_question"],
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cache_read_tokens": details_in.get("cache_read", 0),
                "rewrite_input_tokens": rw_in,
                "rewrite_output_tokens": rw_out,
                "graph_input_tokens": g_in,
                "graph_output_tokens": g_out,
                "graph_calls": graph_usage.get("calls", 0),
            },
            "n_rerank_calls": prep["n_rerank_calls"],
            "models": {
                "generation": GEMINI_MODEL,
                "rewriter": REWRITER_MODEL,
                **({"controller": GRAPH_MODEL} if USE_GRAPH else {}),
            },
            "est_cost_usd": est_cost,
            "timings": {
                "rewrite_s": round(prep["t_rewrite"], 3),
                "retrieve_s": round(prep["t_retrieve"], 3),
                "rerank_s": round(prep["t_rerank"], 3),
                "generate_s": round(t_generate, 3),
                "total_s": round(time.perf_counter() - prep["t_start"], 3),
            },
            "n_context_docs": len(prep["docs"]),
            "context_mode": CONTEXT_MODE,
            "n_subqueries": len(prep["sub_queries"]),
            "sub_queries": prep["sub_queries"],
            "decompose_fallback": prep["decompose_fallback"],
            "rewrite_skipped": prep["rewrite_skipped"],
            "rerank_skipped": prep["rerank_skipped"],
            "pool_consensus": prep["pool_consensus"],
            "retrieval_mode": "fanout" if prep["use_fanout"] else "single",
            "n_candidates": prep["n_candidates"],
            "n_scoped_candidates": prep["n_scoped_candidates"],
            "fanin_swaps": prep["fanin_swaps"],
            "route": prep.get("route"),
            "graph_hops": prep.get("graph_hops", 0),
            "graph_trace": prep.get("graph_trace"),
            "cached": False,
            **prep["context_stats"],
        }

        log_entry = {k: v for k, v in meta.items() if k not in ("answer", "docs")}
        logger.info("query_usage %s", json.dumps(log_entry, ensure_ascii=False))

        # Cache first-turn answers for exact-match reuse. Skip truncated ones —
        # a MAX_TOKENS answer likely lost its Sources section. Follow-ups carry
        # cacheable=False (set in _prepare_turn), so only no-history turns land.
        if (prep.get("cacheable") and self.response_cache is not None
                and finish_reason != "MAX_TOKENS"):
            self.response_cache.set(normalize_question(prep["question"]), answer)

        return meta

    @traceable(name="bcit_query")
    def query_with_meta(self, question: str, memory: Optional[SessionMemory] = None) -> dict:
        if memory is None:
            memory = self.memory

        prep = self._prepare_turn(question, memory)
        if "short_circuit" in prep:
            return prep["short_circuit"]

        t0 = time.perf_counter()
        msg = self.llm.invoke(prep["prompt_value"])
        t_generate = time.perf_counter() - t0

        return self._finalize_turn(
            prep,
            memory,
            answer=_message_text(msg),
            usage=dict(msg.usage_metadata or {}),
            finish_reason=str((msg.response_metadata or {}).get("finish_reason", "")),
            t_generate=t_generate,
        )

    @traceable(name="bcit_query", reduce_fn=_reduce_stream_outputs)
    def query_stream(self, question: str, memory: Optional[SessionMemory] = None):
        """Streaming twin of query_with_meta: yields ("delta", text) as
        tokens arrive, then ("done", meta) where meta is exactly what the
        blocking path returns (memory write + query_usage log included).

        Callers must drain the generator to completion — the server does so
        on a worker thread, so a disconnected client still leaves consistent
        session history and a query_usage log line."""
        if memory is None:
            memory = self.memory

        prep = self._prepare_turn(question, memory)
        if "short_circuit" in prep:
            sc = prep["short_circuit"]
            yield ("delta", sc["answer"])
            yield ("done", sc)
            return

        t0 = time.perf_counter()
        parts: List[str] = []
        usage: dict = {}
        finish_reason = ""
        for chunk in self.llm.stream(prep["prompt_value"]):
            text = _message_text(chunk)
            if text:
                parts.append(text)
                yield ("delta", text)
            # Usage must be SUMMED, not overwritten. ChatVertexAI reported the
            # whole turn's usage once, on the final chunk, so keeping the last
            # value seen was correct for it. ChatGoogleGenerativeAI does the
            # opposite: the Gemini API returns a running cumulative count on
            # every chunk and the integration subtracts the previous one, so
            # each chunk carries a DELTA and the final chunk holds only the last
            # few tokens. Overwriting here would have under-reported every
            # streamed turn's cost by roughly the whole answer. add_usage is
            # langchain-core's reducer for these dicts and folds the nested
            # input/output_token_details (where the reasoning count lives) too.
            # Missing usage still degrades to zero counts, never a crash.
            if getattr(chunk, "usage_metadata", None):
                usage = dict(add_usage(usage or None, chunk.usage_metadata))
            chunk_finish = str(
                (getattr(chunk, "response_metadata", None) or {}).get("finish_reason", "")
            )
            if chunk_finish:
                finish_reason = chunk_finish
        t_generate = time.perf_counter() - t0

        yield ("done", self._finalize_turn(
            prep,
            memory,
            answer="".join(parts),
            usage=usage,
            finish_reason=finish_reason,
            t_generate=t_generate,
        ))

    def query(self, question: str, memory: Optional[SessionMemory] = None) -> str:
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
