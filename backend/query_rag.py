import os
import warnings
import pickle

os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from typing import List, Set

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory

from embeddings import BGEM3Embeddings
from reranker import CrossEncoderReranker
from hybrid_retriever import create_hybrid_retriever
from config import (
    VECTOR_STORE_DIR,
    DOCUMENTS_PICKLE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    GEMINI_MODEL,
    GEMINI_API_KEY,
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
    RERANKER_CANDIDATES,
    RERANKER_TOP_K
)


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
        self._create_chain()

        print("\nCommands: 'quit', 'exit', 'q' to exit\n")

    def _validate_requirements(self):

        if not VECTOR_STORE_DIR.exists():
            raise FileNotFoundError(f"Vectorstore not found: {VECTOR_STORE_DIR}")

        if USE_HYBRID_SEARCH and not DOCUMENTS_PICKLE.exists():
            raise FileNotFoundError(f"Documents pickle not found: {DOCUMENTS_PICKLE}")

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")

        print("Requirements validated\n")

    def _load_embeddings(self):
        import logging
        logging.getLogger('FlagEmbedding').setLevel(logging.ERROR)

        self.embeddings = BGEM3Embeddings(
            model_name=EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
            use_fp16=True,
            normalize_embeddings=True
        )
        print("Embeddings loaded\n")

    def _load_vectorstore(self):
        self.vectorstore = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vectorstore loaded ({self.vectorstore.index.ntotal:,} vectors)\n")

    def _load_documents(self):
        if USE_HYBRID_SEARCH:
            with open(DOCUMENTS_PICKLE, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents):,} documents\n")
        else:
            self.documents = None

    def _initialize_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS
        )
        print("LLM initialized\n")

    def _initialize_memory(self):
        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True
        )
        print("Memory initialized")

    def _initialize_reranker(self):
        if USE_RERANKING:
            self.reranker = CrossEncoderReranker(
                model_name=RERANKER_MODEL,
                device=EMBEDDING_DEVICE
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

    def _format_chat_history(self) -> str:
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation."

        formatted = []
        for msg in messages:
            role = "Student" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def _rewrite_query(self, question: str) -> str:
        chat_history = self._format_chat_history()

        if chat_history == "No previous conversation.":
            return question

        rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)
        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

        try:
            rewritten = rewrite_chain.invoke({
                "chat_history": chat_history,
                "question": question
            })

            if rewritten.strip() != question.strip():
                print(f"\n[Query Rewritten]")
                print(f"Original: {question}")
                print(f"Rewritten: {rewritten}")

            return rewritten.strip()
        except:
            return question

    def _retrieve_with_rewrite(self, question: str) -> List[Document]:
        standalone_question = self._rewrite_query(question)

        docs = self.base_retriever.invoke(standalone_question)

        if USE_RERANKING and self.reranker:
            docs = self.reranker.rerank(
                query=standalone_question,
                documents=docs,
                top_k=RERANKER_TOP_K
            )
        return docs

    def _create_chain(self):
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.chain = (
                {
                    "context": lambda x: self._format_docs_full(
                        self._retrieve_with_rewrite(x)
                    ),
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: self._format_chat_history()
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

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

        print("\nRETRIEVED DOCUMENTS:")


        formatted = []
        for i, source_path in enumerate(sorted(unique_sources), 1):
            metadata = source_to_metadata.get(source_path, {})
            full_content = self._load_full_document(source_path)

            if not full_content:
                continue

            filename = metadata.get('filename', 'N/A')
            print(f"{i}. {filename}")

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

        print("-" * 80 + "\n")

        return "\n\n" + "-" * 60 + "\n\n".join(formatted)

    def query(self, question: str) -> str:
        easter_eggs = {
        "WHO IS THE BEST INSTRUCTOR AT BCIT": "Chi En Huang",
        "WHO IS THE BEST INSTRUCTOR AT BCIT?": "Chi En Huang"
        }

        normalized = question.strip().upper()
        if normalized in easter_eggs:
            answer = easter_eggs[normalized]
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            return answer

        answer = self.chain.invoke(question)

        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)

        return answer

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
        print("1. Run: python build_vectorstore.py")
        print("2. Check GEMINI_API_KEY in .env")
        print("3. Check: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
