import { MessageSquare, ArrowRight, Database, Layers, Cpu, GitMerge, Server } from "lucide-react";
import "./Blog.css";

export default function Blog() {
  return (
    <div className="blog-page">
      <nav className="blog-nav">
        <span className="blog-nav-brand">BCIT AI Advisor</span>
        <a href="/chat" className="blog-nav-cta">
          <MessageSquare size={17} />
          <span>Open Chatbot</span>
        </a>
      </nav>

      <header className="blog-hero">
        <p className="blog-kicker">Engineering Deep Dive</p>
        <h1>Building a Production RAG Chatbot for BCIT</h1>
        <p className="blog-subtitle">
          How hybrid retrieval, a managed reranker, and Gemini turn 5,000+ pages of
          BCIT program and course documentation into an academic advisor that
          answers in seconds — on a CPU-only VM.
        </p>
        <a href="/chat" className="blog-hero-cta">
          Try the live chatbot <ArrowRight size={18} />
        </a>
      </header>

      <article className="blog-article">
        <section>
          <h2>The problem</h2>
          <p>
            BCIT's website spans hundreds of programs and thousands of courses. A
            prospective student asking <em>"What math do I need for the Computing
            diploma, and can I transfer into the degree later?"</em> has to stitch the
            answer together from a dozen pages. A general-purpose LLM either
            hallucinates the details or refuses to commit. The fix is
            retrieval-augmented generation: ground every answer in the actual
            institutional documents, retrieved at question time.
          </p>
        </section>

        <section>
          <h2>Architecture at a glance</h2>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><Layers size={20} /></div>
              <div>
                <strong>1 · Embed the question</strong>
                <p>Vertex AI <code>gemini-embedding-001</code> — 1536-dim vectors</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Database size={20} /></div>
              <div>
                <strong>2 · Hybrid retrieval</strong>
                <p>pgvector HNSW (dense) + BM25 (sparse), fused with Reciprocal Rank Fusion</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><GitMerge size={20} /></div>
              <div>
                <strong>3 · Rerank</strong>
                <p>Vertex AI Ranking API <code>semantic-ranker-default-004</code> → top 10</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Cpu size={20} /></div>
              <div>
                <strong>4 · Generate</strong>
                <p><code>gemini-3.5-flash</code> via ChatVertexAI, grounded in retrieved chunks</p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h2>Hybrid retrieval: dense + sparse, fused by rank</h2>
          <p>
            Dense vector search is great at paraphrase ("cost of tuition" finds
            "program fees") but weak on exact identifiers like <strong>COMP 1510</strong>.
            BM25 is the opposite. So the retriever runs both in parallel and merges
            them with <strong>Reciprocal Rank Fusion</strong>: each document scores{" "}
            <code>1 / (60 + rank)</code> in each list, weighted 50/50 between the
            two retrievers. RRF only looks at <em>ranks</em>, so there's no need to
            normalize incompatible similarity scores against BM25 scores.
          </p>
          <p>
            The dense side lives in <strong>Cloud SQL PostgreSQL with pgvector</strong>,
            indexed with HNSW. One production gotcha worth sharing: pgvector's default{" "}
            <code>ef_search</code> of 40 silently caps how many candidates an HNSW scan
            returns — our MMR retrieval requests 50 candidates, so we raise it to 100
            per session. The sparse side is an in-process BM25 index
            (<code>rank_bm25</code>) built over the same chunks, loaded from a pickle
            at startup — no extra search infrastructure to operate.
          </p>
          <p>
            Documents are chunked at 1,024 characters with 130 overlap, and embedded
            once by a resumable batch job — re-running it skips chunks already in the
            database, so an interrupted hour-long indexing run costs nothing to resume.
          </p>
        </section>

        <section>
          <h2>Reranking: the cheap accuracy win</h2>
          <p>
            Hybrid retrieval pulls a generous candidate pool (13 chunks), then the{" "}
            <strong>Vertex AI Ranking API</strong> (<code>semantic-ranker-default-004</code>)
            re-scores each candidate against the query with a cross-encoder and keeps
            the top 10. Cross-encoders read the query and the document <em>together</em>,
            so they catch relevance that bi-encoder embeddings miss — at a price that's
            only paid on a handful of candidates, not the whole corpus. Using the managed
            API means no GPU, no model weights, no serving stack: the whole pipeline
            stays CPU-only.
          </p>
        </section>

        <section>
          <h2>Generation and conversation memory</h2>
          <p>
            The final answer comes from <code>gemini-3.5-flash</code> through
            LangChain's <code>ChatVertexAI</code>, with the reranked chunks injected as
            grounding context. Each browser session gets its own
            five-turn conversation window, so follow-ups like{" "}
            <em>"and what about the part-time option?"</em> resolve correctly. Sessions
            expire after 30 minutes of inactivity. Auth is handled by Application
            Default Credentials end to end — there is not a single API key in the
            codebase or the environment.
          </p>
        </section>

        <section>
          <h2>Serving it</h2>
          <div className="infra-card">
            <div className="infra-icon"><Server size={20} /></div>
            <div>
              <p>
                Everything runs on one <strong>e2-standard-2 GCE VM</strong> — FastAPI +
                uvicorn behind systemd, with a companion <code>cloud-sql-proxy</code> unit
                providing an IAM-authenticated tunnel to Cloud SQL. The React frontend is
                a static Vite build served by the same FastAPI process. Chat requests run
                in a thread pool so the event loop never blocks on a RAG query, and the
                heavyweight pieces (embeddings, reranking, LLM) are all managed Vertex AI
                services, which is what makes a CPU-only box sufficient.
              </p>
            </div>
          </div>
        </section>

        <section className="blog-cta-section">
          <h2>See it answer for yourself</h2>
          <p>
            Ask it about admissions, program requirements, tuition, or campus life —
            and ask a follow-up to watch the conversation memory work.
          </p>
          <a href="/chat" className="blog-hero-cta">
            Open the chatbot <ArrowRight size={18} />
          </a>
        </section>
      </article>

      <footer className="blog-footer">
        <p>
          BCIT AI Advisor · RAG chatbot built on Google Cloud (Vertex AI, Cloud SQL
          pgvector) · Not affiliated with the British Columbia Institute of Technology
        </p>
      </footer>
    </div>
  );
}
