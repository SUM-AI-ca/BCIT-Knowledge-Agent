import { MessageSquare, ArrowRight, Database, Layers, Cpu, GitMerge, Server, Search, Zap, FileText } from "lucide-react";
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
          How query decomposition, hybrid retrieval, a managed reranker, and Gemini
          turn 11,000+ pages of BCIT program and course documentation into an
          academic advisor that answers in ~5 seconds on a CPU-only VM — at 87%
          less LLM cost than the first production version.
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
              <div className="arch-icon"><Search size={20} /></div>
              <div>
                <strong>1 · Rewrite & decompose</strong>
                <p>One schema-constrained JSON call resolves pronouns from history and splits multipart questions into up to 4 self-contained sub-queries</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Layers size={20} /></div>
              <div>
                <strong>2 · Embed</strong>
                <p>Vertex AI <code>gemini-embedding-001</code> — 1536-dim vectors, per sub-query</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Database size={20} /></div>
              <div>
                <strong>3 · Hybrid retrieval</strong>
                <p>pgvector HNSW (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion — multipart questions fan out in parallel threads</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><GitMerge size={20} /></div>
              <div>
                <strong>4 · Pooled rerank</strong>
                <p>ONE Vertex AI Ranking API call scores the merged pool; a coverage quota guarantees every sub-question keeps its best evidence → top 10</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Cpu size={20} /></div>
              <div>
                <strong>5 · Assemble & generate</strong>
                <p>Selected chunks + their document neighbors (small-to-big) go to <code>gemini-3.5-flash</code> — ~6k input tokens instead of whole source files</p>
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
            <code>1 / (60 + rank)</code> in each list, weighted between the
            two retrievers. RRF only looks at <em>ranks</em>, so there's no need to
            normalize incompatible similarity scores against BM25 scores.
          </p>
          <p>
            Multipart questions get special treatment: <em>"admission requirements,
            tuition, AND housing?"</em> decomposes into three self-contained
            sub-queries that retrieve <strong>in parallel</strong> on a shared thread
            pool, then merge into one deduplicated candidate pool. Single questions
            skip the fan-out and keep the full-width single retrieval.
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
            Hybrid retrieval pulls a generous candidate pool (25 chunks for single
            questions, up to 40 pooled across sub-queries), then the{" "}
            <strong>Vertex AI Ranking API</strong> (<code>semantic-ranker-default-004</code>)
            re-scores each candidate against the query with a cross-encoder and keeps
            the top 10. Cross-encoders read the query and the document <em>together</em>,
            so they catch relevance that bi-encoder embeddings miss — at a price that's
            only paid on a handful of candidates, not the whole corpus.
          </p>
          <p>
            The API bills <em>per query</em>, so the pipeline makes exactly{" "}
            <strong>one ranking call per turn</strong> no matter how many sub-queries
            fan out: all candidates are scored in a single pooled request, and a
            per-sub-query <strong>coverage quota</strong> swaps each sub-question's
            best evidence back in if the global top-10 squeezed it out. Using the
            managed API means no GPU, no model weights, no serving stack: the whole
            pipeline stays CPU-only.
          </p>
        </section>

        <section>
          <h2>Cutting LLM cost 87% — measured, not vibes</h2>
          <p>
            The first production version injected <em>entire source documents</em> into
            the prompt (whichever files the top chunks came from) — safe for answer
            completeness, but ~44,000 input tokens per query. The June 2026 overhaul
            replaced that with the retrieved chunks themselves plus their{" "}
            <strong>document neighbors</strong> (±2 adjacent chunks, resolved from an
            in-process index — no DB migration), deduplicating the chunk overlaps as
            runs merge. A 40-case golden set with retrieval hit-rate, key-fact recall,
            citation precision, token, and latency metrics gated every change:
          </p>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><Zap size={20} /></div>
              <div>
                <strong>−87% input tokens</strong>
                <p>44,322 → 5,884 per query (mean), thinking tokens 896 → 0</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Search size={20} /></div>
              <div>
                <strong>Accuracy went up</strong>
                <p>Retrieval hit-rate 0.892 → 0.963 · key-fact recall 0.848 → 0.885</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Cpu size={20} /></div>
              <div>
                <strong>2× faster</strong>
                <p>p95 latency 10.8s → 5.2s (p50 4.1s), zero truncated answers</p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><FileText size={20} /></div>
              <div>
                <strong>Fully reversible</strong>
                <p>The legacy pipeline is one env flag away — every change ships behind a switch</p>
              </div>
            </div>
          </div>
          <p>
            Two findings the eval surfaced that intuition wouldn't have: Gemini's
            thinking tokens count <em>against</em> <code>max_output_tokens</code> (the
            default thinking budget silently truncated every capped answer — and for
            this extract-and-summarize workload, disabling thinking measured{" "}
            <em>better</em> on recall and halved latency); and precise chunks beat
            9,000-token document dumps on factual recall — more context was actively
            hurting retrieval-grounded answers.
          </p>
        </section>

        <section>
          <h2>Generation and conversation memory</h2>
          <p>
            The final answer comes from <code>gemini-3.5-flash</code> through
            LangChain's <code>ChatVertexAI</code>, grounded in the assembled chunk
            context. The prompt puts the static instructions first and the variable
            inputs last, so Gemini's implicit caching can reuse the shared prefix;
            multipart questions get their sub-questions enumerated so every part is
            answered. Each browser session gets its own five-turn conversation
            window — saved history is stripped of citation lists and capped, since
            it is re-sent on every turn. Follow-ups like{" "}
            <em>"and what about the part-time option?"</em> resolve correctly.
            Sessions expire after 30 minutes of inactivity. Auth is handled by
            Application Default Credentials end to end — there is not a single API
            key in the codebase or the environment.
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
