import {
  MessageSquare, ArrowRight, Database, Layers, Cpu, GitMerge, Server,
  Search, FileText, ShieldCheck, BookOpen, Languages, Scale
} from "lucide-react";
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
        <p className="blog-kicker">About this project</p>
        <h1>An AI academic advisor, grounded in BCIT's own pages</h1>
        <p className="blog-subtitle">
          Ask about programs, admission requirements, courses, tuition, or campus
          life — in your own words. Every answer is built from 11,000+ official
          BCIT web pages retrieved at question time, cites its sources, and shows
          exactly what it cost to produce.
        </p>
        <a href="/chat" className="blog-hero-cta">
          Try the live chatbot <ArrowRight size={18} />
        </a>
      </header>

      <article className="blog-article">
        <section>
          <h2>What it does</h2>
          <p>
            BCIT's website spans hundreds of programs and thousands of course
            pages. A prospective student asking <em>"What math do I need for the
            Computing diploma, and can I transfer in credits from another
            school?"</em> normally has to stitch that answer together from a dozen
            different pages. This advisor does the stitching: it reads the
            question, finds the relevant official pages, and writes one direct
            answer — with links to every page it used, so anything can be
            double-checked in a click.
          </p>
          <p>
            It handles real questions the way people actually type them:
            multi-part questions ("requirements, tuition, and housing?"),
            follow-ups that lean on the conversation ("and the part-time
            option?"), abbreviations ("CIT entrance reqs?"), typos, and questions
            asked in another language.
          </p>
        </section>

        <section>
          <h2>Why you can trust the answers</h2>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><BookOpen size={20} /></div>
              <div>
                <strong>Grounded, not guessed</strong>
                <p>
                  The AI is only allowed to state BCIT facts that appear in the
                  retrieved pages. If the documents don't contain the answer, it
                  says so instead of inventing one.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><FileText size={20} /></div>
              <div>
                <strong>Sources on every answer</strong>
                <p>
                  Each reply ends with the bcit.ca pages it drew from. In
                  benchmark testing, 100% of cited pages were genuinely part of
                  the retrieved evidence — no fabricated links.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Scale size={20} /></div>
              <div>
                <strong>Measured accuracy</strong>
                <p>
                  Quality is tracked with two hand-verified test sets — one of
                  clean questions, one of messy real-world phrasing (typos,
                  acronyms, other languages). Current results: the right page is
                  retrieved 97–100% of the time across the two benchmarks, and
                  92–95% of expected key facts appear in answers.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><ShieldCheck size={20} /></div>
              <div>
                <strong>Transparent cost</strong>
                <p>
                  Every reply shows its own token usage and price — well under
                  half a cent per question — so the economics of running this at
                  student-services scale are visible in the product itself.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h2>How it works</h2>
          <p>
            Under the hood this is a retrieval-augmented generation (RAG) system:
            instead of asking an AI model to remember BCIT, the pipeline fetches
            the relevant institutional documents for each question and instructs
            the model to answer only from them.
          </p>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><Search size={20} /></div>
              <div>
                <strong>1 · Understand the question</strong>
                <p>
                  A language model rewrites the question into search-ready form:
                  pronouns are resolved from the conversation, abbreviations are
                  expanded, typos corrected, other languages translated, and
                  multi-part questions split into up to four self-contained
                  sub-queries.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Database size={20} /></div>
              <div>
                <strong>2 · Search two ways at once</strong>
                <p>
                  Every sub-query runs a semantic vector search (meaning-based,
                  great at paraphrase) and a keyword search (exact codes like
                  "COMP 1510") in parallel over 100,000+ passages, fused by rank.
                  Both indexes carry each passage's page identity — the keyword
                  index and the meaning vectors alike — so "entrance
                  requirements" for one program doesn't drown in the same section
                  from 528 sibling program pages.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><GitMerge size={20} /></div>
              <div>
                <strong>3 · Verify with a second opinion</strong>
                <p>
                  A dedicated semantic ranking model re-scores the pooled
                  candidates against the question in a single call, and a
                  coverage rule guarantees every part of a multi-part question
                  keeps its best evidence in the final selection.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Layers size={20} /></div>
              <div>
                <strong>4 · Assemble the evidence</strong>
                <p>
                  The winning passages are expanded with their neighboring
                  passages from the same page, so the model sees complete
                  sections rather than fragments — typically ~6,000 tokens of
                  tightly relevant context.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Cpu size={20} /></div>
              <div>
                <strong>5 · Write the grounded answer</strong>
                <p>
                  A language model composes the reply from that evidence only,
                  enumerating each part of multi-part questions, and appends the
                  source links. Each browser session keeps a private five-turn
                  conversation memory so follow-ups resolve naturally.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h2>Engineering notes</h2>
          <p>
            For the technically curious, the load-bearing details. Retrieval
            fuses dense vectors (PostgreSQL + pgvector, HNSW index) with an
            in-process BM25 index using <strong>Reciprocal Rank Fusion</strong> —
            rank-based fusion sidesteps normalizing incompatible score scales.
            Both arms are <em>identity-aware</em>: the BM25 index is fit on
            title-augmented text, and the corpus is embedded with each passage's
            page title and category prefixed to the embedded text — while the
            served passages stay byte-identical. Giving deep chunks their page
            identity took multi-part retrieval to 1.000 on the clean benchmark
            and the messy-query benchmark to a perfect page hit rate.
          </p>
          <p>
            The semantic reranker is called <strong>once per question</strong> on
            the merged candidate pool — one billed call regardless of how many
            sub-queries fanned out. Multi-part questions retrieve their
            sub-queries in parallel threads, and a per-sub-query quota prevents
            any one part from monopolizing the context.
          </p>
          <p>
            Every change to this pipeline ships behind a configuration flag and
            must pass a gated evaluation before adoption: retrieval hit rate,
            key-fact recall, citation precision, cost, and latency, measured on
            both test sets and archived with the code. Several intuitive
            "improvements" — bigger candidate pools, cheaper rewrite models,
            query-expansion tricks, a rerank-skipping shortcut — measured worse
            and were rejected; the numbers, not the vibes, decide.
          </p>
        </section>

        <section>
          <h2>What it costs to run</h2>
          <p>
            About <strong>$0.004 per question</strong> — roughly $4 per thousand
            student questions — with answers typically arriving in 3–5 seconds.
            Everything runs on a single small CPU-only virtual machine: the
            heavyweight AI stages (embeddings, ranking, generation) are managed
            services on Google Cloud's Gemini Enterprise Agent Platform
            (formerly Vertex AI), authenticated end-to-end with workload
            identity. There is not a single API key in the codebase or the
            environment.
          </p>
          <div className="infra-card">
            <div className="infra-icon"><Server size={20} /></div>
            <div>
              <p>
                One GCE VM runs FastAPI + uvicorn behind systemd, with a
                companion cloud-sql-proxy unit providing an IAM-authenticated
                tunnel to Cloud SQL. The React frontend is a static build served
                by the same process. Chat requests run in a thread pool so the
                event loop never blocks, sessions are isolated per browser and
                expire after 30 minutes, and the corpus is versioned for
                blue-green reindexing with instant rollback.
              </p>
            </div>
          </div>
        </section>

        <section className="blog-cta-section">
          <h2>See it answer for yourself</h2>
          <p>
            Ask about admissions, program requirements, tuition, or campus life —
            then ask a follow-up, try an abbreviation, or ask in another language.
            Check the sources and the cost line under every reply.
          </p>
          <a href="/chat" className="blog-hero-cta">
            Open the chatbot <ArrowRight size={18} />
          </a>
        </section>
      </article>

      <footer className="blog-footer">
        <p>
          BCIT AI Advisor · Retrieval-augmented chatbot built on Google Cloud
          (Gemini Enterprise Agent Platform, Cloud SQL pgvector) · Not
          affiliated with the British Columbia Institute of Technology
        </p>
        <p>
          Made by{" "}
          <a href="https://sumai.ca" target="_blank" rel="noopener noreferrer">
            SUM AI
          </a>
        </p>
      </footer>
    </div>
  );
}
