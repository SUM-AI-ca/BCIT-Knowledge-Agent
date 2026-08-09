import {
  MessageSquare, ArrowRight, Database, Layers, Cpu, GitMerge, Server,
  Search, FileText, ShieldCheck, BookOpen, Scale, GitBranch, Repeat
} from "lucide-react";
import "./Blog.css";

export default function Blog() {
  return (
    <div className="blog-page">
      <nav className="blog-nav">
        <span className="blog-nav-brand">BCIT Knowledge Agent</span>
        <a href="/chat" className="blog-nav-cta">
          <MessageSquare size={17} />
          <span>Open Chatbot</span>
        </a>
      </nav>

      <header className="blog-hero">
        <p className="blog-kicker">About this project</p>
        <h1>An AI academic advisor, grounded in BCIT's own pages</h1>
        <p className="blog-subtitle">
          A retrieval-augmented chatbot answering questions about BCIT programs,
          courses, admission and campus services from 11,129 official pages —
          100,515 indexed passages — fetched at question time. Every answer
          cites the pages it used. Every claim about its quality on this page is
          a measured number, taken from benchmark runs committed alongside the
          code.
        </p>
        <a href="/chat" className="blog-hero-cta">
          Try the live chatbot <ArrowRight size={18} />
        </a>
      </header>

      <article className="blog-article">
        <section>
          <h2>The problem</h2>
          <p>
            BCIT's website spans hundreds of programs and thousands of course
            pages. A question like <em>"What math do I need for the Computing
            diploma, and can I transfer credits in?"</em> is answered across a
            dozen of them. This system does the stitching: it decides whether a
            question needs the corpus at all, finds the relevant pages, and
            writes one direct answer with a link to everything it used. It
            handles multi-part questions, follow-ups that depend on the
            conversation, abbreviations, typos, and questions asked in another
            language.
          </p>
        </section>

        <section>
          <h2>Measured quality</h2>
          <p>
            Four hand-verified benchmark sets, each written against the actual
            corpus so every expected page and fact was confirmed to exist. The
            sets get progressively less forgiving:
          </p>
          <div className="metrics-table-wrap">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Benchmark</th>
                  <th>Correct page retrieved</th>
                  <th>Key facts present</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Standard questions <span>(40)</span></td>
                  <td>97.5%</td>
                  <td>100%</td>
                </tr>
                <tr>
                  <td>Messy phrasing <span>(25)</span></td>
                  <td>100%</td>
                  <td>98.0%</td>
                </tr>
                <tr>
                  <td>Targeted edge cases <span>(24)</span></td>
                  <td>97.8%</td>
                  <td>94.3%</td>
                </tr>
                <tr>
                  <td>Adversarial phrasing <span>(16)</span></td>
                  <td>83.3%</td>
                  <td>94.4%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="metrics-note">
            Messy phrasing covers acronyms, typos and non-English questions.
            Targeted edge cases cover multi-hop reasoning, unanswerable
            questions and out-of-scope requests. Adversarial phrasing is the
            newest and hardest set — the same question classes written the way
            people actually type, <code>"comp 4537 prereqs of prereqs?"</code>{" "}
            rather than a well-formed sentence — and it is deliberately where
            the numbers are worst. Across all four sets, every cited link came
            from evidence the system had actually retrieved (citation precision
            0.98–1.00), and it declined every out-of-scope and unanswerable
            question rather than inventing an answer.
          </p>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><BookOpen size={20} /></div>
              <div>
                <strong>Grounded, not remembered</strong>
                <p>
                  The model may only state BCIT facts that appear in the pages
                  retrieved for that question. When the corpus does not contain
                  the answer, it says so — including for questions it has to
                  search first to know are unanswerable, such as graduation
                  rates.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><FileText size={20} /></div>
              <div>
                <strong>Sources on every answer</strong>
                <p>
                  Each reply ends with the bcit.ca pages behind it, follow-ups
                  included. A turn that could have answered from conversation
                  history alone still retrieves, because a fact without a
                  citable source is not one this system will assert.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Scale size={20} /></div>
              <div>
                <strong>Gated adoption</strong>
                <p>
                  Every pipeline change ships behind a configuration flag and
                  must clear a pre-registered evaluation — retrieval hit rate,
                  key-fact recall, citation precision, refusal behaviour, cost
                  and latency — on all four sets before it is turned on. Runs
                  are archived with the commit that adopted them.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><ShieldCheck size={20} /></div>
              <div>
                <strong>Transparent cost</strong>
                <p>
                  Every reply reports its own token usage and price, computed
                  from live per-token rates rather than a quoted average, so the
                  economics of running this at student-services scale are
                  visible in the product itself.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h2>How it works</h2>
          <p>
            Rather than asking a model to remember BCIT, the pipeline retrieves
            the relevant institutional pages for each question and constrains
            the model to answer from them. An LLM controller decides whether to
            search, whether what came back is enough, and what to search for
            next.
          </p>
          <div className="arch-diagram">
            <div className="arch-step">
              <div className="arch-icon"><GitBranch size={20} /></div>
              <div>
                <strong>1 · Decide whether to search at all</strong>
                <p>
                  One schema-constrained call routes the turn: answer directly
                  (greetings, questions about the assistant), decline as outside
                  BCIT's scope, or retrieve. On the targeted benchmark, 6 of 24
                  turns skip retrieval entirely — no rewrite, no search, no
                  billed ranking call.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Search size={20} /></div>
              <div>
                <strong>2 · Understand the question</strong>
                <p>
                  A model rewrites it into search-ready form: pronouns resolved
                  from the conversation, abbreviations expanded, typos
                  corrected, other languages translated, and multi-part
                  questions split into up to four self-contained sub-queries.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Database size={20} /></div>
              <div>
                <strong>3 · Search two ways at once</strong>
                <p>
                  Each sub-query runs a semantic vector search (strong on
                  paraphrase) and a keyword search (exact identifiers like
                  "COMP 1510") in parallel, fused by rank. Both indexes carry
                  each passage's page identity, so "entrance requirements" for
                  one program does not drown in the same section from 528
                  sibling pages. A third arm scopes directly to a named course,
                  program or instructor when the question mentions one.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><GitMerge size={20} /></div>
              <div>
                <strong>4 · Re-rank with a second opinion</strong>
                <p>
                  A dedicated semantic ranking model re-scores the pooled
                  candidates against the question in a single call, and a
                  coverage rule guarantees every part of a multi-part question
                  keeps its best evidence.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Repeat size={20} /></div>
              <div>
                <strong>5 · Check coverage, search again if needed</strong>
                <p>
                  The controller re-reads a compact digest of what came back —
                  page identities plus the structured{" "}
                  <code>Prerequisite(s) | …</code> lines course outlines carry
                  (present on 3,258 of 3,262 outlines) — and names what is still
                  missing. <em>"What are the prerequisites of COMP 4537's own
                  prerequisites?"</em> takes two passes: the first learns which
                  courses they are, the second looks those up. Multi-hop page
                  retrieval went from 60.0% to 93.3% when this was added.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Layers size={20} /></div>
              <div>
                <strong>6 · Assemble the evidence</strong>
                <p>
                  The selected passages are expanded with their neighbours from
                  the same page, so the model sees whole sections rather than
                  fragments. Turns that took a second pass get a wider context
                  budget; turns that route straight through pay nothing extra.
                </p>
              </div>
            </div>
            <div className="arch-step">
              <div className="arch-icon"><Cpu size={20} /></div>
              <div>
                <strong>7 · Write the grounded answer</strong>
                <p>
                  A model composes the reply from that evidence only, addresses
                  each part of a multi-part question, and appends the source
                  links. The answer streams as it is written. Each browser
                  session keeps a private five-turn memory so follow-ups resolve
                  naturally.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h2>Engineering notes</h2>
          <p>
            Retrieval fuses dense vectors (PostgreSQL with pgvector, HNSW index)
            and an in-process BM25 index using{" "}
            <strong>Reciprocal Rank Fusion</strong>, which compares ranks and so
            sidesteps normalising incompatible score scales. Both arms are{" "}
            <em>identity-aware</em>: the BM25 index is fit on title-augmented
            text and the corpus is embedded with each passage's page title and
            category prefixed, while the passages actually served stay
            byte-identical. Giving deep chunks their page identity is what took
            multi-part retrieval to 100% on the standard set. A separate
            in-process index maps 1,291 instructors to the pages that name them{" "}
            <em>as the instructor</em> — deliberately excluding the approval
            signatures that name the same people in a different role, which
            outnumber the instructor mentions roughly fifteen to one.
          </p>
          <p>
            The semantic re-ranker is called <strong>once per question</strong>{" "}
            over the merged candidate pool — one billed call however many
            sub-queries fanned out. The controller loop runs entirely{" "}
            <em>before</em> generation, which is what keeps a retrieval loop
            compatible with token streaming: by the time the first token leaves,
            routing and any second pass have already settled, so the answer is
            still one uninterrupted stream.
          </p>
          <p>
            Two of the controller's constraints had to be enforced in code
            rather than asked for in the prompt. A follow-up may not answer from
            conversation history — five benchmark cases lost their citations
            that way, and two prompt revisions took it from five to two, not to
            zero. And a second search is refused unless the controller can name
            a concrete target, which is what stops an unanswerable question from
            looping. The model's reasoning in both cases was not wrong on its
            own terms; it simply did not own the requirement. A prompt states a
            preference, and only code states a requirement.
          </p>
          <p>
            Measurement decides, and it has rejected more than it has adopted:
            bigger candidate pools, cheaper rewrite models, query-expansion
            tricks, and a re-rank-skipping shortcut all measured worse. A
            candidate-retention rule intended to fix multi-answer questions
            fired zero times in twelve runs, because its premise had already
            been made false by an earlier change. Folding the router into an
            existing model call — the obvious way to claw back its cost — saves
            $0.00035 on searched turns and costs $0.00053 on exactly the turns
            routing was supposed to make cheap, netting between nothing and six
            per cent; it was costed and dropped rather than built.
          </p>
          <p>
            The most useful finding was about the tests themselves. The
            adversarial-phrasing set was added after a bug reached production
            that every tidy benchmark scored perfectly, and it immediately found
            another: the deployed system was answering{" "}
            <code>"how do i center a div in css"</code> with flexbox
            instructions, while the well-formed version of the same out-of-scope
            question was declined every time. A guard set written in tidy
            phrasing is itself an overfitting instrument.
          </p>
        </section>

        <section>
          <h2>Performance and cost</h2>
          <p>
            Between $0.0045 and $0.0062 per question depending on the benchmark
            — the exact figure is measured and shown under every answer. Three
            models are priced separately: the controller, the query rewriter and
            the answer generator. Measured on the live endpoint, a
            retrieval-backed answer begins streaming in about 7 seconds and
            completes in 7 to 8; a question needing a second retrieval pass
            takes around 10; a directly routed reply arrives in about 2, since
            it skips search and ranking altogether. An exact repeat of an
            earlier question returns from cache in under 100 ms at no API cost.
          </p>
          <p>
            Everything runs on a single small CPU-only virtual machine. The
            heavyweight stages — embeddings, ranking, generation — are managed
            services on Google Cloud's Gemini Enterprise Agent Platform
            (formerly Vertex AI), authenticated end to end with workload
            identity. There is no API key in the codebase or the environment.
          </p>
          <div className="infra-card">
            <div className="infra-icon"><Server size={20} /></div>
            <div>
              <p>
                One GCE VM runs FastAPI and uvicorn under systemd, with a
                companion cloud-sql-proxy unit providing an IAM-authenticated
                tunnel to Cloud SQL. The React frontend is a static build served
                by the same process. Chat requests run in a thread pool so the
                event loop never blocks, sessions are isolated per browser and
                expire after 30 minutes, and the corpus is versioned for
                blue-green reindexing with an instant rollback.
              </p>
            </div>
          </div>
        </section>

        <section className="blog-cta-section">
          <h2>See it answer for yourself</h2>
          <p>
            Ask about admissions, program requirements, tuition or campus life,
            then try a follow-up, an abbreviation, or a question in another
            language. Check the sources and the cost line under each reply.
          </p>
          <a href="/chat" className="blog-hero-cta">
            Open the chatbot <ArrowRight size={18} />
          </a>
        </section>
      </article>

      <footer className="blog-footer">
        <p>
          BCIT Knowledge Agent · Retrieval-augmented chatbot built on Google Cloud
          (Gemini Enterprise Agent Platform, Cloud SQL pgvector) · Not
          affiliated with, endorsed by, or sponsored by the British Columbia
          Institute of Technology
        </p>
        <p>
          Answers are AI-generated and may be wrong or out of date — confirm
          anything that matters on{" "}
          <a href="https://www.bcit.ca" target="_blank" rel="noopener noreferrer">
            bcit.ca
          </a>
          . Source content © British Columbia Institute of Technology, used for
          informational, non-commercial purposes. &ldquo;BCIT&rdquo; and
          &ldquo;British Columbia Institute of Technology&rdquo; are registered
          trade marks of BCIT.
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
