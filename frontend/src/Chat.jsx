import { useState, useRef, useEffect } from "react";
import { ArrowUp, ArrowLeft, ThumbsUp, ThumbsDown, Sparkles } from "lucide-react";
import "./App.css";

// Simple markdown-like formatting
function formatMessage(text) {
  if (!text) return "";

  const lines = text.split('\n');
  const elements = [];
  let inList = false;
  let listItems = [];
  let inSources = false;

  lines.forEach((line, idx) => {
    // Sources section: bold heading, then one "- url" link per line —
    // normalizes any "Document N [URL: ...]" wrappers the model may emit.
    if (line.replace(/[*#:\s]/g, '').toLowerCase() === 'sources') {
      if (inList) {
        elements.push(<ul key={`list-${idx}`}>{listItems}</ul>);
        listItems = [];
        inList = false;
      }
      inSources = true;
      elements.push(<p key={idx} className="msg-sources-heading">Sources</p>);
      return;
    }
    if (inSources) {
      if (!line.trim()) return;
      const urlMatch = line.match(/https?:\/\/[^\s\])"']+/);
      if (urlMatch) {
        const url = urlMatch[0].replace(/[.,;]+$/, '');
        elements.push(
          <p key={idx} className="msg-source-line">
            -&nbsp;<a href={url} target="_blank" rel="noopener noreferrer">{url}</a>
          </p>
        );
      } else {
        elements.push(<p key={idx} className="msg-source-line">{line.trim()}</p>);
      }
      return;
    }
    // Headers
    if (line.startsWith('### ')) {
      if (inList) {
        elements.push(<ul key={`list-${idx}`}>{listItems}</ul>);
        listItems = [];
        inList = false;
      }
      elements.push(<h4 key={idx}>{line.slice(4)}</h4>);
    } else if (line.startsWith('## ')) {
      if (inList) {
        elements.push(<ul key={`list-${idx}`}>{listItems}</ul>);
        listItems = [];
        inList = false;
      }
      elements.push(<h3 key={idx}>{line.slice(3)}</h3>);
    } else if (line.startsWith('# ')) {
      if (inList) {
        elements.push(<ul key={`list-${idx}`}>{listItems}</ul>);
        listItems = [];
        inList = false;
      }
      elements.push(<h2 key={idx}>{line.slice(2)}</h2>);
    }
    // Bullet points
    else if (line.match(/^[-*]\s/)) {
      inList = true;
      listItems.push(<li key={idx}>{formatInline(line.slice(2))}</li>);
    }
    // Numbered lists
    else if (line.match(/^\d+\.\s/)) {
      inList = true;
      listItems.push(<li key={idx}>{formatInline(line.replace(/^\d+\.\s/, ''))}</li>);
    }
    // Regular paragraph
    else {
      if (inList) {
        elements.push(<ul key={`list-${idx}`}>{listItems}</ul>);
        listItems = [];
        inList = false;
      }
      if (line.trim()) {
        elements.push(<p key={idx}>{formatInline(line)}</p>);
      }
    }
  });

  if (inList && listItems.length > 0) {
    elements.push(<ul key="list-final">{listItems}</ul>);
  }

  return elements;
}

// Format inline elements (markdown links, bold, bare URLs).
// Markdown links must be matched before bare URLs, or the URL inside
// [label](url) gets auto-linked with the closing ")" glued to the href.
function formatInline(text) {
  const tokenRe = /(\[[^\]]+\]\(https?:\/\/[^\s)]+\))|(\*\*[^*]+\*\*)|(https?:\/\/[^\s]+)/g;
  const elements = [];
  let last = 0;
  let key = 0;
  let m;
  while ((m = tokenRe.exec(text)) !== null) {
    if (m.index > last) {
      elements.push(text.slice(last, m.index));
    }
    if (m[1]) {
      const link = m[1].match(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/);
      elements.push(
        <a key={key++} href={link[2]} target="_blank" rel="noopener noreferrer">{link[1]}</a>
      );
    } else if (m[2]) {
      elements.push(<strong key={key++}>{m[2].slice(2, -2)}</strong>);
    } else {
      const url = m[3].replace(/[).,;:!?\]]+$/, '');
      elements.push(
        <a key={key++} href={url} target="_blank" rel="noopener noreferrer">{url}</a>
      );
      const trail = m[3].slice(url.length);
      if (trail) {
        elements.push(trail);
      }
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    elements.push(text.slice(last));
  }
  return elements;
}

const MAX_MESSAGE_CHARS = 2000;   // matches the backend cap (config.MAX_MESSAGE_CHARS)
const CLIENT_TIMEOUT_MS = 110000; // a beat above the server's CHAT_TIMEOUT_S
const STORAGE_KEY = "bcit-chat-v1";

const GREETING = {
  id: "1",
  text: "Hello! I'm the BCIT AI Advisor. Ask me about programs, courses, admissions, or campus life.",
  sender: "assistant"
};

const STARTER_QUESTIONS = [
  "What are the entrance requirements for Computer Systems Technology?",
  "How much is tuition for international students?",
  "What housing options does BCIT offer on the Burnaby campus?"
];

// Tab-scoped restore so a refresh doesn't wipe the conversation. Server
// sessions expire after 30 min; sending an expired id is safe — the backend
// re-adopts it as a fresh empty session.
function loadSavedChat() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY));
    if (!Array.isArray(saved?.messages) || saved.messages.length === 0) return {};
    return saved;
  } catch {
    return {};
  }
}

// Pull complete SSE frames out of the buffer; returns [events, remainder].
function parseSSEBuffer(buffer) {
  const events = [];
  const frames = buffer.split("\n\n");
  const remainder = frames.pop();
  for (const frame of frames) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) continue;
    try {
      events.push({ event, data: JSON.parse(dataLines.join("\n")) });
    } catch {
      // malformed frame — skip it
    }
  }
  return [events, remainder];
}

export default function Chat() {
  const [messages, setMessages] = useState(() => loadSavedChat().messages || [GREETING]);
  const [sessionId, setSessionId] = useState(() => loadSavedChat().sessionId || null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    try {
      sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ sessionId, messages: messages.slice(-40) })
      );
    } catch {
      // storage unavailable/full — chat still works, just not across refresh
    }
  }, [messages, sessionId]);

  // Append-or-update the assistant message for the in-flight turn, so
  // streaming deltas, the stats footer, and error text all land in one bubble.
  function upsertAssistant(id, updater) {
    setMessages(prev => {
      const idx = prev.findIndex(m => m.id === id);
      if (idx === -1) {
        return [...prev, updater({ id, text: "", sender: "assistant" })];
      }
      const next = [...prev];
      next[idx] = updater(next[idx]);
      return next;
    });
  }

  // Returns {handled: false} when the streaming endpoint is unavailable and
  // the caller should retry via plain POST /chat; throws on real errors.
  async function sendStreaming(text, assistantId) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);
    let res;
    try {
      res = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal
      });
    } catch (err) {
      clearTimeout(timer);
      if (err?.name === "AbortError") throw err;
      return { handled: false }; // network-level failure — try /chat once
    }

    const contentType = res.headers.get("content-type") || "";
    if (!res.ok || !res.body || !contentType.includes("text/event-stream")) {
      clearTimeout(timer);
      if (res.status === 404 || res.status === 405) return { handled: false }; // backend without /chat/stream
      let detail = "";
      try { detail = (await res.json()).detail || ""; } catch { /* not JSON */ }
      throw new Error(detail || `Request failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamedAny = false;
    let errorDetail = null;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const [events, remainder] = parseSSEBuffer(buffer);
        buffer = remainder;
        for (const ev of events) {
          if (ev.event === "session" && ev.data.session_id) {
            setSessionId(ev.data.session_id);
          } else if (ev.event === "delta") {
            streamedAny = true;
            upsertAssistant(assistantId, m => ({ ...m, text: m.text + ev.data.text }));
          } else if (ev.event === "done") {
            upsertAssistant(assistantId, m => ({ ...m, stats: ev.data.stats || null }));
          } else if (ev.event === "error") {
            errorDetail = ev.data.detail || "The request failed.";
          }
        }
      }
    } catch (err) {
      if (streamedAny) {
        upsertAssistant(assistantId, m => ({
          ...m,
          text: m.text + "\n\n(The answer was interrupted — please ask again.)"
        }));
        return { handled: true };
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }

    if (errorDetail) {
      if (!streamedAny) throw new Error(errorDetail);
      upsertAssistant(assistantId, m => ({
        ...m,
        text: m.text + "\n\n(The answer was interrupted — please ask again.)"
      }));
    } else if (!streamedAny) {
      return { handled: false }; // stream produced nothing — fall back once
    }
    return { handled: true };
  }

  async function sendBlocking(text, assistantId) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal
      });
      if (!res.ok) {
        let detail = "";
        try { detail = (await res.json()).detail || ""; } catch { /* not JSON */ }
        throw new Error(detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      if (data.session_id) setSessionId(data.session_id);
      upsertAssistant(assistantId, m => ({ ...m, text: data.reply, stats: data.stats || null }));
    } finally {
      clearTimeout(timer);
    }
  }

  async function handleSend(textOverride) {
    const text = (textOverride ?? input).trim();
    if (!text || isLoading) return;

    const now = Date.now();
    setMessages(prev => [...prev, { id: String(now), text, sender: "user" }]);
    if (textOverride === undefined) setInput("");
    setIsLoading(true);

    const assistantId = String(now + 1);
    try {
      const { handled } = await sendStreaming(text, assistantId);
      if (!handled) await sendBlocking(text, assistantId);
    } catch (err) {
      const aborted = err?.name === "AbortError";
      const fallbackText = aborted
        ? "The answer took too long — please try again."
        : err?.message && !err.message.startsWith("Request failed")
          ? err.message
          : "Sorry — something went wrong while answering. Please try again in a moment.";
      upsertAssistant(assistantId, m => ({ ...m, text: m.text || fallbackText }));
    }
    setIsLoading(false);
  }

  function questionFor(assistantId) {
    const idx = messages.findIndex(m => m.id === assistantId);
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].sender === "user") return messages[i].text;
    }
    return "";
  }

  async function sendFeedback(message, verdict) {
    setMessages(prev => prev.map(m => (m.id === message.id ? { ...m, feedback: verdict } : m)));
    try {
      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          verdict,
          question: questionFor(message.id),
          answer_excerpt: (message.text || "").slice(0, 1000)
        })
      });
    } catch {
      // feedback is best-effort
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const awaitingFirstDelta =
    isLoading && messages[messages.length - 1]?.sender === "user";

  return (
    <main id="chat-container">
      <header className="chat-header">
        <div>
          <h1>BCIT AI Advisor</h1>
          <p>An independent AI guide to BCIT&apos;s public web pages</p>
        </div>
        <a href="/" className="chat-home-link">
          <ArrowLeft size={16} />
          <span>About this project</span>
        </a>
      </header>

      <div className="chat-messages" aria-live="polite">
        {messages.map(m => (
          <div
            key={m.id}
            className={m.sender === "user" ? "msg msg-user" : "msg msg-advisor"}
          >
            <div className="msg-role">
              {m.sender === "user"
                ? "You"
                : <><Sparkles size={13} /> Advisor</>}
            </div>
            <div className="msg-content">
              {m.sender === "assistant" ? formatMessage(m.text) : m.text}
              {m.stats && (
                <div
                  className="msg-stats"
                  title={`input ${m.stats.input_tokens.toLocaleString()} tokens · output ${m.stats.output_tokens.toLocaleString()} tokens (incl. query rewriting) · cost includes search reranking and embedding fees`}
                >
                  <span>{m.stats.total_tokens.toLocaleString()} tokens</span>
                  <span className="msg-stats-sep">·</span>
                  <span>{m.stats.currency === "CAD" ? "CAD$" : "US$"}{m.stats.cost_usd.toFixed(4)}</span>
                  <span className="msg-stats-sep">·</span>
                  <span>{m.stats.latency_s.toFixed(1)}s</span>
                </div>
              )}
              {m.sender === "assistant" && m.stats && (
                <div className="msg-feedback">
                  <button
                    className={m.feedback === "up" ? "feedback-btn selected" : "feedback-btn"}
                    onClick={() => sendFeedback(m, "up")}
                    disabled={!!m.feedback}
                    aria-label="Helpful answer"
                    title="Helpful"
                  >
                    <ThumbsUp size={14} />
                  </button>
                  <button
                    className={m.feedback === "down" ? "feedback-btn selected" : "feedback-btn"}
                    onClick={() => sendFeedback(m, "down")}
                    disabled={!!m.feedback}
                    aria-label="Unhelpful answer"
                    title="Not helpful"
                  >
                    <ThumbsDown size={14} />
                  </button>
                  {m.feedback && <span className="feedback-thanks">Thanks!</span>}
                </div>
              )}
            </div>
          </div>
        ))}

        {messages.length === 1 && !isLoading && (
          <div className="chat-chips" aria-label="Suggested questions">
            {STARTER_QUESTIONS.map(q => (
              <button key={q} className="chat-chip" onClick={() => handleSend(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        {awaitingFirstDelta && (
          <div className="msg msg-advisor">
            <div className="msg-role"><Sparkles size={13} /> Advisor</div>
            <div className="msg-content typing-content">
              <span className="typing-dots">
                <span>.</span><span>.</span><span>.</span>
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <footer className="chat-footer">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about programs, admissions, campus life..."
          maxLength={MAX_MESSAGE_CHARS}
        />
        <button
          className="chat-send"
          onClick={() => handleSend()}
          disabled={isLoading || !input.trim()}
        >
          <ArrowUp size={22} />
        </button>
      </footer>

      {/* This is the page that actually serves BCIT-derived content, so the
          attribution and the accuracy warning belong here rather than only on
          the About page. Kept to one line so it stays out of the way. */}
      <p className="chat-disclaimer">
        Independent project — not affiliated with, endorsed by, or sponsored by
        BCIT &middot; AI-generated answers can be wrong or out of date; confirm
        anything that matters on{" "}
        <a href="https://www.bcit.ca" target="_blank" rel="noopener noreferrer">
          bcit.ca
        </a>{" "}
        &middot; Source content &copy; British Columbia Institute of Technology
      </p>
    </main>
  );
}
