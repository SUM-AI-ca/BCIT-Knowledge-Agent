import { useState, useRef, useEffect } from "react";
import { Send, ArrowLeft } from "lucide-react";
import Blog from "./Blog";
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

export default function App() {
  const path = window.location.pathname;
  if (path === "/chat" || path.startsWith("/chat/")) {
    return <Chat />;
  }
  return <Blog />;
}

function Chat() {
  const [messages, setMessages] = useState([
    {
      id: "1",
      text: "Hello! I'm the BCIT AI Advisor. Ask me about programs, courses, admissions, or campus life.",
      sender: "assistant",
      timestamp: new Date()
    }
  ]);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  async function handleSend() {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: input,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.text,
          session_id: sessionId
        })
      });

      const data = await res.json();

      if (data.session_id) {
        setSessionId(data.session_id);
      }

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        text: data.reply,
        sender: "assistant",
        stats: data.stats || null,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          text: "Something went wrong. Please make sure the backend server is running on port 8000.",
          sender: "assistant",
          timestamp: new Date()
        }
      ]);
    }

    setIsLoading(false);
  }

  function handleKeyPress(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <main id="chat-container">
      <header className="chat-header">
        <div>
          <h1>BCIT AI Advisor</h1>
          <p>Ask me about anything in BCIT</p>
        </div>
        <a href="/" className="chat-home-link">
          <ArrowLeft size={16} />
          <span>About this project</span>
        </a>
      </header>

      <div className="chat-messages">
        {messages.map(m => (
          <div
            key={m.id}
            className={m.sender === "user" ? "msg-row user-row" : "msg-row assistant-row"}
          >
            <div className={m.sender === "user" ? "msg-bubble user-bubble" : "msg-bubble assistant-bubble"}>
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
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="msg-row assistant-row">
            <div className="msg-bubble assistant-bubble typing-bubble">
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
          onKeyPress={handleKeyPress}
          placeholder="Ask about programs, admissions, campus life..."
          disabled={isLoading}
        />
        <button
          className="chat-send"
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
        >
          <Send size={20} />
        </button>
      </footer>
    </main>
  );
}
