import { useEffect, useRef, useState } from "react";

export default function Chat({ sendChat }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const data = await sendChat(text);
      setMessages((m) => [
        ...m,
        {
          role: "ai",
          text: data.response,
          sources: data.sources,
          latency: data.latency_ms,
        },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "ai", text: err.message, failed: true }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card chat">
      <h2>Chat with the paper</h2>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.failed ? " failed" : ""}`}>
            <div className="bubble">{m.text}</div>
            {m.sources?.length > 0 && (
              <details>
                <summary>
                  {m.sources.length} sources · {m.latency} ms
                </summary>
                <ul>
                  {m.sources.map((s, j) => (
                    <li key={j}>
                      <strong>{s.section}</strong>: {s.snippet}…
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        {busy && <div className="msg ai"><div className="bubble">Thinking…</div></div>}
        <div ref={endRef} />
      </div>
      <form className="row" onSubmit={submit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the paper…"
        />
        <button disabled={busy}>Send</button>
      </form>
    </section>
  );
}
