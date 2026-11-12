import { useRef, useState } from "react";
import { getSummary, sendChat, uploadPdf } from "./api.js";
import Chat from "./components/Chat.jsx";

export default function App() {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [topics, setTopics] = useState([]);
  const [topic, setTopic] = useState("");
  const [summary, setSummary] = useState("");
  const [summarizing, setSummarizing] = useState(false);

  async function handleFile(file) {
    if (!file) return;
    setError("");
    setUploading(true);
    setTopics([]);
    setSummary("");
    try {
      const data = await uploadPdf(file);
      setFileName(file.name);
      setTopics(data.topics);
      setTopic(data.topics[0] || "");
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSummarize() {
    if (!topic) return;
    setSummarizing(true);
    setSummary("");
    try {
      const data = await getSummary(topic);
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setSummarizing(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Research Paper Analyzer</h1>
        <p>Upload a paper, summarize its sections, and chat with it.</p>
      </header>

      <section
        className="card upload"
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {uploading
          ? "Processing and indexing PDF…"
          : fileName
          ? `Loaded: ${fileName} (click to replace)`
          : "Click or drop a PDF here"}
      </section>

      {error && <p className="error">{error}</p>}

      {topics.length > 0 && (
        <>
          <section className="card">
            <h2>Section summary</h2>
            <div className="row">
              <select value={topic} onChange={(e) => setTopic(e.target.value)}>
                {topics.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
              <button onClick={handleSummarize} disabled={summarizing}>
                {summarizing ? "Summarizing…" : "Summarize"}
              </button>
            </div>
            {summary && <pre className="summary">{summary}</pre>}
          </section>

          <Chat sendChat={sendChat} />
        </>
      )}
    </main>
  );
}
