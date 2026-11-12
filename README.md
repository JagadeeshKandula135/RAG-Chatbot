# RAG Chatbot

Full-stack Retrieval-Augmented Generation (RAG) chatbot for research papers, built with **React**, **FastAPI**, **LangChain**, **ChromaDB** and the **OpenAI API**. Upload a PDF, get section summaries, and hold context-aware multi-turn conversations grounded in the document.

## Architecture

```
React (Vite)  ──REST/JSON──▶  FastAPI  ──▶  LangChain ConversationalRetrievalChain
 X-Session-ID header            │                 │                 │
                                │          OpenAI chat model   MMR retriever
                         SessionStore                              │
                      (per-user state, TTL)            ChromaDB (cosine HNSW,
                                                      one collection per session)
                                                                   ▲
                                         OpenAI embeddings (batched + disk-cached)
```

* **Scalable REST design**: no global state. Each upload creates a session (`X-Session-ID`) holding its own text, sections, Chroma collection and conversation memory. Sessions expire after a TTL and their collections are deleted. Blocking LLM/embedding work runs in a threadpool, so sessions are served concurrently; turns within one session are serialized.
* **Vector DB schema** (`src/create_vector_db.py`): one collection per session using a cosine HNSW index (tuned `M`, `construction_ef`, `search_ef`). Each chunk stores `session_id`, `source`, `section`, `chunk_index`, `start_index`, `n_chars` as metadata. Chunking is section-aware.
* **Embedding pipeline**: `RecursiveCharacterTextSplitter` (800/120) → OpenAI `text-embedding-3-small`, batched, behind a `CacheBackedEmbeddings` disk cache (documents and queries).
* **Retrieval / multi-turn context**: MMR search (fetch 20 by similarity, return 4 diverse), a question-condensing prompt to resolve follow-ups, and windowed conversation memory per session.
* **Latency & optimization**: indexing happens at upload so the first question is fast; embedding cache; per-request `latency_ms` and an `X-Process-Time-Ms` header; `benchmark.py` measures chunking/reranking strategies on Hit Rate@3, MRR, accuracy and latency.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/upload` | Upload PDF → returns `session_id` and detected sections |
| POST | `/api/summary` | Summarize a section (`X-Session-ID`) |
| POST | `/api/chat` | Multi-turn RAG answer + sources + latency |
| DELETE | `/api/session` | End the session and drop its vectors |
| GET | `/api/health` | Health check |

## Setup

```bash
pip install -r requirement.txt
cp .env.example .env        # set OPENAI_API_KEY

# Frontend
cd frontend && npm install && npm run build && cd ..

# Run (serves the built React app and the API)
python app.py               # http://127.0.0.1:8000

# Frontend dev server with hot reload (API proxied to :8000)
cd frontend && npm run dev  # http://localhost:5173
```

Optional benchmark: `python benchmark.py`

## Project structure

```text
app.py                      FastAPI app, REST routes, session handling
benchmark.py                Retrieval evaluation (chunking, reranking)
frontend/                   React (Vite) UI
src/config.py               Settings, OpenAI LLM + cached embeddings
src/session_store.py        Per-session state with TTL eviction
src/create_vector_db.py     Chunking, metadata schema, Chroma collections
src/RAG_retrival_chain.py   MMR retriever, prompts, conversational chain
src/get_summary.py          Section summarization
src/detect_and_split_sections.py, src/load_and_extract_text.py   PDF parsing
```
