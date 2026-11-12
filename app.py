import json
import os
import re
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    MAX_UPLOAD_MB,
    SESSION_TTL_SECONDS,
    UPLOAD_FOLDER,
    get_embedder,
    get_llm,
)
from src.create_vector_db import create_vector_db
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.get_summary import generate_detailed_summary
from src.load_and_extract_text import extract_pdf_sections, extract_text_from_pdf
from src.RAG_retrival_chain import get_conversational_chain
from src.session_store import Session, SessionStore

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Shared, stateless model clients (safe to reuse across sessions).
llm = get_llm()
embedder = get_embedder()

# All per-user state lives in the session store, not in module globals.
sessions = SessionStore(ttl_seconds=SESSION_TTL_SECONDS)

app = FastAPI(title="Research Paper Analyzer", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.0f}"
    return response


class SummaryRequest(BaseModel):
    topic: str


class ChatRequest(BaseModel):
    message: str


def require_session(session_id: Optional[str]) -> Session:
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(
            status_code=401, detail="Unknown or expired session. Please upload a PDF."
        )
    return s


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text + sections and index the paper in a new isolated session."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB.")

    session = sessions.create()
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(file.filename))
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(data)

    def process():
        text = extract_text_from_pdf(file_path)
        sections = extract_pdf_sections(full_text=text)
        refined = refine_sections(json.dumps(sections), llm=llm)
        topics = split_sections_with_content(text, refined)
        # Index at upload time so the first chat question is fast.
        db = create_vector_db(
            text=text,
            embedder=embedder,
            session_id=session.id,
            sections=topics,
            source=safe_name,
        )
        return text, topics, db

    try:
        text, topics, db = await run_in_threadpool(process)
    except Exception as e:
        sessions.delete(session.id)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass

    session.full_text, session.topics, session.vector_db = text, topics, db
    session.filename = safe_name
    session.chain = get_conversational_chain(vectordb=db, llm=llm)

    return {
        "session_id": session.id,
        "message": "File processed successfully",
        "topics": list(topics.keys()),
    }


@app.post("/api/summary")
async def get_summary(
    payload: SummaryRequest, x_session_id: Optional[str] = Header(None)
):
    session = require_session(x_session_id)
    content = session.topics.get(payload.topic)
    if not content:
        raise HTTPException(status_code=404, detail=f"Topic '{payload.topic}' not found.")
    try:
        summary = await run_in_threadpool(generate_detailed_summary, content, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")
    return {"topic": payload.topic, "summary": summary}


@app.post("/api/chat")
async def chat(payload: ChatRequest, x_session_id: Optional[str] = Header(None)):
    """Multi-turn conversational RAG against this session's indexed paper."""
    session = require_session(x_session_id)
    if not session.chain:
        raise HTTPException(status_code=400, detail="Document is not indexed yet.")

    def run():
        # Memory is per-session and mutable: serialize turns within a session
        # while different sessions run fully in parallel.
        with session.lock:
            t0 = time.perf_counter()
            result = session.chain.invoke({"question": payload.message})
            return result, (time.perf_counter() - t0) * 1000

    try:
        result, ms = await run_in_threadpool(run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {e}")

    sources = [
        {
            "section": d.metadata.get("section"),
            "chunk_index": d.metadata.get("chunk_index"),
            "snippet": d.page_content[:200],
        }
        for d in result.get("source_documents", [])
    ]
    return {
        "response": result.get("answer", "I don't know."),
        "sources": sources,
        "latency_ms": round(ms),
    }


@app.delete("/api/session")
def end_session(x_session_id: Optional[str] = Header(None)):
    return {"deleted": sessions.delete(x_session_id or "")}


# Serve the built React app (run `npm run build` in frontend/).
DIST = os.path.join("frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = os.path.join(DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
