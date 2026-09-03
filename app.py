import json
import os
import shutil
from typing import Dict, List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.create_vector_db import create_vector_db
from src.detect_and_split_sections import (
    refine_sections,
    split_sections_with_content,
)
from src.get_summary import generate_detailed_summary
from src.load_and_extract_text import extract_pdf_sections, extract_text_from_pdf
from src.RAG_retrival_chain import get_qa_chain, get_conversational_chain


load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Models
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Session state
full_text = ""
Research_paper_topics: Dict[str, str] = {}
vector_db = None

# Initialize FastAPI & Jinja2 Templates
app = FastAPI(title="Research Paper Analyzer")

# Create static dir if not existing so StaticFiles mount doesn't crash
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class SummaryRequest(BaseModel):
    topic: str


class ChatRequest(BaseModel):
    message: str



@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF file upload, text extraction, and section detection."""
    global full_text, Research_paper_topics, vector_db

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_text_from_pdf(file_path)
        full_text = extracted_text
        vector_db = None  # Reset index for fresh PDF

        extracted_sections = extract_pdf_sections(full_text=extracted_text)
        refined_sections = refine_sections(
            json.dumps(extracted_sections), llm=llm
        )
        section_with_content = split_sections_with_content(
            extracted_text, refined_sections
        )

        Research_paper_topics = section_with_content

        return {
            "message": "File processed successfully",
            "topics": list(Research_paper_topics.keys()),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/summary")
async def get_summary(payload: SummaryRequest):
    """Generates detailed summary for selected section."""
    global Research_paper_topics

    if not Research_paper_topics:
        raise HTTPException(
            status_code=400, detail="No paper has been uploaded yet."
        )

    topic_content = Research_paper_topics.get(payload.topic)
    if not topic_content:
        raise HTTPException(
            status_code=404, detail=f"Topic '{payload.topic}' not found."
        )

    try:
        summary = generate_detailed_summary(topic_content, llm)
        return {"topic": payload.topic, "summary": summary}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate summary: {str(e)}"
        )


conversational_chain = None


@app.post("/chat")
async def chat(payload: ChatRequest):
    """Multi-turn conversational RAG query against the uploaded paper."""
    global full_text, vector_db, conversational_chain

    if not full_text:
        raise HTTPException(
            status_code=400, detail="Please upload a PDF document first."
        )

    try:
        
        if not vector_db:
            vector_db = create_vector_db(text=full_text, embedder=embedder)

        
        if not conversational_chain:
            conversational_chain = get_conversational_chain(
                vectordb=vector_db, llm=llm
            )

        
        result = conversational_chain.invoke({"question": payload.message})

        ai_response = result.get("answer", "I don't know.")
        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve answer: {str(e)}"
        )


import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)