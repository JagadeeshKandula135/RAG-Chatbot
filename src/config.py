"""Central configuration and model factories (OpenAI LLM + embeddings)."""
import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

UPLOAD_FOLDER = "uploads"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db_store")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", ".embedding_cache")

# Retrieval tuning
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "6"))

# Session management
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


def get_embedder() -> CacheBackedEmbeddings:
    """OpenAI embeddings behind a disk cache so repeated texts/queries are free
    and instant (lower latency + lower API cost)."""
    base = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        chunk_size=256,  # batch size per embeddings API call
    )
    return CacheBackedEmbeddings.from_bytes_store(
        base,
        LocalFileStore(EMBEDDING_CACHE_DIR),
        namespace=EMBEDDING_MODEL,
        query_embedding_cache=True,
        key_encoder="sha256",
    )
