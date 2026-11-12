"""Vector DB schema + embedding pipeline on ChromaDB.

Schema (one Chroma collection per session, cosine HNSW index):
    document : chunk text
    embedding: OpenAI embedding vector
    metadata : session_id, source, section, chunk_index, start_index, n_chars
"""
import re
from typing import Dict, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PERSIST_DIR

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# HNSW parameters tuned for small/medium per-document collections.
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 128,
    "hnsw:search_ef": 64,
    "hnsw:M": 16,
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def collection_name(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)[:48]
    return f"paper_{safe}"


def build_documents(
    text: str,
    sections: Optional[Dict[str, str]] = None,
    source: str = "document",
    session_id: str = "",
) -> list:
    """Section-aware chunking: each chunk carries its section as metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    units = sections if sections else {"Full_Paper": text}
    docs, idx = [], 0
    for section, content in units.items():
        for d in splitter.split_documents([Document(page_content=content)]):
            d.metadata = {
                "session_id": session_id,
                "source": source,
                "section": section,
                "chunk_index": idx,
                "start_index": d.metadata.get("start_index", 0),
                "n_chars": len(d.page_content),
            }
            docs.append(d)
            idx += 1
    return docs


def create_vector_db(
    text: str,
    embedder: Embeddings,
    session_id: str = "default",
    sections: Optional[Dict[str, str]] = None,
    source: str = "document",
) -> Chroma:
    """Chunk -> embed (batched) -> store in this session's Chroma collection."""
    client = _get_client()
    name = collection_name(session_id)
    try:  # fresh index for a freshly uploaded paper
        client.delete_collection(name)
    except Exception:
        pass

    docs = build_documents(text, sections, source, session_id)
    vectordb = Chroma(
        client=client,
        collection_name=name,
        embedding_function=embedder,
        collection_metadata=COLLECTION_METADATA,
    )
    ids = [f"{name}-{d.metadata['chunk_index']}" for d in docs]
    vectordb.add_documents(docs, ids=ids)
    return vectordb


def delete_vector_db(session_id: str) -> None:
    try:
        _get_client().delete_collection(collection_name(session_id))
    except Exception:
        pass
