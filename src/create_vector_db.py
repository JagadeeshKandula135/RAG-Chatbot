import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = "chroma_db_store"


def create_vector_db(text: str, embedder: Embeddings) -> Chroma:
    """Splits input text into chunks, generates vector embeddings,

    and stores them in a persistent Chroma vector database.
    """
    # Optional: Clear previous document collection if storing single session
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # 1. Wrap raw text in Document object
    doc = Document(page_content=text)

    # 2. Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    docs = splitter.split_documents([doc])
    print(f"Created {len(docs)} document chunks.")

    # 3. Create & persist Chroma vector database
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name="research_paper_collection",
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Chroma DB saved to '{CHROMA_PERSIST_DIR}'.")

    return vectordb