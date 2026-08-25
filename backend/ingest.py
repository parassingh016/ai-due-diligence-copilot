"""
ingest.py
Handles: PDF text extraction -> chunking -> embedding -> storing in ChromaDB
"""

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model once (free, runs locally, ~80MB download on first run)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent local vector database (free, no cloud account needed)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("due_diligence_docs")


def extract_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks for better retrieval context."""
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_document(pdf_path: str, doc_name: str) -> int:
    """Extract, chunk, embed, and store a document. Returns number of chunks stored."""
    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text found in PDF (it may be a scanned image).")

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def list_documents() -> list[str]:
    """Return unique document names currently stored in the vector DB."""
    all_items = collection.get()
    sources = {m["source"] for m in all_items["metadatas"]} if all_items["metadatas"] else set()
    return sorted(sources)


def delete_document(doc_name: str):
    """Remove all chunks belonging to a document."""
    collection.delete(where={"source": doc_name})