"""RAG adapter: fast offline lexical retrieval, optionally semantic FAISS.

Offline retrieval is always available and doesn't download embedding weights.
Opt in to semantic retrieval by setting RAG_BACKEND=semantic and installing
requirements-semantic.txt. Semantic indexes are rebuilt in this process and
never loaded via unsafe pickle deserialization.
"""
from __future__ import annotations
import os
from functools import lru_cache
from dataclasses import dataclass
from src.knowledge import DOCS_DIR, search

@dataclass
class Document:
    page_content: str
    metadata: dict

class OfflineRetriever:
    def __init__(self, k: int = 3):
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        return [Document(page_content=f"{p.title}\n{p.text}", metadata={"source": p.source, "score": p.score})
                for p in search(query, limit=self.k)]

@lru_cache(maxsize=2)
def get_retriever(k: int = 3):
    if os.getenv("RAG_BACKEND", "lexical").lower() == "semantic":
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_core.documents import Document as LCDocument
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            texts = [LCDocument(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
                     for p in sorted(DOCS_DIR.glob("*.md"))]
            chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60).split_documents(texts)
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # In-memory only: no unsafe FAISS.load_local pickle dependency.
            return FAISS.from_documents(chunks, embedding).as_retriever(search_kwargs={"k": k})
        except (ImportError, RuntimeError, OSError) as exc:
            import logging
            logging.getLogger(__name__).warning("Semantic retrieval unavailable: %s", type(exc).__name__)
    return OfflineRetriever(k)
