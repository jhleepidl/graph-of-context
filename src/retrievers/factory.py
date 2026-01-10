from __future__ import annotations
from typing import List, Optional
from .base import TextItem, TextRetriever
from .bm25_retriever import BM25Retriever
from .faiss_retriever import FaissRetriever, FaissConfig

def build_retriever(kind: str, items: List[TextItem], faiss_dim: int = 384) -> TextRetriever:
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return BM25Retriever(items)
    if kind == "faiss":
        return FaissRetriever(items, cfg=FaissConfig(dim=faiss_dim))
    raise ValueError(f"Unknown retriever kind: {kind}. Use bm25 or faiss.")
