from __future__ import annotations
from typing import List, Tuple
from .base import TextRetriever, TextItem
from ..bm25 import BM25Index

class BM25Retriever(TextRetriever):
    def __init__(self, items: List[TextItem]):
        docs = [{"docid": it.id, "url": it.meta.get("url") if it.meta else "", "title": it.meta.get("title") if it.meta else "", "content": it.text} for it in items]
        self._index = BM25Index(docs)
        self._n = len(items)

    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        return self._index.search(query, topk=topk)

    def size(self) -> int:
        return self._n
