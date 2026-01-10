from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

from .retrievers.base import TextItem, TextRetriever
from .retrievers.factory import build_retriever

@dataclass
class CorpusEnv:
    corpus: List[Dict[str, Any]]
    retriever: TextRetriever
    docid_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    url_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.docid_map:
            self.docid_map = {d["docid"]: d for d in self.corpus}
        if not self.url_map:
            self.url_map = {d["url"]: d for d in self.corpus if "url" in d}

    @classmethod
    def from_json(cls, path: str, retriever_kind: str = "bm25", faiss_dim: int = 384) -> "CorpusEnv":
        corpus = json.load(open(path, "r", encoding="utf-8"))
        items: List[TextItem] = []
        for d in corpus:
            items.append(TextItem(
                id=d["docid"],
                text=d.get("content", "") or "",
                meta={"url": d.get("url",""), "title": d.get("title","")}
            ))
        retriever = build_retriever(retriever_kind, items, faiss_dim=faiss_dim)
        return cls(corpus=corpus, retriever=retriever)

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        hits = self.retriever.search(query, topk=topk)
        out = []
        for docid, score in hits:
            d = self.docid_map.get(docid)
            if not d:
                continue
            snippet = (d.get("content","") or "")[:400].replace("\n", " ")
            out.append({
                "docid": docid,
                "url": d.get("url", ""),
                "title": d.get("title", ""),
                "snippet": snippet,
                "score": float(score),
            })
        return out

    def open_page(self, docid: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        d = None
        if docid:
            d = self.docid_map.get(docid)
        if d is None and url:
            d = self.url_map.get(url)
        if not d:
            return {"docid": docid or "", "url": url or "", "title": "", "content": ""}
        return {"docid": d.get("docid",""), "url": d.get("url",""), "title": d.get("title",""), "content": d.get("content","")}
