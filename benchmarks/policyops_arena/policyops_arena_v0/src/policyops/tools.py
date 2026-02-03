from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None

from collections import Counter

from .schemas import Clause, World

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_snippet(text: str, length: int = 160) -> str:
    snippet = text.strip().replace("\n", " ")
    if len(snippet) <= length:
        return snippet
    return snippet[:length].rstrip() + "..."


class _FallbackBM25:
    def __init__(self, corpus: List[List[str]]) -> None:
        self.corpus = corpus
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: List[Counter[str]] = []
        for doc in corpus:
            counts = Counter(doc)
            self.term_freqs.append(counts)
            self.doc_freq.update(counts.keys())
        self.num_docs = len(corpus)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores: List[float] = []
        query_counts = Counter(query_tokens)
        for doc_counts in self.term_freqs:
            score = 0.0
            for term, qcount in query_counts.items():
                tf = doc_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = 1.0 + (self.num_docs / (1 + self.doc_freq.get(term, 0)))
                score += qcount * tf * idf
            scores.append(score)
        return scores


class ClauseIndex:
    def __init__(self, world: World) -> None:
        self.world = world
        self.clauses: List[Clause] = list(world.clauses.values())
        self.clause_ids: List[str] = [clause.clause_id for clause in self.clauses]
        corpus = [tokenize(clause.text) for clause in self.clauses]
        if BM25Okapi is None:
            self.bm25 = _FallbackBM25(corpus)
        else:
            self.bm25 = BM25Okapi(corpus)
        self.doc_by_id = {doc.doc_id: doc for doc in world.documents}

    def _passes_filters(self, clause: Clause, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        doc = self.doc_by_id.get(clause.doc_id)
        for key in ["slot", "kind", "authority"]:
            if key in filters and getattr(clause, key) != filters[key]:
                return False
        if "doc_type" in filters and doc and doc.doc_type != filters["doc_type"]:
            return False
        for key in ["region", "product", "tier", "data_type", "purpose"]:
            if key in filters:
                allowed = clause.applies_if.get(key)
                if allowed and filters[key] not in allowed:
                    return False
        return True

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        scored: List[Dict[str, Any]] = []
        for idx, clause in enumerate(self.clauses):
            if not self._passes_filters(clause, filters):
                continue
            scored.append(
                {
                    "clause_id": clause.clause_id,
                    "score": float(scores[idx]),
                    "snippet": make_snippet(clause.text),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
