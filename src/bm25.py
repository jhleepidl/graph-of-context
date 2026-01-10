from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, DefaultDict
import math
import heapq
from collections import defaultdict

from .utils import tokenize

@dataclass
class BM25Index:
    """A small BM25 implementation.

    v6 improvement:
    - Build an inverted index (postings) so queries do NOT score every document (avoid O(N) per query).
    - If a query matches no terms, fall back to returning the first topk docs with score 0.0.

    Note: For very large corpora, consider swapping this with a vector DB or a dedicated IR engine.
    """
    docs: List[Dict[str, Any]]
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self):
        self.N = len(self.docs)
        self.doc_tokens: List[List[str]] = [tokenize((d.get("content", "") or "") + " " + (d.get("title", "") or "")) for d in self.docs]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)

        # df + postings
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_i, tf)]
        for i, toks in enumerate(self.doc_tokens):
            tf: Dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, f in tf.items():
                self.df[w] = self.df.get(w, 0) + 1
                self.postings.setdefault(w, []).append((i, f))

        # idf
        self.idf: Dict[str, float] = {}
        for w, df in self.df.items():
            self.idf[w] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        q = tokenize(query)
        if self.N == 0:
            return []

        scores: DefaultDict[int, float] = defaultdict(float)

        for w in q:
            plist = self.postings.get(w)
            if not plist:
                continue
            idf = self.idf.get(w, 0.0)
            for doc_i, f in plist:
                dl = self.doc_len[doc_i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                scores[doc_i] += idf * (f * (self.k1 + 1)) / denom

        if not scores:
            # No overlap; return first docs with 0 score
            return [(self.docs[i]["docid"], 0.0) for i in range(min(topk, self.N))]

        # Take topk efficiently
        best = heapq.nlargest(topk, scores.items(), key=lambda x: x[1])
        return [(self.docs[i]["docid"], float(s)) for i, s in best]
