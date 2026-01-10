from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import hashlib
import re

from .base import TextRetriever, TextItem

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.U)

def _stable_hash(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic hashing embedding (no external models needed).

    Not a semantic encoder; intended as a lightweight FAISS-ready baseline.
    """
    vec = np.zeros((dim,), dtype=np.float32)
    for tok in _WORD_RE.findall(text.lower()):
        hv = _stable_hash(tok)
        idx = hv % dim
        sign = 1.0 if (hv >> 63) == 0 else -1.0
        vec[idx] += sign
    # L2 normalize
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec

@dataclass
class FaissConfig:
    dim: int = 384
    use_cosine: bool = True

class FaissRetriever(TextRetriever):
    """FAISS vector retriever with a lightweight hashing embedder.

    Requirements:
      pip install faiss-cpu

    This is a structural integration point: you can swap `_hash_embed` with a real
    embedding model (e.g., sentence-transformers) later.
    """

    def __init__(self, items: List[TextItem], cfg: Optional[FaissConfig] = None):
        self.cfg = cfg or FaissConfig()
        self.ids: List[str] = [it.id for it in items]

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("FaissRetriever requires faiss-cpu. Install via `pip install faiss-cpu`.") from e

        self.faiss = faiss

        # Build vectors
        X = np.stack([_hash_embed(it.text, self.cfg.dim) for it in items], axis=0).astype("float32")
        # Cosine similarity via inner product on normalized vectors
        self.index = faiss.IndexFlatIP(self.cfg.dim)
        self.index.add(X)
        self._n = len(items)

    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        qv = _hash_embed(query, self.cfg.dim).reshape(1, -1).astype("float32")
        scores, idxs = self.index.search(qv, topk)
        out: List[Tuple[str, float]] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self.ids):
                continue
            out.append((self.ids[i], float(score)))
        return out

    def size(self) -> int:
        return self._n
