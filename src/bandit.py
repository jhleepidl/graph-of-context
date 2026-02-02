from __future__ import annotations

"""Lightweight contextual bandit utilities.

This repo's GoC memory already produces structured unfold candidates with strong
signals (retrieval score, token cost, closure size, doc_ref hits). For
fold/unfold *selection*, we often do not need an LLM; a linear contextual bandit
works well and is cheap.

We implement LinUCB with a Sherman–Morrison update for A^{-1}.
The bandit is *global* (one theta for all dynamic candidates).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np


@dataclass
class LinUCBBandit:
    """Global linear contextual bandit (LinUCB).

    Maintains:
      A = lambda*I + sum x x^T
      b = sum r x
      theta = A^{-1} b

    Selection score:
      p(x) = theta^T x + alpha * sqrt(x^T A^{-1} x)
    """

    dim: int
    alpha: float = 1.0
    ridge: float = 1.0

    def __post_init__(self):
        d = int(self.dim)
        self.dim = d
        self.alpha = float(self.alpha)
        self.ridge = float(self.ridge)
        self.A = self.ridge * np.eye(d, dtype=np.float64)
        self.A_inv = (1.0 / self.ridge) * np.eye(d, dtype=np.float64)
        self.b = np.zeros((d,), dtype=np.float64)

    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def score(self, x: np.ndarray) -> float:
        x = x.reshape(-1)
        mu = float(self.theta().dot(x))
        u = float(np.sqrt(max(0.0, x @ self.A_inv @ x)))
        return mu + self.alpha * u

    def update(self, x: np.ndarray, reward: float):
        """Online update with Sherman–Morrison.

        A <- A + x x^T
        A_inv <- A_inv - (A_inv x x^T A_inv)/(1 + x^T A_inv x)
        b <- b + r x
        """
        x = x.reshape(-1, 1).astype(np.float64)
        if x.shape[0] != self.dim:
            raise ValueError(f"x dim mismatch: got {x.shape[0]} expected {self.dim}")

        Ax = self.A_inv @ x
        denom = float(1.0 + (x.T @ Ax))
        if denom <= 1e-12:
            # Extremely unlikely with ridge>0; fall back to stable rebuild.
            self.A = self.A + (x @ x.T)
            self.b = self.b + float(reward) * x.reshape(-1)
            self.A_inv = np.linalg.inv(self.A)
            return

        self.A_inv = self.A_inv - (Ax @ Ax.T) / denom
        self.A = self.A + (x @ x.T)
        self.b = self.b + float(reward) * x.reshape(-1)

    # ---- persistence ----
    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "kind": "linucb",
            "dim": int(self.dim),
            "alpha": float(self.alpha),
            "ridge": float(self.ridge),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_json_dict(cls, dct: Dict[str, Any]) -> "LinUCBBandit":
        dim = int(dct.get("dim") or 0)
        if dim <= 0:
            raise ValueError("Invalid bandit dim")
        obj = cls(dim=dim, alpha=float(dct.get("alpha", 1.0)), ridge=float(dct.get("ridge", 1.0)))
        # Restore A_inv and b; rebuild A for consistency.
        obj.A_inv = np.array(dct.get("A_inv"), dtype=np.float64)
        obj.b = np.array(dct.get("b"), dtype=np.float64)
        if obj.A_inv.shape != (dim, dim):
            raise ValueError("Invalid A_inv shape")
        if obj.b.shape != (dim,):
            raise ValueError("Invalid b shape")
        obj.A = np.linalg.inv(obj.A_inv)
        return obj

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LinUCBBandit":
        dct = json.loads(open(path, "r", encoding="utf-8").read())
        return cls.from_json_dict(dct)
