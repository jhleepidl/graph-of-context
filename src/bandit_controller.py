from __future__ import annotations

"""Bandit-based unfold controller for GoC.

This controller replaces the (costly) small-LLM controller in Option-B with a
lightweight contextual bandit. It consumes the structured candidate list
produced by GoCMemory.compute_unfold_candidates() and returns a list of seed_ids
to unfold.

Design goals:
  - No LLM calls
  - Deterministic, inspectable features
  - Works with dynamic candidates (global linear theta)
  - Can be trained from trace logs (offline) or online
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import random

import numpy as np

from .bandit import LinUCBBandit


def _safe_log1p(x: float) -> float:
    try:
        return float(math.log1p(max(0.0, float(x))))
    except Exception:
        return 0.0


@dataclass
class BanditUnfoldController:
    """Contextual-bandit controller for selecting unfold seed ids.

    Feature version v1 (dim=10):
      0  bias
      1  log1p(score)
      2  -log1p(cost_tokens)
      3  log1p(closure_size)
      4  score / max(1, cost_tokens)
      5  is_tool_seed (seed_kind == 'tool')
      6  is_summary_seed (seed_kind == 'summary')
      7  has_title_docref (any docid startswith 'TITLE:')
      8  hits_committed_title (seed_docids intersects committed TITLE anchors)
      9  recency_norm (1 / (1 + max(0, now_step - seed_step)))

    Notes:
      - We use a *global* theta (LinUCB). Candidates are transient so we can't
        maintain per-arm stats.
      - The bandit is still useful because these features generalize across
        tasks and steps.
    """

    alpha: float = 1.0
    epsilon: float = 0.05
    ridge: float = 1.0
    feature_version: str = "v1"
    seed: int = 7

    bandit: LinUCBBandit = field(init=False)

    def __post_init__(self):
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        dim = self.feature_dim(self.feature_version)
        self.bandit = LinUCBBandit(dim=dim, alpha=float(self.alpha), ridge=float(self.ridge))

    # ---- features ----
    @staticmethod
    def feature_dim(version: str) -> int:
        v = (version or "v1").strip().lower()
        if v == "v1":
            return 10
        raise ValueError(f"Unknown feature_version: {version}")

    def featurize(
        self,
        cand: Dict[str, Any],
        *,
        now_step: int,
        committed_titles: Optional[List[str]] = None,
    ) -> np.ndarray:
        v = (self.feature_version or "v1").strip().lower()
        if v != "v1":
            raise ValueError(f"Unsupported feature_version: {self.feature_version}")

        score = float(cand.get("score", 0.0) or 0.0)
        cost = float(cand.get("cost_tokens", 0.0) or 0.0)
        closure = float(cand.get("closure_size", cand.get("closure_size", 0.0)) or 0.0)
        kind = str(cand.get("seed_kind") or "").lower().strip()
        docids = [str(d) for d in (cand.get("seed_docids") or [])]

        has_title = any(d.startswith("TITLE:") for d in docids)
        committed = committed_titles or []
        committed_keys = {f"TITLE:{t}" for t in committed}
        hit_commit = any(d in committed_keys for d in docids)

        seed_step = int(cand.get("seed_step", -1) or -1)
        delta = max(0, int(now_step) - seed_step) if seed_step >= 0 else 9999
        recency = 1.0 / (1.0 + float(delta))

        x = np.array(
            [
                1.0,
                _safe_log1p(score),
                -_safe_log1p(cost),
                _safe_log1p(closure),
                float(score) / float(max(1.0, cost)),
                1.0 if kind == "tool" else 0.0,
                1.0 if kind == "summary" else 0.0,
                1.0 if has_title else 0.0,
                1.0 if hit_commit else 0.0,
                float(recency),
            ],
            dtype=np.float64,
        )
        return x

    # ---- selection ----
    def select_seed_ids(
        self,
        *,
        candidates: List[Dict[str, Any]],
        k: int,
        budget_unfold: int,
        now_step: int,
        committed_titles: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Pick <=k seed_ids.

        Returns: (picked_seed_ids, debug)
        """
        kk = max(1, int(k or 1))
        bud = max(0, int(budget_unfold or 0))
        remaining = bud

        pool = list(candidates or [])
        picked: List[str] = []
        picked_rows: List[Dict[str, Any]] = []

        # Precompute features + bandit scores for all candidates; we will
        # re-score only if we later want stateful features (v2+).
        scored: List[Tuple[float, Dict[str, Any], np.ndarray]] = []
        for c in pool:
            try:
                x = self.featurize(c, now_step=now_step, committed_titles=committed_titles)
                p = float(self.bandit.score(x))
                scored.append((p, c, x))
            except Exception:
                continue
        scored.sort(key=lambda t: t[0], reverse=True)

        # Epsilon-greedy over the top candidates.
        i = 0
        while i < len(scored) and len(picked) < kk:
            # Choose an index: either best available or random among top 5.
            if random.random() < float(self.epsilon):
                topn = min(5, len(scored))
                j = random.randrange(0, topn)
            else:
                j = 0

            p, c, x = scored.pop(j)
            sid = str(c.get("seed_id") or "").strip()
            if not sid:
                continue
            if sid in picked:
                continue

            cost = int(c.get("cost_tokens", 0) or 0)
            if remaining - cost < 0:
                # Can't afford this seed; try next.
                continue

            picked.append(sid)
            remaining -= int(max(0, cost))
            picked_rows.append(
                {
                    "seed_id": sid,
                    "bandit_score": float(p),
                    "features": x.tolist(),
                    "cost_tokens": int(cost),
                    "score": float(c.get("score", 0.0) or 0.0),
                }
            )
            i += 1

        dbg = {
            "feature_version": self.feature_version,
            "alpha": float(self.alpha),
            "epsilon": float(self.epsilon),
            "budget_unfold": int(bud),
            "budget_remaining": int(remaining),
            "picked": picked_rows,
        }
        return picked, dbg

    # ---- updates (optional; online learning) ----
    def update_from_features(self, features: List[float], reward: float):
        x = np.array(features, dtype=np.float64)
        self.bandit.update(x, float(reward))

    # ---- persistence ----
    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "kind": "bandit_unfold_controller",
            "feature_version": self.feature_version,
            "alpha": float(self.alpha),
            "epsilon": float(self.epsilon),
            "ridge": float(self.ridge),
            "seed": int(self.seed),
            "bandit": self.bandit.to_json_dict(),
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, path: str) -> "BanditUnfoldController":
        """Convenience loader."""
        return cls().load_json(path)

    def load_json(self, path: str):
        dct = json.loads(open(path, "r", encoding="utf-8").read())
        fv = str(dct.get("feature_version") or "v1")
        self.feature_version = fv
        self.alpha = float(dct.get("alpha", self.alpha))
        self.epsilon = float(dct.get("epsilon", self.epsilon))
        self.ridge = float(dct.get("ridge", self.ridge))
        self.seed = int(dct.get("seed", self.seed))
        self.bandit = LinUCBBandit.from_json_dict(dct.get("bandit") or {})
        return self
