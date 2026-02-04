from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

UPDATE_KW = ["effective immediately", "supersede", "revoke", "prior guidance", "amend"]
COND_KW = ["allowed only after", "conditions:", "provided that", "must", "require"]
DEF_KW = ["definition", "means", "for the purposes of"]
EXC_KW = ["except", "unless", "however"]

KEYWORD_WEIGHTS = {
    "update": 3.0,
    "condition": 2.5,
    "definition": 2.0,
    "exception": 1.5,
}


def _keyword_hit(snippet: str, keywords: List[str]) -> bool:
    return any(word in snippet for word in keywords)


def _merge_results(seed_results: List[Dict[str, Any]], expanded_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in seed_results + expanded_results:
        clause_id = item.get("clause_id")
        if not clause_id:
            continue
        current = merged.get(clause_id)
        score = float(item.get("score", 0.0))
        snippet = item.get("snippet", "")
        if current is None:
            merged[clause_id] = {"clause_id": clause_id, "score": score, "snippet": snippet}
        else:
            if score > current.get("score", 0.0):
                current["score"] = score
            if not current.get("snippet") and snippet:
                current["snippet"] = snippet
    return merged


@dataclass
class Controller:
    actions: List[str] = field(
        default_factory=lambda: [
            "update_first",
            "definition_heavy",
            "exception_first",
            "recency_biased",
            "balanced",
            "alpha_0_2",
            "alpha_0_5",
            "alpha_0_8",
        ]
    )
    epsilon: float = 0.1
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {
                "global": {action: {"count": 0, "total_reward": 0.0} for action in self.actions},
                "contexts": {},
            }

    def _context_key(self, context_features: Dict[str, Any]) -> str:
        parts = [
            context_features.get("slot"),
            context_features.get("region"),
            context_features.get("tier"),
            context_features.get("purpose"),
            context_features.get("data_type"),
            str(context_features.get("open_budget")),
            str(context_features.get("top1_score")),
            str(context_features.get("top5_mean")),
            str(context_features.get("score_gap")),
            str(context_features.get("top5_update_hits")),
            str(context_features.get("top5_condition_hits")),
            str(context_features.get("top5_definition_hits")),
            str(context_features.get("top5_exception_hits")),
        ]
        return "|".join([str(p) for p in parts])

    def build_context_features(
        self,
        task_context: Dict[str, Any],
        open_budget: int,
        search_stats: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        base = {
            "slot": task_context.get("slot"),
            "region": task_context.get("region"),
            "tier": task_context.get("tier"),
            "purpose": task_context.get("purpose"),
            "data_type": task_context.get("data_type"),
            "open_budget": open_budget,
        }
        if search_stats:
            base.update(search_stats)
        return base

    def compute_search_stats(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [float(item.get("score", 0.0)) for item in seed_results]
        top1 = scores[0] if scores else 0.0
        top5 = scores[:5]
        top5_mean = sum(top5) / len(top5) if top5 else 0.0
        score_gap = top1 - top5_mean
        top5_items = seed_results[:5]
        return {
            "top1_score": round(top1, 4),
            "top5_mean": round(top5_mean, 4),
            "score_gap": round(score_gap, 4),
            "top5_update_hits": sum(1 for item in top5_items if _keyword_hit(str(item.get("snippet", "")).lower(), UPDATE_KW)),
            "top5_condition_hits": sum(1 for item in top5_items if _keyword_hit(str(item.get("snippet", "")).lower(), COND_KW)),
            "top5_definition_hits": sum(1 for item in top5_items if _keyword_hit(str(item.get("snippet", "")).lower(), DEF_KW)),
            "top5_exception_hits": sum(1 for item in top5_items if _keyword_hit(str(item.get("snippet", "")).lower(), EXC_KW)),
        }

    def select_action(self, context_features: Dict[str, Any], explore: bool = True) -> str:
        context_key = self._context_key(context_features)
        stats = self.stats.get("contexts", {}).get(context_key) or self.stats.get("global", {})
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        best_action = self.actions[0]
        best_score = float("-inf")
        for action in self.actions:
            entry = stats.get(action, {"count": 0, "total_reward": 0.0})
            count = entry.get("count", 0)
            avg = entry.get("total_reward", 0.0) / count if count else 0.0
            if avg > best_score:
                best_score = avg
                best_action = action
        return best_action

    def update(self, action: str, context_features: Dict[str, Any], reward: float) -> None:
        context_key = self._context_key(context_features)
        contexts = self.stats.setdefault("contexts", {})
        if context_key not in contexts:
            contexts[context_key] = {a: {"count": 0, "total_reward": 0.0} for a in self.actions}
        for scope in [self.stats["global"], contexts[context_key]]:
            entry = scope.setdefault(action, {"count": 0, "total_reward": 0.0})
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["total_reward"] = float(entry.get("total_reward", 0.0)) + float(reward)

    def rank_clause_ids(
        self,
        action: str,
        seed_results: List[Dict[str, Any]],
        expanded_results: List[Dict[str, Any]],
    ) -> List[str]:
        merged = _merge_results(seed_results, expanded_results)
        ranked: List[Tuple[float, str]] = []
        for clause_id, item in merged.items():
            snippet = str(item.get("snippet", "")).lower()
            base_score = float(item.get("score", 0.0))
            update_hit = _keyword_hit(snippet, UPDATE_KW)
            cond_hit = _keyword_hit(snippet, COND_KW)
            def_hit = _keyword_hit(snippet, DEF_KW)
            exc_hit = _keyword_hit(snippet, EXC_KW)
            keyword_priority = (
                (KEYWORD_WEIGHTS["update"] if update_hit else 0.0)
                + (KEYWORD_WEIGHTS["condition"] if cond_hit else 0.0)
                + (KEYWORD_WEIGHTS["definition"] if def_hit else 0.0)
                + (KEYWORD_WEIGHTS["exception"] if exc_hit else 0.0)
            )

            if action == "update_first":
                bonus = (3.0 if update_hit else 0.0) + (1.0 if def_hit else 0.0) + (1.0 if exc_hit else 0.0)
            elif action == "definition_heavy":
                bonus = (3.0 if def_hit else 0.0) + (1.0 if update_hit else 0.0) + (1.0 if exc_hit else 0.0)
            elif action == "exception_first":
                bonus = (3.0 if exc_hit else 0.0) + (1.0 if update_hit else 0.0) + (1.0 if def_hit else 0.0)
            elif action == "recency_biased":
                bonus = 0.0
            else:  # balanced
                bonus = (2.0 if update_hit else 0.0) + (1.5 if def_hit else 0.0) + (1.0 if exc_hit else 0.0)

            if action.startswith("alpha_"):
                try:
                    alpha = float(action.split("_", 1)[1].replace("_", "."))
                except ValueError:
                    alpha = 0.5
                score = alpha * base_score + (1.0 - alpha) * keyword_priority
            else:
                score = base_score + bonus
            ranked.append((score, clause_id))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [clause_id for _, clause_id in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"actions": self.actions, "epsilon": self.epsilon, "stats": self.stats}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, epsilon: float = 0.1) -> "Controller":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            actions = data.get("actions") or [
                "update_first",
                "definition_heavy",
                "exception_first",
                "recency_biased",
                "balanced",
                "alpha_0_2",
                "alpha_0_5",
                "alpha_0_8",
            ]
            stats = data.get("stats", {})
            ctrl = cls(actions=actions, epsilon=float(data.get("epsilon", epsilon)), stats=stats)
            return ctrl
        return cls(epsilon=epsilon)
