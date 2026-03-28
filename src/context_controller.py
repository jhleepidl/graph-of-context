from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import math
import pickle


@dataclass
class ControllerDecision:
    action: str = "none"
    reason: str = "default_none"
    unfold_query: str = ""
    fork_query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextController:
    """Operator controller for GoC.

    Supported actions:
      - none
      - unfold
      - fork
      - unfold_then_fork

    The controller supports both simple heuristic policies and a learned
    phase18-style classifier loaded from a pickle payload produced by
    scripts/train_phase18_controller.py.
    """

    HEURISTIC_POLICIES = {"stage_aware", "budget_aware", "uncertainty_aware"}
    LEARNED_POLICIES = {"learned", "learned_tree", "learned_logreg", "phase18_tree", "phase18_logreg"}
    VALID_POLICIES = HEURISTIC_POLICIES | LEARNED_POLICIES
    VALID_ACTIONS = {"none", "unfold", "fork", "unfold_then_fork"}

    def __init__(
        self,
        *,
        policy: str = "uncertainty_aware",
        support_gap_threshold: float = 0.20,
        budget_pressure_threshold: float = 0.80,
        fork_ambiguity_threshold: float = 0.45,
        active_tokens_fork_threshold: int = 450,
        learned_model_path: Optional[str] = None,
        learned_min_confidence: float = 0.0,
        learned_fallback_action: str = "unfold",
        learned_disable_none_action: bool = False,
    ) -> None:
        p = self._normalize_policy(policy)
        self.policy = p
        self.support_gap_threshold = max(0.0, float(support_gap_threshold))
        self.budget_pressure_threshold = max(0.0, float(budget_pressure_threshold))
        self.fork_ambiguity_threshold = max(0.0, float(fork_ambiguity_threshold))
        self.active_tokens_fork_threshold = max(0, int(active_tokens_fork_threshold))
        self.learned_model_path = str(learned_model_path or "").strip() or None
        self.learned_min_confidence = max(0.0, min(1.0, float(learned_min_confidence)))
        fallback = str(learned_fallback_action or "unfold").strip().lower()
        self.learned_fallback_action = fallback if fallback in self.VALID_ACTIONS else "unfold"
        self.learned_disable_none_action = bool(learned_disable_none_action)

        self.learned_model: Any = None
        self.learned_feature_names: List[str] = []
        self.learned_actions: List[str] = ["none", "unfold", "fork", "unfold_then_fork"]
        self.learned_model_type: str = ""
        if self.policy in self.LEARNED_POLICIES:
            self._load_learned_model(self.learned_model_path)

    @classmethod
    def _normalize_policy(cls, policy: Optional[str]) -> str:
        p = str(policy or "uncertainty_aware").strip().lower()
        aliases = {
            "tree": "learned_tree",
            "logreg": "learned_logreg",
            "phase18": "learned",
            "phase18_learned": "learned",
        }
        p = aliases.get(p, p)
        return p if p in cls.VALID_POLICIES else "uncertainty_aware"

    def _load_learned_model(self, model_path: Optional[str]) -> None:
        if not model_path:
            raise ValueError("learned context controller requires learned_model_path")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"learned context controller model not found: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("learned context controller payload must be a dict")
        model = payload.get("model")
        feature_names = payload.get("feature_names") or []
        actions = payload.get("actions") or []
        if model is None:
            raise ValueError("learned context controller payload is missing 'model'")
        if not isinstance(feature_names, Sequence) or not feature_names:
            raise ValueError("learned context controller payload is missing feature_names")
        actions = [str(x) for x in actions if str(x)]
        if not actions:
            actions = ["none", "unfold", "fork", "unfold_then_fork"]
        self.learned_model = model
        self.learned_feature_names = [str(x) for x in feature_names]
        self.learned_actions = actions
        self.learned_model_type = str(payload.get("model_type") or type(model).__name__)

    def decide(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str] = None,
        commit_titles: Optional[List[str]] = None,
    ) -> ControllerDecision:
        if self.policy in self.LEARNED_POLICIES:
            return self._decide_learned(
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )
        if self.policy == "stage_aware":
            return self._decide_stage_aware(
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )
        if self.policy == "budget_aware":
            return self._decide_budget_aware(
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )
        return self._decide_uncertainty_aware(
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )

    def _base_queries(
        self,
        *,
        current_user_prompt: str,
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
    ) -> Dict[str, str]:
        parts: List[str] = []
        q1 = str(q1_text or "").strip()
        if q1:
            parts.append(q1)
        titles = [str(x).strip() for x in (commit_titles or []) if str(x).strip()]
        if titles:
            parts.append(" ".join(titles[:8]))
        cur = str(current_user_prompt or "").strip()
        if cur:
            parts.append(cur)
        base = " | ".join([p for p in parts if p])
        unfold = base or cur or q1
        fork = " | ".join(
            [
                p
                for p in [
                    unfold,
                    "Return only support-complete evidence for the current pivot decision.",
                ]
                if str(p).strip()
            ]
        )
        return {"unfold_query": unfold, "fork_query": fork}

    def _finalize(
        self,
        *,
        action: str,
        reason: str,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> ControllerDecision:
        queries = self._base_queries(
            current_user_prompt=current_user_prompt,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )
        meta = {
            "policy": self.policy,
            "support_gap_score": float(features.get("support_gap_score", 0.0) or 0.0),
            "budget_utilization": float(features.get("budget_utilization", 0.0) or 0.0),
            "ambiguity_score": float(features.get("ambiguity_score", 0.0) or 0.0),
            "pivot_risk": float(features.get("pivot_risk", 0.0) or 0.0),
            "fork_ready": bool(features.get("fork_ready", False)),
            "fork_gate_reason": str(features.get("fork_gate_reason", "") or ""),
        }
        if extra:
            meta.update(extra)
        return ControllerDecision(
            action=str(action),
            reason=str(reason),
            unfold_query=str(queries["unfold_query"]),
            fork_query=str(queries["fork_query"]),
            metadata=meta,
        )

    @staticmethod
    def _as_float(val: Any, default: float = 0.0) -> float:
        try:
            out = float(val)
        except Exception:
            return float(default)
        return out if math.isfinite(out) else float(default)

    def _final_action_with_guards(
        self,
        *,
        predicted_action: str,
        predicted_confidence: float,
        features: Dict[str, Any],
    ) -> tuple[str, str, Dict[str, Any]]:
        action = str(predicted_action or "none").strip().lower()
        confidence = self._as_float(predicted_confidence, 0.0)
        meta: Dict[str, Any] = {
            "learned_predicted_action": action,
            "learned_confidence": float(confidence),
            "learned_fallback_action": self.learned_fallback_action,
            "learned_disable_none_action": bool(self.learned_disable_none_action),
        }
        reason = "learned_policy"

        if action not in self.VALID_ACTIONS:
            action = self.learned_fallback_action
            reason = "learned_invalid_action_fallback"

        if confidence < self.learned_min_confidence:
            action = self.learned_fallback_action
            reason = "learned_low_confidence_fallback"
            meta["learned_min_confidence"] = float(self.learned_min_confidence)

        if action == "none" and self.learned_disable_none_action:
            action = self.learned_fallback_action
            reason = "learned_none_disabled_fallback"

        fork_ready = bool(features.get("fork_ready", False))
        branch_score = self._as_float(features.get("branch_score", 0.0), 0.0)
        candidate_count = int(features.get("candidate_count", 0) or 0)
        has_conflict = bool(features.get("has_conflict", False))
        if action == "fork" and not fork_ready:
            action = self.learned_fallback_action if self.learned_fallback_action != "fork" else "unfold"
            reason = "learned_fork_gate_blocked"
        elif action == "unfold_then_fork" and not fork_ready:
            action = "unfold"
            reason = "learned_utf_fork_gate_blocked"
        elif action in {"fork", "unfold_then_fork"} and branch_score < 0.12 and candidate_count < 2 and not has_conflict:
            action = self.learned_fallback_action if action == "fork" else "unfold"
            reason = "learned_low_branch_fallback"
            meta["learned_branch_score"] = float(branch_score)

        if action not in self.VALID_ACTIONS:
            action = "unfold"
            reason = "learned_invalid_post_guard"
        return action, reason, meta

    def _decide_learned(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
    ) -> ControllerDecision:
        if self.learned_model is None:
            raise RuntimeError("learned context controller has no loaded model")
        row: List[float] = []
        for name in self.learned_feature_names:
            val = features.get(name, 0.0)
            if isinstance(val, bool):
                row.append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                row.append(self._as_float(val, 0.0))
            else:
                row.append(0.0)

        pred_raw = self.learned_model.predict([row])[0]
        try:
            pred_idx = int(pred_raw)
            predicted_action = self.learned_actions[pred_idx] if 0 <= pred_idx < len(self.learned_actions) else str(pred_raw)
        except Exception:
            predicted_action = str(pred_raw)

        pred_probs: Dict[str, float] = {}
        confidence = 1.0
        if hasattr(self.learned_model, "predict_proba"):
            try:
                proba = self.learned_model.predict_proba([row])[0]
                classes = list(getattr(self.learned_model, "classes_", []))
                if classes and len(classes) == len(proba):
                    for cls, p in zip(classes, proba):
                        try:
                            action = self.learned_actions[int(cls)]
                        except Exception:
                            action = str(cls)
                        pred_probs[str(action)] = float(p)
                    confidence = max(pred_probs.values()) if pred_probs else 1.0
                else:
                    confidence = max(float(x) for x in proba) if len(proba) else 1.0
            except Exception:
                confidence = 1.0

        final_action, reason, meta = self._final_action_with_guards(
            predicted_action=predicted_action,
            predicted_confidence=confidence,
            features=features,
        )
        if pred_probs:
            meta["learned_action_probs"] = {k: float(v) for k, v in pred_probs.items()}
        meta["learned_model_type"] = str(self.learned_model_type or type(self.learned_model).__name__)
        meta["learned_model_path"] = str(self.learned_model_path or "")
        meta["learned_feature_count"] = int(len(self.learned_feature_names))
        return self._finalize(
            action=final_action,
            reason=reason,
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
            extra=meta,
        )

    def _decide_stage_aware(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
    ) -> ControllerDecision:
        is_commit = bool(features.get("is_commit_like", False))
        is_final = bool(features.get("is_final_like", False))
        is_pivot = bool(features.get("is_pivot_like", False))
        fork_ready = bool(features.get("fork_ready", False))
        support_gap = float(features.get("support_gap_score", 0.0) or 0.0)
        ambiguity = float(features.get("ambiguity_score", 0.0) or 0.0)
        active_tokens = int(features.get("active_tokens_est", 0) or 0)

        if is_commit or is_final:
            if fork_ready and ambiguity >= self.fork_ambiguity_threshold and active_tokens >= self.active_tokens_fork_threshold:
                return self._finalize(
                    action="unfold_then_fork",
                    reason="stage_commit_or_final_unfold_then_fork",
                    current_user_prompt=current_user_prompt,
                    features=features,
                    q1_text=q1_text,
                    commit_titles=commit_titles,
                )
            return self._finalize(
                action="unfold",
                reason="stage_commit_or_final_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        if is_pivot and support_gap >= self.support_gap_threshold:
            if fork_ready and ambiguity >= self.fork_ambiguity_threshold:
                return self._finalize(
                    action="unfold_then_fork",
                    reason="stage_pivot_gap_unfold_then_fork",
                    current_user_prompt=current_user_prompt,
                    features=features,
                    q1_text=q1_text,
                    commit_titles=commit_titles,
                )
            return self._finalize(
                action="unfold",
                reason="stage_pivot_gap_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        if is_pivot and fork_ready and ambiguity >= self.fork_ambiguity_threshold:
            return self._finalize(
                action="fork",
                reason="stage_pivot_specialist_fork",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        return self._finalize(
            action="none",
            reason="stage_noop",
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )

    def _decide_budget_aware(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
    ) -> ControllerDecision:
        support_gap = float(features.get("support_gap_score", 0.0) or 0.0)
        budget_pressure = float(features.get("budget_utilization", 0.0) or 0.0)
        ambiguity = float(features.get("ambiguity_score", 0.0) or 0.0)
        fork_ready = bool(features.get("fork_ready", False))
        specialist = bool(features.get("specialist_subtask_flag", False))

        if support_gap >= self.support_gap_threshold:
            if fork_ready and specialist and budget_pressure >= self.budget_pressure_threshold and ambiguity >= self.fork_ambiguity_threshold:
                return self._finalize(
                    action="unfold_then_fork",
                    reason="budget_gap_pressure_unfold_then_fork",
                    current_user_prompt=current_user_prompt,
                    features=features,
                    q1_text=q1_text,
                    commit_titles=commit_titles,
                )
            return self._finalize(
                action="unfold",
                reason="budget_gap_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        if fork_ready and specialist and budget_pressure >= self.budget_pressure_threshold and ambiguity >= self.fork_ambiguity_threshold:
            return self._finalize(
                action="fork",
                reason="budget_pressure_fork",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        return self._finalize(
            action="none",
            reason="budget_noop",
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )

    def _decide_uncertainty_aware(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str],
        commit_titles: Optional[List[str]],
    ) -> ControllerDecision:
        support_gap = float(features.get("support_gap_score", 0.0) or 0.0)
        ambiguity = float(features.get("ambiguity_score", 0.0) or 0.0)
        pivot_risk = float(features.get("pivot_risk", 0.0) or 0.0)
        fork_ready = bool(features.get("fork_ready", False))
        specialist = bool(features.get("specialist_subtask_flag", False))
        has_conflict = bool(features.get("has_conflict", False))
        branch_score = float(features.get("branch_score", 0.0) or 0.0)
        pressure = float(features.get("evidence_pressure_score", 0.0) or 0.0)
        candidate_count = int(features.get("candidate_count", 0) or 0)
        strong_branch = bool(branch_score >= 0.18 or candidate_count >= 2 or has_conflict)
        strong_pressure = bool(pressure >= 0.45 or ambiguity >= self.fork_ambiguity_threshold or pivot_risk >= 0.75 or has_conflict)

        if support_gap >= self.support_gap_threshold:
            if fork_ready and specialist and strong_branch and strong_pressure:
                return self._finalize(
                    action="unfold_then_fork",
                    reason="uncertainty_gap_unfold_then_fork",
                    current_user_prompt=current_user_prompt,
                    features=features,
                    q1_text=q1_text,
                    commit_titles=commit_titles,
                    extra={"branch_score": float(branch_score), "evidence_pressure_score": float(pressure)},
                )
            return self._finalize(
                action="unfold",
                reason="uncertainty_gap_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
                extra={"branch_score": float(branch_score), "evidence_pressure_score": float(pressure)},
            )

        if fork_ready and specialist and strong_branch and strong_pressure and pivot_risk >= 0.40:
            return self._finalize(
                action="fork",
                reason="uncertainty_specialist_fork",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
                extra={"branch_score": float(branch_score), "evidence_pressure_score": float(pressure)},
            )

        if pivot_risk >= 0.65 or pressure >= 0.52:
            return self._finalize(
                action="unfold",
                reason="uncertainty_pivot_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
                extra={"branch_score": float(branch_score), "evidence_pressure_score": float(pressure)},
            )

        return self._finalize(
            action="none",
            reason="uncertainty_noop",
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )
