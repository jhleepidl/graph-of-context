from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ControllerDecision:
    action: str = "none"
    reason: str = "default_none"
    unfold_query: str = ""
    fork_query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextController:
    """Heuristic operator controller for GoC.

    The controller chooses among a small action set:
      - none
      - unfold
      - fork
      - unfold_then_fork

    Design goals for the first implementation:
      - deterministic and inspectable
      - cheap enough to run every step
      - conservative defaults so existing GoC behavior is preserved unless
        explicitly enabled
    """

    VALID_POLICIES = {"stage_aware", "budget_aware", "uncertainty_aware"}

    def __init__(
        self,
        *,
        policy: str = "uncertainty_aware",
        support_gap_threshold: float = 0.20,
        budget_pressure_threshold: float = 0.80,
        fork_ambiguity_threshold: float = 0.45,
        active_tokens_fork_threshold: int = 450,
    ) -> None:
        p = str(policy or "uncertainty_aware").strip().lower()
        self.policy = p if p in self.VALID_POLICIES else "uncertainty_aware"
        self.support_gap_threshold = max(0.0, float(support_gap_threshold))
        self.budget_pressure_threshold = max(0.0, float(budget_pressure_threshold))
        self.fork_ambiguity_threshold = max(0.0, float(fork_ambiguity_threshold))
        self.active_tokens_fork_threshold = max(0, int(active_tokens_fork_threshold))

    def decide(
        self,
        *,
        current_user_prompt: str,
        features: Dict[str, Any],
        q1_text: Optional[str] = None,
        commit_titles: Optional[List[str]] = None,
    ) -> ControllerDecision:
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

        if support_gap >= self.support_gap_threshold:
            if fork_ready and specialist and (ambiguity >= self.fork_ambiguity_threshold or has_conflict or pivot_risk >= 0.75):
                return self._finalize(
                    action="unfold_then_fork",
                    reason="uncertainty_gap_unfold_then_fork",
                    current_user_prompt=current_user_prompt,
                    features=features,
                    q1_text=q1_text,
                    commit_titles=commit_titles,
                )
            return self._finalize(
                action="unfold",
                reason="uncertainty_gap_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        if fork_ready and specialist and (ambiguity >= self.fork_ambiguity_threshold or has_conflict) and pivot_risk >= 0.40:
            return self._finalize(
                action="fork",
                reason="uncertainty_specialist_fork",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        if pivot_risk >= 0.65:
            return self._finalize(
                action="unfold",
                reason="uncertainty_pivot_unfold",
                current_user_prompt=current_user_prompt,
                features=features,
                q1_text=q1_text,
                commit_titles=commit_titles,
            )

        return self._finalize(
            action="none",
            reason="uncertainty_noop",
            current_user_prompt=current_user_prompt,
            features=features,
            q1_text=q1_text,
            commit_titles=commit_titles,
        )
