from __future__ import annotations

from src.context_controller import ContextController


BASE = {
    "support_gap_score": 0.0,
    "budget_utilization": 0.5,
    "ambiguity_score": 0.0,
    "pivot_risk": 0.2,
    "fork_ready": False,
    "fork_gate_reason": "active_lt_min",
    "active_tokens_est": 300,
    "is_commit_like": False,
    "is_final_like": False,
    "is_pivot_like": False,
    "specialist_subtask_flag": False,
    "has_conflict": False,
}


def test_stage_aware_commit_prefers_unfold_then_fork_when_ready() -> None:
    ctl = ContextController(policy="stage_aware")
    feats = dict(BASE)
    feats.update(
        {
            "is_commit_like": True,
            "is_pivot_like": True,
            "fork_ready": True,
            "fork_gate_reason": "ok",
            "ambiguity_score": 0.7,
            "active_tokens_est": 700,
        }
    )
    dec = ctl.decide(current_user_prompt="/ COMMIT decide using evidence", features=feats, q1_text="Q1", commit_titles=["A", "B"])
    assert dec.action == "unfold_then_fork"
    assert "commit" in dec.reason
    assert "support-complete evidence" in dec.fork_query


def test_uncertainty_aware_degrades_to_unfold_when_fork_not_ready() -> None:
    ctl = ContextController(policy="uncertainty_aware")
    feats = dict(BASE)
    feats.update(
        {
            "support_gap_score": 0.55,
            "ambiguity_score": 0.8,
            "pivot_risk": 0.85,
            "specialist_subtask_flag": True,
            "fork_ready": False,
            "fork_gate_reason": "open_lt_min",
        }
    )
    dec = ctl.decide(current_user_prompt="verify final answer with evidence", features=feats)
    assert dec.action == "unfold"
    assert dec.metadata["fork_ready"] is False


def test_budget_aware_forks_under_pressure_when_gap_is_small() -> None:
    ctl = ContextController(policy="budget_aware")
    feats = dict(BASE)
    feats.update(
        {
            "budget_utilization": 0.95,
            "ambiguity_score": 0.7,
            "fork_ready": True,
            "fork_gate_reason": "ok",
            "specialist_subtask_flag": True,
            "pivot_risk": 0.5,
        }
    )
    dec = ctl.decide(current_user_prompt="choose best supported candidate", features=feats)
    assert dec.action == "fork"
    assert dec.reason == "budget_pressure_fork"
