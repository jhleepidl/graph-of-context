from __future__ import annotations

from policyops.traceops_v0.evaluator import _apply_llm_update_evidence_gate
from policyops.traceops_v0.schema import TraceStep, TraceWorldClause


def _mk_step(*, binding_key: str = "budget") -> TraceStep:
    return TraceStep(
        step_id="TR-GATE-S001",
        thread_id="TR-GATE",
        step_idx=1,
        kind="pivot_check",
        message="decide now",
        state={"budget": "low", "retention_tier": "standard"},
        metadata={"binding_key": binding_key},
    )


def test_patch_s_require_condition_without_update_is_allowed_with_rule_support() -> None:
    step = _mk_step(binding_key="budget")
    clauses = {
        "PA": TraceWorldClause(
            clause_id="PA",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="EVIDENCE",
            text="Anchor",
            metadata={"policy_anchor": True},
        ),
        "PC": TraceWorldClause(
            clause_id="PC",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="EVIDENCE",
            text="Codebook",
            metadata={"policy_codebook": True},
        ),
    }
    pred, gate = _apply_llm_update_evidence_gate(
        prediction={"decision": "require_condition", "conditions": [], "evidence": ["PA", "PC"]},
        step=step,
        context_ids=["PA", "PC"],
        clauses=clauses,
    )
    assert pred["decision"] == "require_condition"
    assert gate["gate_forced_needs_more_info"] is False
    assert gate["gate_forced_reason"] == ""


def test_patch_s_allow_deny_with_noop_update_is_forced_to_needs_more_info() -> None:
    step = _mk_step(binding_key="budget")
    clauses = {
        "UN": TraceWorldClause(
            clause_id="UN",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="UPDATE",
            text="No-op update",
            state_key="budget",
            state_value="low",
            metadata={"noop_update": True},
        ),
        "PC": TraceWorldClause(
            clause_id="PC",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="EVIDENCE",
            text="Codebook",
            metadata={"policy_codebook": True},
        ),
    }
    pred, gate = _apply_llm_update_evidence_gate(
        prediction={"decision": "deny", "conditions": [], "evidence": ["UN", "PC"]},
        step=step,
        context_ids=["UN", "PC"],
        clauses=clauses,
    )
    assert pred["decision"] == "needs_more_info"
    assert gate["gate_forced_needs_more_info"] is True
    assert gate["gate_forced_reason"] == "allow_deny_update_is_noop"
    assert gate["allow_deny_commit_without_valid_update"] is True


def test_patch_s_allow_deny_with_wrong_update_key_is_forced() -> None:
    step = _mk_step(binding_key="budget")
    clauses = {
        "UR": TraceWorldClause(
            clause_id="UR",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="UPDATE",
            text="Update retention tier",
            state_key="retention_tier",
            state_value="standard",
        ),
        "PC": TraceWorldClause(
            clause_id="PC",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="EVIDENCE",
            text="Codebook",
            metadata={"policy_codebook": True},
        ),
    }
    pred, gate = _apply_llm_update_evidence_gate(
        prediction={"decision": "allow", "conditions": [], "evidence": ["UR", "PC"]},
        step=step,
        context_ids=["UR", "PC"],
        clauses=clauses,
    )
    assert pred["decision"] == "needs_more_info"
    assert gate["gate_forced_needs_more_info"] is True
    assert gate["gate_forced_reason"] == "allow_deny_update_wrong_key"
    assert gate["allow_deny_commit_without_valid_update"] is True


def test_patch_s_require_condition_with_only_updates_is_forced() -> None:
    step = _mk_step(binding_key="budget")
    clauses = {
        "U1": TraceWorldClause(
            clause_id="U1",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="UPDATE",
            text="Update budget low",
            state_key="budget",
            state_value="low",
        ),
        "U2": TraceWorldClause(
            clause_id="U2",
            thread_id="TR-GATE",
            step_idx=0,
            node_type="UPDATE",
            text="Update budget medium",
            state_key="budget",
            state_value="medium",
        ),
    }
    pred, gate = _apply_llm_update_evidence_gate(
        prediction={"decision": "require_condition", "conditions": [], "evidence": ["U1", "U2"]},
        step=step,
        context_ids=["U1", "U2"],
        clauses=clauses,
    )
    assert pred["decision"] == "needs_more_info"
    assert gate["gate_forced_needs_more_info"] is True
    assert gate["gate_forced_reason"] == "require_condition_missing_rule_evidence"

