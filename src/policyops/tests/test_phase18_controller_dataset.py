from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.controller_dataset import build_dep_scoped_fork_from_base_context, evaluate_pivot_actions
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _args(**overrides: object) -> SimpleNamespace:
    base = {
        "traceops_max_steps": 0,
        "traceops_similarity_topk": 8,
        "goc_enable_avoids": True,
        "goc_applicability_seed_enable": True,
        "goc_applicability_seed_topk": 8,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
        "goc_depwalk_enable": True,
        "goc_depwalk_hops": 2,
        "goc_depwalk_topk_per_hop": 6,
        "goc_smart_context_enable": False,
        "fork_k": 2,
        "fork_max_tokens": 200,
        "fork_include_recent_active": True,
        "fork_recent_active_n": 1,
        "fork_dependency_hops": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _thread_hidden_support() -> TraceThread:
    steps = [
        TraceStep(
            step_id="S0",
            thread_id="TR-HIDDEN",
            step_idx=0,
            kind="explore",
            message="read old rule",
            state={"city": "A"},
            introduced_clause_ids=["C0"],
        ),
        TraceStep(
            step_id="S1",
            thread_id="TR-HIDDEN",
            step_idx=1,
            kind="update",
            message="confirm airport and rent",
            state={"city": "A", "city_a_status": "eligible"},
            introduced_clause_ids=["C1"],
        ),
    ]
    clauses = {
        "C0": TraceWorldClause(
            clause_id="C0",
            thread_id="TR-HIDDEN",
            step_idx=0,
            node_type="DECISION",
            text="Decision checkpoint: allow if airport access and low rent.",
        ),
        "C1": TraceWorldClause(
            clause_id="C1",
            thread_id="TR-HIDDEN",
            step_idx=1,
            node_type="UPDATE",
            text="Update: City A airport access confirmed and rent changed to low.",
            state_key="city_a_status",
            state_value="eligible",
            depends_on=["C0"],
        ),
    }
    # Add many later irrelevant updates so the recent active approximation drops C0/C1.
    for idx in range(2, 10):
        cid = f"C{idx}"
        sid = f"S{idx}"
        steps.append(
            TraceStep(
                step_id=sid,
                thread_id="TR-HIDDEN",
                step_idx=idx,
                kind="update",
                message=f"irrelevant update {idx}",
                state={"city": "A", "city_a_status": "eligible"},
                introduced_clause_ids=[cid],
            )
        )
        clauses[cid] = TraceWorldClause(
            clause_id=cid,
            thread_id="TR-HIDDEN",
            step_idx=idx,
            node_type="UPDATE",
            text=f"Update: City Z museum subsidy changed to level {idx}.",
            state_key=f"city_z_museum_{idx}",
            state_value=f"level_{idx}",
        )

    steps.append(
        TraceStep(
            step_id="SP",
            thread_id="TR-HIDDEN",
            step_idx=10,
            kind="pivot_check",
            message="Should City A be allowed under the airport and rent rule?",
            state={"city": "A", "city_a_status": "eligible"},
            pivot_required_ids=["C0", "C1"],
            gold=TraceGold(
                decision="allow",
                conditions=[],
                evidence_ids=["C0", "C1"],
                evidence_core_ids=["C0", "C1"],
                evidence_meta_ids=[],
            ),
            metadata={"trap_decision_checkpoint_ids": ["C0"]},
        )
    )
    return TraceThread(
        thread_id="TR-HIDDEN",
        level=3,
        scenario="indirect",
        initial_state={"city": "A", "city_a_status": "eligible"},
        steps=steps,
        clauses=clauses,
        meta={},
    )


def test_direct_fork_needs_unfold_to_recover_support() -> None:
    thread = _thread_hidden_support()
    pivot = thread.steps[-1]
    ordered_history = [cid for step in thread.steps[:-1] for cid in step.introduced_clause_ids]

    direct_ids, _ = build_dep_scoped_fork_from_base_context(
        thread=thread,
        step=pivot,
        ordered_history=ordered_history,
        base_context_ids=["C4", "C5", "C6", "C7", "C8", "C9"],
        args=_args(),
        allow_gold_support=False,
    )
    utf_ids, _ = build_dep_scoped_fork_from_base_context(
        thread=thread,
        step=pivot,
        ordered_history=ordered_history,
        base_context_ids=["C0", "C1", "C8", "C9"],
        args=_args(),
        allow_gold_support=False,
    )

    assert "C0" not in direct_ids and "C1" not in direct_ids
    assert {"C0", "C1"}.issubset(set(utf_ids))


def test_pivot_eval_prefers_unfold_then_fork_when_support_is_hidden() -> None:
    thread = _thread_hidden_support()
    history_ids = [cid for step in thread.steps[:-1] for cid in step.introduced_clause_ids]
    invalidated_ids: list[str] = []
    pivot = thread.steps[-1]

    out = evaluate_pivot_actions(
        thread=thread,
        step=pivot,
        history_ids=history_ids,
        invalidated_ids=invalidated_ids,
        args=_args(),
        none_mode="agent_fold",
        token_weight=0.10,
        coverage_weight=0.15,
        leakage_weight=0.0,
        dev_ratio=0.5,
        split_seed=7,
    )

    assert out.actions["fork"].score["answer_correct"] is False
    assert out.actions["unfold_then_fork"].score["answer_correct"] is True
    assert out.best_action in {"unfold", "unfold_then_fork"}
    assert out.actions["unfold_then_fork"].utility >= out.actions["fork"].utility
