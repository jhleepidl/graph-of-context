from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _args(**overrides: object) -> SimpleNamespace:
    base = {
        "traceops_max_steps": 0,
        "traceops_similarity_topk": 8,
        "goc_enable_avoids": True,
        "goc_applicability_seed_enable": False,
        "goc_applicability_seed_topk": 8,
        "goc_unfold_max_nodes": 999,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
        "goc_depwalk_enable": False,
        "goc_smart_context_enable": False,
        "fork_k": 2,
        "fork_max_tokens": 200,
        "fork_include_recent_active": True,
        "fork_recent_active_n": 1,
        "fork_dependency_hops": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _thread() -> TraceThread:
    return TraceThread(
        thread_id="TR-FORK",
        level=3,
        scenario="indirect",
        initial_state={"city": "A"},
        steps=[
            TraceStep(
                step_id="S1",
                thread_id="TR-FORK",
                step_idx=0,
                kind="explore",
                message="read anchor",
                state={"city": "A"},
                introduced_clause_ids=["C1"],
            ),
            TraceStep(
                step_id="S2",
                thread_id="TR-FORK",
                step_idx=1,
                kind="update",
                message="update city a",
                state={"city": "A"},
                introduced_clause_ids=["C2"],
            ),
            TraceStep(
                step_id="S3",
                thread_id="TR-FORK",
                step_idx=2,
                kind="explore",
                message="irrelevant museum update",
                state={"city": "A"},
                introduced_clause_ids=["C3"],
            ),
            TraceStep(
                step_id="S4",
                thread_id="TR-FORK",
                step_idx=3,
                kind="pivot_check",
                message="Should City A be allowed under the airport and rent rule?",
                state={"city": "A"},
                pivot_required_ids=["C1", "C2"],
                gold=TraceGold(
                    decision="allow",
                    conditions=[],
                    evidence_ids=["C1", "C2"],
                    evidence_core_ids=["C1", "C2"],
                    evidence_meta_ids=[],
                ),
                metadata={"trap_decision_checkpoint_ids": ["C1"]},
            ),
        ],
        clauses={
            "C1": TraceWorldClause(
                clause_id="C1",
                thread_id="TR-FORK",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow if airport access and low rent.",
            ),
            "C2": TraceWorldClause(
                clause_id="C2",
                thread_id="TR-FORK",
                step_idx=1,
                node_type="UPDATE",
                text="Update: City A airport access confirmed and rent changed to low.",
                state_key="city_a_status",
                state_value="eligible",
                depends_on=["C1"],
            ),
            "C3": TraceWorldClause(
                clause_id="C3",
                thread_id="TR-FORK",
                step_idx=2,
                node_type="UPDATE",
                text="Update: City Z museum subsidy expanded.",
                state_key="city_z_museum",
                state_value="expanded",
            ),
        },
        meta={},
    )


def test_goc_fork_dep_keeps_support_complete_scope() -> None:
    report = evaluate_traceops_method("goc_fork_dep", [_thread()], args=_args())
    rec = report["records"][0]
    ctx = set(rec["e3_context_clause_ids"])
    assert {"C1", "C2"}.issubset(ctx)
    assert rec["fork_enabled"] is True
    assert rec["fork_scope_mode"] == "dep"
    assert rec["fork_support_coverage"] >= 1.0


def test_goc_fork_sim_drops_dependency_more_often_than_dep() -> None:
    dep = evaluate_traceops_method("goc_fork_dep", [_thread()], args=_args())["records"][0]
    sim = evaluate_traceops_method("goc_fork_sim", [_thread()], args=_args(fork_k=1))["records"][0]
    assert dep["fork_support_coverage"] >= sim["fork_support_coverage"]
    assert dep["fork_scope_leakage"] <= 1.0


def test_goc_fork_full_marks_full_scope() -> None:
    rec = evaluate_traceops_method("goc_fork_full", [_thread()], args=_args())["records"][0]
    assert rec["fork_scope_mode"] == "full"
    assert rec["fork_context_count"] >= 2
