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
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
        "goc_depwalk_enable": True,
        "goc_depwalk_hops": 2,
        "goc_depwalk_topk_per_hop": 6,
        "goc_exception_rescue_topk": 0,
        "goc_update_history_rescue_topk": 0,
        "goc_unfold_max_nodes": 999,
        "traceops_eval_mode": "deterministic",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_goc_depwalk_adds_relevant_neighbor_and_keeps_rescue_off() -> None:
    thread = TraceThread(
        thread_id="TR-DPW",
        level=2,
        scenario="mixed",
        initial_state={"region": "us"},
        steps=[
            TraceStep(
                step_id="TR-DPW-S001",
                thread_id="TR-DPW",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us"},
                introduced_clause_ids=["C002", "C003"],
            ),
            TraceStep(
                step_id="TR-DPW-S002",
                thread_id="TR-DPW",
                step_idx=1,
                kind="commit",
                message="commit",
                state={"region": "us"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-DPW-S003",
                thread_id="TR-DPW",
                step_idx=2,
                kind="pivot_check",
                message="pivot",
                state={"region": "us"},
                pivot_required_ids=["C001"],
                gold=TraceGold(
                    decision="allow",
                    conditions=[],
                    evidence_ids=["C001"],
                    evidence_core_ids=["C001"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR-DPW",
                step_idx=1,
                node_type="DECISION",
                text="Decision checkpoint: allow in region us.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-DPW",
                step_idx=0,
                node_type="EVIDENCE",
                text="Evidence snapshot confirms region us request constraints.",
            ),
            "C003": TraceWorldClause(
                clause_id="C003",
                thread_id="TR-DPW",
                step_idx=0,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]
    assert "C002" in rec["goc_depwalk_added_ids"]
    assert rec["goc_depwalk_added_count"] >= 1
    assert "C002" in rec["e3_context_clause_ids"]
    assert rec["goc_rescue_ran"] is False
    assert rec["goc_rescue_reason_short"] == "not_needed"
