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
        "goc_unfold_max_nodes": 10,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_traceops_full_history_keeps_correct_answer_when_stale_exists() -> None:
    clauses = {
        "C0001": TraceWorldClause(
            clause_id="C0001",
            thread_id="TR0001",
            step_idx=0,
            node_type="ASSUMPTION",
            text="Old assumption: budget stays high.",
        ),
        "C0002": TraceWorldClause(
            clause_id="C0002",
            thread_id="TR0001",
            step_idx=1,
            node_type="DECISION",
            text="Decision checkpoint: allow",
            depends_on=["C0001"],
        ),
        "C0003": TraceWorldClause(
            clause_id="C0003",
            thread_id="TR0001",
            step_idx=2,
            node_type="UPDATE",
            text="Budget changed; old assumption is stale.",
        ),
    }
    steps = [
        TraceStep(
            step_id="TR0001-S001",
            thread_id="TR0001",
            step_idx=0,
            kind="explore",
            message="explore",
            state={"budget": "high"},
            introduced_clause_ids=["C0001"],
        ),
        TraceStep(
            step_id="TR0001-S002",
            thread_id="TR0001",
            step_idx=1,
            kind="commit",
            message="commit",
            state={"budget": "high"},
            introduced_clause_ids=["C0002"],
        ),
        TraceStep(
            step_id="TR0001-S003",
            thread_id="TR0001",
            step_idx=2,
            kind="update",
            message="update",
            state={"budget": "low"},
            introduced_clause_ids=["C0003"],
            avoid_target_ids=["C0001"],
        ),
        TraceStep(
            step_id="TR0001-S004",
            thread_id="TR0001",
            step_idx=3,
            kind="pivot_check",
            message="pivot check",
            state={"budget": "low"},
            pivot_required_ids=["C0002"],
            gold=TraceGold(
                decision="allow",
                conditions=["budget=low"],
                evidence_ids=["C0002"],
                evidence_core_ids=["C0002"],
                evidence_meta_ids=[],
            ),
        ),
    ]
    thread = TraceThread(
        thread_id="TR0001",
        level=1,
        scenario="contradiction",
        initial_state={"budget": "high"},
        steps=steps,
        clauses=clauses,
        meta={},
    )

    report = evaluate_traceops_method("full", [thread], args=_args())
    rec = report["records"][0]

    assert report["metrics"]["pivot_decision_accuracy"] == 1.0
    assert rec["decision_correct"] is True
    assert rec["stale_present"] is True
    assert rec["stale_count"] == 1
