from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _args(**overrides: object) -> SimpleNamespace:
    base = {
        "traceops_max_steps": 0,
        "traceops_similarity_topk": 8,
        "goc_enable_avoids": True,
        "goc_applicability_seed_enable": True,
        "goc_applicability_seed_topk": 4,
        "goc_unfold_max_nodes": 4,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_traceops_required_force_include_is_oracle_only() -> None:
    clauses = {
        "C0001": TraceWorldClause(
            clause_id="C0001",
            thread_id="TR0001",
            step_idx=0,
            node_type="ASSUMPTION",
            text="Legacy assumption for EU policy branch.",
            state_key="region",
            state_value="eu",
        ),
        "C0002": TraceWorldClause(
            clause_id="C0002",
            thread_id="TR0001",
            step_idx=1,
            node_type="UPDATE",
            text="State update: region switched to us.",
            state_key="region",
            state_value="us",
        ),
    }
    steps = [
        TraceStep(
            step_id="TR0001-S001",
            thread_id="TR0001",
            step_idx=0,
            kind="explore",
            message="explore",
            state={"region": "eu"},
            introduced_clause_ids=["C0001"],
        ),
        TraceStep(
            step_id="TR0001-S002",
            thread_id="TR0001",
            step_idx=1,
            kind="update",
            message="update",
            state={"region": "us"},
            introduced_clause_ids=["C0002"],
        ),
        TraceStep(
            step_id="TR0001-S003",
            thread_id="TR0001",
            step_idx=2,
            kind="pivot_check",
            message="pivot check",
            state={"region": "us"},
            pivot_required_ids=["C0001"],
            gold=TraceGold(
                decision="override_invalidated",
                conditions=[],
                evidence_ids=["C0001"],
                evidence_core_ids=["C0001"],
                evidence_meta_ids=[],
            ),
        ),
    ]
    thread = TraceThread(
        thread_id="TR0001",
        level=1,
        scenario="mixed",
        initial_state={"region": "eu"},
        steps=steps,
        clauses=clauses,
        meta={},
    )

    report_goc = evaluate_traceops_method("goc", [thread], args=_args())
    rec_goc = report_goc["records"][0]
    assert rec_goc["goc_required_force_included_ids"] == []
    assert rec_goc["goc_required_force_included_but_inapplicable_ids"] == []
    assert "C0001" not in rec_goc["e3_context_clause_ids"]

    report_oracle = evaluate_traceops_method("goc_oracle", [thread], args=_args())
    rec_oracle = report_oracle["records"][0]
    assert "C0001" in rec_oracle["goc_required_force_included_ids"]
    assert "C0001" in rec_oracle["goc_required_force_included_but_inapplicable_ids"]
    assert "C0001" in rec_oracle["e3_context_clause_ids"]
