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
        "goc_dependency_closure_enable": True,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "world",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_traceops_dependency_closure_world_excludes_future_clause_ids() -> None:
    clauses = {
        "C0001": TraceWorldClause(
            clause_id="C0001",
            thread_id="TR0001",
            step_idx=0,
            node_type="ASSUMPTION",
            text="Assume region is eu.",
        ),
        "C0002": TraceWorldClause(
            clause_id="C0002",
            thread_id="TR0001",
            step_idx=1,
            node_type="DECISION",
            text="Decision checkpoint: allow",
            depends_on=["C0001"],
        ),
        # Future node: if world universe is unrestricted this can leak in via closure.
        "C0003": TraceWorldClause(
            clause_id="C0003",
            thread_id="TR0001",
            step_idx=3,
            node_type="OPTION",
            text="Future option branch tied to prior decision.",
            depends_on=["C0002"],
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
            kind="commit",
            message="commit",
            state={"region": "eu"},
            introduced_clause_ids=["C0002"],
        ),
        TraceStep(
            step_id="TR0001-S003",
            thread_id="TR0001",
            step_idx=2,
            kind="pivot_check",
            message="pivot check",
            state={"region": "eu"},
            pivot_required_ids=["C0002"],
            gold=TraceGold(
                decision="allow",
                conditions=[],
                evidence_ids=["C0002"],
                evidence_core_ids=["C0002"],
                evidence_meta_ids=[],
            ),
        ),
        TraceStep(
            step_id="TR0001-S004",
            thread_id="TR0001",
            step_idx=3,
            kind="explore",
            message="future explore",
            state={"region": "eu"},
            introduced_clause_ids=["C0003"],
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

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]

    assert "C0003" not in rec["goc_dependency_closure_added_ids"]
    assert "C0003" not in rec["e3_context_clause_ids"]
    assert rec["goc_dependency_closure_universe_effective_size"] == 2
