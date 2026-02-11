from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import _clause_applicable, evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _mk_exception(text: str) -> TraceWorldClause:
    return TraceWorldClause(
        clause_id="CEX",
        thread_id="TR",
        step_idx=0,
        node_type="EXCEPTION",
        text=text,
    )


def test_exception_applicability_if_state_pattern() -> None:
    clause = _mk_exception("Exception applies if deadline is tight.")
    assert _clause_applicable(clause, {"deadline": "tight"}) is True
    assert _clause_applicable(clause, {"deadline": "flex"}) is False


def test_exception_applicability_residency_mismatch() -> None:
    clause = _mk_exception("Footnote exception: residency mismatch can be tolerated.")
    assert _clause_applicable(clause, {"residency": "eu", "region": "us"}) is True
    assert _clause_applicable(clause, {"residency": "eu", "region": "eu"}) is False
    assert _clause_applicable(clause, {"budget": "low"}) is True


def test_goc_context_excludes_inapplicable_exception_from_anchor_seed_and_closure() -> None:
    thread = TraceThread(
        thread_id="TR0001",
        level=1,
        scenario="mixed",
        initial_state={"deadline": "flex"},
        steps=[
            TraceStep(
                step_id="TR0001-S001",
                thread_id="TR0001",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"deadline": "flex"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR0001-S002",
                thread_id="TR0001",
                step_idx=1,
                kind="update",
                message="update",
                state={"deadline": "flex"},
                introduced_clause_ids=["C002", "C003"],
            ),
            TraceStep(
                step_id="TR0001-S003",
                thread_id="TR0001",
                step_idx=2,
                kind="pivot_check",
                message="pivot",
                state={"deadline": "flex"},
                pivot_required_ids=["C001"],
                gold=TraceGold(
                    decision="allow",
                    conditions=["deadline=flex"],
                    evidence_ids=["C001"],
                    evidence_core_ids=["C001"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR0001",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR0001",
                step_idx=1,
                node_type="UPDATE",
                text="Update: deadline changed to flex.",
                state_key="deadline",
                state_value="flex",
            ),
            "C003": TraceWorldClause(
                clause_id="C003",
                thread_id="TR0001",
                step_idx=1,
                node_type="EXCEPTION",
                text="Exception activated if deadline is tight.",
                depends_on=["C002"],
            ),
        },
        meta={},
    )

    args = SimpleNamespace(
        traceops_max_steps=0,
        traceops_similarity_topk=8,
        goc_enable_avoids=True,
        goc_applicability_seed_enable=True,
        goc_applicability_seed_topk=8,
        goc_unfold_max_nodes=999,
            goc_dependency_closure_enable=True,
            goc_dependency_closure_topk=12,
            goc_dependency_closure_hops=1,
            goc_dependency_closure_universe="candidates",
            goc_exception_rescue_topk=0,
            goc_update_history_rescue_topk=0,
        )

    report = evaluate_traceops_method("goc", [thread], args=args)
    rec = report["records"][0]

    assert "C003" not in rec["e3_context_clause_ids"]
    assert rec["goc_exception_injected_count"] == 0
    assert rec["goc_exception_rescue_count"] == 0
