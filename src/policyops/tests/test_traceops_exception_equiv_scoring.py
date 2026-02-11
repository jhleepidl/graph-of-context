from __future__ import annotations

from policyops.traceops_v0.evaluator import _score_step
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def test_exception_equiv_scoring_llm_mode() -> None:
    thread = TraceThread(
        thread_id="TR-EQ",
        level=1,
        scenario="mixed",
        initial_state={"region": "us"},
        steps=[],
        clauses={
            "CEX1": TraceWorldClause(
                clause_id="CEX1",
                thread_id="TR-EQ",
                step_idx=0,
                node_type="EXCEPTION",
                text="Footnote exception: residency mismatch can be tolerated with manual review.",
            ),
            "CEX2": TraceWorldClause(
                clause_id="CEX2",
                thread_id="TR-EQ",
                step_idx=1,
                node_type="EXCEPTION",
                text="Footnote exception: residency mismatch can be tolerated with manual review.",
            ),
        },
        meta={},
    )
    step = TraceStep(
        step_id="TR-EQ-S003",
        thread_id="TR-EQ",
        step_idx=2,
        kind="pivot_check",
        message="pivot",
        state={"region": "us", "residency": "eu"},
        pivot_required_ids=["CEX1"],
        gold=TraceGold(
            decision="require_exception",
            conditions=["exception=CEX1"],
            evidence_ids=["CEX1"],
            evidence_core_ids=["CEX1"],
            evidence_meta_ids=[],
        ),
    )
    pred = {
        "decision": "require_condition",
        "conditions": ["exception=CEX2"],
        "evidence": ["CEX2"],
    }

    score = _score_step(
        thread=thread,
        step=step,
        prediction=pred,
        context_ids=["CEX2"],
        invalidated_ids=[],
        eval_mode="llm",
    )

    assert score["conditions_correct_subset"] is False
    assert score["conditions_correct_subset_equiv"] is True
    assert score["conditions_correct"] is True
    assert score["evidence_core_covered_strict"] is False
    assert score["evidence_core_covered_equiv"] is True
    assert score["evidence_core_id_mismatch_but_equiv_present"] is True
