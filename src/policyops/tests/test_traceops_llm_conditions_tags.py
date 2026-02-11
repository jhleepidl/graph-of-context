from __future__ import annotations

from policyops.traceops_v0.evaluator import _build_traceops_llm_prompt, _score_step
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _sample_thread_step() -> tuple[TraceThread, TraceStep]:
    thread = TraceThread(
        thread_id="TRX",
        level=1,
        scenario="mixed",
        initial_state={"region": "us", "budget": "low"},
        steps=[],
        clauses={
            "C1000": TraceWorldClause(
                clause_id="C1000",
                thread_id="TRX",
                step_idx=0,
                node_type="EXCEPTION",
                text="Exception condition note",
            ),
        },
        meta={},
    )
    step = TraceStep(
        step_id="TRX-S001",
        thread_id="TRX",
        step_idx=1,
        kind="pivot_check",
        message="Pivot check question",
        state={"region": "us", "budget": "low"},
        gold=TraceGold(
            decision="allow",
            conditions=["region=us"],
            evidence_ids=["C1000"],
            evidence_core_ids=["C1000"],
            evidence_meta_ids=[],
        ),
    )
    return thread, step


def test_prompt_contains_allowed_conditions_tags() -> None:
    thread, step = _sample_thread_step()
    prompt = _build_traceops_llm_prompt(step, thread, ["C1000"])

    assert "ALLOWED_CONDITIONS:" in prompt
    assert "- apply_latest_update" in prompt
    assert "- region=us" in prompt
    assert "- budget=low" in prompt
    assert "- exception=C1000" in prompt


def test_llm_conditions_subset_is_primary_but_exact_is_diagnostic() -> None:
    thread, step = _sample_thread_step()
    prediction = {
        "decision": "allow",
        "conditions": ["region=us", "budget=low"],
        "evidence": ["C1000"],
    }

    score_llm = _score_step(
        thread=thread,
        step=step,
        prediction=prediction,
        context_ids=["C1000"],
        invalidated_ids=[],
        eval_mode="llm",
    )
    assert score_llm["conditions_correct_subset"] is True
    assert score_llm["conditions_correct_exact"] is False
    assert score_llm["conditions_correct"] is True

    score_deterministic = _score_step(
        thread=thread,
        step=step,
        prediction=prediction,
        context_ids=["C1000"],
        invalidated_ids=[],
        eval_mode="deterministic",
    )
    assert score_deterministic["conditions_correct_subset"] is True
    assert score_deterministic["conditions_correct_exact"] is False
    assert score_deterministic["conditions_correct"] is False
