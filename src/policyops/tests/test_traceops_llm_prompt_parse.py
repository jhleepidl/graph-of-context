from __future__ import annotations

from policyops.traceops_v0.evaluator import _build_traceops_llm_prompt, _parse_llm_json
from policyops.traceops_v0.schema import TraceStep, TraceThread, TraceWorldClause


def _sample_thread_and_step() -> tuple[TraceThread, TraceStep]:
    thread = TraceThread(
        thread_id="TR0001",
        level=1,
        scenario="mixed",
        initial_state={"region": "eu"},
        steps=[],
        clauses={
            "C0001": TraceWorldClause(
                clause_id="C0001",
                thread_id="TR0001",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow under EU residency.",
            ),
            "C0002": TraceWorldClause(
                clause_id="C0002",
                thread_id="TR0001",
                step_idx=1,
                node_type="EXCEPTION",
                text="Exception: manual review can override baseline decision.",
            ),
        },
        meta={},
    )
    step = TraceStep(
        step_id="TR0001-S003",
        thread_id="TR0001",
        step_idx=2,
        kind="pivot_check",
        message="Given the updates, what is the final decision now?",
        state={"region": "eu"},
    )
    return thread, step


def test_traceops_llm_prompt_contains_schema_and_clause_ids() -> None:
    thread, step = _sample_thread_and_step()
    prompt = _build_traceops_llm_prompt(step, thread, ["C0001", "C0002"])

    assert "Output STRICT JSON only" in prompt
    assert '"decision":"allow|deny|require_condition|needs_more_info"' in prompt
    assert "You MUST choose one of the 4 decision labels exactly as written." in prompt
    assert "CLAUSE C0001 (DECISION)" in prompt
    assert "CLAUSE C0002 (EXCEPTION)" in prompt
    assert "Allowed evidence clause IDs" in prompt


def test_traceops_parse_llm_json_perfect_json() -> None:
    parsed = _parse_llm_json(
        '{"decision":"allow","conditions":["region=eu"],"evidence":["C0001"]}'
    )
    assert parsed["decision"] == "allow"
    assert parsed["conditions"] == ["region=eu"]
    assert parsed["evidence"] == ["C0001"]
    assert parsed["parse_error"] is False


def test_traceops_parse_llm_json_with_wrapped_text() -> None:
    parsed = _parse_llm_json(
        "Sure. Here is my answer:\n"
        '{"decision":"deny","conditions":[],"evidence":["C0002"]}\n'
        "Done."
    )
    assert parsed["decision"] == "deny"
    assert parsed["evidence"] == ["C0002"]
    assert parsed["parse_error"] is False


def test_traceops_parse_llm_json_invalid_fallback() -> None:
    parsed = _parse_llm_json("not-json-at-all")
    assert parsed["decision"] == "needs_more_info"
    assert parsed["conditions"] == []
    assert parsed["evidence"] == []
    assert parsed["parse_error"] is True
