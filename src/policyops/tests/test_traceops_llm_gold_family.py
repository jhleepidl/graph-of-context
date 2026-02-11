from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import _gold_decision_family, evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


class _FakeLLMClient:
    def generate_with_usage(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 256):
        return (
            '{"decision":"require_condition","conditions":[],"evidence":["C0001"]}',
            {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )


def _args(cache_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        traceops_max_steps=0,
        traceops_eval_mode="llm",
        traceops_llm_temperature=0.0,
        traceops_llm_max_output_tokens=256,
        traceops_llm_max_pivots=0,
        traceops_llm_cache_dir=cache_dir,
        traceops_similarity_topk=8,
        model="gpt-4.1-mini",
    )


def test_gold_decision_family_mapping() -> None:
    assert _gold_decision_family("defer") == "needs_more_info"
    assert _gold_decision_family("require_exception") == "require_condition"
    assert _gold_decision_family("override_invalidated") == "require_condition"


def test_llm_mode_uses_family_correctness_for_primary_metric(tmp_path) -> None:
    thread = TraceThread(
        thread_id="TR0001",
        level=1,
        scenario="mixed",
        initial_state={"region": "eu"},
        steps=[
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
                kind="pivot_check",
                message="pivot",
                state={"region": "eu"},
                pivot_required_ids=["C0001"],
                gold=TraceGold(
                    decision="require_exception",
                    conditions=[],
                    evidence_ids=["C0001"],
                    evidence_core_ids=["C0001"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C0001": TraceWorldClause(
                clause_id="C0001",
                thread_id="TR0001",
                step_idx=0,
                node_type="EXCEPTION",
                text="Exception clause.",
            )
        },
        meta={},
    )

    report = evaluate_traceops_method(
        "full",
        [thread],
        args=_args(str(tmp_path / "traceops_llm_cache")),
        client=_FakeLLMClient(),
    )
    rec = report["records"][0]

    assert rec["gold_decision_family"] == "require_condition"
    assert rec["decision_correct_exact"] is False
    assert rec["decision_correct_family"] is True
    assert rec["decision_correct"] is True
    assert report["metrics"]["pivot_decision_accuracy"] == 1.0
