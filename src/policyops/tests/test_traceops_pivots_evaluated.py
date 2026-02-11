from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from policyops.traceops_v0.evaluator import evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


class _FakeClient:
    def generate_with_usage(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> tuple[str, dict]:
        _ = (prompt, temperature, max_output_tokens)
        return (
            '{"decision":"allow","conditions":[],"evidence":["C001"]}',
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _thread_two_pivots() -> TraceThread:
    return TraceThread(
        thread_id="TR-PVT",
        level=2,
        scenario="mixed",
        initial_state={"region": "us"},
        steps=[
            TraceStep(
                step_id="TR-PVT-S001",
                thread_id="TR-PVT",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-PVT-S002",
                thread_id="TR-PVT",
                step_idx=1,
                kind="pivot_check",
                message="pivot-1",
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
            TraceStep(
                step_id="TR-PVT-S003",
                thread_id="TR-PVT",
                step_idx=2,
                kind="update",
                message="update",
                state={"region": "us"},
                introduced_clause_ids=["C002"],
            ),
            TraceStep(
                step_id="TR-PVT-S004",
                thread_id="TR-PVT",
                step_idx=3,
                kind="pivot_check",
                message="pivot-2",
                state={"region": "us"},
                pivot_required_ids=["C002"],
                gold=TraceGold(
                    decision="allow",
                    conditions=[],
                    evidence_ids=["C002"],
                    evidence_core_ids=["C002"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR-PVT",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-PVT",
                step_idx=2,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
        },
        meta={},
    )


def _args(cache_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        traceops_max_steps=0,
        traceops_eval_mode="llm",
        traceops_llm_temperature=0.0,
        traceops_llm_max_output_tokens=256,
        traceops_llm_max_pivots=1,
        traceops_llm_cache_dir=str(cache_dir),
        traceops_llm_seed=0,
        traceops_llm_eval_scope="pivots",
        traceops_llm_sample_rate=0.2,
        traceops_similarity_topk=8,
    )


def test_pivots_evaluated_respects_llm_max_pivots_in_pivots_scope(tmp_path: Path) -> None:
    report = evaluate_traceops_method(
        "full",
        [_thread_two_pivots()],
        args=_args(tmp_path / "cache"),
        client=_FakeClient(),
    )
    metrics = dict(report.get("metrics") or {})
    assert int(metrics.get("pivots_available_total", -1)) == 2
    assert int(metrics.get("pivots_evaluated", -1)) == 1
    assert metrics.get("traceops_llm_eval_scope") == "pivots"
