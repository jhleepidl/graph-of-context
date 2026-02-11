from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from policyops.traceops_v0.evaluator import evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_with_usage(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> tuple[str, dict]:
        _ = (prompt, temperature, max_output_tokens)
        self.calls += 1
        return (
            '{"decision":"allow","conditions":[],"evidence":["C001"]}',
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _thread() -> TraceThread:
    return TraceThread(
        thread_id="TR-LLM",
        level=1,
        scenario="mixed",
        initial_state={"region": "us"},
        steps=[
            TraceStep(
                step_id="TR-LLM-S001",
                thread_id="TR-LLM",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-LLM-S002",
                thread_id="TR-LLM",
                step_idx=1,
                kind="update",
                message="update",
                state={"region": "us"},
                introduced_clause_ids=["C002"],
            ),
            TraceStep(
                step_id="TR-LLM-S003",
                thread_id="TR-LLM",
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
                thread_id="TR-LLM",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-LLM",
                step_idx=1,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
        },
        meta={},
    )


def _args(tmp_dir: Path, *, scope: str, sample_rate: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(
        traceops_max_steps=0,
        traceops_eval_mode="llm",
        traceops_llm_temperature=0.0,
        traceops_llm_max_output_tokens=256,
        traceops_llm_max_pivots=0,
        traceops_llm_cache_dir=str(tmp_dir),
        traceops_llm_seed=0,
        traceops_llm_eval_scope=scope,
        traceops_llm_sample_rate=sample_rate,
        traceops_similarity_topk=8,
    )


def test_traceops_llm_eval_scope_all_calls_every_step(tmp_path: Path) -> None:
    client = _FakeClient()
    report = evaluate_traceops_method(
        "full",
        [_thread()],
        args=_args(tmp_path / "cache_all", scope="all"),
        client=client,
    )
    assert client.calls == 3
    assert len(report["records"]) == 1
    assert report["records"][0]["traceops_llm_eval_scope"] == "all"
    assert report["records"][0]["sampled_step"] is True


def test_traceops_llm_eval_scope_pivots_only_calls_pivot_step(tmp_path: Path) -> None:
    client = _FakeClient()
    report = evaluate_traceops_method(
        "full",
        [_thread()],
        args=_args(tmp_path / "cache_pivots", scope="pivots"),
        client=client,
    )
    assert client.calls == 1
    assert len(report["records"]) == 1
    assert report["records"][0]["traceops_llm_eval_scope"] == "pivots"
    assert report["records"][0]["sampled_step"] is True


def test_traceops_llm_eval_scope_sample_can_skip_all_steps(tmp_path: Path) -> None:
    client = _FakeClient()
    report = evaluate_traceops_method(
        "full",
        [_thread()],
        args=_args(tmp_path / "cache_sample", scope="sample", sample_rate=0.0),
        client=client,
    )
    assert client.calls == 0
    assert report["records"] == []
