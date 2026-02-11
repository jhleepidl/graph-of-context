from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import evaluate_traceops_method
from policyops.traceops_v0.schema import TraceGold, TraceStep, TraceThread, TraceWorldClause


def _args(*, depwalk_enable: bool) -> SimpleNamespace:
    return SimpleNamespace(
        traceops_max_steps=0,
        traceops_similarity_topk=8,
        goc_enable_avoids=True,
        goc_applicability_seed_enable=False,
        goc_applicability_seed_topk=8,
        goc_dependency_closure_enable=False,
        goc_dependency_closure_topk=12,
        goc_dependency_closure_hops=1,
        goc_dependency_closure_universe="candidates",
        goc_depwalk_enable=depwalk_enable,
        goc_depwalk_hops=2,
        goc_depwalk_topk_per_hop=6,
        goc_exception_rescue_topk=0,
        goc_update_history_rescue_topk=0,
        goc_unfold_max_nodes=999,
        traceops_eval_mode="deterministic",
    )


def _thread_with_hidden_core() -> TraceThread:
    return TraceThread(
        thread_id="TR-P18",
        level=2,
        scenario="indirect",
        initial_state={"region": "us"},
        steps=[
            TraceStep(
                step_id="TR-P18-S001",
                thread_id="TR-P18",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us"},
                introduced_clause_ids=["C002"],
            ),
            TraceStep(
                step_id="TR-P18-S002",
                thread_id="TR-P18",
                step_idx=1,
                kind="commit",
                message="commit",
                state={"region": "us"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-P18-S003",
                thread_id="TR-P18",
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
                metadata={
                    "core_necessity_flip_count": 2,
                    "core_necessity_all_required": True,
                    "core_necessity_failed": False,
                    "trap_decision_label": "deny",
                    "trap_decision_flip": True,
                    "hidden_core_ids": ["C002"],
                    "hidden_core_parent_ids": ["C001"],
                },
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR-P18",
                step_idx=1,
                node_type="DECISION",
                text="Decision checkpoint: allow",
                depends_on=["C002"],
                metadata={"decision_label": "allow"},
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-P18",
                step_idx=0,
                node_type="ASSUMPTION",
                text="Hidden binding marker",
            ),
        },
        meta={},
    )


def test_phase18_metrics_include_hidden_core_rescue_and_metadata() -> None:
    thread = _thread_with_hidden_core()

    report_depwalk_on = evaluate_traceops_method("goc", [thread], args=_args(depwalk_enable=True))
    rec_on = report_depwalk_on["records"][0]
    metrics_on = report_depwalk_on["metrics"]
    assert rec_on["core_necessity_all_required"] is True
    assert rec_on["core_necessity_flip_count"] == 2
    assert rec_on["trap_decision_flip"] is True
    assert rec_on["hidden_core_ids"] == ["C002"]
    assert "C002" in rec_on["goc_depwalk_added_ids"]
    assert float(metrics_on["hidden_core_rescued_by_depwalk_rate"]) == 1.0

    report_depwalk_off = evaluate_traceops_method("goc", [thread], args=_args(depwalk_enable=False))
    rec_off = report_depwalk_off["records"][0]
    metrics_off = report_depwalk_off["metrics"]
    assert "C002" not in rec_off["e3_context_clause_ids"]
    assert float(metrics_off["hidden_core_missing_without_depwalk_rate"]) == 1.0

