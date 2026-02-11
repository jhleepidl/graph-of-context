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
        "goc_unfold_max_nodes": 999,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_goc_rescues_activation_exception_even_if_inapplicable() -> None:
    thread = TraceThread(
        thread_id="TR-ACT",
        level=2,
        scenario="mixed",
        initial_state={"deadline": "flex"},
        steps=[
            TraceStep(
                step_id="TR-ACT-S001",
                thread_id="TR-ACT",
                step_idx=0,
                kind="commit",
                message="commit",
                state={"deadline": "flex"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-ACT-S002",
                thread_id="TR-ACT",
                step_idx=1,
                kind="update",
                message="update",
                state={"deadline": "flex"},
                introduced_clause_ids=["C002", "C003"],
            ),
            TraceStep(
                step_id="TR-ACT-S003",
                thread_id="TR-ACT",
                step_idx=2,
                kind="pivot_check",
                message="pivot",
                state={"deadline": "flex"},
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
                thread_id="TR-ACT",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-ACT",
                step_idx=1,
                node_type="UPDATE",
                text="Update: deadline changed to flex.",
                state_key="deadline",
                state_value="flex",
            ),
            "C003": TraceWorldClause(
                clause_id="C003",
                thread_id="TR-ACT",
                step_idx=1,
                node_type="EXCEPTION",
                text="Exception activated: manual override allowed if deadline is tight.",
                tags=["exception", "activation"],
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]

    assert "C003" in rec["e3_context_clause_ids"]
    assert "C003" in rec["goc_exception_rescue_ids"]
    assert "activation" in rec["goc_exception_rescue_reason"]


def test_goc_rescues_latent_exception_on_residency_region_mismatch() -> None:
    thread = TraceThread(
        thread_id="TR-LAT",
        level=2,
        scenario="mixed",
        initial_state={"region": "us", "residency": "eu"},
        steps=[
            TraceStep(
                step_id="TR-LAT-S001",
                thread_id="TR-LAT",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us", "residency": "eu"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-LAT-S002",
                thread_id="TR-LAT",
                step_idx=1,
                kind="commit",
                message="commit",
                state={"region": "us", "residency": "eu"},
                introduced_clause_ids=["C002"],
            ),
            TraceStep(
                step_id="TR-LAT-S003",
                thread_id="TR-LAT",
                step_idx=2,
                kind="pivot_check",
                message="pivot",
                state={"region": "us", "residency": "eu"},
                pivot_required_ids=["C001", "C002"],
                gold=TraceGold(
                    decision="require_exception",
                    conditions=["exception=C001"],
                    evidence_ids=["C001", "C002"],
                    evidence_core_ids=["C001", "C002"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR-LAT",
                step_idx=0,
                node_type="EXCEPTION",
                text="Footnote exception: residency mismatch can be tolerated with manual review.",
                tags=["exception", "latent"],
                metadata={"salience": "low"},
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-LAT",
                step_idx=1,
                node_type="DECISION",
                text="Decision checkpoint: allow.",
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]

    assert "C001" in rec["e3_context_clause_ids"]
    assert "C001" in rec["goc_exception_rescue_ids"]
    assert "latent_mismatch" in rec["goc_exception_rescue_reason"]


def test_goc_update_history_rescue_keeps_last_two_updates_for_flipped_key() -> None:
    thread = TraceThread(
        thread_id="TR-FLIP",
        level=2,
        scenario="contradiction",
        initial_state={"region": "eu"},
        steps=[
            TraceStep(
                step_id="TR-FLIP-S001",
                thread_id="TR-FLIP",
                step_idx=0,
                kind="commit",
                message="commit",
                state={"region": "eu"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-FLIP-S002",
                thread_id="TR-FLIP",
                step_idx=1,
                kind="update",
                message="update",
                state={"region": "us"},
                introduced_clause_ids=["C002"],
            ),
            TraceStep(
                step_id="TR-FLIP-S003",
                thread_id="TR-FLIP",
                step_idx=2,
                kind="update",
                message="update",
                state={"region": "apac"},
                introduced_clause_ids=["C003"],
            ),
            TraceStep(
                step_id="TR-FLIP-S004",
                thread_id="TR-FLIP",
                step_idx=3,
                kind="pivot_check",
                message="pivot",
                state={"region": "apac"},
                pivot_required_ids=["C001", "C003"],
                gold=TraceGold(
                    decision="override_invalidated",
                    conditions=["apply_latest_update"],
                    evidence_ids=["C001", "C003"],
                    evidence_core_ids=["C001", "C003"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "C001": TraceWorldClause(
                clause_id="C001",
                thread_id="TR-FLIP",
                step_idx=0,
                node_type="DECISION",
                text="Decision checkpoint: allow.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-FLIP",
                step_idx=1,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
            "C003": TraceWorldClause(
                clause_id="C003",
                thread_id="TR-FLIP",
                step_idx=2,
                node_type="UPDATE",
                text="Update: region changed to apac.",
                state_key="region",
                state_value="apac",
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]

    assert "C002" in rec["goc_update_history_rescue_ids"]
    assert "C002" in rec["e3_context_clause_ids"]
    assert rec["goc_update_history_rescue_count"] >= 1
