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
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
        "goc_depwalk_enable": True,
        "goc_depwalk_hops": 2,
        "goc_depwalk_topk_per_hop": 6,
        "goc_smart_context_enable": False,
        "goc_smart_cap_option": 0,
        "goc_smart_cap_assumption": 2,
        "goc_smart_cap_update": 4,
        "goc_smart_cap_exception": 2,
        "goc_smart_cap_evidence": 2,
        "goc_exception_rescue_topk": 0,
        "goc_update_history_rescue_topk": 0,
        "goc_unfold_max_nodes": 999,
        "traceops_eval_mode": "deterministic",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_goc_depwalk_adds_relevant_neighbor_and_keeps_rescue_off() -> None:
    thread = TraceThread(
        thread_id="TR-DPW",
        level=2,
        scenario="mixed",
        initial_state={"region": "us"},
        steps=[
            TraceStep(
                step_id="TR-DPW-S001",
                thread_id="TR-DPW",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "us"},
                introduced_clause_ids=["C002", "C003"],
            ),
            TraceStep(
                step_id="TR-DPW-S002",
                thread_id="TR-DPW",
                step_idx=1,
                kind="commit",
                message="commit",
                state={"region": "us"},
                introduced_clause_ids=["C001"],
            ),
            TraceStep(
                step_id="TR-DPW-S003",
                thread_id="TR-DPW",
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
                thread_id="TR-DPW",
                step_idx=1,
                node_type="DECISION",
                text="Decision checkpoint: allow in region us.",
            ),
            "C002": TraceWorldClause(
                clause_id="C002",
                thread_id="TR-DPW",
                step_idx=0,
                node_type="EVIDENCE",
                text="Evidence snapshot confirms region us request constraints.",
            ),
            "C003": TraceWorldClause(
                clause_id="C003",
                thread_id="TR-DPW",
                step_idx=0,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method("goc", [thread], args=_args())
    rec = report["records"][0]
    assert "C002" in rec["goc_depwalk_added_ids"]
    assert rec["goc_depwalk_added_count"] >= 1
    assert "C002" in rec["e3_context_clause_ids"]
    assert rec["goc_rescue_ran"] is False
    assert rec["goc_rescue_reason_short"] == "not_needed"


def test_goc_smart_pack_dedups_and_unfolds_minimum_types() -> None:
    thread = TraceThread(
        thread_id="TR-SMART",
        level=2,
        scenario="mixed",
        initial_state={"region": "eu", "budget": "low"},
        steps=[
            TraceStep(
                step_id="TR-SMART-S001",
                thread_id="TR-SMART",
                step_idx=0,
                kind="explore",
                message="explore",
                state={"region": "eu", "budget": "low"},
                introduced_clause_ids=["O1", "O2", "A1", "U1", "E1"],
            ),
            TraceStep(
                step_id="TR-SMART-S002",
                thread_id="TR-SMART",
                step_idx=1,
                kind="update",
                message="update",
                state={"region": "eu", "budget": "low"},
                introduced_clause_ids=["A2", "UINV", "U2", "E2", "D1"],
            ),
            TraceStep(
                step_id="TR-SMART-S003",
                thread_id="TR-SMART",
                step_idx=2,
                kind="pivot_check",
                message="Given latest region and budget update, decide policy.",
                state={"region": "eu", "budget": "low"},
                pivot_required_ids=["D1"],
                gold=TraceGold(
                    decision="allow",
                    conditions=[],
                    evidence_ids=["D1"],
                    evidence_core_ids=["D1"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "O1": TraceWorldClause(
                clause_id="O1",
                thread_id="TR-SMART",
                step_idx=0,
                node_type="OPTION",
                text="Option alpha legacy branch.",
            ),
            "O2": TraceWorldClause(
                clause_id="O2",
                thread_id="TR-SMART",
                step_idx=0,
                node_type="OPTION",
                text="Option beta controlling branch.",
            ),
            "A1": TraceWorldClause(
                clause_id="A1",
                thread_id="TR-SMART",
                step_idx=0,
                node_type="ASSUMPTION",
                text="Assume region eu baseline.",
                state_key="region",
                state_value="eu",
            ),
            "U1": TraceWorldClause(
                clause_id="U1",
                thread_id="TR-SMART",
                step_idx=0,
                node_type="UPDATE",
                text="Update: region changed to us.",
                state_key="region",
                state_value="us",
            ),
            "E1": TraceWorldClause(
                clause_id="E1",
                thread_id="TR-SMART",
                step_idx=0,
                node_type="EVIDENCE",
                text="Evidence snapshot old budget policy.",
            ),
            "A2": TraceWorldClause(
                clause_id="A2",
                thread_id="TR-SMART",
                step_idx=1,
                node_type="ASSUMPTION",
                text="Assume region eu baseline refreshed.",
                state_key="region",
                state_value="eu",
            ),
            "UINV": TraceWorldClause(
                clause_id="UINV",
                thread_id="TR-SMART",
                step_idx=1,
                node_type="UPDATE",
                text="Earlier note is not controlling under current state.",
                state_key="region",
                state_value="us",
                metadata={"invalidates": ["O1"]},
            ),
            "U2": TraceWorldClause(
                clause_id="U2",
                thread_id="TR-SMART",
                step_idx=1,
                node_type="UPDATE",
                text="Update: region changed to eu.",
                state_key="region",
                state_value="eu",
            ),
            "E2": TraceWorldClause(
                clause_id="E2",
                thread_id="TR-SMART",
                step_idx=1,
                node_type="EVIDENCE",
                text="Evidence snapshot current budget policy.",
            ),
            "D1": TraceWorldClause(
                clause_id="D1",
                thread_id="TR-SMART",
                step_idx=1,
                node_type="DECISION",
                text="Final decision follows option beta.",
                depends_on=["O2"],
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method(
        "goc",
        [thread],
        args=_args(
            goc_depwalk_enable=False,
            goc_applicability_seed_enable=True,
            goc_applicability_seed_topk=20,
            goc_dependency_closure_enable=False,
            goc_smart_context_enable=True,
            goc_smart_cap_option=0,
            goc_smart_cap_assumption=1,
            goc_smart_cap_update=1,
            goc_smart_cap_exception=2,
            goc_smart_cap_evidence=0,
        ),
    )
    rec = report["records"][0]
    context_ids = list(rec["e3_context_clause_ids"])

    assert rec["goc_smart_enable"] is True
    assert "O1" not in context_ids
    assert "O2" in context_ids
    assert "A1" not in context_ids
    assert "A2" in context_ids
    assert any(cid.startswith("E") for cid in context_ids)
    assert len(rec["goc_smart_dropped_ids"]) > 0
    assert len(rec["goc_smart_dropped_reasons"]) == len(rec["goc_smart_dropped_ids"])
    assert isinstance(rec["goc_smart_type_counts_before"], dict)
    assert isinstance(rec["goc_smart_type_counts_after"], dict)
    assert rec["goc_smart_injected_ids"]
