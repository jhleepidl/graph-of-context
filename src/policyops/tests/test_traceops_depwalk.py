from __future__ import annotations

from types import SimpleNamespace

from policyops.traceops_v0.evaluator import _goc_smart_pack_context_ids, evaluate_traceops_method
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
        "traceops_delay_to_relevance": 0,
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


def test_goc_smart_update_delay_backfill_and_key_coverage() -> None:
    thread = TraceThread(
        thread_id="TR-SMART-U",
        level=2,
        scenario="mixed",
        initial_state={"retention_tier": "short"},
        steps=[
            TraceStep(
                step_id="TR-SMART-U-S001",
                thread_id="TR-SMART-U",
                step_idx=0,
                kind="explore",
                message="initial",
                state={"retention_tier": "short"},
                introduced_clause_ids=["UOLD", "E1"],
            ),
            TraceStep(
                step_id="TR-SMART-U-S002",
                thread_id="TR-SMART-U",
                step_idx=1,
                kind="update",
                message="early update",
                state={"retention_tier": "short"},
                introduced_clause_ids=["USTABLE"],
            ),
            TraceStep(
                step_id="TR-SMART-U-S003",
                thread_id="TR-SMART-U",
                step_idx=2,
                kind="update",
                message="recent trap update",
                state={"retention_tier": "standard"},
                introduced_clause_ids=["UTRAP"],
            ),
            TraceStep(
                step_id="TR-SMART-U-S004",
                thread_id="TR-SMART-U",
                step_idx=3,
                kind="commit",
                message="decision",
                state={"retention_tier": "standard"},
                introduced_clause_ids=["D1"],
            ),
            TraceStep(
                step_id="TR-SMART-U-S008",
                thread_id="TR-SMART-U",
                step_idx=7,
                kind="pivot_check",
                message="Given handle binding, decide retention policy now.",
                state={"retention_tier": "standard"},
                pivot_required_ids=["D1"],
                gold=TraceGold(
                    decision="deny",
                    conditions=[],
                    evidence_ids=["D1", "USTABLE"],
                    evidence_core_ids=["D1", "USTABLE"],
                    evidence_meta_ids=[],
                ),
            ),
        ],
        clauses={
            "UOLD": TraceWorldClause(
                clause_id="UOLD",
                thread_id="TR-SMART-U",
                step_idx=0,
                node_type="UPDATE",
                text="Update: retention_tier remains short for controlling timeline.",
                state_key="retention_tier",
                state_value="short",
            ),
            "E1": TraceWorldClause(
                clause_id="E1",
                thread_id="TR-SMART-U",
                step_idx=0,
                node_type="EVIDENCE",
                text="Evidence baseline for retention tier decisions.",
            ),
            "USTABLE": TraceWorldClause(
                clause_id="USTABLE",
                thread_id="TR-SMART-U",
                step_idx=1,
                node_type="UPDATE",
                text="Update: retention_tier remains short; earlier notes are not controlling.",
                state_key="retention_tier",
                state_value="short",
            ),
            "UTRAP": TraceWorldClause(
                clause_id="UTRAP",
                thread_id="TR-SMART-U",
                step_idx=2,
                node_type="UPDATE",
                text="Historical trap note: retention_tier changed to standard immediately.",
                state_key="retention_tier",
                state_value="standard",
                metadata={"trap": True, "trap_kind": "stale"},
            ),
            "D1": TraceWorldClause(
                clause_id="D1",
                thread_id="TR-SMART-U",
                step_idx=3,
                node_type="DECISION",
                text="Decision depends on retention timeline bindings.",
                metadata={"binding_key": "retention_tier"},
            ),
        },
        meta={},
    )

    report = evaluate_traceops_method(
        "goc",
        [thread],
        args=_args(
            goc_depwalk_enable=False,
            goc_applicability_seed_enable=False,
            goc_dependency_closure_enable=False,
            goc_smart_context_enable=True,
            goc_smart_cap_option=0,
            goc_smart_cap_assumption=2,
            goc_smart_cap_update=3,
            goc_smart_cap_exception=2,
            goc_smart_cap_evidence=2,
            traceops_delay_to_relevance=6,
        ),
    )
    rec = report["records"][0]
    context_ids = set(rec["e3_context_clause_ids"])
    age_diag = rec.get("goc_update_counts_by_age") or {}

    assert "UTRAP" not in context_ids
    assert "USTABLE" in context_ids
    assert rec.get("goc_update_delay") == 6
    assert "retention_tier" in list(rec.get("goc_update_keys_required") or [])
    assert "retention_tier" in list(rec.get("goc_update_keys_injected") or [])
    assert float(age_diag.get("stable_update_count", 0)) >= 1.0


def test_goc_smart_pack_reads_delay_from_dict_args() -> None:
    step = TraceStep(
        step_id="TR-DICT-S010",
        thread_id="TR-DICT",
        step_idx=10,
        kind="pivot_check",
        message="Decide using update timeline",
        state={"retention_tier": "standard"},
        metadata={"delay_to_relevance": 6},
    )
    clauses = {
        "U_STABLE": TraceWorldClause(
            clause_id="U_STABLE",
            thread_id="TR-DICT",
            step_idx=2,
            node_type="UPDATE",
            text="Update: retention_tier changed to short.",
            state_key="retention_tier",
            state_value="short",
        ),
        "U_RECENT": TraceWorldClause(
            clause_id="U_RECENT",
            thread_id="TR-DICT",
            step_idx=9,
            node_type="UPDATE",
            text="Update: retention_tier changed to standard.",
            state_key="retention_tier",
            state_value="standard",
        ),
        "D1": TraceWorldClause(
            clause_id="D1",
            thread_id="TR-DICT",
            step_idx=8,
            node_type="DECISION",
            text="Decision depends on retention timeline.",
            metadata={"binding_key": "retention_tier"},
        ),
    }
    packed_ids, debug = _goc_smart_pack_context_ids(
        context_ids=["U_STABLE", "U_RECENT", "D1"],
        ordered_history=["U_STABLE", "U_RECENT", "D1"],
        candidates=["U_STABLE", "U_RECENT", "D1"],
        step=step,
        clauses=clauses,
        avoid_set=set(),
        args={
            "traceops_delay_to_relevance": 6,
            "goc_smart_cap_option": 0,
            "goc_smart_cap_assumption": 2,
            "goc_smart_cap_update": 4,
            "goc_smart_cap_exception": 2,
            "goc_smart_cap_evidence": 2,
        },
    )
    assert "U_STABLE" in set(packed_ids)
    assert "U_RECENT" in set(packed_ids)
    assert int(debug.get("goc_update_delay", 0)) == 6


def test_goc_smart_pack_uses_step_delay_metadata_when_arg_missing() -> None:
    step = TraceStep(
        step_id="TR-DICT-S011",
        thread_id="TR-DICT",
        step_idx=10,
        kind="pivot_check",
        message="Decide using update timeline",
        state={"retention_tier": "standard"},
        metadata={"delay_to_relevance": 6},
    )
    clauses = {
        "U_STABLE": TraceWorldClause(
            clause_id="U_STABLE",
            thread_id="TR-DICT",
            step_idx=2,
            node_type="UPDATE",
            text="Update: retention_tier changed to short.",
            state_key="retention_tier",
            state_value="short",
        ),
        "U_RECENT": TraceWorldClause(
            clause_id="U_RECENT",
            thread_id="TR-DICT",
            step_idx=9,
            node_type="UPDATE",
            text="Update: retention_tier changed to standard.",
            state_key="retention_tier",
            state_value="standard",
        ),
    }
    _, debug = _goc_smart_pack_context_ids(
        context_ids=["U_STABLE", "U_RECENT"],
        ordered_history=["U_STABLE", "U_RECENT"],
        candidates=["U_STABLE", "U_RECENT"],
        step=step,
        clauses=clauses,
        avoid_set=set(),
        args={
            "goc_smart_cap_option": 0,
            "goc_smart_cap_assumption": 2,
            "goc_smart_cap_update": 4,
            "goc_smart_cap_exception": 2,
            "goc_smart_cap_evidence": 2,
        },
    )
    counts = debug.get("goc_update_counts_by_age") or {}
    assert int(debug.get("goc_update_delay", 0)) == 6
    assert int(counts.get("stable_update_count", 0)) >= 1
    assert int(counts.get("recent_update_count", 0)) >= 1


def test_goc_smart_pack_protects_policy_anchor_and_codebook_from_quota_drop() -> None:
    step = TraceStep(
        step_id="TR-PROT-S010",
        thread_id="TR-PROT",
        step_idx=10,
        kind="pivot_check",
        message="final decision",
        state={"region": "us"},
        metadata={"delay_to_relevance": 6},
    )
    clauses = {
        "PA": TraceWorldClause(
            clause_id="PA",
            thread_id="TR-PROT",
            step_idx=1,
            node_type="EVIDENCE",
            text="Anchor ledger HZ-13 lane assignment.",
            metadata={"policy_anchor": True},
        ),
        "PC": TraceWorldClause(
            clause_id="PC",
            thread_id="TR-PROT",
            step_idx=1,
            node_type="EVIDENCE",
            text="Codebook HZ-13 lane-alpha => ALLOW; lane-beta => DENY.",
            metadata={"policy_codebook": True},
        ),
        "E1": TraceWorldClause(
            clause_id="E1",
            thread_id="TR-PROT",
            step_idx=9,
            node_type="EVIDENCE",
            text="Recent but non-core evidence.",
        ),
        "D1": TraceWorldClause(
            clause_id="D1",
            thread_id="TR-PROT",
            step_idx=9,
            node_type="DECISION",
            text="Decision depends on policy chain.",
            depends_on=["PA", "PC"],
        ),
        "U1": TraceWorldClause(
            clause_id="U1",
            thread_id="TR-PROT",
            step_idx=9,
            node_type="UPDATE",
            text="Update: region changed to us.",
            state_key="region",
            state_value="us",
        ),
    }

    packed_ids, debug = _goc_smart_pack_context_ids(
        context_ids=["PA", "PC", "E1", "D1", "U1"],
        ordered_history=["PA", "PC", "E1", "D1", "U1"],
        candidates=["PA", "PC", "E1", "D1", "U1"],
        step=step,
        clauses=clauses,
        avoid_set=set(),
        args={
            "goc_smart_cap_option": 0,
            "goc_smart_cap_assumption": 2,
            "goc_smart_cap_update": 4,
            "goc_smart_cap_exception": 2,
            "goc_smart_cap_evidence": 1,
        },
    )
    packed = set(packed_ids)
    assert "PA" in packed
    assert "PC" in packed
    protected_ids = set(debug.get("goc_smart_protected_ids") or [])
    assert {"PA", "PC"}.issubset(protected_ids)
