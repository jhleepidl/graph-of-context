from __future__ import annotations

from policyops.traceops_v0.event_traces import build_event_trace_line


def test_event_trace_line_includes_diag_and_truncation() -> None:
    context_ids = [f"C{i:04d}" for i in range(260)]
    depwalk_ids = [f"D{i:04d}" for i in range(255)]
    avoid_ids = [f"C{i:04d}" for i in range(10, 30)]
    missing_ids = [f"M{i:04d}" for i in range(245)]
    missing_keys = [f"k{i:04d}" for i in range(240)]
    hidden_ids = [f"H{i:04d}" for i in range(222)]
    context_clauses = [
        {
            "id": cid,
            "type": "EXCEPTION" if idx % 2 == 0 else "UPDATE",
            "text": f"clause text {cid}",
        }
        for idx, cid in enumerate(context_ids[:120])
    ]

    base = {
        "task_id": "T1",
        "thread_id": "TR1",
        "step_idx": 3,
        "variant_name": "goc_phase16_depwalk",
        "method_name": "goc",
        "context_clause_ids": context_ids,
        "context_clauses": context_clauses,
        "gold": {"decision": "allow", "conditions": [], "evidence_core_ids": []},
        "pred": {"decision": "allow", "conditions": [], "evidence": []},
        "tokens": {"prompt_est": 123},
    }
    record = {
        "traceops_eval_mode": "llm",
        "traceops_llm_eval_scope": "pivots",
        "sampled_step": False,
        "gold_needs_more_info": True,
        "pred_needs_more_info": True,
        "commit_when_gold_unknown": False,
        "gate_forced_needs_more_info": True,
        "gate_forced_reason": "allow_deny_update_is_noop",
        "allow_deny_commit_without_valid_update": True,
        "noop_update_in_context": True,
        "noop_update_in_context_count": 3,
        "evidence_core_missing_ids_strict": missing_ids,
        "evidence_core_missing_equiv_keys": missing_keys,
        "goc_avoid_target_clause_ids": avoid_ids,
        "goc_depwalk_added_ids": depwalk_ids,
        "goc_depwalk_added_count": len(depwalk_ids),
        "goc_exception_injected_ids": ["C0002", "C0004"],
        "goc_exception_injected_count": 2,
        "goc_exception_applicable_count": 1,
        "goc_smart_enable": True,
        "goc_smart_dropped_ids": [f"X{i:04d}" for i in range(230)],
        "goc_smart_dropped_reasons": ["drop_option_unreferenced"] * 230,
        "goc_smart_injected_ids": [f"Y{i:04d}" for i in range(225)],
        "goc_smart_type_counts_before": {"OPTION": 10, "UPDATE": 8},
        "goc_smart_type_counts_after": {"OPTION": 1, "UPDATE": 3},
        "goc_update_delay": 6,
        "goc_update_counts_by_age": {"stable_update_count": 2, "recent_update_count": 1},
        "goc_noop_update_dropped_count": 2,
        "goc_recent_update_dropped_count": 4,
        "goc_stable_update_kept_count": 2,
        "goc_update_keys_required": ["region", "budget", "retention_tier"],
        "goc_update_keys_injected": ["retention_tier"],
        "goc_update_keys_missing_after_smart": ["budget"],
        "core_necessity_flip_count": 3,
        "core_necessity_all_required": True,
        "core_necessity_failed": False,
        "trap_decision_label": "deny",
        "trap_decision_flip": True,
        "hidden_core_ids": hidden_ids,
        "hidden_core_parent_ids": ["P0001", "P0002"],
        "policy_anchor_id": "A0001",
        "policy_anchor_in_gold_core": True,
        "policy_anchor_in_context": True,
    }

    line = build_event_trace_line(record, base, max_list_items=200)

    assert "diag" in line
    assert "truncation" in line["diag"]
    assert len(line["context_clause_ids"]) == 200
    assert line["diag"]["truncation"]["context_clause_ids_full_count"] == 260
    assert line["diag"]["truncation"]["context_clause_ids_truncated"] is True

    assert len(line["evidence_core_missing_ids_strict"]) == 200
    assert line["diag"]["truncation"]["evidence_core_missing_ids_strict_full_count"] == 245
    assert line["diag"]["truncation"]["evidence_core_missing_ids_strict_truncated"] is True

    goc_diag = line["diag"]["goc"]
    assert len(goc_diag["goc_depwalk_added_ids"]) == 200
    assert line["diag"]["truncation"]["goc_depwalk_added_ids_full_count"] == 255
    assert line["diag"]["truncation"]["goc_depwalk_added_ids_truncated"] is True
    assert goc_diag["goc_smart_enable"] is True
    assert len(goc_diag["goc_smart_dropped_ids"]) == 200
    assert len(goc_diag["goc_smart_dropped_reasons"]) == 200
    assert len(goc_diag["goc_smart_injected_ids"]) == 200
    assert goc_diag["goc_smart_type_counts_before"]["OPTION"] == 10
    assert goc_diag["goc_smart_type_counts_after"]["OPTION"] == 1
    assert goc_diag["goc_update_delay"] == 6
    assert goc_diag["goc_update_counts_by_age"]["stable_update_count"] == 2
    assert goc_diag["goc_noop_update_dropped_count"] == 2
    assert goc_diag["goc_recent_update_dropped_count"] == 4
    assert goc_diag["goc_stable_update_kept_count"] == 2
    assert "retention_tier" in goc_diag["goc_update_keys_required"]
    assert goc_diag["goc_update_keys_injected"] == ["retention_tier"]
    assert goc_diag["goc_update_keys_missing_after_smart"] == ["budget"]

    avoid_diag = line["diag"]["avoid"]
    assert avoid_diag["avoid_target_count"] == 20
    assert avoid_diag["avoid_injected"] is True
    assert line["gold_needs_more_info"] is True
    assert line["pred_needs_more_info"] is True
    assert line["commit_when_gold_unknown"] is False
    assert line["gate_forced_needs_more_info"] is True
    assert line["gate_forced_reason"] == "allow_deny_update_is_noop"
    assert line["allow_deny_commit_without_valid_update"] is True
    assert line["noop_update_in_context"] is True
    assert line["noop_update_in_context_count"] == 3

    assert line["core_necessity_all_required"] is True
    assert line["core_necessity_flip_count"] == 3
    assert line["trap_decision_flip"] is True
    assert line["trap_decision_label"] == "deny"
    assert len(line["hidden_core_ids"]) == 200
    assert line["policy_anchor_id"] == "A0001"
    assert line["policy_anchor_in_gold_core"] is True
    assert line["policy_anchor_in_context"] is True
    assert line["diag"]["truncation"]["hidden_core_ids_full_count"] == 222
    assert line["diag"]["truncation"]["hidden_core_ids_truncated"] is True

    scenario = line["diag"]["scenario_metrics"]
    assert scenario["core_necessity_all_required"] is True
    assert scenario["trap_decision_flip"] is True
    assert len(scenario["hidden_core_ids"]) == 200
    assert scenario["policy_anchor_id"] == "A0001"
    assert scenario["policy_anchor_in_gold_core"] is True
    assert scenario["policy_anchor_in_context"] is True
