from __future__ import annotations

from policyops.traceops_v0.event_traces import build_event_trace_line


def test_event_trace_line_includes_diag_and_truncation() -> None:
    context_ids = [f"C{i:04d}" for i in range(260)]
    depwalk_ids = [f"D{i:04d}" for i in range(255)]
    avoid_ids = [f"C{i:04d}" for i in range(10, 30)]
    missing_ids = [f"M{i:04d}" for i in range(245)]
    missing_keys = [f"k{i:04d}" for i in range(240)]
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
        "evidence_core_missing_ids_strict": missing_ids,
        "evidence_core_missing_equiv_keys": missing_keys,
        "goc_avoid_target_clause_ids": avoid_ids,
        "goc_depwalk_added_ids": depwalk_ids,
        "goc_depwalk_added_count": len(depwalk_ids),
        "goc_exception_injected_ids": ["C0002", "C0004"],
        "goc_exception_injected_count": 2,
        "goc_exception_applicable_count": 1,
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

    avoid_diag = line["diag"]["avoid"]
    assert avoid_diag["avoid_target_count"] == 20
    assert avoid_diag["avoid_injected"] is True
