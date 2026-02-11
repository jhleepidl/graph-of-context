from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def cap_list(xs: Any, max_n: int = 200) -> Tuple[List[Any], int, bool]:
    if isinstance(xs, list):
        vals = list(xs)
    elif isinstance(xs, (tuple, set)):
        vals = list(xs)
    else:
        vals = []
    cap = max(1, int(max_n))
    full_count = len(vals)
    capped = vals[:cap]
    return capped, full_count, full_count > cap


def build_event_trace_line(
    record: Dict[str, Any],
    base_fields: Dict[str, Any],
    *,
    max_list_items: int = 200,
) -> Dict[str, Any]:
    line: Dict[str, Any] = dict(base_fields)
    rec = dict(record or {})
    truncation: Dict[str, Any] = {}

    def _cap(name: str, xs: Any) -> List[Any]:
        capped, full_count, truncated = cap_list(xs, max_n=max_list_items)
        truncation[f"{name}_full_count"] = int(full_count)
        truncation[f"{name}_truncated"] = bool(truncated)
        return capped

    # Common metadata (top-level)
    line["variant_name"] = line.get("variant_name", rec.get("variant_name"))
    line["method_name"] = line.get("method_name", rec.get("method_name"))
    line["traceops_eval_mode"] = rec.get("traceops_eval_mode")
    line["traceops_llm_eval_scope"] = rec.get("traceops_llm_eval_scope")
    line["sampled_step"] = rec.get("sampled_step")

    # Correctness/evidence diagnostics (top-level)
    for key in [
        "decision_correct_exact",
        "decision_correct_family",
        "conditions_correct_exact",
        "conditions_correct_subset",
        "conditions_correct_exact_equiv",
        "conditions_correct_subset_equiv",
        "evidence_core_covered_strict",
        "evidence_core_covered_equiv",
        "evidence_core_id_mismatch_but_equiv_present",
    ]:
        line[key] = rec.get(key)
    line["evidence_core_missing_ids_strict"] = _cap(
        "evidence_core_missing_ids_strict",
        rec.get("evidence_core_missing_ids_strict"),
    )
    line["evidence_core_missing_equiv_keys"] = _cap(
        "evidence_core_missing_equiv_keys",
        rec.get("evidence_core_missing_equiv_keys"),
    )
    for key in [
        "indirection_overlap_gold",
        "best_gold_sim",
        "best_distractor_sim",
        "trap_gap",
        "trap_present",
        "core_size",
    ]:
        line[key] = rec.get(key)
    line["trap_distractor_ids"] = _cap("trap_distractor_ids", rec.get("trap_distractor_ids"))

    raw_context_ids = line.get("context_clause_ids")
    context_clause_ids = _cap("context_clause_ids", raw_context_ids)
    line["context_clause_ids"] = list(context_clause_ids)
    context_set = {str(cid) for cid in context_clause_ids if str(cid).strip()}
    context_clauses, context_clauses_full_count, context_clauses_truncated = cap_list(
        line.get("context_clauses"), max_n=80
    )
    truncation["context_clauses_full_count"] = int(context_clauses_full_count)
    truncation["context_clauses_truncated"] = bool(context_clauses_truncated)
    line["context_clauses"] = list(context_clauses)

    num_exception: int | None = 0
    num_update: int | None = 0
    if context_clauses:
        for clause in context_clauses:
            if not isinstance(clause, dict):
                continue
            ctype = str(clause.get("type", clause.get("node_type", "")) or "")
            if ctype == "EXCEPTION":
                num_exception += 1
            if ctype == "UPDATE":
                num_update += 1
    else:
        num_exception = None
        num_update = None

    avoid_target_ids = rec.get("goc_avoid_target_clause_ids")
    if avoid_target_ids is None:
        avoid_target_ids = rec.get("avoid_target_clause_ids")
    avoid_target_ids_capped = _cap("avoid_target_clause_ids", avoid_target_ids)
    _, avoid_target_count_full, _ = cap_list(avoid_target_ids, max_n=max_list_items)
    avoid_set = {str(cid) for cid in avoid_target_ids_capped if str(cid).strip()}
    avoid_injected = bool(avoid_set & context_set)

    diag: Dict[str, Any] = {
        "context_stats": {
            "context_size": int(len(context_clause_ids)),
            "num_exception_in_context": num_exception,
            "num_update_in_context": num_update,
        },
        "avoid": {
            "avoid_target_clause_ids": list(avoid_target_ids_capped),
            "avoid_target_count": int(
                rec.get("goc_avoid_target_count", avoid_target_count_full)
                if isinstance(rec.get("goc_avoid_target_count"), (int, float))
                else avoid_target_count_full
            ),
            "avoid_injected": bool(avoid_injected),
        },
    }

    goc: Dict[str, Any] = {}
    has_goc = any(str(k).startswith("goc_") for k in rec.keys())
    if has_goc:
        goc["goc_depwalk_added_ids"] = _cap("goc_depwalk_added_ids", rec.get("goc_depwalk_added_ids"))
        goc["goc_depwalk_added_count"] = rec.get("goc_depwalk_added_count")
        goc["goc_depwalk_added_applicable_rate"] = rec.get("goc_depwalk_added_applicable_rate")

        goc["goc_rescue_ran"] = rec.get("goc_rescue_ran")
        goc["goc_rescue_reason_short"] = rec.get("goc_rescue_reason_short")
        goc["goc_exception_rescue_ids"] = _cap(
            "goc_exception_rescue_ids", rec.get("goc_exception_rescue_ids")
        )
        goc["goc_exception_rescue_reason"] = _cap(
            "goc_exception_rescue_reason", rec.get("goc_exception_rescue_reason")
        )
        goc["goc_exception_rescue_count"] = rec.get("goc_exception_rescue_count")
        goc["goc_update_history_rescue_ids"] = _cap(
            "goc_update_history_rescue_ids",
            rec.get("goc_update_history_rescue_ids"),
        )
        goc["goc_update_history_rescue_count"] = rec.get("goc_update_history_rescue_count")

        goc["goc_exception_injected_ids"] = _cap(
            "goc_exception_injected_ids",
            rec.get("goc_exception_injected_ids"),
        )
        injected_count = rec.get("goc_exception_injected_count")
        applicable_count = rec.get("goc_exception_applicable_count")
        goc["goc_exception_injected_count"] = injected_count
        goc["goc_exception_applicable_count"] = applicable_count
        if isinstance(injected_count, (int, float)) and isinstance(applicable_count, (int, float)):
            denom = max(1.0, float(injected_count))
            goc["goc_exception_injected_applicable_rate"] = float(applicable_count) / denom
        else:
            goc["goc_exception_injected_applicable_rate"] = None

        for opt_key in [
            "goc_seed_ids",
            "goc_anchor_ids",
            "goc_closure_added_ids",
            "goc_applicability_seed_ids",
            "goc_dependency_closure_added_ids",
        ]:
            if opt_key in rec:
                goc[opt_key] = _cap(opt_key, rec.get(opt_key))

    if goc:
        diag["goc"] = goc
    scenario_metrics = {
        "indirection_overlap_gold": rec.get("indirection_overlap_gold"),
        "best_gold_sim": rec.get("best_gold_sim"),
        "best_distractor_sim": rec.get("best_distractor_sim"),
        "trap_gap": rec.get("trap_gap"),
        "trap_present": rec.get("trap_present"),
        "core_size": rec.get("core_size"),
        "trap_distractor_ids": _cap("scenario_trap_distractor_ids", rec.get("trap_distractor_ids")),
    }
    if any(v is not None for k, v in scenario_metrics.items() if k != "trap_distractor_ids"):
        diag["scenario_metrics"] = scenario_metrics
    diag["truncation"] = dict(truncation)
    line["diag"] = diag
    return line


__all__ = ["cap_list", "build_event_trace_line"]
