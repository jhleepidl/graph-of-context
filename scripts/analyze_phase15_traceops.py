#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from policyops.traceops_v0.failure_taxonomy import compare_pair
from policyops.traceops_v0.event_traces import build_event_trace_line
from policyops.traceops_v0.schema import normalize_decision


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_threads_by_scenario(phase15_root: Path, scenario: str) -> Dict[str, Any]:
    threads_path = phase15_root / "data" / scenario / "data" / "traceops" / "threads.jsonl"
    if not threads_path.exists():
        return {}
    out: Dict[str, Any] = {}
    with threads_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            tid = str(obj.get("thread_id", "") or "")
            if not tid:
                continue
            out[tid] = obj
    return out


def _resolve_report_path(phase15_root: Path, run: Dict[str, Any]) -> Path | None:
    scenario = str(run.get("traceops_scenario") or "").strip()
    variant = str(run.get("variant") or "").strip()
    if scenario and variant:
        quick_report = (
            phase15_root
            / "quick_access"
            / "reports"
            / f"{scenario}__{variant}.json"
        )
        if quick_report.exists():
            return quick_report
    report_rel = str(run.get("report_json", "") or "").strip()
    if report_rel:
        fallback = phase15_root / Path(report_rel)
        if fallback.exists():
            return fallback
    return None


def _gold_decision_family(gold_decision: str) -> str:
    return normalize_decision(gold_decision)


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _mean(values: List[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _mean_or_zero(values: List[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def _tokset(text: Any) -> set[str]:
    import re

    return {tok for tok in re.findall(r"[a-z0-9_]+", str(text or "").lower()) if tok}


def _jaccard(a: Any, b: Any) -> float:
    ta = _tokset(a)
    tb = _tokset(b)
    if not ta or not tb:
        return 0.0
    denom = len(ta | tb)
    if denom <= 0:
        return 0.0
    return float(len(ta & tb)) / float(denom)


def _compute_scenario_metrics(
    *,
    pivot_question: str,
    gold_core_ids: List[str],
    clauses_map: Dict[str, Any],
    all_prior_clause_ids: List[str],
) -> Dict[str, Any]:
    core_ids = [str(cid) for cid in (gold_core_ids or []) if str(cid)]
    core_text = " ".join(
        str((clauses_map.get(cid) or {}).get("text", "") or "")
        for cid in core_ids
        if isinstance(clauses_map.get(cid), dict)
    )
    indirection_overlap_gold = _jaccard(pivot_question, core_text)
    best_gold_sim = 0.0
    for cid in core_ids:
        clause = clauses_map.get(cid)
        if not isinstance(clause, dict):
            continue
        best_gold_sim = max(best_gold_sim, _jaccard(pivot_question, clause.get("text", "")))

    distractor_ids: List[str] = []
    best_distractor_sim = 0.0
    for cid in all_prior_clause_ids:
        clause = clauses_map.get(cid)
        if not isinstance(clause, dict):
            continue
        tags = {str(tag).strip().lower() for tag in (clause.get("tags") or [])}
        meta = clause.get("metadata") if isinstance(clause.get("metadata"), dict) else {}
        if ("distractor" not in tags and "trap" not in tags and not bool(meta.get("trap", False))):
            continue
        distractor_ids.append(str(cid))
        best_distractor_sim = max(best_distractor_sim, _jaccard(pivot_question, clause.get("text", "")))
    trap_gap = float(best_distractor_sim - best_gold_sim)
    trap_present = bool(len(distractor_ids) > 0 and trap_gap > 0.0)
    return {
        "indirection_overlap_gold": float(indirection_overlap_gold),
        "best_gold_sim": float(best_gold_sim),
        "best_distractor_sim": float(best_distractor_sim),
        "trap_gap": float(trap_gap),
        "trap_present": bool(trap_present),
        "core_size": int(len(core_ids)),
        "trap_distractor_ids": list(dict.fromkeys(distractor_ids)),
    }


def _build_failure_taxonomy(
    phase15_root: Path,
    event_trace_by_file: Dict[str, List[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, List[str]]:
    taxonomy_rows: List[Dict[str, Any]] = []
    case_lines: List[str] = []
    case_lines.append("# TraceOps Failure Taxonomy (pairwise)")
    case_lines.append("")

    by_triplet: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for fname, events in event_trace_by_file.items():
        parts = str(fname).split("__")
        if len(parts) < 4:
            continue
        scenario, variant, method = parts[0], parts[1], parts[2]
        by_triplet[(scenario, variant, method)] = list(events)

    scenarios = sorted({k[0] for k in by_triplet.keys()})
    pair_specs = [
        (("baseline", "full"), ("baseline", "similarity_only")),
        (("baseline", "full"), ("baseline", "agent_fold")),
        (("baseline", "full"), ("goc_phase13_style", "goc")),
        (("baseline", "full"), ("goc_phase16_depwalk", "goc")),
        (("baseline", "full"), ("goc_phase17_depwalk", "goc")),
        (("goc_phase16_depwalk", "goc"), ("baseline", "similarity_only")),
        (("goc_phase17_depwalk", "goc"), ("baseline", "similarity_only")),
        (("goc_phase13_style", "goc"), ("baseline", "similarity_only")),
        (("goc_phase13_style", "goc"), ("baseline", "agent_fold")),
        (("goc_phase17_depwalk", "goc"), ("baseline", "agent_fold")),
        (("goc_phase13_style", "goc"), ("goc_phase17_depwalk", "goc")),
    ]

    for scenario in scenarios:
        for left, right in pair_specs:
            l_variant, l_method = left
            r_variant, r_method = right
            left_events = by_triplet.get((scenario, l_variant, l_method))
            right_events = by_triplet.get((scenario, r_variant, r_method))
            if not left_events or not right_events:
                continue
            variant_a = f"{l_variant}__{l_method}"
            variant_b = f"{r_variant}__{r_method}"
            rows, examples = compare_pair(
                left_events,
                right_events,
                scenario=scenario,
                variant_a=variant_a,
                variant_b=variant_b,
            )
            case_lines.append(f"## {scenario}: {variant_a} vs {variant_b}")
            case_lines.append("")
            if not rows:
                taxonomy_rows.append(
                    {
                        "scenario": str(scenario),
                        "variant_a": str(variant_a),
                        "variant_b": str(variant_b),
                        "direction": "no_directional_delta",
                        "category": "no_failure_delta",
                        "count": 0,
                        "rate_over_direction": 0.0,
                        "example_keys": "",
                    }
                )
                case_lines.append("- no directional correctness delta found for this pair.")
                case_lines.append("")
                continue

            taxonomy_rows.extend(rows)
            for direction in ["A_correct_B_wrong", "B_correct_A_wrong"]:
                buckets = examples.get(direction, {})
                if not buckets:
                    continue
                case_lines.append(f"### {direction}")
                for category in sorted(buckets.keys()):
                    ex_rows = list(buckets.get(category) or [])[:3]
                    case_lines.append(f"- `{category}` ({len(buckets.get(category) or [])})")
                    for ex in ex_rows:
                        key = str(ex.get("key", "") or "")
                        lgold = ex.get("loser_gold") or {}
                        lpred = ex.get("loser_pred") or {}
                        wpred = ex.get("winner_pred") or {}
                        strict_cov = ex.get("loser_evidence_core_covered_strict")
                        equiv_cov = ex.get("loser_evidence_core_covered_equiv")
                        mismatch = ex.get("loser_evidence_core_id_mismatch_but_equiv_present")
                        ctx = ex.get("loser_context_clauses") or []
                        ctx_snip = []
                        for clause in list(ctx)[:4]:
                            if not isinstance(clause, dict):
                                continue
                            ctype = str(clause.get("type", "") or "")
                            ctext = str(clause.get("text", "") or "").strip()
                            ctx_snip.append(f"[{ctype}] {ctext[:120]}")
                        case_lines.append(f"  - key `{key}`")
                        case_lines.append(
                            f"    gold(decision/cond)={lgold.get('decision')}/{lgold.get('conditions')}"
                        )
                        case_lines.append(
                            f"    pred_winner(decision/cond)={wpred.get('decision')}/{wpred.get('conditions')}"
                        )
                        case_lines.append(
                            f"    pred_loser(decision/cond)={lpred.get('decision')}/{lpred.get('conditions')}"
                        )
                        case_lines.append(
                            "    strict_vs_equiv_core="
                            f"{strict_cov}/{equiv_cov} mismatch_but_equiv={mismatch}"
                        )
                        loser_diag = ex.get("loser_diag") if isinstance(ex.get("loser_diag"), dict) else {}
                        ctx_stats = (
                            loser_diag.get("context_stats")
                            if isinstance(loser_diag.get("context_stats"), dict)
                            else {}
                        )
                        avoid_diag = (
                            loser_diag.get("avoid")
                            if isinstance(loser_diag.get("avoid"), dict)
                            else {}
                        )
                        goc_diag = (
                            loser_diag.get("goc")
                            if isinstance(loser_diag.get("goc"), dict)
                            else {}
                        )
                        scenario_diag = (
                            loser_diag.get("scenario_metrics")
                            if isinstance(loser_diag.get("scenario_metrics"), dict)
                            else {}
                        )
                        def _pick_scenario_value(field: str) -> Any:
                            direct = ex.get(field)
                            if isinstance(direct, (int, float, bool, str)):
                                return direct
                            return scenario_diag.get(field)

                        indirection_overlap = _pick_scenario_value("indirection_overlap_gold")
                        trap_present = _pick_scenario_value("trap_present")
                        trap_gap = _pick_scenario_value("trap_gap")
                        core_size = _pick_scenario_value("core_size")
                        trap_ids = ex.get("trap_distractor_ids")
                        if not isinstance(trap_ids, list):
                            trap_ids = scenario_diag.get("trap_distractor_ids")
                        trap_ids_count = len(trap_ids) if isinstance(trap_ids, list) else None
                        case_lines.append("    diag_summary:")
                        case_lines.append(
                            "      - "
                            f"context_size={ctx_stats.get('context_size')} "
                            f"avoid_target_count={avoid_diag.get('avoid_target_count')} "
                            f"avoid_injected={avoid_diag.get('avoid_injected')}"
                        )
                        case_lines.append(
                            "      - "
                            f"depwalk_added_count={goc_diag.get('goc_depwalk_added_count')} "
                            f"exception_rescue_count={goc_diag.get('goc_exception_rescue_count')} "
                            f"update_history_rescue_count={goc_diag.get('goc_update_history_rescue_count')}"
                        )
                        case_lines.append(
                            "      - "
                            f"core_covered(strict/equiv)="
                            f"{strict_cov}/{equiv_cov} "
                            f"id_mismatch_but_equiv={mismatch}"
                        )
                        case_lines.append(
                            "      - "
                            f"scenario(indirection_overlap_gold={indirection_overlap}, "
                            f"trap_present={trap_present}, trap_gap={trap_gap}, "
                            f"core_size={core_size}, trap_distractor_ids_count={trap_ids_count})"
                        )
                        core_need_all_required = _pick_scenario_value("core_necessity_all_required")
                        core_need_flip_count = _pick_scenario_value("core_necessity_flip_count")
                        core_need_failed = _pick_scenario_value("core_necessity_failed")
                        trap_decision_label = _pick_scenario_value("trap_decision_label")
                        trap_decision_flip = _pick_scenario_value("trap_decision_flip")
                        hidden_core_ids = _pick_scenario_value("hidden_core_ids")
                        hidden_core_count = (
                            len(hidden_core_ids)
                            if isinstance(hidden_core_ids, list)
                            else 0
                        )
                        case_lines.append(
                            "      - "
                            f"phase18(core_need: all_required={core_need_all_required} "
                            f"flip_count={core_need_flip_count} failed={core_need_failed}; "
                            f"trap_flip: trap_decision={trap_decision_label} flip={trap_decision_flip}; "
                            f"hidden_core: n={hidden_core_count})"
                        )
                        if ctx_snip:
                            case_lines.append("    context_snippet:")
                            for line in ctx_snip:
                                case_lines.append(f"      - {line}")
                case_lines.append("")

    return pd.DataFrame(taxonomy_rows), case_lines


def _attach_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["tokens_savings_vs_full"] = np.nan
        out["tokens_savings_vs_full_actual"] = np.nan
        out["accuracy_delta_vs_full"] = np.nan
        return out

    out = df.copy()
    key_cols = ["traceops_level", "traceops_scenario"]
    full_ref = out[out["method"] == "full"].groupby(key_cols, dropna=False).agg(
        full_tokens_pivot_mean=("tokens_pivot_mean", "mean"),
        full_tokens_pivot_mean_actual=("tokens_pivot_mean_actual", "mean"),
        full_pivot_e3_only_accuracy=("pivot_e3_only_accuracy", "mean"),
    )
    out = out.merge(full_ref, left_on=key_cols, right_index=True, how="left")
    out["tokens_savings_vs_full"] = out["tokens_pivot_mean"] / out["full_tokens_pivot_mean"]
    out["tokens_savings_vs_full_actual"] = (
        out["tokens_pivot_mean_actual"] / out["full_tokens_pivot_mean_actual"]
    )
    out["accuracy_delta_vs_full"] = (
        out["pivot_e3_only_accuracy"] - out["full_pivot_e3_only_accuracy"]
    )
    out.drop(
        columns=[
            "full_tokens_pivot_mean",
            "full_tokens_pivot_mean_actual",
            "full_pivot_e3_only_accuracy",
        ],
        inplace=True,
        errors="ignore",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase15_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase15_root = Path(args.phase15_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(phase15_root / "run_manifest.json")
    runs = list(manifest.get("runs") or [])
    quick_event_dir = phase15_root / "quick_access" / "event_traces"
    quick_event_dir.mkdir(parents=True, exist_ok=True)
    scenario_threads_cache: Dict[str, Dict[str, Any]] = {}
    event_trace_by_file: Dict[str, List[Dict[str, Any]]] = {}

    rows: List[Dict[str, Any]] = []
    for run in runs:
        report_path = _resolve_report_path(phase15_root, run)
        if report_path is None or not report_path.exists():
            continue
        report = _load_json(report_path)
        method_reports = report.get("method_reports") or {}
        scenario_params = report.get("scenario_params") or {}
        scenario_name = str(
            run.get("traceops_scenario")
            or scenario_params.get("traceops_scenario")
            or "mixed"
        )
        if scenario_name not in scenario_threads_cache:
            scenario_threads_cache[scenario_name] = _load_threads_by_scenario(phase15_root, scenario_name)
        thread_lookup = scenario_threads_cache.get(scenario_name, {})
        variant_name = str(run.get("variant") or "variant")
        for method, mr in method_reports.items():
            metrics = dict(mr.get("metrics") or {})
            records = list(mr.get("records") or [])
            thread_records = list(mr.get("thread_records") or [])
            tokens_pivot_mean_est = _as_float(metrics.get("tokens_pivot_mean_est"))
            if not np.isfinite(tokens_pivot_mean_est):
                tokens_pivot_mean_est = _mean(
                    [
                        _as_float(rec.get("prompt_tokens_est", rec.get("prompt_tokens")))
                        for rec in records
                    ]
                )
            tokens_total_mean_est = _as_float(metrics.get("tokens_total_mean_est"))
            if not np.isfinite(tokens_total_mean_est):
                tokens_total_mean_est = _mean(
                    [_as_float(rec.get("pivot_token_total_est")) for rec in thread_records]
                )
            tokens_pivot_mean_actual = _as_float(metrics.get("tokens_pivot_mean_actual"))
            if not np.isfinite(tokens_pivot_mean_actual):
                tokens_pivot_mean_actual = _mean(
                    [_as_float(rec.get("total_tokens_actual")) for rec in records]
                )
            tokens_total_mean_actual = _as_float(metrics.get("tokens_total_mean_actual"))
            if not np.isfinite(tokens_total_mean_actual):
                tokens_total_mean_actual = _mean(
                    [_as_float(rec.get("pivot_token_total_actual")) for rec in thread_records]
                )
            tokens_pivot_mean = _as_float(metrics.get("tokens_pivot_mean"))
            if not np.isfinite(tokens_pivot_mean):
                tokens_pivot_mean = (
                    tokens_pivot_mean_actual
                    if np.isfinite(tokens_pivot_mean_actual)
                    else tokens_pivot_mean_est
                )
            tokens_total_mean = _as_float(metrics.get("tokens_total_mean"))
            if not np.isfinite(tokens_total_mean):
                tokens_total_mean = (
                    tokens_total_mean_actual
                    if np.isfinite(tokens_total_mean_actual)
                    else tokens_total_mean_est
                )
            row: Dict[str, Any] = dict(run)
            depwalk_added_vals = [
                _as_float((rec.get("goc_depwalk_added_count", 0.0) if isinstance(rec, dict) else 0.0))
                for rec in records
            ]
            exception_rescue_vals = [
                _as_float((rec.get("goc_exception_rescue_count", 0.0) if isinstance(rec, dict) else 0.0))
                for rec in records
            ]
            update_history_rescue_vals = [
                _as_float((rec.get("goc_update_history_rescue_count", 0.0) if isinstance(rec, dict) else 0.0))
                for rec in records
            ]
            row.update(
                {
                    "method": method,
                    "traceops_level": int(
                        row.get("traceops_level")
                        or scenario_params.get("traceops_level")
                        or 1
                    ),
                    "traceops_scenario": str(
                        row.get("traceops_scenario")
                        or scenario_params.get("traceops_scenario")
                        or "mixed"
                    ),
                    "traceops_delay_to_relevance": int(
                        row.get("traceops_delay_to_relevance")
                        or scenario_params.get("traceops_delay_to_relevance")
                        or 0
                    ),
                    "traceops_distractor_branching": int(
                        row.get("traceops_distractor_branching")
                        or scenario_params.get("traceops_distractor_branching")
                        or 0
                    ),
                    "traceops_contradiction_rate": _as_float(
                        row.get("traceops_contradiction_rate")
                        or scenario_params.get("traceops_contradiction_rate")
                    ),
                    "pivot_decision_accuracy": _as_float(
                        metrics.get("pivot_decision_accuracy")
                    ),
                    "pivot_decision_accuracy_strict_raw": _as_float(
                        metrics.get("pivot_decision_accuracy_strict_raw")
                    ),
                    "pivot_e3_only_accuracy": _as_float(
                        metrics.get("pivot_e3_only_accuracy")
                    ),
                    "strict_pivot_accuracy": _as_float(
                        metrics.get("strict_pivot_accuracy")
                    ),
                    "pivots_available_total": _as_float(
                        metrics.get("pivots_available_total")
                    ),
                    "pivots_evaluated": _as_float(
                        metrics.get("pivots_evaluated")
                    ),
                    "steps_available_total": _as_float(
                        metrics.get("steps_available_total")
                    ),
                    "sampled_step_rate": _as_float(
                        metrics.get("sampled_step_rate")
                    ),
                    "sampled_steps_evaluated": _as_float(
                        metrics.get("sampled_steps_evaluated")
                    ),
                    "tokens_pivot_mean": tokens_pivot_mean,
                    "tokens_total_mean": tokens_total_mean,
                    "tokens_pivot_mean_est": tokens_pivot_mean_est,
                    "tokens_total_mean_est": tokens_total_mean_est,
                    "tokens_pivot_mean_actual": tokens_pivot_mean_actual,
                    "tokens_total_mean_actual": tokens_total_mean_actual,
                    "mean_avoid_targets_per_pivot": _as_float(
                        metrics.get("mean_avoid_targets_per_pivot")
                    ),
                    "avoided_injected_rate": _as_float(
                        metrics.get("avoided_injected_rate")
                    ),
                    "exception_injected_rate": _as_float(
                        metrics.get("exception_injected_rate")
                    ),
                    "mean_exception_injected_count": _as_float(
                        metrics.get("mean_exception_injected_count")
                    ),
                    "mean_goc_stable_update_count": _as_float(
                        metrics.get("mean_goc_stable_update_count")
                    ),
                    "mean_goc_recent_update_count": _as_float(
                        metrics.get("mean_goc_recent_update_count")
                    ),
                    "stable_update_present_rate": _as_float(
                        metrics.get("stable_update_present_rate")
                    ),
                    "mean_depwalk_added_count": _mean_or_zero(depwalk_added_vals),
                    "mean_exception_rescue_count": _mean_or_zero(exception_rescue_vals),
                    "mean_update_history_rescue_count": _mean_or_zero(update_history_rescue_vals),
                    "revive_success_rate": _as_float(metrics.get("revive_success_rate")),
                    "mean_indirection_overlap_gold": _as_float(
                        metrics.get("mean_indirection_overlap_gold")
                    ),
                    "mean_trap_gap": _as_float(metrics.get("mean_trap_gap")),
                    "trap_present_rate": _as_float(metrics.get("trap_present_rate")),
                    "mean_trap_injected_count": _as_float(
                        metrics.get("mean_trap_injected_count")
                    ),
                    "mean_trap_injected_rate": _as_float(
                        metrics.get("mean_trap_injected_rate")
                    ),
                    "trap_injected_any_rate": _as_float(
                        metrics.get("trap_injected_any_rate")
                    ),
                    "mean_decision_checkpoint_trap_injected_count": _as_float(
                        metrics.get("mean_decision_checkpoint_trap_injected_count")
                    ),
                    "mean_decision_checkpoint_trap_injected_rate": _as_float(
                        metrics.get(
                            "mean_decision_checkpoint_trap_injected_rate",
                            metrics.get("decision_checkpoint_trap_injected_rate"),
                        )
                    ),
                    "decision_checkpoint_trap_injected_any_rate": _as_float(
                        metrics.get("decision_checkpoint_trap_injected_any_rate")
                    ),
                    "mean_forced_trap_injected_count": _as_float(
                        metrics.get("mean_forced_trap_injected_count")
                    ),
                    "mean_forced_trap_injected_rate": _as_float(
                        metrics.get("mean_forced_trap_injected_rate")
                    ),
                    "forced_trap_injected_any_rate": _as_float(
                        metrics.get("forced_trap_injected_any_rate")
                    ),
                    "mean_core_size": _as_float(metrics.get("mean_core_size")),
                    "gold_core_has_decision_checkpoint_rate": _as_float(
                        metrics.get("gold_core_has_decision_checkpoint_rate")
                    ),
                    "core_necessity_all_required_rate": _as_float(
                        metrics.get("core_necessity_all_required_rate")
                    ),
                    "mean_core_necessity_flip_count": _as_float(
                        metrics.get("mean_core_necessity_flip_count")
                    ),
                    "core_necessity_failed_rate": _as_float(
                        metrics.get("core_necessity_failed_rate")
                    ),
                    "trap_decision_flip_rate": _as_float(
                        metrics.get("trap_decision_flip_rate")
                    ),
                    "hidden_core_present_rate": _as_float(
                        metrics.get("hidden_core_present_rate")
                    ),
                    "policy_anchor_present_rate": _as_float(
                        metrics.get("policy_anchor_present_rate")
                    ),
                    "policy_anchor_in_context_rate": _as_float(
                        metrics.get("policy_anchor_in_context_rate")
                    ),
                    "policy_anchor_retrieved_by_goc_rate": _as_float(
                        metrics.get("policy_anchor_retrieved_by_goc_rate")
                    ),
                    "policy_codebook_present_rate": _as_float(
                        metrics.get("policy_codebook_present_rate")
                    ),
                    "policy_codebook_in_context_rate": _as_float(
                        metrics.get("policy_codebook_in_context_rate")
                    ),
                    "policy_codebook_retrieved_by_goc_rate": _as_float(
                        metrics.get("policy_codebook_retrieved_by_goc_rate")
                    ),
                    "hidden_core_rescued_by_depwalk_rate": _as_float(
                        metrics.get("hidden_core_rescued_by_depwalk_rate")
                    ),
                    "hidden_core_missing_without_depwalk_rate": _as_float(
                        metrics.get("hidden_core_missing_without_depwalk_rate")
                    ),
                }
            )
            rows.append(row)

            event_file_key = f"{scenario_name}__{variant_name}__{method}__event_traces.jsonl"
            event_rows = event_trace_by_file.setdefault(event_file_key, [])
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                thread_id = str(rec.get("thread_id", "") or "")
                step_idx = int(rec.get("step_idx", 0) or 0)
                thread_obj = thread_lookup.get(thread_id) if isinstance(thread_lookup, dict) else None
                steps = thread_obj.get("steps") if isinstance(thread_obj, dict) else []
                step_obj = None
                if isinstance(steps, list):
                    for s in steps:
                        if not isinstance(s, dict):
                            continue
                        if int(s.get("step_idx", -1) or -1) == step_idx:
                            step_obj = s
                            break
                clauses_map = thread_obj.get("clauses") if isinstance(thread_obj, dict) else {}
                clauses_map = clauses_map if isinstance(clauses_map, dict) else {}
                context_clause_ids = rec.get("e3_context_clause_ids") or []
                if not isinstance(context_clause_ids, list):
                    context_clause_ids = []
                context_clause_ids = [str(cid) for cid in context_clause_ids]
                context_clauses: List[Dict[str, Any]] = []
                for cid in context_clause_ids[:80]:
                    clause = clauses_map.get(cid)
                    if not isinstance(clause, dict):
                        continue
                    context_clauses.append(
                        {
                            "id": str(cid),
                            "type": str(clause.get("node_type", "") or ""),
                            "text": str(clause.get("text", "") or ""),
                            "state_key": str(clause.get("state_key", "") or ""),
                            "state_value": str(clause.get("state_value", "") or ""),
                        }
                    )
                all_prior_clause_ids = [
                    str(cid)
                    for cid, cobj in clauses_map.items()
                    if isinstance(cobj, dict) and int(cobj.get("step_idx", 0) or 0) < int(step_idx)
                ]

                gold_decision_raw = str(
                    rec.get("gold_decision_raw", rec.get("gold_decision", "")) or ""
                )
                gold_decision = normalize_decision(
                    str(rec.get("gold_decision", gold_decision_raw) or gold_decision_raw)
                )
                gold_family = str(
                    rec.get("gold_decision_family")
                    or _gold_decision_family(gold_decision_raw or gold_decision)
                )
                gold_core_ids = list(
                    rec.get("pivot_required_clause_ids")
                    or rec.get("gold_evidence_ids")
                    or []
                )
                pivot_question = str(
                    (step_obj or {}).get("message", "")
                    if isinstance(step_obj, dict)
                    else ""
                )
                scenario_metrics = _compute_scenario_metrics(
                    pivot_question=pivot_question,
                    gold_core_ids=gold_core_ids,
                    clauses_map=clauses_map,
                    all_prior_clause_ids=all_prior_clause_ids,
                )
                rec_for_line = dict(rec)
                rec_for_line.update(scenario_metrics)
                base_line: Dict[str, Any] = {
                    "task_id": str(rec.get("task_id", "") or ""),
                    "thread_id": thread_id,
                    "step_idx": step_idx,
                    "variant_name": variant_name,
                    "method_name": str(method),
                    "state": (
                        dict((step_obj or {}).get("state") or {})
                        if isinstance(step_obj, dict)
                        else {}
                    ),
                    "pivot_question": pivot_question,
                    "gold": {
                        "decision_raw": gold_decision_raw,
                        "decision": gold_decision,
                        "decision_family": gold_family,
                        "conditions": list(rec.get("gold_conditions") or []),
                        "evidence_core_ids": list(gold_core_ids),
                    },
                    "pred": {
                        "decision_raw": str(rec.get("pred_decision_raw", rec.get("pred_decision", "")) or ""),
                        "decision": str(rec.get("pred_decision", "") or ""),
                        "conditions": list(rec.get("pred_conditions") or []),
                        "evidence": list(rec.get("pred_evidence") or []),
                    },
                    "decision_correct": bool(rec.get("decision_correct", False)),
                    "decision_correct_strict_raw": bool(
                        rec.get("decision_correct_strict_raw", rec.get("decision_correct_exact", False))
                    ),
                    "conditions_correct": bool(rec.get("conditions_correct", False)),
                    "conditions_correct_exact": bool(rec.get("conditions_correct_exact", False)),
                    "conditions_correct_subset": bool(rec.get("conditions_correct_subset", False)),
                    "conditions_correct_exact_equiv": bool(
                        rec.get("conditions_correct_exact_equiv", False)
                    ),
                    "conditions_correct_subset_equiv": bool(
                        rec.get("conditions_correct_subset_equiv", False)
                    ),
                    "e3_answer_correct": bool(rec.get("e3_answer_correct", False)),
                    "evidence_core_covered_strict": bool(
                        rec.get("evidence_core_covered_strict", False)
                    ),
                    "evidence_core_covered_equiv": bool(
                        rec.get("evidence_core_covered_equiv", False)
                    ),
                    "evidence_core_missing_ids_strict": list(
                        rec.get("evidence_core_missing_ids_strict") or []
                    ),
                    "evidence_core_missing_equiv_keys": list(
                        rec.get("evidence_core_missing_equiv_keys") or []
                    ),
                    "evidence_core_id_mismatch_but_equiv_present": bool(
                        rec.get("evidence_core_id_mismatch_but_equiv_present", False)
                    ),
                    "goc_avoid_target_clause_ids": list(rec.get("goc_avoid_target_clause_ids") or []),
                    "context_clause_ids": list(context_clause_ids),
                    "context_clauses": context_clauses,
                    "llm_output_text": str(rec.get("llm_output_text", "") or ""),
                    "llm_parse_error": bool(rec.get("llm_parse_error", False)),
                    "tokens": {
                        "prompt_actual": rec.get("prompt_tokens_actual"),
                        "completion_actual": rec.get("completion_tokens_actual"),
                        "total_actual": rec.get("total_tokens_actual"),
                        "prompt_est": rec.get("prompt_tokens_est", rec.get("prompt_tokens")),
                    },
                }
                event_rows.append(build_event_trace_line(rec_for_line, base_line, max_list_items=200))

    root_name = str(phase15_root.name).lower()
    if root_name.startswith("phase") and len(root_name) > 5:
        phase_label = root_name
    else:
        phase_label = "phase15"
    df = _attach_vs_full(pd.DataFrame(rows))
    csv_path = out_dir / f"{phase_label}_traceops_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines: List[str] = []
    md_lines.append(f"# {phase_label.capitalize()} TraceOps Summary")
    md_lines.append("")
    if df.empty:
        md_lines.append("No rows found.")
    else:
        md_lines.append("## Raw Rows")
        md_lines.append("")
        raw_cols = [
            "traceops_level",
            "traceops_scenario",
            "method",
            "pivot_decision_accuracy",
            "pivot_decision_accuracy_strict_raw",
            "pivot_e3_only_accuracy",
            "strict_pivot_accuracy",
            "pivots_available_total",
            "pivots_evaluated",
            "steps_available_total",
            "sampled_step_rate",
            "sampled_steps_evaluated",
            "tokens_pivot_mean",
            "tokens_total_mean",
            "tokens_pivot_mean_est",
            "tokens_total_mean_est",
            "tokens_pivot_mean_actual",
            "tokens_total_mean_actual",
            "mean_avoid_targets_per_pivot",
            "avoided_injected_rate",
            "exception_injected_rate",
            "mean_exception_injected_count",
            "mean_goc_stable_update_count",
            "mean_goc_recent_update_count",
            "stable_update_present_rate",
            "mean_depwalk_added_count",
            "mean_exception_rescue_count",
            "mean_update_history_rescue_count",
            "revive_success_rate",
            "mean_indirection_overlap_gold",
            "mean_trap_gap",
            "trap_present_rate",
            "mean_trap_injected_count",
            "mean_trap_injected_rate",
            "trap_injected_any_rate",
            "mean_decision_checkpoint_trap_injected_count",
            "mean_decision_checkpoint_trap_injected_rate",
            "decision_checkpoint_trap_injected_any_rate",
            "mean_forced_trap_injected_count",
            "mean_forced_trap_injected_rate",
            "forced_trap_injected_any_rate",
            "mean_core_size",
            "gold_core_has_decision_checkpoint_rate",
            "core_necessity_all_required_rate",
            "mean_core_necessity_flip_count",
            "core_necessity_failed_rate",
            "trap_decision_flip_rate",
            "hidden_core_present_rate",
            "policy_anchor_present_rate",
            "policy_anchor_in_context_rate",
            "policy_anchor_retrieved_by_goc_rate",
            "policy_codebook_present_rate",
            "policy_codebook_in_context_rate",
            "policy_codebook_retrieved_by_goc_rate",
            "hidden_core_rescued_by_depwalk_rate",
            "hidden_core_missing_without_depwalk_rate",
            "tokens_savings_vs_full",
            "tokens_savings_vs_full_actual",
            "accuracy_delta_vs_full",
        ]
        cols = [c for c in raw_cols if c in df.columns]
        md_lines.append(df[cols].to_markdown(index=False))
        md_lines.append("")

        group_cols = ["traceops_level", "traceops_scenario", "method"]
        agg_cols = [
            "pivot_decision_accuracy",
            "pivot_decision_accuracy_strict_raw",
            "pivot_e3_only_accuracy",
            "strict_pivot_accuracy",
            "pivots_available_total",
            "pivots_evaluated",
            "steps_available_total",
            "sampled_step_rate",
            "sampled_steps_evaluated",
            "tokens_pivot_mean",
            "tokens_total_mean",
            "tokens_pivot_mean_est",
            "tokens_total_mean_est",
            "tokens_pivot_mean_actual",
            "tokens_total_mean_actual",
            "mean_avoid_targets_per_pivot",
            "avoided_injected_rate",
            "exception_injected_rate",
            "mean_exception_injected_count",
            "mean_goc_stable_update_count",
            "mean_goc_recent_update_count",
            "stable_update_present_rate",
            "mean_depwalk_added_count",
            "mean_exception_rescue_count",
            "mean_update_history_rescue_count",
            "revive_success_rate",
            "mean_indirection_overlap_gold",
            "mean_trap_gap",
            "trap_present_rate",
            "mean_trap_injected_count",
            "mean_trap_injected_rate",
            "trap_injected_any_rate",
            "mean_decision_checkpoint_trap_injected_count",
            "mean_decision_checkpoint_trap_injected_rate",
            "decision_checkpoint_trap_injected_any_rate",
            "mean_forced_trap_injected_count",
            "mean_forced_trap_injected_rate",
            "forced_trap_injected_any_rate",
            "mean_core_size",
            "gold_core_has_decision_checkpoint_rate",
            "core_necessity_all_required_rate",
            "mean_core_necessity_flip_count",
            "core_necessity_failed_rate",
            "trap_decision_flip_rate",
            "hidden_core_present_rate",
            "policy_anchor_present_rate",
            "policy_anchor_in_context_rate",
            "policy_anchor_retrieved_by_goc_rate",
            "policy_codebook_present_rate",
            "policy_codebook_in_context_rate",
            "policy_codebook_retrieved_by_goc_rate",
            "hidden_core_rescued_by_depwalk_rate",
            "hidden_core_missing_without_depwalk_rate",
        ]
        grouped = (
            df.groupby(group_cols, dropna=False)[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(group_cols)
        )
        grouped = _attach_vs_full(grouped)
        md_lines.append("## Aggregated By Level/Scenario/Method")
        md_lines.append("")
        md_lines.append(grouped.to_markdown(index=False))
        md_lines.append("")

        def _bucket_delay(v: Any) -> str:
            try:
                val = int(v)
            except Exception:
                return "unknown"
            if val <= 2:
                return "short"
            if val <= 5:
                return "mid"
            return "long"

        def _bucket_branch(v: Any) -> str:
            try:
                val = int(v)
            except Exception:
                return "unknown"
            if val <= 1:
                return "low"
            if val <= 3:
                return "mid"
            return "high"

        def _bucket_contra(v: Any) -> str:
            if not isinstance(v, (int, float)):
                return "unknown"
            if v < 0.2:
                return "low"
            if v < 0.5:
                return "mid"
            return "high"

        bucket_df = df.copy()
        bucket_df["delay_bucket"] = bucket_df["traceops_delay_to_relevance"].map(_bucket_delay)
        bucket_df["branch_bucket"] = bucket_df["traceops_distractor_branching"].map(_bucket_branch)
        bucket_df["contra_bucket"] = bucket_df["traceops_contradiction_rate"].map(_bucket_contra)

        bucket_group = (
            bucket_df.groupby(["delay_bucket", "branch_bucket", "contra_bucket", "method"], dropna=False)[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["delay_bucket", "branch_bucket", "contra_bucket", "method"])
        )
        md_lines.append("## Aggregated By Knob Buckets")
        md_lines.append("")
        md_lines.append(bucket_group.to_markdown(index=False))

    md_path = out_dir / f"{phase_label}_traceops_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    for filename, lines in event_trace_by_file.items():
        out_path = quick_event_dir / filename
        with out_path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=True) + "\n")

    taxonomy_df, case_lines = _build_failure_taxonomy(phase15_root, event_trace_by_file)
    taxonomy_csv = phase15_root / "quick_access" / "failure_taxonomy.csv"
    taxonomy_md = phase15_root / "quick_access" / "failure_cases.md"
    if taxonomy_df.empty:
        taxonomy_df = pd.DataFrame(
            columns=["scenario", "variant", "method", "category", "count", "example_keys"]
        )
    taxonomy_df.to_csv(taxonomy_csv, index=False)
    taxonomy_md.write_text("\n".join(case_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {quick_event_dir}")
    print(f"Wrote: {taxonomy_csv}")
    print(f"Wrote: {taxonomy_md}")


if __name__ == "__main__":
    main()
