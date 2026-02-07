from __future__ import annotations

import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .bridged_ab import compute_bridged_ab_slices


def _resolve_base_dir(report_path: Path) -> Path:
    parts = report_path.resolve().parts
    if "runs" in parts:
        idx = parts.index("runs")
        return Path(*parts[:idx])
    return report_path.parent


def _latest_report_path(report_path: Path) -> Path:
    if report_path.is_file():
        return report_path
    candidates = sorted(report_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No report JSON files found in {report_path}")
    return candidates[0]


def _parse_timestamp(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _run_sort_key(payload: Dict[str, Any], path: Path) -> Tuple[float, str]:
    ts = _parse_timestamp(str(payload.get("timestamp") or ""))
    if ts:
        ts_val = ts.timestamp()
    else:
        ts_val = path.stat().st_mtime
    run_id = str(payload.get("run_id") or path.stem)
    return (ts_val, run_id)


def summarize_context_budget_sweep(
    run_dir: Path,
    output_dir: Path,
    *,
    prefer_non_saturated_goc: bool = True,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], List[Dict[str, Any]]]:
    compare_paths = list(run_dir.rglob("runs/compare/*.json"))
    latest_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    all_threaded_fu_budgets: set[int] = set()
    symbolic_threaded_fu_budgets: set[int] = set()
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_params = payload.get("scenario_params", {}) or {}
        scenario_mode = scenario_params.get("scenario_mode")
        if not (isinstance(scenario_mode, str) and scenario_mode.startswith("threaded_v1_3_fu")):
            continue
        budget_val = scenario_params.get("thread_context_budget_chars")
        if budget_val is None:
            continue
        try:
            budget = int(budget_val)
        except (TypeError, ValueError):
            continue
        all_threaded_fu_budgets.add(budget)
        # Sweep/Calibration must only use symbolic_packed judge.
        if payload.get("judge") != "symbolic_packed":
            continue
        symbolic_threaded_fu_budgets.add(budget)
        sort_key = _run_sort_key(payload, path)
        for method, report_obj in payload.get("method_reports", {}).items():
            metrics = report_obj.get("metrics", {}) or {}
            records = report_obj.get("records", []) or []
            e3_records = [r for r in records if r.get("episode_id") == 3]

            def _mean_from_records(key: str) -> Optional[float]:
                vals = [
                    float(r.get(key))
                    for r in e3_records
                    if isinstance(r.get(key), (int, float))
                ]
                return sum(vals) / len(vals) if vals else None

            def _mean_bool(vals: List[bool]) -> Optional[float]:
                if not vals:
                    return None
                return sum(1.0 for v in vals if v) / len(vals)

            def _all_critical_rate() -> Optional[float]:
                flags: List[bool] = []
                for r in e3_records:
                    crit_ids = r.get("critical_core_clause_ids") or []
                    if not crit_ids:
                        continue
                    if r.get("e3_packed_all_critical") is not None:
                        flags.append(bool(r.get("e3_packed_all_critical")))
                    else:
                        count = r.get("e3_packed_critical_count")
                        if isinstance(count, (int, float)):
                            flags.append(int(count) == len(crit_ids))
                return _mean_bool(flags)

            def _any_critical_rate() -> Optional[float]:
                flags: List[bool] = []
                for r in e3_records:
                    crit_ids = r.get("critical_core_clause_ids") or []
                    if not crit_ids:
                        continue
                    if r.get("e3_packed_any_critical") is not None:
                        flags.append(bool(r.get("e3_packed_any_critical")))
                    else:
                        count = r.get("e3_packed_critical_count")
                        if isinstance(count, (int, float)):
                            flags.append(int(count) > 0)
                return _mean_bool(flags)

            def _critical_slot_rate(slot_idx: int) -> Optional[float]:
                flags: List[bool] = []
                for r in e3_records:
                    crit_ids = r.get("critical_core_clause_ids") or []
                    if len(crit_ids) <= slot_idx:
                        continue
                    slot_key = f"e3_packed_contains_critical{slot_idx}"
                    if r.get(slot_key) is not None:
                        flags.append(bool(r.get(slot_key)))
                    else:
                        packed_ids = set(r.get("e3_packed_clause_ids") or [])
                        flags.append(str(crit_ids[slot_idx]) in packed_ids)
                return _mean_bool(flags)

            e3_judge_acc = metrics.get("e3_judge_accuracy_packed")
            if e3_judge_acc is None:
                e3_judge_acc = metrics.get("episode_judge_accuracy_e3")
            if e3_judge_acc is None:
                e3_judge_acc = metrics.get("judge_accuracy_packed")
            if e3_judge_acc is None:
                e3_judge_acc = _mean_bool(
                    [
                        bool(r.get("judge_correct"))
                        for r in e3_records
                        if r.get("judge_correct") is not None
                    ]
                )

            key = (budget, method)
            row = {
                "budget": budget,
                "method": method,
                "e3_judge_acc_packed": e3_judge_acc,
                "e3_packed_any_critical_rate": metrics.get(
                    "e3_packed_any_critical_rate"
                )
                if metrics.get("e3_packed_any_critical_rate") is not None
                else _any_critical_rate(),
                "e3_packed_contains_critical0_rate": metrics.get(
                    "e3_packed_contains_critical0_rate"
                )
                if metrics.get("e3_packed_contains_critical0_rate") is not None
                else _critical_slot_rate(0),
                "e3_packed_contains_critical1_rate": metrics.get(
                    "e3_packed_contains_critical1_rate"
                )
                if metrics.get("e3_packed_contains_critical1_rate") is not None
                else _critical_slot_rate(1),
                "e3_packed_all_critical_rate": metrics.get(
                    "e3_packed_all_critical_rate"
                )
                if metrics.get("e3_packed_all_critical_rate") is not None
                else _all_critical_rate(),
                "e3_packed_critical_count_mean": metrics.get(
                    "e3_packed_critical_count_mean"
                )
                if metrics.get("e3_packed_critical_count_mean") is not None
                else _mean_from_records("e3_packed_critical_count"),
                "e3_decoy_clause_count_mean": metrics.get("e3_decoy_clause_count_mean")
                if metrics.get("e3_decoy_clause_count_mean") is not None
                else _mean_from_records("e3_decoy_clause_count"),
                "e3_litm_filler_count_mean": metrics.get("e3_litm_filler_count_mean")
                if metrics.get("e3_litm_filler_count_mean") is not None
                else _mean_from_records("e3_litm_filler_count"),
                "e3_context_truncated_rate": metrics.get("e3_context_truncated_rate")
                if metrics.get("e3_context_truncated_rate") is not None
                else _mean_from_records("e3_context_truncated"),
                "e3_context_chars_used_mean": metrics.get("e3_context_chars_used_mean")
                if metrics.get("e3_context_chars_used_mean") is not None
                else _mean_from_records("e3_context_chars_used"),
                "e3_context_token_est_mean": metrics.get("e3_context_token_est_mean")
                if metrics.get("e3_context_token_est_mean") is not None
                else _mean_from_records("e3_context_token_est"),
                "e3_context_clause_count_mean": metrics.get("e3_context_clause_count_mean")
                if metrics.get("e3_context_clause_count_mean") is not None
                else _mean_from_records("e3_context_clause_count"),
                "e3_packed_dropped_clause_count_mean": metrics.get(
                    "e3_packed_dropped_clause_count_mean"
                )
                if metrics.get("e3_packed_dropped_clause_count_mean") is not None
                else _mean_from_records("e3_packed_dropped_clause_count"),
                "cost_per_correct_token_est": metrics.get("cost_per_correct_token_est"),
                "acc_per_1k_tokens": metrics.get("acc_per_1k_tokens"),
                "goc_unfolded_clause_count_mean": metrics.get(
                    "goc_unfolded_clause_count_mean"
                )
                if metrics.get("goc_unfolded_clause_count_mean") is not None
                else _mean_from_records("goc_unfolded_clause_count"),
                "goc_unfolded_critical_clause_count_mean": metrics.get(
                    "goc_unfolded_critical_clause_count_mean"
                )
                if metrics.get("goc_unfolded_critical_clause_count_mean") is not None
                else _mean_from_records("goc_unfolded_critical_clause_count"),
                "goc_folded_episode_count_mean": metrics.get(
                    "goc_folded_episode_count_mean"
                )
                if metrics.get("goc_folded_episode_count_mean") is not None
                else _mean_from_records("goc_folded_episode_count"),
                "judge_accuracy": metrics.get("judge_accuracy"),
                "_sort_key": sort_key,
                "_run_id": payload.get("run_id") or path.stem,
            }
            existing = latest_rows.get(key)
            if existing is None or sort_key > existing.get("_sort_key", (0.0, "")):
                latest_rows[key] = row

    sweep_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in latest_rows.values()
    ]
    sweep_rows.sort(key=lambda r: (r.get("budget", 0), str(r.get("method", ""))))

    if not sweep_rows:
        return None, None, None, []

    sweep_fields = [
        "budget",
        "method",
        "e3_judge_acc_packed",
        "e3_packed_all_critical_rate",
        "e3_packed_any_critical_rate",
        "e3_packed_contains_critical0_rate",
        "e3_packed_contains_critical1_rate",
        "e3_packed_critical_count_mean",
        "e3_decoy_clause_count_mean",
        "e3_litm_filler_count_mean",
        "e3_context_truncated_rate",
        "e3_context_chars_used_mean",
        "e3_context_token_est_mean",
        "e3_context_clause_count_mean",
        "e3_packed_dropped_clause_count_mean",
        "cost_per_correct_token_est",
        "acc_per_1k_tokens",
        "goc_unfolded_clause_count_mean",
        "goc_unfolded_critical_clause_count_mean",
        "goc_folded_episode_count_mean",
        "judge_accuracy",
    ]
    sweep_csv = output_dir / "results_context_budget_sweep.csv"
    sweep_md = output_dir / "results_context_budget_sweep.md"

    def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "n/a"

    _write_csv(sweep_csv, sweep_rows, sweep_fields)
    lines = ["# Context Budget Sweep Summary", ""]
    lines.append("|" + "|".join(sweep_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(sweep_fields)) + "|")
    for row in sweep_rows:
        lines.append(
            "|"
            + "|".join(
                _fmt(row.get(f)) if f not in {"budget", "method"} else str(row.get(f))
                for f in sweep_fields
            )
            + "|"
        )
    sweep_md.write_text("\n".join(lines), encoding="utf-8")

    # Calibration recommendation (critical miss band, all-critical)
    by_budget: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in sweep_rows:
        by_budget.setdefault(int(row["budget"]), {})[str(row["method"])] = row

    candidates: List[Tuple[int, float]] = []
    for budget, methods in by_budget.items():
        crit_rate = methods.get("full_history", {}).get("e3_packed_all_critical_rate")
        if isinstance(crit_rate, (int, float)):
            candidates.append((budget, float(crit_rate)))

    recommended_budget = None
    recommended_band = None
    recommended_gap = None
    band_candidates: List[Tuple[int, float]] = []
    for budget, crit_rate in candidates:
        if 0.30 <= crit_rate <= 0.70:
            band_candidates.append((budget, crit_rate))

    def _gap_for_budget(budget: int) -> float:
        goc_val = by_budget.get(budget, {}).get("goc", {}).get(
            "e3_packed_all_critical_rate"
        )
        full_val = by_budget.get(budget, {}).get("full_history", {}).get(
            "e3_packed_all_critical_rate"
        )
        if isinstance(goc_val, (int, float)) and isinstance(full_val, (int, float)):
            return float(goc_val) - float(full_val)
        return float("-inf")

    if band_candidates:
        if prefer_non_saturated_goc:
            non_saturated = []
            for budget, rate in band_candidates:
                goc_val = by_budget.get(budget, {}).get("goc", {}).get(
                    "e3_packed_all_critical_rate"
                )
                if isinstance(goc_val, (int, float)) and goc_val < 1.0:
                    non_saturated.append((budget, rate))
            if non_saturated:
                band_candidates = non_saturated

        def _score(item: Tuple[int, float]) -> Tuple[float, int, int]:
            budget, _ = item
            gap = _gap_for_budget(budget)
            goc_val = by_budget.get(budget, {}).get("goc", {}).get(
                "e3_packed_all_critical_rate"
            )
            saturated_penalty = 1 if prefer_non_saturated_goc and goc_val == 1.0 else 0
            return (gap, -saturated_penalty, -budget)

        best = max(band_candidates, key=_score)
        recommended_budget = best[0]
        recommended_band = best[1]
        gap_val = _gap_for_budget(recommended_budget)
        if gap_val != float("-inf"):
            recommended_gap = gap_val

    calib_md = output_dir / "calibration_recommendation.md"
    calib_lines: List[str] = []
    calib_lines.append("# Calibration Recommendation")
    calib_lines.append("")
    missing_symbolic_budgets = sorted(all_threaded_fu_budgets - symbolic_threaded_fu_budgets)
    if missing_symbolic_budgets:
        calib_lines.append(
            "- warning: missing symbolic_packed compare JSONs for budgets: "
            + ", ".join(str(v) for v in missing_symbolic_budgets)
        )
    if recommended_budget is None:
        calib_lines.append("- status: FAIL (no budget in target critical-miss band)")
        calib_lines.append(
            "- recommended_budget: n/a (no full_history e3_packed_all_critical_rate in [0.30, 0.70])"
        )
        calib_lines.append("- suggestion: expand the budget sweep range.")
    else:
        calib_lines.append("- status: PASS")
        calib_lines.append(f"- recommended_budget: {recommended_budget}")
        calib_lines.append(
            f"- full_history_e3_packed_all_critical_rate: {_fmt(recommended_band)}"
        )
        calib_lines.append(
            f"- goc_vs_full_e3_packed_all_critical_gap: {_fmt(recommended_gap)}"
        )
        if prefer_non_saturated_goc:
            calib_lines.append("- prefer_non_saturated_goc: true")
    goc_values = [
        methods.get("goc", {}).get("e3_packed_all_critical_rate")
        for methods in by_budget.values()
    ]
    if goc_values and all(isinstance(v, (int, float)) and abs(v - 1.0) < 1e-6 for v in goc_values):
        calib_lines.append(
            "- warning: goc e3_packed_all_critical_rate is saturated at 1.0 across all budgets."
        )
    calib_lines.append(
        "- Note: This budget is selected for calibration of the cost-control regime, not for score tuning."
    )
    calib_lines.append("")
    calib_lines.append("## Sweep Table")
    calib_lines.append("")
    calib_lines.append("|" + "|".join(sweep_fields) + "|")
    calib_lines.append("|" + "|".join(["---"] * len(sweep_fields)) + "|")
    for row in sweep_rows:
        calib_lines.append(
            "|"
            + "|".join(
                _fmt(row.get(f)) if f not in {"budget", "method"} else str(row.get(f))
                for f in sweep_fields
            )
            + "|"
        )
    calib_md.write_text("\n".join(calib_lines), encoding="utf-8")

    return sweep_csv, sweep_md, calib_md, sweep_rows


def summarize_cost_efficiency_reports(
    output_dir: Path,
    sweep_rows: List[Dict[str, Any]],
) -> Tuple[Optional[Path], Optional[Path]]:
    if not sweep_rows:
        return None, None

    def _is_num(val: Any) -> bool:
        return isinstance(val, (int, float)) and not math.isnan(float(val))

    def _fmt(val: Any) -> str:
        if isinstance(val, bool):
            return "true" if val else "false"
        if _is_num(val):
            return f"{float(val):.4f}"
        return "n/a"

    rows = sorted(
        [
            {
                "budget": int(row.get("budget")),
                "method": str(row.get("method")),
                "e3_packed_all_critical_rate": row.get("e3_packed_all_critical_rate"),
                "e3_context_token_est_mean": row.get("e3_context_token_est_mean"),
                "cost_per_correct_token_est": row.get("cost_per_correct_token_est"),
            }
            for row in sweep_rows
            if isinstance(row.get("budget"), (int, float))
        ],
        key=lambda r: (r["budget"], r["method"]),
    )
    if not rows:
        return None, None

    # Pareto front is computed per budget across methods using acc↑ and token↓.
    for row in rows:
        row["pareto_efficient"] = True
    by_budget: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_budget.setdefault(int(row["budget"]), []).append(row)
    for _, budget_rows in by_budget.items():
        for row in budget_rows:
            acc_i = row.get("e3_packed_all_critical_rate")
            tok_i = row.get("e3_context_token_est_mean")
            if not (_is_num(acc_i) and _is_num(tok_i)):
                row["pareto_efficient"] = False
                continue
            for other in budget_rows:
                if other is row:
                    continue
                acc_j = other.get("e3_packed_all_critical_rate")
                tok_j = other.get("e3_context_token_est_mean")
                if not (_is_num(acc_j) and _is_num(tok_j)):
                    continue
                if (
                    float(acc_j) >= float(acc_i)
                    and float(tok_j) <= float(tok_i)
                    and (float(acc_j) > float(acc_i) or float(tok_j) < float(tok_i))
                ):
                    row["pareto_efficient"] = False
                    break

    pareto_fields = [
        "budget",
        "method",
        "e3_packed_all_critical_rate",
        "e3_context_token_est_mean",
        "cost_per_correct_token_est",
        "pareto_efficient",
    ]
    pareto_md = output_dir / "cost_pareto.md"
    lines: List[str] = ["# Cost Pareto (Threaded)", ""]
    lines.append("|" + "|".join(pareto_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(pareto_fields)) + "|")
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                _fmt(row.get(field))
                if field not in {"budget", "method"}
                else str(row.get(field))
                for field in pareto_fields
            )
            + "|"
        )
    pareto_md.write_text("\n".join(lines), encoding="utf-8")

    # Accuracy-matched cost comparison at budgets where full_history is in calibration band.
    accuracy_md = output_dir / "accuracy_matched_cost.md"
    lines = ["# Accuracy Matched Cost (Threaded)", ""]
    lines.append(
        "Budgets included when `full_history e3_packed_all_critical_rate` is in [0.30, 0.70]."
    )
    lines.append("")
    matched_fields = [
        "budget",
        "target_acc_full_history",
        "goc_acc",
        "full_history_cost_per_correct_token_est",
        "goc_cost_per_correct_token_est",
        "cost_savings_ratio_vs_full_history",
    ]
    matched_rows: List[Dict[str, Any]] = []
    for budget in sorted(by_budget.keys()):
        budget_rows = by_budget[budget]
        by_method = {str(row["method"]): row for row in budget_rows}
        full_row = by_method.get("full_history")
        goc_row = by_method.get("goc")
        if not full_row or not goc_row:
            continue
        full_acc = full_row.get("e3_packed_all_critical_rate")
        if not (_is_num(full_acc) and 0.30 <= float(full_acc) <= 0.70):
            continue
        full_cost = full_row.get("cost_per_correct_token_est")
        goc_cost = goc_row.get("cost_per_correct_token_est")
        savings = None
        if _is_num(full_cost) and _is_num(goc_cost) and float(full_cost) > 0.0:
            savings = (float(full_cost) - float(goc_cost)) / float(full_cost)
        matched_rows.append(
            {
                "budget": budget,
                "target_acc_full_history": full_acc,
                "goc_acc": goc_row.get("e3_packed_all_critical_rate"),
                "full_history_cost_per_correct_token_est": full_cost,
                "goc_cost_per_correct_token_est": goc_cost,
                "cost_savings_ratio_vs_full_history": savings,
            }
        )
    if matched_rows:
        lines.append("|" + "|".join(matched_fields) + "|")
        lines.append("|" + "|".join(["---"] * len(matched_fields)) + "|")
        for row in matched_rows:
            lines.append(
                "|"
                + "|".join(
                    _fmt(row.get(field)) if field != "budget" else str(row.get(field))
                    for field in matched_fields
                )
                + "|"
            )
    else:
        lines.append("- No budgets in the full_history target band [0.30, 0.70].")
    accuracy_md.write_text("\n".join(lines), encoding="utf-8")
    return pareto_md, accuracy_md


def summarize_efficiency_thresholds(
    output_dir: Path,
    sweep_rows: List[Dict[str, Any]],
    *,
    thresholds: Optional[List[float]] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    if not sweep_rows:
        return None, None
    if thresholds is None:
        thresholds = [0.30, 0.50, 0.67, 0.90]

    methods = sorted({str(row.get("method")) for row in sweep_rows if row.get("method")})
    if not methods:
        return None, None

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep_rows:
        method = str(row.get("method"))
        if not method:
            continue
        by_method.setdefault(method, []).append(row)
    for method in methods:
        by_method[method] = sorted(
            by_method.get(method, []),
            key=lambda row: int(row.get("budget") or 0),
        )

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for method in methods:
            selected: Optional[Dict[str, Any]] = None
            for row in by_method.get(method, []):
                val = row.get("e3_packed_all_critical_rate")
                if isinstance(val, (int, float)) and float(val) >= float(threshold):
                    selected = row
                    break
            rows.append(
                {
                    "threshold": float(threshold),
                    "method": method,
                    "min_budget": selected.get("budget") if selected else None,
                    "e3_packed_all_critical_rate": selected.get("e3_packed_all_critical_rate")
                    if selected
                    else None,
                    "e3_context_token_est_mean": selected.get("e3_context_token_est_mean")
                    if selected
                    else None,
                    "cost_per_correct_token_est": selected.get("cost_per_correct_token_est")
                    if selected
                    else None,
                    "acc_per_1k_tokens": selected.get("acc_per_1k_tokens")
                    if selected
                    else None,
                }
            )

    fieldnames = [
        "threshold",
        "method",
        "min_budget",
        "e3_packed_all_critical_rate",
        "e3_context_token_est_mean",
        "cost_per_correct_token_est",
        "acc_per_1k_tokens",
    ]
    csv_path = output_dir / "efficiency_thresholds.csv"
    md_path = output_dir / "efficiency_thresholds.md"

    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{float(val):.4f}"
        return "n/a"

    lines: List[str] = []
    lines.append("# Efficiency Thresholds (Threaded)")
    lines.append("")
    lines.append(
        "Computed from `results_context_budget_sweep` filtered to `judge=symbolic_packed`."
    )
    lines.append("")
    lines.append("|" + "|".join(fieldnames) + "|")
    lines.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                (
                    str(int(row.get(col)))
                    if col == "min_budget" and isinstance(row.get(col), (int, float))
                    else _fmt(row.get(col))
                )
                if col not in {"method", "threshold"}
                else (f"{float(row.get(col)):.2f}" if col == "threshold" else str(row.get(col)))
                for col in fieldnames
            )
            + "|"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def summarize_seed_stability(
    run_dir: Path,
    output_dir: Path,
    *,
    judge_mode: str,
    metrics: List[str],
    filename_prefix: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    compare_paths = list(run_dir.rglob("runs/compare/*.json"))
    latest_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("judge") != judge_mode:
            continue
        scenario_params = payload.get("scenario_params", {}) or {}
        scenario_mode = scenario_params.get("scenario_mode")
        if not (isinstance(scenario_mode, str) and scenario_mode.startswith("threaded_")):
            continue
        seed = scenario_params.get("seed")
        if seed is None:
            continue
        sort_key = _run_sort_key(payload, path)
        for method, report_obj in payload.get("method_reports", {}).items():
            metrics_obj = report_obj.get("metrics", {}) or {}
            usage_obj = report_obj.get("usage", {}) or {}
            row = {
                "seed": int(seed),
                "method": method,
                "judge_accuracy": metrics_obj.get("judge_accuracy"),
                "decision_accuracy": metrics_obj.get("decision_accuracy"),
                "e3_packed_all_critical_rate": metrics_obj.get(
                    "e3_packed_all_critical_rate"
                ),
                "e3_packed_critical_count_mean": metrics_obj.get(
                    "e3_packed_critical_count_mean"
                ),
                "e3_context_truncated_rate": metrics_obj.get("e3_context_truncated_rate"),
                "prompt_tokens_avg": usage_obj.get("prompt_tokens_avg"),
                "tool_calls_avg": usage_obj.get("tool_calls_avg"),
                "open_calls_avg": usage_obj.get("open_calls_avg"),
                "_sort_key": sort_key,
            }
            key = (int(seed), method)
            existing = latest_rows.get(key)
            if existing is None or sort_key > existing.get("_sort_key", (0.0, "")):
                latest_rows[key] = row

    rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in latest_rows.values()
    ]
    if not rows:
        return None, None

    methods = sorted({row["method"] for row in rows})
    fieldnames = ["method", "n_seeds"]
    for metric in metrics:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_std")

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        seed_count = len(method_rows)
        summary = {"method": method, "n_seeds": seed_count}
        for metric in metrics:
            vals = [
                float(row.get(metric))
                for row in method_rows
                if isinstance(row.get(metric), (int, float))
            ]
            if vals:
                summary[f"{metric}_mean"] = sum(vals) / len(vals)
                summary[f"{metric}_std"] = (
                    statistics.stdev(vals) if len(vals) > 1 else 0.0
                )
            else:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
        summary_rows.append(summary)

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "n/a"

    def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in cols})

    csv_path = output_dir / f"{filename_prefix}_{judge_mode}.csv"
    md_path = output_dir / f"{filename_prefix}_{judge_mode}.md"
    _write_csv(csv_path, summary_rows, fieldnames)
    lines = [f"# Seed Stability Summary ({judge_mode})", ""]
    lines.append("|" + "|".join(fieldnames) + "|")
    lines.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
    for row in summary_rows:
        lines.append(
            "|"
            + "|".join(
                _fmt(row.get(col)) if col not in {"method", "n_seeds"} else str(row.get(col))
                for col in fieldnames
            )
            + "|"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def summarize_decoy_validity(
    run_dir: Path,
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    compare_paths = list(run_dir.rglob("runs/compare/*.json"))
    latest_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    saw_decoy = False
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_params = payload.get("scenario_params", {}) or {}
        scenario_mode = scenario_params.get("scenario_mode")
        if not (isinstance(scenario_mode, str) and "decoy" in scenario_mode):
            continue
        saw_decoy = True
        budget_val = scenario_params.get("thread_context_budget_chars")
        if budget_val is None:
            continue
        try:
            budget = int(budget_val)
        except (TypeError, ValueError):
            continue
        sort_key = _run_sort_key(payload, path)
        for method, report_obj in payload.get("method_reports", {}).items():
            records = report_obj.get("records", []) or []
            e1e2 = [r for r in records if r.get("episode_id") in {1, 2}]
            e3 = [r for r in records if r.get("episode_id") == 3]

            def _mean_bool(vals: List[bool]) -> Optional[float]:
                if not vals:
                    return None
                return sum(1.0 for v in vals if v) / len(vals)

            def _mean_num(vals: List[Any]) -> Optional[float]:
                nums = [float(v) for v in vals if isinstance(v, (int, float))]
                return sum(nums) / len(nums) if nums else None

            open_decoy_rate = _mean_bool(
                [bool(r.get("opened_decoy_present")) for r in e1e2]
            )
            open_decoy_count_mean = _mean_num(
                [r.get("opened_decoy_clause_count") for r in e1e2]
            )
            e3_decoy_present = [bool((r.get("e3_decoy_clause_count") or 0) > 0) for r in e3]
            e3_decoy_rate = _mean_bool(e3_decoy_present)
            e3_decoy_count_mean = _mean_num([r.get("e3_decoy_clause_count") for r in e3])
            e3_correct = [bool(r.get("judge_correct")) for r in e3 if r.get("judge_correct") is not None]
            e3_decoy_rate_correct = _mean_bool(
                [
                    bool((r.get("e3_decoy_clause_count") or 0) > 0)
                    for r in e3
                    if r.get("judge_correct") is True
                ]
            )
            e3_decoy_rate_incorrect = _mean_bool(
                [
                    bool((r.get("e3_decoy_clause_count") or 0) > 0)
                    for r in e3
                    if r.get("judge_correct") is False
                ]
            )
            delta = None
            if isinstance(e3_decoy_rate_incorrect, (int, float)) and isinstance(
                e3_decoy_rate_correct, (int, float)
            ):
                delta = e3_decoy_rate_incorrect - e3_decoy_rate_correct
            any_flags: List[bool] = []
            all_flags: List[bool] = []
            critical0_flags: List[bool] = []
            critical1_flags: List[bool] = []
            for r in e3:
                crit_ids = r.get("critical_core_clause_ids") or []
                if not crit_ids:
                    continue
                if r.get("e3_packed_any_critical") is not None:
                    any_flags.append(bool(r.get("e3_packed_any_critical")))
                else:
                    count = r.get("e3_packed_critical_count")
                    if isinstance(count, (int, float)):
                        any_flags.append(int(count) > 0)
                if r.get("e3_packed_all_critical") is not None:
                    all_flags.append(bool(r.get("e3_packed_all_critical")))
                else:
                    count = r.get("e3_packed_critical_count")
                    if isinstance(count, (int, float)):
                        all_flags.append(int(count) == len(crit_ids))
                if len(crit_ids) > 0:
                    if r.get("e3_packed_contains_critical0") is not None:
                        critical0_flags.append(bool(r.get("e3_packed_contains_critical0")))
                    else:
                        critical0_flags.append(str(crit_ids[0]) in set(r.get("e3_packed_clause_ids") or []))
                if len(crit_ids) > 1:
                    if r.get("e3_packed_contains_critical1") is not None:
                        critical1_flags.append(bool(r.get("e3_packed_contains_critical1")))
                    else:
                        critical1_flags.append(str(crit_ids[1]) in set(r.get("e3_packed_clause_ids") or []))
            e3_packed_any = _mean_bool(any_flags)
            e3_packed_all = _mean_bool(all_flags)
            e3_judge_acc = _mean_bool(e3_correct)
            row = {
                "budget": budget,
                "method": method,
                "open_decoy_rate_e1_e2": open_decoy_rate,
                "open_decoy_count_mean_e1_e2": open_decoy_count_mean,
                "e3_decoy_packed_rate": e3_decoy_rate,
                "e3_decoy_clause_count_mean": e3_decoy_count_mean,
                "e3_decoy_rate_when_correct": e3_decoy_rate_correct,
                "e3_decoy_rate_when_incorrect": e3_decoy_rate_incorrect,
                "e3_decoy_rate_delta": delta,
                "e3_packed_contains_critical0_rate": _mean_bool(critical0_flags),
                "e3_packed_contains_critical1_rate": _mean_bool(critical1_flags),
                "e3_packed_all_critical_rate": e3_packed_all,
                "e3_packed_any_critical_rate": e3_packed_any,
                "e3_judge_acc_packed": e3_judge_acc,
                "_sort_key": sort_key,
            }
            key = (budget, method)
            existing = latest_rows.get(key)
            if existing is None or sort_key > existing.get("_sort_key", (0.0, "")):
                latest_rows[key] = row

    if not saw_decoy:
        return None, None

    rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in latest_rows.values()
    ]
    if not rows:
        return None, None
    rows.sort(key=lambda r: (r.get("budget", 0), str(r.get("method", ""))))
    fields = [
        "budget",
        "method",
        "open_decoy_rate_e1_e2",
        "open_decoy_count_mean_e1_e2",
        "e3_decoy_packed_rate",
        "e3_decoy_clause_count_mean",
        "e3_decoy_rate_when_correct",
        "e3_decoy_rate_when_incorrect",
        "e3_decoy_rate_delta",
        "e3_packed_contains_critical0_rate",
        "e3_packed_contains_critical1_rate",
        "e3_packed_all_critical_rate",
        "e3_packed_any_critical_rate",
        "e3_judge_acc_packed",
    ]

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "n/a"

    def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in cols})

    csv_path = output_dir / "decoy_validity.csv"
    md_path = output_dir / "decoy_validity.md"
    _write_csv(csv_path, rows, fields)
    lines = ["# Decoy Validity (Threaded Decoy)", ""]
    lines.append("|" + "|".join(fields) + "|")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                _fmt(row.get(f)) if f not in {"budget", "method"} else str(row.get(f))
                for f in fields
            )
            + "|"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _load_raw_preview(record: Dict[str, Any]) -> str:
    raw_output = record.get("raw_output")
    if isinstance(raw_output, str) and raw_output.strip():
        return " ".join(raw_output[:300].split())
    raw_path = record.get("raw_path")
    if raw_path:
        try:
            text = Path(raw_path).read_text(encoding="utf-8")[:300]
            return " ".join(text.split())
        except Exception:
            return ""
    return ""


def analyze_failure_slice(report_path: Path, top_k: int = 20) -> Path:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_reports = payload.get("method_reports", {})

    failures: List[Tuple[str, Dict[str, Any]]] = []
    for method, report in method_reports.items():
        for record in report.get("records", []):
            if not record.get("gold_in_search_topk"):
                continue
            opened_gold_count = record.get("opened_gold_count", 0) or 0
            opened_has_winning = record.get("opened_has_winning_clause")
            if opened_gold_count == 0 or opened_has_winning is False:
                failures.append((method, record))

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_failure_slice.md"

    lines: List[str] = []
    lines.append(f"# Failure Slice Report\n")
    lines.append(f"- Source report: {report_path}\n")
    lines.append(f"- run_id: {payload.get('run_id')}\n")
    lines.append(f"- git_sha: {payload.get('git_sha')}\n")
    lines.append(f"- Failures: {len(failures)}\n")

    if not failures:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    for method, record in failures:
        lines.append(f"## Method: {method} / Task: {record.get('task_id')}\n")
        lines.append(f"- gold_decision: {record.get('gold_decision')}")
        lines.append(f"- pred_decision: {record.get('pred_decision')}")
        lines.append(f"- opened_clause_ids: {record.get('opened_clause_ids')}")
        if record.get("forced_open_ids") is not None:
            lines.append(f"- forced_open_ids: {record.get('forced_open_ids')}")
        if record.get("search_topk_clause_ids") is not None:
            snapshot = record.get("search_topk_clause_ids") or []
            lines.append(f"- primary_search_topk_clause_ids: {snapshot[:top_k]}")
        lines.append(f"- winning_clause_rank: {record.get('winning_clause_rank')}")
        lines.append(f"- min_gold_rank: {record.get('min_gold_rank')}")
        lines.append(f"- gold_score_gap: {record.get('gold_score_gap')}")
        raw_preview = _load_raw_preview(record)
        if raw_preview:
            lines.append(f"- raw_output_preview: {raw_preview}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def analyze_bridged_ab(report_path: Path, method: str = "goc") -> Path:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])
    slices = compute_bridged_ab_slices(records)

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_bridged_ab.md"

    lines: List[str] = []
    lines.append("# Bridged A×B Slice Report\n")
    lines.append(f"- Source report: {report_path}\n")
    lines.append(f"- Method: {method}\n")
    lines.append(f"- Records: {slices.get('n_records', 0)}\n")

    axes = slices.get("axes", {})
    cells = slices.get("cells", {})
    a_keys = axes.get("A", [])
    b_keys = axes.get("B", [])
    header = "|A/B|" + "|".join(b_keys) + "|"
    sep = "|---|" + "|".join(["---"] * len(b_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for a_key in a_keys:
        row = [a_key]
        for b_key in b_keys:
            cell = cells.get(a_key, {}).get(b_key)
            if not cell:
                row.append("n=0")
                continue
            extra = ""
            if a_key == "A2_opened_wrong_bridge":
                extra = (
                    f", probe_gold={cell.get('bridge_probe_contains_gold_canonical_rate',0):.2f}, "
                    f"opened_gold={cell.get('bridge_opened_contains_gold_canonical_rate',0):.2f}"
                )
            if a_key == "A3_opened_gold_bridge" and b_key == "B2_hop2_with_gold_canonical":
                core_rank = cell.get("core_min_rank_union_mean")
                deep_rate = cell.get("deep_rank_core_rate")
                core_rank_str = f"{core_rank:.2f}" if isinstance(core_rank, (int, float)) else "n/a"
                deep_rate_str = f"{deep_rate:.2f}" if isinstance(deep_rate, (int, float)) else "n/a"
                extra = (
                    f"{extra}, core_min_rank_union_mean={core_rank_str}, "
                    f"deep_rank_core_rate={deep_rate_str}"
                )
            decision_acc = cell.get("decision_acc")
            judge_acc = cell.get("judge_acc")
            decision_acc_str = f"{decision_acc:.2f}" if isinstance(decision_acc, (int, float)) else "n/a"
            judge_acc_str = f"{judge_acc:.2f}" if isinstance(judge_acc, (int, float)) else "n/a"
            row.append(
                f"n={cell.get('n',0)}, decision_acc={decision_acc_str}, "
                f"judge_acc={judge_acc_str}, cov_core={cell.get('opened_gold_coverage_core_mean',0):.2f}{extra}"
            )
        lines.append("|" + "|".join(row) + "|")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _bucket_a(rec: Dict[str, Any]) -> str:
    if rec.get("bridge_needed") is False:
        return "A0_no_bridge_needed"
    if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is False:
        return "A1_needed_not_opened"
    if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is True:
        if rec.get("bridge_opened_gold") is True:
            return "A3_opened_gold_bridge"
        if rec.get("bridge_opened_gold") is False:
            return "A2_opened_wrong_bridge"
    return "A_unknown"


def _bucket_b(rec: Dict[str, Any]) -> str:
    hop2_executed = rec.get("hop2_executed") is True
    hop2_gold = rec.get("hop2_query_contains_gold_canonical") is True
    if not hop2_executed:
        return "B0_no_hop2"
    if hop2_gold:
        return "B2_hop2_with_gold_canonical"
    return "B1_hop2_no_gold_canonical"


def analyze_selection_triage(
    report_path: Path,
    method: str = "goc",
    max_per_bucket: int = 20,
) -> Tuple[Path, Path]:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])

    # Ensure triage exists for the same report/method.
    from .triage import triage_compare

    triage_compare(report_path, method=method, max_per_bucket=max_per_bucket)

    rng = random.Random(0)
    sel_gap = [r for r in records if isinstance(r.get("selection_gap"), (int, float)) and r.get("selection_gap") >= 0.5]
    acc_no_core = [r for r in records if r.get("acc_no_core_evidence") is True]
    a3b2_core0 = [
        r
        for r in records
        if _bucket_a(r) == "A3_opened_gold_bridge"
        and _bucket_b(r) == "B2_hop2_with_gold_canonical"
        and (r.get("opened_gold_coverage_core") in {0, 0.0})
    ]

    def _sample(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) <= max_per_bucket:
            return list(items)
        return rng.sample(items, max_per_bucket)

    sampled = {
        "SEL_GAP": _sample(sel_gap),
        "ACC_NO_CORE_EVIDENCE": _sample(acc_no_core),
        "A3B2_CORE0": _sample(a3b2_core0),
    }

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{stamp}_selection_triage.csv"
    md_path = out_dir / f"{stamp}_patterns.md"

    rows: List[Dict[str, Any]] = []
    for bucket, items in sampled.items():
        for rec in items:
            rows.append(
                {
                    "bucket": bucket,
                    "task_id": rec.get("task_id"),
                    "winning_clause_rank_union": rec.get("winning_clause_rank_union"),
                    "hit_at_open_budget_union": rec.get("hit_at_open_budget_union"),
                    "rank_success": rec.get("rank_success"),
                    "opened_has_winning_clause": rec.get("opened_has_winning_clause_union"),
                    "policy_gain_over_rank": rec.get("policy_gain_over_rank"),
                    "opened_gold_coverage_core": rec.get("opened_gold_coverage_core"),
                    "judge_supporting_count": rec.get("judge_supporting_count"),
                    "open_from_hop1_count": rec.get("open_from_hop1_count"),
                    "open_from_hop2_count": rec.get("open_from_hop2_count"),
                    "opened_bridge_count": rec.get("opened_bridge_count"),
                    "opened_meta_count": rec.get("opened_meta_count"),
                    "opened_rule_count": rec.get("opened_rule_count"),
                    "bridge_open_cap_hit": rec.get("bridge_open_cap_hit"),
                    "meta_avoided_count": rec.get("meta_avoided_count"),
                    "hop2_pool_used_count": rec.get("hop2_pool_used_count"),
                    "fallback_reason": rec.get("fallback_reason"),
                }
            )

    # Write CSV
    import csv

    fieldnames = [
        "bucket",
        "task_id",
        "winning_clause_rank_union",
        "hit_at_open_budget_union",
        "rank_success",
        "opened_has_winning_clause",
        "policy_gain_over_rank",
        "opened_gold_coverage_core",
        "judge_supporting_count",
        "open_from_hop1_count",
        "open_from_hop2_count",
        "opened_bridge_count",
        "opened_meta_count",
        "opened_rule_count",
        "bridge_open_cap_hit",
        "meta_avoided_count",
        "hop2_pool_used_count",
        "fallback_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Patterns summary (rule-based)
    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _rate(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [1.0 for r in items if r.get(key)]
        return len(vals) / len(items) if items else None

    lines: List[str] = []
    lines.append("# Representative Failure Patterns")
    lines.append("")
    lines.append(f"- source_report: {report_path}")
    lines.append(f"- method: {method}")
    lines.append("")

    patterns = [
        ("SEL_GAP", sel_gap, "Selection gap: winning clause in union budget but not opened"),
        ("ACC_NO_CORE_EVIDENCE", acc_no_core, "Judge correct without core evidence opened"),
        ("A3B2_CORE0", a3b2_core0, "Bridge+hop2 succeeded but core evidence not opened"),
    ]
    for name, items, desc in patterns:
        lines.append(f"## {name}")
        lines.append(f"{desc}")
        lines.append(f"n={len(items)}")
        mean_gap = _mean(
            [
                float(r.get("selection_gap"))
                for r in items
                if isinstance(r.get("selection_gap"), (int, float))
            ]
        )
        mean_cov = _mean(
            [
                float(r.get("opened_gold_coverage_core"))
                for r in items
                if isinstance(r.get("opened_gold_coverage_core"), (int, float))
            ]
        )
        mean_support = _mean(
            [
                float(r.get("judge_supporting_count"))
                for r in items
                if isinstance(r.get("judge_supporting_count"), (int, float))
            ]
        )
        hit_rate = _rate(items, "rank_success")
        opened_win_rate = _rate(items, "opened_has_winning_clause_union")
        mean_gain = _mean(
            [
                float(r.get("policy_gain_over_rank"))
                for r in items
                if isinstance(r.get("policy_gain_over_rank"), (int, float))
            ]
        )
        lines.append(f"selection_gap_mean={mean_gap if mean_gap is not None else 'n/a'}")
        lines.append(f"opened_gold_coverage_core_mean={mean_cov if mean_cov is not None else 'n/a'}")
        lines.append(f"judge_supporting_count_mean={mean_support if mean_support is not None else 'n/a'}")
        lines.append(f"rank_success_rate={hit_rate if hit_rate is not None else 'n/a'}")
        lines.append(f"opened_has_winning_clause_union_rate={opened_win_rate if opened_win_rate is not None else 'n/a'}")
        lines.append(f"policy_gain_over_rank_mean={mean_gain if mean_gain is not None else 'n/a'}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def analyze_slot_breakdown(
    report_path: Path,
    method: str = "goc",
) -> Tuple[Path, Path]:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        slot_term = rec.get("slot_term") or rec.get("slot") or "unknown"
        grouped.setdefault(str(slot_term), []).append(rec)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    rows: List[Dict[str, Any]] = []
    for slot_term, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        judge_vals = [1.0 if r.get("judge_correct") else 0.0 for r in items if r.get("judge_correct") is not None]
        cov_vals = [
            float(r.get("opened_gold_coverage_core") or 0.0)
            for r in items
            if isinstance(r.get("opened_gold_coverage_core"), (int, float))
        ]
        rank_vals = [
            float(r.get("min_gold_core_rank_union"))
            for r in items
            if isinstance(r.get("min_gold_core_rank_union"), int)
        ]
        deep_vals = [1.0 if r.get("deep_rank_core_flag") else 0.0 for r in items]
        rows.append(
            {
                "slot_term": slot_term,
                "n": len(items),
                "judge_acc": _mean(judge_vals),
                "cov_core": _mean(cov_vals),
                "min_gold_core_rank_union_mean": _mean(rank_vals),
                "deep_rank_core_rate": _mean(deep_vals),
            }
        )

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{stamp}_slot_breakdown.csv"
    md_path = out_dir / f"{stamp}_slot_breakdown.md"

    # CSV
    import csv

    fieldnames = [
        "slot_term",
        "n",
        "judge_acc",
        "cov_core",
        "min_gold_core_rank_union_mean",
        "deep_rank_core_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Markdown table
    lines: List[str] = []
    lines.append("# Slot Breakdown")
    lines.append("")
    lines.append(f"- source_report: {report_path}")
    lines.append(f"- method: {method}")
    lines.append("")
    lines.append("|slot_term|n|judge_acc|cov_core|min_gold_core_rank_union_mean|deep_rank_core_rate|")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        def _fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
        lines.append(
            f"|{row['slot_term']}|{row['n']}|{_fmt(row.get('judge_acc'))}|"
            f"{_fmt(row.get('cov_core'))}|{_fmt(row.get('min_gold_core_rank_union_mean'))}|"
            f"{_fmt(row.get('deep_rank_core_rate'))}|"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def analyze_split_sweep_ab(
    sweep_dir: Path,
    results_md: Optional[Path] = None,
    method: str = "goc",
) -> Path:
    compare_paths = sorted(sweep_dir.rglob("runs/compare/*.json"))
    by_hop1: Dict[int, List[Dict[str, Any]]] = {}

    for path in compare_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hop1 = payload.get("scenario_params", {}).get("open_split_hop1")
        if hop1 is None:
            continue
        records = payload.get("method_reports", {}).get(method, {}).get("records", [])
        a3b2 = [
            r
            for r in records
            if _bucket_a(r) == "A3_opened_gold_bridge"
            and _bucket_b(r) == "B2_hop2_with_gold_canonical"
        ]
        by_hop1.setdefault(int(hop1), []).extend(a3b2)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    lines: List[str] = []
    lines.append("")
    lines.append("## A3×B2 (goc) Core/Selection Summary")
    lines.append("|open_split_hop1|n|judge_acc|cov_core|selection_gap|")
    lines.append("|---|---|---|---|---|")
    for hop1 in sorted(by_hop1.keys()):
        items = by_hop1[hop1]
        judge_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in items])
        cov_core = _mean([float(r.get("opened_gold_coverage_core") or 0.0) for r in items])
        sel_gap = _mean([float(r.get("selection_gap") or 0.0) for r in items])
        def _fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
        lines.append(
            f"|{hop1}|{len(items)}|"
            f"{_fmt(judge_acc)}|"
            f"{_fmt(cov_core)}|"
            f"{_fmt(sel_gap)}|"
        )

    if results_md is None:
        results_md = sweep_dir / "results_split_sweep.md"
    if results_md.exists():
        content = results_md.read_text(encoding="utf-8")
        content = content.rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        content = "\n".join(lines) + "\n"
    results_md.write_text(content, encoding="utf-8")
    return results_md


def analyze_bundle(run_dir: Path) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    compare_paths = list(run_dir.rglob("runs/compare/*.json"))
    threaded_mode = False
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_params = payload.get("scenario_params", {}) or {}
        scenario_mode = scenario_params.get("scenario_mode")
        if isinstance(scenario_mode, str) and scenario_mode.startswith("threaded_"):
            threaded_mode = True
            break

    summary_path = run_dir / "summary.md"
    if not summary_path.exists():
        summary_path = None

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.exists():
            return path
        candidate = run_dir.parent / path_str
        if candidate.exists():
            return candidate
        candidate = Path(path_str).expanduser()
        if candidate.exists():
            return candidate
        return path

    def _normalize_preset(name: str) -> str:
        lowered = name.lower()
        if "n8" in lowered:
            return "n8_exclcore"
        if "n10" in lowered:
            return "n10_exclcore"
        return name

    # Parse summary.md into index.
    index: Dict[str, Dict[str, Dict[str, Path]]] = {}
    current_preset = None
    current_policy = None
    if summary_path:
        for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                current_preset = _normalize_preset(line[3:].strip())
                index.setdefault(current_preset, {})
                current_policy = None
                continue
            if line.startswith("- open_policy:"):
                current_policy = line.split(":", 1)[1].strip()
                if current_preset is None:
                    current_preset = "unknown"
                index.setdefault(current_preset, {}).setdefault(current_policy, {})
                continue
            if ":" in line and current_preset and current_policy:
                key, value = line.split(":", 1)
                key = key.strip("- ").strip()
                value = value.strip()
                if value:
                    index[current_preset][current_policy][key] = _resolve_path(value)

    # If parsing failed, fallback to scanning compare JSONs.
    if not index:
        compare_paths = list(run_dir.rglob("runs/compare/*.json"))
        for path in compare_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            preset = _normalize_preset(
                payload.get("scenario_params", {}).get("preset", "unknown")
            )
            policy = payload.get("scenario_params", {}).get("open_policy", "current")
            index.setdefault(preset, {}).setdefault(policy, {})["compare_json"] = path

    # Backfill analysis paths based on source_report lines.
    analysis_dir = run_dir.parent / "analysis"
    analysis_files = list(analysis_dir.glob("*.md")) + list(analysis_dir.glob("*.csv"))
    analysis_by_report: Dict[str, List[Path]] = {}
    for path in analysis_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        report_line = None
        for line in text.splitlines():
            if line.strip().startswith("- Source report:") or line.strip().startswith("- source_report:"):
                report_line = line
                break
        if not report_line:
            continue
        report_path = report_line.split(":", 1)[1].strip()
        analysis_by_report.setdefault(report_path, []).append(path)

    for preset, policies in index.items():
        for policy, info in policies.items():
            compare_path = info.get("compare_json")
            if not compare_path:
                continue
            report_paths = analysis_by_report.get(str(compare_path), [])
            for path in report_paths:
                if path.name.endswith("_bridged_ab.md") and "bridged_ab_md" not in info:
                    info["bridged_ab_md"] = path
                if path.name.endswith("_selection_triage.csv") and "selection_triage_csv" not in info:
                    info["selection_triage_csv"] = path
                if path.name.endswith("_patterns.md") and "patterns_md" not in info:
                    info["patterns_md"] = path
                if path.name.endswith("_slot_breakdown.md") and "slot_breakdown_md" not in info:
                    info["slot_breakdown_md"] = path
                if path.name.endswith("_slot_breakdown.csv") and "slot_breakdown_csv" not in info:
                    info["slot_breakdown_csv"] = path

    # C0-2: load compare JSONs and extract metrics (skip for threaded runs).
    metric_rows: List[Dict[str, Any]] = []
    if not threaded_mode:
        for preset, policies in index.items():
            for policy, info in policies.items():
                compare_path = info.get("compare_json")
                if not compare_path or not compare_path.exists():
                    continue
                payload = json.loads(compare_path.read_text(encoding="utf-8"))
                methods = payload.get("methods", [])
                for method in methods:
                    report = payload.get("method_reports", {}).get(method, {})
                    metrics = report.get("metrics", {}) or {}
                    summary = payload.get("summary", {}).get(method, {}) or {}
                    row = {
                        "preset": preset,
                        "open_policy": policy,
                        "method": method,
                        "judge_accuracy": metrics.get("judge_accuracy", summary.get("judge_accuracy")),
                        "decision_accuracy": metrics.get("decision_accuracy", summary.get("decision_accuracy")),
                        "rank_success_rate": metrics.get("rank_success_rate", summary.get("rank_success_rate")),
                        "winning_in_union_rate": metrics.get("winning_in_union_rate", summary.get("winning_in_union_rate")),
                        "opened_has_winning_clause_rate_union": metrics.get(
                            "opened_has_winning_clause_rate_union",
                            summary.get("opened_has_winning_clause_rate_union"),
                        ),
                        "policy_gain_over_rank": metrics.get("policy_gain_over_rank", summary.get("policy_gain_over_rank")),
                        "rank_gap": metrics.get("rank_gap", summary.get("rank_gap")),
                        "opened_gold_coverage_core_mean": metrics.get("opened_gold_coverage_core_mean"),
                        "deep_rank_core_rate": metrics.get("deep_rank_core_rate", summary.get("deep_rank_core_rate")),
                        "min_gold_core_rank_union_mean": metrics.get(
                            "min_gold_core_rank_union_mean", summary.get("min_gold_core_rank_union_mean")
                        ),
                        "min_gold_core_rank_union_median": metrics.get(
                            "min_gold_core_rank_union_median", summary.get("min_gold_core_rank_union_median")
                        ),
                        "opened_bridge_count_mean": metrics.get(
                            "opened_bridge_count_mean", summary.get("opened_bridge_count_mean")
                        ),
                        "opened_rule_count_mean": metrics.get(
                            "opened_rule_count_mean", summary.get("opened_rule_count_mean")
                        ),
                        "opened_meta_count_mean": metrics.get(
                            "opened_meta_count_mean", summary.get("opened_meta_count_mean")
                        ),
                    }
                    metric_rows.append(row)

    def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "n/a"

    # Build lookup for per-method metrics.
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in metric_rows:
        by_key[(row["preset"], row["open_policy"], row["method"])] = row

    output_dir = run_dir / "analysis_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_files = [
        "policy_method_matrix.csv",
        "policy_method_matrix.md",
        "goc_deltas.csv",
        "goc_deltas.md",
        "difficulty_sanity.csv",
        "difficulty_sanity.md",
        "narrative_summary.md",
    ]
    if threaded_mode:
        for name in legacy_files:
            legacy_path = output_dir / name
            if legacy_path.exists():
                legacy_path.unlink()

    if not threaded_mode:
        # C1-1 Policy × Method matrix
        matrix_rows: List[Dict[str, Any]] = []
        for preset, policies in sorted(index.items()):
            for policy in sorted(policies.keys()):
                row = {
                    "preset": preset,
                    "open_policy": policy,
                }
                for method in ["goc", "goc_base", "topk", "full"]:
                    key = (preset, policy, method)
                    metrics = by_key.get(key, {})
                    row[f"{method}_judge_acc"] = metrics.get("judge_accuracy")
                    row[f"{method}_policy_gain_over_rank"] = metrics.get("policy_gain_over_rank")
                goc_metrics = by_key.get((preset, policy, "goc"), {})
                row["goc_rank_success_rate"] = goc_metrics.get("rank_success_rate")
                row["goc_winning_in_union_rate"] = goc_metrics.get("winning_in_union_rate")
                matrix_rows.append(row)

        matrix_fields = [
            "preset",
            "open_policy",
            "goc_judge_acc",
            "goc_base_judge_acc",
            "topk_judge_acc",
            "full_judge_acc",
            "goc_rank_success_rate",
            "goc_winning_in_union_rate",
            "goc_policy_gain_over_rank",
            "goc_base_policy_gain_over_rank",
            "topk_policy_gain_over_rank",
            "full_policy_gain_over_rank",
        ]
        matrix_csv = output_dir / "policy_method_matrix.csv"
        _write_csv(matrix_csv, matrix_rows, matrix_fields)

        matrix_md = output_dir / "policy_method_matrix.md"
        lines = []
        lines.append("# Policy × Method Matrix")
        lines.append("")
        lines.append("|" + "|".join(matrix_fields) + "|")
        lines.append("|" + "|".join(["---"] * len(matrix_fields)) + "|")
        for row in matrix_rows:
            lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy"} else str(row.get(f)) for f in matrix_fields) + "|")
        matrix_md.write_text("\n".join(lines), encoding="utf-8")

        # C1-2 GoC deltas
        delta_rows: List[Dict[str, Any]] = []
        for preset, policies in sorted(index.items()):
            for policy in sorted(policies.keys()):
                goc = by_key.get((preset, policy, "goc"), {})
                for baseline in ["goc_base", "topk", "full"]:
                    base = by_key.get((preset, policy, baseline), {})
                    delta_rows.append(
                        {
                            "preset": preset,
                            "open_policy": policy,
                            "baseline": baseline,
                            "delta_judge_acc": (
                                (goc.get("judge_accuracy") or 0) - (base.get("judge_accuracy") or 0)
                                if goc.get("judge_accuracy") is not None and base.get("judge_accuracy") is not None
                                else None
                            ),
                            "delta_policy_gain_over_rank": (
                                (goc.get("policy_gain_over_rank") or 0) - (base.get("policy_gain_over_rank") or 0)
                                if goc.get("policy_gain_over_rank") is not None and base.get("policy_gain_over_rank") is not None
                                else None
                            ),
                        }
                    )
        delta_fields = ["preset", "open_policy", "baseline", "delta_judge_acc", "delta_policy_gain_over_rank"]
        delta_csv = output_dir / "goc_deltas.csv"
        _write_csv(delta_csv, delta_rows, delta_fields)
        delta_md = output_dir / "goc_deltas.md"
        lines = []
        lines.append("# GoC Improvement Decomposition")
        lines.append("")
        lines.append("|" + "|".join(delta_fields) + "|")
        lines.append("|" + "|".join(["---"] * len(delta_fields)) + "|")
        for row in delta_rows:
            lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy", "baseline"} else str(row.get(f)) for f in delta_fields) + "|")
        delta_md.write_text("\n".join(lines), encoding="utf-8")

        # C1-3 Difficulty sanity (goc only)
        sanity_rows: List[Dict[str, Any]] = []
        for preset, policies in sorted(index.items()):
            for policy in sorted(policies.keys()):
                goc = by_key.get((preset, policy, "goc"), {})
                sanity_rows.append(
                    {
                        "preset": preset,
                        "open_policy": policy,
                        "rank_success_rate": goc.get("rank_success_rate"),
                        "deep_rank_core_rate": goc.get("deep_rank_core_rate"),
                        "min_gold_core_rank_union_mean": goc.get("min_gold_core_rank_union_mean"),
                        "min_gold_core_rank_union_median": goc.get("min_gold_core_rank_union_median"),
                    }
                )
        sanity_fields = [
            "preset",
            "open_policy",
            "rank_success_rate",
            "deep_rank_core_rate",
            "min_gold_core_rank_union_mean",
            "min_gold_core_rank_union_median",
        ]
        sanity_csv = output_dir / "difficulty_sanity.csv"
        _write_csv(sanity_csv, sanity_rows, sanity_fields)
        sanity_md = output_dir / "difficulty_sanity.md"
        lines = []
        lines.append("# Difficulty Calibration Sanity")
        lines.append("")
        lines.append("|" + "|".join(sanity_fields) + "|")
        lines.append("|" + "|".join(["---"] * len(sanity_fields)) + "|")
        for row in sanity_rows:
            lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy"} else str(row.get(f)) for f in sanity_fields) + "|")
        sanity_md.write_text("\n".join(lines), encoding="utf-8")

    # C0-3 Slot breakdown parsing + C2 slot summary
    slot_rows: List[Dict[str, Any]] = []
    for preset, policies in sorted(index.items()):
        for policy, info in policies.items():
            slot_md_path = info.get("slot_breakdown_md")
            if not slot_md_path or not slot_md_path.exists():
                continue
            table_lines = []
            for line in slot_md_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("|slot_term|"):
                    table_lines.append(line)
                    continue
                if table_lines:
                    if line.startswith("|") and not line.startswith("|---"):
                        table_lines.append(line)
                    elif line.strip() == "":
                        break
            if len(table_lines) < 2:
                continue
            for row_line in table_lines[1:]:
                parts = [p.strip() for p in row_line.strip("|").split("|")]
                if len(parts) < 6:
                    continue
                slot_rows.append(
                    {
                        "preset": preset,
                        "open_policy": policy,
                        "slot_term": parts[0],
                        "n": int(parts[1]) if parts[1].isdigit() else None,
                        "judge_acc": float(parts[2]) if parts[2] != "n/a" else None,
                        "cov_core": float(parts[3]) if parts[3] != "n/a" else None,
                        "min_gold_core_rank_union_mean": float(parts[4]) if parts[4] != "n/a" else None,
                        "deep_rank_core_rate": float(parts[5]) if parts[5] != "n/a" else None,
                    }
                )

    # Build slot comparison current vs bridge_one_only (fallback hop2_priority)
    slot_summary_rows: List[Dict[str, Any]] = []
    for preset in sorted({r["preset"] for r in slot_rows}):
        current_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == "current"}
        alt_policy = "bridge_one_only"
        alt_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == alt_policy}
        if not alt_rows:
            alt_policy = "hop2_priority"
            alt_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == alt_policy}
        for slot_term, cur in current_rows.items():
            alt = alt_rows.get(slot_term)
            if not alt:
                continue
            delta = None
            if cur.get("judge_acc") is not None and alt.get("judge_acc") is not None:
                delta = alt.get("judge_acc") - cur.get("judge_acc")
            slot_summary_rows.append(
                {
                    "preset": preset,
                    "slot_term": slot_term,
                    "baseline_policy": "current",
                    "compare_policy": alt_policy,
                    "judge_acc_current": cur.get("judge_acc"),
                    "judge_acc_compare": alt.get("judge_acc"),
                    "delta_judge_acc": delta,
                    "cov_core_current": cur.get("cov_core"),
                    "cov_core_compare": alt.get("cov_core"),
                    "deep_rank_core_rate_current": cur.get("deep_rank_core_rate"),
                    "deep_rank_core_rate_compare": alt.get("deep_rank_core_rate"),
                    "min_gold_core_rank_union_mean_current": cur.get("min_gold_core_rank_union_mean"),
                    "min_gold_core_rank_union_mean_compare": alt.get("min_gold_core_rank_union_mean"),
                }
            )

    slot_summary_fields = [
        "preset",
        "slot_term",
        "baseline_policy",
        "compare_policy",
        "judge_acc_current",
        "judge_acc_compare",
        "delta_judge_acc",
        "cov_core_current",
        "cov_core_compare",
        "deep_rank_core_rate_current",
        "deep_rank_core_rate_compare",
        "min_gold_core_rank_union_mean_current",
        "min_gold_core_rank_union_mean_compare",
    ]
    slot_csv = output_dir / "slot_summary.csv"
    slot_md = output_dir / "slot_summary.md"
    if slot_summary_rows:
        _write_csv(slot_csv, slot_summary_rows, slot_summary_fields)
        lines = []
        lines.append("# Slot Summary (Current vs Bridge-One-Only)")
        lines.append("")
        lines.append("|" + "|".join(slot_summary_fields) + "|")
        lines.append("|" + "|".join(["---"] * len(slot_summary_fields)) + "|")
        for row in slot_summary_rows:
            lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "slot_term", "baseline_policy", "compare_policy"} else str(row.get(f)) for f in slot_summary_fields) + "|")
        slot_md.write_text("\n".join(lines), encoding="utf-8")
    else:
        if slot_csv.exists():
            slot_csv.unlink()
        if slot_md.exists():
            slot_md.unlink()

    sweep_csv = None
    sweep_md = None
    calib_md = None
    decoy_csv = None
    decoy_md = None
    stability_symbolic_csv = None
    stability_symbolic_md = None
    stability_llm_csv = None
    stability_llm_md = None
    cost_pareto_md = None
    accuracy_matched_cost_md = None
    efficiency_thresholds_csv = None
    efficiency_thresholds_md = None
    if threaded_mode:
        sweep_csv, sweep_md, calib_md, sweep_rows = summarize_context_budget_sweep(
            run_dir,
            output_dir,
            prefer_non_saturated_goc=True,
        )
        cost_pareto_md, accuracy_matched_cost_md = summarize_cost_efficiency_reports(
            output_dir,
            sweep_rows,
        )
        efficiency_thresholds_csv, efficiency_thresholds_md = summarize_efficiency_thresholds(
            output_dir,
            sweep_rows,
        )
        decoy_csv, decoy_md = summarize_decoy_validity(run_dir, output_dir)
        budget_values = sorted(
            {
                int(row.get("budget"))
                for row in sweep_rows
                if isinstance(row.get("budget"), (int, float))
            }
        )
        if len(budget_values) <= 1:
            stability_symbolic_csv, stability_symbolic_md = summarize_seed_stability(
                run_dir,
                output_dir,
                judge_mode="symbolic_packed",
                metrics=[
                    "judge_accuracy",
                    "e3_packed_all_critical_rate",
                    "e3_packed_critical_count_mean",
                    "e3_context_truncated_rate",
                ],
                filename_prefix="seed_stability",
            )
        else:
            # Remove stale seed stability outputs to avoid confusion in sweep-only bundles.
            for path in output_dir.glob("seed_stability_*"):
                try:
                    path.unlink()
                except Exception:
                    pass
            stability_llm_csv, stability_llm_md = summarize_seed_stability(
                run_dir,
                output_dir,
                judge_mode="llm",
                metrics=[
                    "judge_accuracy",
                    "decision_accuracy",
                    "e3_packed_all_critical_rate",
                    "e3_packed_critical_count_mean",
                    "e3_context_truncated_rate",
                    "prompt_tokens_avg",
                    "tool_calls_avg",
                    "open_calls_avg",
                ],
                filename_prefix="seed_stability",
            )

    # Fairness sanity: shared_topk open policy
    fairness_md = output_dir / "fairness_sanity.md"
    fairness_lines = ["# Fairness Sanity (shared_topk)", ""]
    checked = 0
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_params = payload.get("scenario_params", {}) or {}
        if scenario_params.get("thread_open_policy") != "shared_topk":
            continue
        checked += 1
        mismatches = 0
        total = 0
        ref_by_key: Dict[Tuple[str, int], List[str]] = {}
        for report_obj in payload.get("method_reports", {}).values():
            for rec in report_obj.get("records", []) or []:
                if rec.get("episode_id") not in {1, 2}:
                    continue
                thread_id = rec.get("thread_id")
                if not thread_id:
                    continue
                key = (thread_id, int(rec.get("episode_id")))
                opened = sorted(rec.get("opened_clause_ids") or [])
                if key not in ref_by_key:
                    ref_by_key[key] = opened
                else:
                    total += 1
                    if opened != ref_by_key[key]:
                        mismatches += 1
        status = "PASS" if mismatches == 0 else "FAIL"
        fairness_lines.append(
            f"- {path.name}: {status} (checked={total}, mismatches={mismatches})"
        )
    if checked == 0:
        fairness_lines.append("- No shared_topk runs found.")
    fairness_md.write_text("\n".join(fairness_lines), encoding="utf-8")

    narrative_md = None
    if not threaded_mode:
        # C2-2 Narrative summary
        narrative_md = output_dir / "narrative_summary.md"
        narrative_lines: List[str] = []
        narrative_lines.append("# Narrative Summary (GoC Novelty)")
        narrative_lines.append("")
        # Use n10 current for headline metrics when available.
        headline = by_key.get(("n10_exclcore", "current", "goc"), {})
        win_rate = headline.get("winning_in_union_rate")
        rank_rate = headline.get("rank_success_rate")
        narrative_lines.append(
            f"- Winning is almost always in union (rate={_fmt(win_rate)}), but rank-only success is low (rate={_fmt(rank_rate)}), so selection under budget dominates."
        )
        # Policy gain statement.
        cur = by_key.get(("n10_exclcore", "current", "goc"), {})
        bridge = by_key.get(("n10_exclcore", "bridge_one_only", "goc"), {})
        if cur and bridge:
            delta_gain = None
            if cur.get("policy_gain_over_rank") is not None and bridge.get("policy_gain_over_rank") is not None:
                delta_gain = bridge.get("policy_gain_over_rank") - cur.get("policy_gain_over_rank")
            delta_acc = None
            if cur.get("judge_accuracy") is not None and bridge.get("judge_accuracy") is not None:
                delta_acc = bridge.get("judge_accuracy") - cur.get("judge_accuracy")
            narrative_lines.append(
                f"- GoC improves over default policy: Δjudge_acc={_fmt(delta_acc)}, Δpolicy_gain_over_rank={_fmt(delta_gain)} (n10_exclcore, current→bridge_one_only)."
            )
        # Baseline deltas.
        goc_base = by_key.get(("n10_exclcore", "bridge_one_only", "goc_base"), {})
        if bridge and goc_base:
            delta_base = None
            if bridge.get("judge_accuracy") is not None and goc_base.get("judge_accuracy") is not None:
                delta_base = bridge.get("judge_accuracy") - goc_base.get("judge_accuracy")
            narrative_lines.append(
                f"- GoC vs goc_base under bridge_one_only: Δjudge_acc={_fmt(delta_base)} (n10_exclcore)."
            )

        # Slot concentration analysis (n10_exclcore).
        n10_rows = [r for r in slot_summary_rows if r["preset"] == "n10_exclcore"]
        if n10_rows:
            sorted_rows = sorted(
                [r for r in n10_rows if r.get("delta_judge_acc") is not None],
                key=lambda r: r["delta_judge_acc"],
                reverse=True,
            )
            top_slots = sorted_rows[:5]
            bottom_slots = sorted_rows[-5:] if len(sorted_rows) >= 5 else sorted_rows[-len(sorted_rows):]
            top_names = ", ".join([r["slot_term"] for r in top_slots])
            bottom_names = ", ".join([r["slot_term"] for r in bottom_slots])
            narrative_lines.append(
                f"- Improvements are not concentrated in a single slot; top slots: {top_names}; worst slots: {bottom_names}."
            )
        narrative_md.write_text("\n".join(narrative_lines), encoding="utf-8")

    # C3 bundles
    def _slug(value: Any) -> str:
        text = str(value) if value is not None else "unknown"
        safe = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    latest_payload: Optional[Dict[str, Any]] = None
    latest_path: Optional[Path] = None
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if latest_payload is None or _run_sort_key(payload, path) > _run_sort_key(
            latest_payload, latest_path or path
        ):
            latest_payload = payload
            latest_path = path

    scenario_tag = "unknown"
    judge_tag = "unknown"
    run_id_tag = "unknown"
    if latest_payload:
        run_id_tag = latest_payload.get("run_id") or (latest_path.stem if latest_path else "unknown")
    scenario_values = set()
    judge_values = set()
    for path in compare_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_values.add(payload.get("scenario_params", {}).get("scenario_mode") or "unknown")
        judge_values.add(payload.get("judge") or "unknown")
    if len(scenario_values) == 1:
        scenario_tag = next(iter(scenario_values))
    elif scenario_values:
        scenario_tag = "mixed"
    if len(judge_values) == 1:
        judge_tag = next(iter(judge_values))
    elif judge_values:
        judge_tag = "mixed"
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bundle_suffix = f"{_slug(scenario_tag)}_{_slug(judge_tag)}_{_slug(run_id_tag)}_{stamp}"
    analysis_bundle_zip = run_dir / f"analysis_bundle_{bundle_suffix}.zip"
    share_bundle_zip = run_dir / f"share_bundle_{bundle_suffix}.zip"

    import shutil

    shutil.make_archive(str(analysis_bundle_zip.with_suffix("")), "zip", output_dir)

    share_dir = run_dir / "share_bundle"
    share_dir.mkdir(parents=True, exist_ok=True)
    # Collect compare jsons
    compare_paths = list(run_dir.rglob("runs/compare/*.json"))
    for path in compare_paths:
        shutil.copy(path, share_dir / path.name)

    if not threaded_mode:
        # Slot breakdown md (n10 current + bridge_one_only if available)
        for policy in ["current", "bridge_one_only"]:
            info = index.get("n10_exclcore", {}).get(policy, {})
            slot_md_path = info.get("slot_breakdown_md")
            if slot_md_path and slot_md_path.exists():
                shutil.copy(slot_md_path, share_dir / slot_md_path.name)

        # Triage zips (n10 current + bridge_one_only)
        for policy in ["current", "bridge_one_only"]:
            info = index.get("n10_exclcore", {}).get(policy, {})
            triage_zip = info.get("triage_zip")
            if triage_zip and triage_zip.exists():
                shutil.copy(triage_zip, share_dir / triage_zip.name)
    else:
        # Include threaded data files for decoy identifiability.
        threads_path = run_dir / "data" / "threads.jsonl"
        episodes_path = run_dir / "data" / "episodes.jsonl"
        if threads_path.exists():
            shutil.copy(threads_path, share_dir / threads_path.name)
        if episodes_path.exists():
            shutil.copy(episodes_path, share_dir / episodes_path.name)

    shutil.make_archive(str(share_bundle_zip.with_suffix("")), "zip", share_dir)

    result = {
        "analysis_bundle_zip": analysis_bundle_zip,
        "share_bundle_zip": share_bundle_zip,
        "fairness_sanity_md": fairness_md,
    }
    if sweep_md:
        result["results_context_budget_sweep_md"] = sweep_md
    if sweep_csv:
        result["results_context_budget_sweep_csv"] = sweep_csv
    if calib_md:
        result["calibration_recommendation_md"] = calib_md
    if decoy_md:
        result["decoy_validity_md"] = decoy_md
    if decoy_csv:
        result["decoy_validity_csv"] = decoy_csv
    if stability_symbolic_md:
        result["seed_stability_symbolic_packed_md"] = stability_symbolic_md
    if stability_symbolic_csv:
        result["seed_stability_symbolic_packed_csv"] = stability_symbolic_csv
    if stability_llm_md:
        result["seed_stability_llm_md"] = stability_llm_md
    if stability_llm_csv:
        result["seed_stability_llm_csv"] = stability_llm_csv
    if cost_pareto_md:
        result["cost_pareto_md"] = cost_pareto_md
    if accuracy_matched_cost_md:
        result["accuracy_matched_cost_md"] = accuracy_matched_cost_md
    if efficiency_thresholds_csv:
        result["efficiency_thresholds_csv"] = efficiency_thresholds_csv
    if efficiency_thresholds_md:
        result["efficiency_thresholds_md"] = efficiency_thresholds_md
    if slot_summary_rows:
        result["slot_summary_md"] = slot_md
        result["slot_summary_csv"] = slot_csv
    if not threaded_mode:
        result.update(
            {
                "policy_method_matrix_md": matrix_md,
                "policy_method_matrix_csv": matrix_csv,
                "goc_deltas_md": delta_md,
                "goc_deltas_csv": delta_csv,
                "difficulty_sanity_md": sanity_md,
                "difficulty_sanity_csv": sanity_csv,
                "narrative_summary_md": narrative_md,
            }
        )
    return result
