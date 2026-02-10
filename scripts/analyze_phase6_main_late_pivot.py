#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "none":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def _load_policyops_runtime():
    repo_root = Path(__file__).resolve().parents[1]
    policyops_src = repo_root / "src"
    if str(policyops_src) not in sys.path:
        sys.path.insert(0, str(policyops_src))
    from policyops.judges import judge_output  # type: ignore
    from policyops.world import load_tasks, load_world  # type: ignore

    return judge_output, load_world, load_tasks


def _find_sweep_csv(config_run_dir: Path) -> Optional[Path]:
    cands = list((config_run_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _find_data_paths(config_run_dir: Path) -> Tuple[Path, Path]:
    world_dir = config_run_dir / "data" / "worlds"
    tasks_path = config_run_dir / "data" / "tasks" / "tasks.jsonl"
    if world_dir.exists() and tasks_path.exists():
        return world_dir, tasks_path
    raise RuntimeError(f"Missing data under {config_run_dir}")


def _read_raw_text(rec: Dict[str, Any]) -> str:
    raw_path = rec.get("raw_path")
    if isinstance(raw_path, str) and raw_path:
        p = Path(raw_path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return str(rec.get("raw_output") or "")


def _extract_opened_clause_ids(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for key in [
        "e3_packed_clause_ids",
        "opened_for_prompt_clause_ids",
        "opened_total_clause_ids",
        "opened_clause_ids",
        "evidence_after_pad",
    ]:
        vals = rec.get(key)
        if not isinstance(vals, list):
            continue
        for item in vals:
            cid = str(item).strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
    return out


def _is_final_episode(rec: Dict[str, Any]) -> bool:
    return _to_int(rec.get("episode_id")) == 3


def _is_pivot(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("is_pivot_task"))


def _mean_bool(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for rec in records:
        b = _to_bool(rec.get(key))
        if b is None:
            continue
        vals.append(1.0 if b else 0.0)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _annotate_record(
    rec: Dict[str, Any],
    task_lookup: Dict[str, Any],
    world: Any,
    judge_output_fn: Any,
) -> Dict[str, Any]:
    out = dict(rec)
    out["correct_updated"] = _to_bool(out.get("judge_correct"))
    out["correct_initial"] = None
    out["stale_by_judge"] = None
    out["pivot_helped_by_update"] = None
    out["pivot_wrong_both"] = None
    out["pivot_right_both"] = None

    if not (_is_final_episode(out) and _is_pivot(out)):
        return out

    task_id = str(out.get("task_id") or "")
    task = task_lookup.get(task_id)
    if task is None:
        return out

    ticket_initial = str(
        out.get("ticket_initial")
        or getattr(task, "ticket_initial", None)
        or getattr(task, "user_ticket", "")
        or ""
    )
    ticket_updated = str(
        out.get("ticket_updated")
        or getattr(task, "ticket_updated", None)
        or getattr(task, "user_ticket", "")
        or ""
    )
    raw_text = _read_raw_text(out)
    opened_clause_ids = _extract_opened_clause_ids(out)

    try:
        initial_eval = judge_output_fn(
            task,
            raw_text,
            world=world,
            judge_name="symbolic_packed",
            ticket_override=ticket_initial,
            opened_clause_ids=opened_clause_ids,
        )
        out["correct_initial"] = _to_bool(initial_eval.get("correct"))
    except Exception:
        out["correct_initial"] = None

    try:
        updated_eval = judge_output_fn(
            task,
            raw_text,
            world=world,
            judge_name="symbolic_packed",
            ticket_override=ticket_updated,
            opened_clause_ids=opened_clause_ids,
        )
        # Keep existing compare outcome as the primary updated-correct label, but
        # expose rejudge result for diagnostics.
        out["correct_updated_rejudged"] = _to_bool(updated_eval.get("correct"))
    except Exception:
        out["correct_updated_rejudged"] = None

    ci = _to_bool(out.get("correct_initial"))
    cu = _to_bool(out.get("correct_updated"))
    if ci is not None and cu is not None:
        out["stale_by_judge"] = bool(ci and not cu)
        out["pivot_helped_by_update"] = bool((not ci) and cu)
        out["pivot_wrong_both"] = bool((not ci) and (not cu))
        out["pivot_right_both"] = bool(ci and cu)
    return out


def _resolve_report_path(report_json_value: str, sweep_csv_path: Path) -> Path:
    raw = Path(str(report_json_value))
    if raw.is_absolute():
        return raw
    return (sweep_csv_path.parent / raw).resolve()


def _compute_row_metrics(
    annotated_records: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    final_records = [r for r in annotated_records if _is_final_episode(r)]
    final_pivot = [r for r in final_records if _is_pivot(r)]
    final_nonpivot = [r for r in final_records if not _is_pivot(r)]

    final_tasks = len(final_records)
    pivot_final_tasks = len(final_pivot)
    return {
        "final_tasks": float(final_tasks),
        "pivot_final_tasks": float(pivot_final_tasks),
        "pivot_rate_final": (
            float(pivot_final_tasks) / float(final_tasks) if final_tasks > 0 else None
        ),
        "final_accuracy": _mean_bool(final_records, "judge_correct"),
        "final_nonpivot_accuracy": _mean_bool(final_nonpivot, "judge_correct"),
        "final_pivot_correct_updated_rate": _mean_bool(final_pivot, "correct_updated"),
        "final_pivot_stale_by_judge_rate": _mean_bool(final_pivot, "stale_by_judge"),
        "final_pivot_wrong_both_rate": _mean_bool(final_pivot, "pivot_wrong_both"),
        "final_pivot_helped_by_update_rate": _mean_bool(
            final_pivot, "pivot_helped_by_update"
        ),
    }


def analyze_phase6(phase6_root: Path, out_dir: Path) -> Tuple[Path, Path, Path, List[Dict[str, Any]]]:
    judge_output_fn, load_world, load_tasks = _load_policyops_runtime()
    runs_root = phase6_root / "runs"
    if not runs_root.exists():
        raise RuntimeError(f"Runs directory not found: {runs_root}")

    summary_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    report_cache: Dict[Path, Dict[str, Any]] = {}
    method_cache: Dict[Tuple[Path, str], List[Dict[str, Any]]] = {}

    for config_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        if config_dir.name.startswith("smoke_"):
            continue
        sweep_csv = _find_sweep_csv(config_dir)
        if sweep_csv is None:
            continue
        world_dir, tasks_path = _find_data_paths(config_dir)
        world = load_world(world_dir)
        tasks = load_tasks(tasks_path)
        task_lookup = {str(t.task_id): t for t in tasks}

        with sweep_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method = str(row.get("method") or "")
                report_json_value = row.get("report_json")
                if not method or not report_json_value:
                    continue
                report_path = _resolve_report_path(str(report_json_value), sweep_csv)
                if not report_path.exists():
                    continue
                if report_path not in report_cache:
                    report_cache[report_path] = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                report_obj = report_cache[report_path]
                method_report = (report_obj.get("method_reports") or {}).get(method) or {}
                records = list(method_report.get("records") or [])
                cache_key = (report_path, method)
                if cache_key not in method_cache:
                    method_cache[cache_key] = [
                        _annotate_record(rec, task_lookup, world, judge_output_fn)
                        for rec in records
                    ]
                annotated_records = method_cache[cache_key]
                computed = _compute_row_metrics(annotated_records)

                scenario_params = report_obj.get("scenario_params") or {}
                pivot_type = scenario_params.get("pivot_type")
                pivot_rate_requested = scenario_params.get("pivot_rate_requested")
                metrics = method_report.get("metrics") or {}

                out_row: Dict[str, Any] = {
                    "config_id": config_dir.name,
                    "pivot_type": pivot_type,
                    "pivot_rate_requested": pivot_rate_requested,
                    "budget": _to_int(row.get("budget")),
                    "method": method,
                    "final_tasks": _to_int(computed.get("final_tasks")),
                    "pivot_final_tasks": _to_int(computed.get("pivot_final_tasks")),
                    "pivot_rate_final": computed.get("pivot_rate_final"),
                    "final_accuracy": computed.get("final_accuracy"),
                    "final_nonpivot_accuracy": computed.get("final_nonpivot_accuracy"),
                    "final_pivot_correct_updated_rate": computed.get(
                        "final_pivot_correct_updated_rate"
                    ),
                    "final_pivot_stale_by_judge_rate": computed.get(
                        "final_pivot_stale_by_judge_rate"
                    ),
                    "final_pivot_wrong_both_rate": computed.get(
                        "final_pivot_wrong_both_rate"
                    ),
                    "final_pivot_helped_by_update_rate": computed.get(
                        "final_pivot_helped_by_update_rate"
                    ),
                    "e3_context_token_est_mean": metrics.get("e3_context_token_est_mean"),
                    "acc_per_1k_tokens": metrics.get("acc_per_1k_tokens"),
                    "cost_per_correct_token_est": metrics.get("cost_per_correct_token_est"),
                    "report_json": str(report_path),
                    "sweep_csv": str(sweep_csv),
                }
                summary_rows.append(out_row)

                for rec in annotated_records:
                    if not (_is_final_episode(rec) and _is_pivot(rec)):
                        continue
                    failure_rows.append(
                        {
                            "config_id": config_dir.name,
                            "pivot_type": pivot_type,
                            "pivot_rate_requested": pivot_rate_requested,
                            "budget": _to_int(row.get("budget")),
                            "method": method,
                            "task_id": rec.get("task_id"),
                            "thread_id": rec.get("thread_id"),
                            "pivot_old_days": rec.get("pivot_old_days"),
                            "pivot_updated_days": rec.get("pivot_updated_days"),
                            "correct_initial": rec.get("correct_initial"),
                            "correct_updated": rec.get("correct_updated"),
                            "stale_by_judge": rec.get("stale_by_judge"),
                            "pivot_helped_by_update": rec.get("pivot_helped_by_update"),
                            "pivot_wrong_both": rec.get("pivot_wrong_both"),
                            "pivot_right_both": rec.get("pivot_right_both"),
                            "prompt_path": rec.get("prompt_path"),
                            "raw_path": rec.get("raw_path"),
                        }
                    )

    summary_rows.sort(
        key=lambda r: (
            str(r.get("config_id") or ""),
            float(_to_float(r.get("budget")) or 0.0),
            str(r.get("method") or ""),
        )
    )
    failure_rows.sort(
        key=lambda r: (
            str(r.get("config_id") or ""),
            float(_to_float(r.get("budget")) or 0.0),
            str(r.get("method") or ""),
            str(r.get("thread_id") or ""),
            str(r.get("task_id") or ""),
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "phase6_summary.csv"
    out_md = out_dir / "phase6_summary.md"
    out_failures = out_dir / "phase6_failures.csv"

    summary_fields = [
        "config_id",
        "pivot_type",
        "pivot_rate_requested",
        "budget",
        "method",
        "final_tasks",
        "pivot_final_tasks",
        "pivot_rate_final",
        "final_accuracy",
        "final_nonpivot_accuracy",
        "final_pivot_correct_updated_rate",
        "final_pivot_stale_by_judge_rate",
        "final_pivot_wrong_both_rate",
        "final_pivot_helped_by_update_rate",
        "e3_context_token_est_mean",
        "acc_per_1k_tokens",
        "cost_per_correct_token_est",
        "report_json",
        "sweep_csv",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in summary_fields})

    failure_fields = [
        "config_id",
        "pivot_type",
        "pivot_rate_requested",
        "budget",
        "method",
        "task_id",
        "thread_id",
        "pivot_old_days",
        "pivot_updated_days",
        "correct_initial",
        "correct_updated",
        "stale_by_judge",
        "pivot_helped_by_update",
        "pivot_wrong_both",
        "pivot_right_both",
        "prompt_path",
        "raw_path",
    ]
    with out_failures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow({k: row.get(k) for k in failure_fields})

    lines: List[str] = []
    lines.append("# Phase6 Late-Pivot Main Summary")
    lines.append("")
    lines.append(
        "| config_id | pivot_type | pivot_rate_requested | budget | method | final_accuracy | final_nonpivot_accuracy | pivot_rate_final | final_pivot_correct_updated_rate | final_pivot_stale_by_judge_rate | final_pivot_wrong_both_rate | final_pivot_helped_by_update_rate |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {config_id} | {pivot_type} | {pivot_rate_requested} | {budget} | {method} | {final_accuracy} | {final_nonpivot_accuracy} | {pivot_rate_final} | {final_pivot_correct_updated_rate} | {final_pivot_stale_by_judge_rate} | {final_pivot_wrong_both_rate} | {final_pivot_helped_by_update_rate} |".format(
                config_id=row.get("config_id"),
                pivot_type=row.get("pivot_type"),
                pivot_rate_requested=_fmt_float(_to_float(row.get("pivot_rate_requested"))),
                budget=row.get("budget"),
                method=row.get("method"),
                final_accuracy=_fmt_float(_to_float(row.get("final_accuracy"))),
                final_nonpivot_accuracy=_fmt_float(
                    _to_float(row.get("final_nonpivot_accuracy"))
                ),
                pivot_rate_final=_fmt_float(_to_float(row.get("pivot_rate_final"))),
                final_pivot_correct_updated_rate=_fmt_float(
                    _to_float(row.get("final_pivot_correct_updated_rate"))
                ),
                final_pivot_stale_by_judge_rate=_fmt_float(
                    _to_float(row.get("final_pivot_stale_by_judge_rate"))
                ),
                final_pivot_wrong_both_rate=_fmt_float(
                    _to_float(row.get("final_pivot_wrong_both_rate"))
                ),
                final_pivot_helped_by_update_rate=_fmt_float(
                    _to_float(row.get("final_pivot_helped_by_update_rate"))
                ),
            )
        )
    lines.append("")
    lines.append(f"- summary_csv: {out_csv}")
    lines.append(f"- failures_csv: {out_failures}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_csv, out_md, out_failures, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase6 late-pivot main runs.")
    parser.add_argument("--phase6_root", required=True, type=Path, help="Path to <bundle>/phase6")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: <phase6_root>/analysis")
    args = parser.parse_args()

    phase6_root = args.phase6_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (phase6_root / "analysis")
    out_csv, out_md, out_failures, rows = analyze_phase6(phase6_root, out_dir)
    print(f"PHASE6_SUMMARY_CSV={out_csv}")
    print(f"PHASE6_SUMMARY_MD={out_md}")
    print(f"PHASE6_FAILURES_CSV={out_failures}")
    print(f"ROWS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

