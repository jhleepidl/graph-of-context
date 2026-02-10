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


def _find_report_json(run_dir: Path) -> Optional[Path]:
    run_dir = run_dir.resolve()
    direct = run_dir.parent / f"{run_dir.name}.json"
    if direct.exists():
        return direct
    candidates = list(run_dir.parent.glob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_data_paths(run_dir: Path) -> Tuple[Path, Path]:
    for parent in [run_dir] + list(run_dir.parents):
        world_dir = parent / "data" / "worlds"
        tasks_path = parent / "data" / "tasks" / "tasks.jsonl"
        if world_dir.exists() and tasks_path.exists():
            return world_dir, tasks_path
    raise RuntimeError(f"Could not locate data/worlds and data/tasks from run_dir={run_dir}")


def _load_policyops_runtime():
    repo_root = Path(__file__).resolve().parents[1]
    policyops_src = repo_root / "src"
    if str(policyops_src) not in sys.path:
        sys.path.insert(0, str(policyops_src))
    from policyops.judges import judge_output  # type: ignore
    from policyops.world import load_tasks, load_world  # type: ignore

    return judge_output, load_world, load_tasks


def _read_raw_text(rec: Dict[str, Any]) -> str:
    raw_path = rec.get("raw_path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    inline = rec.get("raw_output")
    return str(inline or "")


def _extract_opened_clause_ids(rec: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in [
        "e3_packed_clause_ids",
        "opened_for_prompt_clause_ids",
        "opened_total_clause_ids",
        "opened_clause_ids",
        "evidence_after_pad",
    ]:
        vals = rec.get(key)
        if isinstance(vals, list):
            for v in vals:
                s = str(v).strip()
                if s:
                    candidates.append(s)
    out: List[str] = []
    seen = set()
    for cid in candidates:
        if cid in seen:
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
    out["pivot_compliant_text"] = _to_bool(out.get("pivot_compliant"))
    out["stale_evidence_text"] = _to_bool(out.get("stale_evidence"))
    out["correct_updated"] = _to_bool(out.get("judge_correct"))
    out["correct_initial"] = None
    out["correct_updated_rejudged"] = None
    out["stale_by_judge"] = None
    out["pivot_helped_by_update"] = None
    out["pivot_wrong_both"] = None
    out["pivot_right_both"] = None
    out["pivot_compliant_by_judge"] = None

    if not (_is_final_episode(out) and _is_pivot(out)):
        return out

    task_id = str(out.get("task_id") or "")
    task = task_lookup.get(task_id)
    if task is None:
        return out
    raw_text = _read_raw_text(out)
    opened_clause_ids = _extract_opened_clause_ids(out)

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
        out["correct_updated_rejudged"] = _to_bool(updated_eval.get("correct"))
    except Exception:
        out["correct_updated_rejudged"] = None

    correct_initial = _to_bool(out.get("correct_initial"))
    correct_updated = _to_bool(out.get("correct_updated"))
    if correct_initial is not None and correct_updated is not None:
        out["stale_by_judge"] = bool(correct_initial and not correct_updated)
        out["pivot_helped_by_update"] = bool((not correct_initial) and correct_updated)
        out["pivot_wrong_both"] = bool((not correct_initial) and (not correct_updated))
        out["pivot_right_both"] = bool(correct_initial and correct_updated)
        out["pivot_compliant_by_judge"] = bool(correct_updated)
    return out


def _collect_summary_rows(
    report: Dict[str, Any],
    task_lookup: Dict[str, Any],
    world: Any,
    judge_output_fn: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    rows: List[Dict[str, Any]] = []
    annotated_by_method: Dict[str, List[Dict[str, Any]]] = {}

    for method, method_report in sorted((report.get("method_reports") or {}).items()):
        metrics = method_report.get("metrics") or {}
        records = list(method_report.get("records") or [])
        annotated = [
            _annotate_record(rec, task_lookup, world, judge_output_fn) for rec in records
        ]
        annotated_by_method[method] = annotated

        all_records = annotated
        all_pivot = [r for r in all_records if _is_pivot(r)]
        all_nonpivot = [r for r in all_records if not _is_pivot(r)]
        final_records = [r for r in annotated if _is_final_episode(r)]
        final_pivot = [r for r in final_records if _is_pivot(r)]
        final_nonpivot = [r for r in final_records if not _is_pivot(r)]

        final_tasks = len(final_records)
        pivot_final_tasks = len(final_pivot)
        pivot_rate_final = (
            float(pivot_final_tasks) / float(final_tasks) if final_tasks > 0 else None
        )

        row = {
            "method": method,
            "judge_accuracy_packed": _to_float(metrics.get("judge_accuracy_packed")),
            "pivot_compliance_rate": _to_float(metrics.get("pivot_compliance_rate")),
            "stale_evidence_rate": _to_float(metrics.get("stale_evidence_rate")),
            "pivot_rate_actual": _to_float(metrics.get("pivot_rate_actual")),
            "tasks": len(all_records),
            "pivot_tasks": len(all_pivot),
            "final_tasks": final_tasks,
            "pivot_final_tasks": pivot_final_tasks,
            "pivot_rate_final": pivot_rate_final,
            "final_accuracy": _mean_bool(final_records, "judge_correct"),
            "final_pivot_accuracy": _mean_bool(final_pivot, "judge_correct"),
            "final_nonpivot_accuracy": _mean_bool(final_nonpivot, "judge_correct"),
            "final_pivot_correct_updated_rate": _mean_bool(final_pivot, "correct_updated"),
            "final_pivot_correct_initial_rate": _mean_bool(final_pivot, "correct_initial"),
            "final_pivot_stale_by_judge_rate": _mean_bool(final_pivot, "stale_by_judge"),
            "final_pivot_helped_by_update_rate": _mean_bool(final_pivot, "pivot_helped_by_update"),
            "final_pivot_wrong_both_rate": _mean_bool(final_pivot, "pivot_wrong_both"),
            "final_pivot_right_both_rate": _mean_bool(final_pivot, "pivot_right_both"),
            "all_episodes_tasks": len(all_records),
            "all_episodes_pivot_tasks": len(all_pivot),
            "all_episodes_pivot_rate": (
                float(len(all_pivot)) / float(len(all_records)) if all_records else None
            ),
            "all_episodes_accuracy": _mean_bool(all_records, "judge_correct"),
            "all_episodes_pivot_accuracy": _mean_bool(all_pivot, "judge_correct"),
            "all_episodes_nonpivot_accuracy": _mean_bool(all_nonpivot, "judge_correct"),
            "pivot_compliance_rate_text_heuristic": _to_float(
                metrics.get("pivot_compliance_rate")
            ),
            "stale_evidence_rate_text_heuristic": _to_float(
                metrics.get("stale_evidence_rate")
            ),
        }
        rows.append(row)

    return rows, annotated_by_method


def _collect_failure_sections(
    annotated_by_method: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    section_updated_correct: List[Dict[str, Any]] = []
    section_stale_by_judge: List[Dict[str, Any]] = []
    section_wrong_both: List[Dict[str, Any]] = []

    for method, records in sorted(annotated_by_method.items()):
        for rec in records:
            if not (_is_final_episode(rec) and _is_pivot(rec)):
                continue
            row = {
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
                "pivot_compliant_by_judge": rec.get("pivot_compliant_by_judge"),
                "pivot_compliant_text": rec.get("pivot_compliant_text"),
                "stale_evidence_text": rec.get("stale_evidence_text"),
                "prompt_path": rec.get("prompt_path"),
                "raw_path": rec.get("raw_path"),
            }
            if _to_bool(rec.get("correct_updated")) is True:
                section_updated_correct.append(row)
            if _to_bool(rec.get("stale_by_judge")) is True:
                section_stale_by_judge.append(row)
            if _to_bool(rec.get("pivot_wrong_both")) is True:
                section_wrong_both.append(row)

    def _sort(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows.sort(
            key=lambda r: (
                str(r.get("method") or ""),
                str(r.get("thread_id") or ""),
                str(r.get("task_id") or ""),
            )
        )
        return rows

    return {
        "updated_correct": _sort(section_updated_correct),
        "stale_by_judge": _sort(section_stale_by_judge),
        "wrong_both": _sort(section_wrong_both),
    }


def _collect_pivot_task_rows(
    annotated_by_method: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, records in sorted(annotated_by_method.items()):
        for rec in records:
            if not (_is_final_episode(rec) and _is_pivot(rec)):
                continue
            rows.append(
                {
                    "method": method,
                    "task_id": rec.get("task_id"),
                    "thread_id": rec.get("thread_id"),
                    "pivot_old_days": rec.get("pivot_old_days"),
                    "pivot_updated_days": rec.get("pivot_updated_days"),
                    "correct_initial": rec.get("correct_initial"),
                    "correct_updated": rec.get("correct_updated"),
                    "stale_by_judge": rec.get("stale_by_judge"),
                    "pivot_helped_by_update": rec.get("pivot_helped_by_update"),
                    "prompt_path": rec.get("prompt_path"),
                    "raw_path": rec.get("raw_path"),
                }
            )
    rows.sort(
        key=lambda r: (
            str(r.get("method") or ""),
            str(r.get("thread_id") or ""),
            str(r.get("task_id") or ""),
        )
    )
    return rows


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "judge_accuracy_packed",
        "pivot_compliance_rate",
        "stale_evidence_rate",
        "pivot_rate_actual",
        "tasks",
        "pivot_tasks",
        "final_tasks",
        "pivot_final_tasks",
        "pivot_rate_final",
        "final_accuracy",
        "final_pivot_accuracy",
        "final_nonpivot_accuracy",
        "final_pivot_correct_updated_rate",
        "final_pivot_correct_initial_rate",
        "final_pivot_stale_by_judge_rate",
        "final_pivot_helped_by_update_rate",
        "final_pivot_wrong_both_rate",
        "final_pivot_right_both_rate",
        "all_episodes_tasks",
        "all_episodes_pivot_tasks",
        "all_episodes_pivot_rate",
        "all_episodes_accuracy",
        "all_episodes_pivot_accuracy",
        "all_episodes_nonpivot_accuracy",
        "pivot_compliance_rate_text_heuristic",
        "stale_evidence_rate_text_heuristic",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_failures_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task_id",
        "thread_id",
        "pivot_old_days",
        "pivot_updated_days",
        "correct_initial",
        "correct_updated",
        "stale_by_judge",
        "pivot_helped_by_update",
        "prompt_path",
        "raw_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _render_failure_lines(rows: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    if not rows:
        return ["- none"]
    out: List[str] = []
    for row in rows[:limit]:
        out.append(
            "- [{method}] task_id={task_id} thread_id={thread_id} old={old} updated={updated} "
            "prompt={prompt} raw={raw}".format(
                method=row.get("method"),
                task_id=row.get("task_id"),
                thread_id=row.get("thread_id"),
                old=row.get("pivot_old_days"),
                updated=row.get("pivot_updated_days"),
                prompt=row.get("prompt_path"),
                raw=row.get("raw_path"),
            )
        )
    return out


def _write_summary_md(
    path: Path,
    rows: List[Dict[str, Any]],
    sections: Dict[str, List[Dict[str, Any]]],
) -> None:
    lines: List[str] = []
    lines.append("# Phase5 Pilot Late-Pivot Summary")
    lines.append("")
    lines.append("## Primary Pivot Metrics (Judge-Based, Final Episode Only)")
    lines.append("")
    lines.append(
        "| method | final_tasks | pivot_final_tasks | pivot_rate_final | final_accuracy | final_nonpivot_accuracy | final_pivot_correct_updated_rate | final_pivot_correct_initial_rate | final_pivot_stale_by_judge_rate | final_pivot_helped_by_update_rate | final_pivot_wrong_both_rate | final_pivot_right_both_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {final_tasks} | {pivot_final_tasks} | {pivot_rate_final} | {final_accuracy} | "
            "{final_nonpivot_accuracy} | {updated} | {initial} | {stale} | {helped} | {wrong_both} | {right_both} |".format(
                method=row.get("method"),
                final_tasks=row.get("final_tasks"),
                pivot_final_tasks=row.get("pivot_final_tasks"),
                pivot_rate_final=_fmt_float(_to_float(row.get("pivot_rate_final"))),
                final_accuracy=_fmt_float(_to_float(row.get("final_accuracy"))),
                final_nonpivot_accuracy=_fmt_float(_to_float(row.get("final_nonpivot_accuracy"))),
                updated=_fmt_float(_to_float(row.get("final_pivot_correct_updated_rate"))),
                initial=_fmt_float(_to_float(row.get("final_pivot_correct_initial_rate"))),
                stale=_fmt_float(_to_float(row.get("final_pivot_stale_by_judge_rate"))),
                helped=_fmt_float(_to_float(row.get("final_pivot_helped_by_update_rate"))),
                wrong_both=_fmt_float(_to_float(row.get("final_pivot_wrong_both_rate"))),
                right_both=_fmt_float(_to_float(row.get("final_pivot_right_both_rate"))),
            )
        )
    lines.append("")
    lines.append("## Compatibility Metrics (All-Episodes + Text-Heuristic)")
    lines.append("")
    lines.append(
        "| method | judge_accuracy_packed | pivot_compliance_rate_text_heuristic | stale_evidence_rate_text_heuristic | pivot_rate_actual |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {judge_accuracy_packed} | {pivot_compliance_rate_text_heuristic} | "
            "{stale_evidence_rate_text_heuristic} | {pivot_rate_actual} |".format(
                method=row.get("method"),
                judge_accuracy_packed=_fmt_float(_to_float(row.get("judge_accuracy_packed"))),
                pivot_compliance_rate_text_heuristic=_fmt_float(
                    _to_float(row.get("pivot_compliance_rate_text_heuristic"))
                ),
                stale_evidence_rate_text_heuristic=_fmt_float(
                    _to_float(row.get("stale_evidence_rate_text_heuristic"))
                ),
                pivot_rate_actual=_fmt_float(_to_float(row.get("pivot_rate_actual"))),
            )
        )
    lines.append("")
    lines.append("## Failure Listing (Final Pivot Tasks)")
    lines.append("")
    lines.append(
        f"### (1) Updated-correct (episode3 & is_pivot_task & correct_updated==True) [{len(sections['updated_correct'])}]"
    )
    lines.extend(_render_failure_lines(sections["updated_correct"], limit=10))
    lines.append("")
    lines.append(
        f"### (2) Stale-by-judge (correct_initial && !correct_updated) [{len(sections['stale_by_judge'])}]"
    )
    lines.extend(_render_failure_lines(sections["stale_by_judge"], limit=10))
    lines.append("")
    lines.append(
        f"### (3) Wrong-both (!correct_initial && !correct_updated) [{len(sections['wrong_both'])}]"
    )
    lines.extend(_render_failure_lines(sections["wrong_both"], limit=10))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase5 late-pivot outputs with judge-based stale metrics.")
    parser.add_argument(
        "--run_dir",
        required=True,
        type=Path,
        help="Compare run dir (e.g., .../runs/compare/<run_id>)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/phase5_pilot_analysis)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "phase5_pilot_analysis")
    report_json = _find_report_json(run_dir)
    if report_json is None:
        raise RuntimeError(f"Could not find compare report JSON for run_dir={run_dir}")
    report = json.loads(report_json.read_text(encoding="utf-8"))

    judge_output_fn, load_world, load_tasks = _load_policyops_runtime()
    world_dir, tasks_path = _find_data_paths(run_dir)
    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)
    task_lookup = {str(t.task_id): t for t in tasks}

    summary_rows, annotated_by_method = _collect_summary_rows(
        report, task_lookup, world, judge_output_fn
    )
    sections = _collect_failure_sections(annotated_by_method)
    pivot_task_rows = _collect_pivot_task_rows(annotated_by_method)

    out_csv = out_dir / "phase5_pilot_summary.csv"
    out_md = out_dir / "phase5_pilot_summary.md"
    out_failures_csv = out_dir / "phase5_pilot_failures.csv"
    _write_summary_csv(out_csv, summary_rows)
    _write_summary_md(out_md, summary_rows, sections)
    _write_failures_csv(out_failures_csv, pivot_task_rows)

    print(f"PHASE5_PILOT_SUMMARY_CSV={out_csv}")
    print(f"PHASE5_PILOT_SUMMARY_MD={out_md}")
    print(f"PHASE5_PILOT_FAILURES_CSV={out_failures_csv}")
    print(f"PIVOT_FINAL_ROWS={len(pivot_task_rows)}")
    for row in summary_rows:
        print(
            "METHOD={method} final_pivot_correct_updated_rate={updated} "
            "final_pivot_stale_by_judge_rate={stale}".format(
                method=row.get("method"),
                updated=row.get("final_pivot_correct_updated_rate"),
                stale=row.get("final_pivot_stale_by_judge_rate"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
