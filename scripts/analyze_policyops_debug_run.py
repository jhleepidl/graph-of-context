#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TaskMeta:
    thread_id: Optional[str]
    episode_id: Optional[int]
    core_clause_ids: List[str]


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return None


def _safe_list_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return out


def _find_tasks_jsonl(run_dir: Path) -> Optional[Path]:
    for parent in [run_dir] + list(run_dir.parents):
        candidate = parent / "data" / "tasks" / "tasks.jsonl"
        if candidate.exists():
            return candidate
    return None


def _load_task_meta(tasks_jsonl: Optional[Path]) -> Dict[str, TaskMeta]:
    if tasks_jsonl is None or not tasks_jsonl.exists():
        return {}
    out: Dict[str, TaskMeta] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = str(obj.get("task_id") or "")
            if not task_id:
                continue
            thread_id = obj.get("thread_id")
            if thread_id is not None:
                thread_id = str(thread_id)
            episode_id_raw = obj.get("episode_id")
            episode_id: Optional[int]
            if isinstance(episode_id_raw, int):
                episode_id = episode_id_raw
            else:
                episode_id = None
            core = _safe_list_str(obj.get("critical_core_clause_ids"))
            if not core:
                gold = obj.get("gold") if isinstance(obj.get("gold"), dict) else {}
                core = _safe_list_str(gold.get("gold_evidence_core"))
            out[task_id] = TaskMeta(
                thread_id=thread_id,
                episode_id=episode_id,
                core_clause_ids=core,
            )
    return out


def _parse_trace_file(path: Path) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "task_id": path.stem,
        "method": None,
        "trace_path": str(path),
        "thread_id": None,
        "episode_id": None,
        "prompt_path": None,
        "raw_path": None,
        "judge_correct": None,
        "packed_all_critical": None,
        "selected_clause_ids": [],
        "unfolded_clause_ids": [],
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = ev.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            task_id = ev.get("task_id")
            if isinstance(task_id, str) and task_id:
                parsed["task_id"] = task_id
            method = ev.get("method")
            if isinstance(method, str) and method:
                parsed["method"] = method
            event_type = ev.get("event_type") or ev.get("type") or ""
            event_type = str(event_type).upper()
            if event_type == "INIT":
                if payload.get("thread_id") is not None:
                    parsed["thread_id"] = str(payload.get("thread_id"))
                episode_id_raw = payload.get("episode_id")
                if isinstance(episode_id_raw, int):
                    parsed["episode_id"] = episode_id_raw
            elif event_type == "PROMPT":
                prompt_path = payload.get("prompt_path")
                if isinstance(prompt_path, str) and prompt_path:
                    parsed["prompt_path"] = prompt_path
            elif event_type == "PREDICTION":
                raw_path = payload.get("raw_path")
                if isinstance(raw_path, str) and raw_path:
                    parsed["raw_path"] = raw_path
                parsed["judge_correct"] = _coerce_bool(payload.get("judge_correct"))
                pac = payload.get("packed_all_critical")
                if pac is None:
                    pac = payload.get("judge_correct_packed_allcritical")
                parsed["packed_all_critical"] = _coerce_bool(pac)
            elif event_type == "SNAPSHOT":
                selected = _safe_list_str(payload.get("selected_clause_ids"))
                unfolded = _safe_list_str(payload.get("unfolded_clause_ids"))
                if selected:
                    parsed["selected_clause_ids"] = selected
                if unfolded:
                    parsed["unfolded_clause_ids"] = unfolded
    return parsed


def _collect_method_traces(run_dir: Path, method: str) -> Dict[str, Dict[str, Any]]:
    trace_root = run_dir / method / "event_traces"
    if not trace_root.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for trace_path in sorted(trace_root.rglob("*.jsonl")):
        parsed = _parse_trace_file(trace_path)
        task_id = str(parsed.get("task_id") or trace_path.stem)
        if not task_id:
            continue
        out[task_id] = parsed
    return out


def _p90(values: List[int]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = max(0, int(math.ceil(len(vals) * 0.9)) - 1)
    return float(vals[idx])


def _dist(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    vals = [int(v) for v in values]
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "median": float(median(vals)),
        "p90": _p90(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _write_failures_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "task_id",
        "thread_id",
        "episode_id",
        "sim_prompt_path",
        "sim_raw_path",
        "sim_trace_path",
        "goc_prompt_path",
        "goc_raw_path",
        "goc_trace_path",
        "selected_clause_ids_count",
        "unfolded_clause_ids_count",
        "missing_core_clause_ids_count",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _fmt_dist(d: Dict[str, Optional[float]]) -> str:
    if not d or not d.get("n"):
        return "n=0"
    return (
        f"n={int(d['n'])}, mean={d['mean']:.2f}, median={d['median']:.2f}, "
        f"p90={d['p90']:.2f}, min={d['min']:.0f}, max={d['max']:.0f}"
    )


def _write_failures_md(
    out_md: Path,
    *,
    total_tasks: int,
    failure_rows: List[Dict[str, Any]],
    goc_only_correct_count: int,
    missing_core_counter: Counter[str],
    selected_fail_dist: Dict[str, Optional[float]],
    unfolded_fail_dist: Dict[str, Optional[float]],
    selected_success_dist: Dict[str, Optional[float]],
    unfolded_success_dist: Dict[str, Optional[float]],
    core_available_tasks: int,
) -> None:
    lines: List[str] = []
    lines.append("# PolicyOps Debug Failure Analysis")
    lines.append("")
    lines.append(f"- total tasks: {total_tasks}")
    lines.append(f"- failure cases (sim true, goc false): {len(failure_rows)}")
    lines.append(f"- goc-only-correct cases (sim false, goc true): {goc_only_correct_count}")
    lines.append(f"- tasks with available core clause ids: {core_available_tasks}")
    lines.append("")
    lines.append("## Top Missing Core Clause IDs")
    if missing_core_counter:
        for cid, count in missing_core_counter.most_common(10):
            lines.append(f"- {cid}: {count}")
    else:
        lines.append("- none (core ids unavailable or no missing core ids)")
    lines.append("")
    lines.append("## GOC Count Distributions")
    lines.append("")
    lines.append("| Group | selected_clause_ids_count | unfolded_clause_ids_count |")
    lines.append("|---|---:|---:|")
    lines.append(f"| failures | {_fmt_dist(selected_fail_dist)} | {_fmt_dist(unfolded_fail_dist)} |")
    lines.append(f"| successes | {_fmt_dist(selected_success_dist)} | {_fmt_dist(unfolded_success_dist)} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir: Path, out_dir: Optional[Path]) -> Tuple[Path, Path, Path, int, int]:
    run_dir = run_dir.resolve()
    if out_dir is None:
        out_dir = run_dir / "failure_analysis"
    out_dir = out_dir.resolve()
    tasks_jsonl = _find_tasks_jsonl(run_dir)
    task_meta = _load_task_meta(tasks_jsonl)
    sim_map = _collect_method_traces(run_dir, "similarity_only")
    goc_map = _collect_method_traces(run_dir, "goc")

    all_task_ids = sorted(set(sim_map.keys()) | set(goc_map.keys()))
    failure_rows: List[Dict[str, Any]] = []
    missing_core_counter: Counter[str] = Counter()
    core_available_tasks = 0
    goc_only_correct_count = 0
    selected_fail_counts: List[int] = []
    unfolded_fail_counts: List[int] = []
    selected_success_counts: List[int] = []
    unfolded_success_counts: List[int] = []

    for task_id in all_task_ids:
        sim = sim_map.get(task_id, {})
        goc = goc_map.get(task_id, {})
        sim_ok = _coerce_bool(sim.get("judge_correct"))
        goc_ok = _coerce_bool(goc.get("judge_correct"))
        selected_ids = _safe_list_str(goc.get("selected_clause_ids"))
        unfolded_ids = _safe_list_str(goc.get("unfolded_clause_ids"))
        selected_count = len(selected_ids)
        unfolded_count = len(unfolded_ids)
        if goc_ok is True:
            selected_success_counts.append(selected_count)
            unfolded_success_counts.append(unfolded_count)
        elif goc_ok is False:
            selected_fail_counts.append(selected_count)
            unfolded_fail_counts.append(unfolded_count)
        if sim_ok is False and goc_ok is True:
            goc_only_correct_count += 1

        if not (sim_ok is True and goc_ok is False):
            continue

        meta = task_meta.get(task_id)
        thread_id = sim.get("thread_id") or goc.get("thread_id")
        episode_id = sim.get("episode_id") or goc.get("episode_id")
        if meta is not None:
            if meta.thread_id is not None:
                thread_id = meta.thread_id
            if meta.episode_id is not None:
                episode_id = meta.episode_id
        missing_core_count: Optional[int] = None
        if meta is not None and meta.core_clause_ids:
            core_available_tasks += 1
            covered = set(selected_ids) | set(unfolded_ids)
            missing_core = sorted(set(meta.core_clause_ids) - covered)
            missing_core_count = len(missing_core)
            for cid in missing_core:
                missing_core_counter[cid] += 1
        row = {
            "task_id": task_id,
            "thread_id": thread_id,
            "episode_id": episode_id,
            "sim_prompt_path": sim.get("prompt_path"),
            "sim_raw_path": sim.get("raw_path"),
            "sim_trace_path": sim.get("trace_path"),
            "goc_prompt_path": goc.get("prompt_path"),
            "goc_raw_path": goc.get("raw_path"),
            "goc_trace_path": goc.get("trace_path"),
            "selected_clause_ids_count": selected_count,
            "unfolded_clause_ids_count": unfolded_count,
            "missing_core_clause_ids_count": missing_core_count,
        }
        failure_rows.append(row)

    failures_csv = out_dir / "failures.csv"
    failures_md = out_dir / "failures.md"
    summary_json = out_dir / "summary.json"
    _write_failures_csv(failure_rows, failures_csv)
    _write_failures_md(
        failures_md,
        total_tasks=len(all_task_ids),
        failure_rows=failure_rows,
        goc_only_correct_count=goc_only_correct_count,
        missing_core_counter=missing_core_counter,
        selected_fail_dist=_dist(selected_fail_counts),
        unfolded_fail_dist=_dist(unfolded_fail_counts),
        selected_success_dist=_dist(selected_success_counts),
        unfolded_success_dist=_dist(unfolded_success_counts),
        core_available_tasks=core_available_tasks,
    )
    summary = {
        "run_dir": str(run_dir),
        "tasks_jsonl": str(tasks_jsonl) if tasks_jsonl else None,
        "total_tasks": len(all_task_ids),
        "failure_cases": len(failure_rows),
        "goc_only_correct_cases": goc_only_correct_count,
        "core_available_tasks": core_available_tasks,
        "files": {
            "failures_csv": str(failures_csv),
            "failures_md": str(failures_md),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        failures_csv,
        failures_md,
        summary_json,
        int(summary["failure_cases"]),
        int(summary["goc_only_correct_cases"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze PolicyOps debug run failures from event traces.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Compare run directory (e.g. runs/compare/<run_id>).")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: <run_dir>/failure_analysis).")
    args = parser.parse_args()
    failures_csv, failures_md, summary_json, failure_cases, goc_only_correct = analyze(
        args.run_dir, args.out_dir
    )
    print(f"FAILURES_CSV={failures_csv}")
    print(f"FAILURES_MD={failures_md}")
    print(f"SUMMARY_JSON={summary_json}")
    print(f"FAILURE_CASES={failure_cases}")
    print(f"GOC_ONLY_CORRECT={goc_only_correct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
