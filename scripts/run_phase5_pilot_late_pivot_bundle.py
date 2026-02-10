#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import string
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from policyops_bundle_layout import build_bundle_quick_access


PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rand8() -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(8))


def _run_command(cmd: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    extra = "src"
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{py_path}" if py_path else extra
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def _load_thread_counts(tasks_jsonl: Path) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = str(obj.get("thread_id") or f"__task__:{obj.get('task_id', 'unknown')}")
            counts[tid] = counts.get(tid, 0) + 1
    ordered = sorted(counts.keys())
    return ordered, counts


def _choose_k_for_target(
    ordered_threads: List[str],
    counts: Dict[str, int],
    *,
    low: int = 45,
    high: int = 55,
    target: int = 50,
) -> Tuple[int, int]:
    total = 0
    for idx, tid in enumerate(ordered_threads, start=1):
        total += int(counts.get(tid, 0))
        if low <= total <= high:
            return idx, total
    total = 0
    best = (0, 0, 10**9)
    for idx, tid in enumerate(ordered_threads, start=1):
        total += int(counts.get(tid, 0))
        if total > high:
            break
        dist = abs(total - target)
        if dist < best[2]:
            best = (idx, total, dist)
    if best[0] > 0:
        return best[0], best[1]
    total_all = sum(int(counts.get(tid, 0)) for tid in ordered_threads)
    return len(ordered_threads), total_all


def _discover_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    compare_jsons = list((out_dir / "runs" / "compare").glob("*.json"))
    report_json = max(compare_jsons, key=lambda p: p.stat().st_mtime) if compare_jsons else None
    run_dir = None
    if report_json is not None:
        candidate = out_dir / "runs" / "compare" / report_json.stem
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        run_dirs = [p for p in (out_dir / "runs" / "compare").glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    sweep_csvs = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(sweep_csvs, key=lambda p: p.stat().st_mtime) if sweep_csvs else None
    graphs_internal = None
    if run_dir is not None:
        candidates = list((run_dir / "goc" / "graphs_internal").glob("*.jsonl"))
        if candidates:
            graphs_internal = max(candidates, key=lambda p: p.stat().st_mtime)
    return report_json, run_dir, sweep_csv, graphs_internal


def _zip_dir(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(root.parent))
                zf.write(str(full), arcname=arc)


def _read_summary_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _symlink_or_copy_data(source_data_dir: Path, target_data_dir: Path) -> None:
    if target_data_dir.exists() or target_data_dir.is_symlink():
        if target_data_dir.is_symlink() or target_data_dir.is_file():
            target_data_dir.unlink()
        else:
            shutil.rmtree(target_data_dir)
    try:
        target_data_dir.symlink_to(source_data_dir.resolve(), target_is_directory=True)
    except Exception:
        shutil.copytree(source_data_dir, target_data_dir, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase5 pilot late-pivot experiment bundle.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pivot_rate", type=float, default=0.5)
    parser.add_argument("--generate_n_threads", type=int, default=40)
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = f"goc_policyops_phase5_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase5_root = bundle_root / "phase5"
    data_root = phase5_root / "data"
    runs_root = phase5_root / "runs"
    analysis_root = phase5_root / "analysis"
    data_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    generate_out = data_root / "pilot_dataset"
    generate_log = phase5_root / "generate.log"
    generate_cmd = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "generate",
        "--preset",
        PRESET,
        "--seed",
        str(args.seed),
        "--n_threads",
        str(args.generate_n_threads),
        "--pivot_rate",
        str(args.pivot_rate),
        "--pivot_type",
        "retention_flip",
        "--out_dir",
        str(generate_out),
    ]
    rc_generate = _run_command(generate_cmd, cwd=repo_root, log_path=generate_log)
    if rc_generate != 0:
        raise RuntimeError(f"generate failed: rc={rc_generate}, log={generate_log}")

    tasks_jsonl = generate_out / "data" / "tasks" / "tasks.jsonl"
    if not tasks_jsonl.exists():
        raise RuntimeError(f"tasks.jsonl missing: {tasks_jsonl}")
    ordered_threads, thread_counts = _load_thread_counts(tasks_jsonl)
    k_threads, total_tasks = _choose_k_for_target(ordered_threads, thread_counts)

    compare_out = runs_root / "pilot_compare"
    _symlink_or_copy_data(generate_out / "data", compare_out / "data")
    compare_log = phase5_root / "compare.log"
    compare_cmd = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "compare",
        "--preset",
        PRESET,
        "--llm",
        "openai",
        "--model",
        args.model,
        "--judge",
        "symbolic_packed",
        "--methods",
        "full",
        "similarity_only",
        "goc",
        "--thread_context_budget_sweep",
        "1600",
        "--parallel_workers",
        str(args.parallel_workers),
        "--n_threads",
        str(k_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "1.0",
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "1.0",
        "--goc_activity_filter",
        "--goc_mmr_lambda",
        "0.35",
        "--no_goc_anchor_top1_lexical",
        "--goc_budget_follow_sweep",
        "--goc_budget_chars_to_tokens_divisor",
        "4",
        "--dotenv",
        args.dotenv,
        "--out_dir",
        str(compare_out),
    ]
    rc_compare = _run_command(compare_cmd, cwd=repo_root, log_path=compare_log)
    if rc_compare != 0:
        raise RuntimeError(f"compare failed: rc={rc_compare}, log={compare_log}")

    report_json, run_dir, sweep_csv, graphs_internal = _discover_artifacts(compare_out)
    if report_json is None:
        raise RuntimeError("compare report json missing")
    if sweep_csv is None:
        raise RuntimeError("results_context_budget_sweep.csv missing")
    if run_dir is None:
        raise RuntimeError("compare run dir missing")
    if graphs_internal is None:
        raise RuntimeError("graphs_internal jsonl missing")

    analyze_log = phase5_root / "analyze.log"
    analyze_cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_phase5_pilot_late_pivot.py",
        "--run_dir",
        str(run_dir),
        "--out_dir",
        str(analysis_root),
    ]
    rc_analyze = _run_command(analyze_cmd, cwd=repo_root, log_path=analyze_log)
    if rc_analyze != 0:
        raise RuntimeError(f"analysis failed: rc={rc_analyze}, log={analyze_log}")

    summary_csv = analysis_root / "phase5_pilot_summary.csv"
    summary_md = analysis_root / "phase5_pilot_summary.md"
    if not summary_csv.exists() or not summary_md.exists():
        raise RuntimeError("analysis outputs missing")
    summary_rows = _read_summary_csv(summary_csv)

    index_md = bundle_root / "INDEX.md"
    lines: List[str] = []
    lines.append(f"# {run_id}")
    lines.append("")
    lines.append("## Run Config")
    lines.append(f"- preset: {PRESET}")
    lines.append(f"- model: {args.model}")
    lines.append(f"- pivot_rate: {args.pivot_rate}")
    lines.append(f"- pivot_type: retention_flip")
    lines.append(f"- parallel_workers: {args.parallel_workers}")
    lines.append(f"- n_threads_selected: {k_threads}")
    lines.append(f"- total_tasks_selected: {total_tasks}")
    lines.append("")
    lines.append("## Key Artifacts")
    lines.append(f"- generate_log: {generate_log}")
    lines.append(f"- compare_log: {compare_log}")
    lines.append(f"- compare_report_json: {report_json}")
    lines.append(f"- sweep_csv: {sweep_csv}")
    lines.append(f"- graphs_internal_sample: {graphs_internal}")
    lines.append(f"- summary_csv: {summary_csv}")
    lines.append(f"- summary_md: {summary_md}")
    lines.append("")
    lines.append("## Metrics (per method)")
    if summary_rows:
        for row in summary_rows:
            lines.append(
                "- {method}: judge_accuracy_packed={judge_accuracy_packed}, "
                "pivot_compliance_rate={pivot_compliance_rate}, "
                "stale_evidence_rate={stale_evidence_rate}, "
                "pivot_rate_actual={pivot_rate_actual}".format(**row)
            )
    else:
        lines.append("- no summary rows")
    index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = bundle_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"K_THREADS={k_threads}")
    print(f"TOTAL_TASKS={total_tasks}")
    for row in summary_rows:
        print(
            "METRIC method={method} acc={judge_accuracy_packed} pivot_comp={pivot_compliance_rate} "
            "stale={stale_evidence_rate} pivot_rate={pivot_rate_actual}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
