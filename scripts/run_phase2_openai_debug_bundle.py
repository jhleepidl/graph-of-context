#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"
METHODS = ["similarity_only", "goc"]
BUDGET = "2000"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rand8() -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(8))


def _rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return "None"
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _find_latest_phase1_data(repo_root: Path) -> Optional[Path]:
    candidates = [
        p
        for p in (repo_root / "experiment_bundles").glob("**/phase1_dummy/data")
        if p.is_dir() and (p / "tasks" / "tasks.jsonl").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_thread_counts(tasks_jsonl: Path) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            thread_id = str(obj.get("thread_id") or "")
            if not thread_id:
                thread_id = f"__task__:{obj.get('task_id', 'unknown')}"
            counts[thread_id] = counts.get(thread_id, 0) + 1
    ordered = sorted(counts.keys())
    return ordered, counts


def _choose_k_main(
    ordered_threads: List[str],
    counts: Dict[str, int],
    lo: int = 80,
    hi: int = 120,
    target: int = 100,
) -> Tuple[int, int]:
    running = 0
    for idx, thread_id in enumerate(ordered_threads, start=1):
        running += int(counts.get(thread_id, 0))
        if lo <= running <= hi:
            return idx, running

    running = 0
    best = (0, 0, 10**9)
    for idx, thread_id in enumerate(ordered_threads, start=1):
        running += int(counts.get(thread_id, 0))
        if running > hi:
            break
        dist = abs(running - target)
        if dist < best[2]:
            best = (idx, running, dist)
    if best[0] > 0:
        return best[0], best[1]
    total_all = sum(int(counts.get(tid, 0)) for tid in ordered_threads)
    return len(ordered_threads), total_all


def _run_command(cmd: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    extra = "src/benchmarks/policyops_arena_v0/src:src"
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


def _find_latest_file(base: Path, pattern: str) -> Optional[Path]:
    files = list(base.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                full = Path(root) / name
                arcname = str(full.relative_to(src_dir.parent))
                zf.write(str(full), arcname=arcname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused Phase2 debug compare and bundle outputs.")
    parser.add_argument("--phase1_data_dir", type=str, default="", help="Optional path to phase1 data directory.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = f"goc_policyops_phase2_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase2_root = bundle_root / "phase2_openai"
    debug_out = phase2_root / "debug"
    debug_out.mkdir(parents=True, exist_ok=True)

    if args.phase1_data_dir:
        phase1_data = Path(args.phase1_data_dir)
    else:
        phase1_data = _find_latest_phase1_data(repo_root)
    if phase1_data is None or not phase1_data.exists():
        raise RuntimeError("Could not find phase1 data snapshot; provide --phase1_data_dir.")

    shutil.copytree(phase1_data, phase2_root / "data_source_snapshot", dirs_exist_ok=True)
    shutil.copytree(phase1_data, debug_out / "data", dirs_exist_ok=True)

    tasks_jsonl = debug_out / "data" / "tasks" / "tasks.jsonl"
    ordered_threads, counts = _load_thread_counts(tasks_jsonl)
    if not ordered_threads:
        raise RuntimeError(f"No tasks found in {tasks_jsonl}")
    n_threads, total_tasks = _choose_k_main(ordered_threads, counts, lo=80, hi=120, target=100)

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
        *METHODS,
        "--thread_context_budget_sweep",
        BUDGET,
        "--parallel_workers",
        str(args.parallel_workers),
        "--n_threads",
        str(n_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "1.0",
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "1.0",
        "--goc_budget_follow_sweep",
        "--goc_budget_chars_to_tokens_divisor",
        "4",
        "--goc_activity_filter",
        "--goc_activity_debug_in_snapshot",
        "--dotenv",
        args.dotenv,
        "--out_dir",
        str(debug_out),
    ]
    compare_log = debug_out / "compare_sweep_openai.log"
    compare_rc = _run_command(compare_cmd, cwd=repo_root, log_path=compare_log)

    compare_root = debug_out / "runs" / "compare"
    report_json = _find_latest_file(compare_root, "*.json")
    run_dir: Optional[Path] = None
    if report_json is not None:
        candidate = compare_root / report_json.stem
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        run_dirs = [p for p in compare_root.glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    sweep_csv = _find_latest_file(
        debug_out / "runs" / "context_budget_sweep",
        "*/results_context_budget_sweep.csv",
    )
    graphs_internal_sample = None
    if run_dir is not None:
        graphs_internal_sample = _find_latest_file(run_dir / "goc" / "graphs_internal", "*.jsonl")

    if compare_rc != 0 or report_json is None or sweep_csv is None or run_dir is None:
        raise RuntimeError(
            "compare failed or expected outputs missing: "
            f"rc={compare_rc} report_json={report_json} sweep_csv={sweep_csv} run_dir={run_dir}"
        )

    analysis_log = debug_out / "analysis.log"
    analysis_cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_policyops_debug_run.py",
        "--run_dir",
        str(run_dir),
    ]
    analysis_rc = _run_command(analysis_cmd, cwd=repo_root, log_path=analysis_log)
    summary_json = run_dir / "failure_analysis" / "summary.json"
    if analysis_rc != 0 or not summary_json.exists():
        raise RuntimeError(f"analysis failed: rc={analysis_rc}, summary_json={summary_json}")
    analysis_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    failure_cases = int(analysis_summary.get("failure_cases") or 0)
    goc_only_correct_cases = int(analysis_summary.get("goc_only_correct_cases") or 0)

    bundle_scripts = bundle_root / "scripts"
    bundle_scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "scripts" / "analyze_policyops_debug_run.py", bundle_scripts / "analyze_policyops_debug_run.py")
    shutil.copy2(repo_root / "scripts" / "run_phase2_openai_debug_bundle.py", bundle_scripts / "run_phase2_openai_debug_bundle.py")

    index_lines: List[str] = []
    index_lines.append(f"# Phase2 Debug Bundle: {run_id}")
    index_lines.append("")
    index_lines.append("## Config")
    index_lines.append(f"- preset: `{PRESET}`")
    index_lines.append(f"- methods: `{','.join(METHODS)}`")
    index_lines.append(f"- budget: `{BUDGET}`")
    index_lines.append(f"- model: `{args.model}`")
    index_lines.append(f"- parallel_workers: `{args.parallel_workers}`")
    index_lines.append(f"- n_threads: `{n_threads}`")
    index_lines.append(f"- total_tasks_targeted: `{total_tasks}`")
    index_lines.append(f"- phase1_data_source: `{_rel(Path(phase1_data), repo_root)}`")
    index_lines.append("")
    index_lines.append("## Compare Artifacts")
    index_lines.append(f"- compare_log: `{_rel(compare_log, repo_root)}`")
    index_lines.append(f"- report_json: `{_rel(report_json, repo_root)}`")
    index_lines.append(f"- run_dir: `{_rel(run_dir, repo_root)}`")
    index_lines.append(f"- sweep_csv: `{_rel(sweep_csv, repo_root)}`")
    index_lines.append(f"- graphs_internal_sample: `{_rel(graphs_internal_sample, repo_root)}`")
    index_lines.append("")
    index_lines.append("## Analysis Artifacts")
    index_lines.append(f"- analysis_log: `{_rel(analysis_log, repo_root)}`")
    index_lines.append(f"- summary_json: `{_rel(summary_json, repo_root)}`")
    index_lines.append(
        f"- failures_csv: `{_rel(Path(analysis_summary['files']['failures_csv']), repo_root)}`"
    )
    index_lines.append(
        f"- failures_md: `{_rel(Path(analysis_summary['files']['failures_md']), repo_root)}`"
    )
    index_lines.append(f"- failure_cases: `{failure_cases}`")
    index_lines.append(f"- goc_only_correct_cases: `{goc_only_correct_cases}`")
    (bundle_root / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    zip_path = bundle_root.with_suffix(".zip")
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"FAILURE_CASES={failure_cases}")
    print(f"GOC_ONLY_CORRECT={goc_only_correct_cases}")
    print(f"RUN_DIR={run_dir}")
    print(f"N_THREADS={n_threads}")
    print(f"TOTAL_TASKS={total_tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
