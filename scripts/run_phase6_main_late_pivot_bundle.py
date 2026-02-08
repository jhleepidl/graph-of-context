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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"
BUDGET_SWEEP = "900,1100,1350,1600,2000"
PIVOT_RATES = [0.1, 0.3, 0.5]
PIVOT_TYPES = ["retention_flip", "entity_switch", "constraint_add"]


@dataclass
class RunArtifacts:
    config_id: str
    out_dir: Path
    generate_log: Path
    compare_log: Path
    report_json: Optional[Path]
    run_dir: Optional[Path]
    sweep_csv: Optional[Path]
    event_trace_sample: Optional[Path]
    tasks_total: int
    final_tasks: int
    n_threads_used: int
    success: bool
    note: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rand8() -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(8))


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


def _zip_dir(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(root.parent))
                zf.write(str(full), arcname=arc)


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


def _count_tasks(tasks_jsonl: Path) -> Tuple[int, int]:
    total = 0
    final = 0
    with tasks_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            if int(obj.get("episode_id") or 0) == 3:
                final += 1
    return total, final


def _generate_dataset_with_sizing(
    repo_root: Path,
    out_dir: Path,
    *,
    seed: int,
    pivot_rate: float,
    pivot_type: str,
    n_threads_init: int = 80,
    max_iters: int = 8,
) -> Tuple[int, int, int, Path]:
    n_threads = int(n_threads_init)
    generate_log = out_dir / "generate.log"
    tasks_path = out_dir / "data" / "tasks" / "tasks.jsonl"
    for _ in range(max_iters):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "policyops.run",
            "generate",
            "--preset",
            PRESET,
            "--seed",
            str(seed),
            "--n_threads",
            str(n_threads),
            "--pivot_rate",
            str(pivot_rate),
            "--pivot_type",
            pivot_type,
            "--out_dir",
            str(out_dir),
        ]
        rc = _run_command(cmd, cwd=repo_root, log_path=generate_log)
        if rc != 0:
            raise RuntimeError(f"generate failed rc={rc} (log: {generate_log})")
        if not tasks_path.exists():
            raise RuntimeError(f"missing tasks.jsonl: {tasks_path}")
        total, final = _count_tasks(tasks_path)
        if 180 <= total <= 300 and final >= 60:
            return total, final, n_threads, generate_log
        if total < 180 or final < 60:
            # Approximately +3 tasks per thread for threaded_v1_3.
            delta_threads = max(2, int(((180 - total) + 2) // 3))
            n_threads = min(120, n_threads + delta_threads)
            continue
        if total > 300:
            delta_threads = max(2, int(((total - 300) + 2) // 3))
            n_threads = max(60, n_threads - delta_threads)
            continue
    total, final = _count_tasks(tasks_path)
    return total, final, n_threads, generate_log


def _discover_compare_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    report_candidates = list((out_dir / "runs" / "compare").glob("*.json"))
    report_json = max(report_candidates, key=lambda p: p.stat().st_mtime) if report_candidates else None
    run_dir = None
    if report_json is not None:
        candidate = out_dir / "runs" / "compare" / report_json.stem
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        run_dirs = [p for p in (out_dir / "runs" / "compare").glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    sweep_csv_candidates = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(sweep_csv_candidates, key=lambda p: p.stat().st_mtime) if sweep_csv_candidates else None
    event_trace_sample = None
    if run_dir is not None:
        trace_candidates = list(run_dir.glob("*/event_traces/**/*.jsonl"))
        if trace_candidates:
            event_trace_sample = max(trace_candidates, key=lambda p: p.stat().st_mtime)
    return report_json, run_dir, sweep_csv, event_trace_sample


def _run_compare(
    repo_root: Path,
    out_dir: Path,
    *,
    dotenv: str,
    model: str,
    parallel_workers: int,
    compare_log: Path,
) -> int:
    cmd = [
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
        model,
        "--judge",
        "symbolic_packed",
        "--methods",
        "full",
        "similarity_only",
        "goc",
        "--thread_context_budget_sweep",
        BUDGET_SWEEP,
        "--parallel_workers",
        str(parallel_workers),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "1.0",
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "0.25",
        "--goc_activity_filter",
        "--no_goc_anchor_top1_lexical",
        "--goc_mmr_lambda",
        "0.35",
        "--goc_budget_follow_sweep",
        "--goc_budget_chars_to_tokens_divisor",
        "4",
        "--dotenv",
        dotenv,
        "--out_dir",
        str(out_dir),
    ]
    return _run_command(cmd, cwd=repo_root, log_path=compare_log)


def _run_dummy_smoke(repo_root: Path, smoke_dir: Path) -> None:
    data_dir = smoke_dir / "data_smoke"
    cmp_dir = smoke_dir / "compare_smoke"
    gen_log = smoke_dir / "generate_dummy.log"
    cmp_log = smoke_dir / "compare_dummy.log"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    gen_cmd = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "generate",
        "--preset",
        PRESET,
        "--seed",
        "0",
        "--n_threads",
        "6",
        "--pivot_rate",
        "0.3",
        "--pivot_type",
        "retention_flip",
        "--out_dir",
        str(data_dir),
    ]
    rc_gen = _run_command(gen_cmd, cwd=repo_root, log_path=gen_log)
    if rc_gen != 0:
        raise RuntimeError(f"dummy generate failed (log={gen_log})")
    _symlink_or_copy_data(data_dir / "data", cmp_dir / "data")
    cmp_cmd = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "compare",
        "--preset",
        PRESET,
        "--llm",
        "dummy",
        "--model",
        "dummy",
        "--judge",
        "symbolic_packed",
        "--methods",
        "full",
        "similarity_only",
        "goc",
        "--thread_context_budget_sweep",
        "1100",
        "--parallel_workers",
        "4",
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "1.0",
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "0.5",
        "--out_dir",
        str(cmp_dir),
    ]
    rc_cmp = _run_command(cmp_cmd, cwd=repo_root, log_path=cmp_log)
    if rc_cmp != 0:
        raise RuntimeError(f"dummy compare failed (log={cmp_log})")
    _, run_dir, _, trace_sample = _discover_compare_artifacts(cmp_dir)
    if run_dir is None or trace_sample is None:
        raise RuntimeError("dummy smoke missing run_dir or event trace sample")


def _read_summary_rows(summary_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not summary_csv.exists():
        return rows
    with summary_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase6 main late-pivot bundle.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_threads_init", type=int, default=80)
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = f"goc_policyops_phase6_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase6_root = bundle_root / "phase6"
    data_root = phase6_root / "data"
    runs_root = phase6_root / "runs"
    analysis_root = phase6_root / "analysis"
    smoke_root = phase6_root / "smoke_dummy"
    for p in [data_root, runs_root, analysis_root]:
        p.mkdir(parents=True, exist_ok=True)

    # Quick pipeline/trace smoke with dummy backend.
    _run_dummy_smoke(repo_root, smoke_root)

    run_artifacts: List[RunArtifacts] = []
    for pivot_type in PIVOT_TYPES:
        for pivot_rate in PIVOT_RATES:
            config_id = f"{pivot_type}_r{str(pivot_rate).replace('.', 'p')}"
            config_data_dir = data_root / config_id
            config_run_dir = runs_root / config_id
            config_run_dir.mkdir(parents=True, exist_ok=True)
            total_tasks, final_tasks, n_threads_used, generate_log = _generate_dataset_with_sizing(
                repo_root,
                config_data_dir,
                seed=args.seed,
                pivot_rate=float(pivot_rate),
                pivot_type=pivot_type,
                n_threads_init=args.n_threads_init,
            )
            _symlink_or_copy_data(config_data_dir / "data", config_run_dir / "data")
            compare_log = config_run_dir / "compare_openai.log"
            rc = _run_compare(
                repo_root,
                config_run_dir,
                dotenv=args.dotenv,
                model=args.model,
                parallel_workers=args.parallel_workers,
                compare_log=compare_log,
            )
            report_json, run_dir, sweep_csv, trace_sample = _discover_compare_artifacts(config_run_dir)
            success = bool(
                rc == 0
                and report_json is not None
                and sweep_csv is not None
                and trace_sample is not None
            )
            note = ""
            if rc != 0:
                note += f"compare_rc={rc};"
            if report_json is None:
                note += "missing_report_json;"
            if sweep_csv is None:
                note += "missing_sweep_csv;"
            if trace_sample is None:
                note += "missing_event_trace;"
            run_artifacts.append(
                RunArtifacts(
                    config_id=config_id,
                    out_dir=config_run_dir,
                    generate_log=generate_log,
                    compare_log=compare_log,
                    report_json=report_json,
                    run_dir=run_dir,
                    sweep_csv=sweep_csv,
                    event_trace_sample=trace_sample,
                    tasks_total=total_tasks,
                    final_tasks=final_tasks,
                    n_threads_used=n_threads_used,
                    success=success,
                    note=note,
                )
            )

    # Analyze all successful runs.
    analyze_log = phase6_root / "analyze.log"
    analyze_cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_phase6_main_late_pivot.py",
        "--phase6_root",
        str(phase6_root),
        "--out_dir",
        str(analysis_root),
    ]
    analyze_rc = _run_command(analyze_cmd, cwd=repo_root, log_path=analyze_log)
    if analyze_rc != 0:
        raise RuntimeError(f"phase6 analyzer failed (log={analyze_log})")

    summary_csv = analysis_root / "phase6_summary.csv"
    summary_md = analysis_root / "phase6_summary.md"
    failures_csv = analysis_root / "phase6_failures.csv"
    if not summary_csv.exists() or not summary_md.exists() or not failures_csv.exists():
        raise RuntimeError("phase6 analysis outputs missing")
    summary_rows = _read_summary_rows(summary_csv)

    index_path = bundle_root / "INDEX.md"
    lines: List[str] = []
    lines.append(f"# {run_id}")
    lines.append("")
    lines.append("## Settings")
    lines.append(f"- preset: {PRESET}")
    lines.append(f"- model: {args.model}")
    lines.append(f"- parallel_workers: {args.parallel_workers}")
    lines.append(f"- budgets: {BUDGET_SWEEP}")
    lines.append(f"- pivot_rates: {','.join(str(v) for v in PIVOT_RATES)}")
    lines.append(f"- pivot_types: {','.join(PIVOT_TYPES)}")
    lines.append("- methods: full, similarity_only, goc")
    lines.append("")
    lines.append("## Smoke")
    lines.append(f"- smoke_dir: {smoke_root}")
    lines.append("")
    lines.append("## Config Runs")
    for art in run_artifacts:
        lines.append(
            "- {cfg}: success={ok}, tasks_total={total}, final_tasks={final}, n_threads={nt}, "
            "report_json={report}, sweep_csv={sweep}, event_trace_sample={trace}, note={note}".format(
                cfg=art.config_id,
                ok=art.success,
                total=art.tasks_total,
                final=art.final_tasks,
                nt=art.n_threads_used,
                report=art.report_json,
                sweep=art.sweep_csv,
                trace=art.event_trace_sample,
                note=art.note or "none",
            )
        )
    lines.append("")
    lines.append("## Analysis")
    lines.append(f"- phase6_summary_csv: {summary_csv}")
    lines.append(f"- phase6_summary_md: {summary_md}")
    lines.append(f"- phase6_failures_csv: {failures_csv}")
    lines.append(f"- analyze_log: {analyze_log}")
    lines.append("")
    lines.append(f"- summary_rows: {len(summary_rows)}")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = bundle_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"CONFIG_RUNS={len(run_artifacts)}")
    print(f"SUCCESS_RUNS={sum(1 for a in run_artifacts if a.success)}")
    print(f"SUMMARY_ROWS={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

