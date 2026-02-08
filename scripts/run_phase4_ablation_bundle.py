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
VARIANTS = ["V1_baseline", "V2_high_distractor", "V3_n20_docs"]
BUDGETS = "900,1600,2000"


@dataclass
class RunResult:
    variant: str
    method_label: str
    out_dir: Path
    return_code: int
    ok: bool
    log_path: Path
    report_json: Optional[Path]
    sweep_csv: Optional[Path]
    run_dir: Optional[Path]
    n_threads: int
    tasks_targeted: int
    notes: str = ""


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


def _find_latest_phase3_data(repo_root: Path) -> Optional[Path]:
    candidates = [
        p
        for p in (repo_root / "experiment_bundles").glob("**/phase3/data")
        if p.is_dir() and all((p / v / "data" / "tasks" / "tasks.jsonl").exists() for v in VARIANTS)
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
            tid = str(obj.get("thread_id") or "")
            if not tid:
                tid = f"__task__:{obj.get('task_id', 'unknown')}"
            counts[tid] = counts.get(tid, 0) + 1
    return sorted(counts.keys()), counts


def _choose_k(
    ordered_threads: List[str],
    counts: Dict[str, int],
    *,
    lo: int = 110,
    hi: int = 130,
    target: int = 120,
) -> Tuple[int, int]:
    total = 0
    for idx, tid in enumerate(ordered_threads, start=1):
        total += int(counts.get(tid, 0))
        if lo <= total <= hi:
            return idx, total
    total = 0
    best = (0, 0, 10**9)
    for idx, tid in enumerate(ordered_threads, start=1):
        total += int(counts.get(tid, 0))
        if total > hi:
            break
        dist = abs(total - target)
        if dist < best[2]:
            best = (idx, total, dist)
    if best[0] > 0:
        return best[0], best[1]
    total_all = sum(int(counts.get(tid, 0)) for tid in ordered_threads)
    return len(ordered_threads), total_all


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


def _build_compare_cmd(
    out_dir: Path,
    *,
    dotenv_path: str,
    model: str,
    n_threads: int,
    parallel_workers: int,
    methods: List[str],
    extra_flags: List[str],
) -> List[str]:
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
        *methods,
        "--thread_context_budget_sweep",
        BUDGETS,
        "--parallel_workers",
        str(parallel_workers),
        "--n_threads",
        str(n_threads),
        "--save_event_trace",
        "--event_trace_sample_rate",
        "0.25",
        "--save_prompts",
        "--save_raw",
        "--goc_budget_follow_sweep",
        "--goc_budget_chars_to_tokens_divisor",
        "4",
        "--dotenv",
        dotenv_path,
        "--out_dir",
        str(out_dir),
    ]
    cmd.extend(extra_flags)
    return cmd


def _discover_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    report_candidates = list((out_dir / "runs" / "compare").glob("*.json"))
    report_json = max(report_candidates, key=lambda p: p.stat().st_mtime) if report_candidates else None
    run_dir: Optional[Path] = None
    if report_json is not None:
        c = out_dir / "runs" / "compare" / report_json.stem
        if c.is_dir():
            run_dir = c
    if run_dir is None:
        run_dirs = [p for p in (out_dir / "runs" / "compare").glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    sweep_candidates = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(sweep_candidates, key=lambda p: p.stat().st_mtime) if sweep_candidates else None
    return report_json, sweep_csv, run_dir


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(src_dir):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(src_dir.parent))
                zf.write(str(full), arcname=arc)


def _write_matrix(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase4 ablation bundle for PolicyOps.")
    parser.add_argument("--phase3_data_dir", type=str, default="", help="Optional path to phase3/data directory.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = f"goc_policyops_phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase4_root = bundle_root / "phase4"
    data_root = phase4_root / "data"
    runs_root = phase4_root / "runs"
    analysis_root = phase4_root / "analysis"
    data_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    source_phase3_data = Path(args.phase3_data_dir) if args.phase3_data_dir else _find_latest_phase3_data(repo_root)
    if source_phase3_data is None or not source_phase3_data.exists():
        raise RuntimeError("Phase3 data directory not found. Provide --phase3_data_dir.")

    # Copy data once into this bundle.
    for variant in VARIANTS:
        src = source_phase3_data / variant / "data"
        dst = data_root / variant / "data"
        if not src.exists():
            raise RuntimeError(f"Missing variant data: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=True)

    method_configs = [
        {
            "method_label": "similarity_only",
            "methods": ["similarity_only"],
            "extra_flags": [],
        },
        {
            "method_label": "goc_v1",
            "methods": ["goc"],
            "extra_flags": [
                "--save_goc_graph",
                "--save_goc_dot",
                "--goc_graph_sample_rate",
                "0.25",
                "--goc_activity_filter",
                "--goc_anchor_top1_lexical",
                "--goc_mmr_lambda",
                "0.35",
            ],
        },
        {
            "method_label": "goc_no_activity",
            "methods": ["goc"],
            "extra_flags": [
                "--save_goc_graph",
                "--save_goc_dot",
                "--goc_graph_sample_rate",
                "0.25",
                "--no_goc_anchor_top1_lexical",
                "--goc_mmr_lambda",
                "0.0",
            ],
        },
        {
            "method_label": "goc_no_mmr",
            "methods": ["goc"],
            "extra_flags": [
                "--save_goc_graph",
                "--save_goc_dot",
                "--goc_graph_sample_rate",
                "0.25",
                "--goc_activity_filter",
                "--goc_anchor_top1_lexical",
                "--goc_mmr_lambda",
                "0.0",
            ],
        },
        {
            "method_label": "goc_no_anchor",
            "methods": ["goc"],
            "extra_flags": [
                "--save_goc_graph",
                "--save_goc_dot",
                "--goc_graph_sample_rate",
                "0.25",
                "--goc_activity_filter",
                "--no_goc_anchor_top1_lexical",
                "--goc_mmr_lambda",
                "0.35",
            ],
        },
    ]

    run_results: List[RunResult] = []
    matrix_rows: List[Dict[str, object]] = []
    total_runs = 0
    success_runs = 0

    for variant in VARIANTS:
        tasks_jsonl = data_root / variant / "data" / "tasks" / "tasks.jsonl"
        ordered_threads, counts = _load_thread_counts(tasks_jsonl)
        n_threads, tasks_targeted = _choose_k(ordered_threads, counts, lo=110, hi=130, target=120)
        for cfg in method_configs:
            total_runs += 1
            method_label = str(cfg["method_label"])
            out_dir = runs_root / variant / method_label
            out_dir.mkdir(parents=True, exist_ok=True)
            _symlink_or_copy_data(data_root / variant / "data", out_dir / "data")

            log_path = out_dir / "compare.log"
            cmd = _build_compare_cmd(
                out_dir,
                dotenv_path=args.dotenv,
                model=args.model,
                n_threads=n_threads,
                parallel_workers=int(args.parallel_workers),
                methods=list(cfg["methods"]),
                extra_flags=list(cfg["extra_flags"]),
            )
            rc = _run_command(cmd, cwd=repo_root, log_path=log_path)
            report_json, sweep_csv, run_dir = _discover_artifacts(out_dir)
            ok = rc == 0 and report_json is not None and sweep_csv is not None
            if ok:
                success_runs += 1
            result = RunResult(
                variant=variant,
                method_label=method_label,
                out_dir=out_dir,
                return_code=rc,
                ok=ok,
                log_path=log_path,
                report_json=report_json,
                sweep_csv=sweep_csv,
                run_dir=run_dir,
                n_threads=int(n_threads),
                tasks_targeted=int(tasks_targeted),
                notes="",
            )
            run_results.append(result)
            matrix_rows.append(
                {
                    "variant": variant,
                    "method_label": method_label,
                    "return_code": rc,
                    "ok": ok,
                    "n_threads": int(n_threads),
                    "tasks_targeted": int(tasks_targeted),
                    "log": str(log_path),
                    "report_json": str(report_json) if report_json else "",
                    "sweep_csv": str(sweep_csv) if sweep_csv else "",
                    "run_dir": str(run_dir) if run_dir else "",
                }
            )

    _write_matrix(analysis_root / "phase4_run_matrix.csv", matrix_rows)

    analyze_log = analysis_root / "analyze_phase4.log"
    analyze_cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_phase4_ablation.py",
        "--phase4_root",
        str(phase4_root),
        "--out_dir",
        str(analysis_root),
    ]
    analyze_rc = _run_command(analyze_cmd, cwd=repo_root, log_path=analyze_log)

    index_lines: List[str] = []
    index_lines.append(f"# Phase4 Ablation Bundle: {run_id}")
    index_lines.append("")
    index_lines.append("## Config")
    index_lines.append(f"- source_phase3_data: `{_rel(source_phase3_data, repo_root)}`")
    index_lines.append(f"- variants: `{', '.join(VARIANTS)}`")
    index_lines.append(f"- budgets: `{BUDGETS}`")
    index_lines.append("- methods: `similarity_only, goc_v1, goc_no_activity, goc_no_mmr, goc_no_anchor`")
    index_lines.append(f"- model: `{args.model}`")
    index_lines.append(f"- parallel_workers: `{args.parallel_workers}`")
    index_lines.append("")
    index_lines.append("## Run Summary")
    index_lines.append(f"- total_runs: `{total_runs}`")
    index_lines.append(f"- successful_runs: `{success_runs}`")
    index_lines.append(f"- analyze_rc: `{analyze_rc}`")
    index_lines.append("")
    index_lines.append("## Artifacts")
    index_lines.append(f"- run_matrix: `{_rel(analysis_root / 'phase4_run_matrix.csv', repo_root)}`")
    index_lines.append(f"- ablation_csv: `{_rel(analysis_root / 'phase4_ablation_summary.csv', repo_root)}`")
    index_lines.append(f"- ablation_md: `{_rel(analysis_root / 'phase4_ablation_summary.md', repo_root)}`")
    index_lines.append(f"- analyze_log: `{_rel(analyze_log, repo_root)}`")
    (bundle_root / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    scripts_dir = bundle_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "scripts" / "run_phase4_ablation_bundle.py", scripts_dir / "run_phase4_ablation_bundle.py")
    shutil.copy2(repo_root / "scripts" / "analyze_phase4_ablation.py", scripts_dir / "analyze_phase4_ablation.py")

    zip_path = bundle_root.with_suffix(".zip")
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"TOTAL_RUNS={total_runs}")
    print(f"SUCCESSFUL_RUNS={success_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

