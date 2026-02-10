#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from policyops_bundle_layout import build_bundle_quick_access


PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"
BUDGETS = "900,1100,1350,1600,2000"
METHODS = ["full", "similarity_only", "goc"]


@dataclass
class CompareResult:
    ok: bool
    return_code: int
    log_path: Path
    report_json: Optional[Path]
    sweep_csv: Optional[Path]
    run_dir: Optional[Path]
    graphs_internal_sample: Optional[Path]
    used_parallel_workers: int
    fallback_used: bool
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


def _count_429(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pats = [r"HTTP Error 429", r"Too Many Requests", r"\b429\b"]
    return sum(len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in pats)


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
            tid = str(obj.get("thread_id") or "")
            if not tid:
                tid = f"__task__:{obj.get('task_id', 'unknown')}"
            counts[tid] = counts.get(tid, 0) + 1
    return sorted(counts.keys()), counts


def _choose_k(
    ordered_threads: List[str],
    counts: Dict[str, int],
    *,
    lo: int,
    hi: int,
    target: int,
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


def _build_generate_cmd(
    out_dir: Path,
    *,
    seed: int,
    n_threads_generate: int,
    n_docs: Optional[int] = None,
    branch_distractor_rate: Optional[float] = None,
) -> List[str]:
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
        str(n_threads_generate),
        "--out_dir",
        str(out_dir),
    ]
    if n_docs is not None:
        cmd.extend(["--n_docs", str(int(n_docs))])
    if branch_distractor_rate is not None:
        cmd.extend(["--branch_distractor_rate", str(float(branch_distractor_rate))])
    return cmd


def _build_compare_cmd(
    out_dir: Path,
    *,
    dotenv_path: str,
    model: str,
    n_threads: int,
    parallel_workers: int,
) -> List[str]:
    return [
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
        *METHODS,
        "--thread_context_budget_sweep",
        BUDGETS,
        "--parallel_workers",
        str(parallel_workers),
        "--n_threads",
        str(n_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "0.25",
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "0.25",
        "--goc_budget_follow_sweep",
        "--goc_budget_chars_to_tokens_divisor",
        "4",
        "--goc_activity_filter",
        "--goc_anchor_top1_lexical",
        "--goc_mmr_lambda",
        "0.35",
        "--dotenv",
        dotenv_path,
        "--out_dir",
        str(out_dir),
    ]


def _discover_compare_artifacts(out_dir: Path, log_path: Path, used_workers: int, fallback_used: bool) -> CompareResult:
    report_candidates = list((out_dir / "runs" / "compare").glob("*.json"))
    report_json = max(report_candidates, key=lambda p: p.stat().st_mtime) if report_candidates else None
    run_dir: Optional[Path] = None
    if report_json is not None:
        candidate = out_dir / "runs" / "compare" / report_json.stem
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        run_dirs = [p for p in (out_dir / "runs" / "compare").glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    sweep_candidates = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(sweep_candidates, key=lambda p: p.stat().st_mtime) if sweep_candidates else None
    graphs_internal_sample = None
    if run_dir is not None:
        candidates = list((run_dir / "goc" / "graphs_internal").glob("*.jsonl"))
        if candidates:
            graphs_internal_sample = max(candidates, key=lambda p: p.stat().st_mtime)
    notes = ""
    if report_json is None:
        notes += "missing_report_json;"
    if sweep_csv is None:
        notes += "missing_sweep_csv;"
    if graphs_internal_sample is None:
        notes += "missing_graphs_internal;"
    ok = bool(report_json and sweep_csv)
    return CompareResult(
        ok=ok,
        return_code=0 if ok else 1,
        log_path=log_path,
        report_json=report_json,
        sweep_csv=sweep_csv,
        run_dir=run_dir,
        graphs_internal_sample=graphs_internal_sample,
        used_parallel_workers=used_workers,
        fallback_used=fallback_used,
        notes=notes.strip(),
    )


def _run_compare_with_fallback(
    repo_root: Path,
    out_dir: Path,
    *,
    dotenv_path: str,
    model: str,
    n_threads: int,
    parallel_workers: int,
    fallback_parallel_workers: int,
    log_name: str,
) -> CompareResult:
    log_path = out_dir / log_name
    cmd = _build_compare_cmd(
        out_dir,
        dotenv_path=dotenv_path,
        model=model,
        n_threads=n_threads,
        parallel_workers=parallel_workers,
    )
    rc = _run_command(cmd, cwd=repo_root, log_path=log_path)
    if rc == 0:
        result = _discover_compare_artifacts(out_dir, log_path, parallel_workers, fallback_used=False)
        result.return_code = rc
        return result

    count429 = _count_429(log_path)
    if count429 >= 8 and fallback_parallel_workers < parallel_workers:
        fb_log = out_dir / f"{Path(log_name).stem}_fallback.log"
        fb_cmd = _build_compare_cmd(
            out_dir,
            dotenv_path=dotenv_path,
            model=model,
            n_threads=n_threads,
            parallel_workers=fallback_parallel_workers,
        )
        fb_rc = _run_command(fb_cmd, cwd=repo_root, log_path=fb_log)
        result = _discover_compare_artifacts(out_dir, fb_log, fallback_parallel_workers, fallback_used=True)
        result.return_code = fb_rc
        if count429 > 0:
            result.notes = (result.notes + f" retry_after_429={count429}").strip()
        return result

    result = _discover_compare_artifacts(out_dir, log_path, parallel_workers, fallback_used=False)
    result.return_code = rc
    if count429 > 0:
        result.notes = (result.notes + f" observed_429={count429}").strip()
    return result


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(src_dir):
            for name in files:
                full = Path(base) / name
                arc = str(full.relative_to(src_dir.parent))
                zf.write(str(full), arcname=arc)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase3 paper-style PolicyOps evaluation bundle runner.")
    parser.add_argument("--phase1_data_dir", type=str, default="")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--fallback_parallel_workers", type=int, default=8)
    parser.add_argument("--n_threads_generate", type=int, default=140)
    parser.add_argument("--reuse_phase1_v1", action="store_true", default=True)
    parser.add_argument("--no_reuse_phase1_v1", action="store_false", dest="reuse_phase1_v1")
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = f"goc_policyops_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase3_root = bundle_root / "phase3"
    data_root = phase3_root / "data"
    runs_root = phase3_root / "runs"
    analysis_root = phase3_root / "analysis"
    data_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    phase1_data = Path(args.phase1_data_dir) if args.phase1_data_dir else _find_latest_phase1_data(repo_root)
    variants = [
        {
            "name": "V1_baseline",
            "reuse_phase1": True,
            "n_docs": None,
            "branch_distractor_rate": None,
        },
        {
            "name": "V2_high_distractor",
            "reuse_phase1": False,
            "n_docs": None,
            "branch_distractor_rate": 0.8,
        },
        {
            "name": "V3_n20_docs",
            "reuse_phase1": False,
            "n_docs": 20,
            "branch_distractor_rate": None,
        },
    ]

    run_rows: List[Dict[str, Any]] = []
    generate_rows: List[Dict[str, Any]] = []
    successful_runs = 0
    total_runs_planned = len(variants) * 2
    compare_runs_executed = 0

    for v in variants:
        variant_name = str(v["name"])
        variant_root = data_root / variant_name
        variant_root.mkdir(parents=True, exist_ok=True)
        variant_data_dir = variant_root / "data"

        if (
            variant_name == "V1_baseline"
            and bool(args.reuse_phase1_v1)
            and phase1_data is not None
            and phase1_data.exists()
        ):
            shutil.copytree(phase1_data, variant_data_dir, dirs_exist_ok=True)
            generate_rows.append(
                {
                    "variant": variant_name,
                    "source": "phase1_reuse",
                    "phase1_data": str(phase1_data),
                    "generate_log": "",
                    "generate_rc": 0,
                    "ok": True,
                }
            )
        else:
            gen_log = variant_root / "generate.log"
            gen_cmd = _build_generate_cmd(
                variant_root,
                seed=int(args.seed),
                n_threads_generate=int(args.n_threads_generate),
                n_docs=v.get("n_docs"),
                branch_distractor_rate=v.get("branch_distractor_rate"),
            )
            gen_rc = _run_command(gen_cmd, cwd=repo_root, log_path=gen_log)
            ok = gen_rc == 0 and (variant_data_dir / "tasks" / "tasks.jsonl").exists()
            generate_rows.append(
                {
                    "variant": variant_name,
                    "source": "generate",
                    "phase1_data": str(phase1_data) if phase1_data else "",
                    "generate_log": str(gen_log),
                    "generate_rc": gen_rc,
                    "ok": ok,
                }
            )
            if not ok:
                run_rows.append(
                    {
                        "variant": variant_name,
                        "split": "dev",
                        "skipped": True,
                        "reason": "generate_failed",
                    }
                )
                run_rows.append(
                    {
                        "variant": variant_name,
                        "split": "main",
                        "skipped": True,
                        "reason": "generate_failed",
                    }
                )
                continue

        tasks_jsonl = variant_data_dir / "tasks" / "tasks.jsonl"
        if not tasks_jsonl.exists():
            run_rows.append(
                {
                    "variant": variant_name,
                    "split": "dev",
                    "skipped": True,
                    "reason": "missing_tasks_jsonl",
                }
            )
            run_rows.append(
                {
                    "variant": variant_name,
                    "split": "main",
                    "skipped": True,
                    "reason": "missing_tasks_jsonl",
                }
            )
            continue

        ordered_threads, counts = _load_thread_counts(tasks_jsonl)
        k_dev, n_dev = _choose_k(ordered_threads, counts, lo=70, hi=90, target=80)
        k_main, n_main = _choose_k(ordered_threads, counts, lo=180, hi=220, target=200)

        for split, k_threads, n_tasks in [("dev", k_dev, n_dev), ("main", k_main, n_main)]:
            split_out = runs_root / variant_name / split
            split_out.mkdir(parents=True, exist_ok=True)
            shutil.copytree(variant_data_dir, split_out / "data", dirs_exist_ok=True)
            compare_runs_executed += 1
            result = _run_compare_with_fallback(
                repo_root,
                split_out,
                dotenv_path=args.dotenv,
                model=args.model,
                n_threads=int(k_threads),
                parallel_workers=int(args.parallel_workers),
                fallback_parallel_workers=int(args.fallback_parallel_workers),
                log_name="compare.log",
            )
            if result.return_code == 0 and result.ok:
                successful_runs += 1
            run_rows.append(
                {
                    "variant": variant_name,
                    "split": split,
                    "n_threads": int(k_threads),
                    "tasks_targeted": int(n_tasks),
                    "return_code": int(result.return_code),
                    "ok": bool(result.return_code == 0 and result.ok),
                    "parallel_workers_used": int(result.used_parallel_workers),
                    "fallback_used": bool(result.fallback_used),
                    "log": str(result.log_path),
                    "report_json": str(result.report_json) if result.report_json else "",
                    "sweep_csv": str(result.sweep_csv) if result.sweep_csv else "",
                    "run_dir": str(result.run_dir) if result.run_dir else "",
                    "graphs_internal_sample": str(result.graphs_internal_sample)
                    if result.graphs_internal_sample
                    else "",
                    "notes": result.notes,
                }
            )

    _write_rows_csv(analysis_root / "phase3_run_matrix.csv", run_rows)
    _write_rows_csv(analysis_root / "phase3_generate_matrix.csv", generate_rows)

    analyze_log = analysis_root / "analyze_phase3.log"
    analyze_cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_phase3_results.py",
        "--phase3_root",
        str(phase3_root),
        "--out_dir",
        str(analysis_root),
        "--split",
        "main",
    ]
    analyze_rc = _run_command(analyze_cmd, cwd=repo_root, log_path=analyze_log)
    analysis_summary_json = analysis_root / "phase3_summary.json"
    analysis_summary: Dict[str, Any] = {}
    if analyze_rc == 0 and analysis_summary_json.exists():
        analysis_summary = json.loads(analysis_summary_json.read_text(encoding="utf-8"))

    index_lines: List[str] = []
    index_lines.append(f"# Phase3 Paper Eval Bundle: {run_id}")
    index_lines.append("")
    index_lines.append("## Config")
    index_lines.append(f"- preset: `{PRESET}`")
    index_lines.append(f"- model: `{args.model}`")
    index_lines.append(f"- budgets: `{BUDGETS}`")
    index_lines.append(f"- methods: `{','.join(METHODS)}`")
    index_lines.append(f"- seed: `{args.seed}`")
    index_lines.append(f"- parallel_workers: `{args.parallel_workers}`")
    index_lines.append(f"- variants: `{', '.join(v['name'] for v in variants)}`")
    index_lines.append("")
    index_lines.append("## Run Summary")
    index_lines.append(f"- total_runs_planned: `{total_runs_planned}`")
    index_lines.append(f"- compare_runs_executed: `{compare_runs_executed}`")
    index_lines.append(f"- compare_runs_successful: `{successful_runs}`")
    index_lines.append(f"- analyze_rc: `{analyze_rc}`")
    if analysis_summary:
        index_lines.append(f"- analysis_rows: `{analysis_summary.get('rows')}`")
        index_lines.append(f"- goc_wins: `{analysis_summary.get('goc_wins')}`")
    index_lines.append("")
    index_lines.append("## Artifacts")
    index_lines.append(f"- run_matrix: `{_rel(analysis_root / 'phase3_run_matrix.csv', repo_root)}`")
    index_lines.append(f"- generate_matrix: `{_rel(analysis_root / 'phase3_generate_matrix.csv', repo_root)}`")
    index_lines.append(f"- summary_csv: `{_rel(analysis_root / 'phase3_summary.csv', repo_root)}`")
    index_lines.append(f"- summary_md: `{_rel(analysis_root / 'phase3_summary.md', repo_root)}`")
    index_lines.append(f"- summary_json: `{_rel(analysis_root / 'phase3_summary.json', repo_root)}`")
    index_lines.append(f"- analyze_log: `{_rel(analyze_log, repo_root)}`")
    (bundle_root / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    bundle_scripts = bundle_root / "scripts"
    bundle_scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "scripts" / "run_phase3_paper_eval_bundle.py", bundle_scripts / "run_phase3_paper_eval_bundle.py")
    shutil.copy2(repo_root / "scripts" / "analyze_phase3_results.py", bundle_scripts / "analyze_phase3_results.py")

    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = bundle_root.with_suffix(".zip")
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"TOTAL_RUNS_EXECUTED={compare_runs_executed}")
    print(f"TOTAL_RUNS_SUCCESSFUL={successful_runs}")
    if analysis_summary:
        print(f"ANALYSIS_ROWS={analysis_summary.get('rows')}")
        print(f"GOC_WINS={analysis_summary.get('goc_wins')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

