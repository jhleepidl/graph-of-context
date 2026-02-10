#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


THREADED_PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"
SMOKE_BUDGETS = "1100"
MAIN_BUDGETS = "900,1100,1350,1600,2000"
METHODS = ["full", "similarity_only", "goc"]


@dataclass
class RunArtifacts:
    ok: bool
    report_json: Optional[Path]
    sweep_csv: Optional[Path]
    graphs_internal_sample: Optional[Path]
    log_path: Path
    notes: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rand8() -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(8))


def _rel(p: Optional[Path], root: Path) -> str:
    if p is None:
        return "None"
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _find_latest_phase1_data(repo_root: Path) -> Optional[Path]:
    cands = [
        p
        for p in (repo_root / "experiment_bundles").glob("**/phase1_dummy/data")
        if p.is_dir() and (p / "tasks" / "tasks.jsonl").exists()
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_thread_counts(tasks_jsonl: Path) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    with tasks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = str(obj.get("thread_id") or "")
            if not tid:
                # Non-threaded fallback: treat each task as independent pseudo-thread.
                tid = f"__task__:{obj.get('task_id', 'unknown')}"
            counts[tid] = counts.get(tid, 0) + 1
    ordered = sorted(counts.keys())
    return ordered, counts


def _choose_k_smoke(ordered_threads: List[str], counts: Dict[str, int], max_tasks: int = 10) -> Tuple[int, int]:
    # Choose the largest prefix under cap so smoke remains meaningful while cost-bounded.
    total = 0
    k = 0
    for tid in ordered_threads:
        c = int(counts.get(tid, 0))
        if total + c > max_tasks:
            break
        total += c
        k += 1
    if k == 0 and ordered_threads:
        k = 1
        total = int(counts.get(ordered_threads[0], 0))
    return k, total


def _choose_k_main(
    ordered_threads: List[str],
    counts: Dict[str, int],
    lo: int = 80,
    hi: int = 120,
    target: int = 100,
) -> Tuple[int, int]:
    total = 0
    for idx, tid in enumerate(ordered_threads, start=1):
        total += int(counts.get(tid, 0))
        if lo <= total <= hi:
            return idx, total

    # Fallback if an exact in-range prefix doesn't exist.
    running = 0
    best = (0, 0, 10**9)  # k, total, distance
    for idx, tid in enumerate(ordered_threads, start=1):
        running += int(counts.get(tid, 0))
        if running > hi:
            break
        dist = abs(running - target)
        if dist < best[2]:
            best = (idx, running, dist)
    if best[0] > 0:
        return best[0], best[1]
    # If everything exceeded hi early or not enough tasks, return all threads.
    total_all = sum(int(counts.get(tid, 0)) for tid in ordered_threads)
    return len(ordered_threads), total_all


def _build_compare_cmd(
    out_dir: Path,
    *,
    dotenv_path: str,
    n_threads: int,
    budgets: str,
    parallel_workers: int,
    model: str,
    is_smoke: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "compare",
        "--preset",
        THREADED_PRESET,
        "--llm",
        "openai",
        "--model",
        model,
        "--judge",
        "symbolic_packed",
        "--methods",
        *METHODS,
        "--thread_context_budget_sweep",
        budgets,
        "--save_goc_graph",
        "--save_goc_dot",
        "--goc_graph_sample_rate",
        "0.25",
        "--save_prompts",
        "--save_raw",
        "--parallel_workers",
        str(parallel_workers),
        "--n_threads",
        str(n_threads),
        "--dotenv",
        dotenv_path,
        "--out_dir",
        str(out_dir),
    ]
    if is_smoke:
        cmd.extend(["--debug_n", "10"])
    return cmd


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


def _discover_artifacts(out_dir: Path, log_path: Path) -> RunArtifacts:
    report_cands = list((out_dir / "runs" / "compare").glob("*.json"))
    report_cands += list((out_dir / "runs").glob("**/report*.json"))
    report_json = max(report_cands, key=lambda p: p.stat().st_mtime) if report_cands else None

    csv_cands = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(csv_cands, key=lambda p: p.stat().st_mtime) if csv_cands else None

    gi_cands = list((out_dir / "runs" / "compare").glob("*/goc/graphs_internal/*.jsonl"))
    graphs_internal_sample = max(gi_cands, key=lambda p: p.stat().st_mtime) if gi_cands else None

    ok = report_json is not None and sweep_csv is not None and graphs_internal_sample is not None
    notes = ""
    if report_json is None:
        notes += "missing_report_json; "
    if sweep_csv is None:
        notes += "missing_sweep_csv; "
    if graphs_internal_sample is None:
        notes += "missing_graphs_internal; "
    return RunArtifacts(
        ok=ok,
        report_json=report_json,
        sweep_csv=sweep_csv,
        graphs_internal_sample=graphs_internal_sample,
        log_path=log_path,
        notes=notes.strip(),
    )


def _count_429(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pats = [
        r"HTTP Error 429",
        r"Too Many Requests",
        r"\b429\b",
    ]
    return sum(len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in pats)


def _render_one_internal_graph(repo_root: Path, snapshot_jsonl: Optional[Path], out_log: Path) -> None:
    if snapshot_jsonl is None or not snapshot_jsonl.exists():
        return
    cmd = [sys.executable, "scripts/render_internal_graph.py", str(snapshot_jsonl)]
    _run_command(cmd, cwd=repo_root, log_path=out_log)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(src_dir):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(src_dir.parent))
                zf.write(str(full), arcname=arc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PolicyOps Phase2 OpenAI with parallel compare and bundle outputs.")
    parser.add_argument("--phase1_data_dir", type=str, default="", help="Optional phase1_dummy/data path.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--fallback_parallel_workers", type=int, default=8)
    args = parser.parse_args()

    repo_root = _repo_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"goc_policyops_phase2_{ts}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase2_root = bundle_root / "phase2_openai"
    smoke_dir = phase2_root / "smoke"
    main_dir = phase2_root / "main"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)

    if args.phase1_data_dir:
        phase1_data = Path(args.phase1_data_dir)
    else:
        phase1_data = _find_latest_phase1_data(repo_root)
    if phase1_data is None or not phase1_data.exists():
        raise RuntimeError("Could not find phase1_dummy/data. Provide --phase1_data_dir.")

    # Keep source snapshot + copy data into both smoke/main out_dir so compare can reuse identical dataset.
    shutil.copytree(phase1_data, phase2_root / "data_source_snapshot", dirs_exist_ok=True)
    shutil.copytree(phase1_data, smoke_dir / "data", dirs_exist_ok=True)
    shutil.copytree(phase1_data, main_dir / "data", dirs_exist_ok=True)

    tasks_jsonl = smoke_dir / "data" / "tasks" / "tasks.jsonl"
    ordered_threads, counts = _load_thread_counts(tasks_jsonl)
    if not ordered_threads:
        raise RuntimeError(f"No tasks found in {tasks_jsonl}")

    smoke_k, smoke_tasks = _choose_k_smoke(ordered_threads, counts, max_tasks=10)
    main_k, main_tasks = _choose_k_main(ordered_threads, counts, lo=80, hi=120, target=100)

    index_lines: List[str] = []
    index_lines.append(f"# Phase2 OpenAI Parallel Bundle: {run_id}")
    index_lines.append("")
    index_lines.append("## Config")
    index_lines.append(f"- preset: `{THREADED_PRESET}`")
    index_lines.append(f"- model: `{args.model}`")
    index_lines.append(f"- dotenv: `{args.dotenv}`")
    index_lines.append(f"- phase1_data_source: `{_rel(Path(phase1_data), repo_root)}`")
    index_lines.append(f"- smoke_parallel_workers: `{args.parallel_workers}`")
    index_lines.append(f"- main_parallel_workers_initial: `{args.parallel_workers}`")
    index_lines.append(f"- main_parallel_workers_fallback: `{args.fallback_parallel_workers}`")
    index_lines.append(f"- smoke_n_threads(K): `{smoke_k}`")
    index_lines.append(f"- smoke_total_tasks: `{smoke_tasks}`")
    index_lines.append(f"- main_n_threads(K2): `{main_k}`")
    index_lines.append(f"- main_total_tasks: `{main_tasks}`")
    index_lines.append("")

    # SMOKE
    smoke_log = smoke_dir / "compare_sweep_openai.log"
    smoke_cmd = _build_compare_cmd(
        smoke_dir,
        dotenv_path=args.dotenv,
        n_threads=smoke_k,
        budgets=SMOKE_BUDGETS,
        parallel_workers=args.parallel_workers,
        model=args.model,
        is_smoke=True,
    )
    smoke_rc = _run_command(smoke_cmd, cwd=repo_root, log_path=smoke_log)
    smoke_art = _discover_artifacts(smoke_dir, smoke_log)
    _render_one_internal_graph(repo_root, smoke_art.graphs_internal_sample, smoke_dir / "render_internal_graph.log")

    index_lines.append("## Smoke")
    index_lines.append(f"- return_code: `{smoke_rc}`")
    index_lines.append(f"- report_json: `{_rel(smoke_art.report_json, repo_root)}`")
    index_lines.append(f"- sweep_csv: `{_rel(smoke_art.sweep_csv, repo_root)}`")
    index_lines.append(f"- graphs_internal_sample: `{_rel(smoke_art.graphs_internal_sample, repo_root)}`")
    index_lines.append(f"- log: `{_rel(smoke_art.log_path, repo_root)}`")
    if smoke_art.notes:
        index_lines.append(f"- notes: `{smoke_art.notes}`")
    index_lines.append(f"- success: `{bool(smoke_rc == 0 and smoke_art.ok)}`")
    index_lines.append("")

    main_used_workers = args.parallel_workers
    main_rc = None
    main_art: Optional[RunArtifacts] = None
    fallback_used = False
    fallback_reason = ""

    if smoke_rc == 0 and smoke_art.ok:
        main_log = main_dir / "compare_sweep_openai.log"
        main_cmd = _build_compare_cmd(
            main_dir,
            dotenv_path=args.dotenv,
            n_threads=main_k,
            budgets=MAIN_BUDGETS,
            parallel_workers=args.parallel_workers,
            model=args.model,
            is_smoke=False,
        )
        main_rc = _run_command(main_cmd, cwd=repo_root, log_path=main_log)
        main_art = _discover_artifacts(main_dir, main_log)
        _render_one_internal_graph(repo_root, main_art.graphs_internal_sample, main_dir / "render_internal_graph.log")

        # Rate-limit fallback: rerun MAIN once with fewer workers if many 429s are observed.
        count429 = _count_429(main_log)
        if count429 >= 8 and args.fallback_parallel_workers < args.parallel_workers:
            fallback_used = True
            fallback_reason = f"observed_429_count={count429}"
            main_used_workers = args.fallback_parallel_workers
            main_log_fb = main_dir / "compare_sweep_openai_fallback.log"
            main_cmd_fb = _build_compare_cmd(
                main_dir,
                dotenv_path=args.dotenv,
                n_threads=main_k,
                budgets=MAIN_BUDGETS,
                parallel_workers=args.fallback_parallel_workers,
                model=args.model,
                is_smoke=False,
            )
            main_rc = _run_command(main_cmd_fb, cwd=repo_root, log_path=main_log_fb)
            main_art = _discover_artifacts(main_dir, main_log_fb)
            _render_one_internal_graph(
                repo_root,
                main_art.graphs_internal_sample,
                main_dir / "render_internal_graph_fallback.log",
            )

        index_lines.append("## Main")
        index_lines.append(f"- return_code: `{main_rc}`")
        index_lines.append(f"- report_json: `{_rel(main_art.report_json, repo_root) if main_art else 'None'}`")
        index_lines.append(f"- sweep_csv: `{_rel(main_art.sweep_csv, repo_root) if main_art else 'None'}`")
        index_lines.append(
            f"- graphs_internal_sample: `{_rel(main_art.graphs_internal_sample, repo_root) if main_art else 'None'}`"
        )
        index_lines.append(f"- log: `{_rel((main_art.log_path if main_art else main_log), repo_root)}`")
        if main_art and main_art.notes:
            index_lines.append(f"- notes: `{main_art.notes}`")
        index_lines.append(f"- fallback_used: `{fallback_used}`")
        if fallback_used:
            index_lines.append(f"- fallback_reason: `{fallback_reason}`")
        index_lines.append(f"- parallel_workers_used: `{main_used_workers}`")
        index_lines.append(f"- success: `{bool(main_rc == 0 and main_art and main_art.ok)}`")
        index_lines.append("")
    else:
        index_lines.append("## Main")
        index_lines.append("- skipped: `true`")
        index_lines.append("- reason: `smoke_failed_or_missing_artifacts`")
        index_lines.append("")

    index_path = bundle_root / "INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = bundle_root.with_suffix(".zip")
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(
        "SUMMARY "
        f"smoke_K={smoke_k} smoke_tasks={smoke_tasks} "
        f"main_K2={main_k} main_tasks={main_tasks} "
        f"parallel_workers_smoke={args.parallel_workers} "
        f"parallel_workers_main={main_used_workers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
