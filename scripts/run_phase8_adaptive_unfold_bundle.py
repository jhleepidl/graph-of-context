#!/usr/bin/env python3
"""
Phase 8 runner: adaptive unfold policy + candidate-pool sweep for GoC (late-pivot stress).

Core idea:
- Keep external context budget fixed (large enough for FullHistory).
- Control GoC via graph-unfold knobs:
  - fixed: (K=max_nodes, H=hops)
  - adaptive_pivot: (default K/H) for normal tasks, (pivot K/H) when a ticket update is present.

We sweep GoC candidate_pool_size to validate that K/H controls remain meaningful as retrieval expands.

Outputs (bundle):
- experiment_bundles/<bundle_name>/phase8/{data,runs,analysis}
- run_manifest.json (paths to report JSONs)
- analysis outputs via scripts/analyze_phase8_adaptive_unfold.py
- zip of entire bundle

Usage:
  python scripts/run_phase8_adaptive_unfold_bundle.py --dotenv .env
  python scripts/run_phase8_adaptive_unfold_bundle.py --dotenv .env --mode smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from policyops_bundle_layout import build_bundle_quick_access

DEFAULT_PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"


def _rand_suffix(n: int = 8) -> str:
    import random, string
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _zip_folder(folder: Path, zip_path: Path) -> None:
    import zipfile

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))


def _discover_compare_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Return (report_json, run_dir, sweep_csv, event_trace_sample).
    Tries to be robust across compare output layouts.
    """
    report_json = None
    run_dir = None
    sweep_csv = None
    trace_sample = None

    # report json pointers (usually under runs/compare/*.json)
    compare_root = out_dir / "runs" / "compare"
    if compare_root.exists():
        jsons = sorted(compare_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if jsons:
            report_json = jsons[0]
        # run dir is usually runs/compare/<timestamp_dir>/...
        run_dirs = [p for p in compare_root.glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    # sweep csv
    sweep_root = out_dir / "runs" / "context_budget_sweep"
    if sweep_root.exists():
        csvs = sorted(sweep_root.glob("*/results_context_budget_sweep.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csvs:
            sweep_csv = csvs[0]

    # event trace sample: search under run_dir
    if run_dir and run_dir.exists():
        candidates = list(run_dir.glob("event_traces/**/*.jsonl")) + list(run_dir.glob("*/event_traces/**/*.jsonl"))
        if candidates:
            trace_sample = max(candidates, key=lambda p: p.stat().st_mtime)

    return report_json, run_dir, sweep_csv, trace_sample



def _link_data_dir(data_out: Path, run_out: Path) -> None:
    """
    policyops.run compare expects a dataset at <out_dir>/data/ (e.g., data/worlds/documents.jsonl).
    The generator writes to <data_out>/data/, so we symlink it into each compare out_dir.
    """
    src = data_out / "data"
    if not src.exists():
        raise FileNotFoundError(f"Generated data dir missing: {src}")
    dst = run_out / "data"
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src.resolve())


@dataclass
class RunEntry:
    pivot_type: str
    variant: str  # baseline | goc_fixed | goc_adaptive
    pool_size: int
    fixed_k: Optional[int]
    fixed_h: Optional[int]
    policy: Optional[str]
    default_k: Optional[int]
    default_h: Optional[int]
    pivot_k: Optional[int]
    pivot_h: Optional[int]
    report_json: str
    compare_root: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase8 adaptive-unfold + pool-size sweep bundle.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    parser.add_argument("--budget", type=int, default=4000, help="External budget given to all methods (fixed).")
    parser.add_argument("--pivot_rate", type=float, default=0.5)
    parser.add_argument("--pivot_types", type=str, default="retention_flip entity_switch constraint_add")
    parser.add_argument("--seed", type=int, default=0)

    # Phase sizing modes
    parser.add_argument("--mode", choices=["smoke", "main"], default="main")
    parser.add_argument("--n_threads", type=int, default=40, help="Threads in generated dataset & compare.")
    parser.add_argument("--total_tasks", type=int, default=120, help="Total tasks to sample for compare.")
    parser.add_argument("--event_trace_sample_rate", type=float, default=1.0)

    # GoC sweeps
    parser.add_argument("--pool_sizes", type=str, default="16 64 128", help="candidate_pool_size values to sweep.")
    parser.add_argument("--fixed_configs", type=str, default="4,2", help="Space-separated list like '4,2 16,3'.")
    parser.add_argument("--adaptive_default", type=str, default="4,2", help="default K,H for adaptive policy.")
    parser.add_argument("--adaptive_pivot", type=str, default="16,3", help="pivot K,H for adaptive policy.")
    parser.add_argument("--bundle_root", type=str, default="", help="Optional explicit bundle name under experiment_bundles/")
    args = parser.parse_args()

    # Mode overrides for smoke (small + fast)
    if args.mode == "smoke":
        args.pivot_types = "entity_switch"
        args.pool_sizes = "32"
        args.fixed_configs = "4,2"
        args.n_threads = min(args.n_threads, 10)
        args.total_tasks = min(args.total_tasks, 20)
        args.parallel_workers = min(args.parallel_workers, 4)

    repo_root = Path(__file__).resolve().parents[1]
    exp_root = repo_root / "experiment_bundles"
    _ensure_dir(exp_root)

    if args.bundle_root:
        bundle_root = exp_root / args.bundle_root
    else:
        bundle_root = exp_root / f"goc_policyops_phase8_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand_suffix()}"
    phase8_root = bundle_root / "phase8"
    data_root = phase8_root / "data"
    runs_root = phase8_root / "runs"
    analysis_root = phase8_root / "analysis"
    for p in [data_root, runs_root, analysis_root]:
        _ensure_dir(p)

    # env for subprocess
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")

    pivot_types = args.pivot_types.split()
    pool_sizes = [int(x) for x in args.pool_sizes.split()]
    fixed_cfgs: List[Tuple[int, int]] = []
    for item in args.fixed_configs.split():
        k, h = item.split(",")
        fixed_cfgs.append((int(k), int(h)))

    def_k, def_h = [int(x) for x in args.adaptive_default.split(",")]
    piv_k, piv_h = [int(x) for x in args.adaptive_pivot.split(",")]

    manifest: Dict[str, Any] = {
        "phase": 8,
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
        "model": args.model,
        "preset": args.preset,
        "budget": args.budget,
        "pivot_rate": args.pivot_rate,
        "pivot_types": pivot_types,
        "pool_sizes": pool_sizes,
        "fixed_configs": fixed_cfgs,
        "adaptive_default": [def_k, def_h],
        "adaptive_pivot": [piv_k, piv_h],
        "runs": [],
    }

    # 1) Generate per pivot_type, then compare baselines + GoC sweeps
    for pivot_type in pivot_types:
        print(f"\n=== Phase8 pivot_type={pivot_type} ===")

        # Generate dataset
        data_out = data_root / pivot_type
        if data_out.exists():
            shutil.rmtree(data_out)
        _ensure_dir(data_out)

        gen_cmd = [
            sys.executable,
            "-u",
            "-m",
            "policyops.run",
            "generate",
            "--preset", args.preset,
            "--seed", str(args.seed),
            "--n_threads", str(args.n_threads),
            "--n_tasks", str(args.total_tasks),
            "--pivot_rate", str(args.pivot_rate),
            "--pivot_type", pivot_type,
            "--out_dir", str(data_out),
        ]
        _run(gen_cmd, cwd=repo_root, env=env)

        # Baseline compare (full + similarity_only) once per pivot_type
        base_out = runs_root / pivot_type / "baseline"
        _ensure_dir(base_out)
        _link_data_dir(data_out, base_out)
        base_cmp_cmd = [
            sys.executable,
            "-u",
            "-m",
            "policyops.run",
            "compare",
            "--preset", args.preset,
            "--llm", "openai",
            "--model", args.model,
            "--judge", "symbolic_packed",
            "--methods", "full", "similarity_only",
            "--thread_context_budget_sweep", str(args.budget),
            "--parallel_workers", str(args.parallel_workers),
            "--n_threads", str(args.n_threads),
            "--save_prompts",
            "--save_raw",
            "--save_event_trace",
            "--event_trace_sample_rate", str(args.event_trace_sample_rate),
            "--dotenv", args.dotenv,
            "--out_dir", str(base_out),
        ]
        _run(base_cmp_cmd, cwd=repo_root, env=env)
        base_report, base_run_dir, base_sweep_csv, _ = _discover_compare_artifacts(base_out)
        if not base_report:
            raise RuntimeError(f"Could not find report json for baseline: {base_out}")
        # One baseline entry per compare run (analyzer will expand per-method rows).
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="baseline",
            pool_size=0,
            fixed_k=None,
            fixed_h=None,
            policy="fixed",
            default_k=None,
            default_h=None,
            pivot_k=None,
            pivot_h=None,
            report_json=str(base_report),
            compare_root=str(base_out / "runs" / "compare"),
        )))

        # GoC runs
        for pool in pool_sizes:
            # Fixed configs
            for (k, h) in fixed_cfgs:
                out_dir = runs_root / pivot_type / f"goc_fixed_pool{pool}_K{k}_H{h}"
                _ensure_dir(out_dir)
                _link_data_dir(data_out, out_dir)
                cmp_cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "policyops.run",
                    "compare",
                    "--preset", args.preset,
                    "--llm", "openai",
                    "--model", args.model,
                    "--judge", "symbolic_packed",
                    "--methods", "goc",
                    "--thread_context_budget_sweep", str(args.budget),
                    "--parallel_workers", str(args.parallel_workers),
                    "--n_threads", str(args.n_threads),
                    "--save_prompts",
                    "--save_raw",
                    "--save_event_trace",
                    "--event_trace_sample_rate", str(args.event_trace_sample_rate),
                    "--goc_candidate_pool_size", str(pool),
                    "--goc_activity_filter",
                    "--goc_anchor_top1_lexical",
                    "--goc_mmr_lambda", "0.35",
                    "--goc_unfold_policy", "fixed",
                    "--goc_unfold_max_nodes", str(k),
                    "--goc_unfold_hops", str(h),
                    "--goc_unfold_budget_mode", "nodes_and_hops",
                    "--dotenv", args.dotenv,
                    "--out_dir", str(out_dir),
                ]
                _run(cmp_cmd, cwd=repo_root, env=env)
                report, run_dir, sweep_csv, _ = _discover_compare_artifacts(out_dir)
                if not report:
                    raise RuntimeError(f"Could not find report json for goc fixed run: {out_dir}")
                manifest["runs"].append(asdict(RunEntry(
                    pivot_type=pivot_type,
                    variant="goc_fixed",
                    pool_size=pool,
                    fixed_k=k,
                    fixed_h=h,
                    policy="fixed",
                    default_k=None,
                    default_h=None,
                    pivot_k=None,
                    pivot_h=None,
                    report_json=str(report),
                    compare_root=str(out_dir / "runs" / "compare"),
                )))

            # Adaptive policy
            out_dir = runs_root / pivot_type / f"goc_adaptive_pool{pool}_D{def_k}_{def_h}_P{piv_k}_{piv_h}"
            _ensure_dir(out_dir)
            _link_data_dir(data_out, out_dir)
            cmp_cmd = [
                sys.executable,
                "-u",
                "-m",
                "policyops.run",
                "compare",
                "--preset", args.preset,
                "--llm", "openai",
                "--model", args.model,
                "--judge", "symbolic_packed",
                "--methods", "goc",
                "--thread_context_budget_sweep", str(args.budget),
                "--parallel_workers", str(args.parallel_workers),
                "--n_threads", str(args.n_threads),
                "--save_prompts",
                "--save_raw",
                "--save_event_trace",
                "--event_trace_sample_rate", str(args.event_trace_sample_rate),
                "--goc_candidate_pool_size", str(pool),
                "--goc_activity_filter",
                "--goc_anchor_top1_lexical",
                "--goc_mmr_lambda", "0.35",
                "--goc_unfold_policy", "adaptive_pivot",
                "--goc_unfold_default_max_nodes", str(def_k),
                "--goc_unfold_default_hops", str(def_h),
                "--goc_unfold_pivot_max_nodes", str(piv_k),
                "--goc_unfold_pivot_hops", str(piv_h),
                "--goc_unfold_budget_mode", "nodes_and_hops",
                "--dotenv", args.dotenv,
                "--out_dir", str(out_dir),
            ]
            _run(cmp_cmd, cwd=repo_root, env=env)
            report, run_dir, sweep_csv, _ = _discover_compare_artifacts(out_dir)
            if not report:
                raise RuntimeError(f"Could not find report json for goc adaptive run: {out_dir}")
            manifest["runs"].append(asdict(RunEntry(
                pivot_type=pivot_type,
                variant="goc_adaptive",
                pool_size=pool,
                fixed_k=None,
                fixed_h=None,
                policy="adaptive_pivot",
                default_k=def_k,
                default_h=def_h,
                pivot_k=piv_k,
                pivot_h=piv_h,
                report_json=str(report),
                compare_root=str(out_dir / "runs" / "compare"),
            )))

    manifest_path = phase8_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2) Analyze
    analyze_cmd = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "analyze_phase8_adaptive_unfold.py"),
        "--phase8_root",
        str(phase8_root),
        "--out_dir",
        str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    # 3) INDEX
    idx_md = bundle_root / "INDEX.md"
    idx_md.write_text(
        "\n".join(
            [
                f"# Phase 8 Bundle",
                "",
                f"- Created: {manifest['created_at']}",
                f"- Mode: {args.mode}",
                f"- Model: {args.model}",
                f"- Preset: {args.preset}",
                f"- Budget: {args.budget}",
                f"- Pivot rate: {args.pivot_rate}",
                f"- Pivot types: {', '.join(pivot_types)}",
                f"- Pool sizes: {', '.join(map(str, pool_sizes))}",
                f"- Fixed configs: {', '.join([f'K{k} H{h}' for (k,h) in fixed_cfgs])}",
                f"- Adaptive default: K{def_k} H{def_h}",
                f"- Adaptive pivot: K{piv_k} H{piv_h}",
                "",
                "Outputs:",
                f"- Manifest: phase8/run_manifest.json",
                f"- Summary: phase8/analysis/phase8_summary.csv, phase8/analysis/phase8_summary.md",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 4) Zip bundle
    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = Path(str(bundle_root) + ".zip")
    _zip_folder(bundle_root, zip_path)
    print(f"\nDONE. Bundle: {bundle_root}")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()