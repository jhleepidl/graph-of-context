#!/usr/bin/env python3
"""Phase 9 runner: GoC unfolding *policy frontier*.

Phase 8 lesson:
- A naive "pivot => bigger K" adaptive policy can sometimes *hurt* (e.g., mixing stale/conflicting evidence).

Phase 9 goal:
- Compare a *frontier* of unfolding policies under a fixed large external budget
  (so FullHistory is not artificially budget-limited):

  Baselines:
    - full, similarity_only (budget fixed, large)

  GoC policies:
    - FIXED light:         K=4,  H=2
    - FIXED heavy:         K=16, H=3  (upper-bound style)
    - ADAPTIVE heavy:      default=(4,2),  pivot=(16,3)
    - ADAPTIVE H-only:     default=(4,2),  pivot=(4,3)   (increase closure without increasing nodes)
    - ADAPTIVE K-small:    default=(4,2),  pivot=(8,3)

We also sweep candidate_pool_size to decouple retrieval candidate quality from unfolding.

Outputs (bundle):
- experiment_bundles/<bundle_name>/phase9/{data,runs,analysis}
- run_manifest.json (paths to report JSONs)
- analysis outputs via scripts/analyze_phase9_frontier.py
- zip of the entire bundle

Usage:
  python scripts/run_phase9_frontier_bundle.py --dotenv .env
  python scripts/run_phase9_frontier_bundle.py --dotenv .env --mode smoke
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
    """Return (report_json, run_dir, sweep_csv, event_trace_sample)."""
    report_json = None
    run_dir = None
    sweep_csv = None
    trace_sample = None

    compare_root = out_dir / "runs" / "compare"
    if compare_root.exists():
        jsons = sorted(compare_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if jsons:
            report_json = jsons[0]
        run_dirs = [p for p in compare_root.glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    sweep_root = out_dir / "runs" / "context_budget_sweep"
    if sweep_root.exists():
        csvs = sorted(sweep_root.glob("*/results_context_budget_sweep.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csvs:
            sweep_csv = csvs[0]

    if run_dir and run_dir.exists():
        candidates = list(run_dir.glob("event_traces/**/*.jsonl")) + list(run_dir.glob("*/event_traces/**/*.jsonl"))
        if candidates:
            trace_sample = max(candidates, key=lambda p: p.stat().st_mtime)

    return report_json, run_dir, sweep_csv, trace_sample


def _link_data_dir(data_out: Path, run_out: Path) -> None:
    """Symlink <data_out>/data -> <run_out>/data for policyops.run compare."""
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
    variant: str  # baseline | goc_fixed_light | goc_fixed_heavy | goc_adaptive_* ...
    pool_size: int
    policy: str   # fixed | adaptive_pivot
    fixed_k: Optional[int]
    fixed_h: Optional[int]
    default_k: Optional[int]
    default_h: Optional[int]
    pivot_k: Optional[int]
    pivot_h: Optional[int]
    report_json: str
    compare_root: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase9 GoC policy frontier bundle.")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--parallel_workers", type=int, default=12)
    ap.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--pivot_rate", type=float, default=0.2)
    ap.add_argument("--pivot_types", type=str, default="retention_flip entity_switch constraint_add")
    ap.add_argument("--seed", type=int, default=0)

    # sizing
    ap.add_argument("--mode", choices=["smoke", "main"], default="main")
    ap.add_argument("--n_threads", type=int, default=40)
    ap.add_argument("--total_tasks", type=int, default=120)
    ap.add_argument("--event_trace_sample_rate", type=float, default=1.0)

    # sweep
    ap.add_argument("--pool_sizes", type=str, default="64", help="candidate_pool_size values (space-separated).")
    ap.add_argument("--bundle_root", type=str, default="", help="Optional explicit bundle name under experiment_bundles/")

    # fixed configs
    ap.add_argument("--fixed_light", type=str, default="4,2")
    ap.add_argument("--fixed_heavy", type=str, default="16,3")

    # adaptive configs
    ap.add_argument("--adaptive_default", type=str, default="4,2")
    ap.add_argument("--adaptive_heavy", type=str, default="16,3")
    ap.add_argument("--adaptive_h_only", type=str, default="4,3")
    ap.add_argument("--adaptive_k_small", type=str, default="8,3")

    args = ap.parse_args()

    # smoke overrides
    if args.mode == "smoke":
        args.pivot_types = "entity_switch"
        args.pool_sizes = "32"
        args.n_threads = min(args.n_threads, 10)
        args.total_tasks = min(args.total_tasks, 20)
        args.parallel_workers = min(args.parallel_workers, 4)
        args.pivot_rate = 0.5

    repo_root = Path(__file__).resolve().parents[1]
    exp_root = repo_root / "experiment_bundles"
    _ensure_dir(exp_root)

    if args.bundle_root:
        bundle_root = exp_root / args.bundle_root
    else:
        bundle_root = exp_root / f"goc_policyops_phase9_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand_suffix()}"

    phase9_root = bundle_root / "phase9"
    data_root = phase9_root / "data"
    runs_root = phase9_root / "runs"
    analysis_root = phase9_root / "analysis"
    for p in [data_root, runs_root, analysis_root]:
        _ensure_dir(p)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")

    pivot_types = args.pivot_types.split()
    pool_sizes = [int(x) for x in args.pool_sizes.split()]

    light_k, light_h = [int(x) for x in args.fixed_light.split(",")]
    heavy_k, heavy_h = [int(x) for x in args.fixed_heavy.split(",")]

    def_k, def_h = [int(x) for x in args.adaptive_default.split(",")]
    piv_heavy_k, piv_heavy_h = [int(x) for x in args.adaptive_heavy.split(",")]
    piv_honly_k, piv_honly_h = [int(x) for x in args.adaptive_h_only.split(",")]
    piv_ksmall_k, piv_ksmall_h = [int(x) for x in args.adaptive_k_small.split(",")]

    # GoC selection quality flags (carry over from Phase7+)
    base_goc_flags = [
        "--goc_activity_filter",
        "--goc_anchor_top1_lexical",
        "--goc_mmr_lambda", "0.35",
    ]

    manifest: Dict[str, Any] = {
        "phase": 9,
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
        "model": args.model,
        "preset": args.preset,
        "budget": args.budget,
        "pivot_rate": args.pivot_rate,
        "pivot_types": pivot_types,
        "pool_sizes": pool_sizes,
        "n_threads": args.n_threads,
        "total_tasks": args.total_tasks,
        "parallel_workers": args.parallel_workers,
        "policies": {
            "fixed_light": {"K": light_k, "H": light_h},
            "fixed_heavy": {"K": heavy_k, "H": heavy_h},
            "adaptive_default": {"K": def_k, "H": def_h},
            "adaptive_heavy": {"K": piv_heavy_k, "H": piv_heavy_h},
            "adaptive_h_only": {"K": piv_honly_k, "H": piv_honly_h},
            "adaptive_k_small": {"K": piv_ksmall_k, "H": piv_ksmall_h},
        },
        "runs": [],
    }

    for pivot_type in pivot_types:
        print(f"\n=== Phase9 pivot_type={pivot_type} ===")

        # Generate dataset
        data_out = data_root / pivot_type
        if data_out.exists():
            shutil.rmtree(data_out)
        _ensure_dir(data_out)

        gen_cmd = [
            sys.executable, "-u", "-m", "policyops.run", "generate",
            "--preset", args.preset,
            "--seed", str(args.seed),
            "--n_threads", str(args.n_threads),
            "--n_tasks", str(args.total_tasks),
            "--pivot_rate", str(args.pivot_rate),
            "--pivot_type", pivot_type,
            "--out_dir", str(data_out),
        ]
        _run(gen_cmd, cwd=repo_root, env=env)

        # Baseline compare once per pivot_type
        base_out = runs_root / pivot_type / "baseline"
        _ensure_dir(base_out)
        _link_data_dir(data_out, base_out)
        base_cmp_cmd = [
            sys.executable, "-u", "-m", "policyops.run", "compare",
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
        base_report, _, _, _ = _discover_compare_artifacts(base_out)
        if not base_report:
            raise RuntimeError(f"Could not find report json for baseline: {base_out}")

        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="baseline",
            pool_size=0,
            policy="fixed",
            fixed_k=None,
            fixed_h=None,
            default_k=None,
            default_h=None,
            pivot_k=None,
            pivot_h=None,
            report_json=str(base_report.relative_to(phase9_root)),
            compare_root=str(base_out.relative_to(phase9_root)),
        )))

        def _run_goc_compare(*, out_dir: Path, variant: str, pool: int, policy: str,
                             fixed: Optional[Tuple[int, int]] = None,
                             adaptive: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> None:
            _ensure_dir(out_dir)
            _link_data_dir(data_out, out_dir)

            cmd = [
                sys.executable, "-u", "-m", "policyops.run", "compare",
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
                "--goc_unfold_budget_mode", "nodes_and_hops",
                *base_goc_flags,
            ]

            if policy == "fixed":
                if fixed is None:
                    raise ValueError("fixed policy requires fixed=(K,H)")
                k, h = fixed
                cmd += [
                    "--goc_unfold_policy", "fixed",
                    "--goc_unfold_max_nodes", str(k),
                    "--goc_unfold_hops", str(h),
                ]
                fk, fh = k, h
                dk = dh = pk = ph = None
            elif policy == "adaptive_pivot":
                if adaptive is None:
                    raise ValueError("adaptive policy requires adaptive=((Dk,Dh),(Pk,Ph))")
                (dk, dh), (pk, ph) = adaptive
                cmd += [
                    "--goc_unfold_policy", "adaptive_pivot",
                    "--goc_unfold_default_max_nodes", str(dk),
                    "--goc_unfold_default_hops", str(dh),
                    "--goc_unfold_pivot_max_nodes", str(pk),
                    "--goc_unfold_pivot_hops", str(ph),
                ]
                fk = fh = None
            else:
                raise ValueError(f"Unknown policy: {policy}")

            cmd += ["--dotenv", args.dotenv, "--out_dir", str(out_dir)]
            _run(cmd, cwd=repo_root, env=env)

            report_json, _, _, _ = _discover_compare_artifacts(out_dir)
            if not report_json:
                raise RuntimeError(f"Missing report json: {out_dir}")

            manifest["runs"].append(asdict(RunEntry(
                pivot_type=pivot_type,
                variant=variant,
                pool_size=pool,
                policy=policy,
                fixed_k=fk,
                fixed_h=fh,
                default_k=dk,
                default_h=dh,
                pivot_k=pk,
                pivot_h=ph,
                report_json=str(report_json.relative_to(phase9_root)),
                compare_root=str(out_dir.relative_to(phase9_root)),
            )))

        for pool in pool_sizes:
            # fixed light/heavy
            _run_goc_compare(
                out_dir=runs_root / pivot_type / f"goc_fixed_light_pool{pool}_K{light_k}_H{light_h}",
                variant="goc_fixed_light",
                pool=pool,
                policy="fixed",
                fixed=(light_k, light_h),
            )
            _run_goc_compare(
                out_dir=runs_root / pivot_type / f"goc_fixed_heavy_pool{pool}_K{heavy_k}_H{heavy_h}",
                variant="goc_fixed_heavy",
                pool=pool,
                policy="fixed",
                fixed=(heavy_k, heavy_h),
            )

            # adaptive variants
            _run_goc_compare(
                out_dir=runs_root / pivot_type / f"goc_adaptive_heavy_pool{pool}_D{def_k}{def_h}_P{piv_heavy_k}{piv_heavy_h}",
                variant="goc_adaptive_heavy",
                pool=pool,
                policy="adaptive_pivot",
                adaptive=((def_k, def_h), (piv_heavy_k, piv_heavy_h)),
            )
            _run_goc_compare(
                out_dir=runs_root / pivot_type / f"goc_adaptive_h_only_pool{pool}_D{def_k}{def_h}_P{piv_honly_k}{piv_honly_h}",
                variant="goc_adaptive_h_only",
                pool=pool,
                policy="adaptive_pivot",
                adaptive=((def_k, def_h), (piv_honly_k, piv_honly_h)),
            )
            _run_goc_compare(
                out_dir=runs_root / pivot_type / f"goc_adaptive_k_small_pool{pool}_D{def_k}{def_h}_P{piv_ksmall_k}{piv_ksmall_h}",
                variant="goc_adaptive_k_small",
                pool=pool,
                policy="adaptive_pivot",
                adaptive=((def_k, def_h), (piv_ksmall_k, piv_ksmall_h)),
            )

    # write manifest
    manifest_path = phase9_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # analyze
    analyze_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_phase9_frontier.py"),
        "--phase9_root", str(phase9_root),
        "--out_dir", str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    # index
    idx_lines: List[str] = []
    idx_lines.append(f"# Phase 9 Bundle: {bundle_root.name}")
    idx_lines.append("")
    idx_lines.append("## What this measures")
    idx_lines.append("- GoC unfolding policy frontier: fixed light/heavy vs adaptive variants (heavy, H-only, K-small).")
    idx_lines.append("- candidate_pool_size sweep to decouple retrieval candidate quality from unfold policy.")
    idx_lines.append("")
    idx_lines.append("## Entry points")
    idx_lines.append(f"- Manifest: {manifest_path.relative_to(bundle_root)}")
    idx_lines.append(f"- Analysis CSV: {(analysis_root / 'phase9_summary.csv').relative_to(bundle_root)}")
    idx_lines.append(f"- Analysis MD: {(analysis_root / 'phase9_summary.md').relative_to(bundle_root)}")
    idx_lines.append("")
    idx_lines.append("## Run configuration")
    idx_lines.append(f"- preset={args.preset} model={args.model} budget={args.budget} pivot_rate={args.pivot_rate}")
    idx_lines.append(f"- n_threads={args.n_threads} total_tasks={args.total_tasks} parallel_workers={args.parallel_workers}")
    idx_lines.append(f"- pool_sizes={pool_sizes}")
    (bundle_root / "INDEX.md").write_text("\n".join(idx_lines), encoding="utf-8")

    # zip bundle
    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = bundle_root.with_suffix(".zip")
    _zip_folder(bundle_root, zip_path)
    print(f"\nBundle folder: {bundle_root}")
    print(f"Zip path: {zip_path}")


if __name__ == "__main__":
    main()
