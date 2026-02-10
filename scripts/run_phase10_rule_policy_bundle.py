#!/usr/bin/env python3
"""Phase 10 runner: Rule-based GoC unfolding policy (pivot-type aware).

Goal:
- Compare baselines (full, similarity_only) vs GoC fixed light/heavy vs adaptive-heavy vs rule-v1.
- Keep baseline context budget generous (thread_context_budget_sweep=4000 by default).
- Control GoC by unfold knobs (K=max nodes, H=hops) rather than budget compression.

Rule policy (v1) (applied only on final episode via adaptive_pivot):
- non-pivot: (K=4, H=2)
- pivot:
  - retention_flip: (16, 3)
  - entity_switch: (4, 3)  # H-only
  - constraint_add: (4, 2)  # keep light (avoid interference)

Outputs (bundle):
- experiment_bundles/<bundle>/phase10/{data,runs,analysis}
- phase10/run_manifest.json (paths to report JSONs)
- phase10/analysis/phase10_summary.csv + .md + figures
- bundle zip

Usage:
  python scripts/run_phase10_rule_policy_bundle.py --dotenv .env
  python scripts/run_phase10_rule_policy_bundle.py --dotenv .env --mode smoke
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

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


DEFAULT_PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _load_env(dotenv_path: Optional[str], repo_root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    if dotenv_path and load_dotenv is not None:
        load_dotenv(dotenv_path, override=False)
        env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Ensure local policyops package is importable
    env["PYTHONPATH"] = str(repo_root / "src")
    return env


def _link_data_dir(data_out: Path, run_out: Path) -> None:
    """Ensure compare out_dir has a ./data pointing to generated data."""
    src = data_out / "data"
    dst = run_out / "data"
    if dst.exists() or dst.is_symlink():
        # If it already points correctly, keep it; otherwise replace.
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def _discover_compare_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Return (report_json, sweep_csv, prompts_dir, raw_dir) if discovered."""
    report_json = None
    sweep_csv = None
    prompts_dir = None
    raw_dir = None

    # report JSON typically lives at <out_dir>/runs/compare/<timestamp>.json
    runs_dir = out_dir / "runs" / "compare"
    if runs_dir.exists():
        cand = sorted(runs_dir.glob("*.json"))
        if cand:
            report_json = cand[-1]

    # sweep csv: results_context_budget_sweep.csv in out_dir or compare run dir
    cand1 = out_dir / "results_context_budget_sweep.csv"
    if cand1.exists():
        sweep_csv = cand1
    else:
        if runs_dir.exists():
            cand2 = sorted(runs_dir.glob("*/results_context_budget_sweep.csv"))
            if cand2:
                sweep_csv = cand2[-1]

    # prompts/raw are usually in out_dir/prompts, out_dir/raw_outputs
    if (out_dir / "prompts").exists():
        prompts_dir = out_dir / "prompts"
    if (out_dir / "raw_outputs").exists():
        raw_dir = out_dir / "raw_outputs"

    return report_json, sweep_csv, prompts_dir, raw_dir


@dataclass
class RunEntry:
    pivot_type: str
    variant: str               # baseline | goc_fixed_light | goc_fixed_heavy | goc_adaptive_heavy | goc_rule_v1
    pool_size: int
    policy: str                # fixed | adaptive_pivot | rule_v1
    fixed_k: Optional[int]
    fixed_h: Optional[int]
    default_k: Optional[int]
    default_h: Optional[int]
    pivot_k: Optional[int]
    pivot_h: Optional[int]
    report_json: str            # relative to phase10 root
    compare_root: str           # relative to phase10 root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--parallel_workers", type=int, default=12)
    ap.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--pivot_rate", type=float, default=0.2)
    ap.add_argument("--pivot_types", type=str, default="retention_flip entity_switch constraint_add")
    ap.add_argument("--pool_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event_trace_sample_rate", type=float, default=0.2)
    ap.add_argument("--mode", choices=["smoke", "main"], default="main")

    # sizing
    ap.add_argument("--n_threads", type=int, default=40)
    ap.add_argument("--total_tasks", type=int, default=120)

    args = ap.parse_args()

    if args.mode == "smoke":
        args.n_threads = min(args.n_threads, 10)
        args.total_tasks = min(args.total_tasks, 20)
        args.parallel_workers = min(args.parallel_workers, 4)
        args.pivot_rate = 0.5
        args.event_trace_sample_rate = 1.0

    repo_root = Path(__file__).resolve().parents[1]
    bundles_root = repo_root / "experiment_bundles"
    _ensure_dir(bundles_root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(4).hex()
    bundle_name = f"goc_policyops_phase10_rule_{stamp}_{rand}"
    bundle_root = bundles_root / bundle_name
    phase10_root = bundle_root / "phase10"
    data_root = phase10_root / "data"
    runs_root = phase10_root / "runs"
    analysis_root = phase10_root / "analysis"
    figs_root = analysis_root / "figures"

    _ensure_dir(data_root)
    _ensure_dir(runs_root)
    _ensure_dir(analysis_root)
    _ensure_dir(figs_root)

    env = _load_env(args.dotenv, repo_root)

    # GoC selection quality flags (carry over from Phase7+)
    base_goc_flags = [
        "--goc_activity_filter",
        "--goc_anchor_top1_lexical",
        "--goc_mmr_lambda", "0.35",
    ]

    # Rule mapping (pivot_type -> (pivot_k, pivot_h))
    rule_map: Dict[str, Tuple[int, int]] = {
        "retention_flip": (16, 3),
        "entity_switch": (4, 3),
        "constraint_add": (4, 2),
    }

    manifest: Dict[str, Any] = {
        "phase": 10,
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
        "model": args.model,
        "preset": args.preset,
        "budget": args.budget,
        "pivot_rate": args.pivot_rate,
        "pivot_types": args.pivot_types.split(),
        "pool_size": args.pool_size,
        "seed": args.seed,
        "n_threads": args.n_threads,
        "total_tasks": args.total_tasks,
        "parallel_workers": args.parallel_workers,
        "event_trace_sample_rate": args.event_trace_sample_rate,
        "rule_map": rule_map,
        "runs": [],
    }

    pivot_types = args.pivot_types.split()

    for pivot_type in pivot_types:
        print(f"\n=== Phase10 pivot_type={pivot_type} ===")

        # 1) generate data (per pivot_type)
        data_out = data_root / pivot_type
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

        # 2) baseline compare (full + similarity_only)
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
            "--dotenv", args.dotenv or ".env",
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
            report_json=str(base_report.relative_to(phase10_root)),
            compare_root=str(base_out.relative_to(phase10_root)),
        )))

        def _run_goc_compare(*, out_dir: Path, variant: str, policy: str,
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
                "--goc_candidate_pool_size", str(args.pool_size),
                "--goc_unfold_budget_mode", "nodes_and_hops",
                "--dotenv", args.dotenv or ".env",
                "--out_dir", str(out_dir),
            ] + base_goc_flags

            fk = fh = dk = dh = pk = ph = None

            if policy == "fixed":
                if fixed is None:
                    raise ValueError("fixed policy requires fixed=(K,H)")
                fk, fh = fixed
                cmd += [
                    "--goc_unfold_policy", "fixed",
                    "--goc_unfold_max_nodes", str(fk),
                    "--goc_unfold_hops", str(fh),
                ]
            elif policy in ("adaptive_pivot", "rule_v1"):
                if adaptive is None:
                    raise ValueError("adaptive policy requires adaptive=((defK,defH),(pivK,pivH))")
                (dk, dh), (pk, ph) = adaptive
                cmd += [
                    "--goc_unfold_policy", "adaptive_pivot",
                    "--goc_unfold_default_max_nodes", str(dk),
                    "--goc_unfold_default_hops", str(dh),
                    "--goc_unfold_pivot_max_nodes", str(pk),
                    "--goc_unfold_pivot_hops", str(ph),
                ]
            else:
                raise ValueError(f"Unknown policy: {policy}")

            _run(cmd, cwd=repo_root, env=env)
            report_json, _, _, _ = _discover_compare_artifacts(out_dir)
            if not report_json:
                raise RuntimeError(f"Could not find report json for {variant}: {out_dir}")

            manifest["runs"].append(asdict(RunEntry(
                pivot_type=pivot_type,
                variant=variant,
                pool_size=args.pool_size,
                policy=policy,
                fixed_k=fk,
                fixed_h=fh,
                default_k=dk,
                default_h=dh,
                pivot_k=pk,
                pivot_h=ph,
                report_json=str(report_json.relative_to(phase10_root)),
                compare_root=str(out_dir.relative_to(phase10_root)),
            )))

        # 3) GoC fixed light (K4,H2)
        out_light = runs_root / pivot_type / f"goc_fixed_light_pool{args.pool_size}_K4_H2"
        _run_goc_compare(out_dir=out_light, variant="goc_fixed_light", policy="fixed", fixed=(4, 2))

        # 4) GoC fixed heavy (K16,H3)
        out_heavy = runs_root / pivot_type / f"goc_fixed_heavy_pool{args.pool_size}_K16_H3"
        _run_goc_compare(out_dir=out_heavy, variant="goc_fixed_heavy", policy="fixed", fixed=(16, 3))

        # 5) Naive adaptive heavy: default(4,2) pivot(16,3)
        out_adapt_heavy = runs_root / pivot_type / f"goc_adaptive_heavy_pool{args.pool_size}_D4_2_P16_3"
        _run_goc_compare(out_dir=out_adapt_heavy, variant="goc_adaptive_heavy", policy="adaptive_pivot", adaptive=((4, 2), (16, 3)))

        # 6) Rule policy (v1): default(4,2) pivot(rule_map[pivot_type])
        piv_k, piv_h = rule_map.get(pivot_type, (16, 3))
        out_rule = runs_root / pivot_type / f"goc_rule_v1_pool{args.pool_size}_D4_2_P{piv_k}_{piv_h}"
        _run_goc_compare(out_dir=out_rule, variant="goc_rule_v1", policy="rule_v1", adaptive=((4, 2), (piv_k, piv_h)))

    # write manifest
    manifest_path = phase10_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # analyze
    analyze_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_phase10_rule_policy.py"),
        "--phase10_root", str(phase10_root),
        "--out_dir", str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    # write index
    idx_lines = []
    idx_lines.append(f"# Phase 10 Rule Policy Bundle: {bundle_name}")
    idx_lines.append("")
    idx_lines.append("## Outputs")
    idx_lines.append(f"- Manifest: {(phase10_root / 'run_manifest.json').relative_to(bundle_root)}")
    idx_lines.append(f"- Analysis CSV: {(analysis_root / 'phase10_summary.csv').relative_to(bundle_root)}")
    idx_lines.append(f"- Analysis MD: {(analysis_root / 'phase10_summary.md').relative_to(bundle_root)}")
    idx_lines.append("")
    idx_lines.append("## Run configuration")
    idx_lines.append(f"- preset={args.preset} model={args.model} budget={args.budget} pivot_rate={args.pivot_rate}")
    idx_lines.append(f"- n_threads={args.n_threads} total_tasks={args.total_tasks} parallel_workers={args.parallel_workers}")
    idx_lines.append(f"- pool_size={args.pool_size}")
    idx_lines.append(f"- rule_map={rule_map}")
    (bundle_root / "INDEX.md").write_text("\n".join(idx_lines), encoding="utf-8")

    # zip bundle
    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = bundle_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(bundle_root), "zip", root_dir=str(bundle_root))
    print(f"\nBundle folder: {bundle_root}")
    print(f"Zip path: {zip_path}")


if __name__ == "__main__":
    main()
