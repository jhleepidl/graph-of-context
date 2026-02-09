#!/usr/bin/env python3
"""Phase 11: E2E (episode 1~3) evaluation bundle runner."""

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
    env["PYTHONPATH"] = str(repo_root / "src/benchmarks/policyops_arena_v0/src") + os.pathsep + str(repo_root / "src")
    return env


def _link_data_dir(data_out: Path, run_out: Path) -> None:
    """Ensure compare out_dir has a ./data pointing to generated data."""
    src = data_out / "data"
    dst = run_out / "data"
    if dst.exists() or dst.is_symlink():
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


def _discover_report_json(out_dir: Path) -> Optional[Path]:
    runs_dir = out_dir / "runs" / "compare"
    if not runs_dir.exists():
        return None
    cand = sorted(runs_dir.glob("*.json"))
    return cand[-1] if cand else None


@dataclass
class RunEntry:
    pivot_type: str
    variant: str
    pool_size: int
    policy: str
    fixed_k: Optional[int] = None
    fixed_h: Optional[int] = None
    default_k: Optional[int] = None
    default_h: Optional[int] = None
    pivot_k: Optional[int] = None
    pivot_h: Optional[int] = None
    report_json: str = ""
    compare_root: str = ""


def _rule_map_v1(pivot_type: str) -> Tuple[int, int]:
    """Pivot-only (K,H) for rule v1."""
    if pivot_type == "retention_flip":
        return 16, 3
    if pivot_type == "entity_switch":
        return 4, 3
    if pivot_type == "constraint_add":
        return 4, 2
    return 4, 2


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
    bundle_name = f"goc_policyops_phase11_e2e_{stamp}_{rand}"
    bundle_root = bundles_root / bundle_name

    phase11_root = bundle_root / "phase11"
    data_root = phase11_root / "data"
    runs_root = phase11_root / "runs"
    analysis_root = phase11_root / "analysis"

    _ensure_dir(data_root)
    _ensure_dir(runs_root)
    _ensure_dir(analysis_root)

    env = _load_env(args.dotenv, repo_root)

    manifest: Dict[str, Any] = {
        "phase": 11,
        "mode": args.mode,
        "model": args.model,
        "preset": args.preset,
        "budget": args.budget,
        "pivot_rate": args.pivot_rate,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "parallel_workers": args.parallel_workers,
        "n_threads": args.n_threads,
        "total_tasks": args.total_tasks,
        "runs": [],
    }

    base_goc_flags = [
        "--goc_activity_filter",
        "--goc_anchor_top1_lexical",
        "--goc_mmr_lambda", "0.35",
    ]

    pivot_types = args.pivot_types.split()

    for pivot_type in pivot_types:
        print(f"\n=== Phase11 pivot_type={pivot_type} ===")

        # 1) generate data
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
        base_report = _discover_report_json(base_out)
        if not base_report:
            raise RuntimeError(f"Could not find report json for baseline: {base_out}")

        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="baseline",
            pool_size=0,
            policy="fixed",
            report_json=str(base_report.relative_to(phase11_root)),
            compare_root=str(base_out.relative_to(phase11_root)),
        )))

        # Common compare header for goc variants
        def _goc_compare(out_dir: Path, extra_flags: List[str]) -> Path:
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
            ] + base_goc_flags + extra_flags + [
                "--dotenv", args.dotenv or ".env",
                "--out_dir", str(out_dir),
            ]
            _run(cmd, cwd=repo_root, env=env)
            rep = _discover_report_json(out_dir)
            if not rep:
                raise RuntimeError(f"Could not find report json: {out_dir}")
            return rep

        # 3) goc_fixed_light (K4,H2)
        out_light = runs_root / pivot_type / f"goc_fixed_light_pool{args.pool_size}_K4_H2"
        rep_light = _goc_compare(out_light, [
            "--goc_unfold_policy", "fixed",
            "--goc_unfold_max_nodes", "4",
            "--goc_unfold_hops", "2",
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="goc_fixed_light",
            pool_size=args.pool_size,
            policy="fixed",
            fixed_k=4,
            fixed_h=2,
            report_json=str(rep_light.relative_to(phase11_root)),
            compare_root=str(out_light.relative_to(phase11_root)),
        )))

        # 4) goc_fixed_heavy (K16,H3)
        out_heavy = runs_root / pivot_type / f"goc_fixed_heavy_pool{args.pool_size}_K16_H3"
        rep_heavy = _goc_compare(out_heavy, [
            "--goc_unfold_policy", "fixed",
            "--goc_unfold_max_nodes", "16",
            "--goc_unfold_hops", "3",
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="goc_fixed_heavy",
            pool_size=args.pool_size,
            policy="fixed",
            fixed_k=16,
            fixed_h=3,
            report_json=str(rep_heavy.relative_to(phase11_root)),
            compare_root=str(out_heavy.relative_to(phase11_root)),
        )))

        # 5) goc_adaptive_heavy: D(4,2) -> P(16,3)
        out_adapt = runs_root / pivot_type / f"goc_adaptive_heavy_pool{args.pool_size}_D4_2_P16_3"
        rep_adapt = _goc_compare(out_adapt, [
            "--goc_unfold_policy", "adaptive_pivot",
            "--goc_unfold_max_nodes", "4",
            "--goc_unfold_hops", "2",
            "--goc_unfold_pivot_max_nodes", "16",
            "--goc_unfold_pivot_hops", "3",
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="goc_adaptive_heavy",
            pool_size=args.pool_size,
            policy="adaptive_pivot",
            default_k=4,
            default_h=2,
            pivot_k=16,
            pivot_h=3,
            report_json=str(rep_adapt.relative_to(phase11_root)),
            compare_root=str(out_adapt.relative_to(phase11_root)),
        )))

        # 6) goc_rule_v1: D(4,2) -> P(rule)
        pk, ph = _rule_map_v1(pivot_type)
        out_rule = runs_root / pivot_type / f"goc_rule_v1_pool{args.pool_size}_D4_2_P{pk}_{ph}"
        rep_rule = _goc_compare(out_rule, [
            "--goc_unfold_policy", "adaptive_pivot",
            "--goc_unfold_max_nodes", "4",
            "--goc_unfold_hops", "2",
            "--goc_unfold_pivot_max_nodes", str(pk),
            "--goc_unfold_pivot_hops", str(ph),
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            variant="goc_rule_v1",
            pool_size=args.pool_size,
            policy="adaptive_pivot",
            default_k=4,
            default_h=2,
            pivot_k=pk,
            pivot_h=ph,
            report_json=str(rep_rule.relative_to(phase11_root)),
            compare_root=str(out_rule.relative_to(phase11_root)),
        )))

    # Save manifest
    (phase11_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Analyze
    analyze_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_phase11_e2e.py"),
        "--phase11_root", str(phase11_root),
        "--out_dir", str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    # INDEX
    idx_lines: List[str] = []
    idx_lines.append(f"# Phase 11 E2E bundle: {bundle_name}\n")
    idx_lines.append(f"- preset={args.preset} model={args.model} budget={args.budget} pool={args.pool_size} pivot_rate={args.pivot_rate}")
    idx_lines.append(f"- n_threads={args.n_threads} total_tasks={args.total_tasks} parallel_workers={args.parallel_workers}")
    idx_lines.append("\n## Outputs")
    idx_lines.append(f"- phase11_root: {phase11_root}")
    idx_lines.append(f"- manifest: {phase11_root / 'run_manifest.json'}")
    idx_lines.append(f"- analysis: {analysis_root}")
    (bundle_root / "INDEX.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    # Zip bundle folder
    zip_path = shutil.make_archive(str(bundle_root), "zip", root_dir=str(bundle_root.parent), base_dir=bundle_root.name)
    print(f"\nBundle folder: {bundle_root}")
    print(f"Zip path: {zip_path}")


if __name__ == "__main__":
    main()
