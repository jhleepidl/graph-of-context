#!/usr/bin/env python3
"""Phase 13 runner: PolicyOps end-to-end (episode 1~3) evaluation.

Derived from Phase 12, with Phase 13 GoC knobs turned on:
- Episode 3 closed-book universe restriction: `--goc_closedbook_universe` (default: memory)
- Commit retrieval improvement: `--goc_graph_frontier*` (default: enabled)

Outputs a self-contained experiment bundle under experiment_bundles/ and zips it.

Usage:
  python scripts/run_phase13_e2e_universe_frontier_bundle.py --dotenv .env

Notes:
- This runner intentionally keeps baseline comparisons unchanged.
- Use --include_ablations to additionally run GoC w/ frontier off and/or universe=world.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional
from policyops_bundle_layout import build_bundle_quick_access


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _link_data_dir(src_root: Path, dst_out_dir: Path) -> None:
    """Ensure compare out_dir has a data/ directory (worlds/ + tasks/).

    The generator typically writes <out_dir>/data/worlds/*.jsonl.
    Some earlier runners passed either <out_dir> or <out_dir>/data. We auto-detect.
    """

    candidates = [src_root, src_root / "data"]
    src_data: Optional[Path] = None
    for cand in candidates:
        if (cand / "worlds").exists() or (cand / "tasks").exists():
            src_data = cand
            break
    if src_data is None:
        src_data = src_root / "data" if (src_root / "data").exists() else src_root

    dst_data = dst_out_dir / "data"

    if dst_data.exists() or dst_data.is_symlink():
        try:
            ok = (dst_data / "worlds").exists() or (dst_data / "tasks").exists()
        except Exception:
            ok = False
        if ok:
            return
        if dst_data.is_symlink() or dst_data.is_file():
            dst_data.unlink(missing_ok=True)
        else:
            shutil.rmtree(dst_data, ignore_errors=True)

    try:
        dst_data.symlink_to(src_data, target_is_directory=True)
    except Exception:
        shutil.copytree(src_data, dst_data)


def _discover_report_json(out_dir: Path) -> Optional[Path]:
    cand = list(out_dir.rglob("runs/compare/*.json"))
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


@dataclass
class RunEntry:
    pivot_type: str
    pivot_rate: float
    variant: str
    method: str
    pool_size: int
    budget: int
    commit_anchor_source: str
    commit_anchor_max_ids: int
    commit_anchor_require_opened: bool
    commit_anchor_fallback_opened: bool
    goc_closedbook_universe: str
    goc_graph_frontier: bool
    goc_graph_frontier_hops: int
    goc_graph_frontier_max_nodes: int
    goc_graph_frontier_seed_top_n: int
    goc_graph_frontier_score_frac: float
    report_json: str
    compare_root: str
    # optional knobs for goc
    policy: str = ""
    fixed_k: int = 0
    fixed_h: int = 0
    default_k: int = 0
    default_h: int = 0
    pivot_k: int = 0
    pivot_h: int = 0


def _rule_map_v1(pivot_type: str) -> tuple[int, int]:
    """Rule policy derived from Phase 9/10 findings."""
    if pivot_type == "retention_flip":
        return (16, 3)
    if pivot_type == "entity_switch":
        return (4, 3)
    if pivot_type == "constraint_add":
        return (4, 2)
    return (16, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="threaded_v1_3_fu_decoy_calib_jitter_n10")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--pool_size", type=int, default=64)
    ap.add_argument("--n_threads", type=int, default=40)
    ap.add_argument("--total_tasks", type=int, default=120)
    ap.add_argument("--pivot_rate", type=float, default=0.2, help="pivot rate for non-retention_flip")
    ap.add_argument("--pivot_rate_retention_flip", type=float, default=0.5)
    ap.add_argument("--parallel_workers", type=int, default=12)
    ap.add_argument("--event_trace_sample_rate", type=float, default=0.2)
    ap.add_argument("--save_goc_graph", action="store_true", default=True)
    ap.add_argument("--goc_graph_sample_rate", type=float, default=0.2)
    ap.add_argument("--dotenv", type=str, default=".env")

    # Phase 12 commit-anchor knobs
    ap.add_argument(
        "--commit_anchor_source",
        type=str,
        choices=["opened_supporting", "model_evidence", "hybrid"],
        default="model_evidence",
    )
    ap.add_argument("--commit_anchor_max_ids", type=int, default=4)
    ap.add_argument("--commit_anchor_require_opened", action="store_true", default=True)
    ap.add_argument("--no_commit_anchor_require_opened", action="store_false", dest="commit_anchor_require_opened")
    ap.add_argument("--commit_anchor_fallback_opened", action="store_true", default=False)
    ap.add_argument("--no_commit_anchor_fallback_opened", action="store_false", dest="commit_anchor_fallback_opened")

    # Phase 13 knobs
    ap.add_argument("--goc_closedbook_universe", choices=["memory", "world"], default="memory")
    ap.add_argument("--goc_graph_frontier", action="store_true", default=True)
    ap.add_argument("--no_goc_graph_frontier", action="store_false", dest="goc_graph_frontier")
    ap.add_argument("--goc_graph_frontier_hops", type=int, default=2)
    ap.add_argument("--goc_graph_frontier_max_nodes", type=int, default=50)
    ap.add_argument("--goc_graph_frontier_seed_top_n", type=int, default=6)
    ap.add_argument("--goc_graph_frontier_score_frac", type=float, default=0.7)

    # Optional ablations
    ap.add_argument("--include_ablations", action="store_true", help="Run extra GoC ablations (frontier off / universe=world)")

    # Debug sizing
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--debug_n", type=int, default=0, help="if >0, pass --debug_n to compare (subsample tasks)")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bundle_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"goc_policyops_phase13_e2e_universe_frontier_{ts}_{bundle_id}"
    bundle_root = repo_root / "experiment_bundles" / bundle_name
    phase13_root = bundle_root / "phase13"
    data_root = phase13_root / "data"
    runs_root = phase13_root / "runs"
    analysis_root = phase13_root / "analysis"
    _ensure_dir(data_root)
    _ensure_dir(runs_root)
    _ensure_dir(analysis_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["PYTHONUNBUFFERED"] = "1"

    # Network robustness defaults for long parallel runs (override via env vars as needed).
    env.setdefault("POLICYOPS_OPENAI_TIMEOUT", "180")
    env.setdefault("POLICYOPS_OPENAI_MAX_RETRIES", "8")
    env.setdefault("POLICYOPS_OPENAI_BACKOFF_BASE", "1.0")
    env.setdefault("POLICYOPS_OPENAI_BACKOFF_MAX", "30.0")

    # Sizing
    n_threads = args.n_threads
    total_tasks = args.total_tasks
    if args.smoke:
        n_threads = min(8, n_threads)
        total_tasks = min(24, total_tasks)

    pivot_types = ["retention_flip", "entity_switch", "constraint_add"]

    manifest: Dict[str, Any] = {
        "bundle_name": bundle_name,
        "created_at": ts,
        "preset": args.preset,
        "model": args.model,
        "budget": args.budget,
        "pool_size": args.pool_size,
        "n_threads": n_threads,
        "total_tasks": total_tasks,
        "parallel_workers": args.parallel_workers,
        "event_trace_sample_rate": args.event_trace_sample_rate,
        "save_goc_graph": bool(args.save_goc_graph),
        "goc_graph_sample_rate": args.goc_graph_sample_rate,
        "commit_anchor_source": args.commit_anchor_source,
        "commit_anchor_max_ids": args.commit_anchor_max_ids,
        "commit_anchor_require_opened": bool(args.commit_anchor_require_opened),
        "commit_anchor_fallback_opened": bool(args.commit_anchor_fallback_opened),
        "goc_closedbook_universe": args.goc_closedbook_universe,
        "goc_graph_frontier": bool(args.goc_graph_frontier),
        "goc_graph_frontier_hops": args.goc_graph_frontier_hops,
        "goc_graph_frontier_max_nodes": args.goc_graph_frontier_max_nodes,
        "goc_graph_frontier_seed_top_n": args.goc_graph_frontier_seed_top_n,
        "goc_graph_frontier_score_frac": args.goc_graph_frontier_score_frac,
        "runs": [],
    }

    def _commit_anchor_flags() -> List[str]:
        flags = [
            "--commit_anchor_source", args.commit_anchor_source,
            "--commit_anchor_max_ids", str(args.commit_anchor_max_ids),
        ]
        flags.append("--commit_anchor_require_opened" if args.commit_anchor_require_opened else "--no_commit_anchor_require_opened")
        flags.append("--commit_anchor_fallback_opened" if args.commit_anchor_fallback_opened else "--no_commit_anchor_fallback_opened")
        return flags

    base_compare_flags = [
        "--preset", args.preset,
        "--llm", "openai",
        "--model", args.model,
        "--judge", "symbolic_packed",
        "--thread_context_budget_sweep", str(args.budget),
        "--parallel_workers", str(args.parallel_workers),
        "--n_threads", str(n_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate", str(args.event_trace_sample_rate),
    ] + _commit_anchor_flags()

    if args.save_goc_graph:
        base_compare_flags += [
            "--save_goc_graph",
            "--goc_graph_sample_rate", str(args.goc_graph_sample_rate),
        ]

    if args.debug_n and args.debug_n > 0:
        base_compare_flags += ["--debug_n", str(args.debug_n)]

    # GoC base flags
    base_goc_flags = [
        "--goc_candidate_pool_size", str(args.pool_size),
        "--goc_unfold_budget_mode", "nodes_and_hops",
        "--goc_closedbook_universe", str(args.goc_closedbook_universe),
    ]

    # Frontier flags
    if args.goc_graph_frontier:
        base_goc_flags += [
            "--goc_graph_frontier",
            "--goc_graph_frontier_hops", str(args.goc_graph_frontier_hops),
            "--goc_graph_frontier_max_nodes", str(args.goc_graph_frontier_max_nodes),
            "--goc_graph_frontier_seed_top_n", str(args.goc_graph_frontier_seed_top_n),
            "--goc_graph_frontier_score_frac", str(args.goc_graph_frontier_score_frac),
        ]
    else:
        base_goc_flags += ["--no_goc_graph_frontier"]

    for pivot_type in pivot_types:
        pivot_rate = float(args.pivot_rate_retention_flip if pivot_type == "retention_flip" else args.pivot_rate)
        data_out = data_root / pivot_type
        _ensure_dir(data_out)

        # 1) Generate
        gen_cmd = [
            sys.executable, "-u", "-m", "policyops.run", "generate",
            "--preset", args.preset,
            "--seed", "0",
            "--n_threads", str(n_threads),
            "--n_tasks", str(total_tasks),
            "--pivot_rate", str(pivot_rate),
            "--pivot_type", pivot_type,
            "--out_dir", str(data_out),
        ]
        _run(gen_cmd, cwd=repo_root, env=env)

        # 2) Baselines: full + similarity_only
        out_base = runs_root / pivot_type / "baseline"
        _ensure_dir(out_base)
        _link_data_dir(data_out, out_base)
        base_cmd = [sys.executable, "-u", "-m", "policyops.run", "compare"] + base_compare_flags + [
            "--methods", "full", "similarity_only",
            "--dotenv", args.dotenv,
            "--out_dir", str(out_base),
        ]
        _run(base_cmd, cwd=repo_root, env=env)
        rep_base = _discover_report_json(out_base)
        if not rep_base:
            raise RuntimeError(f"Could not find baseline report json: {out_base}")
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            pivot_rate=pivot_rate,
            variant="baseline",
            method="full/similarity_only",
            pool_size=args.pool_size,
            budget=args.budget,
            commit_anchor_source=args.commit_anchor_source,
            commit_anchor_max_ids=args.commit_anchor_max_ids,
            commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
            commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
            goc_closedbook_universe=str(args.goc_closedbook_universe),
            goc_graph_frontier=bool(args.goc_graph_frontier),
            goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
            goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
            goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
            goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
            report_json=str(rep_base.relative_to(phase13_root)),
            compare_root=str(out_base.relative_to(phase13_root)),
        )))

        def _goc_compare(out_dir: Path, extra_flags: List[str]) -> Path:
            _ensure_dir(out_dir)
            _link_data_dir(data_out, out_dir)
            cmd = [sys.executable, "-u", "-m", "policyops.run", "compare"] + base_compare_flags + [
                "--methods", "goc",
            ] + base_goc_flags + extra_flags + [
                "--dotenv", args.dotenv,
                "--out_dir", str(out_dir),
            ]
            _run(cmd, cwd=repo_root, env=env)
            rep = _discover_report_json(out_dir)
            if not rep:
                raise RuntimeError(f"Could not find report json: {out_dir}")
            return rep

        # Core GoC variants (same as Phase 12)
        out_light = runs_root / pivot_type / f"goc_fixed_light_pool{args.pool_size}_K4_H2"
        rep_light = _goc_compare(out_light, [
            "--goc_unfold_policy", "fixed",
            "--goc_unfold_max_nodes", "4",
            "--goc_unfold_hops", "2",
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            pivot_rate=pivot_rate,
            variant="goc_fixed_light",
            method="goc",
            pool_size=args.pool_size,
            budget=args.budget,
            commit_anchor_source=args.commit_anchor_source,
            commit_anchor_max_ids=args.commit_anchor_max_ids,
            commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
            commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
            goc_closedbook_universe=str(args.goc_closedbook_universe),
            goc_graph_frontier=bool(args.goc_graph_frontier),
            goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
            goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
            goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
            goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
            policy="fixed",
            fixed_k=4,
            fixed_h=2,
            report_json=str(rep_light.relative_to(phase13_root)),
            compare_root=str(out_light.relative_to(phase13_root)),
        )))

        out_heavy = runs_root / pivot_type / f"goc_fixed_heavy_pool{args.pool_size}_K16_H3"
        rep_heavy = _goc_compare(out_heavy, [
            "--goc_unfold_policy", "fixed",
            "--goc_unfold_max_nodes", "16",
            "--goc_unfold_hops", "3",
        ])
        manifest["runs"].append(asdict(RunEntry(
            pivot_type=pivot_type,
            pivot_rate=pivot_rate,
            variant="goc_fixed_heavy",
            method="goc",
            pool_size=args.pool_size,
            budget=args.budget,
            commit_anchor_source=args.commit_anchor_source,
            commit_anchor_max_ids=args.commit_anchor_max_ids,
            commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
            commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
            goc_closedbook_universe=str(args.goc_closedbook_universe),
            goc_graph_frontier=bool(args.goc_graph_frontier),
            goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
            goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
            goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
            goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
            policy="fixed",
            fixed_k=16,
            fixed_h=3,
            report_json=str(rep_heavy.relative_to(phase13_root)),
            compare_root=str(out_heavy.relative_to(phase13_root)),
        )))

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
            pivot_rate=pivot_rate,
            variant="goc_adaptive_heavy",
            method="goc",
            pool_size=args.pool_size,
            budget=args.budget,
            commit_anchor_source=args.commit_anchor_source,
            commit_anchor_max_ids=args.commit_anchor_max_ids,
            commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
            commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
            goc_closedbook_universe=str(args.goc_closedbook_universe),
            goc_graph_frontier=bool(args.goc_graph_frontier),
            goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
            goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
            goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
            goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
            policy="adaptive_pivot",
            default_k=4,
            default_h=2,
            pivot_k=16,
            pivot_h=3,
            report_json=str(rep_adapt.relative_to(phase13_root)),
            compare_root=str(out_adapt.relative_to(phase13_root)),
        )))

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
            pivot_rate=pivot_rate,
            variant="goc_rule_v1",
            method="goc",
            pool_size=args.pool_size,
            budget=args.budget,
            commit_anchor_source=args.commit_anchor_source,
            commit_anchor_max_ids=args.commit_anchor_max_ids,
            commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
            commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
            goc_closedbook_universe=str(args.goc_closedbook_universe),
            goc_graph_frontier=bool(args.goc_graph_frontier),
            goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
            goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
            goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
            goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
            policy="adaptive_pivot",
            default_k=4,
            default_h=2,
            pivot_k=pk,
            pivot_h=ph,
            report_json=str(rep_rule.relative_to(phase13_root)),
            compare_root=str(out_rule.relative_to(phase13_root)),
        )))

        if args.include_ablations:
            # A1) Frontier OFF (diagnostic: confirms retrieval improvement is doing work)
            out_rule_nof = runs_root / pivot_type / f"goc_rule_v1_NOFRONTIER_pool{args.pool_size}_D4_2_P{pk}_{ph}"
            rep_rule_nof = _goc_compare(out_rule_nof, [
                "--no_goc_graph_frontier",
                "--goc_unfold_policy", "adaptive_pivot",
                "--goc_unfold_max_nodes", "4",
                "--goc_unfold_hops", "2",
                "--goc_unfold_pivot_max_nodes", str(pk),
                "--goc_unfold_pivot_hops", str(ph),
            ])
            manifest["runs"].append(asdict(RunEntry(
                pivot_type=pivot_type,
                pivot_rate=pivot_rate,
                variant="goc_rule_v1_NOFRONTIER",
                method="goc",
                pool_size=args.pool_size,
                budget=args.budget,
                commit_anchor_source=args.commit_anchor_source,
                commit_anchor_max_ids=args.commit_anchor_max_ids,
                commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
                commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
                goc_closedbook_universe=str(args.goc_closedbook_universe),
                goc_graph_frontier=False,
                goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
                goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
                goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
                goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
                policy="adaptive_pivot",
                default_k=4,
                default_h=2,
                pivot_k=pk,
                pivot_h=ph,
                report_json=str(rep_rule_nof.relative_to(phase13_root)),
                compare_root=str(out_rule_nof.relative_to(phase13_root)),
            )))

            # A2) Universe WORLD (diagnostic only: should increase unseen leakage)
            out_rule_world = runs_root / pivot_type / f"goc_rule_v1_UNIVERSE_WORLD_pool{args.pool_size}_D4_2_P{pk}_{ph}"
            rep_rule_world = _goc_compare(out_rule_world, [
                "--goc_closedbook_universe", "world",
                "--goc_unfold_policy", "adaptive_pivot",
                "--goc_unfold_max_nodes", "4",
                "--goc_unfold_hops", "2",
                "--goc_unfold_pivot_max_nodes", str(pk),
                "--goc_unfold_pivot_hops", str(ph),
            ])
            manifest["runs"].append(asdict(RunEntry(
                pivot_type=pivot_type,
                pivot_rate=pivot_rate,
                variant="goc_rule_v1_UNIVERSE_WORLD",
                method="goc",
                pool_size=args.pool_size,
                budget=args.budget,
                commit_anchor_source=args.commit_anchor_source,
                commit_anchor_max_ids=args.commit_anchor_max_ids,
                commit_anchor_require_opened=bool(args.commit_anchor_require_opened),
                commit_anchor_fallback_opened=bool(args.commit_anchor_fallback_opened),
                goc_closedbook_universe="world",
                goc_graph_frontier=bool(args.goc_graph_frontier),
                goc_graph_frontier_hops=int(args.goc_graph_frontier_hops),
                goc_graph_frontier_max_nodes=int(args.goc_graph_frontier_max_nodes),
                goc_graph_frontier_seed_top_n=int(args.goc_graph_frontier_seed_top_n),
                goc_graph_frontier_score_frac=float(args.goc_graph_frontier_score_frac),
                policy="adaptive_pivot",
                default_k=4,
                default_h=2,
                pivot_k=pk,
                pivot_h=ph,
                report_json=str(rep_rule_world.relative_to(phase13_root)),
                compare_root=str(out_rule_world.relative_to(phase13_root)),
            )))

    # Save manifest
    (phase13_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Analyze
    analyze_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_phase13_e2e_universe_frontier.py"),
        "--phase13_root", str(phase13_root),
        "--out_dir", str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    # INDEX
    idx_lines: List[str] = []
    idx_lines.append(f"# Phase 13 E2E (universe+frontier) bundle: {bundle_name}\n")
    idx_lines.append(
        "- "
        + f"preset={args.preset} model={args.model} budget={args.budget} pool={args.pool_size} "
        + f"n_threads={n_threads} total_tasks={total_tasks} parallel_workers={args.parallel_workers}"
    )
    idx_lines.append(
        "- "
        + f"commit_anchor_source={args.commit_anchor_source} max_ids={args.commit_anchor_max_ids} "
        + f"require_opened={bool(args.commit_anchor_require_opened)} fallback_opened={bool(args.commit_anchor_fallback_opened)}"
    )
    idx_lines.append(
        "- "
        + f"goc_closedbook_universe={args.goc_closedbook_universe} "
        + f"goc_graph_frontier={bool(args.goc_graph_frontier)} hops={args.goc_graph_frontier_hops} "
        + f"max_nodes={args.goc_graph_frontier_max_nodes} seed_top_n={args.goc_graph_frontier_seed_top_n} "
        + f"score_frac={args.goc_graph_frontier_score_frac}"
    )
    idx_lines.append(f"- pivot_rate(retention_flip)={args.pivot_rate_retention_flip} pivot_rate(other)={args.pivot_rate}")
    idx_lines.append("\n## Outputs")
    idx_lines.append(f"- phase13_root: {phase13_root}")
    idx_lines.append(f"- manifest: {phase13_root / 'run_manifest.json'}")
    idx_lines.append(f"- analysis: {analysis_root}")
    (bundle_root / "INDEX.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    # Zip bundle folder
    quick_access_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_access_dirs:
        print(f"Quick access: {qd}")

    zip_path = shutil.make_archive(str(bundle_root), "zip", root_dir=str(bundle_root.parent), base_dir=bundle_root.name)
    print(f"\nBundle folder: {bundle_root}")
    print(f"Zip path: {zip_path}")


if __name__ == "__main__":
    main()
