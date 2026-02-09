#!/usr/bin/env python3
"""
Phase 7 (AB) runner: generate multi-hop dependency data (A) + run GoC candidate-pool + (K,H) unfold sweeps (B).

This script is meant to replace ad-hoc bash for reproducible runs.
It produces:
- experiment_bundles/<bundle_name>/phase7/{data,runs,analysis}
- run_manifest.json
- analysis outputs via scripts/analyze_phase7_unfold_controls.py
- a zip of the entire bundle folder

Usage (example):
  python scripts/run_phase7_ab_unfold_controls_bundle.py --dotenv .env
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"


@dataclass
class RunEntry:
    pivot_type: str
    run_type: str  # "baseline" | "goc"
    goc_k: Optional[int]
    goc_h: Optional[int]
    out_dir: str
    compare_log: str
    rc: int
    report_json: Optional[str]
    run_dir: Optional[str]
    sweep_csv: Optional[str]
    event_trace_sample: Optional[str]
    success: bool
    note: str


def _rand_suffix(n: int = 8) -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _tee_run(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    """
    Run command and tee stdout/stderr to both console and a log file.
    """
    _ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        return int(proc.returncode)


def _zip_dir(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(root.parent))
                zf.write(full, arcname=arc)


def _symlink_data(data_src: Path, out_dir: Path) -> None:
    """
    policyops.run compare expects data at <out_dir>/data, so link it.
    """
    dst = out_dir / "data"
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(data_src.resolve())


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


def _make_env(repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src' / 'benchmarks' / 'policyops_arena_v0' / 'src'}:{repo_root / 'src'}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_generate(
    repo_root: Path,
    *,
    out_dir: Path,
    preset: str,
    seed: int,
    n_threads: int,
    pivot_rate: float,
    pivot_type: str,
    definition_dependency_depth: int,
    definition_dependency_extra_terms: int,
    force_exception_chain_depth: int,
    force_exception_chain_all_apply: bool,
) -> int:
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "generate",
        "--preset",
        preset,
        "--seed",
        str(seed),
        "--n_threads",
        str(n_threads),
        "--pivot_rate",
        str(pivot_rate),
        "--pivot_type",
        pivot_type,
        "--definition_dependency_depth",
        str(definition_dependency_depth),
        "--definition_dependency_extra_terms",
        str(definition_dependency_extra_terms),
        "--force_exception_chain_depth",
        str(force_exception_chain_depth),
    ]
    if force_exception_chain_all_apply:
        cmd.append("--force_exception_chain_all_apply")
    cmd.extend(["--out_dir", str(out_dir)])
    log_path = out_dir / "generate.log"
    env = _make_env(repo_root)
    return _tee_run(cmd, cwd=repo_root, env=env, log_path=log_path)


def _run_compare(
    repo_root: Path,
    *,
    out_dir: Path,
    preset: str,
    dotenv: str,
    model: str,
    parallel_workers: int,
    n_threads: int,
    budget: int,
    methods: List[str],
    compare_log: Path,
    # GoC knobs
    goc_k: Optional[int] = None,
    goc_h: Optional[int] = None,
    goc_candidate_pool_size: Optional[int] = None,
    event_trace_sample_rate: float = 0.25,
    goc_graph_sample_rate: float = 0.25,
) -> int:
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "policyops.run",
        "compare",
        "--preset",
        preset,
        "--llm",
        "openai",
        "--model",
        model,
        "--judge",
        "symbolic_packed",
        "--methods",
        *methods,
        "--thread_context_budget_sweep",
        str(budget),
        "--parallel_workers",
        str(parallel_workers),
        "--n_threads",
        str(n_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        str(event_trace_sample_rate),
        "--dotenv",
        dotenv,
        "--out_dir",
        str(out_dir),
    ]

    if "goc" in methods:
        cmd.extend(
            [
                "--save_goc_graph",
                "--save_goc_dot",
                "--goc_graph_sample_rate",
                str(goc_graph_sample_rate),
                "--goc_activity_filter",
                "--goc_mmr_lambda",
                "0.35",
                "--no_goc_anchor_top1_lexical",
                "--goc_unfold_budget_mode",
                "nodes_and_hops",
            ]
        )
        if goc_k is not None:
            cmd.extend(["--goc_unfold_max_nodes", str(goc_k)])
        if goc_h is not None:
            cmd.extend(["--goc_unfold_hops", str(goc_h)])
        if goc_candidate_pool_size is not None and int(goc_candidate_pool_size) > 0:
            cmd.extend(["--goc_candidate_pool_size", str(int(goc_candidate_pool_size))])

    env = _make_env(repo_root)
    return _tee_run(cmd, cwd=repo_root, env=env, log_path=compare_log)


def _run_analyzer(repo_root: Path, phase7_root: Path, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        "-u",
        "scripts/analyze_phase7_unfold_controls.py",
        "--phase7_root",
        str(phase7_root),
        "--out_dir",
        str(out_dir),
    ]
    env = _make_env(repo_root)
    return _tee_run(cmd, cwd=repo_root, env=env, log_path=out_dir / "analyze.log")


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out: List[int] = []
    for p in parts:
        if p:
            out.append(int(p))
    return out


def _parse_str_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    return [p for p in parts if p]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase7 (AB) unfold-control experiments as a bundle.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--n_threads", type=int, default=40, help="Threads in generated dataset & compare.")
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    parser.add_argument("--budget", type=int, default=4000, help="External budget given to all methods (fixed).")
    parser.add_argument("--pivot_rate", type=float, default=0.5)
    parser.add_argument("--pivot_types", type=str, default="retention_flip entity_switch constraint_add")
    parser.add_argument("--seed", type=int, default=0)

    # GoC knobs
    parser.add_argument("--goc_candidate_pool_size", type=int, default=64)
    parser.add_argument("--k_list", type=str, default="4 8 16 32")
    parser.add_argument("--h_list", type=str, default="1 2 3")

    # Sampling / artifacts
    parser.add_argument("--event_trace_sample_rate", type=float, default=0.25)
    parser.add_argument("--goc_graph_sample_rate", type=float, default=0.25)

    # (A) dependency options for generation + judge evidence
    parser.add_argument("--definition_dependency_depth", type=int, default=3)
    parser.add_argument("--definition_dependency_extra_terms", type=int, default=1)
    parser.add_argument("--force_exception_chain_depth", type=int, default=4)
    parser.add_argument("--force_exception_chain_all_apply", action="store_true")

    # Bundle root
    parser.add_argument("--bundle_root", type=str, default="", help="Optional explicit bundle root under experiment_bundles/")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exp_root = repo_root / "experiment_bundles"
    _ensure_dir(exp_root)

    if args.bundle_root:
        bundle_root = exp_root / args.bundle_root
    else:
        bundle_root = exp_root / f"goc_policyops_phase7_AB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand_suffix()}"
    phase7_root = bundle_root / "phase7"
    data_root = phase7_root / "data"
    runs_root = phase7_root / "runs"
    analysis_root = phase7_root / "analysis"
    for p in (data_root, runs_root, analysis_root):
        _ensure_dir(p)

    pivot_types = _parse_str_list(args.pivot_types)
    k_list = _parse_int_list(args.k_list)
    h_list = _parse_int_list(args.h_list)

    entries: List[RunEntry] = []

    # 1) Generate per pivot type (A)
    for pivot in pivot_types:
        out_dir = data_root / pivot
        _ensure_dir(out_dir)
        print(f"\n=== GENERATE pivot_type={pivot} -> {out_dir} ===")
        rc = _run_generate(
            repo_root,
            out_dir=out_dir,
            preset=args.preset,
            seed=int(args.seed),
            n_threads=int(args.n_threads),
            pivot_rate=float(args.pivot_rate),
            pivot_type=pivot,
            definition_dependency_depth=int(args.definition_dependency_depth),
            definition_dependency_extra_terms=int(args.definition_dependency_extra_terms),
            force_exception_chain_depth=int(args.force_exception_chain_depth),
            force_exception_chain_all_apply=bool(args.force_exception_chain_all_apply),
        )
        (out_dir / "rc.txt").write_text(str(rc), encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"generate failed for pivot_type={pivot} (see {out_dir / 'generate.log'})")

    # 2) Compare baselines + GoC sweeps (B)
    for pivot in pivot_types:
        gen_dir = data_root / pivot
        data_dir = gen_dir / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Expected data_dir missing: {data_dir}")

        # Baselines
        baseline_out = runs_root / pivot / "baseline"
        _ensure_dir(baseline_out)
        _symlink_data(data_dir, baseline_out)
        baseline_log = baseline_out / "compare_openai.log"
        print(f"\n=== COMPARE baseline pivot_type={pivot} -> {baseline_out} ===")
        rc = _run_compare(
            repo_root,
            out_dir=baseline_out,
            preset=args.preset,
            dotenv=args.dotenv,
            model=args.model,
            parallel_workers=int(args.parallel_workers),
            n_threads=int(args.n_threads),
            budget=int(args.budget),
            methods=["full", "similarity_only"],
            compare_log=baseline_log,
            event_trace_sample_rate=float(args.event_trace_sample_rate),
            goc_graph_sample_rate=float(args.goc_graph_sample_rate),
        )
        (baseline_out / "rc.txt").write_text(str(rc), encoding="utf-8")
        rep, run_dir, sweep, trace = _discover_compare_artifacts(baseline_out)
        success = bool(rc == 0 and rep is not None and sweep is not None)
        note = "ok"
        if trace is None:
            note = "missing_event_trace"
        entries.append(
            RunEntry(
                pivot_type=pivot,
                run_type="baseline",
                goc_k=None,
                goc_h=None,
                out_dir=str(baseline_out),
                compare_log=str(baseline_log),
                rc=rc,
                report_json=str(rep) if rep else None,
                run_dir=str(run_dir) if run_dir else None,
                sweep_csv=str(sweep) if sweep else None,
                event_trace_sample=str(trace) if trace else None,
                success=success,
                note=note,
            )
        )

        # GoC grid
        for k in k_list:
            for h in h_list:
                goc_out = runs_root / pivot / f"goc_K{k}_H{h}"
                _ensure_dir(goc_out)
                _symlink_data(data_dir, goc_out)
                goc_log = goc_out / "compare_openai.log"
                print(f"\n=== COMPARE goc pivot_type={pivot} K={k} H={h} pool={args.goc_candidate_pool_size} -> {goc_out} ===")
                rc = _run_compare(
                    repo_root,
                    out_dir=goc_out,
                    preset=args.preset,
                    dotenv=args.dotenv,
                    model=args.model,
                    parallel_workers=int(args.parallel_workers),
                    n_threads=int(args.n_threads),
                    budget=int(args.budget),
                    methods=["goc"],
                    compare_log=goc_log,
                    goc_k=int(k),
                    goc_h=int(h),
                    goc_candidate_pool_size=int(args.goc_candidate_pool_size),
                    event_trace_sample_rate=float(args.event_trace_sample_rate),
                    goc_graph_sample_rate=float(args.goc_graph_sample_rate),
                )
                (goc_out / "rc.txt").write_text(str(rc), encoding="utf-8")
                rep, run_dir, sweep, trace = _discover_compare_artifacts(goc_out)
                success = bool(rc == 0 and rep is not None and sweep is not None)
                notes: List[str] = [f"goc_candidate_pool_size={int(args.goc_candidate_pool_size)}"]
                if trace is None:
                    notes.append("missing_event_trace")
                entries.append(
                    RunEntry(
                        pivot_type=pivot,
                        run_type="goc",
                        goc_k=int(k),
                        goc_h=int(h),
                        out_dir=str(goc_out),
                        compare_log=str(goc_log),
                        rc=rc,
                        report_json=str(rep) if rep else None,
                        run_dir=str(run_dir) if run_dir else None,
                        sweep_csv=str(sweep) if sweep else None,
                        event_trace_sample=str(trace) if trace else None,
                        success=success,
                        note=";".join(notes),
                    )
                )

    # 3) run_manifest + analyzer
    manifest_path = phase7_root / "run_manifest.json"
    manifest_obj = {
        "phase": "phase7_unfold_controls_AB",
        "preset": args.preset,
        "pivot_rate": float(args.pivot_rate),
        "budget": int(args.budget),
        "parallel_workers": int(args.parallel_workers),
        "n_threads": int(args.n_threads),
        "goc_candidate_pool_size": int(args.goc_candidate_pool_size),
        "k_list": k_list,
        "h_list": h_list,
        "generator_cfg": {
            "definition_dependency_depth": int(args.definition_dependency_depth),
            "definition_dependency_extra_terms": int(args.definition_dependency_extra_terms),
            "force_exception_chain_depth": int(args.force_exception_chain_depth),
            "force_exception_chain_all_apply": bool(args.force_exception_chain_all_apply),
        },
        "runs": [asdict(e) for e in entries],
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path} (runs={len(entries)})")

    print(f"\n=== ANALYZE -> {analysis_root} ===")
    analyze_rc = _run_analyzer(repo_root, phase7_root, analysis_root)
    if analyze_rc != 0:
        raise RuntimeError(f"phase7 analyzer failed (see {analysis_root / 'analyze.log'})")

    # 4) zip bundle
    zip_path = bundle_root.with_suffix(".zip")
    print(f"\n=== ZIP -> {zip_path} ===")
    _zip_dir(bundle_root, zip_path)

    print("\nDONE")
    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    print(f"SUMMARY_CSV={analysis_root / 'phase7_summary.csv'}")
    print(f"SUMMARY_MD={analysis_root / 'phase7_summary.md'}")


if __name__ == "__main__":
    main()
