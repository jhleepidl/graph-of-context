#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from policyops_bundle_layout import build_bundle_quick_access


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _link_data_dir(src_root: Path, dst_out_dir: Path) -> None:
    src_data = src_root / "data"
    dst_data = dst_out_dir / "data"
    if dst_data.exists() or dst_data.is_symlink():
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
    traceops_level: int
    traceops_scenario: str
    traceops_threads: int
    traceops_trace_len: int
    traceops_delay_to_relevance: int
    traceops_distractor_branching: int
    traceops_contradiction_rate: float
    traceops_exception_density: float
    traceops_state_flip_count: int
    variant: str
    method: str
    pivot_gold_mode: str
    goc_enable_avoids: bool
    goc_avoids_mode: str
    goc_applicability_seed_enable: bool
    goc_applicability_seed_topk: int
    goc_dependency_closure_enable: bool
    goc_dependency_closure_topk: int
    goc_dependency_closure_hops: int
    goc_dependency_closure_universe: str
    report_json: str
    compare_root: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--traceops_level", type=int, default=1)
    ap.add_argument("--traceops_scenarios", type=str, default="mixed")
    ap.add_argument("--traceops_seed", type=int, default=0)
    ap.add_argument("--traceops_threads", type=int, default=40)
    ap.add_argument("--traceops_trace_len", type=int, default=0)
    ap.add_argument("--traceops_delay_to_relevance", type=int, default=0)
    ap.add_argument("--traceops_distractor_branching", type=int, default=2)
    ap.add_argument("--traceops_contradiction_rate", type=float, default=0.35)
    ap.add_argument("--traceops_exception_density", type=float, default=0.35)
    ap.add_argument("--traceops_state_flip_count", type=int, default=1)
    ap.add_argument("--max_threads", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--pivot_gold_mode", choices=["original", "respect_ticket_updated", "both"], default="respect_ticket_updated")
    ap.add_argument("--goc_enable_avoids", action="store_true", default=True)
    ap.add_argument("--no_goc_enable_avoids", action="store_false", dest="goc_enable_avoids")
    ap.add_argument("--goc_avoids_mode", choices=["legacy_commit", "applicability", "off"], default="applicability")
    ap.add_argument("--traceops_eval_mode", choices=["deterministic", "llm"], default="deterministic")
    ap.add_argument("--traceops_llm_max_pivots", type=int, default=0)
    ap.add_argument("--include_ablations", action="store_true")
    ap.add_argument("--include_oracle", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bundle_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"goc_traceops_phase15_{ts}_{bundle_id}"
    bundle_root = repo_root / "experiment_bundles" / bundle_name
    phase15_root = bundle_root / "phase15"
    data_root = phase15_root / "data"
    runs_root = phase15_root / "runs"
    analysis_root = phase15_root / "analysis"
    _ensure_dir(data_root)
    _ensure_dir(runs_root)
    _ensure_dir(analysis_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["PYTHONUNBUFFERED"] = "1"

    scenarios = [s.strip() for s in str(args.traceops_scenarios).split(",") if s.strip()]
    if not scenarios:
        scenarios = ["mixed"]

    traceops_threads = int(args.traceops_threads)
    if args.max_threads and int(args.max_threads) > 0:
        traceops_threads = min(traceops_threads, int(args.max_threads))
    if args.smoke:
        traceops_threads = min(traceops_threads, 4)

    manifest: Dict[str, Any] = {
        "bundle_name": bundle_name,
        "created_at": ts,
        "benchmark": "traceops_v0",
        "model": args.model,
        "traceops_level": int(args.traceops_level),
        "traceops_seed": int(args.traceops_seed),
        "traceops_threads": int(traceops_threads),
        "traceops_scenarios": list(scenarios),
        "traceops_trace_len": int(args.traceops_trace_len),
        "traceops_delay_to_relevance": int(args.traceops_delay_to_relevance),
        "traceops_distractor_branching": int(args.traceops_distractor_branching),
        "traceops_contradiction_rate": float(args.traceops_contradiction_rate),
        "traceops_exception_density": float(args.traceops_exception_density),
        "traceops_state_flip_count": int(args.traceops_state_flip_count),
        "pivot_gold_mode": str(args.pivot_gold_mode),
        "goc_enable_avoids": bool(args.goc_enable_avoids),
        "goc_avoids_mode": str(args.goc_avoids_mode),
        "traceops_eval_mode": str(args.traceops_eval_mode),
        "traceops_llm_max_pivots": int(args.traceops_llm_max_pivots),
        "max_steps": int(args.max_steps),
        "runs": [],
    }

    for scenario in scenarios:
        data_out = data_root / scenario
        _ensure_dir(data_out)
        gen_cmd = [
            sys.executable,
            "-u",
            "-m",
            "policyops.run",
            "generate",
            "--benchmark",
            "traceops_v0",
            "--traceops_level",
            str(args.traceops_level),
            "--traceops_scenarios",
            scenario,
            "--traceops_seed",
            str(args.traceops_seed),
            "--traceops_threads",
            str(traceops_threads),
            "--traceops_distractor_branching",
            str(args.traceops_distractor_branching),
            "--traceops_contradiction_rate",
            str(args.traceops_contradiction_rate),
            "--traceops_exception_density",
            str(args.traceops_exception_density),
            "--traceops_state_flip_count",
            str(args.traceops_state_flip_count),
            "--out_dir",
            str(data_out),
        ]
        if int(args.traceops_trace_len) > 0:
            gen_cmd += ["--traceops_trace_len", str(args.traceops_trace_len)]
        if int(args.traceops_delay_to_relevance) > 0:
            gen_cmd += ["--traceops_delay_to_relevance", str(args.traceops_delay_to_relevance)]
        _run(gen_cmd, cwd=repo_root, env=env)

        base_compare_flags = [
            "--benchmark",
            "traceops_v0",
            "--pivot_gold_mode",
            args.pivot_gold_mode,
            "--goc_avoids_mode",
            str(args.goc_avoids_mode),
            "--traceops_eval_mode",
            str(args.traceops_eval_mode),
            "--parallel_workers",
            "1",
        ]
        if str(args.traceops_eval_mode) == "llm":
            base_compare_flags += [
                "--llm",
                "openai",
                "--model",
                args.model,
                "--traceops_llm_max_pivots",
                str(int(args.traceops_llm_max_pivots)),
                "--traceops_llm_temperature",
                "0.0",
                "--traceops_llm_max_output_tokens",
                "256",
                "--traceops_llm_cache_dir",
                ".cache/traceops_llm",
                "--traceops_llm_seed",
                "0",
            ]
        else:
            base_compare_flags += ["--model", args.model]
        if args.goc_enable_avoids:
            base_compare_flags += ["--goc_enable_avoids"]
        else:
            base_compare_flags += ["--no_goc_enable_avoids"]
        if int(args.max_steps) > 0:
            base_compare_flags += ["--traceops_max_steps", str(int(args.max_steps))]

        out_base = runs_root / scenario / "baseline"
        _ensure_dir(out_base)
        _link_data_dir(data_out, out_base)
        base_cmd = [
            sys.executable,
            "-u",
            "-m",
            "policyops.run",
            "compare",
        ] + base_compare_flags + [
            "--methods",
            "full",
            "similarity_only",
            "agent_fold",
            "--dotenv",
            args.dotenv,
            "--out_dir",
            str(out_base),
        ]
        _run(base_cmd, cwd=repo_root, env=env)
        rep_base = _discover_report_json(out_base)
        if not rep_base:
            raise RuntimeError(f"missing compare report: {out_base}")
        manifest["runs"].append(
            asdict(
                RunEntry(
                    traceops_level=int(args.traceops_level),
                    traceops_scenario=scenario,
                    traceops_threads=int(traceops_threads),
                    traceops_trace_len=int(args.traceops_trace_len),
                    traceops_delay_to_relevance=int(args.traceops_delay_to_relevance),
                    traceops_distractor_branching=int(args.traceops_distractor_branching),
                    traceops_contradiction_rate=float(args.traceops_contradiction_rate),
                    traceops_exception_density=float(args.traceops_exception_density),
                    traceops_state_flip_count=int(args.traceops_state_flip_count),
                    variant="baseline",
                    method="full/similarity_only/agent_fold",
                    pivot_gold_mode=str(args.pivot_gold_mode),
                    goc_enable_avoids=bool(args.goc_enable_avoids),
                    goc_avoids_mode=str(args.goc_avoids_mode),
                    goc_applicability_seed_enable=False,
                    goc_applicability_seed_topk=0,
                    goc_dependency_closure_enable=False,
                    goc_dependency_closure_topk=0,
                    goc_dependency_closure_hops=0,
                    goc_dependency_closure_universe="candidates",
                    report_json=str(rep_base.relative_to(phase15_root)),
                    compare_root=str(out_base.relative_to(phase15_root)),
                )
            )
        )

        def _run_goc_variant(
            variant: str,
            *,
            method_name: str,
            seed_enable: bool,
            seed_topk: int,
            closure_enable: bool,
            closure_topk: int,
            closure_hops: int,
            closure_universe: str,
            unfold_max_nodes: int,
            unfold_hops: int,
        ) -> None:
            out_dir = runs_root / scenario / variant
            _ensure_dir(out_dir)
            _link_data_dir(data_out, out_dir)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "policyops.run",
                "compare",
            ] + base_compare_flags + [
                "--methods",
                method_name,
                "--goc_unfold_max_nodes",
                str(unfold_max_nodes),
                "--goc_unfold_hops",
                str(unfold_hops),
                "--goc_unfold_budget_mode",
                "nodes_and_hops",
                "--goc_applicability_seed_topk",
                str(seed_topk),
                "--goc_dependency_closure_topk",
                str(closure_topk),
                "--goc_dependency_closure_hops",
                str(closure_hops),
                "--goc_dependency_closure_universe",
                str(closure_universe),
                "--dotenv",
                args.dotenv,
                "--out_dir",
                str(out_dir),
            ]
            if seed_enable:
                cmd += ["--goc_applicability_seed_enable"]
            if closure_enable:
                cmd += ["--goc_dependency_closure_enable"]
            _run(cmd, cwd=repo_root, env=env)
            rep = _discover_report_json(out_dir)
            if not rep:
                raise RuntimeError(f"missing compare report: {out_dir}")
            manifest["runs"].append(
                asdict(
                    RunEntry(
                        traceops_level=int(args.traceops_level),
                        traceops_scenario=scenario,
                        traceops_threads=int(traceops_threads),
                        traceops_trace_len=int(args.traceops_trace_len),
                        traceops_delay_to_relevance=int(args.traceops_delay_to_relevance),
                        traceops_distractor_branching=int(args.traceops_distractor_branching),
                        traceops_contradiction_rate=float(args.traceops_contradiction_rate),
                        traceops_exception_density=float(args.traceops_exception_density),
                        traceops_state_flip_count=int(args.traceops_state_flip_count),
                        variant=variant,
                        method=str(method_name),
                        pivot_gold_mode=str(args.pivot_gold_mode),
                        goc_enable_avoids=bool(args.goc_enable_avoids),
                        goc_avoids_mode=str(args.goc_avoids_mode),
                        goc_applicability_seed_enable=bool(seed_enable),
                        goc_applicability_seed_topk=int(seed_topk),
                        goc_dependency_closure_enable=bool(closure_enable),
                        goc_dependency_closure_topk=int(closure_topk),
                        goc_dependency_closure_hops=int(closure_hops),
                        goc_dependency_closure_universe=str(closure_universe),
                        report_json=str(rep.relative_to(phase15_root)),
                        compare_root=str(out_dir.relative_to(phase15_root)),
                    )
                )
            )

        _run_goc_variant(
            "goc_phase13_style",
            method_name="goc",
            seed_enable=False,
            seed_topk=8,
            closure_enable=False,
            closure_topk=12,
            closure_hops=1,
            closure_universe="candidates",
            unfold_max_nodes=999,
            unfold_hops=1,
        )
        _run_goc_variant(
            "goc_phase15_seed_closure",
            method_name="goc",
            seed_enable=True,
            seed_topk=8,
            closure_enable=True,
            closure_topk=12,
            closure_hops=1,
            closure_universe="candidates",
            unfold_max_nodes=999,
            unfold_hops=1,
        )
        if args.include_ablations:
            _run_goc_variant(
                "goc_phase15_seed_closure_world",
                method_name="goc",
                seed_enable=True,
                seed_topk=8,
                closure_enable=True,
                closure_topk=12,
                closure_hops=1,
                closure_universe="world",
                unfold_max_nodes=999,
                unfold_hops=1,
            )
        if args.include_oracle:
            _run_goc_variant(
                "goc_oracle_phase15_seed_closure",
                method_name="goc_oracle",
                seed_enable=True,
                seed_topk=8,
                closure_enable=True,
                closure_topk=12,
                closure_hops=1,
                closure_universe="candidates",
                unfold_max_nodes=999,
                unfold_hops=1,
            )

    (phase15_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    analyze_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_phase15_traceops.py"),
        "--phase15_root",
        str(phase15_root),
        "--out_dir",
        str(analysis_root),
    ]
    _run(analyze_cmd, cwd=repo_root, env=env)

    idx_lines = [
        f"# Phase15 TraceOps bundle: {bundle_name}",
        "",
        f"- model={args.model}",
        f"- traceops_level={int(args.traceops_level)} scenarios={','.join(scenarios)} threads={int(traceops_threads)}",
        f"- delay_to_relevance={int(args.traceops_delay_to_relevance)} distractor_branching={int(args.traceops_distractor_branching)}",
        f"- contradiction_rate={float(args.traceops_contradiction_rate)} exception_density={float(args.traceops_exception_density)}",
        f"- state_flip_count={int(args.traceops_state_flip_count)} max_steps={int(args.max_steps)}",
        f"- traceops_eval_mode={str(args.traceops_eval_mode)} traceops_llm_max_pivots={int(args.traceops_llm_max_pivots)}",
        (
            "- traceops_v0 deterministic mode (no LLM calls); token fields are estimates."
            if str(args.traceops_eval_mode) == "deterministic"
            else "- traceops_v0 llm mode enabled for pivot_check; usage tokens prefer actual API usage."
        ),
        "",
        "## Outputs",
        f"- phase15_root: {phase15_root}",
        f"- manifest: {phase15_root / 'run_manifest.json'}",
        f"- analysis: {analysis_root}",
    ]
    (bundle_root / "INDEX.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    quick_dirs = build_bundle_quick_access(bundle_root)
    for qd in quick_dirs:
        print(f"Quick access: {qd}")

    zip_path = shutil.make_archive(str(bundle_root), "zip", root_dir=str(bundle_root.parent), base_dir=bundle_root.name)
    print(f"\nBundle folder: {bundle_root}")
    print(f"Zip path: {zip_path}")


if __name__ == "__main__":
    main()
