#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import string
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PRESET = "threaded_v1_3_fu_decoy_calib_jitter_n10"
PIVOT_TYPES = ["retention_flip", "entity_switch", "constraint_add"]
PIVOT_RATE = "0.5"
BUDGET = "4000"
GOC_K_VALUES = [4, 8, 16, 32]
GOC_H_VALUES = [1, 2, 3]


@dataclass
class RunEntry:
    pivot_type: str
    run_type: str  # baseline | goc
    goc_K: Optional[int]
    goc_H: Optional[int]
    out_dir: str
    compare_log: str
    report_json: Optional[str]
    run_dir: Optional[str]
    sweep_csv: Optional[str]
    event_trace_sample: Optional[str]
    success: bool
    note: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rand8() -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(8))


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


def _zip_dir(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for fn in files:
                full = Path(base) / fn
                arc = str(full.relative_to(root.parent))
                zf.write(str(full), arcname=arc)


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


def _find_latest_phase6_data_root(repo_root: Path) -> Path:
    candidates = sorted(
        (repo_root / "experiment_bundles").glob("goc_policyops_phase6_main_*/phase6/data"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise RuntimeError("No phase6 data root found under experiment_bundles/")


def _resolve_phase6_source_data(
    phase6_data_root: Path,
    pivot_type: str,
) -> Path:
    config_dir = phase6_data_root / f"{pivot_type}_r0p5"
    source_data = config_dir / "data"
    if source_data.is_dir():
        return source_data
    raise RuntimeError(f"Missing phase6 source data for {pivot_type}: {source_data}")


def _discover_compare_artifacts(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    report_candidates = list((out_dir / "runs" / "compare").glob("*.json"))
    report_json = max(report_candidates, key=lambda p: p.stat().st_mtime) if report_candidates else None
    run_dir = None
    if report_json is not None:
        candidate = out_dir / "runs" / "compare" / report_json.stem
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        run_dirs = [p for p in (out_dir / "runs" / "compare").glob("*") if p.is_dir()]
        if run_dirs:
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    sweep_csv_candidates = list((out_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    sweep_csv = max(sweep_csv_candidates, key=lambda p: p.stat().st_mtime) if sweep_csv_candidates else None
    event_trace_sample = None
    if run_dir is not None:
        trace_candidates = list(run_dir.glob("*/event_traces/**/*.jsonl"))
        if trace_candidates:
            event_trace_sample = max(trace_candidates, key=lambda p: p.stat().st_mtime)
    return report_json, run_dir, sweep_csv, event_trace_sample


def _run_compare(
    repo_root: Path,
    *,
    out_dir: Path,
    dotenv: str,
    model: str,
    parallel_workers: int,
    n_threads: int,
    methods: List[str],
    compare_log: Path,
    goc_k: Optional[int] = None,
    goc_h: Optional[int] = None,
) -> int:
    cmd: List[str] = [
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
        BUDGET,
        "--parallel_workers",
        str(parallel_workers),
        "--n_threads",
        str(n_threads),
        "--save_prompts",
        "--save_raw",
        "--save_event_trace",
        "--event_trace_sample_rate",
        "0.25",
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
                "0.25",
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
    return _run_command(cmd, cwd=repo_root, log_path=compare_log)


def _build_entry(
    *,
    pivot_type: str,
    run_type: str,
    goc_k: Optional[int],
    goc_h: Optional[int],
    out_dir: Path,
    compare_log: Path,
    rc: int,
) -> RunEntry:
    report_json, run_dir, sweep_csv, trace_sample = _discover_compare_artifacts(out_dir)
    success = bool(
        rc == 0
        and report_json is not None
        and sweep_csv is not None
        and trace_sample is not None
    )
    notes: List[str] = []
    if rc != 0:
        notes.append(f"compare_rc={rc}")
    if report_json is None:
        notes.append("missing_report_json")
    if sweep_csv is None:
        notes.append("missing_sweep_csv")
    if trace_sample is None:
        notes.append("missing_event_trace")
    return RunEntry(
        pivot_type=pivot_type,
        run_type=run_type,
        goc_K=goc_k,
        goc_H=goc_h,
        out_dir=str(out_dir),
        compare_log=str(compare_log),
        report_json=str(report_json) if report_json else None,
        run_dir=str(run_dir) if run_dir else None,
        sweep_csv=str(sweep_csv) if sweep_csv else None,
        event_trace_sample=str(trace_sample) if trace_sample else None,
        success=success,
        note=";".join(notes) if notes else "ok",
    )


def _log_contains_network_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    markers = [
        "temporary failure in name resolution",
        "urlerror",
        "name or service not known",
        "failed to resolve",
    ]
    return any(marker in text for marker in markers)


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
    return _run_command(cmd, cwd=repo_root, log_path=out_dir / "analyze.log")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7 unfold-control experiments as a bundle.")
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--parallel_workers", type=int, default=12)
    parser.add_argument("--n_threads", type=int, default=12)
    parser.add_argument("--phase6_data_root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = _repo_root()
    phase6_data_root = (
        args.phase6_data_root.resolve()
        if args.phase6_data_root
        else _find_latest_phase6_data_root(repo_root)
    )

    run_id = f"goc_policyops_phase7_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand8()}"
    bundle_root = repo_root / "experiment_bundles" / run_id
    phase7_root = bundle_root / "phase7"
    data_root = phase7_root / "data"
    runs_root = phase7_root / "runs"
    analysis_root = phase7_root / "analysis"
    for p in [data_root, runs_root, analysis_root]:
        p.mkdir(parents=True, exist_ok=True)

    entries: List[RunEntry] = []
    network_blocked = False
    for pivot_type in PIVOT_TYPES:
        pivot_data_src = _resolve_phase6_source_data(phase6_data_root, pivot_type)
        pivot_data_dst = data_root / pivot_type / "data"
        _symlink_or_copy_data(pivot_data_src, pivot_data_dst)

        # Baselines at high external budget.
        baseline_out = runs_root / pivot_type / "baseline"
        baseline_out.mkdir(parents=True, exist_ok=True)
        _symlink_or_copy_data(pivot_data_dst, baseline_out / "data")
        baseline_log = baseline_out / "compare_openai.log"
        if network_blocked:
            rc = 999
            baseline_log.write_text(
                "Skipped due to earlier network resolution failure.\n",
                encoding="utf-8",
            )
        else:
            rc = _run_compare(
                repo_root,
                out_dir=baseline_out,
                dotenv=args.dotenv,
                model=args.model,
                parallel_workers=int(args.parallel_workers),
                n_threads=int(args.n_threads),
                methods=["full", "similarity_only"],
                compare_log=baseline_log,
            )
            if rc != 0 and _log_contains_network_failure(baseline_log):
                network_blocked = True
        entries.append(
            _build_entry(
                pivot_type=pivot_type,
                run_type="baseline",
                goc_k=None,
                goc_h=None,
                out_dir=baseline_out,
                compare_log=baseline_log,
                rc=rc,
            )
        )

        # GoC grid runs.
        for k in GOC_K_VALUES:
            for h in GOC_H_VALUES:
                goc_out = runs_root / pivot_type / f"goc_K{k}_H{h}"
                goc_out.mkdir(parents=True, exist_ok=True)
                _symlink_or_copy_data(pivot_data_dst, goc_out / "data")
                goc_log = goc_out / "compare_openai.log"
                if network_blocked:
                    rc = 999
                    goc_log.write_text(
                        "Skipped due to earlier network resolution failure.\n",
                        encoding="utf-8",
                    )
                else:
                    rc = _run_compare(
                        repo_root,
                        out_dir=goc_out,
                        dotenv=args.dotenv,
                        model=args.model,
                        parallel_workers=int(args.parallel_workers),
                        n_threads=int(args.n_threads),
                        methods=["goc"],
                        compare_log=goc_log,
                        goc_k=k,
                        goc_h=h,
                    )
                    if rc != 0 and _log_contains_network_failure(goc_log):
                        network_blocked = True
                entries.append(
                    _build_entry(
                        pivot_type=pivot_type,
                        run_type="goc",
                        goc_k=k,
                        goc_h=h,
                        out_dir=goc_out,
                        compare_log=goc_log,
                        rc=rc,
                    )
                )

    manifest_path = phase7_root / "run_manifest.json"
    manifest_obj = {
        "phase": "phase7_unfold_controls",
        "preset": PRESET,
        "pivot_rate": PIVOT_RATE,
        "budget": BUDGET,
        "parallel_workers": int(args.parallel_workers),
        "n_threads": int(args.n_threads),
        "phase6_data_root": str(phase6_data_root),
        "runs": [asdict(e) for e in entries],
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")

    analyze_rc = _run_analyzer(repo_root, phase7_root, analysis_root)
    if analyze_rc != 0:
        raise RuntimeError(f"phase7 analyzer failed (see {analysis_root / 'analyze.log'})")
    summary_csv = analysis_root / "phase7_summary.csv"
    summary_md = analysis_root / "phase7_summary.md"
    pareto_json = analysis_root / "phase7_pareto.json"
    if not summary_csv.exists() or not summary_md.exists() or not pareto_json.exists():
        raise RuntimeError("phase7 analysis artifacts missing")

    pareto_obj: Dict[str, Any] = json.loads(pareto_json.read_text(encoding="utf-8"))
    index_path = bundle_root / "INDEX.md"
    lines: List[str] = []
    lines.append(f"# {run_id}")
    lines.append("")
    lines.append("## Settings")
    lines.append(f"- preset: {PRESET}")
    lines.append(f"- pivot_rate: {PIVOT_RATE}")
    lines.append(f"- pivot_types: {', '.join(PIVOT_TYPES)}")
    lines.append(f"- budget: {BUDGET}")
    lines.append(f"- model: {args.model}")
    lines.append(f"- parallel_workers: {args.parallel_workers}")
    lines.append(f"- n_threads: {args.n_threads}")
    lines.append(f"- phase6_data_root: {phase6_data_root}")
    lines.append(f"- network_blocked: {network_blocked}")
    lines.append("")
    lines.append("## Run Entries")
    for e in entries:
        lines.append(
            "- pivot={p} type={t} K={k} H={h} success={s} report={r} sweep={sw} trace={tr} note={n}".format(
                p=e.pivot_type,
                t=e.run_type,
                k=e.goc_K,
                h=e.goc_H,
                s=e.success,
                r=e.report_json,
                sw=e.sweep_csv,
                tr=e.event_trace_sample,
                n=e.note,
            )
        )
    lines.append("")
    lines.append("## Analysis")
    lines.append(f"- summary_csv: {summary_csv}")
    lines.append(f"- summary_md: {summary_md}")
    lines.append(f"- pareto_json: {pareto_json}")
    lines.append("")
    lines.append("## Best Pareto Configs")
    for pivot_type in PIVOT_TYPES:
        pobj = (pareto_obj.get(pivot_type) or {}) if isinstance(pareto_obj, dict) else {}
        best = pobj.get("best_pareto")
        if isinstance(best, dict):
            lines.append(
                "- {pivot}: K={k}, H={h}, final_pivot_acc={acc}, tokens={tok}".format(
                    pivot=pivot_type,
                    k=best.get("goc_K"),
                    h=best.get("goc_H"),
                    acc=best.get("final_pivot_correct_updated_rate"),
                    tok=best.get("e3_context_token_est_mean"),
                )
            )
        else:
            lines.append(f"- {pivot_type}: no pareto config available")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = bundle_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    _zip_dir(bundle_root, zip_path)

    print(f"BUNDLE_ROOT={bundle_root}")
    print(f"ZIP_PATH={zip_path}")
    for pivot_type in PIVOT_TYPES:
        pobj = (pareto_obj.get(pivot_type) or {}) if isinstance(pareto_obj, dict) else {}
        best = pobj.get("best_pareto")
        if isinstance(best, dict):
            print(
                "BEST_PARETO[{pivot}] K={k} H={h} final_pivot_acc={acc} tokens={tok}".format(
                    pivot=pivot_type,
                    k=best.get("goc_K"),
                    h=best.get("goc_H"),
                    acc=best.get("final_pivot_correct_updated_rate"),
                    tok=best.get("e3_context_token_est_mean"),
                )
            )
        else:
            print(f"BEST_PARETO[{pivot_type}] none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
