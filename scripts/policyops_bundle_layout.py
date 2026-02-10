#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _safe_symlink(link_path: Path, target_path: Path) -> None:
    _remove_path(link_path)
    link_path.symlink_to(target_path.resolve(), target_is_directory=target_path.is_dir())


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    cands = list(root.glob(pattern))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _discover_compare_artifacts(run_out_dir: Path) -> Dict[str, Optional[Path]]:
    compare_root = run_out_dir / "runs" / "compare"
    report_json = _latest_file(compare_root, "*.json")

    compare_run_dir: Optional[Path] = None
    if report_json is not None:
        candidate = compare_root / report_json.stem
        if candidate.exists() and candidate.is_dir():
            compare_run_dir = candidate
    if compare_run_dir is None and compare_root.exists():
        run_dirs = [p for p in compare_root.iterdir() if p.is_dir()]
        if run_dirs:
            compare_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    sweep_csv = _latest_file(run_out_dir / "runs" / "context_budget_sweep", "*/results_context_budget_sweep.csv")
    if sweep_csv is None:
        sweep_csv = _latest_file(run_out_dir, "results_context_budget_sweep.csv")

    return {
        "report_json": report_json,
        "compare_run_dir": compare_run_dir,
        "sweep_csv": sweep_csv,
    }


def build_run_quick_access(run_out_dir: Path) -> Optional[Path]:
    artifacts = _discover_compare_artifacts(run_out_dir)
    compare_run_dir = artifacts["compare_run_dir"]
    if compare_run_dir is None:
        return None

    quick_root = run_out_dir / "quick_access"
    _remove_path(quick_root)
    _ensure_dir(quick_root)

    report_json = artifacts["report_json"]
    if report_json is not None and report_json.exists():
        _safe_symlink(quick_root / "report.json", report_json)

    sweep_csv = artifacts["sweep_csv"]
    if sweep_csv is not None and sweep_csv.exists():
        _safe_symlink(quick_root / "results_context_budget_sweep.csv", sweep_csv)

    _safe_symlink(quick_root / "compare_run", compare_run_dir)

    methods = [p for p in compare_run_dir.iterdir() if p.is_dir()]
    methods.sort(key=lambda p: p.name)

    for asset in ["event_traces", "prompts", "raw_outputs", "graphs", "graphs_internal"]:
        _ensure_dir(quick_root / asset)

    index_lines: List[str] = []
    index_lines.append(f"# Quick Access: {run_out_dir.name}")
    index_lines.append("")
    index_lines.append(f"- run_out_dir: {run_out_dir}")
    index_lines.append(f"- compare_run: {compare_run_dir}")
    if report_json is not None:
        index_lines.append(f"- report_json: {report_json}")
    if sweep_csv is not None:
        index_lines.append(f"- sweep_csv: {sweep_csv}")
    index_lines.append("")
    index_lines.append("## Method Artifacts")

    for method_dir in methods:
        method = method_dir.name
        index_lines.append(f"- {method}:")

        traces_root = method_dir / "event_traces"
        trace_target = traces_root / method if (traces_root / method).exists() else traces_root
        if trace_target.exists():
            _safe_symlink(quick_root / "event_traces" / method, trace_target)
            index_lines.append(f"  event_traces={trace_target}")

        for asset in ["prompts", "raw_outputs", "graphs", "graphs_internal"]:
            target = method_dir / asset
            if target.exists():
                _safe_symlink(quick_root / asset / method, target)
                index_lines.append(f"  {asset}={target}")

    (quick_root / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return quick_root


def build_phase_quick_access(phase_root: Path) -> Optional[Path]:
    runs_root = phase_root / "runs"
    if not runs_root.exists():
        return None

    run_out_dirs: List[Path] = []
    for compare_dir in runs_root.rglob("runs/compare"):
        run_out_dirs.append(compare_dir.parent.parent)

    if not run_out_dirs:
        return None

    unique_dirs = sorted({p.resolve() for p in run_out_dirs})

    phase_quick = phase_root / "quick_access"
    _remove_path(phase_quick)
    _ensure_dir(phase_quick)
    _ensure_dir(phase_quick / "runs")
    _ensure_dir(phase_quick / "event_traces")
    _ensure_dir(phase_quick / "reports")

    index_lines: List[str] = []
    index_lines.append(f"# Phase Quick Access: {phase_root.name}")
    index_lines.append("")

    for run_out_dir in unique_dirs:
        try:
            rel = run_out_dir.relative_to(runs_root.resolve())
            run_key = "__".join(rel.parts)
        except ValueError:
            run_key = run_out_dir.name

        _safe_symlink(phase_quick / "runs" / run_key, run_out_dir)
        run_quick = build_run_quick_access(run_out_dir)
        index_lines.append(f"- {run_key}: {run_out_dir}")

        if run_quick is None:
            continue

        report_link = run_quick / "report.json"
        if report_link.exists():
            _safe_symlink(phase_quick / "reports" / f"{run_key}.json", report_link.resolve())

        et_root = run_quick / "event_traces"
        if et_root.exists():
            for method_link in sorted(et_root.iterdir(), key=lambda p: p.name):
                if method_link.exists() or method_link.is_symlink():
                    event_key = f"{run_key}__{method_link.name}"
                    _safe_symlink(phase_quick / "event_traces" / event_key, method_link.resolve())

    (phase_quick / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return phase_quick


def build_bundle_quick_access(bundle_root: Path) -> List[Path]:
    out: List[Path] = []
    for child in sorted(bundle_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not child.name.startswith("phase"):
            continue
        quick = build_phase_quick_access(child)
        if quick is not None:
            out.append(quick)

    index_path = bundle_root / "INDEX.md"
    if out and index_path.exists():
        marker = "## Quick Access"
        content = index_path.read_text(encoding="utf-8")
        if marker not in content:
            lines = [content.rstrip(), "", marker]
            for quick_dir in out:
                lines.append(f"- {quick_dir}")
            lines.append("")
            index_path.write_text("\n".join(lines), encoding="utf-8")

    return out
