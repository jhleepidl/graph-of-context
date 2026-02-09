#!/usr/bin/env python3
"""Analyze Phase 9 GoC policy frontier bundles.

This script reads <phase9_root>/run_manifest.json (emitted by
scripts/run_phase9_frontier_bundle.py) and computes **final-episode** metrics
(episode_id == 3) from each compare report.

Outputs (in --out_dir):
- phase9_summary.csv: one row per (manifest run) × (method)
- phase9_summary.md: pivot-wise tables + overall mean summary
- figures/: simple scatter plots (tokens vs accuracy) for paper/debug

Notes:
- Pivot tasks are identified via record field `is_pivot_task`.
- GoC per-task used knobs are taken from record fields `goc_unfold_max_nodes` / `goc_unfold_hops`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None]
    if not xs2:
        return None
    return sum(xs2) / float(len(xs2))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics(method_report: Dict[str, Any]) -> Dict[str, Any]:
    records = list(method_report.get("records") or [])
    final = [r for r in records if _to_int(r.get("episode_id")) == 3]
    if not final:
        return {}

    def _is_bool(v: Any) -> bool:
        return isinstance(v, bool)

    final_judge = [r for r in final if _is_bool(r.get("judge_correct"))]
    final_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in final_judge])

    pivot_final = [r for r in final if bool(r.get("is_pivot_task"))]
    nonpivot_final = [r for r in final if not bool(r.get("is_pivot_task"))]

    pivot_judge = [r for r in pivot_final if _is_bool(r.get("judge_correct"))]
    nonpivot_judge = [r for r in nonpivot_final if _is_bool(r.get("judge_correct"))]

    pivot_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in pivot_judge])
    nonpivot_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in nonpivot_judge])

    tok_mean = _mean([_to_float(r.get("e3_context_token_est")) for r in final])
    pivot_tok = _mean([_to_float(r.get("e3_context_token_est")) for r in pivot_final])
    nonpivot_tok = _mean([_to_float(r.get("e3_context_token_est")) for r in nonpivot_final])

    stale = [r for r in pivot_final if isinstance(r.get("stale_evidence"), bool)]
    stale_rate = _mean([1.0 if r.get("stale_evidence") else 0.0 for r in stale])

    goc_k_all = _mean([_to_float(r.get("goc_unfold_max_nodes")) for r in final])
    goc_h_all = _mean([_to_float(r.get("goc_unfold_hops")) for r in final])
    goc_k_pivot = _mean([_to_float(r.get("goc_unfold_max_nodes")) for r in pivot_final])
    goc_h_pivot = _mean([_to_float(r.get("goc_unfold_hops")) for r in pivot_final])
    goc_k_nonpivot = _mean([_to_float(r.get("goc_unfold_max_nodes")) for r in nonpivot_final])
    goc_h_nonpivot = _mean([_to_float(r.get("goc_unfold_hops")) for r in nonpivot_final])

    return {
        "final_tasks": len(final),
        "final_accuracy": final_acc,
        "final_token_mean": tok_mean,
        "pivot_final_tasks": len(pivot_final),
        "pivot_rate_final": len(pivot_final) / float(len(final)) if final else None,
        "final_pivot_accuracy": pivot_acc,
        "final_pivot_token_mean": pivot_tok,
        "final_nonpivot_accuracy": nonpivot_acc,
        "final_nonpivot_token_mean": nonpivot_tok,
        "final_pivot_stale_rate": stale_rate,
        "goc_K_mean_all": goc_k_all,
        "goc_H_mean_all": goc_h_all,
        "goc_K_mean_pivot": goc_k_pivot,
        "goc_H_mean_pivot": goc_h_pivot,
        "goc_K_mean_nonpivot": goc_k_nonpivot,
        "goc_H_mean_nonpivot": goc_h_nonpivot,
    }


def _label(run: Dict[str, Any], method: str) -> str:
    variant = str(run.get("variant") or "")
    if variant == "baseline":
        return method
    pool = int(run.get("pool_size") or 0)
    return f"{variant}_pool{pool}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _md_table(rows: List[Dict[str, Any]]) -> List[str]:
    cols = [
        "label",
        "policy",
        "pool_size",
        "method",
        "final_accuracy",
        "final_pivot_accuracy",
        "final_token_mean",
        "final_pivot_token_mean",
        "final_nonpivot_token_mean",
        "goc_K_mean_pivot",
        "goc_H_mean_pivot",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]

    def _fmt(r: Dict[str, Any], k: str) -> str:
        v = r.get(k)
        if v is None:
            return ""
        if k in ("final_accuracy", "final_pivot_accuracy"):
            return f"{float(v):.3f}"
        if k.endswith("token_mean"):
            return f"{float(v):.1f}"
        if k.startswith("goc_"):
            return f"{float(v):.1f}"
        return str(v)

    for r in rows:
        lines.append("| " + " | ".join(_fmt(r, c) for c in cols) + " |")
    return lines


def _plot_frontier(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Create simple scatter plots: tokens(all) vs pivot_acc."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skipping plots")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pivots = sorted({r.get("pivot_type") for r in rows if r.get("pivot_type")})

    def _scatter(points: List[Dict[str, Any]], title: str, path: Path, y_key: str) -> None:
        xs = [float(r.get("final_token_mean") or 0.0) for r in points]
        ys = [float(r.get(y_key) or 0.0) for r in points]
        plt.figure()
        plt.scatter(xs, ys)
        for r, x, y in zip(points, xs, ys):
            plt.annotate(str(r.get("label")), (x, y), fontsize=7)
        plt.xlabel("Tokens (episode 3 mean)")
        plt.ylabel("Pivot accuracy (episode 3)" if y_key == "final_pivot_accuracy" else "Final accuracy (episode 3)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    for pt in pivots:
        pts = [r for r in rows if r.get("pivot_type") == pt]
        _scatter(pts, f"Phase9: {pt} pivot_acc vs tokens", fig_dir / f"frontier_{pt}_pivotacc.png", "final_pivot_accuracy")
        _scatter(pts, f"Phase9: {pt} final_acc vs tokens", fig_dir / f"frontier_{pt}_finalacc.png", "final_accuracy")

    # overall mean by label
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(str(r.get("label")), []).append(r)

    mean_rows: List[Dict[str, Any]] = []
    for lab, rs in by_label.items():
        mean_rows.append({
            "label": lab,
            "final_token_mean": _mean([_to_float(x.get("final_token_mean")) for x in rs]),
            "final_pivot_accuracy": _mean([_to_float(x.get("final_pivot_accuracy")) for x in rs]),
            "final_accuracy": _mean([_to_float(x.get("final_accuracy")) for x in rs]),
        })
    mean_rows = sorted(mean_rows, key=lambda r: (-(r.get("final_pivot_accuracy") or 0.0), r.get("final_token_mean") or 1e9))
    _scatter(mean_rows, "Phase9: mean pivot_acc vs tokens", fig_dir / "frontier_mean_pivotacc.png", "final_pivot_accuracy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase9_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase9_root = Path(args.phase9_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = phase9_root / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []
    for run in runs:
        report_path = phase9_root / str(run.get("report_json"))
        if not report_path.exists():
            report_path = Path(str(run.get("report_json")))
        if not report_path.exists():
            print(f"[WARN] missing report: {run.get('report_json')}")
            continue

        report = _load_json(report_path)
        method_reports = report.get("method_reports") or {}
        if not isinstance(method_reports, dict):
            continue

        for method, mr in method_reports.items():
            if not isinstance(mr, dict):
                continue
            metrics = _compute_metrics(mr)
            if not metrics:
                continue
            row = dict(run)
            row["method"] = method
            row["label"] = _label(run, method)
            row.update(metrics)
            rows.append(row)

    csv_path = out_dir / "phase9_summary.csv"
    md_path = out_dir / "phase9_summary.md"

    if rows:
        _write_csv(csv_path, rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Phase 9 Summary")
    lines.append("")
    lines.append(f"- Method-rows: {len(rows)}")
    lines.append("")

    pivots = sorted({r.get("pivot_type") for r in rows if r.get("pivot_type")})
    for pt in pivots:
        lines.append(f"## pivot_type={pt}")
        lines.append("")
        pt_rows = [r for r in rows if r.get("pivot_type") == pt]

        def _key(r: Dict[str, Any]) -> Tuple[int, int, str]:
            lab = str(r.get("label") or "")
            pool = int(r.get("pool_size") or 0)
            order = 99
            if lab in ("full", "similarity_only"):
                order = 0
            elif "goc_fixed_light" in lab:
                order = 1
            elif "goc_fixed_heavy" in lab:
                order = 2
            elif "goc_adaptive_h_only" in lab:
                order = 3
            elif "goc_adaptive_k_small" in lab:
                order = 4
            elif "goc_adaptive_heavy" in lab:
                order = 5
            return (order, pool, lab)

        pt_rows = sorted(pt_rows, key=_key)
        lines.extend(_md_table(pt_rows))
        lines.append("")

    # overall mean
    lines.append("## Overall mean across pivot types")
    lines.append("")

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(str(r.get("label")), []).append(r)

    mean_rows: List[Dict[str, Any]] = []
    for lab, rs in by_label.items():
        mean_rows.append({
            "label": lab,
            "policy": rs[0].get("policy"),
            "pool_size": rs[0].get("pool_size"),
            "method": rs[0].get("method"),
            "final_accuracy": _mean([_to_float(x.get("final_accuracy")) for x in rs]),
            "final_pivot_accuracy": _mean([_to_float(x.get("final_pivot_accuracy")) for x in rs]),
            "final_token_mean": _mean([_to_float(x.get("final_token_mean")) for x in rs]),
            "final_pivot_token_mean": _mean([_to_float(x.get("final_pivot_token_mean")) for x in rs]),
            "final_nonpivot_token_mean": _mean([_to_float(x.get("final_nonpivot_token_mean")) for x in rs]),
            "goc_K_mean_pivot": _mean([_to_float(x.get("goc_K_mean_pivot")) for x in rs]),
            "goc_H_mean_pivot": _mean([_to_float(x.get("goc_H_mean_pivot")) for x in rs]),
        })

    mean_rows = sorted(mean_rows, key=lambda r: (-(r.get("final_pivot_accuracy") or 0.0), r.get("final_token_mean") or 1e9))
    lines.extend(_md_table(mean_rows))
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    _plot_frontier(rows, out_dir)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
