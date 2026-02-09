#!/usr/bin/env python3
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


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _load_report(path: Path) -> Dict[str, Any]:
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

    # GoC per-task used knobs (if present)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase8_root = Path(args.phase8_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = phase8_root / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []
    for run in runs:
        report_path = Path(run["report_json"])
        if not report_path.exists():
            # try relative to phase8 root
            alt = phase8_root / report_path
            if alt.exists():
                report_path = alt
            else:
                print(f"[WARN] missing report: {run['report_json']}")
                continue

        report = _load_report(report_path)
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
            row.update(metrics)
            rows.append(row)

    # write CSV
    csv_path = out_dir / "phase8_summary.csv"
    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
    else:
        csv_path.write_text("", encoding="utf-8")

    # write MD (compact)
    md_path = out_dir / "phase8_summary.md"
    lines: List[str] = []
    lines.append("# Phase 8 Summary")
    lines.append("")
    lines.append(f"- Runs: {len(rows)} method-rows")
    lines.append("")

    # pivot-wise tables
    pivots = sorted({r.get("pivot_type") for r in rows if r.get("pivot_type")})
    for pt in pivots:
        lines.append(f"## pivot_type={pt}")
        lines.append("")
        pt_rows = [r for r in rows if r.get("pivot_type") == pt]
        # sort: baseline first, then goc_fixed by pool, then goc_adaptive by pool
        def _key(r: Dict[str, Any]) -> Tuple[int, int, str]:
            var = r.get("variant") or ""
            order = 0
            if var == "baseline":
                order = 0
            elif var == "goc_fixed":
                order = 1
            else:
                order = 2
            pool = int(r.get("pool_size") or 0)
            meth = str(r.get("method") or "")
            return (order, pool, meth)

        pt_rows = sorted(pt_rows, key=_key)
        # header
        lines.append("| variant | pool | method | final_acc | pivot_acc | tokens(all) | tokens(pivot) | goc_K(pivot) | goc_H(pivot) |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in pt_rows:
            lines.append(
                "| {variant} | {pool} | {method} | {final_acc:.3f} | {pivot_acc:.3f} | {tok_all:.1f} | {tok_piv:.1f} | {k_piv:.1f} | {h_piv:.1f} |".format(
                    variant=str(r.get("variant")),
                    pool=int(r.get("pool_size") or 0),
                    method=str(r.get("method")),
                    final_acc=float(r.get("final_accuracy") or 0.0),
                    pivot_acc=float(r.get("final_pivot_accuracy") or 0.0),
                    tok_all=float(r.get("final_token_mean") or 0.0),
                    tok_piv=float(r.get("final_pivot_token_mean") or 0.0),
                    k_piv=float(r.get("goc_K_mean_pivot") or 0.0),
                    h_piv=float(r.get("goc_H_mean_pivot") or 0.0),
                )
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
