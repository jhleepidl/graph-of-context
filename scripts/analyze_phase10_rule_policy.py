#!/usr/bin/env python3
"""Analyze Phase 10 rule-policy runs.

Reads phase10/run_manifest.json and each report JSON, computes final-episode metrics (episode_id==3),
and writes:
- phase10_summary.csv
- phase10_summary.md
- figures/*.png (pivot_acc vs tokens, final_acc vs tokens per pivot_type)

This intentionally uses judge_correct from the report (already judged against updated ticket for final episode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    # Matplotlib is optional. When unavailable, we still write CSV/MD summaries.
    plt = None  # type: ignore
    _HAS_MPL = False
import pandas as pd


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    vs = [x for x in xs if isinstance(x, (int, float))]
    if not vs:
        return None
    return sum(vs) / float(len(vs))


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


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

    pivot_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in pivot_judge]) if pivot_judge else None
    nonpivot_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in nonpivot_judge]) if nonpivot_judge else None

    # tokens (episode 3 context estimate)
    tok_all = _mean([_to_float(r.get("e3_context_token_est")) for r in final])
    tok_piv = _mean([_to_float(r.get("e3_context_token_est")) for r in pivot_final]) if pivot_final else None
    tok_non = _mean([_to_float(r.get("e3_context_token_est")) for r in nonpivot_final]) if nonpivot_final else None

    # stale evidence rate (pivot tasks)
    piv_stale = None
    if pivot_final:
        stale_flags = [r.get("stale_evidence") for r in pivot_final if _is_bool(r.get("stale_evidence"))]
        if stale_flags:
            piv_stale = _mean([1.0 if x else 0.0 for x in stale_flags])

    avoids_eval = []
    for r in final:
        should_avoid = bool(r.get("goc_avoids_edge_injected"))
        avoid_ids = r.get("goc_avoid_target_clause_ids") or []
        if (not should_avoid) and isinstance(avoid_ids, list) and avoid_ids:
            should_avoid = True
        if not should_avoid:
            continue
        injected = r.get("goc_avoided_node_injected")
        if _is_bool(injected):
            avoids_eval.append(1.0 if injected else 0.0)
            continue
        if isinstance(avoid_ids, list):
            ctx = set(map(str, r.get("e3_context_clause_ids") or []))
            avoids_eval.append(1.0 if (set(map(str, avoid_ids)) & ctx) else 0.0)
    avoided_node_injected_rate = _mean(avoids_eval) if avoids_eval else None

    # unfold usage (GoC only)
    def _mean_int(field: str, subset: List[Dict[str, Any]]) -> Optional[float]:
        vs = []
        for r in subset:
            v = _to_int(r.get(field))
            if v is not None:
                vs.append(float(v))
        return _mean(vs)

    goc_K_mean_pivot = _mean_int("goc_unfold_max_nodes", pivot_final) if pivot_final else None
    goc_H_mean_pivot = _mean_int("goc_unfold_hops", pivot_final) if pivot_final else None
    goc_K_mean_nonpivot = _mean_int("goc_unfold_max_nodes", nonpivot_final) if nonpivot_final else None
    goc_H_mean_nonpivot = _mean_int("goc_unfold_hops", nonpivot_final) if nonpivot_final else None

    return {
        "final_tasks": len(final),
        "pivot_final_tasks": len(pivot_final),
        "pivot_rate_final": (len(pivot_final) / float(len(final))) if final else None,
        "final_accuracy": final_acc,
        "final_pivot_accuracy": pivot_acc,
        "final_nonpivot_accuracy": nonpivot_acc,
        "final_token_mean": tok_all,
        "final_pivot_token_mean": tok_piv,
        "final_nonpivot_token_mean": tok_non,
        "final_pivot_stale_rate": piv_stale,
        "avoided_node_injected_rate": avoided_node_injected_rate,
        "goc_K_mean_pivot": goc_K_mean_pivot,
        "goc_H_mean_pivot": goc_H_mean_pivot,
        "goc_K_mean_nonpivot": goc_K_mean_nonpivot,
        "goc_H_mean_nonpivot": goc_H_mean_nonpivot,
    }


def _label(row: Dict[str, Any]) -> str:
    v = row.get("variant")
    if v == "baseline":
        return str(row.get("method"))
    if v == "goc_fixed_light":
        return "goc_fixed_light(K4,H2)"
    if v == "goc_fixed_heavy":
        return "goc_fixed_heavy(K16,H3)"
    if v == "goc_adaptive_heavy":
        return "goc_adaptive_heavy(D4,2->P16,3)"
    if v == "goc_rule_v1":
        pk = row.get("pivot_k"); ph = row.get("pivot_h")
        return f"goc_rule_v1(D4,2->P{pk},{ph})"
    return str(v)


def _scatter(df: pd.DataFrame, title: str, x: str, y: str, out: Path) -> None:
    """Write a simple scatter plot if matplotlib is available."""
    if not _HAS_MPL:
        return
    # mypy: plt may be None when _HAS_MPL is False
    plt.figure()  # type: ignore
    plt.scatter(df[x], df[y])  # type: ignore
    for _, r in df.iterrows():
        plt.annotate(r["label"], (r[x], r[y]), fontsize=7)  # type: ignore
    plt.xlabel(x)  # type: ignore
    plt.ylabel(y)  # type: ignore
    plt.title(title)  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(out, dpi=220)  # type: ignore
    plt.close()  # type: ignore



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase10_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase10_root = Path(args.phase10_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    if _HAS_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(phase10_root / "run_manifest.json")
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []
    for run in runs:
        report_path = phase10_root / Path(run["report_json"])
        if not report_path.exists():
            print(f"[WARN] missing report: {report_path}")
            continue
        report = _load_json(report_path)

        method_reports = report.get("method_reports") or {}
        for method, mr in method_reports.items():
            row: Dict[str, Any] = dict(run)
            row["method"] = method
            row.update(_compute_metrics(mr))
            row["label"] = _label(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows extracted from reports")

    # Save CSV/MD
    csv_path = out_dir / "phase10_summary.csv"
    df.to_csv(csv_path, index=False)

    # Markdown summary grouped by pivot_type
    md_lines: List[str] = []
    md_lines.append("# Phase 10 Rule Policy Summary\n")
    md_lines.append("Metrics are computed on final episode only (episode_id==3).\n")
    if not _HAS_MPL:
        md_lines.append("(Note) matplotlib is not installed; figures were skipped.\n")
    for pivot_type in sorted(df["pivot_type"].unique()):
        md_lines.append(f"## pivot_type = {pivot_type}\n")
        sub = df[df["pivot_type"] == pivot_type].copy()
        # Only keep the relevant methods (baseline full/sim + goc)
        show_cols = [
            "label",
            "final_accuracy",
            "final_pivot_accuracy",
            "final_nonpivot_accuracy",
            "final_token_mean",
            "final_pivot_token_mean",
            "final_nonpivot_token_mean",
            "avoided_node_injected_rate",
            "pivot_rate_final",
        ]
        sub = sub.sort_values(["final_pivot_accuracy","final_accuracy","final_token_mean"], ascending=[False,False,True])
        md_lines.append(sub[show_cols].to_markdown(index=False))
        md_lines.append("")

        # figures
        if _HAS_MPL:
            sub_plot = sub.dropna(subset=["final_token_mean", "final_pivot_accuracy"])
            if len(sub_plot) >= 2:
                _scatter(sub_plot, f"{pivot_type}: pivot_acc vs tokens", "final_token_mean", "final_pivot_accuracy", fig_dir / f"{pivot_type}_pivotacc_vs_tokens.png")
                sub_plot2 = sub.dropna(subset=["final_token_mean", "final_accuracy"])
                _scatter(sub_plot2, f"{pivot_type}: final_acc vs tokens", "final_token_mean", "final_accuracy", fig_dir / f"{pivot_type}_finalacc_vs_tokens.png")

    (out_dir / "phase10_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
