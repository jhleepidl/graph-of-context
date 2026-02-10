#!/usr/bin/env python3
"""Phase 11 analyzer: end-to-end (episode 1~3) accuracy + cost.

Input: phase11/run_manifest.json + report JSONs produced by policyops.run compare.
Output: phase11_summary.csv/.md and optional figures.

Key metrics (thread-level):
- final_accuracy: mean(thread_judge_correct)
- strict_e2e_accuracy: mean(commit1_correct & commit2_correct & thread_judge_correct)
- commit_resilient_accuracy: mean((commit1_correct | commit2_correct) & thread_judge_correct)

Cost metrics:
- tokens_total_*: sum(prompt_tokens) across episodes 1..3 per thread (mean/p95/p99)
- tokens_e3_*: episode-3 prompt_tokens per thread (mean/p95/p99)

Notes:
- We use thread_records for correctness, and per-task records for prompt_tokens.
- Some backends may not populate completion tokens; we intentionally use prompt_tokens only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting
_HAS_MPL = False
try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(v: float) -> float:
    return float(v) if v is not None else float("nan")


def _safe_mean_bool(xs: List[bool]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(xs))


def _pctl(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.array(xs, dtype=float), q))


def _label(row: Dict[str, Any]) -> str:
    v = row.get("variant")
    m = row.get("method")
    if v == "baseline":
        return str(m)
    if v == "goc_fixed_light":
        return "goc_fixed_light(K4,H2)"
    if v == "goc_fixed_heavy":
        return "goc_fixed_heavy(K16,H3)"
    if v == "goc_adaptive_heavy":
        return "goc_adaptive_heavy(D4,2->P16,3)"
    if v == "goc_rule_v1":
        pk = row.get("pivot_k")
        ph = row.get("pivot_h")
        return f"goc_rule_v1(D4,2->P{pk},{ph})"
    return str(v)


def _scatter(df: pd.DataFrame, title: str, x: str, y: str, out: Path) -> None:
    if not _HAS_MPL:
        return
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


def _thread_tokens(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Return per-thread token sums by episode and total: {thread_id: {e1,e2,e3,total}}"""
    out: Dict[str, Dict[str, float]] = {}
    for r in records:
        th = r.get("thread_id")
        ep = int(r.get("episode_id") or 0)
        pt = r.get("prompt_tokens")
        if th is None:
            continue
        if pt is None:
            continue
        out.setdefault(th, {"e1": 0.0, "e2": 0.0, "e3": 0.0, "total": 0.0})
        if ep == 1:
            out[th]["e1"] += float(pt)
        elif ep == 2:
            out[th]["e2"] += float(pt)
        elif ep == 3:
            out[th]["e3"] += float(pt)
        out[th]["total"] += float(pt)
    return out


def _pivot_threads(records: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Map thread_id -> is_pivot_task based on episode-3 records."""
    mp: Dict[str, bool] = {}
    for r in records:
        if int(r.get("episode_id") or 0) != 3:
            continue
        th = r.get("thread_id")
        if not th:
            continue
        is_pivot = bool(r.get("is_pivot_task"))
        # If multiple, treat thread as pivot if any ep3 record is pivot
        mp[th] = mp.get(th, False) or is_pivot
    return mp


def _avoided_node_injected_rate(task_recs: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for r in task_recs:
        if int(r.get("episode_id") or 0) != 3:
            continue
        avoid_ids = r.get("goc_avoid_target_clause_ids") or []
        if not isinstance(avoid_ids, list) or not avoid_ids:
            continue
        injected = r.get("goc_avoided_node_injected")
        if isinstance(injected, bool):
            vals.append(1.0 if injected else 0.0)
            continue
        e3_ids = set(map(str, r.get("e3_context_clause_ids") or []))
        vals.append(1.0 if (set(map(str, avoid_ids)) & e3_ids) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase11_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase11_root = Path(args.phase11_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    if _HAS_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(phase11_root / "run_manifest.json")
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []

    for run in runs:
        report_path = phase11_root / Path(run["report_json"])
        if not report_path.exists():
            print(f"[WARN] missing report: {report_path}")
            continue
        report = _load_json(report_path)
        method_reports = report.get("method_reports") or {}

        # Baseline report contains 2 methods; goc reports contain only goc.
        for method_name, mr in method_reports.items():
            thread_recs = list(mr.get("thread_records") or [])
            task_recs = list(mr.get("records") or [])

            # Thread-level correctness
            final_correct = [bool(t.get("thread_judge_correct")) for t in thread_recs]
            c1 = [bool(t.get("commit1_correct")) for t in thread_recs]
            c2 = [bool(t.get("commit2_correct")) for t in thread_recs]
            strict = [a and b and c for a, b, c in zip(c1, c2, final_correct)]
            resil = [((a or b) and c) for a, b, c in zip(c1, c2, final_correct)]

            # Pivot threads (episode-3)
            pivot_mp = _pivot_threads(task_recs)
            pivot_flags = [pivot_mp.get(t.get("thread_id"), False) for t in thread_recs]
            pivot_threads = [t for t, is_p in zip(thread_recs, pivot_flags) if is_p]
            nonpivot_threads = [t for t, is_p in zip(thread_recs, pivot_flags) if not is_p]

            pivot_final_correct = [bool(t.get("thread_judge_correct")) for t in pivot_threads]
            nonpivot_final_correct = [bool(t.get("thread_judge_correct")) for t in nonpivot_threads]

            # Tokens (prompt-only)
            tok_mp = _thread_tokens(task_recs)
            total_tok = [tok_mp.get(t.get("thread_id"), {}).get("total", float("nan")) for t in thread_recs]
            e3_tok = [tok_mp.get(t.get("thread_id"), {}).get("e3", float("nan")) for t in thread_recs]
            e1_tok = [tok_mp.get(t.get("thread_id"), {}).get("e1", float("nan")) for t in thread_recs]
            e2_tok = [tok_mp.get(t.get("thread_id"), {}).get("e2", float("nan")) for t in thread_recs]

            # filter out NaNs for percentiles
            total_tok_f = [float(x) for x in total_tok if x == x]
            e3_tok_f = [float(x) for x in e3_tok if x == x]
            e1_tok_f = [float(x) for x in e1_tok if x == x]
            e2_tok_f = [float(x) for x in e2_tok if x == x]
            avoided_injected_rate = _avoided_node_injected_rate(task_recs)

            row: Dict[str, Any] = dict(run)
            row.update(
                {
                    "method": method_name,
                    "label": "",
                    "final_threads": len(thread_recs),
                    "pivot_threads": int(sum(bool(x) for x in pivot_flags)),
                    "pivot_rate_final": float(np.mean([1.0 if x else 0.0 for x in pivot_flags])) if pivot_flags else float("nan"),
                    "final_accuracy": _safe_mean_bool(final_correct),
                    "final_pivot_accuracy": _safe_mean_bool(pivot_final_correct),
                    "final_nonpivot_accuracy": _safe_mean_bool(nonpivot_final_correct),
                    "commit1_accuracy": _safe_mean_bool(c1),
                    "commit2_accuracy": _safe_mean_bool(c2),
                    "strict_e2e_accuracy": _safe_mean_bool(strict),
                    "commit_resilient_accuracy": _safe_mean_bool(resil),
                    "tokens_total_mean": float(np.mean(total_tok_f)) if total_tok_f else float("nan"),
                    "tokens_total_p95": _pctl(total_tok_f, 95),
                    "tokens_total_p99": _pctl(total_tok_f, 99),
                    "tokens_e3_mean": float(np.mean(e3_tok_f)) if e3_tok_f else float("nan"),
                    "tokens_e3_p95": _pctl(e3_tok_f, 95),
                    "tokens_e3_p99": _pctl(e3_tok_f, 99),
                    "tokens_e1_mean": float(np.mean(e1_tok_f)) if e1_tok_f else float("nan"),
                    "tokens_e2_mean": float(np.mean(e2_tok_f)) if e2_tok_f else float("nan"),
                    "avoided_node_injected_rate": avoided_injected_rate,
                }
            )
            row["label"] = _label(row)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = out_dir / "phase11_summary.csv"
    df.to_csv(csv_path, index=False)

    # Markdown summary
    md_lines: List[str] = []
    md_lines.append("# Phase 11 E2E summary\n")
    md_lines.append("Key columns:")
    md_lines.append("- final_accuracy: mean thread_judge_correct (episode 3)\n")
    md_lines.append("- strict_e2e_accuracy: commit1 & commit2 & final\n")
    md_lines.append("- commit_resilient_accuracy: (commit1|commit2) & final\n")
    md_lines.append("- tokens_total_*: sum of prompt_tokens across episodes 1..3 per thread\n")
    md_lines.append("- avoided_node_injected_rate: avoid-target leakage in episode-3 context\n")

    show_cols = [
        "pivot_type",
        "label",
        "final_accuracy",
        "strict_e2e_accuracy",
        "commit_resilient_accuracy",
        "tokens_total_mean",
        "tokens_total_p95",
        "tokens_e3_mean",
        "pivot_rate_final",
        "avoided_node_injected_rate",
    ]
    if not df.empty:
        md_lines.append("## By pivot_type\n")
        for pivot_type in sorted(df["pivot_type"].unique()):
            sub = df[df["pivot_type"] == pivot_type].copy()
            sub = sub.sort_values("tokens_total_mean")
            md_lines.append(f"### {pivot_type}\n")
            md_lines.append(sub[show_cols].to_markdown(index=False))
            md_lines.append("\n")

            # Figures
            _scatter(
                sub,
                title=f"Phase11 {pivot_type}: strict E2E vs total tokens",
                x="tokens_total_mean",
                y="strict_e2e_accuracy",
                out=fig_dir / f"scatter_{pivot_type}_strict_e2e_vs_total_tokens.png",
            )
            _scatter(
                sub,
                title=f"Phase11 {pivot_type}: final acc vs total tokens",
                x="tokens_total_mean",
                y="final_accuracy",
                out=fig_dir / f"scatter_{pivot_type}_finalacc_vs_total_tokens.png",
            )

    md_path = out_dir / "phase11_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if _HAS_MPL:
        print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
