#!/usr/bin/env python3
"""Phase 13 analyzer: end-to-end (episode 1~3) accuracy + cost under commit-anchor-by-evidence,
plus Phase 13 integrity/diagnostics.

Input: phase13/run_manifest.json + report JSONs produced by policyops.run compare.
Output: phase13_summary.csv/.md (+ optional figures).

Adds to Phase 12 summary:
- commit_both_rate: mean(commit1_correct & commit2_correct)
- p_final_given_commits: P(final_correct | both commits ok)
- unseen_e3_rate/unseen_e3_mean: whether Episode 3 context contains clauses unseen in episodes 1..2
  (computed as |e3_context_clause_ids \ e12_opened_clause_ids|)
- stale_evidence_rate (pivot threads): mean(stale_evidence) on episode-3 records

Optional:
- Frontier attribution (GoC only): when goc_graph_jsonl_path exists (requires --save_goc_graph and sample_rate),
  compute fraction of retrieved docs first seen from stage=graph_frontier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HAS_MPL = False
try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_mean_bool(xs: List[bool]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(xs))


def _pctl(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.array(xs, dtype=float), q))


def _scatter(df: pd.DataFrame, title: str, x: str, y: str, out: Path) -> None:
    if not _HAS_MPL:
        return
    plt.figure()  # type: ignore
    plt.scatter(df[x], df[y])  # type: ignore
    for _, r in df.iterrows():
        plt.annotate(str(r.get("label", "")), (r[x], r[y]), fontsize=7)  # type: ignore
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
        if th is None or pt is None:
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
        mp[th] = mp.get(th, False) or is_pivot
    return mp


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
    if v == "goc_rule_v1_no_frontier":
        pk = row.get("pivot_k")
        ph = row.get("pivot_h")
        return f"goc_rule_v1_no_frontier(D4,2->P{pk},{ph})"
    if v == "goc_rule_v1_world_universe":
        pk = row.get("pivot_k")
        ph = row.get("pivot_h")
        return f"goc_rule_v1_world_universe(D4,2->P{pk},{ph})"
    return str(v)


def _episode3_unseen_stats(task_recs: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Return (unseen_rate, unseen_mean_count) over episode-3 records.

    unseen_count = |e3_context_clause_ids \ e12_opened_clause_ids|
    - If fields are missing, those records are skipped.
    """
    counts: List[int] = []
    for r in task_recs:
        if int(r.get("episode_id") or 0) != 3:
            continue
        e3_ids = r.get("e3_context_clause_ids") or []
        e12_ids = r.get("e12_opened_clause_ids") or []
        if not isinstance(e3_ids, list) or not isinstance(e12_ids, list):
            continue
        unseen = set(map(str, e3_ids)) - set(map(str, e12_ids))
        counts.append(len(unseen))
    if not counts:
        return (float("nan"), float("nan"))
    unseen_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in counts]))
    unseen_mean = float(np.mean(counts))
    return unseen_rate, unseen_mean


def _stale_evidence_pivot(task_recs: List[Dict[str, Any]]) -> float:
    """Mean stale_evidence on episode-3 pivot tasks."""
    vals: List[float] = []
    for r in task_recs:
        if int(r.get("episode_id") or 0) != 3:
            continue
        if not bool(r.get("is_pivot_task")):
            continue
        if r.get("stale_evidence") is None:
            continue
        vals.append(1.0 if bool(r.get("stale_evidence")) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _avoided_node_injected_rate(task_recs: List[Dict[str, Any]]) -> float:
    """Fraction of episode-3 tasks where avoid-target nodes were still injected."""
    vals: List[float] = []
    for r in task_recs:
        if int(r.get("episode_id") or 0) != 3:
            continue
        should_avoid = bool(r.get("goc_avoids_edge_injected"))
        avoid_ids = r.get("goc_avoid_target_clause_ids") or []
        if (not should_avoid) and isinstance(avoid_ids, list) and avoid_ids:
            should_avoid = True
        if not should_avoid:
            continue
        injected = r.get("goc_avoided_node_injected")
        if isinstance(injected, bool):
            vals.append(1.0 if injected else 0.0)
            continue
        if isinstance(avoid_ids, list):
            e3_ids = set(map(str, r.get("e3_context_clause_ids") or []))
            vals.append(1.0 if (set(map(str, avoid_ids)) & e3_ids) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _frontier_first_seen_rate_from_graph(graph_jsonl: Path) -> Optional[float]:
    """Best-effort: fraction of *doc nodes* whose first_seen_stage == 'graph_frontier'.

    This requires GoC graph jsonl saving. If the file is missing or schema differs, return None.
    """
    try:
        if not graph_jsonl.exists():
            return None
        total = 0
        frontier = 0
        with graph_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") != "node":
                    continue
                if obj.get("node_type") != "doc":
                    continue
                total += 1
                stage = obj.get("attrs", {}).get("first_seen_stage") or obj.get("first_seen_stage")
                if stage == "graph_frontier":
                    frontier += 1
        if total == 0:
            return None
        return float(frontier / total)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase13_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--frontier_graph_sample_only", action="store_true", help="Only use tasks with saved goc_graph_jsonl_path for frontier stats")
    args = ap.parse_args()

    phase13_root = Path(args.phase13_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    if _HAS_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(phase13_root / "run_manifest.json")
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []

    for run in runs:
        report_path = phase13_root / Path(run["report_json"])
        if not report_path.exists():
            print(f"[WARN] missing report: {report_path}")
            continue
        report = _load_json(report_path)
        method_reports = report.get("method_reports") or {}

        for method_name, mr in method_reports.items():
            thread_recs = list(mr.get("thread_records") or [])
            task_recs = list(mr.get("records") or [])

            final_correct = [bool(t.get("thread_judge_correct")) for t in thread_recs]
            c1 = [bool(t.get("commit1_correct")) for t in thread_recs]
            c2 = [bool(t.get("commit2_correct")) for t in thread_recs]
            both = [a and b for a, b in zip(c1, c2)]
            strict = [a and b and c for a, b, c in zip(c1, c2, final_correct)]
            resil = [((a or b) and c) for a, b, c in zip(c1, c2, final_correct)]

            commit_both_rate = _safe_mean_bool(both)
            p_final_given_commits = float("nan")
            idx = [i for i, ok in enumerate(both) if ok]
            if idx:
                p_final_given_commits = float(np.mean([1.0 if final_correct[i] else 0.0 for i in idx]))

            pivot_mp = _pivot_threads(task_recs)
            pivot_flags = [pivot_mp.get(t.get("thread_id"), False) for t in thread_recs]
            pivot_threads = [t for t, is_p in zip(thread_recs, pivot_flags) if is_p]
            nonpivot_threads = [t for t, is_p in zip(thread_recs, pivot_flags) if not is_p]

            pivot_final_correct = [bool(t.get("thread_judge_correct")) for t in pivot_threads]
            nonpivot_final_correct = [bool(t.get("thread_judge_correct")) for t in nonpivot_threads]

            tok_mp = _thread_tokens(task_recs)
            total_tok = [tok_mp.get(t.get("thread_id"), {}).get("total", float("nan")) for t in thread_recs]
            e3_tok = [tok_mp.get(t.get("thread_id"), {}).get("e3", float("nan")) for t in thread_recs]
            e1_tok = [tok_mp.get(t.get("thread_id"), {}).get("e1", float("nan")) for t in thread_recs]
            e2_tok = [tok_mp.get(t.get("thread_id"), {}).get("e2", float("nan")) for t in thread_recs]

            total_tok_f = [float(x) for x in total_tok if x == x]
            e3_tok_f = [float(x) for x in e3_tok if x == x]
            e1_tok_f = [float(x) for x in e1_tok if x == x]
            e2_tok_f = [float(x) for x in e2_tok if x == x]

            unseen_rate, unseen_mean = _episode3_unseen_stats(task_recs)
            stale_pivot = _stale_evidence_pivot(task_recs)
            avoided_injected_rate = _avoided_node_injected_rate(task_recs)

            # Frontier graph attribution (best-effort, GoC only)
            frontier_rates: List[float] = []
            if method_name == "goc":
                for r in task_recs:
                    if int(r.get("episode_id") or 0) not in {1, 2}:
                        continue
                    gp = r.get("goc_graph_jsonl_path")
                    if not gp:
                        continue
                    val = _frontier_first_seen_rate_from_graph((phase13_root / gp) if not Path(gp).is_absolute() else Path(gp))
                    if val is not None:
                        frontier_rates.append(val)
                frontier_first_seen_rate = float(np.mean(frontier_rates)) if frontier_rates else float("nan")
            else:
                frontier_first_seen_rate = float("nan")

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
                    "commit_both_rate": commit_both_rate,
                    "p_final_given_commits": p_final_given_commits,
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
                    "unseen_e3_rate": unseen_rate,
                    "unseen_e3_mean_count": unseen_mean,
                    "stale_evidence_pivot_rate": stale_pivot,
                    "avoided_node_injected_rate": avoided_injected_rate,
                    "frontier_first_seen_rate": frontier_first_seen_rate,
                }
            )
            row["label"] = _label(row)
            rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = out_dir / "phase13_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines: List[str] = []
    md_lines.append("# Phase 13 E2E summary\n")
    md_lines.append("Added columns vs Phase 12:")
    md_lines.append("- commit_both_rate, p_final_given_commits\n")
    md_lines.append("- unseen_e3_rate/unseen_e3_mean_count (closed-book integrity)\n")
    md_lines.append("- stale_evidence_pivot_rate (pivot robustness)\n")
    md_lines.append("- avoided_node_injected_rate (pivot avoids integrity)\n")
    md_lines.append("- frontier_first_seen_rate (if GoC graphs were saved)\n")

    show_cols = [
        "pivot_type",
        "label",
        "final_accuracy",
        "strict_e2e_accuracy",
        "commit_both_rate",
        "p_final_given_commits",
        "unseen_e3_rate",
        "tokens_total_mean",
        "tokens_total_p95",
        "tokens_e3_mean",
        "stale_evidence_pivot_rate",
        "avoided_node_injected_rate",
        "frontier_first_seen_rate",
    ]

    if not df.empty:
        md_lines.append("## By pivot_type\n")
        for pivot_type in sorted(df["pivot_type"].unique()):
            sub = df[df["pivot_type"] == pivot_type].copy().sort_values("tokens_total_mean")
            md_lines.append(f"### {pivot_type}\n")
            cols = [c for c in show_cols if c in sub.columns]
            md_lines.append(sub[cols].to_markdown(index=False))
            md_lines.append("\n")

            _scatter(
                sub,
                title=f"Phase13 {pivot_type}: strict E2E vs total tokens",
                x="tokens_total_mean",
                y="strict_e2e_accuracy",
                out=fig_dir / f"scatter_{pivot_type}_strict_e2e_vs_total_tokens.png",
            )
            _scatter(
                sub,
                title=f"Phase13 {pivot_type}: final acc vs total tokens",
                x="tokens_total_mean",
                y="final_accuracy",
                out=fig_dir / f"scatter_{pivot_type}_finalacc_vs_total_tokens.png",
            )

    md_path = out_dir / "phase13_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if _HAS_MPL:
        print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
