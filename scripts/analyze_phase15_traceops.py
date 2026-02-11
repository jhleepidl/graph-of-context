#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase15_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    phase15_root = Path(args.phase15_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(phase15_root / "run_manifest.json")
    runs = list(manifest.get("runs") or [])

    rows: List[Dict[str, Any]] = []
    for run in runs:
        report_path = phase15_root / Path(str(run.get("report_json", "")))
        if not report_path.exists():
            continue
        report = _load_json(report_path)
        method_reports = report.get("method_reports") or {}
        scenario_params = report.get("scenario_params") or {}
        for method, mr in method_reports.items():
            metrics = dict(mr.get("metrics") or {})
            row: Dict[str, Any] = dict(run)
            row.update(
                {
                    "method": method,
                    "traceops_level": int(
                        row.get("traceops_level")
                        or scenario_params.get("traceops_level")
                        or 1
                    ),
                    "traceops_scenario": str(
                        row.get("traceops_scenario")
                        or scenario_params.get("traceops_scenario")
                        or "mixed"
                    ),
                    "traceops_delay_to_relevance": int(
                        row.get("traceops_delay_to_relevance")
                        or scenario_params.get("traceops_delay_to_relevance")
                        or 0
                    ),
                    "traceops_distractor_branching": int(
                        row.get("traceops_distractor_branching")
                        or scenario_params.get("traceops_distractor_branching")
                        or 0
                    ),
                    "traceops_contradiction_rate": _as_float(
                        row.get("traceops_contradiction_rate")
                        or scenario_params.get("traceops_contradiction_rate")
                    ),
                    "pivot_decision_accuracy": _as_float(
                        metrics.get("pivot_decision_accuracy")
                    ),
                    "pivot_e3_only_accuracy": _as_float(
                        metrics.get("pivot_e3_only_accuracy")
                    ),
                    "strict_pivot_accuracy": _as_float(
                        metrics.get("strict_pivot_accuracy")
                    ),
                    "tokens_pivot_mean": _as_float(metrics.get("tokens_pivot_mean")),
                    "tokens_total_mean": _as_float(metrics.get("tokens_total_mean")),
                    "mean_avoid_targets_per_pivot": _as_float(
                        metrics.get("mean_avoid_targets_per_pivot")
                    ),
                    "avoided_injected_rate": _as_float(
                        metrics.get("avoided_injected_rate")
                    ),
                    "revive_success_rate": _as_float(metrics.get("revive_success_rate")),
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "phase15_traceops_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines: List[str] = []
    md_lines.append("# Phase15 TraceOps Summary")
    md_lines.append("")
    if df.empty:
        md_lines.append("No rows found.")
    else:
        md_lines.append("## Raw Rows")
        md_lines.append("")
        raw_cols = [
            "traceops_level",
            "traceops_scenario",
            "method",
            "pivot_decision_accuracy",
            "pivot_e3_only_accuracy",
            "strict_pivot_accuracy",
            "tokens_pivot_mean",
            "tokens_total_mean",
            "mean_avoid_targets_per_pivot",
            "avoided_injected_rate",
            "revive_success_rate",
        ]
        cols = [c for c in raw_cols if c in df.columns]
        md_lines.append(df[cols].to_markdown(index=False))
        md_lines.append("")

        group_cols = ["traceops_level", "traceops_scenario", "method"]
        agg_cols = [
            "pivot_decision_accuracy",
            "pivot_e3_only_accuracy",
            "strict_pivot_accuracy",
            "tokens_pivot_mean",
            "tokens_total_mean",
            "mean_avoid_targets_per_pivot",
            "avoided_injected_rate",
            "revive_success_rate",
        ]
        grouped = (
            df.groupby(group_cols, dropna=False)[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(group_cols)
        )
        md_lines.append("## Aggregated By Level/Scenario/Method")
        md_lines.append("")
        md_lines.append(grouped.to_markdown(index=False))
        md_lines.append("")

        def _bucket_delay(v: Any) -> str:
            try:
                val = int(v)
            except Exception:
                return "unknown"
            if val <= 2:
                return "short"
            if val <= 5:
                return "mid"
            return "long"

        def _bucket_branch(v: Any) -> str:
            try:
                val = int(v)
            except Exception:
                return "unknown"
            if val <= 1:
                return "low"
            if val <= 3:
                return "mid"
            return "high"

        def _bucket_contra(v: Any) -> str:
            if not isinstance(v, (int, float)):
                return "unknown"
            if v < 0.2:
                return "low"
            if v < 0.5:
                return "mid"
            return "high"

        bucket_df = df.copy()
        bucket_df["delay_bucket"] = bucket_df["traceops_delay_to_relevance"].map(_bucket_delay)
        bucket_df["branch_bucket"] = bucket_df["traceops_distractor_branching"].map(_bucket_branch)
        bucket_df["contra_bucket"] = bucket_df["traceops_contradiction_rate"].map(_bucket_contra)

        bucket_group = (
            bucket_df.groupby(["delay_bucket", "branch_bucket", "contra_bucket", "method"], dropna=False)[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["delay_bucket", "branch_bucket", "contra_bucket", "method"])
        )
        md_lines.append("## Aggregated By Knob Buckets")
        md_lines.append("")
        md_lines.append(bucket_group.to_markdown(index=False))

    md_path = out_dir / "phase15_traceops_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
