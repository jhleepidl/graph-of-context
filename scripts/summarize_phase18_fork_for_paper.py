#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

MAIN_METHOD_ORDER = [
    "full",
    "similarity_only",
    "agent_fold",
    "goc",
    "goc_fork_dep",
    "goc_fork_sim",
    "goc_fork_full",
]

MAIN_COLUMNS = [
    "traceops_scenario",
    "traceops_delay_to_relevance",
    "variant",
    "method",
    "fork_scope_mode",
    "fork_max_tokens",
    "pivot_decision_accuracy",
    "strict_pivot_accuracy",
    "pred_commit_rate",
    "pred_committed_accuracy",
    "tokens_total_mean",
    "tokens_savings_vs_full",
    "accuracy_delta_vs_full",
]

BUDGET_COLUMNS = [
    "traceops_scenario",
    "traceops_delay_to_relevance",
    "method",
    "fork_scope_mode",
    "fork_max_tokens",
    "pivot_decision_accuracy",
    "strict_pivot_accuracy",
    "tokens_total_mean",
    "tokens_savings_vs_full",
    "accuracy_delta_vs_full",
]


def _safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _sort_methods(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(MAIN_METHOD_ORDER)}
    out = df.copy()
    out["_method_order"] = out["method"].map(lambda x: order.get(str(x), 999))
    sort_cols = [c for c in ["traceops_delay_to_relevance", "_method_order", "fork_max_tokens"] if c in out.columns]
    out = out.sort_values(sort_cols).drop(columns=["_method_order"])
    return out


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)\n"
    return df.to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize phase18 fork results into paper-friendly CSV/MD tables.")
    ap.add_argument("--summary_csv", type=Path, required=True)
    ap.add_argument("--scenario", type=str, default="")
    ap.add_argument("--delay", type=int, default=None)
    ap.add_argument("--output_prefix", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    if args.scenario:
        df = df.loc[df.get("traceops_scenario", "").astype(str) == str(args.scenario)].copy()
    if args.delay is not None and "traceops_delay_to_relevance" in df.columns:
        df = df.loc[df["traceops_delay_to_relevance"] == int(args.delay)].copy()

    # Main comparison table: one row per main method, prefer goc_phase18_depwalk over older goc variant.
    main_df = df.loc[df.get("method", "").astype(str).isin(MAIN_METHOD_ORDER)].copy()
    if not main_df.empty:
        # Prefer the phase18 depwalk goc row when multiple goc rows exist.
        if "variant" in main_df.columns:
            main_df["_variant_rank"] = 1
            main_df.loc[main_df["variant"].astype(str).str.contains("phase18", case=False, na=False), "_variant_rank"] = 0
            dedupe_keys = [c for c in ["traceops_scenario", "traceops_delay_to_relevance", "method", "fork_max_tokens"] if c in main_df.columns]
            main_df = main_df.sort_values(dedupe_keys + ["_variant_rank"]).drop_duplicates(subset=dedupe_keys, keep="first")
            main_df = main_df.drop(columns=["_variant_rank"])
        main_df = _sort_methods(main_df)
        main_df = main_df[_safe_cols(main_df, MAIN_COLUMNS)]

    # Budget table: fork methods only.
    budget_df = df.loc[df.get("method", "").astype(str).isin(["goc_fork_dep", "goc_fork_sim", "goc_fork_full"])].copy()
    if not budget_df.empty:
        budget_df = budget_df[_safe_cols(budget_df, BUDGET_COLUMNS)]
        sort_cols = [c for c in ["traceops_delay_to_relevance", "method", "fork_max_tokens"] if c in budget_df.columns]
        budget_df = budget_df.sort_values(sort_cols)

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    main_csv = out_prefix.with_name(out_prefix.name + "_main.csv")
    main_md = out_prefix.with_name(out_prefix.name + "_main.md")
    budget_csv = out_prefix.with_name(out_prefix.name + "_budget.csv")
    budget_md = out_prefix.with_name(out_prefix.name + "_budget.md")

    main_df.to_csv(main_csv, index=False)
    main_md.write_text(_to_md(main_df), encoding="utf-8")
    budget_df.to_csv(budget_csv, index=False)
    budget_md.write_text(_to_md(budget_df), encoding="utf-8")

    print(f"[OK] wrote {main_csv}")
    print(f"[OK] wrote {main_md}")
    print(f"[OK] wrote {budget_csv}")
    print(f"[OK] wrote {budget_md}")


if __name__ == "__main__":
    main()
