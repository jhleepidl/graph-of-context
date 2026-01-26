"""Analyze HotpotQA "levels" sweeps and produce a compact flip-map report.

This script is intended for the workflow described in v68 notes:
  - Fix a task set (e.g., tradeoff_v3 taskwise_unique_SimilarityOnly.txt)
  - Run multiple level configurations (anaphoric/noise/trap/repeat)
  - Summarize where methods flip winners

It reads per-run `llm_results.jsonl` and emits:
  - levels_summary.md: condition -> accuracy by method
  - flip_map.csv: condition -> (GoC - SimilarityOnly) delta

Usage:
  python scripts/analyze_levels_flipmap.py --root /path/to/levels_runs --out /path/to/out_dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def parse_run_name(name: str) -> Dict[str, int]:
    """Best-effort parse of the canonical levels folder naming scheme."""
    m = re.match(r"levels_.*?_b(\d+)_a(\d+)_t(\d+)_r(\d+)_n(\d+)", name)
    if not m:
        return {}
    return {
        "budget": int(m.group(1)),
        "anaphoric": int(m.group(2)),
        "trap": int(m.group(3)),
        "repeat": int(m.group(4)),
        "noise": int(m.group(5)),
    }


def load_results(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "llm_results.jsonl"
    if not p.exists():
        return pd.DataFrame()
    params = parse_run_name(run_dir.name)
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            obj.update(params)
            obj["run"] = run_dir.name
            rows.append(obj)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Directory containing many level run folders")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dfs = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            df = load_results(d)
            if len(df):
                dfs.append(df)

    if not dfs:
        raise SystemExit(f"No llm_results.jsonl found under: {root}")

    df = pd.concat(dfs, ignore_index=True)

    group_cols = ["anaphoric", "trap", "repeat", "noise"]
    g = (
        df.groupby(group_cols + ["method"], dropna=False)
        .agg(n=("task_id", "count"), acc=("correct", "mean"), acc_strict=("correct_strict", "mean"))
        .reset_index()
    )

    pivot = g.pivot_table(index=group_cols, columns="method", values="acc")
    pivot_strict = g.pivot_table(index=group_cols, columns="method", values="acc_strict")

    # Flip-map: focus on GoC vs SimilarityOnly if present
    flip = pd.DataFrame(index=pivot.index).reset_index()
    if "GoC" in pivot.columns and "SimilarityOnly" in pivot.columns:
        flip["delta_GoC_minus_Sim"] = pivot["GoC"].values - pivot["SimilarityOnly"].values
    else:
        flip["delta_GoC_minus_Sim"] = float("nan")

    flip.to_csv(out / "flip_map.csv", index=False)

    # Markdown summary
    md = []
    md.append(f"# Levels sweep summary\n\nRoot: `{root}`\n\n")
    md.append("## Accuracy (non-strict)\n\n")
    md.append(pivot.reset_index().to_markdown(index=False))
    md.append("\n\n## Accuracy (strict)\n\n")
    md.append(pivot_strict.reset_index().to_markdown(index=False))
    md.append("\n\n## Flip-map (GoC - SimilarityOnly)\n\n")
    md.append(flip.to_markdown(index=False))
    (out / "levels_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote:")
    print(" -", (out / "levels_summary.md").resolve())
    print(" -", (out / "flip_map.csv").resolve())


if __name__ == "__main__":
    main()
