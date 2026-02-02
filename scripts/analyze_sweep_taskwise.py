#!/usr/bin/env python3
"""Analyze a sweep directory and emit task-wise artifacts.

This script is useful when you want to answer:
  - In each run, which tasks are GoC-only wins vs FullHistory-only wins?
  - Across runs (e.g., different difficulty levels), which tasks are consistently
    won by one method?

It reads per-run `llm_results.jsonl` and writes:
  - per-run `taskwise/` artifacts
  - a sweep-level `taskwise_sweep.jsonl` (one row per task_id per run)
  - a sweep-level `taskwise_sweep_summary.json` (counts per run)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path so `import src.*` works when running as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis.taskwise import build_taskwise, load_jsonl, write_taskwise_artifacts


def _read_run_cfg(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True, help="Sweep output dir (contains 12-char run_id folders)")
    ap.add_argument("--pair", default="GoC,FullHistory", help="Comparison pair, e.g. GoC,FullHistory")
    ap.add_argument("--out", default=None, help="Optional output directory (defaults to sweep_dir)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_root = Path(args.out) if args.out else sweep_dir
    out_root.mkdir(parents=True, exist_ok=True)

    pair = tuple([x.strip() for x in args.pair.split(",") if x.strip()])
    if len(pair) != 2:
        raise SystemExit("--pair must be like 'GoC,FullHistory'")
    pair_t: Tuple[str, str] = (pair[0], pair[1])

    run_dirs = [p for p in sweep_dir.iterdir() if p.is_dir() and (p / "llm_results.jsonl").exists()]
    run_dirs.sort(key=lambda p: p.name)
    if not run_dirs:
        raise SystemExit(f"No run dirs with llm_results.jsonl under {sweep_dir}")

    sweep_rows: List[Dict[str, Any]] = []
    sweep_summary: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        rows = load_jsonl(run_dir / "llm_results.jsonl")
        cfg = _read_run_cfg(run_dir)
        methods = cfg.get("methods")
        summ = build_taskwise(rows, methods=methods, pair=pair_t)
        taskwise_dir = run_dir / "taskwise"
        artifacts = write_taskwise_artifacts(summ, taskwise_dir, prefix="taskwise")

        # per-run summary row
        bw = ((cfg.get("params") or {}).get("bench_kwargs") or cfg.get("bench_kwargs") or {})
        sweep_summary.append({
            "run_id": run_dir.name,
            "pair": pair_t,
            "counts": summ.counts,
            "budget_active": (cfg.get("params") or {}).get("budget_active"),
            "bench_kwargs": bw,
            "taskwise_dir": str(taskwise_dir),
            **artifacts,
        })

        # per-task per-run rows
        for t in summ.tasks:
            sweep_rows.append({
                "run_id": run_dir.name,
                "task_id": t.get("task_id"),
                "winner_vs_pair": t.get("winner_vs_pair"),
                "methods_correct": t.get("methods_correct"),
                "unique_winner": t.get("unique_winner"),
                "delta_tokens_A_minus_B": t.get("delta_tokens_A_minus_B"),
                "delta_steps_A_minus_B": t.get("delta_steps_A_minus_B"),
                "correct_by_method": t.get("correct_by_method"),
                "tokens_by_method": t.get("tokens_by_method"),
                "steps_by_method": t.get("steps_by_method"),
            })

    out_jsonl = out_root / "taskwise_sweep.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in sweep_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_sum = out_root / "taskwise_sweep_summary.json"
    out_sum.write_text(json.dumps(sweep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote:\n- {out_jsonl}\n- {out_sum}")


if __name__ == "__main__":
    main()
