"""Rebuild sweep_master_*.jsonl by scanning run directories.

Why this exists:
- You may run multiple sweeps into the same out_dir, or manually copy run folders.
- The master summary JSONL can end up incomplete or out-of-sync.

This script treats on-disk run folders as the source of truth.

Usage:
  python rebuild_sweep_master.py --out_dir sweeps
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summarize_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Return per-method summary like run_sweep._summarize_jsonl."""
    rows = load_jsonl(path)
    if not rows:
        return []

    # Group by method
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("method") or "")
        by_method.setdefault(m, []).append(r)

    out: List[Dict[str, Any]] = []
    for m, ms in sorted(by_method.items(), key=lambda kv: kv[0]):
        n = len(ms)
        if n == 0:
            continue
        correct = sum(1 for r in ms if bool(r.get("correct")))
        # usage.total_tokens is present for llm runs
        toks = []
        for r in ms:
            u = r.get("usage") or {}
            if isinstance(u, dict) and ("total_tokens" in u):
                try:
                    toks.append(float(u["total_tokens"]))
                except Exception:
                    pass
        out.append({
            "method": m,
            "n": n,
            "acc": correct / n,
            "avg_total_tokens": (sum(toks) / len(toks)) if toks else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="sweeps")
    ap.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional output master path. Default: sweep_master_<benchmark>_<runner>.jsonl in out_dir, if benchmark/runner are discoverable; otherwise sweep_master_rebuilt.jsonl.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit(f"out_dir does not exist: {out_dir}")

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    # Identify run dirs by presence of run_config.json
    run_dirs = [p for p in run_dirs if (p / "run_config.json").exists()]
    if not run_dirs:
        raise SystemExit(f"No run directories found under: {out_dir}")

    # Try to infer benchmark/runner from first run_config
    first_cfg = json.loads((run_dirs[0] / "run_config.json").read_text(encoding="utf-8"))
    bench = str(first_cfg.get("benchmark") or "unknown")
    runner = str(first_cfg.get("runner") or "unknown")

    if args.master:
        master_path = Path(args.master)
    else:
        master_path = out_dir / f"sweep_master_{bench}_{runner}.jsonl"

    master_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for rd in sorted(run_dirs, key=lambda p: p.name):
        cfg = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
        run_id = str(cfg.get("run_id") or rd.name)
        done = (rd / "DONE").exists()

        # Summarize
        runner = str(cfg.get("runner") or "llm")
        results_path = rd / ("llm_results.jsonl" if runner == "llm" else "results.jsonl")
        summary_by_method = summarize_results_jsonl(results_path)

        records.append({
            "run_id": run_id,
            "name": cfg.get("name"),
            "status": "ok" if done else "partial",
            "benchmark": cfg.get("benchmark"),
            "runner": cfg.get("runner"),
            "methods": cfg.get("methods"),
            "params": cfg.get("params"),
            "bench_kwargs": cfg.get("bench_kwargs"),
            "config_path": str(cfg.get("config_path") or ""),
            "artifacts": {},
            "summary_by_method": summary_by_method,
            "taskwise_counts": None,
            "taskwise_artifacts": {},
            "session_stamp": None,
            "run_index": None,
        })

    with open(master_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Rebuilt master summary: {master_path} (runs={len(records)})")


if __name__ == "__main__":
    main()
