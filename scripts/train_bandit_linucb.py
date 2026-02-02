#!/usr/bin/env python3
"""Train a LinUCB unfold controller from an offline dataset.

Input dataset format: JSONL rows as produced by scripts/build_bandit_dataset.py:
  {"x": [...], "reward": float, "meta": {...}}

This simply replays the logged interactions and updates A^{-1}, b online.
It is effectively ridge regression with uncertainty, packaged as a controller
that GoC-Bandit can load.

Example:
  python scripts/train_bandit_linucb.py \
    --dataset runs/hp/bandit_dataset.jsonl \
    --out_model runs/hp/bandit_model.json \
    --alpha 1.2 --ridge 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path so `import src...` works when invoked as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.bandit_controller import BanditUnfoldController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--feature_version", type=str, default="v1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0, help="Exploration for inference; keep 0 during offline replay")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--reward_clip", type=float, default=None, help="If set, clip rewards to [-clip, +clip]")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ln in ds_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
        if args.max_rows and len(rows) >= int(args.max_rows):
            break

    if args.shuffle:
        random.seed(int(args.seed))
        random.shuffle(rows)

    ctl = BanditUnfoldController(
        alpha=float(args.alpha),
        epsilon=float(args.epsilon),
        ridge=float(args.ridge),
        feature_version=str(args.feature_version),
        seed=int(args.seed),
    )

    n = 0
    r_sum = 0.0
    r_sq = 0.0
    for row in rows:
        x = row.get("x")
        if not isinstance(x, list):
            continue
        r = float(row.get("reward") or 0.0)
        if args.reward_clip is not None:
            c = float(args.reward_clip)
            r = max(-c, min(c, r))
        try:
            ctl.update_from_features(x, r)
            n += 1
            r_sum += r
            r_sq += r * r
        except Exception:
            continue

    ctl.save_json(str(out_path))

    mean = (r_sum / n) if n else 0.0
    var = (r_sq / n - mean * mean) if n else 0.0
    print(f"Updated on {n} rows")
    print(f"Reward mean={mean:.4f} var={var:.4f}")
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
