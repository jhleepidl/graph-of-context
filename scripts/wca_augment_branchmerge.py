"""Create a *derived* WebChoreArena JSON split with an explicit branch-merge requirement.

Motivation
----------
Vanilla WebChoreArena already stresses memory (Massive/Long-Term). However, our
synthetic benchmark intentionally creates a *branch-merge* decision where two
independent sub-traces must both be preserved to answer a final question.

This script produces an **augmented** split that is still *state-evaluable* like
the original (we do not change the underlying env/evaluator fields), but makes
the agent's *reasoning trace* naturally branchy:

  1) The agent must explore two alternatives (A and B) before committing.
  2) It must remember a key detail from both branches.
  3) It must merge them into a final decision and execute the original task.

Important caveat
---------------
Because WebChoreArena's official scoring is environment-state-based, this
augmentation primarily serves as a *stress test for context management* rather
than an official leaderboard submission.

Usage
-----
python scripts/wca_augment_branchmerge.py \
  --in_json <WebChoreArena>/BrowserGym/config_files/<tasks>.json \
  --out_json wca_branchmerge_aug.json \
  --only_type_main "Long-Term Memory" \
  --max_tasks 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _augment_intent(original_intent: str) -> str:
    # Keep it concise to avoid making tasks *much* harder.
    return (
        "Before executing the task, do the following:\n"
        "(A) Explore Option A: collect the key info you would use to solve the task.\n"
        "(B) Explore Option B: collect an alternative key info path (a different candidate / page / approach).\n"
        "Then MERGE: explicitly compare A vs B and decide, and execute the task.\n\n"
        "While exploring, remember at least one concrete detail from A and one from B "
        "(e.g., a product name, a number, a policy line, a username, a setting).\n\n"
        "Original task:\n"
        + original_intent.strip()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--only_type_main", default=None, help='Filter tasks by type_main (exact match)')
    ap.add_argument("--max_tasks", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    tasks: List[Dict[str, Any]] = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []

    for t in tasks:
        if args.only_type_main and t.get("type_main") != args.only_type_main:
            continue

        t2 = dict(t)
        t2["task_id"] = f"{t.get('task_id')}_goc_branchmerge"
        t2["intent"] = _augment_intent(str(t.get("intent", "")))
        # Keep a structured tag (useful for analysis scripts)
        t2.setdefault("goc_meta", {})
        try:
            t2["goc_meta"]["aug"] = "branchmerge_v1"
        except Exception:
            t2["goc_meta"] = {"aug": "branchmerge_v1"}
        out.append(t2)

        if args.max_tasks and len(out) >= args.max_tasks:
            break

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} tasks to {args.out_json}")


if __name__ == "__main__":
    main()
