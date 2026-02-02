"""
Multi-commit contract checker.

Usage:
  python scripts/check_multicommit_contract.py --trace_glob "runs/**/traces/*.jsonl" --strict

What it checks (best-effort, trace-driven):
- For commit-flow tasks: stage advancement is only via return-injection (no auto-inject).
- SUBTASK/MERGE/FINAL markers appear (progression sanity).
- No merge-stage gating deadlocks (return_blocked loops with stage_kind == "merge").
- Return tool events show injected correctly when used.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # keep going (traces can include partial lines in rare cases)
                continue

def scan_trace(path: str):
    stats = defaultdict(int)
    markers = defaultdict(int)

    # For stage order sanity (very lightweight)
    saw_subtask = False
    saw_merge = False
    saw_final = False

    for ev in iter_jsonl(path):
        # 1) user turn injected events
        if ev.get("type") == "user_turn_injected":
            stats["user_turn_injected"] += 1
            reason = ev.get("reason")
            if reason == "auto":
                stats["inject_auto"] += 1
            elif reason == "return":
                stats["inject_return"] += 1

        # 2) return blocked
        if ev.get("type") == "return_blocked":
            stats["return_blocked"] += 1
            stage_kind = ev.get("stage_kind")
            if stage_kind == "merge":
                stats["return_blocked_merge"] += 1
            elif stage_kind == "subtask":
                stats["return_blocked_subtask"] += 1

        # 3) tool events (return/finish)
        tool = ev.get("tool")
        if tool == "return":
            stats["tool_return"] += 1
            if ev.get("ignored") is True:
                stats["tool_return_ignored"] += 1
            if ev.get("injected") is True:
                stats["tool_return_injected_true"] += 1
        if tool == "finish":
            stats["tool_finish"] += 1

        # 4) prompt markers
        # Depending on implementation, the prompt text might appear under different keys.
        text = ""
        if isinstance(ev.get("prompt"), str):
            text = ev["prompt"]
        elif isinstance(ev.get("user_prompt"), str):
            text = ev["user_prompt"]
        elif isinstance(ev.get("content"), str):
            text = ev["content"]

        if text:
            if "[SUBTASK" in text:
                markers["subtask_marker"] += 1
                saw_subtask = True
            if "[MERGE" in text:
                markers["merge_marker"] += 1
                saw_merge = True
            if "[FINAL" in text:
                markers["final_marker"] += 1
                saw_final = True

    # derive simple booleans
    stats["saw_subtask"] = int(saw_subtask)
    stats["saw_merge"] = int(saw_merge)
    stats["saw_final"] = int(saw_final)
    stats["markers_subtask"] = markers["subtask_marker"]
    stats["markers_merge"] = markers["merge_marker"]
    stats["markers_final"] = markers["final_marker"]

    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_glob", required=True, help='Glob like "runs/**/traces/*.jsonl"')
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if contract violations found")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.trace_glob, recursive=True))
    if not paths:
        print(json.dumps({"ok": False, "error": "no traces matched", "glob": args.trace_glob}, indent=2))
        return 2

    agg = defaultdict(int)
    per = {}
    violations = []

    for p in paths:
        st = scan_trace(p)
        per[p] = dict(st)
        for k, v in st.items():
            agg[k] += int(v) if isinstance(v, int) else 0

        # --- Contract checks (best effort) ---
        # 1) no auto-inject for commit-flow
        if st.get("inject_auto", 0) > 0:
            violations.append({"trace": p, "rule": "no_auto_inject", "value": st["inject_auto"]})

        # 2) should see subtask marker at least once
        if st.get("saw_subtask", 0) == 0:
            violations.append({"trace": p, "rule": "must_see_subtask", "value": 0})

        # 3) merge gating deadlock should be absent
        if st.get("return_blocked_merge", 0) > 0:
            violations.append({"trace": p, "rule": "no_merge_return_gating", "value": st["return_blocked_merge"]})

    out = {
        "ok": len(violations) == 0,
        "n_traces": len(paths),
        "aggregate": dict(agg),
        "violations": violations[:50],  # cap
    }
    print(json.dumps(out, indent=2))

    if args.strict and violations:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())