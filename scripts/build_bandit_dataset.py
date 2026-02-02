#!/usr/bin/env python3
"""Build an offline contextual-bandit dataset for GoC unfold selection.

Why?
  - Option-B uses a small controller LLM to pick which GoC storage seeds to unfold.
  - This script lets you replace that controller with a cheap *contextual bandit*.

What it does
  1) Reads LLM runner results (llm_results.jsonl) to get episode-level reward signals.
  2) Reads per-task trace JSONLs (--log_dir) and extracts GoC 'unfold' events that include:
       - candidates (seed list with score/cost/closure_size/docids)
       - chosen_seed_ids (teacher action)
  3) Converts each chosen seed into a featurized interaction row:
       {"x": [...], "reward": r, "meta": {...}}

Notes
  - This is *offline / logged* learning: we learn from whatever policy generated the traces.
    For true online exploration-learning, see scripts/train_bandit_online.py (optional).

Example
  python run_benchmark.py --benchmark hotpotqa --methods GoC --log_dir runs/hp_traces --out_dir runs/hp
  python scripts/build_bandit_dataset.py \
    --results_path runs/hp/llm_results.jsonl \
    --log_dir runs/hp_traces \
    --out_path runs/hp/bandit_dataset.jsonl

  python scripts/train_bandit_linucb.py --dataset runs/hp/bandit_dataset.jsonl --out_model runs/hp/bandit_model.json
  python run_benchmark.py --benchmark hotpotqa --methods GoC-Bandit --bandit_model_path runs/hp/bandit_model.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so `import src...` works when invoked as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.bandit_controller import BanditUnfoldController


def _iter_trace_files(log_dir: Path) -> List[Path]:
    return sorted([Path(p) for p in glob.glob(str(log_dir / "trace_*.jsonl"))])


def _load_results(results_path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Map (run_tag, method, task_id) -> result row."""
    m: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for ln in results_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        key = (str(row.get("run_tag") or ""), str(row.get("method") or ""), str(row.get("task_id") or ""))
        if all(key):
            m[key] = row
    return m


def _extract_committed_titles_from_return_message(msg: Any) -> List[str]:
    """Best-effort parse of stage-1 committed titles."""
    if msg is None:
        return []
    if isinstance(msg, dict):
        for k in ["supporting_titles", "evidence_titles", "titles"]:
            v = msg.get(k)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        for k in ["evidence_title", "title", "primary_title"]:
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
        return []
    if isinstance(msg, str):
        s = msg.strip()
        # Try to extract {...}
        if "{" in s and "}" in s:
            i = s.find("{")
            j = s.rfind("}")
            cand = s[i : j + 1] if 0 <= i < j else s
            # Strip code fences
            cand = re.sub(r"^```(?:json)?\s*", "", cand, flags=re.I)
            cand = re.sub(r"\s*```\s*$", "", cand)
            try:
                obj = json.loads(cand)
                return _extract_committed_titles_from_return_message(obj)
            except Exception:
                return []
    return []


def _reward_from_result(row: Dict[str, Any], *, w_acc: float, w_cov: float, w_tok: float) -> float:
    correct = 1.0 if bool(row.get("correct")) else 0.0
    cov = float(row.get("docid_cov") or 0.0)
    usage = row.get("usage") or {}
    toks = float(usage.get("total_tokens") or 0.0)
    # Penalize 1k tokens as 1.0 reward unit by default weight.
    return float(w_acc) * correct + float(w_cov) * cov - float(w_tok) * (toks / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", type=str, required=True, help="Path to llm_results.jsonl")
    ap.add_argument("--log_dir", type=str, required=True, help="Trace dir produced by --log_dir")
    ap.add_argument("--out_path", type=str, required=True, help="Output JSONL dataset path")
    ap.add_argument("--feature_version", type=str, default="v1")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--w_acc", type=float, default=1.0, help="Weight on correctness")
    ap.add_argument("--w_cov", type=float, default=0.0, help="Weight on docid coverage")
    ap.add_argument("--w_tok", type=float, default=0.15, help="Weight on token penalty (per 1k tokens)")
    ap.add_argument(
        "--empty_action",
        type=str,
        default="skip",
        choices=["skip", "min_cost", "max_score", "best_ratio"],
        help=(
            "What to do if an unfold event has candidates but no chosen_seed_ids. "
            "Default 'skip' (recommended for real training). "
            "Other options are for smoke tests / pipeline debugging."
        ),
    )
    args = ap.parse_args()

    results_path = Path(args.results_path)
    log_dir = Path(args.log_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    controller = BanditUnfoldController(alpha=1.0, epsilon=0.0, feature_version=str(args.feature_version))

    trace_files = _iter_trace_files(log_dir)
    if args.max_files:
        trace_files = trace_files[: int(args.max_files)]

    written = 0
    skipped_no_result = 0
    skipped_no_unfold = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for tf in trace_files:
            committed_titles: List[str] = []
            # We'll infer the episode key from trace events (run_tag, method, task_id).
            ep_key: Optional[Tuple[str, str, str]] = None

            lines = tf.read_text(encoding="utf-8").splitlines()
            # First pass: find ep metadata + commit titles.
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    ev = json.loads(ln)
                except Exception:
                    continue

                if ep_key is None:
                    rt = str(ev.get("run_tag") or "")
                    md = str(ev.get("method") or "")
                    tid = str(ev.get("task_id") or "")
                    if rt and md and tid:
                        ep_key = (rt, md, tid)

                if ev.get("type") == "tool" and str(ev.get("tool") or "") == "return":
                    msg = None
                    try:
                        msg = (ev.get("args") or {}).get("message")
                    except Exception:
                        msg = None
                    titles = _extract_committed_titles_from_return_message(msg)
                    if titles:
                        committed_titles = titles

            if not ep_key or ep_key not in results:
                skipped_no_result += 1
                continue

            reward = _reward_from_result(results[ep_key], w_acc=args.w_acc, w_cov=args.w_cov, w_tok=args.w_tok)
            had_any = False

            # Second pass: unfold events.
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    ev = json.loads(ln)
                except Exception:
                    continue

                unfold_ev = None
                trace_step = None
                if ev.get("type") == "unfold" and ev.get("mem") == "GoC":
                    unfold_ev = ev
                    trace_step = ev.get("step")
                elif ev.get("type") == "mem_event" and ev.get("ev_type") == "unfold":
                    unfold_ev = ev.get("event")
                    trace_step = ev.get("step")
                if not isinstance(unfold_ev, dict):
                    continue
                if (unfold_ev.get("mem") or "GoC") != "GoC":
                    # The raw unfold event has mem='GoC'. Wrapped mem_event keeps it inside.
                    pass

                cands = unfold_ev.get("candidates") or []
                chosen = unfold_ev.get("chosen_seed_ids") or unfold_ev.get("picked_seed_ids") or []
                if not cands:
                    continue

                pseudo_action = False
                if not chosen:
                    # Some traces may not record teacher actions (e.g., budget too low). For real training
                    # you should skip these; for smoke tests you can synthesize a single action.
                    if str(args.empty_action).lower() == "skip":
                        continue
                    pseudo_action = True
                    bud = int(unfold_ev.get("budget_unfold") or 0)
                    # Prefer affordable candidates when possible.
                    affordable = [c for c in cands if int(c.get("cost_tokens", 0) or 0) <= bud] or list(cands)
                    mode = str(args.empty_action).lower()
                    if mode == "min_cost":
                        best = min(affordable, key=lambda c: int(c.get("cost_tokens", 10**9) or 10**9))
                    elif mode == "max_score":
                        best = max(affordable, key=lambda c: float(c.get("score", 0.0) or 0.0))
                    else:  # best_ratio
                        best = max(
                            affordable,
                            key=lambda c: float(c.get("score", 0.0) or 0.0) / float(max(1, int(c.get("cost_tokens", 0) or 0))),
                        )
                    sid = str(best.get("seed_id") or "").strip()
                    if not sid:
                        continue
                    chosen = [sid]

                # Use outer trace step if present; else fall back to memory global_step.
                now_step = int(trace_step) if trace_step is not None else int(unfold_ev.get("global_step") or unfold_ev.get("global_step") or 0)
                cand_by_id = {str(c.get("seed_id") or ""): c for c in cands if isinstance(c, dict) and c.get("seed_id")}

                for sid in chosen:
                    s = str(sid)
                    c = cand_by_id.get(s)
                    if not c:
                        continue
                    x = controller.featurize(c, now_step=now_step, committed_titles=committed_titles).tolist()
                    row = {
                        "x": x,
                        "reward": float(reward),
                        "meta": {
                            "trace_file": tf.name,
                            "run_tag": ep_key[0],
                            "method": ep_key[1],
                            "task_id": ep_key[2],
                            "decision_step": int(now_step),
                            "seed_id": s,
                            "budget_unfold": int(unfold_ev.get("budget_unfold") or 0),
                            "k": int(unfold_ev.get("k") or len(chosen)),
                            "have_committed_titles": bool(committed_titles),
                            "pseudo_action": bool(pseudo_action),
                            "empty_action_mode": str(args.empty_action),
                        },
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    had_any = True
                    if args.max_rows and written >= int(args.max_rows):
                        break
                if args.max_rows and written >= int(args.max_rows):
                    break

            if not had_any:
                skipped_no_unfold += 1
            if args.max_rows and written >= int(args.max_rows):
                break

    print(f"Wrote {written} bandit rows to {out_path}")
    print(f"Skipped (no matching result): {skipped_no_result}")
    print(f"Skipped (no unfold events): {skipped_no_unfold}")


if __name__ == "__main__":
    main()
