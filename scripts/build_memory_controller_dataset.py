#!/usr/bin/env python3
"""Build SFT-style datasets for an agentic fold/unfold controller (Option-B).

This script reads per-task JSONL traces written by the LLM runner (--log_dir) and
extracts controller decision points for UNFOLD selection.

Two data sources:
  - teacher: uses memory events of type 'unfold' that include candidates + chosen_seed_ids.
            (Requires GoCMemory.trace_unfold_candidates=True.)
  - controller: uses agent events of type 'controller_unfold' (candidate list + picked_seed_ids).
  - auto: prefer teacher; fall back to controller.

Output format: JSONL, each row:
  {
    "messages": [
      {"role":"system","content":...},
      {"role":"user","content":...},
      {"role":"assistant","content":...}
    ],
    "meta": {...}
  }

You can feed the resulting JSONL into an OpenAI chat fine-tuning job or adapt it
to your preferred trainer.

Example:
  python scripts/build_memory_controller_dataset.py \
      --log_dir runs/hotpot_traces \
      --out_path runs/controller_sft.jsonl \
      --source auto

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so `import src.*` works when running as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


SYS_PROMPT = (
    "You are a memory-controller that selects which stored memory nodes to UNFOLD. "
    "Return exactly one JSON object and nothing else."
)


def _format_candidates(cands: List[Dict[str, Any]], max_cand: int = 24) -> str:
    lines: List[str] = []
    for i, c in enumerate(cands[:max_cand], start=1):
        sid = c.get("seed_id") or c.get("id") or ""
        score = float(c.get("score", 0.0))
        cost = int(c.get("cost_tokens", 0))
        step = int(c.get("seed_step", -1))
        docids = c.get("seed_docids") or []
        docids_s = ",".join([str(d) for d in docids[:4]])
        preview = str(c.get("preview") or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."
        lines.append(f"{i}. id={sid} score={score:.4f} cost={cost} step={step} docids={docids_s} | {preview}")
    return "\n".join(lines)


def _build_user_prompt(query: str, budget_unfold: int, k: int, cands: List[Dict[str, Any]], max_sel: int = 6) -> str:
    return (
        f"QUERY: {query}\n"
        f"BUDGET_UNFOLD_TOKENS: {int(budget_unfold)}\n"
        f"TARGET_K: {int(k)} (select <= {int(max_sel)} seed ids)\n\n"
        "Each candidate is a SEED node; selecting it will unfold its dependency closure.\n"
        "Pick a small set that likely contains the needed evidence with minimal cost.\n\n"
        "CANDIDATES:\n" + _format_candidates(cands) + "\n\n"
        "Return JSON with the key select_seed_ids as a list of ids from the candidates.\n"
        "Example: {\"select_seed_ids\":[\"N_12\",\"N_51\"]}\n"
    )


def _assistant_json(seed_ids: List[str]) -> str:
    return json.dumps({"select_seed_ids": seed_ids}, ensure_ascii=False)


def _iter_trace_files(log_dir: Path) -> List[Path]:
    return sorted([Path(p) for p in glob.glob(str(log_dir / "trace_*.jsonl"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--source", type=str, default="auto", choices=["auto", "teacher", "controller"])
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--include_fallback", action="store_true", help="Include controller_unfold rows where used_fallback=True")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_files = _iter_trace_files(log_dir)
    if args.max_files:
        trace_files = trace_files[: int(args.max_files)]

    written = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for tf in trace_files:
            for line in tf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue

                # Teacher source: memory emits type='unfold' with candidates + chosen_seed_ids.
                # Newer traces may wrap memory events as {type:'mem_event', ev_type:'unfold', event:{...}}.
                teacher_ev = None
                if ev.get("type") == "unfold" and ev.get("mem") == "GoC":
                    teacher_ev = ev
                elif ev.get("type") == "mem_event" and ev.get("ev_type") == "unfold":
                    teacher_ev = ev.get("event")

                if args.source in ("auto", "teacher") and isinstance(teacher_ev, dict) and teacher_ev.get("type") == "unfold" and teacher_ev.get("mem") == "GoC":
                    cands = teacher_ev.get("candidates") or []
                    chosen = teacher_ev.get("chosen_seed_ids") or []
                    if not cands or not chosen:
                        continue
                    query = str(teacher_ev.get("query") or "")
                    bud = int(teacher_ev.get("budget_unfold") or 0)
                    k = int(teacher_ev.get("k") or len(chosen) or 6)
                    user = _build_user_prompt(query, bud, k, cands)
                    row = {
                        "messages": [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": _assistant_json([str(x) for x in chosen])},
                        ],
                        "meta": {
                            "source": "teacher",
                            "trace_file": tf.name,
                            "task_id": ev.get("task_id"),
                            "method": ev.get("method"),
                            "step": ev.get("step"),
                        },
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if args.max_examples and written >= int(args.max_examples):
                        break

                # Controller source: agent emits type='controller_unfold'
                if args.source in ("auto", "controller") and ev.get("type") == "controller_unfold":
                    if (not args.include_fallback) and bool(ev.get("used_fallback")):
                        continue
                    cands = ev.get("candidates") or []
                    chosen = ev.get("picked_seed_ids") or []
                    if not cands or not chosen:
                        continue
                    query = str(ev.get("query") or "")
                    bud = int(ev.get("budget_unfold") or 0)
                    k = int(ev.get("k") or len(chosen) or 6)
                    user = _build_user_prompt(query, bud, k, cands)
                    row = {
                        "messages": [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": _assistant_json([str(x) for x in chosen])},
                        ],
                        "meta": {
                            "source": "controller",
                            "trace_file": tf.name,
                            "task_id": ev.get("task_id"),
                            "method": ev.get("method"),
                            "step": ev.get("step"),
                            "reason": ev.get("reason"),
                        },
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if args.max_examples and written >= int(args.max_examples):
                        break

            if args.max_examples and written >= int(args.max_examples):
                break

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
