#!/usr/bin/env python3
"""Classify per-task failure types from experiment outputs.

Reads JSONL files produced by:
  - LLM runner: llm_results.jsonl
  - Deterministic runner: results.jsonl

If --input is a directory, the script searches recursively for those files.

Outputs (to --out_dir):
  - failure_by_task.csv : one row per task result (method x task_id)
  - failure_summary.csv : counts by method x failure_type
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ANSWER_PAIR_RE = re.compile(r"\b(Project_\d{4})\b\s*\|\s*\b([A-Za-z]+_\d+)\b")
PROJ_RE = re.compile(r"\bProject_\d{4}\b")
CITY_RE = re.compile(r"\b([A-Za-z]+_\d+)\b")

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            # ignore malformed lines
            continue
    return rows

def _find_result_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    out: List[Path] = []
    for name in ("llm_results.jsonl", "results.jsonl"):
        out.extend(list(inp.rglob(name)))
    return sorted(set(out))

def _maybe_extract_answer_from_expl(expl: str) -> Optional[str]:
    s = (expl or "").strip()
    if not s:
        return None
    m = ANSWER_PAIR_RE.search(s)
    if m:
        return f"{m.group(1)} | {m.group(2)}"
    # common phrasing: "Project_0041 ... headquarters is City_42"
    m2 = re.search(r"\b(Project_\d{4})\b.*?\bheadquarters\b.*?\b([A-Za-z]+_\d+)\b", s, flags=re.I | re.S)
    if m2:
        return f"{m2.group(1)} | {m2.group(2)}"
    return None

def _classify_one(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (failure_type, extra_flags)."""
    pred = (row.get("pred") or row.get("answer") or "").strip()
    gold = (row.get("gold") or "").strip()
    correct = bool(row.get("correct"))
    expl = (row.get("explanation") or "")
    ts = row.get("tool_stats") or {}
    steps = row.get("steps") or 0

    # Derived signals
    has_open = int(ts.get("open_page_calls") or 0) > 0
    has_search = int(ts.get("search_calls") or 0) > 0
    rep_search = int(ts.get("repeated_search_count") or 0)
    search_calls = int(ts.get("search_calls") or 0)
    open_cache_hits = int(ts.get("open_page_cache_hits") or 0)
    blocked_search = int(ts.get("blocked_search_query") or 0)
    cooled = int(ts.get("search_queries_cooled_down") or 0)
    unproductive = int(ts.get("unproductive_searches") or 0)
    dup_open_blocked = int(ts.get("duplicate_open_blocked") or 0)
    constraints_written = int(ts.get("constraints_written") or 0)
    invalid_blocks = int(ts.get("finish_invalid_project_blocked") or 0)
    premature_blocks = int(ts.get("premature_finish_blocked") or 0)
    salvaged = int(ts.get("finish_answer_salvaged") or 0)
    docid_cov = float(row.get("docid_cov") or 0.0)

    flags: Dict[str, Any] = {
        "has_open_page": has_open,
        "has_search": has_search,
        "repeated_search_count": rep_search,
        "open_page_cache_hits": open_cache_hits,
        "blocked_search_query": blocked_search,
        "search_queries_cooled_down": cooled,
        "unproductive_searches": unproductive,
        "duplicate_open_blocked": dup_open_blocked,
        "constraints_written": constraints_written,
        "finish_invalid_project_blocked": invalid_blocks,
        "finish_answer_salvaged": salvaged,
        "premature_finish_blocked": premature_blocks,
        "docid_cov": docid_cov,
    }

    # Primary failure classification (priority order)
    if correct:
        return "correct", flags

    if "max_steps reached" in (expl or "").lower():
        if cooled > 0 or blocked_search > 0 or dup_open_blocked > 0:
            return "no_finish_with_policy_intervention", flags
        return "no_finish", flags

    if not pred:
        # Detect the common bug: answer was written in explanation but pred is empty.
        inferred = _maybe_extract_answer_from_expl(expl)
        if inferred:
            flags["answer_in_explanation"] = True
            flags["inferred_answer"] = inferred
            return "empty_answer_field_but_present_in_explanation", flags
        if salvaged > 0:
            return "answer_salvage_attempted_but_pred_empty", flags
        return "empty_pred", flags

    if invalid_blocks > 0:
        return "invalid_project_finish_blocked", flags

    if not has_open:
        return "no_open_page_evidence", flags

    if docid_cov == 0.0:
        return "no_gold_doc_opened", flags

    # Heuristic: search loop (many repeats vs searches)
    if search_calls >= 4 and rep_search >= max(3, int(0.5 * search_calls)):
        if cooled > 0 or blocked_search > 0:
            return "search_loop_with_cooldown", flags
        return "search_loop", flags

    # Heuristic: tool protocol misuse
    if int(ts.get("return_in_main_ignored") or 0) >= 2:
        return "tool_protocol_return_in_main", flags

    # Default: wrong answer (with non-empty pred)
    return "wrong_answer", flags

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union keys in stable order
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main() -> None:
    ap = argparse.ArgumentParser(description="Classify per-task failure types from llm_results.jsonl.")
    ap.add_argument("--input", type=str, required=True, help="Path to llm_results.jsonl OR a directory containing sweep runs.")
    ap.add_argument("--out_dir", type=str, default="failure_report", help="Output directory for CSV reports.")
    ap.add_argument("--truncate_explanation", type=int, default=220, help="Characters of explanation to include in failure_by_task.csv.")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _find_result_files(inp)
    if not files:
        raise SystemExit(f"No result files found under: {inp}")

    by_task_rows: List[Dict[str, Any]] = []
    summary_counter: Dict[Tuple[str, str], int] = {}

    for fp in files:
        rows = _read_jsonl(fp)
        for r in rows:
            method = str(r.get("method") or "")
            task_id = str(r.get("task_id") or "")
            failure_type, flags = _classify_one(r)

            summary_counter[(method, failure_type)] = summary_counter.get((method, failure_type), 0) + 1

            usage = r.get("usage") or {}
            ts = r.get("tool_stats") or {}

            by_task_rows.append({
                "source_file": str(fp),
                "run_tag": r.get("run_tag", ""),
                "method": method,
                "task_id": task_id,
                "failure_type": failure_type,
                "correct": bool(r.get("correct")),
                "gold": (r.get("gold") or ""),
                "pred": (r.get("pred") or ""),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "steps": int(r.get("steps") or 0),
                "search_calls": int(ts.get("search_calls") or 0),
                "repeated_search_count": int(ts.get("repeated_search_count") or 0),
                "open_page_calls": int(ts.get("open_page_calls") or 0),
                "open_page_cache_hits": int(ts.get("open_page_cache_hits") or 0),
                "blocked_search_query": int(ts.get("blocked_search_query") or 0),
                "search_queries_cooled_down": int(ts.get("search_queries_cooled_down") or 0),
                "unproductive_searches": int(ts.get("unproductive_searches") or 0),
                "duplicate_open_blocked": int(ts.get("duplicate_open_blocked") or 0),
                "constraints_written": int(ts.get("constraints_written") or 0),
                "tool_calls_proposed_total": int(ts.get("tool_calls_proposed_total") or 0),
                "finish_answer_salvaged": int(ts.get("finish_answer_salvaged") or 0),
                "finish_invalid_project_blocked": int(ts.get("finish_invalid_project_blocked") or 0),
                "premature_finish_blocked": int(ts.get("premature_finish_blocked") or 0),
                "docid_cov": float(r.get("docid_cov") or 0.0),
                "explanation_head": (r.get("explanation") or "")[: int(args.truncate_explanation)],
                **{f"flag_{k}": v for k, v in flags.items()},
            })

    # Sort for readability
    by_task_rows.sort(key=lambda x: (x.get("method",""), x.get("task_id","")))

    summary_rows: List[Dict[str, Any]] = []
    for (method, ftype), n in sorted(summary_counter.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        summary_rows.append({"method": method, "failure_type": ftype, "count": n})

    _write_csv(out_dir / "failure_by_task.csv", by_task_rows)
    _write_csv(out_dir / "failure_summary.csv", summary_rows)

    print("Wrote:", out_dir / "failure_by_task.csv") 
    print("Wrote:", out_dir / "failure_summary.csv")

if __name__ == "__main__":
    main()
