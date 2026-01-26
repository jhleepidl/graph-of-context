"""Task-wise analysis for multi-method runs.

This module helps answer questions like:
  - Which tasks did GoC solve that FullHistory did not (and vice versa)?
  - For those tasks, did token usage/steps/tool usage differ?
  - Which tasks are consistently hard/easy across a sweep?

The runner produces one JSONL row per (method, task_id). We aggregate those
rows into one record per task_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            # ignore malformed lines
            continue
    return rows


@dataclass
class TaskwiseSummary:
    """Aggregated per-task matrix + a few convenience counts."""

    tasks: List[Dict[str, Any]]
    counts: Dict[str, int]
    methods: List[str]
    pair: Tuple[str, str]


def build_taskwise(rows: Iterable[Dict[str, Any]],
                  methods: Optional[List[str]] = None,
                  pair: Tuple[str, str] = ("GoC", "FullHistory"),
                  include_fields: Optional[List[str]] = None) -> TaskwiseSummary:
    """Aggregate rows (one per method+task) into per-task records.

    Each per-task record includes:
      - correct_by_method
      - tokens_by_method
      - steps_by_method
      - docid_cov_by_method
      - json_fail_by_method
      - json_recover_by_method
      - winner_vs_pair: one of {"A_only", "B_only", "both", "neither", "tie"}
      - delta_tokens_A_minus_B
    """
    include_fields = include_fields or []
    # Determine methods if not provided
    seen_methods: List[str] = []
    if methods is None:
        for r in rows:
            m = r.get("method")
            if isinstance(m, str) and m not in seen_methods:
                seen_methods.append(m)
        methods = seen_methods
    else:
        methods = [m for m in methods if m]

    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    # Keep optional task_index fields if present in rows.
    task_meta: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        tid = r.get("task_id")
        m = r.get("method")
        if not tid or not m:
            continue
        by_task.setdefault(str(tid), {})[str(m)] = r
        # Optional extras to carry along
        extras: Dict[str, Any] = {}
        for k in include_fields:
            if k in r:
                extras[k] = r.get(k)
        if extras:
            task_meta.setdefault(str(tid), {}).update(extras)

    A, B = pair
    tasks_out: List[Dict[str, Any]] = []

    counts = {
        f"{A}_only": 0,
        f"{B}_only": 0,
        "both": 0,
        "neither": 0,
        "tie": 0,
        "tasks": 0,
    }

    for tid, by_m in by_task.items():
        rec: Dict[str, Any] = {
            "task_id": tid,
            "correct_by_method": {},
            "tokens_by_method": {},
            "steps_by_method": {},
            "docid_cov_by_method": {},
            "json_fail_by_method": {},
            "json_recover_by_method": {},
        }
        if tid in task_meta:
            rec.update(task_meta[tid])

        for m in methods:
            r = by_m.get(m)
            if not r:
                continue
            rec["correct_by_method"][m] = bool(r.get("correct"))
            u = r.get("usage") or {}
            rec["tokens_by_method"][m] = _safe_int(u.get("total_tokens"), 0)
            rec["steps_by_method"][m] = _safe_int(r.get("steps"), 0)
            rec["docid_cov_by_method"][m] = _safe_float(r.get("docid_cov"), 0.0)
            ts = r.get("tool_stats") or {}
            rec["json_fail_by_method"][m] = _safe_int(ts.get("json_parse_failures"), 0)
            rec["json_recover_by_method"][m] = _safe_int(ts.get("json_recoveries"), 0)
            # Some extra tool stats that are handy for debugging
            if "tool_calls_by_method" not in rec:
                rec["tool_calls_by_method"] = {}
            rec["tool_calls_by_method"][m] = _safe_int(ts.get("tool_calls_total"), 0)
            if "open_page_calls_by_method" not in rec:
                rec["open_page_calls_by_method"] = {}
            rec["open_page_calls_by_method"][m] = _safe_int(ts.get("open_page_calls"), 0)
            if "search_calls_by_method" not in rec:
                rec["search_calls_by_method"] = {}
            rec["search_calls_by_method"][m] = _safe_int(ts.get("search_calls"), 0)

        # Which methods solved this task?
        solved = [m for m, ok in rec["correct_by_method"].items() if ok]
        rec["methods_correct"] = solved
        rec["unique_winner"] = solved[0] if len(solved) == 1 else None

        a_ok = bool(rec["correct_by_method"].get(A, False))
        b_ok = bool(rec["correct_by_method"].get(B, False))
        if a_ok and (not b_ok):
            label = f"{A}_only"
        elif b_ok and (not a_ok):
            label = f"{B}_only"
        elif a_ok and b_ok:
            label = "both"
        else:
            label = "neither"
        # tie (both wrong or both correct) is still useful for overlap analysis
        if a_ok == b_ok:
            counts["tie"] += 1
        counts[label] += 1
        counts["tasks"] += 1

        rec["winner_vs_pair"] = label
        rec["delta_tokens_A_minus_B"] = _safe_int(rec["tokens_by_method"].get(A, 0), 0) - _safe_int(rec["tokens_by_method"].get(B, 0), 0)
        rec["delta_steps_A_minus_B"] = _safe_int(rec["steps_by_method"].get(A, 0), 0) - _safe_int(rec["steps_by_method"].get(B, 0), 0)
        tasks_out.append(rec)

    # Stable ordering: show pair-wins first
    def _rank(r: Dict[str, Any]) -> Tuple[int, int]:
        lab = r.get("winner_vs_pair")
        if lab == f"{A}_only":
            return (0, -_safe_int(r.get("delta_tokens_A_minus_B"), 0))
        if lab == f"{B}_only":
            return (1, _safe_int(r.get("delta_tokens_A_minus_B"), 0))
        if lab == "both":
            return (2, 0)
        return (3, 0)

    tasks_out.sort(key=_rank)
    return TaskwiseSummary(tasks=tasks_out, counts=counts, methods=methods, pair=pair)


def write_taskwise_artifacts(summary: TaskwiseSummary, out_dir: Path, prefix: str = "taskwise") -> Dict[str, str]:
    """Write taskwise artifacts (jsonl + txt lists + markdown)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in summary.tasks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    A, B = summary.pair
    # Write task id lists
    lists = {
        f"{A}_only": [],
        f"{B}_only": [],
        "both": [],
        "neither": [],
    }
    for r in summary.tasks:
        lab = r.get("winner_vs_pair")
        if lab in lists:
            lists[lab].append(r.get("task_id"))

    list_paths: Dict[str, str] = {}
    for lab, ids in lists.items():
        p = out_dir / f"{prefix}_{lab}.txt"
        p.write_text("\n".join([x for x in ids if x]) + "\n", encoding="utf-8")
        list_paths[lab] = str(p)

    # Unique winners across *all* methods
    unique_by_method: Dict[str, List[str]] = {m: [] for m in summary.methods}
    for r in summary.tasks:
        uw = r.get("unique_winner")
        if uw in unique_by_method:
            unique_by_method[uw].append(r.get("task_id"))
    for m, ids in unique_by_method.items():
        p = out_dir / f"{prefix}_unique_{m}.txt"
        p.write_text("\n".join([x for x in ids if x]) + "\n", encoding="utf-8")
        list_paths[f"unique_{m}"] = str(p)

    # Markdown report (short)
    md = []
    md.append(f"# Taskwise Report ({A} vs {B})\n")
    md.append("## Counts\n")
    for k in [f"{A}_only", f"{B}_only", "both", "neither", "tie", "tasks"]:
        md.append(f"- **{k}**: {summary.counts.get(k, 0)}")
    md.append("\n## Top examples\n")
    md.append(f"### {A}_only (first 20)\n")
    md.append("\n".join([f"- {tid}" for tid in lists[f"{A}_only"][:20]] or ["- (none)"]))
    md.append(f"\n\n### {B}_only (first 20)\n")
    md.append("\n".join([f"- {tid}" for tid in lists[f"{B}_only"][:20]] or ["- (none)"]))
    md.append("\n\n### both (first 20)\n")
    md.append("\n".join([f"- {tid}" for tid in lists["both"][:20]] or ["- (none)"]))
    md.append("\n\n### neither (first 20)\n")
    md.append("\n".join([f"- {tid}" for tid in lists["neither"][:20]] or ["- (none)"]))

    md.append("\n\n## Unique winners across all methods\n")
    for m in summary.methods:
        md.append(f"- **{m}** unique wins: {len(unique_by_method.get(m, []))}")
    md_path = out_dir / f"{prefix}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "taskwise_jsonl": str(jsonl_path),
        "taskwise_md": str(md_path),
        **{f"task_ids_{k}": v for k, v in list_paths.items()},
    }
