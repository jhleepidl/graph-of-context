#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def _iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


def _load_results(artifact_dir: Path):
    for p in artifact_dir.rglob("llm_results.jsonl"):
        return list(_iter_jsonl(p))
    return []


def _trace_map(artifact_dir: Path):
    trace_map = {}
    for p in artifact_dir.rglob("traces/*.jsonl"):
        # Default: trace_<ts>_<method>_<task_id>.jsonl
        parts = p.stem.split("_")
        method = None
        task_id = None
        if len(parts) >= 5 and parts[0] == "trace":
            method = parts[3]
            task_id = "_".join(parts[4:])
        if method and task_id:
            trace_map[(method, task_id)] = p
    return trace_map


def _infer_max_steps(row):
    explanation = str(row.get("explanation") or "").lower()
    return "max_steps" in explanation


def _scan_trace(path: Path):
    return_blocked = Counter()
    schema_errors = Counter()
    stage = {"subtask": False, "merge": False, "final": False}
    tool_calls = 0

    for ev in _iter_jsonl(path):
        if ev.get("type") == "return_blocked":
            reason = ev.get("reason") or "unknown"
            return_blocked[str(reason)] += 1
            se = ev.get("schema_error_type")
            if isinstance(se, str) and se:
                schema_errors[se] += 1
        if ev.get("tool"):
            tool_calls += 1

        # stage markers appear in prompt/messages content
        txts = []
        if isinstance(ev.get("prompt"), str):
            txts.append(ev["prompt"])
        if isinstance(ev.get("user_prompt"), str):
            txts.append(ev["user_prompt"])
        if isinstance(ev.get("content"), str):
            txts.append(ev["content"])
        if isinstance(ev.get("messages"), list):
            for m in ev["messages"]:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    txts.append(m["content"])
        for t in txts:
            if "[SUBTASK" in t:
                stage["subtask"] = True
            if "[MERGE" in t:
                stage["merge"] = True
            if "[FINAL" in t:
                stage["final"] = True

    return return_blocked, schema_errors, stage, tool_calls


def main():
    ap = argparse.ArgumentParser(description="Report max_steps_exit outliers for a run-group.")
    ap.add_argument("--artifact_dir", required=True, help="runs/<exp_id>")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists():
        raise SystemExit(f"artifact_dir not found: {artifact_dir}")

    results = _load_results(artifact_dir)
    trace_map = _trace_map(artifact_dir)

    outliers = []
    stage_counts = Counter()
    type_counts = Counter()

    for row in results:
        if not _infer_max_steps(row):
            continue
        method = str(row.get("method") or "UNKNOWN")
        task_id = str(row.get("task_id") or "")
        trace = trace_map.get((method, task_id))
        rb_counts = {}
        se_counts = {}
        stage = {"subtask": False, "merge": False, "final": False}
        tool_calls = 0
        if trace and trace.exists():
            rb, se, stage, tool_calls = _scan_trace(trace)
            rb_counts = dict(rb)
            se_counts = dict(se)
            if stage["subtask"]:
                stage_counts["subtask"] += 1
            if stage["merge"]:
                stage_counts["merge"] += 1
            if stage["final"]:
                stage_counts["final"] += 1
        type_counts["max_steps_exit"] += 1

        outliers.append({
            "task_id": task_id,
            "method": method,
            "trace": str(trace) if trace else None,
            "return_blocked_counts": rb_counts,
            "schema_error_type_counts": se_counts,
            "max_return_blocked": max(rb_counts.values()) if rb_counts else 0,
            "tool_calls": tool_calls,
            "stage_reached": stage,
        })

    summary = {
        "n_outliers": len(outliers),
        "outlier_types": dict(type_counts),
        "stage_reached_counts": dict(stage_counts),
    }

    out_json = {
        "artifact_dir": str(artifact_dir),
        "summary": summary,
        "outliers": outliers,
    }

    json_path = artifact_dir / "outlier_report.json"
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Outlier Report ({artifact_dir.name})",
        "",
        f"- Outliers (max_steps_exit): {len(outliers)}",
        f"- Stage reached counts: {summary['stage_reached_counts']}",
        "",
        "## Outliers",
    ]

    for o in outliers:
        md_lines.append(f"- {o['method']} | {o['task_id']} | trace: {o['trace']}")
        md_lines.append(f"  - return_blocked: {o['return_blocked_counts']}")
        md_lines.append(f"  - schema_error_type: {o['schema_error_type_counts']}")
        md_lines.append(f"  - max_return_blocked: {o['max_return_blocked']} | tool_calls: {o['tool_calls']} | stage: {o['stage_reached']}")

    md_path = artifact_dir / "outlier_report.md"
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
