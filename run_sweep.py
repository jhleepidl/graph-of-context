import argparse
import sys
import json
import itertools
import hashlib
import traceback
import subprocess
import zipfile
import re
from math import floor, ceil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm
from src.analysis.taskwise import build_taskwise, load_jsonl, write_taskwise_artifacts


def _load_jsonl_safe(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _iter_jsonl(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
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


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    k = (len(vs) - 1) * (p / 100.0)
    f = int(floor(k))
    c = int(ceil(k))
    if f == c:
        return float(vs[f])
    d = k - f
    return float(vs[f] + (vs[c] - vs[f]) * d)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _collect_trace_files(out_dir: Path) -> List[Path]:
    return sorted(out_dir.rglob("traces/*.jsonl"))


def _scan_trace_files(trace_files: List[Path]) -> Tuple[
    Dict[Tuple[str, str], Path],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], bool],
    int,
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], int],
]:
    trace_map: Dict[Tuple[str, str], Path] = {}
    return_blocked_map: Dict[Tuple[str, str], int] = {}
    finish_map: Dict[Tuple[str, str], bool] = {}
    auto_inject_total = 0
    autofix_map: Dict[Tuple[str, str], int] = {}
    schema_error_map: Dict[Tuple[str, str], Dict[str, int]] = {}
    key_inject_map: Dict[Tuple[str, str], int] = {}
    key_trunc_map: Dict[Tuple[str, str], int] = {}

    for p in trace_files:
        method = None
        task_id = None
        return_blocked = 0
        autofix = 0
        schema_errors: Dict[str, int] = {}
        key_inject = 0
        key_trunc = 0
        saw_finish = False
        for ev in _iter_jsonl(p):
            if method is None:
                method = ev.get("method") or method
            if task_id is None:
                task_id = ev.get("task_id") or task_id
            if ev.get("type") == "return_blocked":
                return_blocked += 1
                se = ev.get("schema_error_type")
                if isinstance(se, str) and se:
                    schema_errors[se] = schema_errors.get(se, 0) + 1
            if ev.get("type") == "schema_autofix":
                autofix += 1
            if ev.get("tool") == "finish":
                saw_finish = True
            if ev.get("type") == "user_turn_injected" and ev.get("reason") == "auto":
                auto_inject_total += 1
            if ev.get("type") == "candidate_commits_injected":
                key_inject += 1
                if bool(ev.get("key_truncated")):
                    key_trunc += 1

        if method is None or task_id is None:
            stem = p.stem
            parts = stem.split("_")
            if len(parts) >= 5 and parts[0] == "trace":
                method = method or parts[3]
                task_id = task_id or "_".join(parts[4:])

        if method and task_id:
            key = (str(method), str(task_id))
            trace_map[key] = p
            return_blocked_map[key] = int(return_blocked)
            finish_map[key] = bool(saw_finish)
            if autofix:
                autofix_map[key] = int(autofix)
            if schema_errors:
                schema_error_map[key] = dict(schema_errors)
            if key_inject:
                key_inject_map[key] = int(key_inject)
            if key_trunc:
                key_trunc_map[key] = int(key_trunc)

    return trace_map, return_blocked_map, finish_map, auto_inject_total, autofix_map, schema_error_map, key_inject_map, key_trunc_map


def _collect_result_rows(out_dir: Path, runner: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if runner == "llm":
        for p in out_dir.rglob("llm_results.jsonl"):
            rows.extend(_load_jsonl_safe(p))
    else:
        for p in out_dir.rglob("results.jsonl"):
            rows.extend(_load_jsonl_safe(p))
    return rows


def _infer_completion(row: Dict[str, Any]) -> bool:
    pred = row.get("pred")
    if isinstance(pred, str):
        return len(pred.strip()) > 0
    return pred is not None


def _infer_fail_reason(row: Dict[str, Any], return_blocked: int) -> str:
    if _infer_completion(row):
        return ""
    explanation = str(row.get("explanation") or "").lower()
    if "max_steps" in explanation:
        return "max_steps_exit"
    if "merge" in explanation:
        return "merge_error"
    tool_stats = row.get("tool_stats") or {}
    if int(tool_stats.get("malformed_return_args", 0) or 0) > 0:
        return "schema_drift"
    if int(tool_stats.get("malformed_branch_args", 0) or 0) > 0:
        return "schema_drift"
    if int(tool_stats.get("finish_hotpot_json_salvage_errors", 0) or 0) > 0:
        return "schema_drift"
    if int(return_blocked) >= 10:
        return "return_blocked_loop"
    return "other"


def _build_summary_min(exp_id: str, out_dir: Path, runner: str) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[Tuple[str, str], Path],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], bool],
    int,
]:
    rows = _collect_result_rows(out_dir, runner)
    trace_files = _collect_trace_files(out_dir)
    (
        trace_map,
        return_blocked_map,
        finish_map,
        auto_inject_total,
        autofix_map,
        schema_error_map,
        key_inject_map,
        key_trunc_map,
    ) = _scan_trace_files(trace_files)

    by_method: Dict[str, Dict[str, Any]] = {}
    total_tasks_by_method: Dict[str, int] = {}

    for row in rows:
        method = str(row.get("method") or "UNKNOWN")
        task_id = str(row.get("task_id") or "")
        key = (method, task_id)

        if method not in by_method:
            by_method[method] = {
                "tokens": [],
                "steps": [],
                "open_page_calls": [],
                "return_blocked": [],
                "schema_autofix": 0,
                "schema_error_counts": {},
                "merge_key_inject": 0,
                "merge_key_trunc": 0,
                "completion": 0,
                "correct": 0,
                "total": 0,
                "fail_counts": {
                    "max_steps_exit": 0,
                    "schema_drift": 0,
                    "return_blocked_loop": 0,
                    "merge_error": 0,
                    "other": 0,
                },
            }

        stats = by_method[method]
        stats["total"] += 1
        total_tasks_by_method[method] = stats["total"]

        completion = _infer_completion(row)
        if completion:
            stats["completion"] += 1

        correct = row.get("correct")
        if isinstance(correct, bool):
            if correct:
                stats["correct"] += 1
        else:
            correct_strict = row.get("correct_strict")
            if isinstance(correct_strict, bool) and correct_strict:
                stats["correct"] += 1

        usage = row.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, (int, float)):
            stats["tokens"].append(float(total_tokens))

        steps = row.get("steps")
        if isinstance(steps, (int, float)):
            stats["steps"].append(float(steps))

        tool_stats = row.get("tool_stats") or {}
        open_page_calls = None
        for k in ("open_page_calls", "open_page_tool_calls", "open_page_tool_calls_total", "open_page_calls_total"):
            if k in tool_stats:
                open_page_calls = tool_stats.get(k)
                break
        if isinstance(open_page_calls, (int, float)):
            stats["open_page_calls"].append(float(open_page_calls))
        else:
            stats["open_page_calls"].append(0.0)

        rb = int(return_blocked_map.get(key, 0))
        stats["return_blocked"].append(float(rb))

        if key in autofix_map:
            stats["schema_autofix"] += int(autofix_map.get(key, 0) or 0)
        if key in schema_error_map:
            for k, v in schema_error_map.get(key, {}).items():
                stats["schema_error_counts"][k] = stats["schema_error_counts"].get(k, 0) + int(v or 0)
        if key in key_inject_map:
            stats["merge_key_inject"] += int(key_inject_map.get(key, 0) or 0)
        if key in key_trunc_map:
            stats["merge_key_trunc"] += int(key_trunc_map.get(key, 0) or 0)

        fail_reason = _infer_fail_reason(row, rb)
        if fail_reason:
            stats["fail_counts"][fail_reason] = stats["fail_counts"].get(fail_reason, 0) + 1

    summary_methods: Dict[str, Any] = {}
    for method, stats in by_method.items():
        total = int(stats["total"] or 0)
        completion_rate = float(stats["completion"]) / total if total else 0.0
        accuracy = float(stats["correct"]) / total if total else 0.0
        token_p50 = _percentile(stats["tokens"], 50.0)
        token_p90 = _percentile(stats["tokens"], 90.0)
        avg_steps = _safe_mean(stats["steps"])
        avg_open_page_calls = _safe_mean(stats["open_page_calls"])
        avg_return_blocked = _safe_mean(stats["return_blocked"])
        max_return_blocked = max(stats["return_blocked"]) if stats["return_blocked"] else 0.0

        summary_methods[method] = {
            "completion_rate": completion_rate,
            "accuracy": accuracy,
            "token_p50": token_p50,
            "token_p90": token_p90,
            "avg_steps": avg_steps,
            "avg_open_page_tool_calls": avg_open_page_calls,
            "avg_return_blocked": avg_return_blocked,
            "max_return_blocked": max_return_blocked,
            "schema_autofix_commit_mismatch_count": int(stats["schema_autofix"]),
            "schema_error_type_counts": stats["schema_error_counts"],
            "merge_key_injection_count": int(stats["merge_key_inject"]),
            "merge_key_truncation_count": int(stats["merge_key_trunc"]),
            "fail_counts": stats["fail_counts"],
        }

    n_tasks = 0
    if total_tasks_by_method:
        n_tasks = max(total_tasks_by_method.values())

    summary_min = {
        "exp_id": exp_id,
        "n_tasks": n_tasks,
        "methods": summary_methods,
    }

    return summary_min, summary_methods, trace_map, return_blocked_map, finish_map, auto_inject_total


def _collect_grep_lines(trace_files: List[Path], pattern: str, max_lines: int, regex: bool = False) -> List[str]:
    lines: List[str] = []
    rx = re.compile(pattern) if regex else None
    for p in trace_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    hit = False
                    if rx is not None:
                        if rx.search(line):
                            hit = True
                    else:
                        if pattern in line:
                            hit = True
                    if hit:
                        lines.append(f"{p}:{idx}:{line.strip()}")
                        if len(lines) >= max_lines:
                            return lines
        except Exception:
            continue
    return lines


def _write_evidence(out_dir: Path, trace_files: List[Path], evidence_path: Path, sample_notes: Optional[List[str]] = None) -> None:
    def _section(title: str, lines: List[str]) -> List[str]:
        out = [f"## {title}"]
        if lines:
            out.extend(lines)
        else:
            out.append("(no matches)")
        out.append("")
        return out

    out_lines: List[str] = []
    out_lines.extend(_section("Stage marker progression: [SUBTASK]", _collect_grep_lines(trace_files, "[SUBTASK", 50)))
    out_lines.extend(_section("Stage marker progression: [MERGE]", _collect_grep_lines(trace_files, "[MERGE", 50)))
    out_lines.extend(_section("Stage marker progression: [FINAL]", _collect_grep_lines(trace_files, "[FINAL", 50)))

    out_lines.extend(_section("Turn injection: user_turn_injected", _collect_grep_lines(trace_files, "\"type\": \"user_turn_injected\"", 50)))
    out_lines.extend(_section("Turn injection: reason=auto (should be EMPTY for commit-flow)", _collect_grep_lines(trace_files, "\"reason\": \"auto\"", 20)))

    out_lines.extend(_section("Return blocked diagnostics", _collect_grep_lines(trace_files, "\"type\": \"return_blocked\"", 80)))

    out_lines.extend(_section("GoC fold events (if present)", _collect_grep_lines(trace_files, r"\"type\": \"goc_.*fold", 50, regex=True)))
    out_lines.extend(_section("GoC unfold events (if present)", _collect_grep_lines(trace_files, r"\"type\": \"goc_.*unfold", 50, regex=True)))

    if sample_notes:
        out_lines.append("## Trace sample selection notes")
        out_lines.extend(sample_notes)
        out_lines.append("")

    evidence_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def _select_trace_samples(
    rows: List[Dict[str, Any]],
    trace_map: Dict[Tuple[str, str], Path],
    return_blocked_map: Dict[Tuple[str, str], int],
    finish_map: Dict[Tuple[str, str], bool],
) -> Tuple[Dict[str, Optional[Path]], List[str]]:
    sample_notes: List[str] = []
    used: set = set()

    def _key(row: Dict[str, Any]) -> Tuple[str, str]:
        return (str(row.get("method") or "UNKNOWN"), str(row.get("task_id") or ""))

    def _pick(predicate, allow_used: bool = False) -> Optional[Path]:
        for row in rows:
            k = _key(row)
            if (not allow_used) and (k in used):
                continue
            if not predicate(row, k):
                continue
            p = trace_map.get(k)
            if p is not None:
                if not allow_used:
                    used.add(k)
                return p
        return None

    success = _pick(lambda row, k: _infer_completion(row) and finish_map.get(k, False))
    if success is None:
        success = _pick(lambda row, k: _infer_completion(row))
        if success is not None:
            sample_notes.append("SUCCESS_1.jsonl: finish marker missing; selected completed trace.")

    def _fail_reason(row: Dict[str, Any], k: Tuple[str, str]) -> str:
        rb = int(return_blocked_map.get(k, 0))
        return _infer_fail_reason(row, rb)

    fail_maxsteps = _pick(lambda row, k: _fail_reason(row, k) == "max_steps_exit")
    fail_loop_or_schema = _pick(lambda row, k: _fail_reason(row, k) in {"return_blocked_loop", "schema_drift", "merge_error"})

    if fail_loop_or_schema is None:
        fail_loop_or_schema = _pick(lambda row, k: not _infer_completion(row))
        if fail_loop_or_schema is not None:
            sample_notes.append("FAIL_LOOP_OR_SCHEMA_1.jsonl: fallback to any incomplete trace.")

    if success is None:
        success = _pick(lambda row, k: True)
        if success is not None:
            sample_notes.append("SUCCESS_1.jsonl: no completed trace found; fallback used.")

    if fail_maxsteps is None:
        fail_maxsteps = _pick(lambda row, k: not _infer_completion(row))
        if fail_maxsteps is not None:
            sample_notes.append("FAIL_MAXSTEPS_1.jsonl: no max_steps trace found; fallback used.")

    # If any sample is still missing, allow reuse of existing traces.
    if success is None:
        success = _pick(lambda row, k: True, allow_used=True)
        if success is not None:
            sample_notes.append("SUCCESS_1.jsonl: reused trace due to limited samples.")
    if fail_maxsteps is None:
        fail_maxsteps = _pick(lambda row, k: True, allow_used=True)
        if fail_maxsteps is not None:
            sample_notes.append("FAIL_MAXSTEPS_1.jsonl: reused trace due to limited samples.")
    if fail_loop_or_schema is None:
        fail_loop_or_schema = _pick(lambda row, k: True, allow_used=True)
        if fail_loop_or_schema is not None:
            sample_notes.append("FAIL_LOOP_OR_SCHEMA_1.jsonl: reused trace due to limited samples.")

    return {
        "SUCCESS_1.jsonl": success,
        "FAIL_MAXSTEPS_1.jsonl": fail_maxsteps,
        "FAIL_LOOP_OR_SCHEMA_1.jsonl": fail_loop_or_schema,
    }, sample_notes


def _write_traces_zip(out_dir: Path, samples: Dict[str, Optional[Path]]) -> None:
    zip_path = out_dir / "traces_samples.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, src in samples.items():
            if src is None or (not src.exists()):
                continue
            zf.write(src, arcname=name)


def _run_contract_check(repo_root: Path, out_dir: Path) -> Dict[str, Any]:
    trace_glob = str(out_dir / "**" / "traces" / "*.jsonl")
    cmd = ["python", str(repo_root / "scripts" / "check_multicommit_contract.py"), "--trace_glob", trace_glob, "--strict"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if stdout:
            try:
                data = json.loads(stdout)
            except Exception:
                data = {"ok": proc.returncode == 0, "raw": stdout}
        else:
            data = {"ok": proc.returncode == 0, "raw": ""}
        if stderr:
            data["stderr"] = stderr
        data["returncode"] = proc.returncode
        return data
    except Exception as e:
        return {"ok": False, "error": str(e), "returncode": -1}


def _append_run_manifest(repo_root: Path, entry: Dict[str, Any]) -> None:
    auto_dir = repo_root / "research_ops" / "_auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = auto_dir / "run_manifest.json"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _debug_packet(
    out_dir: Path,
    runner: str,
    cfg_path: Path,
    command: str,
    status: str,
    repo_root: Path,
) -> None:
    exp_id = out_dir.name
    summary_min, summary_methods, trace_map, return_blocked_map, finish_map, auto_inject_total = _build_summary_min(exp_id, out_dir, runner)

    summary_path = out_dir / "summary_min.json"
    summary_path.write_text(json.dumps(summary_min, ensure_ascii=False, indent=2), encoding="utf-8")

    trace_files = _collect_trace_files(out_dir)
    rows = _collect_result_rows(out_dir, runner)
    samples, sample_notes = _select_trace_samples(rows, trace_map, return_blocked_map, finish_map)
    _write_traces_zip(out_dir, samples)

    evidence_path = out_dir / "evidence.txt"
    _write_evidence(out_dir, trace_files, evidence_path, sample_notes=sample_notes)

    contract_data = _run_contract_check(repo_root, out_dir)
    contract_path = out_dir / "contract_check.json"
    contract_path.write_text(json.dumps(contract_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Determine triage label (prioritize contract violations, then dominant fail reason)
    triage = "FAIL: other"
    all_fail_counts: Dict[str, int] = {}
    for m in summary_methods.values():
        for k, v in (m.get("fail_counts") or {}).items():
            all_fail_counts[k] = all_fail_counts.get(k, 0) + int(v or 0)

    if contract_data.get("ok") is False:
        viols = contract_data.get("violations") or []
        if any(v.get("rule") == "no_auto_inject" for v in viols):
            triage = "FAIL: unexpected_auto_inject"
    elif any(v > 0 for v in all_fail_counts.values()):
        dom = max(all_fail_counts.items(), key=lambda kv: kv[1])[0]
        if dom == "max_steps_exit":
            triage = "FAIL: max_steps_exit"
        elif dom == "return_blocked_loop":
            triage = "FAIL: return_blocked_loop"
        elif dom == "schema_drift":
            triage = "FAIL: schema_drift"
        elif dom == "merge_error":
            triage = "FAIL: merge_error"
        else:
            triage = "FAIL: other"
    elif status == "success":
        triage = "SUCCESS"

    note_suffix = ""
    if auto_inject_total > 0:
        note_suffix = f"auto_inject={auto_inject_total}"
    elif contract_data.get("ok") is False:
        note_suffix = "contract_check_failed"

    note = f"{triage}"
    if note_suffix:
        note = f"{note} - {note_suffix}"

    # Git SHA (best effort)
    git_sha = "UNKNOWN"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        git_sha = "UNKNOWN"

    entry = {
        "exp_id": exp_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "preset_path": str(cfg_path),
        "command": command,
        "artifact_path": str(out_dir),
        "status": status,
        "notes": note,
    }
    _append_run_manifest(repo_root, entry)


def _infer_taskwise(run_dir: Path) -> (Optional[Dict[str, Any]], Dict[str, str]):
    """Best-effort: infer taskwise counts + artifact paths from existing files."""
    td = run_dir / "taskwise"
    jpath = td / "taskwise.jsonl"
    artifacts: Dict[str, str] = {}
    if not jpath.exists():
        return None, artifacts

    # Artifacts paths (if they exist)
    def _maybe(k: str, p: Path):
        if p.exists():
            artifacts[k] = str(p)

    _maybe("taskwise_jsonl", jpath)
    _maybe("taskwise_md", td / "taskwise.md")
    for p in td.glob("taskwise_*.txt"):
        artifacts[p.stem] = str(p)

    rows = _load_jsonl_safe(jpath)
    if not rows:
        return None, artifacts

    counts = {"GoC_only": 0, "FullHistory_only": 0, "both": 0, "neither": 0, "tie": 0, "tasks": len(rows)}
    for r in rows:
        key = str(r.get("winner_vs_pair") or "")
        if key in counts:
            counts[key] += 1
    return counts, artifacts


def _sync_master_from_disk(out_dir: Path, master_path: Path) -> int:
    """Rewrite master JSONL based on on-disk run folders.

    This prevents confusing situations where run folders exist (e.g., multiple budgets) but the master JSONL
    only contains runs from the latest invocation.
    """
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir() and (p / "run_config.json").exists()]
    if not run_dirs:
        return 0

    records: List[Dict[str, Any]] = []
    for rd in run_dirs:
        try:
            cfg = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
        except Exception:
            continue

        run_id = str(cfg.get("run_id") or rd.name)
        runner = str(cfg.get("runner") or "llm")
        results_path = rd / ("llm_results.jsonl" if runner == "llm" else "results.jsonl")
        summary = _summarize_jsonl(results_path, runner=runner) if results_path.exists() else []

        done = (rd / "DONE").exists()
        taskwise_counts, taskwise_artifacts = _infer_taskwise(rd)

        params = cfg.get("params") or {}
        # Stable sort key: name -> budget_active -> run_id
        budget = params.get("budget_active")

        records.append({
            "run_id": run_id,
            "name": cfg.get("name"),
            "status": "ok" if done else "partial",
            "error": None,
            "benchmark": cfg.get("benchmark"),
            "runner": cfg.get("runner"),
            "methods": cfg.get("methods"),
            "params": params,
            "bench_kwargs": cfg.get("bench_kwargs"),
            "config_path": str(cfg.get("config_path") or ""),
            "artifacts": {
                "out_results": str(results_path) if results_path.exists() else None,
                "out_report": str((rd / ("llm_report.md" if runner == "llm" else "report.md"))) if (rd / ("llm_report.md" if runner == "llm" else "report.md")).exists() else None,
            },
            "summary_by_method": summary,
            "taskwise_counts": taskwise_counts,
            "taskwise_artifacts": taskwise_artifacts,
            "session_stamp": cfg.get("session_stamp"),
            "run_index": cfg.get("run_index"),
            "_sort": (str(cfg.get("name") or ""), int(budget) if budget is not None else 0, str(run_id)),
        })

    records.sort(key=lambda r: r.get("_sort"))
    for r in records:
        r.pop("_sort", None)

    master_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = master_path.with_suffix(master_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(master_path)
    return len(records)

def _load_json(path: str) -> Dict[str, Any]:
    return json.load(open(path, "r", encoding="utf-8"))

def _product(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge (b overrides a)."""
    out = dict(a or {})
    out.update(b or {})
    return out

def _summarize_jsonl(results_path: Path, runner: str) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def _avg(xs): return sum(xs)/max(1,len(xs))
    out = []
    for method, rs in by_method.items():
        n = len(rs)
        acc = sum(1 for r in rs if r.get("correct")) / max(1, n)
        # Strict accuracy is optional (only present when benchmark supplies it)
        strict_vals = [r.get("correct_strict") for r in rs if r.get("correct_strict") is not None]
        acc_strict = (sum(1 for v in strict_vals if v) / max(1, len(strict_vals))) if strict_vals else None
        cov = _avg([float(r.get("docid_cov", 0.0)) for r in rs])

        if runner == "llm":
            avg_tok = _avg([float((r.get("usage") or {}).get("total_tokens") or 0) for r in rs])
            avg_steps = _avg([float(r.get("steps") or 0) for r in rs])
            avg_tools = _avg([float((r.get("tool_stats") or {}).get("tool_calls_total") or 0) for r in rs])
            avg_search = _avg([float((r.get("tool_stats") or {}).get("search_calls") or 0) for r in rs])
            avg_open = _avg([float((r.get("tool_stats") or {}).get("open_page_calls") or 0) for r in rs])
            json_fail = sum(int((r.get("tool_stats") or {}).get("json_parse_failures") or 0) for r in rs)
            json_rec = sum(int((r.get("tool_stats") or {}).get("json_recoveries") or 0) for r in rs)
            avg_elapsed = _avg([float(r.get("elapsed_sec") or 0.0) for r in rs])
            out.append({
                "method": method,
                "n": n,
                "accuracy": acc,
                "accuracy_strict": acc_strict,
                "avg_total_tokens": avg_tok,
                "avg_steps": avg_steps,
                "avg_tool_calls": avg_tools,
                "avg_search": avg_search,
                "avg_open": avg_open,
                "json_fail": json_fail,
                "json_recover": json_rec,
                "avg_elapsed_sec": avg_elapsed,
                "avg_docid_coverage": cov,
            })
        else:
            metrics = [r.get("metrics") or {} for r in rs]
            avg_tok = _avg([(m.get("llm_in_tokens", 0) + m.get("llm_out_tokens", 0)) for m in metrics])
            avg_peak = _avg([m.get("peak_active_tokens", 0) for m in metrics])
            avg_tools = _avg([m.get("tool_calls", 0) for m in metrics])
            out.append({
                "method": method,
                "n": n,
                "accuracy": acc,
                "avg_total_tokens": avg_tok,
                "avg_peak_active_tokens": avg_peak,
                "avg_tool_calls": avg_tools,
                "avg_docid_coverage": cov,
            })
    return out

def main():
    ap = argparse.ArgumentParser(description="Run parameter sweeps and aggregate results into one file.")
    ap.add_argument("--config", type=str, default=None, help="Path to sweep JSON config.")
    ap.add_argument("--preset", type=str, default=None, help="Name of a built-in preset under configs/*.json.")
    ap.add_argument("--list_presets", action="store_true", help="List available presets and exit.")
    ap.add_argument("--out_dir", type=str, default="sweeps", help="Directory to store per-run artifacts + master summary.")
    ap.add_argument("--dry_run", action="store_true", help="Print planned runs (methods + grid combos) and exit.")
    ap.add_argument("--resume", action="store_true", help="Resume: reuse existing run directories and continue incomplete runs.")
    ap.add_argument("--fresh", action="store_true", help="Ignore any existing runs in out_dir and start a new sweep (but does not delete files).")
    ap.add_argument("--fail_fast", action="store_true", help="Stop the sweep on the first error (default: continue and record the error).")
    ap.add_argument("--continue_on_error", action="store_true", help="(Deprecated) kept for backward compatibility. The default behavior already continues on error unless --fail_fast is set.")
    ap.add_argument("--master", type=str, default=None, help="Optional master summary path. If omitted, uses a stable sweep_master_<bench>_<runner>.jsonl in out_dir.")
    ap.add_argument(
        "--no_sync_master",
        action="store_true",
        help=(
            "Do not resync/rebuild the master summary JSONL at the end. "
            "By default, the sweep will scan run folders and rewrite the master file so it includes *all* runs "
            "(useful if runs were copied/moved or the master became out-of-sync)."
        ),
    )
    ap.add_argument("--taskwise_pair", type=str, default="GoC,FullHistory", help="Pair to compare in per-run taskwise reports, e.g. GoC,FullHistory")
    ap.add_argument("--no_taskwise", action="store_true", help="Disable per-run taskwise artifacts.")
    orig_argv = sys.argv[1:]
    args = ap.parse_args()

    preset_dir = Path(__file__).parent / "configs"
    if args.list_presets:
        if not preset_dir.exists():
            print("No configs/ directory found.")
            return
        presets = sorted([p.stem for p in preset_dir.glob("*.json")])
        for p in presets:
            print(p)
        return

    cfg_path: Optional[Path] = None
    if args.preset:
        cfg_path = preset_dir / f"{args.preset}.json"
        if not cfg_path.exists():
            raise SystemExit(f"Unknown preset '{args.preset}'. Expected: {cfg_path}")
    elif args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
    else:
        raise SystemExit("Provide --config <path.json> or --preset <name>. Use --list_presets to see built-ins.")

    cfg = _load_json(str(cfg_path))
    bench_name = cfg.get("benchmark", "synthetic_browsecomp")
    if bench_name not in BENCHMARKS:
        raise SystemExit(f"Unknown benchmark {bench_name}. Available: {list(BENCHMARKS.keys())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = get_benchmark(bench_name)
    data_dir = cfg.get("data_dir", "data")
    runner = cfg.get("runner", "llm")
    methods = cfg.get("methods", ["GoC"])
    # Normalize methods and make behavior explicit.
    def _norm_method(m: str) -> str:
        return (m or "").strip()

    if isinstance(methods, list):
        methods = [_norm_method(m) for m in methods if _norm_method(m)]
        # If user accidentally mixes ALL with other methods, raise to avoid surprises.
        if any(m.upper() == "ALL" for m in methods) and len(methods) > 1:
            raise SystemExit(f"Config error: methods contains ALL mixed with others: {methods}. Use only ['ALL'] or an explicit list.")
    # If methods is string, keep existing parsing below (comma-separated or ALL).
    print("[SWEEP] Loaded methods from config:", methods)


    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(",")] if methods.strip().upper() != "ALL" else ["ALL"]

    base = cfg.get("base", {})
    grid = cfg.get("grid", {})
    runs_cfg = cfg.get("runs", None)
    # Support two sweep styles:
    #  1) grid: cartesian product of scalar params
    #  2) runs: a list of named variants (typically bench_kwargs overrides)
    # If both are provided, we run every (run_variant x grid_combo).
    if runs_cfg is None:
        run_variants = [{"name": None, "params_over": {}, "bench_over": {}}]
    elif isinstance(runs_cfg, list):
        run_variants = []
        for i, r in enumerate(runs_cfg):
            if not isinstance(r, dict):
                raise SystemExit(f"Config error: runs[{i}] must be an object, got {type(r)}")
            name = (r.get("name") or f"run{i+1}").strip()

            # Parameter overrides can be given either under "params" or at top-level.
            params_over: Dict[str, Any] = {}
            if isinstance(r.get("params"), dict):
                params_over.update(r["params"])
            for k, v in r.items():
                if k in ("name", "bench_kwargs", "params"):
                    continue
                params_over[k] = v

            bench_over = r.get("bench_kwargs") if isinstance(r.get("bench_kwargs"), dict) else {}
            run_variants.append({"name": name, "params_over": params_over, "bench_over": bench_over})
        if not run_variants:
            run_variants = [{"name": None, "params_over": {}, "bench_over": {}}]
    else:
        raise SystemExit(f"Config error: runs must be a list, got {type(runs_cfg)}")

    grid_combos = list(_product(grid)) if grid else [dict()]
    total_runs = len(grid_combos) * len(run_variants)
    print("[SWEEP] Grid keys:", list(grid.keys()))
    if runs_cfg is not None:
        print("[SWEEP] Run variants:", [rv["name"] for rv in run_variants])
    print("[SWEEP] Total planned runs:", total_runs)



    # Optional prepare
    if cfg.get("prepare", False):
        prep_kwargs = cfg.get("prepare_kwargs", {})
        bench.prepare(data_dir=data_dir, **prep_kwargs)

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Stable master summary path (for resume across multiple invocations)
    master_path = Path(args.master) if args.master else (out_dir / f"sweep_master_{bench_name}_{runner}.jsonl")

    # Auto-resume if we detect existing artifacts, unless --fresh is specified.
    if args.fresh:
        args.resume = False
    else:
        if (args.resume or master_path.exists() or any((p / "run_config.json").exists() for p in out_dir.iterdir() if p.is_dir())):
            if not args.resume:
                print("[SWEEP] Auto-resume enabled (existing runs detected). Use --fresh to ignore existing artifacts.")
            args.resume = True

    def _run_key(params: Dict[str, Any], bench_kwargs: Dict[str, Any]) -> str:
        payload = {
            "benchmark": bench_name,
            "runner": runner,
            "methods": methods,
            "params": params,
            "bench_kwargs": bench_kwargs,
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    if args.dry_run:
        # Print planned parameter combinations (without running).
        i = 0
        for rv in run_variants:
            for combo in grid_combos:
                i += 1
                params = dict(base)
                params.update(combo)
                params.update(rv.get("params_over") or {})

                base_bk = (base.get("bench_kwargs") or {})
                bk = _merge_dict(base_bk, params.get("bench_kwargs") or {})
                bk = _merge_dict(bk, rv.get("bench_over") or {})
                params["bench_kwargs"] = bk

                name = rv.get("name")
                tag = f" name={name}" if name else ""
                print(f"[DRY_RUN] {i:03d}/{total_runs}{tag} params:", params)
        print("[DRY_RUN] Exiting without running any experiments.")
        return
    run_idx = 0
    ok_runs = 0
    error_runs = 0
    for rv in run_variants:
        for combo in grid_combos:
            run_idx += 1

            # Merge parameters
            params = dict(base)
            params.update(combo)
            params.update(rv.get("params_over") or {})

            # Merge bench_kwargs (base -> params.bench_kwargs -> run_variant.bench_over)
            base_bk = (base.get("bench_kwargs") or {})
            bench_kwargs = _merge_dict(base_bk, params.get("bench_kwargs") or {})
            bench_kwargs = _merge_dict(bench_kwargs, rv.get("bench_over") or {})
            params["bench_kwargs"] = bench_kwargs

            run_name = rv.get("name")
            run_id = _run_key(params, bench_kwargs)
            run_dir = out_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Persist run config early so we can resume even if the process crashes mid-run.
            run_cfg_path = run_dir / "run_config.json"
            if (not run_cfg_path.exists()) or (not args.resume):
                run_cfg_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "name": run_name,
                            "params": params,
                            "bench_kwargs": bench_kwargs,
                            "methods": methods,
                            "benchmark": bench_name,
                            "runner": runner,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            done_flag = run_dir / "DONE"
            if args.resume and done_flag.exists():
                print(f"[SWEEP] Skipping completed run_id={run_id} ({run_idx}/{total_runs})")
                continue

            status = "ok"
            err_info = None
            try:
                if runner == "llm":
                    res = run_llm(
                        benchmark=bench,
                        data_dir=data_dir,
                        methods=methods,
                        out_results_path=str(run_dir / "llm_results.jsonl"),
                        out_report_path=str(run_dir / "llm_report.md"),
                        bench_kwargs=bench_kwargs,
                        model=params.get("model", "gpt-4o-mini"),
                        dotenv_path=params.get("dotenv", ".env"),
                        max_steps=int(params.get("max_steps", 35)),
                        max_json_retries=int(params.get("max_json_retries", 2)),
                        budget_active=int(params.get("budget_active", 1200)),
                        budget_unfold=int(params.get("budget_unfold", 650)),
                        unfold_k=int(params.get("unfold_k", 8)),
                        linear_summary_every=int(params.get("linear_summary_every", 8)),
                        agentfold_fold_chunk=int(params.get("agentfold_fold_chunk", 10)),
                        task_limit=params.get("task_limit"),
                        retriever_kind=params.get("retriever_kind", "bm25"),
                        faiss_dim=int(params.get("faiss_dim", 384)),
                        verbose_steps=bool(params.get("verbose_steps", False)),
                        log_dir=str(run_dir / "traces") if params.get("log_traces", False) else None,
                        trace_messages=bool(params.get("trace_messages", True)),
                        trace_message_chars=int(params.get("trace_message_chars", 6000) or 0),
                        trace_output_chars=int(params.get("trace_output_chars", 4000) or 0),
                        prompt_context_chars=int(params.get("prompt_context_chars", 0) or 0),
                        log_context_chars=int(params.get("log_context_chars", 2500) or 2500),

                        # Parallelization (optional)
                        parallel_tasks=int(params.get("parallel_tasks", 1) or 1),

                        # Two-stage commit helpers (HotpotQA/FEVER-style)
                        enforce_committed_supporting_titles=str(params.get("enforce_committed_supporting_titles", "goc_only") or "goc_only"),
                        committed_supporting_titles_n=int(params.get("committed_supporting_titles_n", 2) or 2),
                        stage_aware_unfold_on_final=bool(params.get("stage_aware_unfold_on_final", True)),
                        stage_final_unfold_k=int(params.get("stage_final_unfold_k", 6) or 6),


                        # Difficulty / gating levers (optional)
                        multi_turn_auto_inject=params.get("multi_turn_auto_inject"),
                        multi_turn_min_step=int(params.get("multi_turn_min_step", 8)),
                        multi_turn_min_open_pages=int(params.get("multi_turn_min_open_pages", 3)),
                        min_steps_before_finish=int(params.get("min_steps_before_finish", 2)),
                        min_open_pages_before_finish=int(params.get("min_open_pages_before_finish", 1)),
                        require_docids_in_finish=params.get("require_docids_in_finish"),

                        # Resume
                        resume=bool(args.resume),

                        # Optional task index for per-task analysis
                        out_task_index_path=str(run_dir / "task_index.jsonl"),
                    )
                    summary = _summarize_jsonl(run_dir / "llm_results.jsonl", runner="llm")
                else:
                    res = run_deterministic(
                        benchmark=bench,
                        data_dir=data_dir,
                        methods=methods,
                        out_results_path=str(run_dir / "results.jsonl"),
                        out_report_path=str(run_dir / "report.md"),
                        bench_kwargs=bench_kwargs,
                        budget_active=int(params.get("budget_active", 1200)),
                        budget_unfold=int(params.get("budget_unfold", 650)),
                        unfold_k=int(params.get("unfold_k", 8)),
                        summary_keep_fields=int(params.get("summary_keep_fields", 1)),
                        linear_summary_every=int(params.get("linear_summary_every", 8)),
                        agentfold_fold_chunk=int(params.get("agentfold_fold_chunk", 10)),
                        task_limit=params.get("task_limit"),
                        retriever_kind=params.get("retriever_kind", "bm25"),
                        faiss_dim=int(params.get("faiss_dim", 384)),
                    )
                    summary = _summarize_jsonl(run_dir / "results.jsonl", runner="deterministic")

                # Per-run taskwise artifacts (best effort; never fail the run)
                taskwise_artifacts = {}
                taskwise_counts = None
                if (not args.no_taskwise) and runner == "llm":
                    try:
                        pair_parts = [x.strip() for x in (args.taskwise_pair or "").split(",") if x.strip()]
                        pair = (pair_parts[0], pair_parts[1]) if len(pair_parts) == 2 else ("GoC", "FullHistory")
                        rows = load_jsonl(run_dir / "llm_results.jsonl")
                        # Prefer methods from config; fall back to discovered
                        summ = build_taskwise(rows, methods=methods if isinstance(methods, list) else None, pair=pair)
                        taskwise_artifacts = write_taskwise_artifacts(summ, run_dir / "taskwise", prefix="taskwise")
                        taskwise_counts = summ.counts
                    except Exception:
                        taskwise_artifacts = {}
                        taskwise_counts = None

                # Mark run complete
                done_flag.write_text(json.dumps({"session": session_stamp, "completed": True}, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                status = "error"
                err_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                res = {"error": str(e)}
                summary = []
                taskwise_artifacts = {}
                taskwise_counts = None
                print(f"[SWEEP] ERROR run_id={run_id}: {e}")
                if args.fail_fast:
                    raise

            if status == "ok":
                ok_runs += 1
            else:
                error_runs += 1

        record = {
            "run_id": run_id,
            "name": run_name,
            "status": status,
            "error": err_info,
            "benchmark": bench_name,
            "runner": runner,
            "methods": methods,
            "params": params,
            "bench_kwargs": bench_kwargs,
            "config_path": str(cfg_path),
            "artifacts": res,
            "summary_by_method": summary,
            "taskwise_counts": taskwise_counts,
            "taskwise_artifacts": taskwise_artifacts,
            "session_stamp": session_stamp,
            "run_index": run_idx,
        }

        with open(master_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[SWEEP] run_id={run_id} wrote {run_dir}")

    # Optional: ensure the master file includes *all* runs that exist on disk.
    # This avoids confusing situations where runs were executed in separate invocations
    # (e.g., different budgets) but only the latest runs were appended.
    if not args.no_sync_master:
        try:
            n = _sync_master_from_disk(out_dir, master_path)
            print(f"[SWEEP] Synced master from disk (runs={n})")
        except Exception as e:
            print(f"[SWEEP] WARNING: failed to sync master from disk: {e}")

    print("Wrote master summary:", master_path)

    # Debug packet (run-group level)
    try:
        if ok_runs > 0 and error_runs == 0:
            group_status = "success"
        elif ok_runs > 0 and error_runs > 0:
            group_status = "partial"
        else:
            group_status = "failed"

        repo_root = Path(__file__).parent
        command = " ".join(["python", Path(__file__).name] + (orig_argv or []))
        _debug_packet(
            out_dir=out_dir,
            runner=runner,
            cfg_path=cfg_path,
            command=command,
            status=group_status,
            repo_root=repo_root,
        )
    except Exception as e:
        print(f"[SWEEP] WARNING: failed to write debug packet: {e}")

if __name__ == "__main__":
    main()
