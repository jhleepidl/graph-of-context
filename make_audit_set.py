import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def pick_best_goc_run(master_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''
    Pick the best run (record) by GoC accuracy; tie-break by lower avg_total_tokens.
    Expects each record to have `summary_by_method` and `artifacts` with out_results path.
    '''
    best = None
    best_key = None
    for rec in master_records:
        summaries = rec.get("summary_by_method") or []
        goc = next((s for s in summaries if s.get("method") == "GoC"), None)
        if not goc:
            continue
        acc = float(goc.get("accuracy", 0.0))
        tok = float(goc.get("avg_total_tokens", 1e18))
        key = (acc, -tok)  # maximize acc, minimize tok
        if best is None or key > best_key:
            best = rec
            best_key = key
    if best is None:
        raise SystemExit("Could not find any GoC summary rows in master JSONL.")
    return best

def score_task(row: Dict[str, Any]) -> Tuple[int, float, float, float, float]:
    '''
    Higher is more audit-worthy.
    Primary: incorrect first.
    Then: premature_finish_blocked, repeated_search_count, tool_calls_total, steps.
    '''
    incorrect = 1 if not row.get("correct", False) else 0
    ts = row.get("tool_stats") or {}
    blocked = float(ts.get("premature_finish_blocked", 0))
    rep = float(ts.get("repeated_search_count", 0))
    tools = float(ts.get("tool_calls_total", 0))
    steps = float(row.get("steps", 0))
    return (incorrect, blocked, rep, tools, steps)

def main():
    ap = argparse.ArgumentParser(description="Create an audit task subset (tasks.json + ids) from Stage-1 sweep results.")
    ap.add_argument("--master", type=str, required=True, help="Path to sweep master JSONL (from Stage-1).")
    ap.add_argument("--data_dir", type=str, default="data", help="Original data_dir containing corpus.json/tasks.json.")
    ap.add_argument("--out_dir", type=str, default="audit", help="Output directory to write audit ids + data_audit/ .")
    ap.add_argument("--k", type=int, default=30, help="Total audit tasks to select.")
    ap.add_argument("--min_incorrect", type=int, default=10, help="Ensure at least this many incorrect GoC tasks (if available).")
    ap.add_argument("--prefer_run_id", type=str, default=None, help="If set, pick this run_id instead of auto-picking best GoC.")
    args = ap.parse_args()

    master_path = Path(args.master)
    records = read_jsonl(master_path)

    if args.prefer_run_id:
        rec = next((r for r in records if r.get("run_id") == args.prefer_run_id), None)
        if rec is None:
            raise SystemExit(f"run_id not found: {args.prefer_run_id}")
    else:
        rec = pick_best_goc_run(records)

    run_id = rec.get("run_id")
    artifacts = rec.get("artifacts") or {}
    out_results = artifacts.get("out_results")
    if not out_results:
        raise SystemExit("Master record missing artifacts.out_results; cannot locate llm_results.jsonl.")
    results_path = Path(out_results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    # Load per-task rows and filter GoC only
    rows = read_jsonl(results_path)
    goc_rows = [r for r in rows if r.get("method") == "GoC"]
    if not goc_rows:
        raise SystemExit("No GoC rows found in selected run results.")

    ranked = sorted(goc_rows, key=score_task, reverse=True)
    incorrect = [r for r in ranked if not r.get("correct", False)]

    picked: List[Dict[str, Any]] = []
    picked_ids = set()

    # Take incorrect first
    for r in incorrect:
        if len(picked) >= min(args.k, max(args.min_incorrect, 0)):
            break
        tid = r["task_id"]
        if tid not in picked_ids:
            picked.append(r)
            picked_ids.add(tid)

    # Fill remaining from highest-scored overall
    for r in ranked:
        if len(picked) >= args.k:
            break
        tid = r["task_id"]
        if tid not in picked_ids:
            picked.append(r)
            picked_ids.add(tid)

    task_ids = [r["task_id"] for r in picked]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "audit_task_ids.json").write_text(json.dumps(task_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build data_audit/ by copying corpus and writing subset tasks.json
    src_data = Path(args.data_dir)
    corpus_src = src_data / "corpus.json"
    tasks_src = src_data / "tasks.json"
    if not corpus_src.exists() or not tasks_src.exists():
        raise SystemExit(f"Expected corpus.json and tasks.json in {src_data}. Run prepare first.")

    audit_data = out_dir / "data_audit"
    audit_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus_src, audit_data / "corpus.json")

    all_tasks = json.load(open(tasks_src, "r", encoding="utf-8"))
    subset_by_id = {t.get("id"): t for t in all_tasks if t.get("id") in set(task_ids)}
    ordered_subset = [subset_by_id[tid] for tid in task_ids if tid in subset_by_id]

    json.dump(ordered_subset, open(audit_data / "tasks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    manifest = {
        "source_master": str(master_path),
        "selected_run_id": run_id,
        "selected_results": str(results_path),
        "n_selected": len(task_ids),
        "out_data_dir": str(audit_data),
    }
    (out_dir / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[AUDIT] selected_run_id:", run_id)
    print("[AUDIT] selected_results:", results_path)
    print("[AUDIT] n_selected:", len(task_ids))
    print("[AUDIT] wrote:", out_dir / "audit_task_ids.json")
    print("[AUDIT] wrote:", audit_data / "tasks.json")

if __name__ == "__main__":
    main()
