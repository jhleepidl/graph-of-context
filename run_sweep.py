import argparse
import json
import itertools
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm
from src.analysis.taskwise import build_taskwise, load_jsonl, write_taskwise_artifacts

def _load_json(path: str) -> Dict[str, Any]:
    return json.load(open(path, "r", encoding="utf-8"))

def _product(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

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
    ap.add_argument("--preset", type=str, default=None, help="Name of a built-in preset under sweep_configs/*.json.")
    ap.add_argument("--list_presets", action="store_true", help="List available presets and exit.")
    ap.add_argument("--out_dir", type=str, default="sweeps", help="Directory to store per-run artifacts + master summary.")
    ap.add_argument("--dry_run", action="store_true", help="Print planned runs (methods + grid combos) and exit.")
    ap.add_argument("--resume", action="store_true", help="Resume: reuse existing run directories and continue incomplete runs.")
    ap.add_argument("--fresh", action="store_true", help="Ignore any existing runs in out_dir and start a new sweep (but does not delete files).")
    ap.add_argument("--fail_fast", action="store_true", help="Stop the sweep on the first error (default: continue and record the error).")
    ap.add_argument("--continue_on_error", action="store_true", help="(Deprecated) kept for backward compatibility. The default behavior already continues on error unless --fail_fast is set.")
    ap.add_argument("--master", type=str, default=None, help="Optional master summary path. If omitted, uses a stable sweep_master_<bench>_<runner>.jsonl in out_dir.")
    ap.add_argument("--taskwise_pair", type=str, default="GoC,FullHistory", help="Pair to compare in per-run taskwise reports, e.g. GoC,FullHistory")
    ap.add_argument("--no_taskwise", action="store_true", help="Disable per-run taskwise artifacts.")
    args = ap.parse_args()

    preset_dir = Path(__file__).parent / "sweep_configs"
    if args.list_presets:
        if not preset_dir.exists():
            print("No sweep_configs/ directory found.")
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
    # Compute number of planned runs from grid
    total_runs = 1
    if grid:
        for k, vs in grid.items():
            total_runs *= max(1, len(vs))
    print("[SWEEP] Grid keys:", list(grid.keys()))
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
        combos = list(_product(grid)) if grid else [dict()]
        for i, combo in enumerate(combos, start=1):
            params = dict(base)
            params.update(combo)
            print(f"[DRY_RUN] {i:03d}/{len(combos)} params:", params)
        print("[DRY_RUN] Exiting without running any experiments.")
        return


    run_idx = 0
    for combo in _product(grid) if grid else [dict()]:
        run_idx += 1
        params = dict(base)
        params.update(combo)

        bench_kwargs = params.get("bench_kwargs") or {}

        run_id = _run_key(params, bench_kwargs)
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Persist run config early so we can resume even if the process crashes mid-run.
        run_cfg_path = run_dir / "run_config.json"
        if (not run_cfg_path.exists()) or (not args.resume):
            run_cfg_path.write_text(
                json.dumps({"run_id": run_id, "params": params, "bench_kwargs": bench_kwargs, "methods": methods, "benchmark": bench_name, "runner": runner}, ensure_ascii=False, indent=2),
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
            print(f"[SWEEP] ERROR run_id={run_id}: {e}")
            if args.fail_fast:
                raise

        record = {
            "run_id": run_id,
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

    print("Wrote master summary:", master_path)

if __name__ == "__main__":
    main()
