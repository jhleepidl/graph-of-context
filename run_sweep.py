import argparse
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm

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
    ap.add_argument("--config", type=str, required=True, help="Path to sweep JSON config.")
    ap.add_argument("--out_dir", type=str, default="sweeps", help="Directory to store per-run artifacts + master summary.")
    ap.add_argument("--dry_run", action="store_true", help="Print planned runs (methods + grid combos) and exit.")
    args = ap.parse_args()

    cfg = _load_json(args.config)
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_path = out_dir / f"sweep_summary_{bench_name}_{runner}_{stamp}.jsonl"
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

        run_id = f"{stamp}_{run_idx:03d}"
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if runner == "llm":
            res = run_llm(
                benchmark=bench,
                data_dir=data_dir,
                methods=methods,
                out_results_path=str(run_dir / "llm_results.jsonl"),
                out_report_path=str(run_dir / "llm_report.md"),
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
            )
            summary = _summarize_jsonl(run_dir / "llm_results.jsonl", runner="llm")
        else:
            res = run_deterministic(
                benchmark=bench,
                data_dir=data_dir,
                methods=methods,
                out_results_path=str(run_dir / "results.jsonl"),
                out_report_path=str(run_dir / "report.md"),
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

        record = {
            "run_id": run_id,
            "benchmark": bench_name,
            "runner": runner,
            "methods": methods,
            "params": params,
            "artifacts": res,
            "summary_by_method": summary,
        }

        with open(master_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[SWEEP] run_id={run_id} wrote {run_dir}")

    print("Wrote master summary:", master_path)

if __name__ == "__main__":
    main()
