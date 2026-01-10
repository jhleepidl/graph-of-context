import argparse
from pathlib import Path

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.llm import run_llm

def main():
    ap = argparse.ArgumentParser(description="Legacy LLM benchmark runner (kept for backward compat).")
    ap.add_argument("--benchmark", type=str, default="synthetic_browsecomp", choices=list(BENCHMARKS.keys()))
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--prepare", action="store_true")

    ap.add_argument("--methods", type=str, default="GoC", help="Comma-separated list or ALL")
    ap.add_argument("--task_limit", type=int, default=10)

    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--max_steps", type=int, default=35)
    ap.add_argument("--max_json_retries", type=int, default=2)

    ap.add_argument("--budget_active", type=int, default=1200)
    ap.add_argument("--budget_unfold", type=int, default=650)
    ap.add_argument("--unfold_k", type=int, default=8)
    ap.add_argument("--linear_summary_every", type=int, default=8)
    ap.add_argument("--agentfold_fold_chunk", type=int, default=10)

    ap.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "faiss"])
    ap.add_argument("--faiss_dim", type=int, default=384)

    ap.add_argument("--verbose_steps", action="store_true")
    ap.add_argument("--log_dir", type=str, default=None)

    ap.add_argument("--out_results", type=str, default="llm_results.jsonl")
    ap.add_argument("--out_report", type=str, default="llm_report.md")

    args = ap.parse_args()

    bench = get_benchmark(args.benchmark)
    if args.prepare:
        # Only works for benchmarks that implement prepare; synthetic does.
        bench.prepare(data_dir=args.data_dir)

    methods = [m.strip() for m in args.methods.split(",")] if args.methods.strip().upper() != "ALL" else ["ALL"]

    res = run_llm(
        benchmark=bench,
        data_dir=args.data_dir,
        methods=methods,
        out_results_path=args.out_results,
        out_report_path=args.out_report,
        model=args.model,
        dotenv_path=args.dotenv,
        max_steps=args.max_steps,
        max_json_retries=args.max_json_retries,
        budget_active=args.budget_active,
        budget_unfold=args.budget_unfold,
        unfold_k=args.unfold_k,
        linear_summary_every=args.linear_summary_every,
        agentfold_fold_chunk=args.agentfold_fold_chunk,
        task_limit=args.task_limit,
        retriever_kind=args.retriever,
        faiss_dim=args.faiss_dim,
        verbose_steps=args.verbose_steps,
        log_dir=args.log_dir,
    )
    print("Done:", res)

if __name__ == "__main__":
    main()
