import argparse
from pathlib import Path

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm

def main():
    ap = argparse.ArgumentParser(description="Unified benchmark runner (deterministic or LLM).")
    ap.add_argument("--benchmark", type=str, default="synthetic_browsecomp", choices=list(BENCHMARKS.keys()))
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--prepare", action="store_true", help="Generate/download benchmark data into data_dir.")

    ap.add_argument("--runner", type=str, default="deterministic", choices=["deterministic", "llm"])
    ap.add_argument("--methods", type=str, default="GoC", help="Comma-separated list or ALL. e.g., GoC,FullHistory or ALL")
    ap.add_argument("--task_limit", type=int, default=None)

    # common budgets
    ap.add_argument("--budget_active", type=int, default=1200)
    ap.add_argument("--budget_unfold", type=int, default=650)
    ap.add_argument("--unfold_k", type=int, default=8)

    # deterministic knobs
    ap.add_argument("--summary_keep_fields", type=int, default=1)
    ap.add_argument("--linear_summary_every", type=int, default=8)
    ap.add_argument("--agentfold_fold_chunk", type=int, default=10)

    # prepare knobs for synthetic
    ap.add_argument("--n_entities", type=int, default=80)
    ap.add_argument("--n_tasks", type=int, default=50)
    ap.add_argument("--distractors_per_entity", type=int, default=2)
    ap.add_argument("--noise_docs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)

    # llm knobs
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--max_steps", type=int, default=35)
    ap.add_argument("--max_json_retries", type=int, default=2)

    # retriever knobs
    ap.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "faiss"])
    ap.add_argument("--faiss_dim", type=int, default=384)

    # debugging
    ap.add_argument("--verbose_steps", action="store_true", help="Print per-step progress to stdout (LLM runner only).")
    ap.add_argument("--log_dir", type=str, default=None, help="If set, write per-task LLM prompt/output traces as JSONL files into this directory.")
    ap.add_argument("--prompt_context_chars", type=int, default=0, help="Optional char-level truncation for ACTIVE_CONTEXT in the LLM prompt (0 = no extra truncation).")
    ap.add_argument("--log_context_chars", type=int, default=2500, help="Char-level tail truncation for traces/logs (does not affect prompt when prompt_context_chars==0).")

    args = ap.parse_args()

    bench = get_benchmark(args.benchmark)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        meta = bench.prepare(
            data_dir=str(data_dir),
            n_entities=args.n_entities,
            n_tasks=args.n_tasks,
            distractors_per_entity=args.distractors_per_entity,
            noise_docs=args.noise_docs,
            seed=args.seed,
        )
        print("Prepared:", meta)

    methods = [m.strip() for m in args.methods.split(",")] if args.methods.strip().upper() != "ALL" else ["ALL"]

    if args.runner == "deterministic":
        res = run_deterministic(
            benchmark=bench,
            data_dir=str(data_dir),
            methods=methods,
            out_results_path=str(Path("results.jsonl")),
            out_report_path=str(Path("report.md")),
            budget_active=args.budget_active,
            budget_unfold=args.budget_unfold,
            unfold_k=args.unfold_k,
            summary_keep_fields=args.summary_keep_fields,
            linear_summary_every=args.linear_summary_every,
            agentfold_fold_chunk=args.agentfold_fold_chunk,
            task_limit=args.task_limit,
            retriever_kind=args.retriever,
            faiss_dim=args.faiss_dim,
        )
    else:
        res = run_llm(
            benchmark=bench,
            data_dir=str(data_dir),
            methods=methods,
            out_results_path=str(Path("llm_results.jsonl")),
            out_report_path=str(Path("llm_report.md")),
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
            prompt_context_chars=args.prompt_context_chars,
            log_context_chars=args.log_context_chars,
        )

    print("Done:", res)

if __name__ == "__main__":
    main()
