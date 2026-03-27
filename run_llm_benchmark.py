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
    ap.add_argument("--save_goc_internal_graph", action="store_true")
    ap.add_argument("--enable_unfold_trigger", action="store_true", help="Enable query-vs-context unfolding trigger.")
    ap.add_argument("--unfold_trigger_missing_terms_threshold", type=int, default=3)
    ap.add_argument("--unfold_trigger_min_token_len", type=int, default=4)
    ap.add_argument("--unfold_trigger_max_keywords", type=int, default=48)
    ap.add_argument("--unfold_trigger_k", type=int, default=6)
    ap.add_argument("--unfold_trigger_log_missing_limit", type=int, default=12)
    ap.add_argument(
        "--unfold_trigger_always_on_required_keys",
        action="store_true",
        default=True,
        help="Always trigger unfold when relocation_* required keys appear in the query.",
    )
    ap.add_argument("--fork_trigger_mode", type=str, default="evidence_gated", help="evidence_gated|gate_probe|always|debug_once|debug_once_no_merge|commit_only|final_only|pivot_only|pivot_and_final")
    ap.add_argument("--fork_min_step", type=int, default=4)
    ap.add_argument("--fork_every_k_steps", type=int, default=3)
    ap.add_argument("--fork_min_open_pages", type=int, default=2)
    ap.add_argument("--fork_min_search_calls", type=int, default=1)
    ap.add_argument("--fork_min_active_tokens", type=int, default=500)
    ap.add_argument("--fork_merge_min_confidence", type=float, default=0.67)
    ap.add_argument("--fork_merge_policy", type=str, default="full", help="full|weak|no_merge")
    ap.add_argument("--fork_weak_merge_max_chars", type=int, default=240)
    ap.add_argument("--fork_debug_force_step", type=int, default=10)
    ap.add_argument("--fork_debug_force_max_calls", type=int, default=1)
    ap.add_argument("--fork_gate_trace", action="store_true", default=True)
    ap.add_argument("--no_fork_gate_trace", action="store_false", dest="fork_gate_trace")
    ap.add_argument("--fork_gate_probe_run_on_ready", action="store_true", default=False)

    ap.add_argument("--fork_max_tokens", type=int, default=160)
    ap.add_argument("--fork_k", type=int, default=6)
    ap.add_argument("--fork_include_recent_active", action="store_true", default=True)
    ap.add_argument("--no_fork_include_recent_active", action="store_false", dest="fork_include_recent_active")
    ap.add_argument("--fork_recent_active_n", type=int, default=4)
    ap.add_argument("--enable_context_controller", action="store_true", default=False)
    ap.add_argument("--context_controller_policy", type=str, default="uncertainty_aware", help="stage_aware|budget_aware|uncertainty_aware")
    ap.add_argument("--context_controller_trace", action="store_true", default=True)
    ap.add_argument("--no_context_controller_trace", action="store_false", dest="context_controller_trace")
    ap.add_argument("--context_controller_support_gap_threshold", type=float, default=0.20)
    ap.add_argument("--context_controller_budget_pressure_threshold", type=float, default=0.80)
    ap.add_argument("--context_controller_fork_ambiguity_threshold", type=float, default=0.45)
    ap.add_argument(
        "--no_unfold_trigger_always_on_required_keys",
        action="store_false",
        dest="unfold_trigger_always_on_required_keys",
        help="Disable always-on relocation_* key trigger for unfold decisions.",
    )

    ap.add_argument("--prompt_context_chars", type=int, default=0)
    ap.add_argument("--log_context_chars", type=int, default=2500)

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
        save_goc_internal_graph=bool(args.save_goc_internal_graph),
        enable_unfold_trigger=bool(args.enable_unfold_trigger),
        unfold_trigger_missing_terms_threshold=int(args.unfold_trigger_missing_terms_threshold),
        unfold_trigger_min_token_len=int(args.unfold_trigger_min_token_len),
        unfold_trigger_max_keywords=int(args.unfold_trigger_max_keywords),
        unfold_trigger_k=int(args.unfold_trigger_k),
        unfold_trigger_log_missing_limit=int(args.unfold_trigger_log_missing_limit),
        unfold_trigger_always_on_required_keys=bool(
            args.unfold_trigger_always_on_required_keys
        ),
        fork_trigger_mode=str(args.fork_trigger_mode),
        fork_min_step=int(args.fork_min_step),
        fork_every_k_steps=int(args.fork_every_k_steps),
        fork_min_open_pages=int(args.fork_min_open_pages),
        fork_min_search_calls=int(args.fork_min_search_calls),
        fork_min_active_tokens=int(args.fork_min_active_tokens),
        fork_merge_min_confidence=float(args.fork_merge_min_confidence),
        fork_merge_policy=str(args.fork_merge_policy),
        fork_weak_merge_max_chars=int(args.fork_weak_merge_max_chars),
        fork_debug_force_step=int(args.fork_debug_force_step),
        fork_debug_force_max_calls=int(args.fork_debug_force_max_calls),
        fork_gate_trace=bool(args.fork_gate_trace),
        fork_gate_probe_run_on_ready=bool(args.fork_gate_probe_run_on_ready),
        fork_max_tokens=int(args.fork_max_tokens),
        fork_k=int(args.fork_k),
        fork_include_recent_active=bool(args.fork_include_recent_active),
        fork_recent_active_n=int(args.fork_recent_active_n),
        enable_context_controller=bool(args.enable_context_controller),
        context_controller_policy=str(args.context_controller_policy),
        context_controller_trace=bool(args.context_controller_trace),
        context_controller_support_gap_threshold=float(args.context_controller_support_gap_threshold),
        context_controller_budget_pressure_threshold=float(args.context_controller_budget_pressure_threshold),
        context_controller_fork_ambiguity_threshold=float(args.context_controller_fork_ambiguity_threshold),
    )
    print("Done:", res)

if __name__ == "__main__":
    main()
