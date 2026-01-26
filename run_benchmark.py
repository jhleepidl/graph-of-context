import argparse
import json
from pathlib import Path

from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm

def main():
    ap = argparse.ArgumentParser(description="Unified benchmark runner (deterministic or LLM).")
    ap.add_argument("--benchmark", type=str, default="synthetic_browsecomp", choices=list(BENCHMARKS.keys()))
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--prepare", action="store_true", help="Generate/download benchmark data into data_dir.")

    # Benchmark-specific knobs (passed through to the benchmark implementation).
    # Use JSON so we can add levers without exploding CLI flags.
    ap.add_argument(
        "--bench_cfg",
        type=str,
        default="{}",
        help="JSON string with benchmark-specific config knobs (augmentation, split, etc).",
    )
    ap.add_argument(
        "--bench_cfg_path",
        type=str,
        default=None,
        help="Path to a JSON file with benchmark-specific config knobs (overrides --bench_cfg).",
    )

    ap.add_argument("--runner", type=str, default="deterministic", choices=["deterministic", "llm"])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["GoC"],
        help="Methods to run. Accepts space-separated and/or comma-separated values, or ALL. e.g., --methods GoC FullHistory or --methods GoC,FullHistory or --methods ALL",
    )
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

    # Multi-turn gating / difficulty levers (LLM runner).
    ap.add_argument("--multi_turn_auto_inject", action="store_true", default=None, help="Enable auto-injection of pending user turns (default: enabled).")
    ap.add_argument("--no_multi_turn_auto_inject", action="store_false", dest="multi_turn_auto_inject", help="Disable auto-injection of pending user turns.")
    ap.add_argument("--multi_turn_min_step", type=int, default=8)
    ap.add_argument("--multi_turn_min_open_pages", type=int, default=3)
    ap.add_argument("--min_steps_before_finish", type=int, default=2)
    ap.add_argument("--min_open_pages_before_finish", type=int, default=1)
    ap.add_argument("--require_docids_in_finish", action="store_true", default=None, help="Require evidence docids in finish explanation (default: enabled).")
    ap.add_argument("--no_require_docids_in_finish", action="store_false", dest="require_docids_in_finish", help="Do not require evidence docids in finish explanation.")

    # Two-stage commit helpers (HotpotQA/FEVER-style)
    ap.add_argument(
        "--enforce_committed_supporting_titles",
        type=str,
        default="none",
        choices=["none", "goc_only", "all"],
        help="If stage-1 committed supporting titles are available, enforce them in final finish JSON (avoids supporting_titles drift).",
    )
    ap.add_argument("--committed_supporting_titles_n", type=int, default=2, help="How many committed titles to keep/enforce (default: 2).")
    ap.add_argument("--stage_aware_unfold_on_final", action="store_true", default=True, help="Proactively unfold around Q1/commit anchors on FINAL prompts.")
    ap.add_argument("--no_stage_aware_unfold_on_final", action="store_false", dest="stage_aware_unfold_on_final", help="Disable stage-aware unfold on FINAL prompts.")
    ap.add_argument("--stage_final_unfold_k", type=int, default=6, help="How many nodes to unfold on FINAL stage (default: 6).")

    # retriever knobs
    ap.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "faiss"])
    ap.add_argument("--faiss_dim", type=int, default=384)

    # debugging
    ap.add_argument("--verbose_steps", action="store_true", help="Print per-step progress to stdout (LLM runner only).")
    ap.add_argument("--log_dir", type=str, default=None, help="If set, write per-task LLM prompt/output traces as JSONL files into this directory.")
    ap.add_argument("--trace_dir", type=str, default=None, help="Alias for --log_dir.")
    ap.add_argument("--trace_messages", action="store_true", default=True, help="Include full prompt message list in traces (default: enabled).")
    ap.add_argument("--no_trace_messages", action="store_false", dest="trace_messages", help="Do not store full prompt messages; store only context tail metadata.")
    ap.add_argument("--trace_message_chars", type=int, default=6000, help="Per-message char cap in prompt traces when --trace_messages is enabled.")
    ap.add_argument("--trace_output_chars", type=int, default=4000, help="Char cap for raw model outputs stored in traces (llm_attempts).")
    ap.add_argument("--prompt_context_chars", type=int, default=0, help="Optional char-level truncation for ACTIVE_CONTEXT in the LLM prompt (0 = no extra truncation).")
    ap.add_argument("--log_context_chars", type=int, default=2500, help="Char-level tail truncation for traces/logs (does not affect prompt when prompt_context_chars==0).")

    # output
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, write results/report files into this directory (created if missing). Defaults to current working directory.",
    )

    args = ap.parse_args()

    bench_kwargs = {}
    if args.bench_cfg_path:
        bench_kwargs = json.loads(Path(args.bench_cfg_path).read_text(encoding="utf-8"))
    else:
        try:
            bench_kwargs = json.loads(args.bench_cfg) if args.bench_cfg else {}
        except Exception as e:
            raise SystemExit(f"Invalid --bench_cfg JSON: {e}")

    bench = get_benchmark(args.benchmark)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        meta = bench.prepare(
            data_dir=str(data_dir),
            **bench_kwargs,
            n_entities=args.n_entities,
            n_tasks=args.n_tasks,
            distractors_per_entity=args.distractors_per_entity,
            noise_docs=args.noise_docs,
            seed=args.seed,
        )
        print("Prepared:", meta)

    # Normalize methods: accept ['GoC', 'FullHistory'] or ['GoC,FullHistory'] etc.
    flat: list[str] = []
    for token in (args.methods or []):
        flat.extend([x.strip() for x in str(token).split(",") if x.strip()])
    methods = ["ALL"] if (len(flat) == 1 and flat[0].upper() == "ALL") else flat

    # Output directory (optional)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic runner in this repo is tailored for synthetic_browsecomp only.
    if args.runner == "deterministic" and args.benchmark != "synthetic_browsecomp":
        raise SystemExit(
            "The deterministic runner currently supports only --benchmark synthetic_browsecomp. "
            "For HotpotQA/LostInMiddle/FEVER, please run with: --runner llm"
        )

    if args.runner == "deterministic":
        res = run_deterministic(
            benchmark=bench,
            data_dir=str(data_dir),
            bench_kwargs=bench_kwargs,
            methods=methods,
            out_results_path=str(out_dir / "results.jsonl"),
            out_report_path=str(out_dir / "report.md"),
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
        # trace_dir is an alias for log_dir
        log_dir = args.trace_dir or args.log_dir
        res = run_llm(
            benchmark=bench,
            data_dir=str(data_dir),
            bench_kwargs=bench_kwargs,
            methods=methods,
            out_results_path=str(out_dir / "llm_results.jsonl"),
            out_report_path=str(out_dir / "llm_report.md"),
            model=args.model,
            dotenv_path=args.dotenv,
            max_steps=args.max_steps,
            max_json_retries=args.max_json_retries,
            multi_turn_auto_inject=args.multi_turn_auto_inject,
            multi_turn_min_step=args.multi_turn_min_step,
            multi_turn_min_open_pages=args.multi_turn_min_open_pages,
            min_steps_before_finish=args.min_steps_before_finish,
            min_open_pages_before_finish=args.min_open_pages_before_finish,
            require_docids_in_finish=args.require_docids_in_finish,
            enforce_committed_supporting_titles=args.enforce_committed_supporting_titles,
            committed_supporting_titles_n=args.committed_supporting_titles_n,
            stage_aware_unfold_on_final=args.stage_aware_unfold_on_final,
            stage_final_unfold_k=args.stage_final_unfold_k,
            budget_active=args.budget_active,
            budget_unfold=args.budget_unfold,
            unfold_k=args.unfold_k,
            linear_summary_every=args.linear_summary_every,
            agentfold_fold_chunk=args.agentfold_fold_chunk,
            task_limit=args.task_limit,
            retriever_kind=args.retriever,
            faiss_dim=args.faiss_dim,
            verbose_steps=args.verbose_steps,
            log_dir=log_dir,
            trace_messages=args.trace_messages,
            trace_message_chars=args.trace_message_chars,
            trace_output_chars=args.trace_output_chars,
            prompt_context_chars=args.prompt_context_chars,
            log_context_chars=args.log_context_chars,
        )

    # Be explicit about outputs so users don't look at the wrong working directory.
    print("Done:", res)
    try:
        print("Outputs:")
        print(" -", (Path(res.get("out_results") or "").resolve()))
        print(" -", (Path(res.get("out_report") or "").resolve()))
    except Exception:
        pass

if __name__ == "__main__":
    main()
