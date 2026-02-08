import argparse
import json
from pathlib import Path

from src.version import CODE_VERSION
from src.benchmarks.registry import get_benchmark, BENCHMARKS
from src.runners.deterministic import run_deterministic
from src.runners.llm import run_llm

def main():
    ap = argparse.ArgumentParser(description="Unified benchmark runner (deterministic or LLM).")
    ap.add_argument(
        "--print_version",
        action="store_true",
        help="Print the code version (for debugging patch/override issues) and exit.",
    )
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

    ap.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Alias for --bench_cfg_path (accepts sweep config JSON too).",
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
    ap.add_argument("--controller_model", type=str, default=None, help="Optional smaller controller LLM model for GoC-Agentic unfold selection.")

    # Optional: LLM-assisted GoC annotations (hybrid/trace-first)
    # These flags override per-method defaults when provided.
    ap.add_argument(
        "--goc_annotation_mode",
        type=str,
        default=None,
        choices=["none", "hybrid_depends", "tracefirst"],
        help="Override goc_annotation_mode in ToolLoopConfig (default: per-method).",
    )
    ap.add_argument(
        "--goc_annotation_gate",
        type=str,
        default=None,
        help="Override goc_annotation_gate (comma-separated triggers, e.g. 'doc_switch,pre_finish').",
    )
    ap.add_argument(
        "--goc_annotation_gate_pre_finish_steps",
        type=int,
        default=None,
        help="Override goc_annotation_gate_pre_finish_steps (default: 2).",
    )
    ap.add_argument(
        "--goc_annotation_gate_every_k_steps",
        type=int,
        default=None,
        help="Override goc_annotation_gate_every_k_steps (default: 0).",
    )
    ap.add_argument(
        "--goc_annotation_schema",
        type=str,
        default=None,
        choices=["compact", "legacy"],
        help="Override goc_annotation_schema (default: compact).",
    )
    ap.add_argument(
        "--goc_declared_dep_max_back",
        type=int,
        default=None,
        help="Override goc_declared_dep_max_back (default: 12).",
    )
    ap.add_argument(
        "--goc_declared_dep_max_per_step",
        type=int,
        default=None,
        help="Override goc_declared_dep_max_per_step (default: 6).",
    )

    ap.add_argument(
        "--goc_annotation_force",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Force GoC annotation emission on gated steps (0=off, 1=force goc key, 2=force minimal dep).",
    )

    # Bandit controller (GoC-Bandit)
    ap.add_argument(
        "--bandit_model_path",
        type=str,
        default=None,
        help="Path to a saved bandit model JSON (for GoC-Bandit). If omitted, uses an untrained prior.",
    )
    ap.add_argument("--bandit_alpha", type=float, default=1.0, help="LinUCB exploration alpha (GoC-Bandit).")
    ap.add_argument("--bandit_epsilon", type=float, default=0.05, help="Epsilon-greedy exploration prob (GoC-Bandit).")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--max_steps", type=int, default=35)
    ap.add_argument("--max_json_retries", type=int, default=2)

    # Parallelization
    ap.add_argument("--parallel_tasks", type=int, default=1, help="Run tasks concurrently (thread pool) to speed up eval.")

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
        default="goc_only",
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
    ap.add_argument("--save_goc_internal_graph", action="store_true", help="Write step/final GoC internal snapshots to <out_dir>/graphs_internal/<task_id>.jsonl.")

    # output
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, write results/report files into this directory (created if missing). Defaults to current working directory.",
    )

    args = ap.parse_args()

    if args.print_version:
        print(CODE_VERSION)
        return

    # Convenience alias: --sweep behaves like --bench_cfg_path.
    if getattr(args, 'sweep', None):
        if args.bench_cfg_path:
            raise SystemExit('Provide only one of --bench_cfg_path or --sweep')
        args.bench_cfg_path = args.sweep


    bench_kwargs = {}
    if args.bench_cfg_path:
        bench_kwargs = json.loads(Path(args.bench_cfg_path).read_text(encoding="utf-8"))
    else:
        try:
            bench_kwargs = json.loads(args.bench_cfg) if args.bench_cfg else {}
        except Exception as e:
            raise SystemExit(f"Invalid --bench_cfg JSON: {e}")

    bench = get_benchmark(args.benchmark)
    # Helper: shallow dict merge (b overrides a).
    def _merge_dict(a: dict, b: dict) -> dict:
        out = dict(a or {})
        if isinstance(b, dict):
            out.update(b)
        return out

    # If the user passed a sweep config, flatten it into benchmark kwargs so `run_benchmark.py`
    # works as a convenient entrypoint for quick sanity checks.
    if isinstance(bench_kwargs, dict):
        # Sweep format A (legacy): top-level has `bench_kwargs` + `grid`/`runs`.
        if isinstance(bench_kwargs.get("bench_kwargs"), dict) and (
            "grid" in bench_kwargs or "sweep_name" in bench_kwargs or "runs" in bench_kwargs
        ):
            sweep_cfg = bench_kwargs
            bench_kwargs = dict(sweep_cfg.get("bench_kwargs") or {})
            # If sweep config specifies data_dir and CLI stayed at default, prefer the sweep value.
            if str(args.data_dir) == "data" and isinstance(sweep_cfg.get("data_dir"), str):
                args.data_dir = sweep_cfg["data_dir"]

        # Sweep format B (run_sweep preset): top-level has `base` + `runs` (+ optional `grid`).
        # Benchmark kwargs live under base.bench_kwargs, and each run may override them.
        elif isinstance(bench_kwargs.get("base"), dict) and ("runs" in bench_kwargs or "grid" in bench_kwargs):
            sweep_cfg = bench_kwargs
            base = sweep_cfg.get("base") or {}
            base_bk = base.get("bench_kwargs") if isinstance(base.get("bench_kwargs"), dict) else {}

            rv_over: dict = {}
            runs = sweep_cfg.get("runs")
            if isinstance(runs, list) and runs and isinstance(runs[0], dict):
                rv = runs[0]
                rv_over = rv.get("bench_kwargs") if isinstance(rv.get("bench_kwargs"), dict) else {}

            bench_kwargs = _merge_dict(base_bk, rv_over)

            if str(args.data_dir) == "data" and isinstance(sweep_cfg.get("data_dir"), str):
                args.data_dir = sweep_cfg["data_dir"]

    # Defensive: if a bench config accidentally includes data_dir inside bench_kwargs,
    # remove it to avoid passing data_dir twice to load_tasks().
    if isinstance(bench_kwargs, dict):
        _cfg_dd = bench_kwargs.pop("data_dir", None)
        if str(args.data_dir) == "data" and isinstance(_cfg_dd, str) and _cfg_dd:
            args.data_dir = _cfg_dd

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
            controller_model=args.controller_model,
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
            save_goc_internal_graph=bool(args.save_goc_internal_graph),
            trace_messages=args.trace_messages,
            trace_message_chars=args.trace_message_chars,
            trace_output_chars=args.trace_output_chars,
            prompt_context_chars=args.prompt_context_chars,
            log_context_chars=args.log_context_chars,

            parallel_tasks=args.parallel_tasks,

            bandit_model_path=args.bandit_model_path,
            bandit_alpha=args.bandit_alpha,
            bandit_epsilon=args.bandit_epsilon,

            goc_annotation_mode=args.goc_annotation_mode,
            goc_annotation_gate=args.goc_annotation_gate,
            goc_annotation_gate_pre_finish_steps=args.goc_annotation_gate_pre_finish_steps,
            goc_annotation_gate_every_k_steps=args.goc_annotation_gate_every_k_steps,
            goc_annotation_schema=args.goc_annotation_schema,
            goc_declared_dep_max_back=args.goc_declared_dep_max_back,
            goc_declared_dep_max_per_step=args.goc_declared_dep_max_per_step,
            goc_annotation_force=args.goc_annotation_force,
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
