from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Callable, Optional
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..benchmarks.base import Benchmark
from ..llm_openai import make_openai_client
from ..llm_agent import ToolLoopLLMAgent, ToolLoopConfig
from ..bandit_controller import BanditUnfoldController
from ..memory import (
    FullHistoryMemory,
    ContextFoldingDiscardMemory,
    LinearSummaryMemory,
    AgentFoldRangeMemory,
    GoCMemory,
    SimpleRAGMemory,
    SimilarityOnlyMemory,
    MemoryManagerBase
)

@dataclass
class MethodSpec:
    name: str
    mem_factory: Callable[[], MemoryManagerBase]

def run_llm(
    benchmark: Benchmark,
    data_dir: str,
    methods: List[str],
    out_results_path: str,
    out_report_path: str,
    bench_kwargs: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    dotenv_path: str = ".env",
    controller_model: Optional[str] = None,

    # Bandit controller (GoC-Bandit)
    bandit_model_path: Optional[str] = None,
    bandit_alpha: float = 1.0,
    bandit_epsilon: float = 0.05,
    max_steps: int = 35,
    max_json_retries: int = 2,
    # Difficulty / gating levers
    multi_turn_auto_inject: Optional[bool] = None,
    multi_turn_min_step: int = 8,
    multi_turn_min_open_pages: int = 3,
    min_steps_before_finish: int = 2,
    min_open_pages_before_finish: int = 1,
    require_docids_in_finish: Optional[bool] = None,
    deadline_finish_nudge_steps: int = 3,
    force_finish_on_deadline: bool = True,
    repeat_search_consecutive_threshold: int = 6,
    auto_open_on_repeat_search: bool = True,
    open_page_dedupe: bool = True,
    open_page_fact_header: bool = True,
    fact_header_max_chars: int = 420,
    open_given_projects_on_repeat_search: bool = True,
    validate_answer_in_given_projects: bool = True,
    budget_active: int = 1200,
    budget_unfold: int = 650,
    unfold_k: int = 8,
    linear_summary_every: int = 8,
    agentfold_fold_chunk: int = 10,
    task_limit: Optional[int] = None,
    retriever_kind: str = "bm25",
    faiss_dim: int = 384,
    # Debug
    verbose_steps: bool = False,
    log_dir: Optional[str] = None,

    # Parallelization
    # - parallel_tasks: run tasks concurrently (thread pool).
    parallel_tasks: int = 1,

    # Trace controls
    trace_messages: bool = True,
    trace_message_chars: int = 6000,
    trace_output_chars: int = 4000,

    # Prompt/log truncation (chars). 0 means "no extra truncation".
    prompt_context_chars: int = 0,
    log_context_chars: int = 2500,

    # Two-stage commit helpers (HotpotQA/FEVER-style)
    # Default to "goc_only" so GoC can leverage traceable commitments to
    # eliminate supporting_titles drift without affecting other baselines.
    enforce_committed_supporting_titles: str = "goc_only",
    committed_supporting_titles_n: int = 2,
    stage_aware_unfold_on_final: bool = True,
    stage_final_unfold_k: int = 6,
    stage_aware_unfold_on_commit: bool = True,
    stage_commit_unfold_k: int = 6,

    # Optional: override LLM-assisted GoC annotation settings (prompt gating + schema)
    # If None, defaults are used (and per-method overrides like GoC-HybridDep apply).
    goc_annotation_mode: Optional[str] = None,
    goc_annotation_gate: Optional[str] = None,
    goc_annotation_gate_pre_finish_steps: Optional[int] = None,
    goc_annotation_gate_every_k_steps: Optional[int] = None,
    goc_annotation_schema: Optional[str] = None,
    goc_annotation_force: Optional[int] = None,

    goc_declared_dep_max_back: Optional[int] = None,
    goc_declared_dep_max_per_step: Optional[int] = None,


    # Resume / robustness
    resume: bool = False,

    # Optional per-task index to enable task-wise analysis without loading raw datasets.
    # When set, writes one JSONL row per task: {task_id, question, gold, meta}.
    out_task_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Defensive: allow bench config (or sweep wrapper) to specify data_dir without
    # causing Python to receive data_dir twice (positional + kwarg).
    # If present in bench_kwargs, prefer it and remove from kwargs.
    bench_kwargs = dict(bench_kwargs or {})
    cfg_data_dir = bench_kwargs.pop("data_dir", None)
    if isinstance(cfg_data_dir, str) and cfg_data_dir:
        data_dir = cfg_data_dir
    tasks = benchmark.load_tasks(data_dir, limit=task_limit, **bench_kwargs)

    # Write a lightweight task index (best effort). Useful for later analysis of
    # which tasks each method wins/loses without reloading the dataset.
    if out_task_index_path:
        try:
            p = Path(out_task_index_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for t in tasks:
                    q = (getattr(t, "question", "") or "")
                    q_preview = q if len(q) <= 800 else (q[:800] + "...")
                    meta = getattr(t, "meta", None)
                    meta_small = {}
                    if isinstance(meta, dict):
                        for k, v in meta.items():
                            if isinstance(v, (str, int, float, bool)):
                                meta_small[k] = v
                    row = {
                        "task_id": getattr(t, "id", None),
                        "question": q_preview,
                        "gold": getattr(t, "answer", None),
                        "meta": meta_small,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    if not tasks:
        raise SystemExit(
            f"No tasks were loaded for benchmark='{benchmark.name}'. "
            "This usually means the dataset path/split is wrong or the file format is unsupported. "
            "For HotpotQA, both the official format (context as [[title,[sents]]...]) and the HuggingFace "
            "format (context as {title:[...], sentences:[...]}) are supported. "
            "Double-check --bench_cfg (e.g., {'path': '.../hotpot_dev_distractor_v1.json'}) and that the file is readable."
        )

    # Build an LLM client.
    llm = make_openai_client(
        model=model,
        api_mode="auto",
        temperature=0.0,
        force_json=True,
        dotenv_path=dotenv_path,
    )

    # Optional controller model for agentic unfold/fold (Option-B).
    controller_llm = llm
    if controller_model:
        controller_llm = make_openai_client(
            model=controller_model,
            api_mode="auto",
            temperature=0.0,
            force_json=True,
            dotenv_path=dotenv_path,
        )

    all_method_specs: Dict[str, MethodSpec] = {
        "FullHistory": MethodSpec("FullHistory", lambda: FullHistoryMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        "ContextFolding-Discard": MethodSpec("ContextFolding-Discard", lambda: ContextFoldingDiscardMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        "LinearSummary": MethodSpec("LinearSummary", lambda: LinearSummaryMemory(budget_active=budget_active, budget_unfold=budget_unfold, summary_every=linear_summary_every)),
        "SimpleRAG": MethodSpec(
            "SimpleRAG",
            lambda: SimpleRAGMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                window_last_n=10,
                rag_k=unfold_k,
                retriever_kind=retriever_kind,
                faiss_dim=faiss_dim,
            ),
        ),
        "AgentFold-Range": MethodSpec("AgentFold-Range", lambda: AgentFoldRangeMemory(budget_active=budget_active, budget_unfold=budget_unfold, fold_chunk=agentfold_fold_chunk)),
        "GoC": MethodSpec(
            "GoC",
            lambda: GoCMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                docid_index_mode="docid_title",
                trace_unfold_candidates=True,
            ),
        ),
        # Research variants: allow the acting model to emit optional "goc" annotations
        # (e.g., depends_on_steps) to help connect steps. These are intended for
        # controlled comparisons and should be treated as experimental.
        "GoC-HybridDep": MethodSpec(
            "GoC-HybridDep",
            lambda: GoCMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                docid_index_mode="docid_title",
                dep_closure_edge_types=("depends", "depends_llm", "doc_ref"),
                trace_unfold_candidates=True,
            ),
        ),
        "GoC-TraceFirst": MethodSpec(
            "GoC-TraceFirst",
            lambda: GoCMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                docid_index_mode="docid_title",
                dep_closure_edge_types=("depends", "depends_llm", "doc_ref"),
                trace_unfold_candidates=True,
            ),
        ),
        "GoC-Agentic": MethodSpec(
            "GoC-Agentic",
            lambda: GoCMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                docid_index_mode="docid_title",
                dep_closure_edge_types=("depends", "doc_ref"),
                trace_unfold_candidates=True,
            ),
        ),
        "GoC-Bandit": MethodSpec(
            "GoC-Bandit",
            lambda: GoCMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                docid_index_mode="docid_title",
                dep_closure_edge_types=("depends", "doc_ref"),
                trace_unfold_candidates=True,
            ),
        ),
        "SimilarityOnly": MethodSpec(
            "SimilarityOnly",
            lambda: SimilarityOnlyMemory(
                budget_active=budget_active,
                budget_unfold=budget_unfold,
                unfold_k=unfold_k,
                storage_retriever_kind=retriever_kind,
                storage_faiss_dim=faiss_dim,
            ),
        ),
    }

    selected: List[MethodSpec] = []
    for m in methods:
        if m == "ALL":
            selected = list(all_method_specs.values())
            break
        selected.append(all_method_specs[m])

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Resume support ---
    out_path = Path(out_results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()  # (method, task_id)
    if resume and out_path.exists():
        try:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                done_keys.add((r.get("method"), r.get("task_id")))
        except Exception:
            # If an existing file is corrupted, do not attempt resume.
            done_keys = set()

    # Stream writes so we can recover from crashes without losing completed tasks.
    f_mode = "a" if (resume and out_path.exists() and done_keys) else "w"
    f_out = open(out_path, f_mode, encoding="utf-8")
    written = 0

    # Shared config used for per-task agents.
    require_docids = require_docids_in_finish if (require_docids_in_finish is not None) else True
    mt_auto = multi_turn_auto_inject if (multi_turn_auto_inject is not None) else True
    base_cfg = ToolLoopConfig(
                max_steps=max_steps,
                min_steps_before_finish=min_steps_before_finish,
                min_open_pages_before_finish=min_open_pages_before_finish,
                require_docids_in_finish=require_docids,
                max_json_retries=max_json_retries,
                deadline_finish_nudge_steps=deadline_finish_nudge_steps,
                force_finish_on_deadline=force_finish_on_deadline,
                repeat_search_consecutive_threshold=repeat_search_consecutive_threshold,
                auto_open_on_repeat_search=auto_open_on_repeat_search,
                open_page_dedupe=open_page_dedupe,
                open_page_fact_header=open_page_fact_header,
                fact_header_max_chars=fact_header_max_chars,
                open_given_projects_on_repeat_search=open_given_projects_on_repeat_search,
                validate_answer_in_given_projects=validate_answer_in_given_projects,
                multi_turn_auto_inject=mt_auto,
                multi_turn_min_step=multi_turn_min_step,
                multi_turn_min_open_pages=multi_turn_min_open_pages,
                verbose=verbose_steps,
                log_dir=log_dir,
                log_messages=bool(trace_messages),
                log_message_chars=int(trace_message_chars or 0),
                prompt_context_chars=prompt_context_chars,
                log_context_chars=log_context_chars,
                log_output_chars=int(trace_output_chars or 0),
                enforce_committed_supporting_titles=enforce_committed_supporting_titles,
                committed_supporting_titles_n=int(committed_supporting_titles_n),
                stage_aware_unfold_on_final=bool(stage_aware_unfold_on_final),
                stage_final_unfold_k=int(stage_final_unfold_k),
                stage_aware_unfold_on_commit=bool(stage_aware_unfold_on_commit),
                stage_commit_unfold_k=int(stage_commit_unfold_k),
            )

    # Shared bandit controller instance (read-only during evaluation).
    # Training is typically done offline from traces via scripts/build_bandit_dataset.py
    # and scripts/train_bandit_linucb.py.
    shared_bandit: Optional[BanditUnfoldController] = None
    try:
        # Create on-demand only if any selected method is GoC-Bandit.
        if any(str(ms.name) == "GoC-Bandit" for ms in selected):
            shared_bandit = BanditUnfoldController(alpha=float(bandit_alpha), epsilon=float(bandit_epsilon))
            if bandit_model_path:
                shared_bandit.load_json(bandit_model_path)
    except Exception:
        shared_bandit = None

    # Thread-safe writer for parallel runs.
    write_lock = threading.Lock()

    def _run_one_task(ms_name: str, mem_factory: Callable[[], MemoryManagerBase], t) -> Dict[str, Any]:
        """Run one task with fresh tools+agent (thread-safe)."""
        local_tools = benchmark.build_tools(data_dir, retriever_kind=retriever_kind, faiss_dim=faiss_dim, **bench_kwargs)
        if hasattr(local_tools, "set_task"):
            try:
                local_tools.set_task(t)
            except Exception:
                pass
        mem = mem_factory()
        cfg = base_cfg
        if str(ms_name).lower().startswith('goc-agentic') or str(ms_name).lower().endswith('agentic'):
            cfg = replace(base_cfg, enable_agentic_unfold=True)
        # Experimental: let the acting model declare lightweight graph annotations
        # (depends_on_steps + optional notes) to help GoC connect steps.
        if str(ms_name) == "GoC-HybridDep":
            cfg = replace(cfg, goc_annotation_mode="hybrid_depends")
        elif str(ms_name) == "GoC-TraceFirst":
            cfg = replace(cfg, goc_annotation_mode="tracefirst")

        # CLI overrides (if provided) take precedence over per-method defaults.
        if goc_annotation_mode is not None:
            cfg = replace(cfg, goc_annotation_mode=str(goc_annotation_mode))
        if goc_annotation_gate is not None:
            cfg = replace(cfg, goc_annotation_gate=str(goc_annotation_gate))
        if goc_annotation_gate_pre_finish_steps is not None:
            cfg = replace(cfg, goc_annotation_gate_pre_finish_steps=int(goc_annotation_gate_pre_finish_steps))
        if goc_annotation_gate_every_k_steps is not None:
            cfg = replace(cfg, goc_annotation_gate_every_k_steps=int(goc_annotation_gate_every_k_steps))
        if goc_annotation_schema is not None:
            cfg = replace(cfg, goc_annotation_schema=str(goc_annotation_schema))
        if goc_declared_dep_max_back is not None:
            cfg = replace(cfg, goc_declared_dep_max_back=int(goc_declared_dep_max_back))
        if goc_declared_dep_max_per_step is not None:
            cfg = replace(cfg, goc_declared_dep_max_per_step=int(goc_declared_dep_max_per_step))
        if goc_annotation_force is not None:
            cfg = replace(cfg, goc_annotation_force=bool(int(goc_annotation_force)))
        # Bandit controller mode (GoC-Bandit)
        bandit_ctl = None
        if str(ms_name) == "GoC-Bandit":
            cfg = replace(
                cfg,
                enable_bandit_unfold=True,
                bandit_alpha=float(bandit_alpha),
                bandit_epsilon=float(bandit_epsilon),
                bandit_model_path=str(bandit_model_path) if bandit_model_path else None,
            )
            bandit_ctl = shared_bandit

        agent = ToolLoopLLMAgent(
            llm=llm,
            tools=local_tools,
            mem=mem,
            cfg=cfg,
            controller_llm=controller_llm,
            bandit_controller=bandit_ctl,
        )
        try:
            out = agent.run(
                t.question,
                user_turns=t.turns,
                task_meta=t.meta,
                task_id=t.id,
                method=ms_name,
                run_tag=run_tag,
            )
            pred = (out.get("answer") or "").strip()
            expl = out.get("explanation") or ""
            ev = benchmark.evaluate(pred, expl, t)
            return {
                "benchmark": benchmark.name,
                "method": ms_name,
                "task_id": t.id,
                "pred": pred,
                "gold": t.answer,
                "correct": bool(ev.get("correct")),
                "correct_strict": bool(ev.get("correct_strict")) if ("correct_strict" in ev) else None,
                "pred_norm": ev.get("pred_norm"),
                "gold_norm": ev.get("gold_norm"),
                "docid_cov": float(ev.get("docid_cov", 0.0)),
                "usage": out.get("usage") or {},
                "tool_stats": out.get("tool_stats") or {},
                "steps": out.get("steps"),
                "elapsed_sec": out.get("elapsed_sec"),
                "confidence": out.get("confidence", ""),
                "explanation": expl,
                "run_tag": run_tag,
                "retriever_kind": retriever_kind,
            }
        except Exception as e:
            return {
                "benchmark": benchmark.name,
                "method": ms_name,
                "task_id": t.id,
                "pred": "",
                "gold": t.answer,
                "correct": False,
                "correct_strict": None,
                "pred_norm": None,
                "gold_norm": None,
                "docid_cov": 0.0,
                "usage": {},
                "tool_stats": {"runner_error": True, "runner_error_msg": str(e)},
                "steps": None,
                "elapsed_sec": None,
                "confidence": "",
                "explanation": "",
                "run_tag": run_tag,
                "retriever_kind": retriever_kind,
                "error": str(e),
            }


    for ms in selected:
        if verbose_steps:
            print(f"\n=== METHOD: {ms.name} ===", flush=True)

        # Filter tasks for resume.
        todo = [t for t in tasks if (ms.name, t.id) not in done_keys]
        if not todo:
            continue

        n_workers = max(1, int(parallel_tasks or 1))
        if n_workers == 1:
            # Preserve the previous sequential behavior.
            for t in todo:
                row = _run_one_task(ms.name, ms.mem_factory, t)
                with write_lock:
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                done_keys.add((ms.name, t.id))
                written += 1
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_run_one_task, ms.name, ms.mem_factory, t) for t in todo]
                for fut in as_completed(futs):
                    row = fut.result()
                    with write_lock:
                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_out.flush()
                    done_keys.add((row.get("method"), row.get("task_id")))
                    written += 1

    f_out.close()

    # Load all rows from the (possibly resumed) results file for report aggregation.
    rows: List[Dict[str, Any]] = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    # report aggregate
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def _avg(xs): return sum(xs)/max(1,len(xs))
    lines = []
    lines.append(f"# LLM Report ({benchmark.name})\n")
    lines.append("| method | n | accuracy | accuracy_strict | avg_total_tokens | avg_steps | avg_elapsed_sec | avg_tool_calls | avg_search | avg_open | avg_repeat_search | json_fail | json_recover | avg_docid_coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rs in by_method.items():
        n = len(rs)
        acc = sum(1 for x in rs if x["correct"]) / max(1, n)
        strict_vals = [x.get("correct_strict") for x in rs if x.get("correct_strict") is not None]
        acc_strict = (sum(1 for v in strict_vals if v) / max(1, len(strict_vals))) if strict_vals else 0.0
        avg_tok = _avg([(x.get("usage", {}).get("total_tokens") or 0) for x in rs])
        avg_steps = _avg([(x.get("steps") or 0) for x in rs])
        avg_elapsed = _avg([(x.get("elapsed_sec") or 0.0) for x in rs])
        avg_tools = _avg([(x.get("tool_stats", {}).get("tool_calls_total") or 0) for x in rs])
        avg_search = _avg([(x.get("tool_stats", {}).get("search_calls") or 0) for x in rs])
        avg_open = _avg([(x.get("tool_stats", {}).get("open_page_calls") or 0) for x in rs])
        avg_rep = _avg([(x.get("tool_stats", {}).get("repeated_search_count") or 0) for x in rs])
        json_fail = sum((x.get("tool_stats", {}).get("json_parse_failures") or 0) for x in rs)
        json_rec = sum((x.get("tool_stats", {}).get("json_recoveries") or 0) for x in rs)
        avg_cov = _avg([x.get("docid_cov",0.0) for x in rs])
        lines.append(
            f"| {name} | {n} | {acc:.3f} | {acc_strict:.3f} | {avg_tok:.1f} | {avg_steps:.1f} | {avg_elapsed:.2f} | "
            f"{avg_tools:.1f} | {avg_search:.1f} | {avg_open:.1f} | {avg_rep:.1f} | "
            f"{json_fail} | {json_rec} | {avg_cov:.3f} |"
        )

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"n_tasks": len(tasks), "n_rows": len(rows), "out_results": out_results_path, "out_report": out_report_path, "run_tag": run_tag}