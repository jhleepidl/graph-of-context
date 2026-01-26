from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
import json
from datetime import datetime
from pathlib import Path

from ..benchmarks.base import Benchmark
from ..llm_openai import make_openai_client
from ..llm_agent import ToolLoopLLMAgent, ToolLoopConfig
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

    # Trace controls
    trace_messages: bool = True,
    trace_message_chars: int = 6000,
    trace_output_chars: int = 4000,

    # Prompt/log truncation (chars). 0 means "no extra truncation".
    prompt_context_chars: int = 0,
    log_context_chars: int = 2500,

    # Two-stage commit helpers (HotpotQA/FEVER-style)
    enforce_committed_supporting_titles: str = "none",
    committed_supporting_titles_n: int = 2,
    stage_aware_unfold_on_final: bool = True,
    stage_final_unfold_k: int = 6,

    # Resume / robustness
    resume: bool = False,

    # Optional per-task index to enable task-wise analysis without loading raw datasets.
    # When set, writes one JSONL row per task: {task_id, question, gold, meta}.
    out_task_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    bench_kwargs = bench_kwargs or {}
    tools = benchmark.build_tools(data_dir, retriever_kind=retriever_kind, faiss_dim=faiss_dim, **bench_kwargs)
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

    llm = make_openai_client(
        model=model,
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
        "GoC": MethodSpec("GoC", lambda: GoCMemory(budget_active=budget_active, budget_unfold=budget_unfold, unfold_k=unfold_k)),
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

    for ms in selected:
        if verbose_steps:
            print(f"\n=== METHOD: {ms.name} ===", flush=True)
        mem = ms.mem_factory()
        # Only override optional bool knobs when explicitly provided.
        require_docids = require_docids_in_finish if (require_docids_in_finish is not None) else True
        mt_auto = multi_turn_auto_inject if (multi_turn_auto_inject is not None) else True

        agent = ToolLoopLLMAgent(
            llm=llm,
            tools=tools,
            mem=mem,
            cfg=ToolLoopConfig(
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
            ),
        )

        for t in tasks:
            # Some benchmarks require per-task tool scoping (e.g., each task has its own context set).
            if hasattr(tools, "set_task"):
                try:
                    tools.set_task(t)
                except Exception:
                    # Never crash a full run due to scoping.
                    pass
            if (ms.name, t.id) in done_keys:
                continue

            try:
                out = agent.run(
                    t.question,
                    user_turns=t.turns,
                    task_meta=t.meta,
                    task_id=t.id,
                    method=ms.name,
                    run_tag=run_tag,
                )
                pred = (out.get("answer") or "").strip()
                expl = out.get("explanation") or ""
                ev = benchmark.evaluate(pred, expl, t)

                row = {
                    "benchmark": benchmark.name,
                    "method": ms.name,
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
                # Don't kill the whole run; record the failure and continue.
                row = {
                    "benchmark": benchmark.name,
                    "method": ms.name,
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

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()
            done_keys.add((ms.name, t.id))
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