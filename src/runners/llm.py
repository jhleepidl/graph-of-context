from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
import json
from datetime import datetime
from pathlib import Path

from ..benchmarks.base import Benchmark
from ..llm_openai import OpenAIChatCompletionsClient
from ..llm_agent import ToolLoopLLMAgent, ToolLoopConfig
from ..memory import (
    FullHistoryMemory,
    ContextFoldingDiscardMemory,
    LinearSummaryMemory,
    AgentFoldRangeMemory,
    GoCMemory,
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
    model: str = "gpt-4o-mini",
    dotenv_path: str = ".env",
    max_steps: int = 35,
    max_json_retries: int = 2,
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
) -> Dict[str, Any]:
    tools = benchmark.build_tools(data_dir, retriever_kind=retriever_kind, faiss_dim=faiss_dim)
    tasks = benchmark.load_tasks(data_dir, limit=task_limit)

    llm = OpenAIChatCompletionsClient(model=model, force_json=True, dotenv_path=dotenv_path)

    all_method_specs: Dict[str, MethodSpec] = {
        "FullHistory": MethodSpec("FullHistory", lambda: FullHistoryMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        "ContextFolding-Discard": MethodSpec("ContextFolding-Discard", lambda: ContextFoldingDiscardMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        "LinearSummary": MethodSpec("LinearSummary", lambda: LinearSummaryMemory(budget_active=budget_active, budget_unfold=budget_unfold, summary_every=linear_summary_every)),
        "AgentFold-Range": MethodSpec("AgentFold-Range", lambda: AgentFoldRangeMemory(budget_active=budget_active, budget_unfold=budget_unfold, fold_chunk=agentfold_fold_chunk)),
        "GoC": MethodSpec("GoC", lambda: GoCMemory(budget_active=budget_active, budget_unfold=budget_unfold, unfold_k=unfold_k)),
    }

    selected: List[MethodSpec] = []
    for m in methods:
        if m == "ALL":
            selected = list(all_method_specs.values())
            break
        selected.append(all_method_specs[m])

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for ms in selected:
        if verbose_steps:
            print(f"\n=== METHOD: {ms.name} ===", flush=True)
        mem = ms.mem_factory()
        agent = ToolLoopLLMAgent(
            llm=llm,
            tools=tools,
            mem=mem,
            cfg=ToolLoopConfig(
                max_steps=max_steps,
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
                verbose=verbose_steps,
                log_dir=log_dir,
            ),
        )

        for t in tasks:
            out = agent.run(t.question, task_id=t.id, method=ms.name, run_tag=run_tag)
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
                "docid_cov": float(ev.get("docid_cov", 0.0)),
                "usage": out.get("usage") or {},
                "tool_stats": out.get("tool_stats") or {},
                "steps": out.get("steps"),
                "elapsed_sec": out.get("elapsed_sec"),
                "confidence": out.get("confidence",""),
                "explanation": expl,
                "run_tag": run_tag,
                "retriever_kind": retriever_kind,
            }
            rows.append(row)

    with open(out_results_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report aggregate
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def _avg(xs): return sum(xs)/max(1,len(xs))
    lines = []
    lines.append(f"# LLM Report ({benchmark.name})\n")
    lines.append("| method | n | accuracy | avg_total_tokens | avg_steps | avg_elapsed_sec | avg_tool_calls | avg_search | avg_open | avg_repeat_search | json_fail | json_recover | avg_docid_coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rs in by_method.items():
        n = len(rs)
        acc = sum(1 for x in rs if x["correct"]) / max(1, n)
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
            f"| {name} | {n} | {acc:.3f} | {avg_tok:.1f} | {avg_steps:.1f} | {avg_elapsed:.2f} | "
            f"{avg_tools:.1f} | {avg_search:.1f} | {avg_open:.1f} | {avg_rep:.1f} | "
            f"{json_fail} | {json_rec} | {avg_cov:.3f} |"
        )

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"n_tasks": len(tasks), "n_rows": len(rows), "out_results": out_results_path, "out_report": out_report_path, "run_tag": run_tag}