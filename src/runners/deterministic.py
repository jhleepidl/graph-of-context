from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
import json

from ..benchmarks.base import Benchmark
from ..agent import ContextLimitedAgent, AgentConfig
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

def run_deterministic(
    benchmark: Benchmark,
    data_dir: str,
    methods: List[str],
    out_results_path: str,
    out_report_path: str,
    bench_kwargs: Optional[Dict[str, Any]] = None,
    budget_active: int = 2000,
    budget_unfold: int = 800,
    unfold_k: int = 6,
    summary_keep_fields: int = 1,
    linear_summary_every: int = 8,
    agentfold_fold_chunk: int = 10,
    task_limit: Optional[int] = None,
    retriever_kind: str = "bm25",
    faiss_dim: int = 384,
) -> Dict[str, Any]:
    bench_kwargs = bench_kwargs or {}
    tools = benchmark.build_tools(data_dir, retriever_kind=retriever_kind, faiss_dim=faiss_dim, **bench_kwargs)
    tasks = benchmark.load_tasks(data_dir, limit=task_limit, **bench_kwargs)

    cfg = AgentConfig(topk=5, summary_keep_fields=summary_keep_fields, unfold_k=unfold_k)

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

    rows = []
    for t in tasks:
        if hasattr(tools, "set_task"):
            try:
                tools.set_task(t)
            except Exception:
                pass
        task_dict = {"id": t.id, "question": t.question, "answer": t.answer}
        if t.entities is not None: task_dict["entities"] = t.entities
        if t.required is not None: task_dict["required"] = t.required
        if t.gold_docids is not None: task_dict["gold_docids"] = t.gold_docids

        for ms in selected:
            mem = ms.mem_factory()
            agent = ContextLimitedAgent(tools=tools, mem=mem, cfg=cfg)
            out = agent.solve(task_dict)
            pred = out["answer"]
            expl = out["explanation"]
            ev = benchmark.evaluate(pred, expl, t)
            row = {
                "benchmark": benchmark.name,
                "method": ms.name,
                "task_id": t.id,
                "pred": pred,
                "gold": t.answer,
                "correct": bool(ev.get("correct")),
                "docid_cov": float(ev.get("docid_cov", 0.0)),
                "metrics": out["metrics"],
                "explanation": expl,
                "retriever_kind": retriever_kind,
            }
            rows.append(row)

    with open(out_results_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # simple report
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def _avg(xs): return sum(xs)/max(1,len(xs))
    lines = []
    lines.append(f"# Report ({benchmark.name})\n")
    lines.append("| method | n | accuracy | avg_total_tokens | avg_peak_active_tokens | avg_tool_calls | avg_docid_coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, rs in by_method.items():
        n = len(rs)
        acc = sum(1 for x in rs if x["correct"]) / max(1, n)
        avg_total = _avg([x["metrics"]["llm_in_tokens"] + x["metrics"]["llm_out_tokens"] for x in rs])
        avg_peak = _avg([x["metrics"]["peak_active_tokens"] for x in rs])
        avg_tools = _avg([x["metrics"]["tool_calls"] for x in rs])
        avg_cov = _avg([x.get("docid_cov",0.0) for x in rs])
        lines.append(f"| {name} | {n} | {acc:.3f} | {avg_total:.1f} | {avg_peak:.1f} | {avg_tools:.1f} | {avg_cov:.3f} |")

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"n_tasks": len(tasks), "n_rows": len(rows), "out_results": out_results_path, "out_report": out_report_path}
