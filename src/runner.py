from __future__ import annotations
from typing import Dict, Any, List, Callable
import json
from dataclasses import dataclass

from .env import CorpusEnv
from .tools import ToolBox
from .memory import (
    FullHistoryMemory,
    ContextFoldingDiscardMemory,
    LinearSummaryMemory,
    AgentFoldRangeMemory,
    GoCMemory,
    MemoryManagerBase
)
from .agent import ContextLimitedAgent, AgentConfig
from .metrics import exact_match, docid_coverage, summarize_results

@dataclass
class MethodSpec:
    name: str
    mem_factory: Callable[[], MemoryManagerBase]

def run_one(method: MethodSpec, env: CorpusEnv, task: Dict[str, Any], cfg: AgentConfig) -> Dict[str, Any]:
    tools = ToolBox(env=env)
    mem = method.mem_factory()
    agent = ContextLimitedAgent(tools=tools, mem=mem, cfg=cfg)
    out = agent.solve(task)
    pred = out["answer"]
    gold = task["answer"]
    correct = exact_match(pred, gold)
    cov = docid_coverage(out["explanation"], task.get("gold_docids", []))
    return {
        "method": method.name,
        "task_id": task["id"],
        "pred": pred,
        "gold": gold,
        "correct": correct,
        "docid_cov": cov,
        "metrics": out["metrics"],
        "explanation": out["explanation"]
    }

def run_experiment(
    corpus_path: str,
    tasks_path: str,
    out_results_path: str,
    out_report_path: str,
    budget_active: int = 2000,
    budget_unfold: int = 800,
    unfold_k: int = 6,
    summary_keep_fields: int = 1,
    linear_summary_every: int = 8,
    agentfold_fold_chunk: int = 10,
):
    env = CorpusEnv.from_json(corpus_path)
    tasks = json.load(open(tasks_path, "r", encoding="utf-8"))

    cfg = AgentConfig(topk=5, summary_keep_fields=summary_keep_fields, unfold_k=unfold_k)

    methods: List[MethodSpec] = [
        MethodSpec("FullHistory", lambda: FullHistoryMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        MethodSpec("ContextFolding-Discard", lambda: ContextFoldingDiscardMemory(budget_active=budget_active, budget_unfold=budget_unfold)),
        MethodSpec("LinearSummary", lambda: LinearSummaryMemory(budget_active=budget_active, budget_unfold=budget_unfold, summary_every=linear_summary_every)),
        MethodSpec("AgentFold-Range", lambda: AgentFoldRangeMemory(budget_active=budget_active, budget_unfold=budget_unfold, fold_chunk=agentfold_fold_chunk)),
        MethodSpec("GoC", lambda: GoCMemory(budget_active=budget_active, budget_unfold=budget_unfold, unfold_k=unfold_k)),
    ]

    all_rows: List[Dict[str, Any]] = []
    for task in tasks:
        for m in methods:
            row = run_one(m, env, task, cfg)
            all_rows.append(row)

    with open(out_results_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    report_lines = []
    report_lines.append("# Report\n")
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_method.setdefault(r["method"], []).append(r)

    report_lines.append("| method | n | accuracy | avg_total_tokens | avg_peak_active_tokens | avg_tool_calls | avg_docid_coverage |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, rows in by_method.items():
        s = summarize_results(rows)
        report_lines.append(
            f"| {name} | {s['n']} | {s['accuracy']:.3f} | {s['avg_total_tokens']:.1f} | "
            f"{s['avg_peak_active_tokens']:.1f} | {s['avg_tool_calls']:.1f} | {s['avg_docid_coverage']:.3f} |"
        )

    report_lines.append("\n## Notes\n")
    report_lines.append("- Accuracy differences are driven by *lossy branch summaries* + distractor documents: if a method cannot recover omitted attributes, it re-searches and may be misled.")
    report_lines.append("- GoC preserves folded branch steps in storage and can *unfold a minimal dependency closure* to recover missing facts under a budget (`budget_unfold`).")
    report_lines.append("- AgentFold-Range is a heuristic baseline that folds old contiguous ranges into summaries and discards originals (lossy).")

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {"n_tasks": len(tasks), "n_rows": len(all_rows), "out_results": out_results_path, "out_report": out_report_path}
