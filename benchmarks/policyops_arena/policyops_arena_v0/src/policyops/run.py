from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .baselines import (
    DummyClient,
    LLMClient,
    OpenAIClient,
    _ensure_min_evidence,
    run_full_history,
    run_goc_heuristic,
    run_topk_rag,
)
from .env import PolicyOpsEnv
from .eval import aggregate_metrics, evaluate_prediction, save_report
from .generator import generate_world_and_tasks
from .world import load_tasks, load_world


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def apply_evidence_padding(
    pred: Dict[str, Any],
    opened_ids: List[str],
    mode: str,
    min_count: int,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    raw_evidence = list(pred.get("evidence", []) or [])
    if opened_ids:
        filtered = []
        seen: set[str] = set()
        for cid in raw_evidence:
            if cid in opened_ids and cid not in seen:
                filtered.append(cid)
                seen.add(cid)
    else:
        filtered = raw_evidence

    if mode == "none":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = filtered
        return pred_for_eval, dict(pred_for_eval), filtered, filtered

    evidence_after = list(filtered)
    if opened_ids:
        evidence_after = _ensure_min_evidence(evidence_after, opened_ids, min_count=min_count)

    if mode == "schema_only":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = filtered
        pred_for_record = dict(pred)
        pred_for_record["evidence"] = evidence_after
        return pred_for_eval, pred_for_record, filtered, evidence_after

    if mode == "global":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = evidence_after
        return pred_for_eval, dict(pred_for_eval), filtered, evidence_after

    raise ValueError(f"Unknown evidence padding mode: {mode}")


def cmd_generate(args: argparse.Namespace) -> None:
    generate_world_and_tasks(
        out_dir=args.out_dir,
        seed=args.seed,
        n_docs=args.n_docs,
        clauses_per_doc=args.clauses_per_doc,
        n_tasks=args.n_tasks,
        exception_chain_depth=args.exception_chain_depth,
        update_rate=args.update_rate,
        definition_density=args.definition_density,
        distractor_strength=args.distractor_strength,
    )
    print("Generated PolicyOps Arena v0 data.")


def cmd_eval(args: argparse.Namespace) -> None:
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"

    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)

    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
    else:
        client = DummyClient()
    metrics: List[Dict[str, float]] = []
    tool_calls: List[int] = []
    open_calls: List[int] = []
    prompt_tokens: List[int] = []
    records: List[Dict[str, Any]] = []

    run_dir = Path(base_dir) / "runs" / args.method
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        env = PolicyOpsEnv(
            world,
            tool_call_budget=task.budgets.get("tool_call_budget", 50),
            open_budget=task.budgets.get("open_budget", 5),
        )
        error: str | None = None
        raw_output: str | None = None
        if args.method == "topk":
            pred, opened_ids, prompt, raw_output = run_topk_rag(task, env, client)
        elif args.method == "full":
            pred, opened_ids, prompt, raw_output = run_full_history(task, env, client)
        elif args.method == "goc":
            pred, opened_ids, prompt, raw_output, error = run_goc_heuristic(task, env, client)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        pred_for_eval, pred_for_record, evidence_before, evidence_after = apply_evidence_padding(
            pred,
            opened_ids,
            mode=args.evidence_padding_mode,
            min_count=args.min_evidence_count,
        )
        task_metrics = evaluate_prediction(pred_for_eval, task.gold, world)
        metrics.append(task_metrics)
        tool_calls.append(env.tool_call_count)
        open_calls.append(env.open_count)
        prompt_tokens.append(len(prompt.split()) if prompt else 0)
        pred_decision = pred_for_record.get("decision")
        record: Dict[str, Any] = {
            "task_id": task.task_id,
            "method": args.method,
            "opened_clause_ids": opened_ids,
            "tool_calls": env.tool_call_count,
            "open_calls": env.open_count,
            "pred_decision": pred_decision,
            "gold_decision": task.gold.decision,
            "decision_correct": pred_decision == task.gold.decision,
            "evidence_precision": task_metrics.get("evidence_precision"),
            "evidence_recall": task_metrics.get("evidence_recall"),
            "critical_evidence_hit": task_metrics.get("critical_evidence_hit"),
            "pred_evidence_count": len(pred_for_record.get("evidence", []) or []),
            "gold_evidence_count": len(task.gold.gold_evidence or []),
            "error": error,
            "evidence_padding_mode": args.evidence_padding_mode,
            "min_evidence_count": args.min_evidence_count,
        }
        if args.evidence_padding_mode in {"schema_only", "global"}:
            record["evidence_before_pad"] = evidence_before
            record["evidence_after_pad"] = evidence_after
        if args.save_prompts:
            prompt_dir = run_dir / "prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / f"{task.task_id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            record["prompt_path"] = str(prompt_path)
        if args.save_raw:
            raw_dir = run_dir / "raw_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{task.task_id}.txt"
            raw_path.write_text(raw_output or "", encoding="utf-8")
            record["raw_path"] = str(raw_path)
        records.append(record)

    aggregate = aggregate_metrics(metrics)
    report = {
        "method": args.method,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": aggregate,
        "counts": {"tasks": len(tasks)},
        "usage": {
            "tool_calls_avg": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
            "open_calls_avg": sum(open_calls) / len(open_calls) if open_calls else 0.0,
            "prompt_tokens_avg": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0,
        },
        "tool_calls": sum(tool_calls),
        "open_calls": sum(open_calls),
        "records": records,
    }

    out_path = run_dir / f"{timestamp}.json"
    save_report(out_path, report)

    print("Evaluation complete.")
    for key, value in aggregate.items():
        print(f"{key}: {value:.3f}")
    print(f"Report saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolicyOps Arena v0 runner")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic world and tasks")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--n_docs", type=int, default=30)
    gen.add_argument("--clauses_per_doc", type=int, default=5)
    gen.add_argument("--n_tasks", type=int, default=200)
    gen.add_argument("--exception_chain_depth", type=int, default=2)
    gen.add_argument("--update_rate", type=float, default=0.3)
    gen.add_argument("--definition_density", type=float, default=0.4)
    gen.add_argument("--distractor_strength", type=float, default=0.3)
    gen.add_argument("--out_dir", type=Path, default=None)
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="Evaluate baselines")
    ev.add_argument("--method", choices=["topk", "full", "goc"], required=True)
    ev.add_argument("--model", type=str, default="gpt-4.1-mini")
    ev.add_argument("--llm", choices=["dummy", "openai"], default="dummy")
    ev.add_argument("--dotenv", type=str, default="../../../.env")
    ev.add_argument("--out_dir", type=Path, default=None)
    ev.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    ev.add_argument("--min_evidence_count", type=int, default=2)
    ev.add_argument("--save_prompts", action="store_true", help="Save per-task prompts to files")
    ev.add_argument("--save_raw", action="store_true", help="Save per-task raw outputs to files")
    ev.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
