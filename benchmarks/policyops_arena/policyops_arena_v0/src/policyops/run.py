from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .baselines import DummyClient, LLMClient, run_full_history, run_goc_heuristic, run_topk_rag
from .env import PolicyOpsEnv
from .eval import aggregate_metrics, evaluate_prediction, save_report
from .generator import generate_world_and_tasks
from .world import load_tasks, load_world


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


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

    client: LLMClient = DummyClient()
    metrics: List[Dict[str, float]] = []
    tool_calls: List[int] = []
    open_calls: List[int] = []
    prompt_tokens: List[int] = []

    for task in tasks:
        env = PolicyOpsEnv(
            world,
            tool_call_budget=task.budgets.get("tool_call_budget", 50),
            open_budget=task.budgets.get("open_budget", 5),
        )
        if args.method == "topk":
            pred, prompt, _ = run_topk_rag(task, env, client)
        elif args.method == "full":
            pred, prompt, _ = run_full_history(task, env, client)
        elif args.method == "goc":
            pred, prompt, _ = run_goc_heuristic(task, env)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        metrics.append(evaluate_prediction(pred, task.gold, world))
        tool_calls.append(env.tool_call_count)
        open_calls.append(env.open_count)
        prompt_tokens.append(len(prompt.split()) if prompt else 0)

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
    }

    run_dir = Path(base_dir) / "runs" / args.method
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
    ev.add_argument("--model", type=str, default="dummy")
    ev.add_argument("--out_dir", type=Path, default=None)
    ev.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
