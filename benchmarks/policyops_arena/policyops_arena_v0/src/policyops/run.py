from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .baselines import (
    DummyClient,
    LLMClient,
    OpenAIClient,
    _ensure_min_evidence,
    run_engine_oracle,
    run_full_history,
    run_goc_heuristic,
    run_oracle,
    run_topk_rag,
)
from .controller import Controller
from .diagnostics import compute_retrieval_diagnostics
from .analysis import analyze_failure_slice
from .env import PolicyOpsEnv
from .eval import aggregate_metrics, evaluate_prediction, gold_decision_distribution, save_report
from .generator import generate_world_and_tasks
from .world import load_tasks, load_world


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _avg_p90(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p90": 0.0}
    avg = sum(values) / len(values)
    ordered = sorted(values)
    idx = int(0.9 * (len(ordered) - 1))
    p90 = ordered[idx]
    return {"avg": avg, "p90": float(p90)}


def _resolve_controller_state_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def _apply_open_budget(tasks: List[Any], open_budget: int) -> List[Any]:
    cloned = copy.deepcopy(tasks)
    for task in cloned:
        task.budgets["open_budget"] = open_budget
    return cloned


def _split_tasks(
    tasks: List[Any],
    task_split: str,
    train_ratio: float,
    split_seed: int,
) -> tuple[List[Any], List[Any]]:
    if task_split != "holdout":
        return [], list(tasks)
    rng = __import__("random").Random(split_seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]

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


def _evaluate_method(
    method: str,
    world: Any,
    tasks: List[Any],
    args: argparse.Namespace,
    client: LLMClient,
    run_dir: Path,
    controller: Controller | None = None,
    controller_mode: str = "off",
    llm_backend: str = "dummy",
    client_class: str = "DummyClient",
    resolved_model: str = "dummy",
) -> Dict[str, Any]:
    metrics: List[Dict[str, float]] = []
    tool_calls: List[int] = []
    open_calls: List[int] = []
    prompt_tokens_list: List[int] = []
    records: List[Dict[str, Any]] = []
    controller_actions: Dict[str, int] = {}

    for task in tasks:
        env = PolicyOpsEnv(
            world,
            tool_call_budget=task.budgets.get("tool_call_budget", 50),
            open_budget=task.budgets.get("open_budget", 5),
        )
        error: str | None = None
        raw_output: str | None = None
        controller_action: str | None = None
        diag: Dict[str, Any] = {}
        oracle_meta: Dict[str, Any] = {}
        if method == "topk":
            pred, opened_ids, prompt, raw_output, diag = run_topk_rag(
                task, env, client, primary_top_k=args.primary_search_top_k
            )
        elif method == "full":
            pred, opened_ids, prompt, raw_output, diag = run_full_history(
                task, env, client, primary_top_k=args.primary_search_top_k
            )
        elif method == "goc":
            pred, opened_ids, prompt, raw_output, error, controller_action, diag = run_goc_heuristic(
                task,
                env,
                client,
                controller=controller,
                controller_mode=controller_mode,
                primary_top_k=args.primary_search_top_k,
                force_open_top_n=args.force_open_top_n,
                force_open_source=args.force_open_source,
            )
            if controller_action:
                controller_actions[controller_action] = controller_actions.get(controller_action, 0) + 1
        elif method == "oracle":
            pred, opened_ids, prompt, raw_output, oracle_meta = run_oracle(
                task, env, client, primary_top_k=args.primary_search_top_k
            )
            diag = oracle_meta
        elif method == "engine":
            pred = run_engine_oracle(task, env)
            opened_ids = []
            prompt = ""
            raw_output = None
            diag = {"primary_search_results": [], "primary_search_top_k": 0, "primary_search_query": ""}
        else:
            raise ValueError(f"Unknown method: {method}")

        primary_results = diag.get("primary_search_results", []) if isinstance(diag, dict) else []
        retrieval_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(task.gold.gold_evidence or []),
            search_results=primary_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=args.save_search_snapshot,
            snapshot_k=args.search_snapshot_k,
        )

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
        prompt_tokens = len(prompt.split()) if prompt else 0
        prompt_tokens_list.append(prompt_tokens)
        pred_decision = pred_for_record.get("decision")
        action_reward = None
        action_opened_has_winning = None
        action_opened_gold_coverage = None
        action_critical_kind_hits = None
        if controller_action:
            winning_clause = task.gold.gold_evidence[0] if task.gold.gold_evidence else None
            action_opened_has_winning = (
                bool(winning_clause and winning_clause in opened_ids)
                if opened_ids
                else False
            )
            action_opened_gold_coverage = retrieval_diag.get("opened_gold_coverage")
            gold_ids = set(task.gold.gold_evidence or [])
            opened_id_set = set(opened_ids)
            critical_kinds = {"update", "definition", "exception"}
            gold_kinds = {
                world.clauses[cid].kind
                for cid in gold_ids
                if cid in world.clauses and world.clauses[cid].kind in critical_kinds
            }
            action_critical_kind_hits = 0
            for kind in gold_kinds:
                if any(
                    world.clauses.get(cid) and world.clauses[cid].kind == kind
                    for cid in opened_id_set
                ):
                    action_critical_kind_hits += 1
            action_reward = (
                (1.0 if action_opened_has_winning else 0.0)
                + 0.2 * float(action_opened_gold_coverage or 0.0)
                + 0.1 * min(3, action_critical_kind_hits)
                + 0.01 * len(opened_ids)
            )
        record: Dict[str, Any] = {
            "task_id": task.task_id,
            "method": method,
            "opened_clause_ids": opened_ids,
            "tool_calls": env.tool_call_count,
            "open_calls": env.open_count,
            "prompt_tokens": prompt_tokens,
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
            "controller_action": controller_action,
            "controller_action_reward": action_reward,
            "controller_action_opened_gold_coverage": action_opened_gold_coverage,
            "controller_action_opened_has_winning_clause": action_opened_has_winning,
            "controller_action_critical_kind_hit_count": action_critical_kind_hits,
        }
        if isinstance(diag, dict):
            record.update(diag)
        record.update(retrieval_diag)
        if args.evidence_padding_mode in {"schema_only", "global"}:
            record["evidence_before_pad"] = evidence_before
            record["evidence_after_pad"] = evidence_after
        include_raw_output = bool(
            getattr(args, "debug_n", 0) or getattr(args, "debug_task_ids", "")
        )
        if include_raw_output:
            record["raw_output"] = raw_output or ""
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
    aggregate["gold_decision_distribution"] = gold_decision_distribution(tasks)
    action_reward_mean = {}
    action_counts = {}
    action_cov_mean = {}
    action_winning_rate = {}
    if controller_actions:
        action_rewards: Dict[str, List[float]] = {}
        action_covs: Dict[str, List[float]] = {}
        action_wins: Dict[str, List[float]] = {}
        for rec in records:
            action = rec.get("controller_action")
            if not action:
                continue
            reward = rec.get("controller_action_reward")
            cov = rec.get("controller_action_opened_gold_coverage")
            win = rec.get("controller_action_opened_has_winning_clause")
            if isinstance(reward, (int, float)):
                action_rewards.setdefault(action, []).append(float(reward))
            if isinstance(cov, (int, float)):
                action_covs.setdefault(action, []).append(float(cov))
            if win is not None:
                action_wins.setdefault(action, []).append(1.0 if win else 0.0)
        for action, count in controller_actions.items():
            action_counts[action] = count
            rewards = action_rewards.get(action, [])
            covs = action_covs.get(action, [])
            wins = action_wins.get(action, [])
            action_reward_mean[action] = sum(rewards) / len(rewards) if rewards else 0.0
            action_cov_mean[action] = sum(covs) / len(covs) if covs else 0.0
            action_winning_rate[action] = sum(wins) / len(wins) if wins else 0.0
    report = {
        "method": method,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "metrics": aggregate,
        "counts": {"tasks": len(tasks)},
        "usage": {
            "tool_calls_avg": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
            "open_calls_avg": sum(open_calls) / len(open_calls) if open_calls else 0.0,
            "prompt_tokens_avg": sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0.0,
        },
        "tool_calls": sum(tool_calls),
        "open_calls": sum(open_calls),
        "records": records,
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "controller_actions_distribution": controller_actions,
        "controller_action_reward_mean": action_reward_mean,
        "controller_action_counts": action_counts,
        "controller_action_opened_gold_coverage_mean": action_cov_mean,
        "controller_action_opened_has_winning_clause_rate": action_winning_rate,
    }
    return report

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

    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")

    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)

    controller = None
    controller_mode = "off"
    state_path = None
    if args.use_controller and args.method == "goc":
        state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
        controller = Controller.load(state_path)
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_mode == "train" and train_tasks:
            train_args = argparse.Namespace(**vars(args))
            train_args.save_prompts = False
            train_args.save_raw = False
            _evaluate_method(
                "goc",
                world,
                train_tasks,
                train_args,
                DummyClient(),
                Path(base_dir) / "runs" / "controller_train",
                controller=controller,
                controller_mode="train",
            )
            controller.save(state_path)

    run_dir = Path(base_dir) / "runs" / args.method
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    report = _evaluate_method(
        args.method,
        world,
        eval_tasks,
        args,
        client,
        run_dir,
        controller=controller,
        controller_mode=controller_mode,
        llm_backend=llm_backend,
        client_class=client_class,
        resolved_model=resolved_model,
    )

    report["task_split"] = args.task_split
    report["train_ratio"] = args.train_ratio
    report["split_seed"] = args.split_seed
    report["num_train_tasks"] = len(train_tasks)
    report["num_eval_tasks"] = len(eval_tasks)

    out_path = run_dir / f"{timestamp}.json"
    save_report(out_path, report)

    print("Evaluation complete.")
    for key, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
    print(f"Report saved to {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"

    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)

    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")

    methods = args.methods or ["topk", "full", "goc", "oracle", "engine"]
    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)
    controller = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
        controller = Controller.load(state_path)
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_mode == "train" and train_tasks:
            train_args = argparse.Namespace(**vars(args))
            train_args.save_prompts = False
            train_args.save_raw = False
            _evaluate_method(
                "goc",
                world,
                train_tasks,
                train_args,
                DummyClient(),
                Path(base_dir) / "runs" / "controller_train",
                controller=controller,
                controller_mode="train",
            )
            controller.save(state_path)

    compare_dir = Path(base_dir) / "runs" / "compare"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    compare_run_dir = compare_dir / timestamp
    compare_run_dir.mkdir(parents=True, exist_ok=True)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for method in methods:
        method_dir = compare_run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_report = _evaluate_method(
            method,
            world,
            eval_tasks,
            args,
            client,
            method_dir,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            llm_backend=llm_backend,
            client_class=client_class,
            resolved_model=resolved_model,
        )
        method_reports[method] = method_report

        records = method_report.get("records", [])
        prompt_tokens_vals = [int(r.get("prompt_tokens", 0)) for r in records]
        open_calls_vals = [int(r.get("open_calls", 0)) for r in records]
        tool_calls_vals = [int(r.get("tool_calls", 0)) for r in records]
        opened_gold_cov_vals = [
            float(r.get("opened_gold_coverage"))
            for r in records
            if isinstance(r.get("opened_gold_coverage"), (int, float))
        ]
        gold_in_topk_vals = [
            1.0 if r.get("gold_in_search_topk") else 0.0
            for r in records
            if r.get("gold_in_search_topk") is not None
        ]
        winning_rank_vals = [
            int(r.get("winning_clause_rank"))
            for r in records
            if isinstance(r.get("winning_clause_rank"), int)
        ]
        min_gold_rank_vals = [
            int(r.get("min_gold_rank"))
            for r in records
            if isinstance(r.get("min_gold_rank"), int)
        ]
        gold_score_gap_vals = [
            float(r.get("gold_score_gap"))
            for r in records
            if isinstance(r.get("gold_score_gap"), (int, float))
        ]
        summary[method] = {
            "decision_accuracy": method_report["metrics"].get("decision_accuracy"),
            "condition_f1": method_report["metrics"].get("condition_f1"),
            "evidence_recall": method_report["metrics"].get("evidence_recall"),
            "critical_evidence_hit": method_report["metrics"].get("critical_evidence_hit"),
            "prompt_tokens": _avg_p90(prompt_tokens_vals),
            "open_calls": _avg_p90(open_calls_vals),
            "tool_calls": _avg_p90(tool_calls_vals),
            "opened_gold_coverage_mean": sum(opened_gold_cov_vals) / len(opened_gold_cov_vals)
            if opened_gold_cov_vals
            else 0.0,
            "gold_in_search_topk_rate": sum(gold_in_topk_vals) / len(gold_in_topk_vals)
            if gold_in_topk_vals
            else 0.0,
            "winning_clause_rank_mean": sum(winning_rank_vals) / len(winning_rank_vals)
            if winning_rank_vals
            else None,
            "min_gold_rank_mean": sum(min_gold_rank_vals) / len(min_gold_rank_vals)
            if min_gold_rank_vals
            else None,
            "gold_score_gap_mean": sum(gold_score_gap_vals) / len(gold_score_gap_vals)
            if gold_score_gap_vals
            else None,
        }
        if method == "oracle":
            oracle_cov = [
                r.get("oracle_gold_coverage")
                for r in records
                if isinstance(r.get("oracle_gold_coverage"), (int, float))
            ]
            summary[method]["oracle_gold_coverage"] = (
                sum(oracle_cov) / len(oracle_cov) if oracle_cov else 0.0
            )

    compare_report = {
        "methods": methods,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "gold_decision_distribution": gold_decision_distribution(eval_tasks),
        "summary": summary,
        "method_reports": method_reports,
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }

    out_path = compare_dir / f"{timestamp}.json"
    save_report(out_path, compare_report)
    print("Compare complete.")
    print(f"Report saved to {out_path}")

    if args.debug_n or args.debug_task_ids:
        debug_ids = []
        if args.debug_task_ids:
            debug_ids = [t.strip() for t in args.debug_task_ids.split(",") if t.strip()]
        for method, report in method_reports.items():
            print(f"[debug] method={method}")
            records = report.get("records", [])
            if debug_ids:
                selected = [r for r in records if r.get("task_id") in debug_ids]
            else:
                selected = records[: args.debug_n]
            for rec in selected:
                raw_preview = ""
                if args.save_raw and rec.get("raw_path"):
                    try:
                        raw_preview = Path(rec["raw_path"]).read_text(encoding="utf-8")[:300]
                    except Exception:
                        raw_preview = ""
                elif rec.get("raw_output"):
                    raw_preview = str(rec.get("raw_output", ""))[:300]
                raw_preview = " ".join(raw_preview.split())
                print(
                    f"task_id={rec.get('task_id')} gold={rec.get('gold_decision')} "
                    f"pred={rec.get('pred_decision')} opened={rec.get('opened_clause_ids')} "
                    f"error={rec.get('error')}"
                )
                print(f"raw_preview={raw_preview}")
                if rec.get("prompt_path"):
                    print(f"prompt_path={rec.get('prompt_path')}")
                if rec.get("raw_path"):
                    print(f"raw_path={rec.get('raw_path')}")


def _run_compare_with_tasks(
    world: Any,
    tasks: List[Any],
    args: argparse.Namespace,
    base_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")

    methods = args.methods or ["topk", "full", "goc", "oracle", "engine"]
    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)
    controller = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
        controller = Controller.load(state_path)
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_mode == "train" and train_tasks:
            train_args = argparse.Namespace(**vars(args))
            train_args.save_prompts = False
            train_args.save_raw = False
            _evaluate_method(
                "goc",
                world,
                train_tasks,
                train_args,
                DummyClient(),
                output_dir / "controller_train",
                controller=controller,
                controller_mode="train",
            )
            controller.save(state_path)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_report = _evaluate_method(
            method,
            world,
            eval_tasks,
            args,
            client,
            method_dir,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            llm_backend=llm_backend,
            client_class=client_class,
            resolved_model=resolved_model,
        )
        method_reports[method] = method_report

        records = method_report.get("records", [])
        prompt_tokens_vals = [int(r.get("prompt_tokens", 0)) for r in records]
        open_calls_vals = [int(r.get("open_calls", 0)) for r in records]
        tool_calls_vals = [int(r.get("tool_calls", 0)) for r in records]
        summary[method] = {
            "decision_accuracy": method_report["metrics"].get("decision_accuracy"),
            "condition_f1": method_report["metrics"].get("condition_f1"),
            "evidence_recall": method_report["metrics"].get("evidence_recall"),
            "critical_evidence_hit": method_report["metrics"].get("critical_evidence_hit"),
            "prompt_tokens": _avg_p90(prompt_tokens_vals),
            "open_calls": _avg_p90(open_calls_vals),
            "tool_calls": _avg_p90(tool_calls_vals),
        }
        if method == "oracle":
            oracle_cov = [
                r.get("oracle_gold_coverage")
                for r in records
                if isinstance(r.get("oracle_gold_coverage"), (int, float))
            ]
            summary[method]["oracle_gold_coverage"] = (
                sum(oracle_cov) / len(oracle_cov) if oracle_cov else 0.0
            )

    compare_report = {
        "methods": methods,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "gold_decision_distribution": gold_decision_distribution(eval_tasks),
        "summary": summary,
        "method_reports": method_reports,
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    return compare_report


def cmd_sweep(args: argparse.Namespace) -> None:
    base_dir = args.out_dir or _default_base_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(base_dir) / "runs" / "sweeps" / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = sweep_dir / f"seed={seed}"
        data_ready = (seed_dir / "data" / "worlds" / "documents.jsonl").exists()
        if not args.reuse_data or not data_ready:
            generate_world_and_tasks(
                out_dir=seed_dir,
                seed=seed,
                n_docs=args.n_docs,
                n_tasks=args.n_tasks,
            )

        world = load_world(seed_dir / "data" / "worlds")
        tasks = load_tasks(seed_dir / "data" / "tasks" / "tasks.jsonl")

        for open_budget in args.open_budgets:
            open_dir = seed_dir / f"open={open_budget}"
            open_dir.mkdir(parents=True, exist_ok=True)
            tasks_budget = _apply_open_budget(tasks, open_budget)

            compare_report = _run_compare_with_tasks(
                world,
                tasks_budget,
                args,
                base_dir=seed_dir,
                output_dir=open_dir,
            )
            compare_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            compare_path = open_dir / f"{compare_stamp}.json"
            save_report(compare_path, compare_report)

            for method, metrics in compare_report.get("summary", {}).items():
                row = {
                    "seed": seed,
                    "open_budget": open_budget,
                    "method": method,
                    "decision_accuracy": metrics.get("decision_accuracy"),
                    "condition_f1": metrics.get("condition_f1"),
                    "evidence_recall": metrics.get("evidence_recall"),
                    "critical_evidence_hit": metrics.get("critical_evidence_hit"),
                    "avg_open_calls": metrics.get("open_calls", {}).get("avg"),
                    "p90_open_calls": metrics.get("open_calls", {}).get("p90"),
                    "avg_prompt_tokens": metrics.get("prompt_tokens", {}).get("avg"),
                    "p90_prompt_tokens": metrics.get("prompt_tokens", {}).get("p90"),
                    "oracle_gold_coverage": metrics.get("oracle_gold_coverage"),
                    "opened_gold_coverage_mean": metrics.get("opened_gold_coverage_mean"),
                    "gold_in_search_topk_rate": metrics.get("gold_in_search_topk_rate"),
                    "winning_clause_rank_mean": metrics.get("winning_clause_rank_mean"),
                    "min_gold_rank_mean": metrics.get("min_gold_rank_mean"),
                    "gold_score_gap_mean": metrics.get("gold_score_gap_mean"),
                }
                summary_rows.append(row)

    summary_json_path = sweep_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    summary_csv_path = sweep_dir / "summary.csv"
    fieldnames = [
        "seed",
        "open_budget",
        "method",
        "decision_accuracy",
        "condition_f1",
        "evidence_recall",
        "critical_evidence_hit",
        "avg_open_calls",
        "p90_open_calls",
        "avg_prompt_tokens",
        "p90_prompt_tokens",
        "oracle_gold_coverage",
        "opened_gold_coverage_mean",
        "gold_in_search_topk_rate",
        "winning_clause_rank_mean",
        "min_gold_rank_mean",
        "gold_score_gap_mean",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Sweep complete.")
    print(f"Summary saved to {summary_csv_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    output_path = analyze_failure_slice(Path(args.report), top_k=args.k)
    print(f"Failure slice report saved to {output_path}")


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
    ev.add_argument("--method", choices=["topk", "full", "goc", "oracle", "engine"], required=True)
    ev.add_argument("--model", type=str, default="gpt-4.1-mini")
    ev.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    ev.add_argument("--dotenv", type=str, default=".env")
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
    ev.add_argument("--save_search_snapshot", action="store_true")
    ev.add_argument("--search_snapshot_k", type=int, default=20)
    ev.add_argument("--primary_search_top_k", type=int, default=20)
    ev.add_argument("--force_open_top_n", type=int, default=1)
    ev.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    ev.add_argument("--use_controller", action="store_true")
    ev.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    ev.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    ev.add_argument("--task_split", choices=["none", "holdout"], default="none")
    ev.add_argument("--train_ratio", type=float, default=0.7)
    ev.add_argument("--split_seed", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    cmp = sub.add_parser("compare", help="Compare methods in one run")
    cmp.add_argument("--methods", nargs="+", default=["topk", "full", "goc", "oracle", "engine"])
    cmp.add_argument("--model", type=str, default="gpt-4.1-mini")
    cmp.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    cmp.add_argument("--dotenv", type=str, default=".env")
    cmp.add_argument("--out_dir", type=Path, default=None)
    cmp.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    cmp.add_argument("--min_evidence_count", type=int, default=2)
    cmp.add_argument("--save_prompts", action="store_true", help="Save per-task prompts to files")
    cmp.add_argument("--save_raw", action="store_true", help="Save per-task raw outputs to files")
    cmp.add_argument("--save_search_snapshot", action="store_true")
    cmp.add_argument("--search_snapshot_k", type=int, default=20)
    cmp.add_argument("--primary_search_top_k", type=int, default=20)
    cmp.add_argument("--force_open_top_n", type=int, default=1)
    cmp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    cmp.add_argument("--use_controller", action="store_true")
    cmp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    cmp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    cmp.add_argument("--task_split", choices=["none", "holdout"], default="none")
    cmp.add_argument("--train_ratio", type=float, default=0.7)
    cmp.add_argument("--split_seed", type=int, default=0)
    cmp.add_argument("--debug_n", type=int, default=0)
    cmp.add_argument("--debug_task_ids", type=str, default="")
    cmp.set_defaults(func=cmd_compare)

    swp = sub.add_parser("sweep", help="Run multi-seed/open_budget sweeps")
    swp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    swp.add_argument("--open_budgets", nargs="+", type=int, default=[3, 5, 8])
    swp.add_argument("--n_docs", type=int, default=30)
    swp.add_argument("--n_tasks", type=int, default=200)
    swp.add_argument("--methods", nargs="+", default=["topk", "full", "goc", "oracle", "engine"])
    swp.add_argument("--model", type=str, default="gpt-4.1-mini")
    swp.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    swp.add_argument("--dotenv", type=str, default=".env")
    swp.add_argument("--out_dir", type=Path, default=None)
    swp.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    swp.add_argument("--min_evidence_count", type=int, default=2)
    swp.add_argument("--save_prompts", action="store_true")
    swp.add_argument("--save_raw", action="store_true")
    swp.add_argument("--save_search_snapshot", action="store_true")
    swp.add_argument("--search_snapshot_k", type=int, default=20)
    swp.add_argument("--primary_search_top_k", type=int, default=20)
    swp.add_argument("--force_open_top_n", type=int, default=1)
    swp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    swp.add_argument("--use_controller", action="store_true")
    swp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    swp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    swp.add_argument("--task_split", choices=["none", "holdout"], default="none")
    swp.add_argument("--train_ratio", type=float, default=0.7)
    swp.add_argument("--split_seed", type=int, default=0)
    swp.add_argument("--reuse_data", action="store_true")
    swp.set_defaults(func=cmd_sweep)

    ana = sub.add_parser("analyze", help="Analyze compare report failure slices")
    ana.add_argument("--report", type=str, required=True)
    ana.add_argument("--k", type=int, default=20)
    ana.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
