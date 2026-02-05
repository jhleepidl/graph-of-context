from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import hashlib
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
from .controller import Controller, RerankController
from .diagnostics import compute_retrieval_diagnostics
from .analysis import analyze_failure_slice
try:
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    _root_src = Path(__file__).resolve().parents[4]
    if str(_root_src) not in sys.path:
        sys.path.append(str(_root_src))
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
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


def _resolve_rerank_weights_path(base_dir: Path, path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        return path if path.is_absolute() else base_dir / path
    return base_dir / "runs" / "controller" / "weights.json"


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


def normalize_prediction_schema(
    prediction: Dict[str, Any],
) -> tuple[Dict[str, Any], str | None, str | None, str | None]:
    allowed = {"allow", "deny", "require_condition", "needs_more_info"}
    decision_before = prediction.get("decision")
    conditions = prediction.get("conditions")
    if conditions is None:
        conditions = prediction.get("required_conditions") or []
    if not isinstance(conditions, list):
        conditions = []
    decision = decision_before if decision_before in allowed else "needs_more_info"
    normalize_reason = None
    if conditions and decision in {"allow", "deny", "needs_more_info"}:
        decision = "require_condition"
        normalize_reason = "reqconds_nonempty_force_require_condition"
    elif decision == "require_condition" and not conditions:
        decision = "needs_more_info"
        normalize_reason = "reqconds_empty_force_needs_more_info"
        if not prediction.get("customer_message"):
            prediction["customer_message"] = "Need required conditions."
    prediction["decision"] = decision
    prediction["conditions"] = conditions
    decision_after = decision
    return prediction, normalize_reason, decision_before, decision_after


def _evaluate_method(
    method: str,
    world: Any,
    tasks: List[Any],
    args: argparse.Namespace,
    client: LLMClient,
    run_dir: Path,
    run_id: str | None = None,
    controller: Controller | RerankController | None = None,
    controller_mode: str = "off",
    controller_policy: str = "bandit",
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
        scenario_mode = getattr(task, "scenario_mode", getattr(args, "scenario_mode", "v0"))
        search_score_mode = getattr(args, "search_score_mode", "bm25_plus_bridge_bonus")
        bridge_bonus = float(getattr(args, "bridge_bonus", 1.5))
        env = PolicyOpsEnv(
            world,
            tool_call_budget=task.budgets.get("tool_call_budget", 50),
            open_budget=task.budgets.get("open_budget", 5),
            search_score_mode=search_score_mode,
            bridge_bonus=bridge_bonus,
        )
        error: str | None = None
        raw_output: str | None = None
        controller_action: str | None = None
        diag: Dict[str, Any] = {}
        oracle_meta: Dict[str, Any] = {}
        goc_graph: GoCGraph | None = None
        goc_graph_path: Path | None = None
        goc_graph_task_path: Path | None = None
        goc_graph_dot_path: Path | None = None
        log_graph = False
        if method == "goc" and args.save_goc_graph:
            if args.goc_graph_sample_rate >= 1.0:
                log_graph = True
            elif args.goc_graph_sample_rate <= 0.0:
                log_graph = False
            else:
                digest = hashlib.md5(task.task_id.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
                log_graph = bucket <= float(args.goc_graph_sample_rate)
        if log_graph:
            goc_graph = GoCGraph(goc_graph_version=args.goc_graph_schema)
            if args.goc_graph_dir and args.goc_graph_dir.endswith(".jsonl"):
                goc_graph_path = Path(args.goc_graph_dir)
                goc_graph_task_path = run_dir / "graphs" / f"{task.task_id}.jsonl"
            else:
                goc_graph_task_path = (
                    Path(args.goc_graph_dir)
                    if args.goc_graph_dir
                    else (run_dir / "graphs")
                ) / f"{task.task_id}.jsonl"
                goc_graph_path = goc_graph_task_path
        agent_query_policy = getattr(args, "agent_query_policy", "single_hop")
        if method == "topk":
            pred, opened_ids, prompt, raw_output, diag = run_topk_rag(
                task,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
        elif method == "full":
            pred, opened_ids, prompt, raw_output, diag = run_full_history(
                task,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
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
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
                agent_query_policy=agent_query_policy,
                controller_policy=controller_policy,
            )
            if controller_action:
                controller_actions[controller_action] = controller_actions.get(controller_action, 0) + 1
        elif method == "oracle":
            pred, opened_ids, prompt, raw_output, oracle_meta = run_oracle(
                task,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
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

        if method == "engine":
            normalize_reason = None
            decision_before_norm = pred.get("decision")
            decision_after_norm = pred.get("decision")
        else:
            pred, normalize_reason, decision_before_norm, decision_after_norm = normalize_prediction_schema(pred)
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
        bridge_clause_id = getattr(task, "bridge_clause_id", None)
        bridge_found = bool(bridge_clause_id and bridge_clause_id in opened_ids)
        update_found_when_needed = False
        if getattr(task, "needs_update_resolution", False):
            update_found_when_needed = any(
                world.clauses.get(cid) and world.clauses[cid].kind == "update" for cid in opened_ids
            )
        winning_rank_exists = retrieval_diag.get("winning_clause_rank") is not None
        action_reward = None
        action_opened_has_winning = None
        action_opened_gold_coverage = None
        reward_breakdown = None
        if method == "goc" and controller_mode == "train":
            winning_clause = task.gold.gold_evidence[0] if task.gold.gold_evidence else None
            action_opened_has_winning = (
                bool(winning_clause and winning_clause in opened_ids)
                if opened_ids
                else False
            )
            action_opened_gold_coverage = retrieval_diag.get("opened_gold_coverage")
            r_win = 1.0 if action_opened_has_winning else 0.0
            r_cov = 0.5 * float(action_opened_gold_coverage or 0.0)
            r_open_penalty = -0.01 * float(env.open_count)
            action_reward = r_win + r_cov + r_open_penalty
            reward_breakdown = {
                "r_win": r_win,
                "r_cov": r_cov,
                "r_open_penalty": r_open_penalty,
                "r_total": action_reward,
            }
        record: Dict[str, Any] = {
            "task_id": task.task_id,
            "method": method,
            "opened_clause_ids": opened_ids,
            "tool_calls": env.tool_call_count,
            "open_calls": env.open_count,
            "open_budget": env.open_budget,
            "prompt_tokens": prompt_tokens,
            "pred_decision": pred_decision,
            "decision_before_normalize": decision_before_norm,
            "decision_after_normalize": decision_after_norm,
            "normalize_reason": normalize_reason,
            "gold_decision": task.gold.decision,
            "decision_correct": pred_decision == task.gold.decision,
            "scenario_mode": scenario_mode,
            "slot_hint_alias": getattr(task, "slot_hint_alias", None),
            "canonical_slot_term": getattr(task, "canonical_slot_term", None),
            "bridge_clause_id": bridge_clause_id,
            "needs_update_resolution": getattr(task, "needs_update_resolution", False),
            "bridge_found": bridge_found,
            "canonical_used_in_query2": bool(diag.get("canonical_used_in_query2"))
            if isinstance(diag, dict)
            else False,
            "update_found_when_needed": update_found_when_needed,
            "winning_rank_exists": winning_rank_exists,
            "evidence_precision": task_metrics.get("evidence_precision"),
            "evidence_recall": task_metrics.get("evidence_recall"),
            "critical_evidence_hit": task_metrics.get("critical_evidence_hit"),
            "pred_evidence_count": len(pred_for_record.get("evidence", []) or []),
            "gold_evidence_count": len(task.gold.gold_evidence or []),
            "gold_evidence_ids": list(task.gold.gold_evidence or []),
            "evidence_before_pad": evidence_before,
            "evidence_after_pad": evidence_after,
            "error": error,
            "evidence_padding_mode": args.evidence_padding_mode,
            "min_evidence_count": args.min_evidence_count,
            "controller_action": controller_action,
            "controller_action_reward": action_reward,
            "controller_action_opened_gold_coverage": action_opened_gold_coverage,
            "controller_action_opened_has_winning_clause": action_opened_has_winning,
            "controller_reward_breakdown": reward_breakdown,
        }
        if isinstance(diag, dict):
            record.update(diag)
        if not args.save_search_snapshot:
            record.pop("rewrite_queries", None)
            record.pop("rewrite_used", None)
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
        record["goc_graph_jsonl_path"] = str(goc_graph_task_path) if goc_graph_task_path else None
        record["goc_graph_dot_path"] = str(goc_graph_dot_path) if goc_graph_dot_path else None
        records.append(record)

        if goc_graph and goc_graph_path:
            rid = run_id or run_dir.name
            log_events = "events" in args.goc_graph_mode
            last_event_node: str | None = None
            def _log(event_type: str, payload: Dict[str, Any]) -> None:
                if not log_events:
                    return
                append_event(
                    goc_graph_path,
                    build_event(
                        rid,
                        task.task_id,
                        method,
                        goc_graph.step,
                        event_type,
                        payload,
                    ),
                )
            def _add_event_node(node_id: str, node_type: str, **attrs: Any) -> None:
                nonlocal last_event_node
                if args.goc_graph_schema != "v1":
                    return
                goc_graph.add_node(node_id, node_type, step=goc_graph.step, **attrs)
                if last_event_node:
                    goc_graph.add_edge(
                        f"next:{last_event_node}:{node_id}",
                        last_event_node,
                        node_id,
                        "next",
                    )
                last_event_node = node_id
            episode_id = f"episode:{task.task_id}"
            ticket_id = f"ticket:{task.task_id}"
            goc_graph.add_node(episode_id, "episode", step=goc_graph.step)
            goc_graph.add_node(ticket_id, "ticket", text=task.user_ticket, step=goc_graph.step)
            _log("INIT", {"episode_id": episode_id, "ticket_id": ticket_id})
            goc_graph.step += 1

            gold_node = f"gold:{task.task_id}"
            goc_graph.add_node(gold_node, "gold", decision=task.gold.decision, step=goc_graph.step)
            for cid in task.gold.gold_evidence or []:
                doc_id = f"doc:{cid}"
                goc_graph.add_node(doc_id, "doc_ref", clause_id=cid)
                goc_graph.add_edge(f"gold:{task.task_id}:{cid}", gold_node, doc_id, "gold_evidence")

            primary_results = diag.get("primary_search_results", [])
            rewrite_queries = diag.get("rewrite_queries") or [task.user_ticket]
            first_seen: Dict[str, Dict[str, Any]] = {}
            stages = ["primary", "rewrite1", "rewrite2"]
            rerank_scores: Dict[str, float] = {}
            if controller_policy == "rerank" and isinstance(controller, RerankController):
                secondary_results = diag.get("secondary_search_results", [])
                rerank_scores = controller.score_candidates(
                    primary_results,
                    secondary_results,
                    world,
                    task.context,
                )
            variants = []
            if isinstance(diag, dict) and diag.get("primary_search_variants"):
                variants = list(diag.get("primary_search_variants") or [])
            else:
                for idx, query in enumerate(rewrite_queries[:3]):
                    stage = stages[idx]
                    results = primary_results if idx == 0 else []
                    variants.append({"stage": stage, "query": query, "results": results})

            for variant in variants:
                stage = variant.get("stage") or "primary"
                query = variant.get("query") or ""
                qid = f"query:{task.task_id}:{stage}"
                query_meta = variant.get("query_meta") or {}
                node_attrs = {"text": query, "step": goc_graph.step, "stage": stage}
                if isinstance(query_meta, dict):
                    node_attrs.update(query_meta)
                goc_graph.add_node(qid, "query", **node_attrs)
                tool_call_id = f"tool_call:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(tool_call_id, "tool_call", tool="search", query_id=qid)
                depends_on = variant.get("depends_on")
                if depends_on:
                    goc_graph.add_edge(
                        f"depends:{qid}:{depends_on}",
                        qid,
                        depends_on,
                        "depends_on",
                    )
                results = variant.get("results") or []
                retrieved = []
                for rank, item in enumerate(results[: args.search_snapshot_k], start=1):
                    cid = item.get("clause_id")
                    if not cid:
                        continue
                    doc_id = f"doc:{cid}"
                    clause = world.clauses.get(cid)
                    node_attrs = {
                        "clause_id": cid,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "first_seen_query_id": qid,
                        "first_seen_rank": rank,
                        "first_seen_score": item.get("score"),
                        "step": goc_graph.step,
                    }
                    if args.goc_graph_include_clause_text and cid in world.clauses:
                        node_attrs["text"] = world.clauses[cid].text[:2000]
                    goc_graph.add_node(doc_id, "doc_ref", **node_attrs)
                    if cid not in first_seen:
                        first_seen[cid] = {
                            "query_id": qid,
                            "rank": rank,
                            "score": item.get("score"),
                        }
                    edge_id = f"retrieved:{task.task_id}:{stage}:{cid}"
                    goc_graph.add_edge(
                        edge_id,
                        qid,
                        doc_id,
                        "retrieved",
                        rank=rank,
                        score=item.get("score"),
                        source=stage,
                    )
                    payload = {
                        "doc_id": doc_id,
                        "clause_id": cid,
                        "rank": rank,
                        "score": item.get("score"),
                        "source": stage,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "snippet": (clause.text[:200] if clause else ""),
                    }
                    for key in [
                        "in_base_topk",
                        "base_rank",
                        "base_score",
                        "in_struct_topk",
                        "struct_rank",
                        "struct_score",
                        "merge_score",
                    ]:
                        if key in item:
                            payload[key] = item.get(key)
                    retrieved.append(payload)
                obs_id = f"observation:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    obs_id,
                    "observation",
                    observation_type="search_results_summary",
                    text=f"{len(retrieved)} results",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("SEARCH", {"query_id": qid, "query": query, "results": retrieved})
                goc_graph.step += 1

            secondary_results = diag.get("secondary_search_results", [])
            if secondary_results:
                qid = f"query:{task.task_id}:secondary"
                query = diag.get("secondary_search_query", "")
                goc_graph.add_node(qid, "query", text=query, step=goc_graph.step)
                tool_call_id = f"tool_call:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(tool_call_id, "tool_call", tool="search", query_id=qid)
                retrieved = []
                for rank, item in enumerate(secondary_results[: args.search_snapshot_k], start=1):
                    cid = item.get("clause_id")
                    if not cid:
                        continue
                    doc_id = f"doc:{cid}"
                    clause = world.clauses.get(cid)
                    node_attrs = {
                        "clause_id": cid,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "step": goc_graph.step,
                    }
                    if args.goc_graph_include_clause_text and cid in world.clauses:
                        node_attrs["text"] = world.clauses[cid].text[:2000]
                    goc_graph.add_node(doc_id, "doc_ref", **node_attrs)
                    if cid not in first_seen:
                        first_seen[cid] = {
                            "query_id": qid,
                            "rank": rank,
                            "score": item.get("score"),
                        }
                    edge_id = f"retrieved:{task.task_id}:secondary:{cid}"
                    goc_graph.add_edge(
                        edge_id,
                        qid,
                        doc_id,
                        "retrieved",
                        rank=rank,
                        score=item.get("score"),
                        source="secondary",
                    )
                    retrieved.append(
                        {
                            "doc_id": doc_id,
                            "clause_id": cid,
                            "rank": rank,
                            "score": item.get("score"),
                            "source": "secondary",
                            "kind": getattr(clause, "kind", None),
                            "slot": getattr(clause, "slot", None),
                            "published_at": getattr(clause, "published_at", None),
                            "authority": getattr(clause, "authority", None),
                            "snippet": (clause.text[:200] if clause else ""),
                        }
                    )
                obs_id = f"observation:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    obs_id,
                    "observation",
                    observation_type="search_results_summary",
                    text=f"{len(retrieved)} results",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("SEARCH", {"query_id": qid, "query": query, "results": retrieved})
                goc_graph.step += 1

            forced = set(diag.get("forced_open_ids") or [])
            for idx, cid in enumerate(opened_ids, start=1):
                selected_by = "forced_open" if cid in forced else "heuristic_fallback"
                if controller_policy == "rerank":
                    selected_by = "controller_rerank"
                elif controller_action:
                    selected_by = "controller_bandit"
                reason = selected_by
                bm25_meta = first_seen.get(cid, {})
                rerank_score = rerank_scores.get(cid) if rerank_scores else None
                controller_context_key = None
                if controller_action and hasattr(controller, "_context_key"):
                    try:
                        search_stats = controller.compute_search_stats(primary_results)  # type: ignore[attr-defined]
                        features = controller.build_context_features(task.context, env.open_budget, search_stats)  # type: ignore[attr-defined]
                        controller_context_key = controller._context_key(features)  # type: ignore[attr-defined]
                    except Exception:
                        controller_context_key = None
                tool_call_id = f"tool_call:open:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    tool_call_id,
                    "tool_call",
                    tool="open",
                    clause_id=cid,
                )
                if args.goc_graph_schema == "v1" and cid in world.clauses:
                    obs_id = f"observation:open:{task.task_id}:{goc_graph.step}"
                    _add_event_node(
                        obs_id,
                        "observation",
                        observation_type="opened_clause_excerpt",
                        text=world.clauses[cid].text[:200],
                    )
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                goc_graph.add_edge(
                    f"opened:{task.task_id}:{cid}",
                    episode_id,
                    f"doc:{cid}",
                    "opened",
                    open_step=idx,
                    reason=reason,
                    selected_by=selected_by,
                    from_query_id=bm25_meta.get("query_id"),
                    bm25_rank=bm25_meta.get("rank"),
                    bm25_score=bm25_meta.get("score"),
                    rerank_score=rerank_score,
                    controller_policy=controller_policy,
                    controller_mode=controller_mode,
                    controller_action=controller_action,
                    controller_context_key=controller_context_key,
                )
                _log(
                    "OPEN",
                    {
                        "clause_id": cid,
                        "reason": reason,
                        "open_index": idx,
                        "selected_by": selected_by,
                        "from_query_id": bm25_meta.get("query_id"),
                        "bm25_rank": bm25_meta.get("rank"),
                        "bm25_score": bm25_meta.get("score"),
                        "rerank_score": rerank_score,
                        "controller_policy": controller_policy,
                        "controller_mode": controller_mode,
                        "controller_action": controller_action,
                        "controller_context_key": controller_context_key,
                    },
                )
                goc_graph.step += 1

            prompt_id = f"prompt:{task.task_id}:{goc_graph.step}"
            controller_info = {"controller_policy": controller_policy, "controller_action": controller_action}
            if rerank_scores:
                top_candidates = []
                items = sorted(primary_results, key=lambda it: it.get("score", 0.0), reverse=True)[:5]
                dates = [world.clauses[it["clause_id"]].published_at for it in items if it.get("clause_id") in world.clauses]
                min_date = min(dates) if dates else None
                max_date = max(dates) if dates else None
                for item in items:
                    cid = item.get("clause_id")
                    clause = world.clauses.get(cid) if cid else None
                    recency_score = 0.0
                    if clause and clause.published_at and min_date and max_date and max_date != min_date:
                        recency_score = (clause.published_at > min_date) * 1.0
                    text = clause.text.lower() if clause else ""
                    top_candidates.append(
                        {
                            "clause_id": cid,
                            "total_score": rerank_scores.get(cid),
                            "bm25_score": item.get("score"),
                            "recency_score": recency_score,
                            "slot_match": 1.0 if clause and clause.slot == task.context.get("slot") else 0.0,
                            "is_update": bool(clause and clause.kind == "update"),
                            "is_exception": bool(clause and clause.kind == "exception"),
                            "is_definition": bool(clause and clause.kind == "definition"),
                            "has_conditions_kw": any(k in text for k in ["conditions:", "must", "require", "provided that"]),
                            "kind": clause.kind if clause else None,
                            "slot": clause.slot if clause else None,
                            "published_at": clause.published_at if clause else None,
                        }
                    )
                controller_info["top_candidates"] = top_candidates
            controller_node_id = f"controller:{task.task_id}"
            goc_graph.add_node(controller_node_id, "controller", **controller_info, step=goc_graph.step)
            goc_graph.add_edge(
                f"controls:{task.task_id}",
                episode_id,
                controller_node_id,
                "controller",
            )
            _add_event_node(
                f"summary:{task.task_id}:{goc_graph.step}",
                "summary",
                text="prompt_materialization",
            )
            prompt_payload = goc_graph.materialize_prompt(
                prompt_id,
                [f"doc:{cid}" for cid in opened_ids],
                budget_info={"open_budget": env.open_budget},
                controller_info=controller_info,
            )
            _log("PROMPT", prompt_payload)
            for node_id in prompt_payload.get("included_node_ids", []):
                goc_graph.add_edge(
                    f"selected:{prompt_id}:{node_id}",
                    episode_id,
                    node_id,
                    "selected_for_prompt",
                    prompt_id=prompt_id,
                    created_step=goc_graph.step,
                    budget_info=prompt_payload.get("budget_info"),
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"summarized:{prompt_id}:{node_id}",
                        f"summary:{task.task_id}:{goc_graph.step}",
                        node_id,
                        "summarized",
                    )
            goc_graph.step += 1

            if record.get("raw_path"):
                tool_call_id = f"tool_call:llm:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    tool_call_id,
                    "tool_call",
                    tool="llm_generate",
                )
                if args.goc_graph_schema == "v1":
                    obs_id = f"observation:llm:{task.task_id}:{goc_graph.step}"
                    _add_event_node(
                        obs_id,
                        "observation",
                        observation_type="llm_raw",
                        text="raw_saved",
                    )
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("LLM_RAW", {"raw_path": record.get("raw_path")})
                goc_graph.step += 1

            answer_id = f"answer:{task.task_id}"
            goc_graph.add_node(answer_id, "answer", decision=pred_for_record.get("decision"), step=goc_graph.step)
            for cid in evidence_before:
                goc_graph.add_edge(
                    f"cites_raw:{task.task_id}:{cid}",
                    answer_id,
                    f"doc:{cid}",
                    "cites_evidence_raw",
                )
            for cid in evidence_after:
                goc_graph.add_edge(
                    f"cites:{task.task_id}:{cid}",
                    answer_id,
                    f"doc:{cid}",
                    "cites_evidence",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{answer_id}:{cid}",
                        answer_id,
                        f"doc:{cid}",
                        "depends_on",
                    )
            _log(
                "PREDICTION",
                goc_graph.record_prediction(
                    pred_for_record.get("decision", ""),
                    pred_for_record.get("conditions", []),
                    evidence_before,
                    evidence_after,
                    parse_error=error,
                    raw_path=record.get("raw_path"),
                    prompt_path=record.get("prompt_path"),
                ),
            )
            goc_graph.step += 1

            _log("DONE", {})
            goc_graph.step += 1

            if "final" in args.goc_graph_mode:
                snapshot_payload = goc_graph.to_snapshot_dict()
                append_event(
                    goc_graph_path,
                    build_event(
                        rid,
                        task.task_id,
                        method,
                        goc_graph.step,
                        "SNAPSHOT",
                        snapshot_payload,
                    ),
                )
                goc_graph.step += 1
            if goc_graph_task_path and goc_graph_path != goc_graph_task_path:
                goc_graph_task_path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for line in goc_graph_path.read_text(encoding="utf-8").splitlines():
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if data.get("task_id") == task.task_id:
                        lines.append(line)
                goc_graph_task_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if args.save_goc_dot and goc_graph_task_path:
                from goc_logger.export import export_dot

                goc_graph_dot_path = goc_graph_task_path.with_suffix(".dot")
                export_dot(goc_graph_task_path, task.task_id, goc_graph_dot_path)
                record["goc_graph_dot_path"] = str(goc_graph_dot_path)
            if goc_graph_task_path:
                record["goc_graph_jsonl_path"] = str(goc_graph_task_path)

    aggregate = aggregate_metrics(metrics)
    aggregate["gold_decision_distribution"] = gold_decision_distribution(tasks)
    bridge_records = [r for r in records if r.get("bridge_clause_id")]
    if bridge_records:
        aggregate["bridge_found_rate"] = sum(1 for r in bridge_records if r.get("bridge_found")) / len(
            bridge_records
        )
    else:
        aggregate["bridge_found_rate"] = 0.0
    two_hop_records = [r for r in records if r.get("agent_query_policy") == "two_hop_bridge"]
    if two_hop_records:
        aggregate["canonical_used_in_query2_rate"] = sum(
            1 for r in two_hop_records if r.get("canonical_used_in_query2")
        ) / len(two_hop_records)
    else:
        aggregate["canonical_used_in_query2_rate"] = 0.0
    update_needed_records = [r for r in records if r.get("needs_update_resolution")]
    if update_needed_records:
        aggregate["update_found_when_needed_rate"] = sum(
            1 for r in update_needed_records if r.get("update_found_when_needed")
        ) / len(update_needed_records)
    else:
        aggregate["update_found_when_needed_rate"] = 0.0
    aggregate["winning_rank_exists_rate"] = sum(
        1 for r in records if r.get("winning_rank_exists")
    ) / len(records) if records else 0.0
    records_with_rank = [r for r in records if r.get("winning_rank_exists")]
    if records_with_rank:
        aggregate["decision_accuracy_when_winning_rank_exists"] = sum(
            1 for r in records_with_rank if r.get("decision_correct")
        ) / len(records_with_rank)
    else:
        aggregate["decision_accuracy_when_winning_rank_exists"] = 0.0
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
        "controller_policy": controller_policy,
        "controller_actions_distribution": controller_actions,
        "controller_action_reward_mean": action_reward_mean,
        "controller_action_counts": action_counts,
        "controller_action_opened_gold_coverage_mean": action_cov_mean,
        "controller_action_opened_has_winning_clause_rate": action_winning_rate,
    }
    if controller_policy == "rerank" and isinstance(controller, RerankController):
        report["rerank_weights"] = dict(controller.weights)
        report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
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
        scenario_mode=args.scenario_mode,
        bridge_prob=args.bridge_prob,
        alias_density=args.alias_density,
        canonical_density=args.canonical_density,
        bridge_kind=args.bridge_kind,
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

    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller and args.method == "goc":
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="rerank",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="bandit",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
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
        run_id=timestamp,
        controller=controller,
        controller_mode=controller_mode,
        controller_policy=args.controller_policy,
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
    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="rerank",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="bandit",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
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
            run_id=timestamp,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            controller_policy=args.controller_policy,
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
        opened_has_win_vals = [
            1.0 if r.get("opened_has_winning_clause") else 0.0
            for r in records
            if r.get("opened_has_winning_clause") is not None
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
            "bridge_found_rate": method_report["metrics"].get("bridge_found_rate"),
            "canonical_used_in_query2_rate": method_report["metrics"].get("canonical_used_in_query2_rate"),
            "update_found_when_needed_rate": method_report["metrics"].get("update_found_when_needed_rate"),
            "winning_rank_exists_rate": method_report["metrics"].get("winning_rank_exists_rate"),
            "decision_accuracy_when_winning_rank_exists": method_report["metrics"].get(
                "decision_accuracy_when_winning_rank_exists"
            ),
            "prompt_tokens": _avg_p90(prompt_tokens_vals),
            "open_calls": _avg_p90(open_calls_vals),
            "tool_calls": _avg_p90(tool_calls_vals),
            "opened_gold_coverage_mean": sum(opened_gold_cov_vals) / len(opened_gold_cov_vals)
            if opened_gold_cov_vals
            else 0.0,
            "opened_has_winning_clause_rate": sum(opened_has_win_vals) / len(opened_has_win_vals)
            if opened_has_win_vals
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
        "controller_policy": args.controller_policy,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    if args.controller_policy == "rerank" and isinstance(controller, RerankController):
        compare_report["rerank_weights"] = dict(controller.weights)
        compare_report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

    out_path = compare_dir / f"{timestamp}.json"
    save_report(out_path, compare_report)
    if args.save_goc_graph or args.save_goc_dot:
        readme_path = compare_run_dir / "README_debug.md"
        readme_path.write_text(
            "\n".join(
                [
                    "# Debug Run Commands",
                    "",
                    "## Compare command",
                    f"PYTHONPATH=src/benchmarks/policyops_arena_v0/src:src python -m policyops.run compare "
                    f"--llm {args.llm} --model {args.model} --methods {' '.join(methods)} "
                    f"--save_goc_graph --save_goc_dot",
                    "",
                    "## Triage command",
                    f"PYTHONPATH=src/benchmarks/policyops_arena_v0/src:src python -m policyops.triage "
                    f"--compare_report {out_path} --method goc --max_per_bucket 20",
                    "",
                    "## Bucket checklist",
                    "- A_open_selection_fail: gold in search but not opened",
                    "- B_reasoning_fail: winning clause opened but wrong decision",
                    "- C_retrieval_fail: gold not in search",
                    "- D_decision_confusion: require_condition mispredicted",
                    "- E_budget_edge_fail: winning clause beyond open_budget",
                    "- F_evidence_padding_artifact: evidence_before empty, after non-empty",
                ]
            ),
            encoding="utf-8",
        )
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
    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
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
                    controller_policy="rerank",
                )
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
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
                    controller_policy="bandit",
                )
                controller.save(state_path)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    run_id = output_dir.name
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
            run_id=run_id,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            controller_policy=args.controller_policy,
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
        opened_has_win_vals = [
            1.0 if r.get("opened_has_winning_clause") else 0.0
            for r in records
            if r.get("opened_has_winning_clause") is not None
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
            "opened_has_winning_clause_rate": sum(opened_has_win_vals) / len(opened_has_win_vals)
            if opened_has_win_vals
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
        "controller_policy": args.controller_policy,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    if args.controller_policy == "rerank" and isinstance(controller, RerankController):
        compare_report["rerank_weights"] = dict(controller.weights)
        compare_report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
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


def cmd_ablate_controller(args: argparse.Namespace) -> None:
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"
    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ablate_dir = Path(base_dir) / "runs" / "ablate" / timestamp
    ablate_dir.mkdir(parents=True, exist_ok=True)

    off_args = argparse.Namespace(**vars(args))
    off_args.use_controller = False
    off_args.controller_mode = "off"

    on_args = argparse.Namespace(**vars(args))
    on_args.use_controller = True
    on_args.controller_policy = "rerank"
    on_args.controller_mode = args.controller_mode

    report_off = _run_compare_with_tasks(
        world,
        tasks,
        off_args,
        base_dir=Path(base_dir),
        output_dir=ablate_dir / "off",
    )
    report_on = _run_compare_with_tasks(
        world,
        tasks,
        on_args,
        base_dir=Path(base_dir),
        output_dir=ablate_dir / "on",
    )

    def _get_metric(report: Dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(report.get("summary", {}).get("goc", {}).get(key, default) or 0.0)

    def _get_prompt_avg(report: Dict[str, Any]) -> float:
        prompt = report.get("summary", {}).get("goc", {}).get("prompt_tokens", {})
        return float(prompt.get("avg") or 0.0) if isinstance(prompt, dict) else 0.0

    delta = {
        "decision_accuracy": _get_metric(report_on, "decision_accuracy")
        - _get_metric(report_off, "decision_accuracy"),
        "critical_evidence_hit": _get_metric(report_on, "critical_evidence_hit")
        - _get_metric(report_off, "critical_evidence_hit"),
        "opened_gold_coverage_mean": _get_metric(report_on, "opened_gold_coverage_mean")
        - _get_metric(report_off, "opened_gold_coverage_mean"),
        "gold_in_search_topk_rate": _get_metric(report_on, "gold_in_search_topk_rate")
        - _get_metric(report_off, "gold_in_search_topk_rate"),
        "prompt_tokens_avg": _get_prompt_avg(report_on) - _get_prompt_avg(report_off),
    }

    out_payload = {
        "timestamp": timestamp,
        "controller_policy": "rerank",
        "off_report": report_off,
        "on_report": report_on,
        "delta": delta,
    }
    out_path = ablate_dir / "ablate_summary.json"
    save_report(out_path, out_payload)
    print("Ablation complete.")
    for key, value in delta.items():
        print(f"{key}: {value:.4f}")
    print(f"Ablation report saved to {out_path}")


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
    gen.add_argument("--scenario_mode", choices=["v0", "bridged_v1_1"], default="v0")
    gen.add_argument("--bridge_prob", type=float, default=0.8)
    gen.add_argument("--alias_density", type=float, default=0.9)
    gen.add_argument("--canonical_density", type=float, default=0.95)
    gen.add_argument("--bridge_kind", choices=["definition", "glossary"], default="definition")
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
    ev.add_argument("--save_goc_graph", action="store_true")
    ev.add_argument("--save_goc_dot", action="store_true")
    ev.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    ev.add_argument("--goc_graph_include_clause_text", action="store_true")
    ev.add_argument("--goc_graph_dir", type=str, default="")
    ev.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    ev.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    ev.add_argument("--scenario_mode", choices=["v0", "bridged_v1_1"], default="v0")
    ev.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    ev.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    ev.add_argument("--bridge_bonus", type=float, default=1.5)
    ev.add_argument("--use_query_rewrite", action="store_true", default=True)
    ev.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    ev.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    ev.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    ev.add_argument("--rewrite_queries", type=int, default=3)
    ev.add_argument("--force_open_top_n", type=int, default=1)
    ev.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    ev.add_argument("--use_controller", action="store_true")
    ev.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    ev.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    ev.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    ev.add_argument("--controller_weights_path", type=str, default="")
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
    cmp.add_argument("--save_goc_graph", action="store_true")
    cmp.add_argument("--save_goc_dot", action="store_true")
    cmp.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    cmp.add_argument("--goc_graph_include_clause_text", action="store_true")
    cmp.add_argument("--goc_graph_dir", type=str, default="")
    cmp.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    cmp.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    cmp.add_argument("--scenario_mode", choices=["v0", "bridged_v1_1"], default="v0")
    cmp.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    cmp.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    cmp.add_argument("--bridge_bonus", type=float, default=1.5)
    cmp.add_argument("--use_query_rewrite", action="store_true", default=True)
    cmp.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    cmp.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    cmp.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    cmp.add_argument("--rewrite_queries", type=int, default=3)
    cmp.add_argument("--force_open_top_n", type=int, default=1)
    cmp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    cmp.add_argument("--use_controller", action="store_true")
    cmp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    cmp.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    cmp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    cmp.add_argument("--controller_weights_path", type=str, default="")
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
    swp.add_argument("--scenario_mode", choices=["v0", "bridged_v1_1"], default="v0")
    swp.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    swp.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    swp.add_argument("--bridge_bonus", type=float, default=1.5)
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
    swp.add_argument("--save_goc_graph", action="store_true")
    swp.add_argument("--save_goc_dot", action="store_true")
    swp.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    swp.add_argument("--goc_graph_include_clause_text", action="store_true")
    swp.add_argument("--goc_graph_dir", type=str, default="")
    swp.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    swp.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    swp.add_argument("--use_query_rewrite", action="store_true", default=True)
    swp.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    swp.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    swp.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    swp.add_argument("--rewrite_queries", type=int, default=3)
    swp.add_argument("--force_open_top_n", type=int, default=1)
    swp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    swp.add_argument("--use_controller", action="store_true")
    swp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    swp.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    swp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    swp.add_argument("--controller_weights_path", type=str, default="")
    swp.add_argument("--task_split", choices=["none", "holdout"], default="none")
    swp.add_argument("--train_ratio", type=float, default=0.7)
    swp.add_argument("--split_seed", type=int, default=0)
    swp.add_argument("--reuse_data", action="store_true")
    swp.set_defaults(func=cmd_sweep)

    ana = sub.add_parser("analyze", help="Analyze compare report failure slices")
    ana.add_argument("--report", type=str, required=True)
    ana.add_argument("--k", type=int, default=20)
    ana.set_defaults(func=cmd_analyze)

    abl = sub.add_parser("ablate_controller", help="Compare controller off vs rerank on")
    abl.add_argument("--methods", nargs="+", default=["goc"])
    abl.add_argument("--model", type=str, default="gpt-4.1-mini")
    abl.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    abl.add_argument("--dotenv", type=str, default=".env")
    abl.add_argument("--out_dir", type=Path, default=None)
    abl.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
    )
    abl.add_argument("--min_evidence_count", type=int, default=2)
    abl.add_argument("--save_prompts", action="store_true")
    abl.add_argument("--save_raw", action="store_true")
    abl.add_argument("--save_search_snapshot", action="store_true")
    abl.add_argument("--search_snapshot_k", type=int, default=20)
    abl.add_argument("--primary_search_top_k", type=int, default=20)
    abl.add_argument("--save_goc_graph", action="store_true")
    abl.add_argument("--save_goc_dot", action="store_true")
    abl.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    abl.add_argument("--goc_graph_include_clause_text", action="store_true")
    abl.add_argument("--goc_graph_dir", type=str, default="")
    abl.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    abl.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    abl.add_argument("--use_query_rewrite", action="store_true", default=True)
    abl.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    abl.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    abl.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    abl.add_argument("--rewrite_queries", type=int, default=3)
    abl.add_argument("--force_open_top_n", type=int, default=1)
    abl.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    abl.add_argument("--controller_mode", choices=["off", "eval", "train"], default="train")
    abl.add_argument("--controller_policy", choices=["bandit", "rerank"], default="rerank")
    abl.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    abl.add_argument("--controller_weights_path", type=str, default="")
    abl.add_argument("--task_split", choices=["none", "holdout"], default="holdout")
    abl.add_argument("--train_ratio", type=float, default=0.7)
    abl.add_argument("--split_seed", type=int, default=0)
    abl.set_defaults(func=cmd_ablate_controller)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
