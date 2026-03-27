from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import TraceStep, TraceThread, TraceWorldClause
from .evaluator import (
    _arg_get,
    _choose_traceops_context,
    _infer_prediction,
    _score_step,
    _tokenize,
    _unique_strs,
)


ACTION_NAMES = ("none", "unfold", "fork", "unfold_then_fork")


@dataclass
class ActionEval:
    action: str
    context_ids: List[str]
    debug: Dict[str, Any]
    prediction: Dict[str, Any]
    score: Dict[str, Any]
    stats: Dict[str, Any]
    utility: float


@dataclass
class PivotEval:
    thread_id: str
    step_id: str
    step_idx: int
    split: str
    features: Dict[str, Any]
    actions: Dict[str, ActionEval]
    best_action: str
    best_utility: float


def _hash_to_unit(text: str) -> float:
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return float(int(digest[:16], 16)) / float(16 ** 16)


def assign_split(thread_id: str, *, dev_ratio: float = 0.5, seed: int = 7) -> str:
    frac = _hash_to_unit(f"{seed}:{thread_id}")
    return "dev" if frac < float(dev_ratio) else "test"


def _clause_token_cost(clause: Optional[TraceWorldClause]) -> int:
    if clause is None:
        return 1
    return max(1, len(_tokenize(clause.text)))


def estimate_tokens_for_ids(thread: TraceThread, clause_ids: Sequence[str]) -> int:
    return int(sum(_clause_token_cost(thread.clauses.get(cid)) for cid in _unique_strs(clause_ids)))


def _ids_within_window(thread: TraceThread, clause_ids: Sequence[str], *, step_idx: int, window: int) -> List[str]:
    out: List[str] = []
    low = int(step_idx) - int(window)
    for cid in _unique_strs(clause_ids):
        clause = thread.clauses.get(cid)
        if clause is None:
            continue
        if low <= int(clause.step_idx) < int(step_idx):
            out.append(cid)
    return out


def summarize_context(thread: TraceThread, step: TraceStep, clause_ids: Sequence[str]) -> Dict[str, Any]:
    ids = _unique_strs(clause_ids)
    type_counts: Dict[str, int] = {}
    step_dists: List[int] = []
    dep_edge_count = 0
    query_tokens = _tokenize(step.message)
    overlap_scores: List[int] = []
    applicable_count = 0
    for cid in ids:
        clause = thread.clauses.get(cid)
        if clause is None:
            continue
        ntype = str(clause.node_type or "")
        type_counts[ntype] = int(type_counts.get(ntype, 0) + 1)
        step_dists.append(max(0, int(step.step_idx) - int(clause.step_idx)))
        dep_edge_count += len(list(getattr(clause, "depends_on", []) or []))
        overlap_scores.append(len(query_tokens & _tokenize(clause.text)))
        # import lazily to avoid export churn in evaluator
        from .evaluator import _clause_applicable  # local import

        if _clause_applicable(clause, step.state):
            applicable_count += 1
    token_est = estimate_tokens_for_ids(thread, ids)
    recent_ids = _ids_within_window(thread, ids, step_idx=int(step.step_idx), window=6)
    recent_ratio = float(len(recent_ids)) / float(max(1, len(ids)))
    avg_step_distance = float(sum(step_dists) / len(step_dists)) if step_dists else 0.0
    max_step_distance = int(max(step_dists)) if step_dists else 0
    mean_query_overlap = float(sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0
    return {
        "context_count": int(len(ids)),
        "token_est": int(token_est),
        "recent_context_count_w6": int(len(recent_ids)),
        "recent_context_ratio_w6": float(recent_ratio),
        "avg_step_distance": float(avg_step_distance) if math.isfinite(avg_step_distance) else 0.0,
        "max_step_distance": int(max_step_distance),
        "dep_edge_count": int(dep_edge_count),
        "applicable_ratio": float(applicable_count) / float(max(1, len(ids))),
        "mean_query_overlap": float(mean_query_overlap),
        "decision_count": int(type_counts.get("DECISION", 0)),
        "update_count": int(type_counts.get("UPDATE", 0)),
        "exception_count": int(type_counts.get("EXCEPTION", 0)),
        "assumption_count": int(type_counts.get("ASSUMPTION", 0)),
        "evidence_count": int(type_counts.get("EVIDENCE", 0)),
        "type_counts": type_counts,
    }


def build_dep_scoped_fork_from_base_context(
    *,
    thread: TraceThread,
    step: TraceStep,
    ordered_history: Sequence[str],
    base_context_ids: Sequence[str],
    args: Any,
    allow_gold_support: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    clauses = thread.clauses
    base_context_ids = _unique_strs(base_context_ids)
    ordered_history = _unique_strs(ordered_history)
    query_tokens = _tokenize(step.message)
    recent_n = max(0, int(_arg_get(args, "fork_recent_active_n", 4) or 4))
    include_recent = bool(_arg_get(args, "fork_include_recent_active", True))
    fork_k = max(1, int(_arg_get(args, "fork_k", 6) or 6))
    budget = max(1, int(_arg_get(args, "fork_max_tokens", 160) or 160))
    hops = max(1, int(_arg_get(args, "fork_dependency_hops", 2) or 2))

    allowed_pool = set(base_context_ids)
    if allow_gold_support:
        allowed_pool |= set(_unique_strs(step.pivot_required_ids or []))
        allowed_pool |= set(_unique_strs((step.gold.evidence_ids if step.gold else []) or []))
    ordered_base = [cid for cid in ordered_history if cid in allowed_pool]

    step_meta = step.metadata if isinstance(step.metadata, dict) else {}
    pivot_required = _unique_strs(step.pivot_required_ids or [])
    checkpoint_ids = _unique_strs(step_meta.get("trap_decision_checkpoint_ids") or [])
    seed_pool = _unique_strs(pivot_required + checkpoint_ids)
    seed_pool = [cid for cid in seed_pool if cid in allowed_pool]
    if not seed_pool:
        scored: List[Tuple[int, int, str]] = []
        for idx, cid in enumerate(ordered_base):
            clause = clauses.get(cid)
            if clause is None:
                continue
            score = len(query_tokens & _tokenize(clause.text))
            scored.append((score, -idx, cid))
        scored.sort(reverse=True)
        seed_pool = [cid for _, _, cid in scored[:fork_k]]

    closure: set[str] = set()
    frontier = list(seed_pool)
    for _ in range(hops + 1):
        if not frontier:
            break
        nxt: List[str] = []
        for cid in frontier:
            if cid in closure or cid not in allowed_pool:
                continue
            closure.add(cid)
            clause = clauses.get(cid)
            if clause is None:
                continue
            for dep in _unique_strs(getattr(clause, "depends_on", []) or []):
                if dep in allowed_pool and dep not in closure:
                    nxt.append(dep)
            for cand in ordered_base:
                if cand in closure:
                    continue
                cand_clause = clauses.get(cand)
                if cand_clause is None:
                    continue
                if cid in _unique_strs(getattr(cand_clause, "depends_on", []) or []):
                    nxt.append(cand)
        frontier = nxt

    selected = list(closure)
    if include_recent and recent_n > 0:
        selected.extend(ordered_base[-recent_n:])
    scoped_ids = [cid for cid in ordered_base if cid in set(selected)]

    kept: List[str] = []
    used = 0
    for cid in _unique_strs(scoped_ids):
        add = _clause_token_cost(clauses.get(cid))
        if kept and used + add > budget:
            continue
        kept.append(cid)
        used += add

    debug = {
        "fork_enabled": True,
        "fork_scope_mode": "dep_direct",
        "fork_context_count": int(len(kept)),
        "fork_base_context_count": int(len(base_context_ids)),
        "fork_budget_tokens": int(budget),
        "fork_seed_pool": list(seed_pool),
        "fork_used_gold_support_pool": bool(allow_gold_support),
    }
    return kept, debug


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _compute_utility(
    *,
    score: Mapping[str, Any],
    stats: Mapping[str, Any],
    full_history_tokens: int,
    token_weight: float,
    coverage_weight: float,
    leakage_weight: float,
) -> float:
    answer = 1.0 if bool(score.get("answer_correct", False)) else 0.0
    critical = _safe_float(score.get("critical_coverage"), 0.0)
    leakage_raw = score.get("fork_scope_leakage")
    if isinstance(leakage_raw, (int, float)) and math.isfinite(float(leakage_raw)):
        leakage = float(leakage_raw)
    else:
        leakage = 0.0
    token_norm = float(int(stats.get("token_est", 0) or 0)) / float(max(1, int(full_history_tokens)))
    return float(answer + float(coverage_weight) * critical - float(token_weight) * token_norm - float(leakage_weight) * leakage)


def _base_pivot_features(
    *,
    thread: TraceThread,
    step: TraceStep,
    history_ids: Sequence[str],
    invalidated_ids: Sequence[str],
    none_stats: Mapping[str, Any],
    unfold_stats: Mapping[str, Any],
    fork_stats: Mapping[str, Any],
    utf_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    ordered_history = _unique_strs(history_ids)
    history_token_est = estimate_tokens_for_ids(thread, ordered_history)
    recent_ids = _ids_within_window(thread, ordered_history, step_idx=int(step.step_idx), window=6)
    avoid_set = set(_unique_strs(invalidated_ids))
    trap_ids = _unique_strs((step.metadata or {}).get("trap_decision_checkpoint_ids") or [])
    features = {
        "level": int(thread.level),
        "scenario_mixed": 1 if str(thread.scenario) == "mixed" else 0,
        "scenario_indirect": 1 if str(thread.scenario) == "indirect" else 0,
        "scenario_budget": 1 if str(thread.scenario) == "budget" else 0,
        "step_idx": int(step.step_idx),
        "history_count": int(len(ordered_history)),
        "history_token_est": int(history_token_est),
        "history_recent_count_w6": int(len(recent_ids)),
        "history_invalidated_count": int(len(avoid_set)),
        "pivot_message_token_count": int(len(_tokenize(step.message))),
        "trap_decision_checkpoint_count": int(len(trap_ids)),
        "none_context_count": int(none_stats.get("context_count", 0) or 0),
        "none_token_est": int(none_stats.get("token_est", 0) or 0),
        "none_recent_ratio_w6": _safe_float(none_stats.get("recent_context_ratio_w6"), 0.0),
        "none_update_count": int(none_stats.get("update_count", 0) or 0),
        "none_exception_count": int(none_stats.get("exception_count", 0) or 0),
        "none_mean_query_overlap": _safe_float(none_stats.get("mean_query_overlap"), 0.0),
        "unfold_context_count": int(unfold_stats.get("context_count", 0) or 0),
        "unfold_token_est": int(unfold_stats.get("token_est", 0) or 0),
        "unfold_update_count": int(unfold_stats.get("update_count", 0) or 0),
        "unfold_exception_count": int(unfold_stats.get("exception_count", 0) or 0),
        "fork_context_count": int(fork_stats.get("context_count", 0) or 0),
        "fork_token_est": int(fork_stats.get("token_est", 0) or 0),
        "utf_context_count": int(utf_stats.get("context_count", 0) or 0),
        "utf_token_est": int(utf_stats.get("token_est", 0) or 0),
    }
    features["budget_pressure_proxy"] = float(int(features["none_token_est"]) / float(max(1, int(features["history_token_est"]))))
    features["unfold_cost_jump"] = float(int(features["unfold_token_est"]) - int(features["none_token_est"]))
    features["fork_cost_jump"] = float(int(features["fork_token_est"]) - int(features["none_token_est"]))
    features["utf_cost_jump"] = float(int(features["utf_token_est"]) - int(features["none_token_est"]))
    return features


def _action_payload(eval_obj: ActionEval) -> Dict[str, Any]:
    return {
        "action": str(eval_obj.action),
        "context_ids": list(eval_obj.context_ids),
        "debug": dict(eval_obj.debug),
        "prediction": dict(eval_obj.prediction),
        "score": dict(eval_obj.score),
        "stats": dict(eval_obj.stats),
        "utility": float(eval_obj.utility),
    }


def pivot_eval_to_dict(pivot: PivotEval) -> Dict[str, Any]:
    return {
        "thread_id": str(pivot.thread_id),
        "step_id": str(pivot.step_id),
        "step_idx": int(pivot.step_idx),
        "split": str(pivot.split),
        "features": dict(pivot.features),
        "actions": {name: _action_payload(ev) for name, ev in pivot.actions.items()},
        "best_action": str(pivot.best_action),
        "best_utility": float(pivot.best_utility),
    }


def evaluate_pivot_actions(
    *,
    thread: TraceThread,
    step: TraceStep,
    history_ids: Sequence[str],
    invalidated_ids: Sequence[str],
    args: Any,
    none_mode: str = "agent_fold",
    token_weight: float = 0.10,
    coverage_weight: float = 0.15,
    leakage_weight: float = 0.00,
    dev_ratio: float = 0.5,
    split_seed: int = 7,
) -> PivotEval:
    none_method = "agent_fold" if str(none_mode).strip().lower() == "agent_fold" else "similarity_only"

    none_context_ids, none_debug = _choose_traceops_context(
        none_method,
        thread=thread,
        step=step,
        history_ids=history_ids,
        invalidated_ids=invalidated_ids,
        args=args,
    )
    unfold_context_ids, unfold_debug = _choose_traceops_context(
        "goc",
        thread=thread,
        step=step,
        history_ids=history_ids,
        invalidated_ids=invalidated_ids,
        args=args,
    )
    ordered_history = _unique_strs(history_ids)
    fork_context_ids, fork_debug = build_dep_scoped_fork_from_base_context(
        thread=thread,
        step=step,
        ordered_history=ordered_history,
        base_context_ids=none_context_ids,
        args=args,
        allow_gold_support=False,
    )
    utf_context_ids, utf_debug = build_dep_scoped_fork_from_base_context(
        thread=thread,
        step=step,
        ordered_history=ordered_history,
        base_context_ids=unfold_context_ids,
        args=args,
        allow_gold_support=False,
    )

    full_history_tokens = estimate_tokens_for_ids(thread, ordered_history)
    action_contexts = {
        "none": (none_context_ids, none_debug),
        "unfold": (unfold_context_ids, unfold_debug),
        "fork": (fork_context_ids, fork_debug),
        "unfold_then_fork": (utf_context_ids, utf_debug),
    }
    actions: Dict[str, ActionEval] = {}
    for action_name, (context_ids, debug) in action_contexts.items():
        pred = _infer_prediction(step=step, context_ids=context_ids, invalidated_ids=invalidated_ids)
        score = _score_step(
            thread=thread,
            step=step,
            prediction=pred,
            context_ids=context_ids,
            invalidated_ids=invalidated_ids,
            eval_mode="deterministic",
        )
        stats = summarize_context(thread, step, context_ids)
        utility = _compute_utility(
            score=score,
            stats=stats,
            full_history_tokens=full_history_tokens,
            token_weight=token_weight,
            coverage_weight=coverage_weight,
            leakage_weight=leakage_weight,
        )
        actions[action_name] = ActionEval(
            action=action_name,
            context_ids=list(context_ids),
            debug=dict(debug),
            prediction=dict(pred),
            score=dict(score),
            stats=dict(stats),
            utility=float(utility),
        )

    none_stats = actions["none"].stats
    unfold_stats = actions["unfold"].stats
    fork_stats = actions["fork"].stats
    utf_stats = actions["unfold_then_fork"].stats
    features = _base_pivot_features(
        thread=thread,
        step=step,
        history_ids=history_ids,
        invalidated_ids=invalidated_ids,
        none_stats=none_stats,
        unfold_stats=unfold_stats,
        fork_stats=fork_stats,
        utf_stats=utf_stats,
    )
    split = assign_split(str(thread.thread_id), dev_ratio=dev_ratio, seed=split_seed)
    ranked = sorted(
        ((name, ev.utility, int(ev.stats.get("token_est", 0) or 0)) for name, ev in actions.items()),
        key=lambda item: (-float(item[1]), item[2], item[0]),
    )
    best_action = str(ranked[0][0]) if ranked else "none"
    best_utility = float(ranked[0][1]) if ranked else 0.0
    return PivotEval(
        thread_id=str(thread.thread_id),
        step_id=str(step.step_id),
        step_idx=int(step.step_idx),
        split=split,
        features=features,
        actions=actions,
        best_action=best_action,
        best_utility=best_utility,
    )


def iter_pivot_evals(
    threads: Sequence[TraceThread],
    *,
    args: Any,
    none_mode: str = "agent_fold",
    token_weight: float = 0.10,
    coverage_weight: float = 0.15,
    leakage_weight: float = 0.00,
    dev_ratio: float = 0.5,
    split_seed: int = 7,
) -> Iterable[PivotEval]:
    max_steps = int(_arg_get(args, "traceops_max_steps", 0) or 0)
    for thread in threads:
        history_ids: List[str] = []
        invalidated_ids: List[str] = []
        for step in thread.steps:
            if max_steps > 0 and int(step.step_idx) >= max_steps:
                break
            history_ids = _unique_strs(history_ids + list(step.introduced_clause_ids))
            if step.kind == "update" and step.avoid_target_ids:
                invalidated_ids = _unique_strs(invalidated_ids + list(step.avoid_target_ids))
            if not bool(step.kind == "pivot_check" and step.gold is not None):
                continue
            yield evaluate_pivot_actions(
                thread=thread,
                step=step,
                history_ids=history_ids,
                invalidated_ids=invalidated_ids,
                args=args,
                none_mode=none_mode,
                token_weight=token_weight,
                coverage_weight=coverage_weight,
                leakage_weight=leakage_weight,
                dev_ratio=dev_ratio,
                split_seed=split_seed,
            )


__all__ = [
    "ACTION_NAMES",
    "ActionEval",
    "PivotEval",
    "assign_split",
    "build_dep_scoped_fork_from_base_context",
    "estimate_tokens_for_ids",
    "evaluate_pivot_actions",
    "iter_pivot_evals",
    "pivot_eval_to_dict",
    "summarize_context",
]
