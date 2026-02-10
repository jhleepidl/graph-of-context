from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Tuple


GOC_UNFOLD_BUDGET_MODES = {"nodes_only", "hops_only", "nodes_and_hops"}
GOC_UNFOLD_POLICIES = {"fixed", "adaptive_pivot", "adaptive_heuristic"}


@dataclass(frozen=True)
class GoCInternalBudgets:
    active_tokens: int
    unfold_tokens: int
    unfold_k: int


@dataclass(frozen=True)
class GoCUnfoldControls:
    max_nodes: int
    hops: int
    budget_mode: str


@dataclass(frozen=True)
class GoCUnfoldPolicy:
    policy: str
    default_knobs: Tuple[int, int]
    pivot_knobs: Tuple[int, int]
    budget_mode: str


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def resolve_internal_budgets_from_namespace(args: argparse.Namespace | Any) -> GoCInternalBudgets:
    active_tokens = _int_or_default(getattr(args, "goc_internal_budget_active_tokens", 1200), 1200)
    if active_tokens <= 0:
        active_tokens = 1200
    unfold_tokens = _int_or_default(getattr(args, "goc_internal_budget_unfold_tokens", 650), 650)
    if unfold_tokens <= 0:
        unfold_tokens = 650
    unfold_k = _int_or_default(getattr(args, "goc_internal_unfold_k", 8), 8)
    if unfold_k <= 0:
        unfold_k = 8

    if bool(getattr(args, "goc_budget_follow_sweep", False)):
        divisor = max(1, _int_or_default(getattr(args, "goc_budget_chars_to_tokens_divisor", 4), 4))
        budget_chars = _int_or_default(getattr(args, "thread_context_budget_chars", 0), 0)
        if budget_chars > 0:
            active_tokens = max(200, budget_chars // divisor)
            unfold_tokens = max(100, int(0.55 * active_tokens))

    return GoCInternalBudgets(
        active_tokens=max(0, int(active_tokens)),
        unfold_tokens=max(0, int(unfold_tokens)),
        unfold_k=max(0, int(unfold_k)),
    )


def resolve_unfold_controls_from_namespace(args: argparse.Namespace | Any) -> GoCUnfoldControls:
    max_nodes = max(0, _int_or_default(getattr(args, "goc_unfold_max_nodes", 0), 0))
    hops = max(0, _int_or_default(getattr(args, "goc_unfold_hops", 0), 0))
    mode = str(getattr(args, "goc_unfold_budget_mode", "nodes_and_hops") or "nodes_and_hops")
    if mode not in GOC_UNFOLD_BUDGET_MODES:
        mode = "nodes_and_hops"
    return GoCUnfoldControls(max_nodes=max_nodes, hops=hops, budget_mode=mode)


def resolve_unfold_policy_from_namespace(args: argparse.Namespace | Any) -> GoCUnfoldPolicy:
    policy = str(getattr(args, "goc_unfold_policy", "fixed") or "fixed")
    if policy not in GOC_UNFOLD_POLICIES:
        policy = "fixed"

    fixed = resolve_unfold_controls_from_namespace(args)

    d_nodes = max(0, _int_or_default(getattr(args, "goc_unfold_default_max_nodes", 0), 0))
    d_hops = max(0, _int_or_default(getattr(args, "goc_unfold_default_hops", 0), 0))
    p_nodes = max(0, _int_or_default(getattr(args, "goc_unfold_pivot_max_nodes", 0), 0))
    p_hops = max(0, _int_or_default(getattr(args, "goc_unfold_pivot_hops", 0), 0))

    if d_nodes <= 0:
        d_nodes = fixed.max_nodes
    if d_hops <= 0:
        d_hops = fixed.hops
    if p_nodes <= 0:
        p_nodes = fixed.max_nodes
    if p_hops <= 0:
        p_hops = fixed.hops

    return GoCUnfoldPolicy(
        policy=policy,
        default_knobs=(max(0, d_nodes), max(0, d_hops)),
        pivot_knobs=(max(0, p_nodes), max(0, p_hops)),
        budget_mode=fixed.budget_mode,
    )


def choose_unfold_knobs(
    policy_cfg: GoCUnfoldPolicy,
    *,
    is_pivot_task: bool = False,
    is_update_episode: bool = False,
) -> Tuple[int, int]:
    policy = str(policy_cfg.policy or "fixed")
    if policy in {"adaptive_pivot", "adaptive_heuristic"}:
        use_pivot = bool(is_pivot_task)
        if policy == "adaptive_heuristic":
            use_pivot = bool(is_pivot_task or is_update_episode)
        if use_pivot:
            return policy_cfg.pivot_knobs
        return policy_cfg.default_knobs
    return policy_cfg.default_knobs

