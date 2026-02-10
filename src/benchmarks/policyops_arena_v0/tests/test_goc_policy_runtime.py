from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.goc_policy import (  # noqa: E402
    choose_unfold_knobs,
    resolve_internal_budgets_from_namespace,
    resolve_unfold_controls_from_namespace,
    resolve_unfold_policy_from_namespace,
)


def test_resolve_internal_budgets_with_follow_sweep() -> None:
    args = SimpleNamespace(
        goc_internal_budget_active_tokens=1200,
        goc_internal_budget_unfold_tokens=650,
        goc_internal_unfold_k=8,
        goc_budget_follow_sweep=True,
        goc_budget_chars_to_tokens_divisor=4,
        thread_context_budget_chars=4000,
    )
    cfg = resolve_internal_budgets_from_namespace(args)
    assert cfg.active_tokens == 1000
    assert cfg.unfold_tokens == 550
    assert cfg.unfold_k == 8


def test_resolve_unfold_controls_invalid_mode_falls_back() -> None:
    args = SimpleNamespace(
        goc_unfold_max_nodes=16,
        goc_unfold_hops=3,
        goc_unfold_budget_mode="invalid_mode",
    )
    cfg = resolve_unfold_controls_from_namespace(args)
    assert cfg.max_nodes == 16
    assert cfg.hops == 3
    assert cfg.budget_mode == "nodes_and_hops"


def test_adaptive_pivot_policy_and_knob_choice() -> None:
    args = SimpleNamespace(
        goc_unfold_policy="adaptive_pivot",
        goc_unfold_max_nodes=4,
        goc_unfold_hops=2,
        goc_unfold_budget_mode="nodes_and_hops",
        goc_unfold_default_max_nodes=4,
        goc_unfold_default_hops=2,
        goc_unfold_pivot_max_nodes=16,
        goc_unfold_pivot_hops=3,
    )
    cfg = resolve_unfold_policy_from_namespace(args)
    assert cfg.policy == "adaptive_pivot"
    assert cfg.default_knobs == (4, 2)
    assert cfg.pivot_knobs == (16, 3)

    assert choose_unfold_knobs(cfg, is_pivot_task=False) == (4, 2)
    assert choose_unfold_knobs(cfg, is_pivot_task=True) == (16, 3)

