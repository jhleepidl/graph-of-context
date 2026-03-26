from __future__ import annotations

from src.memory import GoCMemory


def test_build_fork_view_and_rejoin() -> None:
    mem = GoCMemory(budget_active=500, budget_unfold=200, unfold_k=4, fork_default_max_tokens=120)
    q = mem.record_msg("Need support for City A eligibility")
    anchor = mem.record_summary("Rule anchor: allow if airport access and low rent")
    upd = mem.record_summary("Update: City A now has airport access and low rent")
    distractor = mem.record_summary("Distractor: City Z museum subsidy")
    mem.add_edge("depends", upd, anchor)
    mem.add_edge("depends", upd, q)
    fork = mem.build_fork_view("City A eligibility", scope_mode="dep_scoped", deny_kinds=("tool",))
    assert anchor in fork.node_ids
    assert upd in fork.node_ids
    assert distractor in mem.nodes
    rid = mem.record_fork_result(fork, "City A satisfies the rule")
    assert rid in mem.nodes
    assert mem.nodes[rid].text.startswith("[FORK_RESULT")
