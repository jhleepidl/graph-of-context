from __future__ import annotations

from src.memory import GoCMemory


def main() -> None:
    mem = GoCMemory(budget_active=500, budget_unfold=200, unfold_k=4, fork_default_max_tokens=120)
    a = mem.record_msg("User asks about whether City A is eligible for the airport+rent rule.")
    b = mem.record_summary("Rule anchor: allow if airport access and rent below threshold.")
    c = mem.record_summary("Distractor: City Z has a museum subsidy that is irrelevant.")
    d = mem.record_summary("Update: City A rent changed to low and airport access is confirmed.")
    mem.add_edge("depends", d, b)
    mem.add_edge("depends", d, a)
    fork = mem.build_fork_view(
        "City A airport rent eligibility",
        k=3,
        max_tokens=120,
        include_recent_active=True,
        deny_kinds=("tool",),
        scope_mode="dep_scoped",
    )
    assert fork.node_ids
    assert d in fork.node_ids
    assert b in fork.node_ids
    result_id = mem.record_fork_result(fork, "Scoped sub-agent: City A satisfies the airport+rent condition.")
    assert result_id in mem.nodes
    deps = mem.edges_out.get("depends", {}).get(result_id, set())
    assert b in deps or d in deps
    print("smoke_test_fork_scope: OK")


if __name__ == "__main__":
    main()
