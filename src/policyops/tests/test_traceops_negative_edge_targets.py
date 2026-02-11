from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def test_traceops_negative_edge_targets_from_contradiction() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["contradiction"],
        seed=3,
        threads=1,
        trace_len=10,
        contradiction_rate=1.0,
    )
    thread = threads[0]
    intro_step = {cid: clause.step_idx for cid, clause in thread.clauses.items()}

    found = False
    for step in thread.steps:
        if step.kind != "update":
            continue
        if not step.avoid_target_ids:
            continue
        for target in step.avoid_target_ids:
            if target in intro_step and intro_step[target] < step.step_idx:
                found = True
                break
        if found:
            break

    assert found
