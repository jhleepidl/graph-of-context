from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def test_traceops_requires_unfold_with_long_delay() -> None:
    delay = 6
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["latent"],
        seed=11,
        threads=1,
        trace_len=10,
        delay_to_relevance=delay,
    )
    thread = threads[0]
    intro_step = {cid: clause.step_idx for cid, clause in thread.clauses.items()}

    has_long_delayed_required = False
    for step in thread.steps:
        if step.kind != "pivot_check" or step.gold is None:
            continue
        for cid in step.pivot_required_ids:
            if cid not in intro_step:
                continue
            if (step.step_idx - intro_step[cid]) >= delay:
                has_long_delayed_required = True
                break
        if has_long_delayed_required:
            break

    assert has_long_delayed_required
