from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def test_traceops_pivot_check_introduces_no_new_facts() -> None:
    threads, _ = generate_traceops_threads(
        level=4,
        scenarios=["mixed"],
        seed=9,
        threads=1,
        trace_len=16,
        delay_to_relevance=6,
    )
    thread = threads[0]

    seen: set[str] = set()
    for step in thread.steps:
        if step.kind == "pivot_check":
            assert not step.introduced_clause_ids
            if step.gold is not None:
                for cid in step.gold.evidence_ids:
                    assert cid in seen
        else:
            seen.update(step.introduced_clause_ids)
