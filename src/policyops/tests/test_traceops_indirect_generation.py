from __future__ import annotations

import re

from policyops.traceops_v0.generator import generate_traceops_threads


def test_traceops_indirect_generation_has_core_and_trap_properties() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=17,
        threads=6,
        indirection_rate=1.0,
        trap_distractor_count=4,
        trap_similarity_boost=1.0,
        core_size_min=2,
        core_size_max=4,
        alias_chain_len=2,
        indirect_pivot_style="blended",
    )

    pivot_steps = [
        step
        for thread in threads
        for step in thread.steps
        if str(step.kind) == "pivot_check"
    ]
    assert pivot_steps, "expected at least one pivot step"

    assert any(int((step.metadata or {}).get("core_size", 0) or 0) >= 2 for step in pivot_steps)
    assert any(float((step.metadata or {}).get("trap_gap", 0.0) or 0.0) > 0.0 for step in pivot_steps)

    has_pattern = False
    for step in pivot_steps:
        q = str(step.message or "").lower()
        if ("latest change" in q or "second" in q) or re.search(r"(hz|pol|tag|idx)-", q):
            has_pattern = True
            break
    assert has_pattern, "expected ordinal_ref or alias_handle pivot style in questions"
