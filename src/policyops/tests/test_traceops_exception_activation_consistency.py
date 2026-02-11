from __future__ import annotations

import re

from policyops.traceops_v0.generator import generate_traceops_threads


def test_activated_exception_text_matches_step_state() -> None:
    threads, _ = generate_traceops_threads(
        level=4,
        scenarios=["mixed"],
        seed=11,
        threads=24,
        exception_density=1.0,
    )

    checked = 0
    for thread in threads:
        state_by_step = {int(step.step_idx): dict(step.state or {}) for step in thread.steps}
        for clause in thread.clauses.values():
            if str(clause.node_type or "") != "EXCEPTION":
                continue
            text = str(clause.text or "")
            lowered = text.lower()
            if "manual override allowed if deadline is tight" not in lowered:
                continue
            state = state_by_step.get(int(clause.step_idx), {})
            deadline = str(state.get("deadline", "")).strip().lower()
            if lowered.startswith("exception activated:"):
                checked += 1
                assert deadline == "tight"
                m = re.search(r"if\s+(\w+)\s+is\s+(\w+)", lowered)
                assert m is not None
                assert str(state.get(str(m.group(1)).strip(), "")).strip().lower() == str(m.group(2)).strip().lower()
            elif lowered.startswith("exception latent:"):
                assert deadline != "tight"

    assert checked > 0
