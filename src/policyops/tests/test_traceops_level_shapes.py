from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def _shape(level: int) -> tuple[int, int]:
    threads, _ = generate_traceops_threads(level=level, scenarios=["mixed"], seed=7, threads=1)
    thread = threads[0]
    step_count = len(thread.steps)
    pivot_count = sum(1 for s in thread.steps if s.kind == "pivot_check")
    return step_count, pivot_count


def test_traceops_level_shapes() -> None:
    s0, p0 = _shape(0)
    assert s0 == 3
    assert p0 == 1

    s1, p1 = _shape(1)
    assert s1 == 5
    assert p1 == 1

    s2, p2 = _shape(2)
    assert s2 == 6
    assert p2 == 2

    s3, p3 = _shape(3)
    assert 9 <= s3 <= 12
    assert 2 <= p3 <= 3

    s4, p4 = _shape(4)
    assert 15 <= s4 <= 24
    assert 3 <= p4 <= 5
