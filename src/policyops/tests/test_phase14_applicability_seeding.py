from __future__ import annotations

from policyops.run import _compute_phase14_applicability_seed_ids
from policyops.schemas import Clause, World


def _mk_clause(clause_id: str, *, kind: str, applies_if: dict[str, list[str]]) -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id="D1",
        published_at="2024-01-01",
        authority="legal",
        text=f"clause {clause_id}",
        kind=kind,
        slot="retention",
        applies_if=applies_if,
        effect={"decision": "allow"},
        conditions=[],
        targets={"overrides": [], "revokes": [], "defines": []},
        terms_used=[],
    )


def test_applicability_seeding_prefers_applicable_and_excludes_avoid_ids() -> None:
    world = World(
        documents=[],
        clauses={
            "C0001": _mk_clause("C0001", kind="rule", applies_if={"region": ["eu"]}),
            "C0002": _mk_clause("C0002", kind="rule", applies_if={"region": ["us"]}),
            "C0003": _mk_clause("C0003", kind="priority", applies_if={}),
            "C0004": _mk_clause("C0004", kind="rule", applies_if={"region": ["eu"]}),
        },
        meta={},
    )

    seed_ids, applicable_rate = _compute_phase14_applicability_seed_ids(
        candidate_clause_ids=["C0002", "C0001", "C0003", "C0004"],
        world=world,
        effective_context={"region": "eu"},
        avoid_clause_ids=["C0004"],
        top_k=8,
    )

    assert "C0002" not in seed_ids  # inapplicable
    assert "C0004" not in seed_ids  # avoid target
    assert seed_ids[:2] == ["C0003", "C0001"]  # priority boost + stable order
    assert applicable_rate == 1.0
