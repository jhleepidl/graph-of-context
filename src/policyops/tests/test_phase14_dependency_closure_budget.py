from __future__ import annotations

from policyops.run import _compute_phase14_dependency_closure_ids
from policyops.schemas import Clause, World


def _mk_clause(
    clause_id: str,
    *,
    kind: str = "rule",
    applies_if: dict[str, list[str]] | None = None,
    overrides: list[str] | None = None,
) -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id="D1",
        published_at="2024-01-01",
        authority="legal",
        text=f"clause {clause_id}",
        kind=kind,
        slot="retention",
        applies_if=applies_if or {},
        effect={"decision": "allow"},
        conditions=[],
        targets={"overrides": list(overrides or []), "revokes": [], "defines": []},
        terms_used=[],
    )


def test_dependency_closure_respects_budget_avoid_and_applicability() -> None:
    world = World(
        documents=[],
        clauses={
            # seed
            "C0010": _mk_clause("C0010", applies_if={"region": ["eu"]}),
            # hop1 neighbors
            "C0011": _mk_clause("C0011", applies_if={"region": ["eu"]}, overrides=["C0010"]),
            "C0014": _mk_clause("C0014", applies_if={"region": ["eu"]}, overrides=["C0010"]),
            # hop2 inapplicable
            "C0012": _mk_clause("C0012", applies_if={"region": ["us"]}, overrides=["C0011"]),
        },
        meta={},
    )

    closure_added_ids, closure_rate = _compute_phase14_dependency_closure_ids(
        seed_clause_ids=["C0010"],
        unfold_candidate_ids=["C0010", "C0011", "C0014", "C0012"],
        opened_history_ids=["C0010", "C0011", "C0014", "C0012"],
        avoid_clause_ids=["C0011"],
        world=world,
        effective_context={"region": "eu"},
        top_k=1,
        hops=2,
        universe_mode="candidates",
    )

    assert len(closure_added_ids) <= 1  # budget
    assert "C0011" not in closure_added_ids  # avoid filter
    assert "C0012" not in closure_added_ids  # inapplicable filter
    assert closure_added_ids == ["C0014"]
    assert closure_rate == 1.0
