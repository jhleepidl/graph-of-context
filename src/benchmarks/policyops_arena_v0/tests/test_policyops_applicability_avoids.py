from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.policyops.run import (  # noqa: E402
    _clause_applies_to_context,
    _compute_goc_avoid_clause_ids,
)


def test_clause_applies_to_context_retention_bucket() -> None:
    clause = SimpleNamespace(applies_if={"retention_bucket": ["le_30"]})
    assert _clause_applies_to_context(clause, {"retention_bucket": "le_30"}) is True
    assert _clause_applies_to_context(clause, {"retention_bucket": "gt_30"}) is False


def test_applicability_mode_derives_avoids_from_applies_if_mismatch() -> None:
    world = SimpleNamespace(
        clauses={
            "C_APPLIES": SimpleNamespace(applies_if={"retention_bucket": ["gt_30"]}),
            "C_STALE": SimpleNamespace(applies_if={"retention_bucket": ["le_30"]}),
            "C_COMMIT_ONLY": SimpleNamespace(applies_if={"retention_bucket": ["gt_30"]}),
        }
    )
    avoid_ids, inapplicable = _compute_goc_avoid_clause_ids(
        mode="applicability",
        is_pivot_task=True,
        opened_history_ids=["C_APPLIES", "C_STALE"],
        world=world,
        effective_context={"retention_bucket": "gt_30"},
        commit1={"anchor_clause_ids": ["C_COMMIT_ONLY"], "supporting_clause_ids": []},
        commit2={"anchor_clause_ids": [], "supporting_clause_ids": []},
    )
    assert avoid_ids == ["C_STALE"]
    assert inapplicable == ["C_STALE"]
    assert "C_COMMIT_ONLY" not in avoid_ids

