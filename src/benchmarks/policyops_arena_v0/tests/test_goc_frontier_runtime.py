from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.goc_frontier import graph_frontier_candidates, merge_ranked_candidates  # noqa: E402


@dataclass
class _Clause:
    clause_id: str
    targets: Dict[str, List[str]] = field(default_factory=dict)
    terms_used: List[str] = field(default_factory=list)


@dataclass
class _World:
    clauses: Dict[str, _Clause]
    meta: Dict[str, object] = field(default_factory=dict)


def test_graph_frontier_candidates_with_targets_and_term_defs() -> None:
    clauses = {
        "c1": _Clause("c1", targets={"defines": ["c2"]}, terms_used=["T_A"]),
        "c2": _Clause("c2", targets={"overrides": ["c3"]}, terms_used=[]),
        "c3": _Clause("c3", targets={}, terms_used=[]),
        "c4": _Clause("c4", targets={}, terms_used=[]),
    }
    world = _World(
        clauses=clauses,
        meta={"term_definitions": {"T_A": "c4"}},
    )

    out, dist = graph_frontier_candidates(world, ["c1"], max_hops=2, max_nodes=10)
    assert set(out) >= {"c2", "c4"}
    assert dist["c2"] == 1
    assert dist["c4"] == 1


def test_merge_ranked_candidates_uses_max_score() -> None:
    primary = [
        {"clause_id": "c1", "score": 0.2},
        {"clause_id": "c2", "score": 0.7},
    ]
    secondary = [
        {"clause_id": "c1", "score": 0.8, "source": "graph_frontier"},
        {"clause_id": "c3", "score": 0.5, "source": "graph_frontier"},
    ]

    merged = merge_ranked_candidates(primary, secondary)
    by_id = {r["clause_id"]: r for r in merged}
    assert by_id["c1"]["score"] == 0.8
    assert by_id["c3"]["score"] == 0.5
    assert by_id["c1"]["source"] == "graph_frontier"

