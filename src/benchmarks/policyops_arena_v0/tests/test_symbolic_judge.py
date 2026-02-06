from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.symbolic_judge import judge_from_opened_clauses


def test_symbolic_judge_matches_when_evidence_present(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=3,
    )
    task = tasks[0]
    opened_ids = list(task.gold.gold_evidence)
    result = judge_from_opened_clauses(task, opened_ids, world)
    assert result["decision"] == task.gold.decision
    assert isinstance(result.get("supporting_clause_ids"), list)
    assert result.get("supporting_clause_ids")


def test_symbolic_judge_needs_more_info_without_evidence(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=1,
        n_docs=4,
        clauses_per_doc=3,
        n_tasks=2,
    )
    task = tasks[0]
    result = judge_from_opened_clauses(task, [], world)
    assert result["decision"] == "needs_more_info"
    assert isinstance(result.get("supporting_clause_ids"), list)
    assert result.get("supporting_clause_ids") == []
