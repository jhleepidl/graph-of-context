from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.symbolic_judge import judge_from_opened_clauses


def test_exclusive_core_evidence_requires_core(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=1,
        n_docs=8,
        clauses_per_doc=4,
        n_tasks=5,
        scenario_mode="bridged_v1_1",
        bridge_prob=1.0,
        alias_density=1.0,
        canonical_density=1.0,
        exclusive_core_evidence=True,
    )
    task = tasks[0]
    core_ids = set(getattr(task.gold, "gold_evidence_core", []) or [])
    assert core_ids

    opened_ids = [cid for cid in world.clauses if cid not in core_ids]
    judge = judge_from_opened_clauses(task, opened_ids, world)
    assert judge.get("decision") != task.gold.decision
