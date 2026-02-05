from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.query_facets import extract_facets


def test_query_facets_extracts_core_fields(tmp_path: Path) -> None:
    _, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=3,
        scenario_mode="bridged_v1_1",
    )
    task = tasks[0]
    facets = extract_facets(task.user_ticket)
    non_empty = sum(1 for v in facets.values() if v)
    assert non_empty >= 2
    assert facets.get("region")
