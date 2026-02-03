from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyops.generator import generate_world_and_tasks


def test_generator_smoke(tmp_path: Path) -> None:
    world, tasks, world_dir, tasks_path = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=1,
        n_docs=4,
        clauses_per_doc=3,
        n_tasks=5,
    )
    assert world.documents
    assert tasks
    assert (world_dir / "documents.jsonl").exists()
    assert (world_dir / "clauses.jsonl").exists()
    assert tasks_path.exists()

    doc_lines = (world_dir / "documents.jsonl").read_text(encoding="utf-8").strip().splitlines()
    clause_lines = (world_dir / "clauses.jsonl").read_text(encoding="utf-8").strip().splitlines()
    task_lines = tasks_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(doc_lines) == 4
    assert len(clause_lines) == 4 * 3
    assert len(task_lines) == 5
