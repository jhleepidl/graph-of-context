"""Offline smoke test for Multi-Commit DAG task synthesis (HotpotQA).

This does NOT call any LLM. It only verifies that the benchmark loader can
stitch N independent HotpotQA examples into one long-horizon MULTI-COMMIT DAG
episode with COMMIT -> (optional MERGE) -> FINAL turns.

Run:
  python scripts/smoke_test_multicommit_generation.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmarks.hotpotqa import HotpotQA


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    # Minimal official-format HotpotQA examples (3 items) so multi_commit_n=3 can stitch them.
    # Each example MUST have 2 supporting titles and supporting_facts.
    off = [
        {
            "_id": "hp_off_mc_1",
            "question": "What is 1+1?",
            "answer": "2",
            "context": [["Math A", ["1+1 is 2."]], ["Other A", ["Nothing."]]],
            "supporting_facts": [["Math A", 0]],
        },
        {
            "_id": "hp_off_mc_2",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "context": [["France", ["Paris is the capital of France."]], ["Other B", ["Nothing."]]],
            "supporting_facts": [["France", 0]],
        },
        {
            "_id": "hp_off_mc_3",
            "question": "Who wrote Hamlet?",
            "answer": "William Shakespeare",
            "context": [["Hamlet", ["Hamlet was written by William Shakespeare."]], ["Other C", ["Nothing."]]],
            "supporting_facts": [["Hamlet", 0]],
        },
    ]

    path = data_dir / "hotpotqa" / "hotpot_official_multicommit_smoke.json"
    _write(path, off)

    bench = HotpotQA()
    tasks = bench.load_tasks(
        str(data_dir),
        path=str(path),
        limit=1,
        multi_commit_n=3,
        multi_commit_include_merges=True,
        multi_commit_merge_plan="binary_tree",
        multi_commit_compose_rule="reduce_lex_min",
        multi_commit_merge_closed_book=True,
        multi_commit_doc_shuffle=False,
    )

    assert len(tasks) == 1, f"Expected 1 stitched task, got {len(tasks)}"
    t = tasks[0]
    assert t.turns and any("MULTI-COMMIT DAG" in x for x in t.turns), "Expected MULTI-COMMIT intro in turns"
    # Expect: intro + 3 commits + 2 merges (binary tree for N=3) + final = 7 turns
    assert len(t.turns) == 7, f"Expected 7 turns for N=3 binary_tree, got {len(t.turns)}"
    print("OK: Multi-Commit DAG synthesis produced", len(t.turns), "turns")


if __name__ == "__main__":
    main()
