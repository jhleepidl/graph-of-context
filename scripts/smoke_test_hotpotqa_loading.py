"""Quick smoke test for the HotpotQA loader.

This repo supports 2 common HotpotQA file formats:
  1) Official HotpotQA release JSON (context as list-of-pairs)
  2) HuggingFace datasets dump JSON (context as dict of lists)

Run:
  python scripts/smoke_test_hotpotqa_loading.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running the script from repo root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks.hotpotqa import HotpotQA


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    data_dir = Path("data")
    bench = HotpotQA()

    # HuggingFace-style
    hf = [
        {
            "id": "hp_hf_1",
            "question": "What is 2+2?",
            "answer": "4",
            "context": {"title": ["Math", "Other"], "sentences": [["2+2 is 4."], ["Nothing."]]},
            "supporting_facts": {"title": ["Math"], "sent_id": [0]},
        }
    ]
    hf_path = data_dir / "hotpotqa" / "hotpot_hf_smoke.json"
    _write(hf_path, hf)
    tasks = bench.load_tasks(str(data_dir), path=str(hf_path), limit=1)
    assert len(tasks) == 1 and tasks[0].meta and tasks[0].meta.get("docs"), "HF format load failed"

    # Official-style
    off = [
        {
            "_id": "hp_off_1",
            "question": "What is 3+3?",
            "answer": "6",
            "context": [["Math", ["3+3 is 6."]], ["Other", ["Nothing."]]],
            "supporting_facts": [["Math", 0]],
        }
    ]
    off_path = data_dir / "hotpotqa" / "hotpot_official_smoke.json"
    _write(off_path, off)
    tasks2 = bench.load_tasks(str(data_dir), path=str(off_path), limit=1)
    assert len(tasks2) == 1 and tasks2[0].meta and tasks2[0].meta.get("docs"), "Official format load failed"

    print("OK: HotpotQA loader supports both HF and official formats.")


if __name__ == "__main__":
    main()
