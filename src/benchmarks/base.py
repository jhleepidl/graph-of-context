from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

@dataclass
class Task:
    id: str
    question: str
    answer: str
    # Optional multi-turn prompts. If provided, `question` should typically be the first turn.
    turns: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    required: Optional[List[str]] = None
    gold_docids: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

class Benchmark(Protocol):
    """Benchmark interface.

    A benchmark provides:
      - dataset generation (optional)
      - environment/tools construction
      - tasks loading
      - evaluation function(s)
    """

    name: str

    def prepare(self, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Optional: generate or download data into data_dir. Return metadata."""
        ...

    def load_tasks(self, data_dir: str, limit: Optional[int] = None, **kwargs) -> List[Task]:
        ...

    def build_tools(self, data_dir: str, **kwargs):
        """Return a tools object compatible with the runner (must provide search/open_page)."""
        ...

    def evaluate(self, pred_answer: str, pred_expl: str, task: Task) -> Dict[str, Any]:
        """Return metrics like correctness, docid coverage, etc."""
        ...
