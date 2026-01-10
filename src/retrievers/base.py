from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Protocol, Optional, Dict, Any

@dataclass
class TextItem:
    id: str
    text: str
    meta: Optional[Dict[str, Any]] = None

class TextRetriever(Protocol):
    """Minimal retriever interface used by CorpusEnv and GoCMemory storage."""
    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        ...

    def size(self) -> int:
        ...
