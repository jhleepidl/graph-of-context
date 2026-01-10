from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from .env import CorpusEnv

@dataclass
class ToolBox:
    env: CorpusEnv

    # Context-Folding style tools
    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        return self.env.search(query=query, topk=topk)

    def open_page(self, docid: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        return self.env.open_page(docid=docid, url=url)

    # branch/return/finish are handled by the runner/memory manager, not the env.
