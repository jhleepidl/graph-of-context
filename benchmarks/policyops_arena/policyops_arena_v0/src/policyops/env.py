from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import Clause, World
from .tools import ClauseIndex, make_snippet

UPDATE_DOC_TYPES = {"announcement", "release_note"}


class PolicyOpsEnv:
    def __init__(self, world: World, tool_call_budget: int = 50, open_budget: int = 5) -> None:
        self.world = world
        self.index = ClauseIndex(world)
        self.tool_call_budget = tool_call_budget
        self.open_budget = open_budget
        self.tool_call_count = 0
        self.open_count = 0
        self._doc_by_id = {doc.doc_id: doc for doc in world.documents}

    def reset_budgets(self, tool_call_budget: int, open_budget: int) -> None:
        self.tool_call_budget = tool_call_budget
        self.open_budget = open_budget
        self.tool_call_count = 0
        self.open_count = 0

    def _consume_tool(self) -> None:
        self.tool_call_count += 1
        if self.tool_call_count > self.tool_call_budget:
            raise RuntimeError("tool_call_budget exceeded")

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        self._consume_tool()
        return self.index.search(query=query, filters=filters, top_k=top_k)

    def open(self, clause_id: str) -> Dict[str, Any]:
        self._consume_tool()
        self.open_count += 1
        if self.open_count > self.open_budget:
            raise RuntimeError("open_budget exceeded")
        clause = self.world.clauses.get(clause_id)
        if clause is None:
            raise KeyError(f"Unknown clause_id: {clause_id}")
        doc = self._doc_by_id.get(clause.doc_id)
        return {
            "clause_id": clause.clause_id,
            "text": clause.text,
            "kind": clause.kind,
            "slot": clause.slot,
            "authority": clause.authority,
            "published_at": clause.published_at,
            "document": doc.to_dict() if doc else {},
        }

    def list_updates(self, since_date: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        self._consume_tool()
        update_docs = []
        update_doc_ids = set()
        for clause in self.world.clauses.values():
            if clause.kind == "update":
                update_doc_ids.add(clause.doc_id)
        for doc in self.world.documents:
            if doc.doc_type in UPDATE_DOC_TYPES or doc.doc_id in update_doc_ids:
                if since_date and doc.published_at < since_date:
                    continue
                update_docs.append(doc)
        update_docs.sort(key=lambda d: d.published_at, reverse=True)
        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "authority": doc.authority,
                "published_at": doc.published_at,
            }
            for doc in update_docs[:top_k]
        ]
