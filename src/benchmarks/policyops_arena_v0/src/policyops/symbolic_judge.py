from __future__ import annotations

from typing import Any, Dict, List

from .schemas import Clause, World, model_dump
from .world import evaluate_context


def judge_from_opened_clauses(
    task: Any, opened_clause_ids: List[str], world: World
) -> Dict[str, Any]:
    clauses: Dict[str, Clause] = {}
    for cid in opened_clause_ids:
        clause = world.clauses.get(cid)
        if clause:
            clauses[cid] = Clause(**model_dump(clause))
    if not clauses:
        return {
            "decision": "needs_more_info",
            "conditions": [],
            "evidence": [],
            "customer_message": "",
        }
    partial_world = World(
        documents=world.documents,
        clauses=clauses,
        meta=world.meta,
    )
    decision, conditions, evidence, _ = evaluate_context(partial_world, task.context)
    evidence = [cid for cid in evidence if cid in opened_clause_ids]
    return {
        "decision": decision,
        "conditions": conditions,
        "evidence": evidence,
        "customer_message": "",
    }
