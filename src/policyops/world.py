from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .schemas import Clause, Document, Gold, Task, World, USE_PYDANTIC

CRITICAL_KINDS = {"definition", "update", "exception"}


def _applies(applies_if: Dict[str, List[str]], context: Dict[str, Any]) -> bool:
    for key, values in applies_if.items():
        if not values:
            continue
        if context.get(key) not in values:
            return False
    return True


def _authority_priority(meta: Dict[str, Any]) -> Dict[str, int]:
    return meta.get(
        "authority_priority",
        {"security": 0, "legal": 1, "product": 2, "support": 3},
    )


def evaluate_context(
    world: World, context: Dict[str, Any]
) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    slot = context.get("slot")
    candidates = [
        clause for clause in world.clauses.values() if clause.slot == slot
    ]
    active = [clause for clause in candidates if _applies(clause.applies_if, context)]

    debug: Dict[str, Any] = {
        "used_updates": [],
        "used_priority": False,
        "overrides": [],
    }

    if not active:
        return "needs_more_info", [], [], debug

    # Pre-update competitive pool for evidence checks.
    pre_update_active = list(active)
    pre_update_overridden: set[str] = set()
    pre_update_ids = {clause.clause_id for clause in pre_update_active}
    for clause in pre_update_active:
        for target in clause.targets.get("overrides", []):
            if target in pre_update_ids:
                pre_update_overridden.add(target)
    pre_update_pool = [
        clause
        for clause in pre_update_active
        if clause.clause_id not in pre_update_overridden and clause.kind != "procedure"
    ]
    pre_update_pool_ids = {clause.clause_id for clause in pre_update_pool}

    # Remove revoked clauses via updates.
    revoked: set[str] = set()
    update_clauses = [
        clause for clause in active if clause.kind == "update" and clause.targets.get("revokes")
    ]
    updates_affecting: List[Clause] = []
    for clause in update_clauses:
        targets = set(clause.targets.get("revokes", []))
        if targets & pre_update_pool_ids:
            updates_affecting.append(clause)
        revoked.update(targets)
    if updates_affecting:
        debug["used_updates"] = [c.clause_id for c in updates_affecting]
    active = [clause for clause in active if clause.clause_id not in revoked]

    # Apply exception overrides.
    overridden: set[str] = set()
    override_sources: List[str] = []
    active_ids = {clause.clause_id for clause in active}
    for clause in active:
        for target in clause.targets.get("overrides", []):
            if target in active_ids:
                overridden.add(target)
                override_sources.append(clause.clause_id)
    if override_sources:
        debug["overrides"] = override_sources
    active = [clause for clause in active if clause.clause_id not in overridden]

    if not active:
        return "needs_more_info", [], [], debug

    supplemental = [clause for clause in active if clause.kind == "procedure"]
    primary = [clause for clause in active if clause not in supplemental]
    if not primary:
        primary = list(active)

    # Authority priority.
    if len(primary) > 1:
        priority = _authority_priority(world.meta)
        ranks = {clause.clause_id: priority.get(clause.authority, 999) for clause in primary}
        best_rank = min(ranks.values())
        filtered = [clause for clause in primary if ranks[clause.clause_id] == best_rank]
        if len(filtered) < len(primary):
            debug["used_priority"] = True
        primary = filtered

    # Latest published_at tie-breaker.
    if len(primary) > 1:
        latest_date = max(clause.published_at for clause in primary)
        primary = [clause for clause in primary if clause.published_at == latest_date]

    # Stable final choice.
    primary.sort(key=lambda c: c.clause_id)
    winning = primary[0]
    decision = winning.effect.get("decision", "needs_more_info")

    conditions: List[str] = list(winning.conditions)
    for clause in supplemental:
        for cond in clause.conditions:
            if cond not in conditions:
                conditions.append(cond)

    evidence: List[str] = [winning.clause_id]

    # Include override chain evidence.
    for target in winning.targets.get("overrides", []):
        if target not in evidence:
            evidence.append(target)

    # Include updates that revoked competing clauses.
    for clause in updates_affecting:
        if clause.clause_id not in evidence:
            evidence.append(clause.clause_id)

    # Include priority clause evidence only if priority tie-breaker applied.
    if debug.get("used_priority"):
        priority_ids = world.meta.get("priority_clause_ids", [])
        if priority_ids:
            if priority_ids[0] not in evidence:
                evidence.append(priority_ids[0])

    # Include definition clauses for terms used (optionally recursive).
    term_definitions = world.meta.get("term_definitions", {}) or {}
    max_depth = int(world.meta.get("definition_dependency_depth", 1) or 1)
    visited_terms: set[str] = set()

    def _add_term_definition(term_id: str, depth_left: int) -> None:
        if not term_id or term_id in visited_terms or depth_left <= 0:
            return
        visited_terms.add(term_id)
        def_clause_id = term_definitions.get(term_id)
        if def_clause_id and def_clause_id not in evidence:
            evidence.append(def_clause_id)
        if depth_left <= 1 or not def_clause_id:
            return
        clause = world.clauses.get(def_clause_id)
        if not clause:
            return
        for next_term in getattr(clause, "terms_used", []) or []:
            if next_term == term_id:
                continue
            _add_term_definition(str(next_term), depth_left - 1)

    if max_depth > 0:
        for term_id in getattr(winning, "terms_used", []) or []:
            _add_term_definition(str(term_id), max_depth)

    return decision, conditions, evidence, debug


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_world(world_dir: Path) -> World:
    docs_path = world_dir / "documents.jsonl"
    clauses_path = world_dir / "clauses.jsonl"
    meta_path = world_dir / "meta.json"

    documents_data = _load_jsonl(docs_path)
    clauses_data = _load_jsonl(clauses_path)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    documents: List[Document] = [Document(**data) for data in documents_data]
    clauses: Dict[str, Clause] = {data["clause_id"]: Clause(**data) for data in clauses_data}
    return World(documents=documents, clauses=clauses, meta=meta)


def load_tasks(tasks_path: Path) -> List[Task]:
    data = _load_jsonl(tasks_path)
    tasks: List[Task] = []
    for item in data:
        if isinstance(item.get("gold"), dict) and not isinstance(item.get("gold"), Gold):
            item["gold"] = Gold(**item["gold"])
        tasks.append(Task(**item))
    return tasks
