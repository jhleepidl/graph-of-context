from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceGold:
    decision: str
    conditions: List[str]
    evidence_ids: List[str]
    evidence_core_ids: List[str] = field(default_factory=list)
    evidence_meta_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TraceGold":
        return TraceGold(
            decision=str(data.get("decision", "needs_more_info")),
            conditions=list(data.get("conditions") or []),
            evidence_ids=list(data.get("evidence_ids") or []),
            evidence_core_ids=list(data.get("evidence_core_ids") or []),
            evidence_meta_ids=list(data.get("evidence_meta_ids") or []),
        )


@dataclass
class TraceWorldClause:
    clause_id: str
    thread_id: str
    step_idx: int
    node_type: str
    text: str
    state_key: Optional[str] = None
    state_value: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    branch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TraceWorldClause":
        return TraceWorldClause(
            clause_id=str(data.get("clause_id", "")),
            thread_id=str(data.get("thread_id", "")),
            step_idx=int(data.get("step_idx", 0) or 0),
            node_type=str(data.get("node_type", "ASSUMPTION")),
            text=str(data.get("text", "")),
            state_key=(str(data.get("state_key")) if data.get("state_key") is not None else None),
            state_value=(str(data.get("state_value")) if data.get("state_value") is not None else None),
            depends_on=list(data.get("depends_on") or []),
            tags=list(data.get("tags") or []),
            branch_id=(str(data.get("branch_id")) if data.get("branch_id") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TraceStep:
    step_id: str
    thread_id: str
    step_idx: int
    kind: str
    message: str
    state: Dict[str, Any]
    introduced_clause_ids: List[str] = field(default_factory=list)
    avoid_target_ids: List[str] = field(default_factory=list)
    pivot_required_ids: List[str] = field(default_factory=list)
    gold: Optional[TraceGold] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gold"] = self.gold.to_dict() if self.gold else None
        return payload

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TraceStep":
        raw_gold = data.get("gold")
        gold = TraceGold.from_dict(raw_gold) if isinstance(raw_gold, dict) else None
        return TraceStep(
            step_id=str(data.get("step_id", "")),
            thread_id=str(data.get("thread_id", "")),
            step_idx=int(data.get("step_idx", 0) or 0),
            kind=str(data.get("kind", "explore")),
            message=str(data.get("message", "")),
            state=dict(data.get("state") or {}),
            introduced_clause_ids=list(data.get("introduced_clause_ids") or []),
            avoid_target_ids=list(data.get("avoid_target_ids") or []),
            pivot_required_ids=list(data.get("pivot_required_ids") or []),
            gold=gold,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TraceThread:
    thread_id: str
    level: int
    scenario: str
    initial_state: Dict[str, Any]
    steps: List[TraceStep]
    clauses: Dict[str, TraceWorldClause]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "level": int(self.level),
            "scenario": str(self.scenario),
            "initial_state": dict(self.initial_state),
            "steps": [step.to_dict() for step in self.steps],
            "clauses": {cid: clause.to_dict() for cid, clause in self.clauses.items()},
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TraceThread":
        raw_steps = data.get("steps") or []
        raw_clauses = data.get("clauses") or {}
        steps = [TraceStep.from_dict(item) for item in raw_steps if isinstance(item, dict)]
        clauses = {
            str(cid): TraceWorldClause.from_dict(item)
            for cid, item in raw_clauses.items()
            if isinstance(item, dict)
        }
        return TraceThread(
            thread_id=str(data.get("thread_id", "")),
            level=int(data.get("level", 1) or 1),
            scenario=str(data.get("scenario", "mixed")),
            initial_state=dict(data.get("initial_state") or {}),
            steps=steps,
            clauses=clauses,
            meta=dict(data.get("meta") or {}),
        )
