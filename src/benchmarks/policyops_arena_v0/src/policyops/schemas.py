from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    USE_PYDANTIC = True
except Exception:  # pragma: no cover - optional dependency
    BaseModel = object
    Field = None
    USE_PYDANTIC = False

if not USE_PYDANTIC:
    from dataclasses import dataclass, field, asdict


if USE_PYDANTIC:

    class _BaseModel(BaseModel):
        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        class Config:
            extra = "forbid"


    class Document(_BaseModel):
        doc_id: str
        doc_type: str
        title: str
        published_at: str
        authority: str
        jurisdiction: List[str]
        applies_to: Dict[str, List[str]]
        sections: List[str]


    class Clause(_BaseModel):
        clause_id: str
        doc_id: str
        published_at: str
        authority: str
        text: str
        kind: str
        slot: str
        applies_if: Dict[str, List[str]]
        effect: Dict[str, str]
        conditions: List[str]
        targets: Dict[str, List[str]]
        terms_used: List[str]
        canonical_terms: List[str] = Field(default_factory=list)
        aliases: List[str] = Field(default_factory=list)
        bridge_for_slot: Optional[str] = None
        bridge_targets: List[str] = Field(default_factory=list)
        has_update_keywords: bool = False
        is_bridge_doc: bool = False


    class Gold(_BaseModel):
        decision: str
        conditions: List[str]
        gold_evidence: List[str]
        gold_evidence_core: List[str] = Field(default_factory=list)
        gold_evidence_meta: List[str] = Field(default_factory=list)


    class Task(_BaseModel):
        task_id: str
        timestamp: str
        user_ticket: str
        context: Dict[str, Any]
        budgets: Dict[str, int]
        gold: Gold
        scenario_mode: str = "v0"
        slot_hint_alias: Optional[str] = None
        canonical_slot_term: Optional[str] = None
        bridge_clause_id: Optional[str] = None
        needs_update_resolution: bool = False


    class World(_BaseModel):
        documents: List[Document]
        clauses: Dict[str, Clause]
        meta: Dict[str, Any] = Field(default_factory=dict)

else:

    @dataclass
    class Document:
        doc_id: str
        doc_type: str
        title: str
        published_at: str
        authority: str
        jurisdiction: List[str]
        applies_to: Dict[str, List[str]]
        sections: List[str]

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


    @dataclass
    class Clause:
        clause_id: str
        doc_id: str
        published_at: str
        authority: str
        text: str
        kind: str
        slot: str
        applies_if: Dict[str, List[str]]
        effect: Dict[str, str]
        conditions: List[str]
        targets: Dict[str, List[str]]
        terms_used: List[str]
        canonical_terms: List[str] = field(default_factory=list)
        aliases: List[str] = field(default_factory=list)
        bridge_for_slot: Optional[str] = None
        bridge_targets: List[str] = field(default_factory=list)
        has_update_keywords: bool = False
        is_bridge_doc: bool = False

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


    @dataclass
    class Gold:
        decision: str
        conditions: List[str]
        gold_evidence: List[str]
        gold_evidence_core: List[str] = field(default_factory=list)
        gold_evidence_meta: List[str] = field(default_factory=list)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


    @dataclass
    class Task:
        task_id: str
        timestamp: str
        user_ticket: str
        context: Dict[str, Any]
        budgets: Dict[str, int]
        gold: Gold
        scenario_mode: str = "v0"
        slot_hint_alias: Optional[str] = None
        canonical_slot_term: Optional[str] = None
        bridge_clause_id: Optional[str] = None
        needs_update_resolution: bool = False

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


    @dataclass
    class World:
        documents: List[Document]
        clauses: Dict[str, Clause]
        meta: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


def model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if USE_PYDANTIC and hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj
