from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from .baselines import _parse_prediction
from .symbolic_judge import judge_from_opened_clauses

_ALLOWED_DECISIONS = {"allow", "deny", "require_condition", "needs_more_info"}


def _extract_retention_days(text: str | None) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"retention\s*days?\s*[:=]\s*(\d+)", str(text), flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d+)\s*days?\b", str(text), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_decision(decision: Any) -> str:
    d = str(decision or "").strip()
    if d in _ALLOWED_DECISIONS:
        return d
    return "needs_more_info"


def _parse_context_override(ticket_text: str | None) -> Dict[str, str]:
    if not ticket_text:
        return {}
    m = re.search(
        r"updated\s+context\s+override:\s*([a-z_]+)\s*=\s*([a-zA-Z0-9_:+-]+)",
        str(ticket_text),
        flags=re.IGNORECASE,
    )
    if not m:
        return {}
    return {str(m.group(1)).strip(): str(m.group(2)).strip()}


def _parse_required_condition(ticket_text: str | None) -> Optional[str]:
    if not ticket_text:
        return None
    m = re.search(
        r"updated\s+constraint:\s*require_condition\s*=\s*([a-zA-Z0-9_]+)",
        str(ticket_text),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return str(m.group(1)).strip().lower()


def _norm_condition(value: Any) -> str:
    return str(value or "").strip().lower()


def _context_for_ticket(task: Any, ticket_override: str | None) -> Dict[str, Any]:
    base = dict(getattr(task, "context", None) or {})
    if not ticket_override:
        return base
    days = _extract_retention_days(ticket_override)
    if days is not None:
        base["retention_days"] = int(days)
        base["retention_bucket"] = "le_30" if int(days) <= 30 else "gt_30"
    for key, value in _parse_context_override(ticket_override).items():
        base[str(key)] = value
    return base


def judge_output(
    task: Any,
    output_text: str,
    *,
    world: Any,
    judge_name: str = "symbolic_packed",
    ticket_override: str | None = None,
    opened_clause_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if judge_name != "symbolic_packed":
        raise ValueError(f"Unsupported judge_name={judge_name!r}; only symbolic_packed is supported.")

    prediction = _parse_prediction(output_text or "")
    predicted_decision = _normalize_decision(prediction.get("decision"))
    evidence_ids = list(prediction.get("evidence") or [])
    if opened_clause_ids is None:
        opened_clause_ids = evidence_ids
    opened_unique = []
    seen = set()
    for cid in opened_clause_ids:
        sval = str(cid).strip()
        if not sval or sval in seen:
            continue
        seen.add(sval)
        opened_unique.append(sval)

    task_for_judge = copy.deepcopy(task)
    task_for_judge.context = _context_for_ticket(task_for_judge, ticket_override)
    if ticket_override is not None:
        task_for_judge.user_ticket = str(ticket_override)

    judged = judge_from_opened_clauses(task_for_judge, opened_unique, world)
    expected_decision = _normalize_decision(judged.get("decision"))
    correct = predicted_decision == expected_decision
    required_condition = _parse_required_condition(ticket_override)
    if required_condition:
        expected_decision = "require_condition"
        pred_conditions = [
            _norm_condition(c) for c in list(prediction.get("conditions") or [])
        ]
        has_required = required_condition in pred_conditions
        correct = bool(predicted_decision == "require_condition" and has_required)

    return {
        "correct": bool(correct),
        "predicted_decision": predicted_decision,
        "expected_decision": expected_decision,
        "opened_clause_ids": opened_unique,
        "predicted_evidence_ids": evidence_ids,
        "ticket_override_days": _extract_retention_days(ticket_override),
        "ticket_context_override": _parse_context_override(ticket_override),
        "ticket_required_condition": required_condition,
    }
