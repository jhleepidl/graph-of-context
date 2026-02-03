from __future__ import annotations

from typing import Dict, List

SLOT_LABELS = {
    "export_identifiers": "exporting identifiers",
    "export_logs": "exporting logs",
    "share_health_data": "sharing health data",
    "retain_logs_90d": "retaining logs for 90 days",
}


def _format_scope(applies_if: Dict[str, List[str]]) -> List[str]:
    parts: List[str] = []
    if applies_if.get("region"):
        parts.append(f"Regions: {', '.join(applies_if['region'])}.")
    if applies_if.get("product"):
        parts.append(f"Products: {', '.join(applies_if['product'])}.")
    if applies_if.get("tier"):
        parts.append(f"Tiers: {', '.join(applies_if['tier'])}.")
    if applies_if.get("data_type"):
        parts.append(f"Data types: {', '.join(applies_if['data_type'])}.")
    if applies_if.get("purpose"):
        parts.append(f"Purposes: {', '.join(applies_if['purpose'])}.")
    return parts


def render_definition(term_label: str, definition: str) -> str:
    return (
        f"For the purposes of this policy, '{term_label}' include {definition}. "
        "Use this definition when interpreting related rules."
    )


def render_priority(authority_a: str, authority_b: str) -> str:
    return (
        f"In the event of conflict, {authority_a} guidance overrides {authority_b} guidance. "
        "Follow the higher-priority authority when rules disagree."
    )


def render_clause_text(
    *,
    kind: str,
    slot: str,
    applies_if: Dict[str, List[str]],
    decision: str,
    conditions: List[str],
    term_label: str | None = None,
    term_definition: str | None = None,
    override_hint: str | None = None,
) -> str:
    scope_parts = _format_scope(applies_if)
    slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))

    if kind == "definition" and term_label and term_definition:
        return render_definition(term_label, term_definition)

    if kind == "priority" and override_hint:
        return override_hint

    if kind == "update":
        line1 = f"Effective immediately, this notice supersedes prior guidance on {slot_text}."
        line2 = "Use this update when prior documents conflict or are outdated."
        line3 = " ".join(scope_parts) if scope_parts else "Scope matches the affected programs."
        return f"{line1} {line2} {line3}"

    if kind == "exception":
        line1 = f"This is an exception to the general rule about {slot_text}."
    elif kind == "procedure":
        line1 = f"This procedure applies to {slot_text}."
    else:
        line1 = f"This rule covers {slot_text}."

    if decision == "allow":
        line2 = "The activity is allowed under the stated scope."
    elif decision == "deny":
        line2 = "The activity is not allowed under the stated scope."
    elif decision == "require_condition":
        line2 = "The activity is allowed only after the listed conditions are met."
    else:
        line2 = "More information is required before a decision can be made."

    if conditions:
        line2 += f" Conditions: {', '.join(conditions)}."

    scope_line = " ".join(scope_parts) if scope_parts else "Scope is global unless otherwise restricted."
    return f"{line1} {line2} {scope_line}"
