from __future__ import annotations

from typing import Dict, List

SLOT_LABELS = {
    "export_identifiers": "exporting identifiers",
    "export_logs": "exporting logs",
    "share_health_data": "sharing health data",
    "retain_logs_90d": "retaining logs for 90 days",
}

RETENTION_BUCKET_TEXT = {
    "le_30": "Retention days <= 30.",
    "gt_30": "Retention days > 30.",
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
    if applies_if.get("retention_bucket"):
        retention_parts = [
            RETENTION_BUCKET_TEXT.get(bucket, f"Retention bucket: {bucket}.")
            for bucket in applies_if["retention_bucket"]
        ]
        parts.append(" ".join(retention_parts))
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
        scope_summary = " ".join(scope_parts) if scope_parts else "All regions and tiers."
        if decision == "allow":
            if conditions:
                line1 = (
                    f"Effective immediately, this update clarifies that {slot_text} is permitted "
                    f"under the following conditions: {', '.join(conditions)}."
                )
            else:
                line1 = (
                    f"Effective immediately, this update clarifies that {slot_text} is permitted "
                    "without additional conditions."
                )
            line2 = "This update replaces prior guidance and should be followed going forward."
            line3 = scope_summary
        elif decision == "deny":
            line1 = (
                f"Effective immediately, this update states that {slot_text} is not permitted "
                f"for the following scope: {scope_summary}"
            )
            line2 = "This update replaces prior guidance and should be followed going forward."
            line3 = ""
        elif decision == "require_condition":
            if conditions:
                line1 = (
                    f"Effective immediately, {slot_text} is permitted only if the following "
                    f"requirements are met: {', '.join(conditions)}."
                )
            else:
                line1 = (
                    f"Effective immediately, {slot_text} is permitted only if additional "
                    "requirements are met."
                )
            line2 = "This update replaces prior guidance and should be followed going forward."
            line3 = scope_summary
        else:
            line1 = f"Effective immediately, this update revises prior guidance on {slot_text}."
            line2 = "Follow this update when prior documents conflict or are outdated."
            line3 = scope_summary
        return " ".join([part for part in [line1, line2, line3] if part])

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
