from __future__ import annotations

import json
import random
import hashlib
import re
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .render import SLOT_LABELS, render_clause_text, render_priority
from .schemas import Clause, Document, Gold, Task, World
from .world import evaluate_context

SLOTS = ["export_identifiers", "export_logs", "share_health_data", "retain_logs_90d"]

DEFAULT_AUTHORITIES = ["security", "legal", "product", "support"]
DEFAULT_PRODUCTS = ["alpha", "beta"]
DEFAULT_TIERS = ["free", "pro", "enterprise"]
DEFAULT_REGIONS = ["GLOBAL", "EU", "US", "KR"]
DEFAULT_DATA_TYPES = ["logs", "identifiers", "health", "financial", "telemetry"]
DEFAULT_PURPOSES = ["debugging", "analytics", "marketing", "security", "legal_request"]

UPDATE_DOC_TYPES = ["announcement", "release_note"]
POLICY_DOC_TYPES = ["policy", "faq", "notice"]

CONDITION_KEYS = [
    "security_review",
    "legal_approval",
    "dpa_required",
    "data_minimization",
    "consent_required",
    "encryption_required",
]

TERM_LIBRARY = [
    ("TERM_IDENTIFIERS", "Identifiers", "account IDs, device IDs, or user handles"),
    ("TERM_SENSITIVE_DATA", "Sensitive Data", "health, financial, or biometric information"),
    ("TERM_LOGS", "Logs", "event logs, access logs, and audit logs"),
    ("TERM_HEALTH_DATA", "Health Data", "diagnosis, treatment, or clinical notes"),
    ("TERM_FINANCIAL", "Financial Data", "payment details, invoices, or billing status"),
    ("TERM_TELEMETRY", "Telemetry", "system metrics and performance data"),
    ("TERM_EXPORT", "Export", "transfer of data outside the hosting region"),
    ("TERM_SHARE", "Share", "disclosure to external third parties"),
    ("TERM_RETENTION", "Retention Window", "the allowed duration of stored records"),
    ("TERM_LEGAL_HOLD", "Legal Hold", "a preservation requirement for legal matters"),
    ("TERM_SECURITY_INCIDENT", "Security Incident", "confirmed compromise or suspected breach"),
    ("TERM_SUPPORT_CASE", "Support Case", "a customer-reported operational issue"),
    ("TERM_ANALYTICS", "Analytics", "measurement of product usage and behavior"),
    ("TERM_MARKETING", "Marketing Use", "promotional or outreach activities"),
    ("TERM_EU_RESTRICTED", "EU Restricted Data", "data subject to EU localization"),
    ("TERM_US_RESTRICTED", "US Restricted Data", "data subject to US export limits"),
    ("TERM_CONSENT", "Consent", "explicit permission from the data subject"),
    ("TERM_DELETION", "Deletion Request", "a request to erase stored data"),
    ("TERM_MINIMIZATION", "Data Minimization", "limiting data to necessary scope"),
    ("TERM_AUTHORIZED_RECIPIENT", "Authorized Recipient", "a vetted and approved recipient"),
]

SLOT_TERMS = {
    "export_identifiers": ["TERM_IDENTIFIERS", "TERM_EXPORT"],
    "export_logs": ["TERM_LOGS", "TERM_EXPORT"],
    "share_health_data": ["TERM_HEALTH_DATA", "TERM_SHARE"],
    "retain_logs_90d": ["TERM_LOGS", "TERM_RETENTION"],
}

BRIDGE_TERMS = {
    "export_identifiers": ("export identifiers", "identifier export"),
    "export_logs": ("export logs", "telemetry export"),
    "share_health_data": ("health data sharing", "clinical data sharing"),
    "retain_logs_90d": ("retain logs 90 days", "log retention 90 days"),
}

UPDATE_KEYWORDS = ["effective immediately", "supersedes", "amendment", "revoke"]
RETENTION_BUCKETS = ["le_30", "gt_30"]
DECOY_CLAUSE_COUNT = 2
DECOY_MIN_CHARS = 250
DECOY_MAX_CHARS = 320
JITTER_FILLER_FRAGMENT = "(Additional context.)"
JITTER_SCOPE_VALUES = {"decoy_only", "decoy_plus_noncritical", "all"}
LITM_FILLER_POSITIONS = ["pre", "between", "post"]


def _extract_retention_days_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"retention\s*days?\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d+)\s*days?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _make_pivot_ticket_update(
    ticket_initial: str,
    pivot_type: str,
    *,
    rng: random.Random,
    task: Optional[Task] = None,
) -> str:
    initial = str(ticket_initial or "")
    kind_raw = str(pivot_type or "retention_flip")
    kind = {
        "action_flip": "entity_switch",
        "exception_add": "constraint_add",
    }.get(kind_raw, kind_raw)
    if kind == "retention_flip":
        old_days = _extract_retention_days_from_text(initial)
        if old_days is None:
            old_days = 90
        new_days = 30 if old_days > 30 else 90
        updated = initial
        replaced = False
        if re.search(r"retention\s*days?\s*[:=]\s*\d+", updated, flags=re.IGNORECASE):
            updated = re.sub(
                r"(retention\s*days?\s*[:=]\s*)\d+",
                rf"\g<1>{new_days}",
                updated,
                count=1,
                flags=re.IGNORECASE,
            )
            replaced = True
        if not replaced:
            pattern = rf"\b{old_days}\s*days?\b"
            if re.search(pattern, updated, flags=re.IGNORECASE):
                updated = re.sub(
                    pattern,
                    f"{new_days} days",
                    updated,
                    count=1,
                    flags=re.IGNORECASE,
                )
                replaced = True
        if not replaced:
            updated = (
                f"{initial}\n\nUpdated requirement: retain logs for {new_days} days, not {old_days} days."
            )
        return updated
    if kind == "entity_switch":
        context = dict(getattr(task, "context", None) or {})
        switch_spec: List[Tuple[str, List[str]]] = [
            ("region", list(DEFAULT_REGIONS)),
            ("product", list(DEFAULT_PRODUCTS)),
            ("tier", list(DEFAULT_TIERS)),
            ("data_type", list(DEFAULT_DATA_TYPES)),
            ("purpose", list(DEFAULT_PURPOSES)),
        ]
        rng.shuffle(switch_spec)
        chosen_key = None
        old_val = None
        new_val = None
        for key, options in switch_spec:
            cur = str(context.get(key, "") or "")
            if not cur:
                continue
            candidates = [v for v in options if str(v) != cur]
            if not candidates:
                continue
            chosen_key = key
            old_val = cur
            new_val = str(rng.choice(candidates))
            break
        if chosen_key and old_val and new_val:
            updated = initial
            # Replace only one mention to keep ticket shape stable.
            updated = re.sub(
                rf"\b{re.escape(old_val)}\b",
                new_val,
                updated,
                count=1,
            )
            if updated.strip() == initial.strip():
                updated = (
                    f"{initial}\n\nUpdated context override: {chosen_key}={new_val} "
                    f"(was {old_val})."
                )
            else:
                updated = (
                    f"{updated}\n\nUpdated context override: {chosen_key}={new_val} "
                    f"(was {old_val})."
                )
            return updated
        return f"{initial}\n\nUpdated context override: region=EU."
    if kind == "constraint_add":
        cond = str(rng.choice(CONDITION_KEYS))
        return (
            f"{initial}\n\nUpdated constraint: require_condition={cond}. "
            f"This constraint must be reflected in the final answer."
        )
    return initial


def _ensure_real_pivot_change(
    initial: str,
    updated: str,
    *,
    pivot_type: str,
    task: Optional[Task],
    rng: random.Random,
) -> str:
    if str(updated or "").strip() != str(initial or "").strip():
        return updated
    # Bounded retries first.
    for _ in range(4):
        retry = _make_pivot_ticket_update(initial, pivot_type, rng=rng, task=task)
        if str(retry or "").strip() != str(initial or "").strip():
            return retry
    # Deterministic fallback to guarantee a real pivot.
    kind = {
        "action_flip": "entity_switch",
        "exception_add": "constraint_add",
    }.get(str(pivot_type or "retention_flip"), str(pivot_type or "retention_flip"))
    if kind == "retention_flip":
        old_days = _extract_retention_days_from_text(initial)
        if old_days is None:
            old_days = 90
        new_days = 30 if old_days > 30 else 90
        return (
            f"{initial}\n\nUpdated requirement: retain logs for {new_days} days, not {old_days} days."
        )
    if kind == "entity_switch":
        return (
            f"{initial}\n\nUpdated context override: region=EU. "
            "Final decision must follow the updated context."
        )
    if kind == "constraint_add":
        return (
            f"{initial}\n\nUpdated constraint: require_condition=legal_approval."
        )
    return f"{initial}\n\nUpdated requirement: apply latest ticket update."


def _apply_late_pivot_updates(
    tasks: List[Task],
    *,
    pivot_rate: float,
    pivot_type: str,
    seed: int,
) -> None:
    for task in tasks:
        if not getattr(task, "ticket_initial", None):
            task.ticket_initial = str(getattr(task, "user_ticket", "") or "")
        task.user_ticket = str(task.ticket_initial or task.user_ticket or "")
        task.ticket_updated = None
        task.pivot_type = None

    if float(pivot_rate or 0.0) <= 0.0:
        return

    rng = random.Random(seed + 9901)
    by_thread: Dict[str, List[Task]] = {}
    for task in tasks:
        key = str(getattr(task, "thread_id", None) or f"__task__:{task.task_id}")
        by_thread.setdefault(key, []).append(task)

    for key in sorted(by_thread.keys()):
        if rng.random() > float(pivot_rate):
            continue
        group = sorted(
            by_thread[key],
            key=lambda t: (
                int(getattr(t, "episode_id", 0) or 0),
                str(getattr(t, "task_id", "")),
            ),
        )
        pivot_task = next(
            (task for task in reversed(group) if int(getattr(task, "episode_id", 0) or 0) == 3),
            group[-1],
        )
        initial = str(getattr(pivot_task, "ticket_initial", None) or pivot_task.user_ticket or "")
        normalized_pivot_type = {
            "action_flip": "entity_switch",
            "exception_add": "constraint_add",
        }.get(str(pivot_type or "retention_flip"), str(pivot_type or "retention_flip"))
        updated = _make_pivot_ticket_update(
            initial,
            normalized_pivot_type,
            rng=rng,
            task=pivot_task,
        )
        updated = _ensure_real_pivot_change(
            initial,
            updated,
            pivot_type=normalized_pivot_type,
            task=pivot_task,
            rng=rng,
        )
        if not updated or updated.strip() == initial.strip():
            continue
        pivot_task.ticket_updated = updated
        pivot_task.pivot_type = str(normalized_pivot_type or "retention_flip")


def _make_applies_if(
    rng: random.Random,
    regions: List[str],
    products: List[str],
    tiers: List[str],
    data_types: List[str],
    purposes: List[str],
    min_keys: int,
    max_keys: int,
    retention_buckets: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    keys = ["region", "product", "tier", "data_type", "purpose"]
    num_keys = rng.randint(min_keys, max_keys)
    selected = rng.sample(keys, k=num_keys) if num_keys > 0 else []
    applies_if: Dict[str, List[str]] = {}
    if "region" in selected:
        applies_if["region"] = [rng.choice(regions)]
    if "product" in selected:
        applies_if["product"] = [rng.choice(products)]
    if "tier" in selected:
        applies_if["tier"] = [rng.choice(tiers)]
    if "data_type" in selected:
        applies_if["data_type"] = [rng.choice(data_types)]
    if "purpose" in selected:
        applies_if["purpose"] = [rng.choice(purposes)]
    if retention_buckets:
        applies_if["retention_bucket"] = [rng.choice(retention_buckets)]
    return applies_if


def _invert_decision(decision: str) -> str:
    if decision == "deny":
        return "allow"
    if decision == "allow":
        return "deny"
    return "allow"


def _make_conditions(rng: random.Random, max_conditions: int = 2) -> List[str]:
    num = rng.randint(0, max_conditions)
    return rng.sample(CONDITION_KEYS, k=num) if num else []


def _next_clause_id(counter: int) -> str:
    return f"C{counter:04d}"


def _make_decoy_text(
    *,
    slot: str,
    alias: str,
    canonical: str,
    tag: str,
    decoy_label: str,
    case_token: str,
    min_chars: int,
    max_chars: int,
    context: Optional[Dict[str, Any]] = None,
    rng: random.Random,
) -> str:
    slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
    context_line = ""
    if context:
        plan_bits = []
        if context.get("product"):
            plan_bits.append(str(context["product"]))
        if context.get("tier"):
            plan_bits.append(str(context["tier"]))
        plan = " ".join(plan_bits)
        region = context.get("region")
        data_type = context.get("data_type")
        purpose = context.get("purpose")
        scope_parts = []
        if plan:
            scope_parts.append(plan)
        if region:
            scope_parts.append(f"in {region}")
        if scope_parts:
            context_line = f"Scope: {' '.join(scope_parts)}."
        if data_type:
            context_line = f"{context_line} Data type: {data_type}.".strip()
        if purpose:
            context_line = f"{context_line} Purpose: {purpose}.".strip()
    case_line = f"Casecode {case_token}. {case_token}. {case_token}."
    base = (
        f"Case ref {tag} (decoy {decoy_label}): This memo discusses {alias}. "
        f"It summarizes {alias} guidance for {slot_text}. "
        f"{case_line} "
        f"{context_line} "
        "This guidance is informational and does not change policy obligations. "
    )
    filler = [
        "This section provides background context and historical notes.",
        "Operational notes are included to assist support teams and audits.",
        "Examples and non-binding commentary appear below for reference.",
        "Additional narrative details are included for completeness.",
    ]
    text = base
    while len(text) < max_chars:
        text += " " + rng.choice(filler)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def apply_length_jitter(text: str, jitter_chars: int) -> str:
    if jitter_chars <= 0:
        return text
    filler = " " + JITTER_FILLER_FRAGMENT
    repeats = max(1, (int(jitter_chars) + len(filler) - 1) // len(filler))
    return f"{text}{filler * repeats}"


def _deterministic_jitter_chars(
    seed: int,
    clause_id: str,
    max_chars: int,
    *,
    thread_hint: str = "",
) -> int:
    if max_chars <= 0:
        return 0
    step = 50
    values = list(range(0, max_chars + 1, step))
    if values[-1] != max_chars:
        values.append(max_chars)
    digest = hashlib.md5(f"{seed}:{thread_hint}:{clause_id}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(values)
    return int(values[idx])


def _deterministic_int_in_range(
    seed: int,
    key: str,
    min_value: int,
    max_value: int,
) -> int:
    if max_value <= min_value:
        return int(min_value)
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    span = int(max_value) - int(min_value) + 1
    return int(min_value) + (int(digest[:8], 16) % span)


def _make_litm_filler_text(
    *,
    slot: str,
    alias: str,
    canonical: str,
    thread_id: str,
    filler_index: int,
    min_chars: int = 140,
) -> str:
    slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
    base = (
        f"Thread {thread_id} recap note {filler_index}: {alias} / {canonical} "
        f"context for {slot_text}. This note is for readability only and does not "
        "introduce policy conditions or exceptions."
    )
    if len(base) >= min_chars:
        return base
    needed = max(0, min_chars - len(base))
    filler = apply_length_jitter("", needed).strip()
    if filler:
        return f"{base} {filler}"
    return base


def _decoy_applies_if(
    ctx: Dict[str, Any],
    *,
    products: List[str],
    tiers: List[str],
    regions: List[str],
    data_types: List[str],
    purposes: List[str],
    rng: random.Random,
) -> Dict[str, List[str]]:
    def _alt(options: List[str], value: str) -> List[str]:
        candidates = [v for v in options if v != value]
        if candidates:
            return [rng.choice(candidates)]
        return [value]

    applies_if: Dict[str, List[str]] = {}
    if ctx.get("region") is not None:
        applies_if["region"] = _alt(regions, ctx["region"])
    if ctx.get("product") is not None:
        applies_if["product"] = _alt(products, ctx["product"])
    if ctx.get("tier") is not None:
        applies_if["tier"] = _alt(tiers, ctx["tier"])
    if ctx.get("data_type") is not None:
        applies_if["data_type"] = _alt(data_types, ctx["data_type"])
    if ctx.get("purpose") is not None:
        applies_if["purpose"] = _alt(purposes, ctx["purpose"])
    if ctx.get("retention_bucket") is not None:
        applies_if["retention_bucket"] = [
            bucket
            for bucket in RETENTION_BUCKETS
            if bucket != ctx.get("retention_bucket")
        ] or [ctx.get("retention_bucket")]
    return applies_if


def _build_clause(
    clause_id: str,
    kind: str,
    slot: str,
    applies_if: Dict[str, List[str]],
    decision: str,
    conditions: List[str],
    terms_used: List[str],
    targets: Optional[Dict[str, List[str]]] = None,
    term_label: Optional[str] = None,
    term_definition: Optional[str] = None,
    override_hint: Optional[str] = None,
    canonical_terms: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    bridge_for_slot: Optional[str] = None,
    bridge_targets: Optional[List[str]] = None,
) -> Clause:
    targets = targets or {"overrides": [], "revokes": [], "defines": []}
    text = render_clause_text(
        kind=kind,
        slot=slot,
        applies_if=applies_if,
        decision=decision,
        conditions=conditions,
        term_label=term_label,
        term_definition=term_definition,
        override_hint=override_hint,
    )
    canonical_terms = canonical_terms or []
    aliases = aliases or []
    bridge_targets = bridge_targets or []
    has_update_keywords = kind == "update" or any(kw in text.lower() for kw in UPDATE_KEYWORDS)
    return Clause(
        clause_id=clause_id,
        doc_id="",
        published_at="",
        authority="",
        text=text,
        kind=kind,
        slot=slot,
        applies_if=applies_if,
        effect={"decision": decision},
        conditions=conditions,
        targets=targets,
        terms_used=terms_used,
        canonical_terms=canonical_terms,
        aliases=aliases,
        bridge_for_slot=bridge_for_slot,
        bridge_targets=bridge_targets,
        has_update_keywords=has_update_keywords,
    )


def _replace_slot_text(text: str, slot: str, replacement: str) -> str:
    slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
    if slot_text in text:
        return text.replace(slot_text, replacement)
    return text


def generate_world(
    *,
    seed: int = 0,
    n_docs: int = 30,
    clauses_per_doc: int = 5,
    exception_chain_depth: int = 2,
    definition_dependency_depth: int = 1,
    definition_dependency_extra_terms: int = 0,
    force_exception_chain_depth: int = 0,
    force_exception_chain_all_apply: bool = False,
    update_rate: float = 0.3,
    definition_density: float = 0.4,
    distractor_strength: float = 0.3,
    scenario_mode: str = "v0",
    bridge_prob: float = 0.8,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    canonical_density: float = 0.95,
    bridge_kind: str = "definition",
    authorities: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
) -> World:
    rng = random.Random(seed)
    authorities = authorities or DEFAULT_AUTHORITIES
    products = products or DEFAULT_PRODUCTS
    tiers = tiers or DEFAULT_TIERS
    regions = regions or DEFAULT_REGIONS
    data_types = data_types or DEFAULT_DATA_TYPES
    purposes = purposes or DEFAULT_PURPOSES
    retention_buckets = (
        RETENTION_BUCKETS
        if scenario_mode in {"threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}
        else None
    )

    total_slots = n_docs * clauses_per_doc
    clause_counter = 1

    base_date = datetime(2025, 1, 1)
    documents: List[Document] = []
    for idx in range(n_docs):
        doc_id = f"D{idx + 1:04d}"
        published_at = (base_date + timedelta(days=idx * 3)).strftime("%Y-%m-%d")
        authority = rng.choice(authorities)
        documents.append(
            Document(
                doc_id=doc_id,
                doc_type="policy",
                title=f"{authority.title()} Policy {idx + 1}",
                published_at=published_at,
                authority=authority,
                jurisdiction=list(regions),
                applies_to={"products": list(products), "tiers": list(tiers)},
                sections=[],
            )
        )

    doc_capacity = {doc.doc_id: clauses_per_doc for doc in documents}
    doc_lookup = {doc.doc_id: doc for doc in documents}
    update_start_idx = max(0, int(n_docs * 0.6))

    def pick_doc(kind: str) -> Document:
        if kind == "update":
            candidates = [
                doc
                for i, doc in enumerate(documents)
                if i >= update_start_idx and doc_capacity[doc.doc_id] > 0
            ]
        else:
            candidates = [doc for doc in documents if doc_capacity[doc.doc_id] > 0]
        if not candidates:
            candidates = [doc for doc in documents if doc_capacity[doc.doc_id] > 0]
        if not candidates:
            raise RuntimeError("No document slots available")
        doc = rng.choice(candidates)
        doc_capacity[doc.doc_id] -= 1
        return doc

    clauses: List[Clause] = []

    # Base rules for each slot.
    base_clauses: Dict[str, Clause] = {}
    for slot in SLOTS:
        decision = rng.choice(["allow", "deny", "require_condition"])
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        base_applies_if = (
            _make_applies_if(
                rng,
                regions,
                products,
                tiers,
                data_types,
                purposes,
                min_keys=0,
                max_keys=1,
                retention_buckets=retention_buckets,
            )
            if retention_buckets
            else {}
        )
        clause = _build_clause(
            clause_id=clause_id,
            kind="rule",
            slot=slot,
            applies_if=base_applies_if,
            decision=decision,
            conditions=_make_conditions(rng) if decision == "require_condition" else [],
            terms_used=SLOT_TERMS.get(slot, []),
        )
        clauses.append(clause)
        base_clauses[slot] = clause

    # Priority meta rules.
    priority_pairs = [("security", "support"), ("legal", "product")]
    priority_clause_ids: List[str] = []
    for authority_a, authority_b in priority_pairs:
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        clause = _build_clause(
            clause_id=clause_id,
            kind="priority",
            slot="meta",
            applies_if={},
            decision="needs_more_info",
            conditions=[],
            terms_used=[],
            override_hint=render_priority(authority_a, authority_b),
        )
        clauses.append(clause)
        priority_clause_ids.append(clause_id)

    # Exception chains.
    available = total_slots - len(clauses)
    chain_depth = (
        int(force_exception_chain_depth)
        if int(force_exception_chain_depth) > 0
        else int(exception_chain_depth)
    )
    max_chain = min(chain_depth, max(0, available // max(1, len(SLOTS))))
    for slot in SLOTS:
        prev = base_clauses[slot]
        chain_applies_if = (
            dict(getattr(prev, "applies_if", {}) or {}) if force_exception_chain_all_apply else None
        )
        for _ in range(max_chain):
            clause_id = _next_clause_id(clause_counter)
            clause_counter += 1
            if force_exception_chain_all_apply:
                applies_if = dict(chain_applies_if or {})
            else:
                applies_if = _make_applies_if(
                    rng,
                    regions,
                    products,
                    tiers,
                    data_types,
                    purposes,
                    min_keys=2,
                    max_keys=3,
                    retention_buckets=retention_buckets,
                )
            decision = _invert_decision(prev.effect["decision"])
            clause = _build_clause(
                clause_id=clause_id,
                kind="exception",
                slot=slot,
                applies_if=applies_if,
                decision=decision,
                conditions=_make_conditions(rng),
                terms_used=SLOT_TERMS.get(slot, []),
                targets={"overrides": [prev.clause_id], "revokes": [], "defines": []},
            )
            clauses.append(clause)
            prev = clause

    # Update clauses.
    update_candidates = [c for c in clauses if c.kind in {"rule", "exception"}]
    num_updates = min(int(update_rate * len(update_candidates)), total_slots - len(clauses))
    if num_updates > 0:
        for target in rng.sample(update_candidates, k=num_updates):
            clause_id = _next_clause_id(clause_counter)
            clause_counter += 1
            decision = rng.choice(["allow", "deny", "require_condition"])
            clause = _build_clause(
                clause_id=clause_id,
                kind="update",
                slot=target.slot,
                applies_if=target.applies_if,
                decision=decision,
                conditions=_make_conditions(rng) if decision == "require_condition" else [],
                terms_used=SLOT_TERMS.get(target.slot, []),
                targets={"overrides": [], "revokes": [target.clause_id], "defines": []},
            )
            clauses.append(clause)

    # Definition clauses.
    target_defs = int(definition_density * total_slots)
    defs_to_add = min(target_defs, total_slots - len(clauses))
    term_cycle = list(TERM_LIBRARY)
    rng.shuffle(term_cycle)
    term_definitions: Dict[str, str] = {}
    created_definition_terms: List[str] = []
    for idx in range(defs_to_add):
        term_id, label, definition = term_cycle[idx % len(term_cycle)]
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        extra_terms: List[str] = []
        if int(definition_dependency_extra_terms) > 0 and created_definition_terms:
            # Create explicit definition dependencies (multi-hop definition evidence).
            extra_terms.append(created_definition_terms[-1])
            remaining = max(0, int(definition_dependency_extra_terms) - 1)
            if remaining > 0 and len(created_definition_terms) > 1:
                pool = created_definition_terms[:-1]
                take = min(remaining, len(pool))
                if take > 0:
                    extra_terms.extend(rng.sample(pool, k=take))
        terms_used = [term_id] + extra_terms
        clause = _build_clause(
            clause_id=clause_id,
            kind="definition",
            slot="definition",
            applies_if={},
            decision="needs_more_info",
            conditions=[],
            terms_used=terms_used,
            targets={"overrides": [], "revokes": [], "defines": [term_id]},
            term_label=label,
            term_definition=definition,
        )
        clauses.append(clause)
        term_definitions[term_id] = clause_id
        created_definition_terms.append(term_id)

    # Fill remaining slots.
    remaining = total_slots - len(clauses)
    for _ in range(remaining):
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        slot = rng.choice(SLOTS)
        roll = rng.random()
        if roll < distractor_strength:
            applies_if = _make_applies_if(
                rng,
                regions,
                products,
                tiers,
                data_types,
                purposes,
                min_keys=1,
                max_keys=2,
                retention_buckets=retention_buckets,
            )
            decision = rng.choice(["allow", "deny", "require_condition"])
            clause = _build_clause(
                clause_id=clause_id,
                kind="rule",
                slot=slot,
                applies_if=applies_if,
                decision=decision,
                conditions=_make_conditions(rng) if decision == "require_condition" else [],
                terms_used=SLOT_TERMS.get(slot, []),
            )
        elif roll < distractor_strength + 0.2:
            base = base_clauses[slot]
            applies_if = _make_applies_if(
                rng,
                regions,
                products,
                tiers,
                data_types,
                purposes,
                min_keys=1,
                max_keys=2,
                retention_buckets=retention_buckets,
            )
            clause = _build_clause(
                clause_id=clause_id,
                kind="exception",
                slot=slot,
                applies_if=applies_if,
                decision=_invert_decision(base.effect["decision"]),
                conditions=_make_conditions(rng),
                terms_used=SLOT_TERMS.get(slot, []),
                targets={"overrides": [base.clause_id], "revokes": [], "defines": []},
            )
        elif roll < distractor_strength + 0.35:
            applies_if = _make_applies_if(
                rng,
                regions,
                products,
                tiers,
                data_types,
                purposes,
                min_keys=1,
                max_keys=2,
                retention_buckets=retention_buckets,
            )
            clause = _build_clause(
                clause_id=clause_id,
                kind="procedure",
                slot=slot,
                applies_if=applies_if,
                decision="require_condition",
                conditions=_make_conditions(rng, max_conditions=2),
                terms_used=SLOT_TERMS.get(slot, []),
            )
        else:
            decision = rng.choice(["allow", "deny", "require_condition"])
            fallback_applies_if = (
                _make_applies_if(
                    rng,
                    regions,
                    products,
                    tiers,
                    data_types,
                    purposes,
                    min_keys=0,
                    max_keys=1,
                    retention_buckets=retention_buckets,
                )
                if retention_buckets
                else {}
            )
            clause = _build_clause(
                clause_id=clause_id,
                kind="rule",
                slot=slot,
                applies_if=fallback_applies_if,
                decision=decision,
                conditions=_make_conditions(rng) if decision == "require_condition" else [],
                terms_used=SLOT_TERMS.get(slot, []),
            )
        clauses.append(clause)

    # Assign clauses to documents.
    bridge_clause_by_slot: Dict[str, str] = {}
    extra_doc_ids: List[str] = []
    if scenario_mode == "bridged_v1_1":
        extra_docs: List[Document] = []
        for slot in SLOTS:
            alias, canonical = BRIDGE_TERMS.get(slot, (slot.replace("_", " "), slot.replace("_", " ")))
            clause_id = _next_clause_id(clause_counter)
            clause_counter += 1
            bridge_text = (
                f"For the purposes of this policy, '{canonical}' includes {alias} ('{alias}')."
                if bridge_kind == "definition"
                else f"Glossary: '{alias}' refers to '{canonical}'."
            )
            bridge_clause = Clause(
                clause_id=clause_id,
                doc_id="",
                published_at="",
                authority=rng.choice(authorities),
                text=bridge_text,
                kind=bridge_kind,
                slot="definition",
                applies_if={},
                effect={"decision": "needs_more_info"},
                conditions=[],
                targets={"overrides": [], "revokes": [], "defines": []},
                terms_used=[],
                canonical_terms=[canonical],
                aliases=[alias],
                bridge_for_slot=slot,
                bridge_targets=[canonical],
                has_update_keywords=False,
                is_bridge_doc=True,
            )
            clauses.append(bridge_clause)
            bridge_clause_by_slot[slot] = clause_id

            doc_id = f"D{len(documents) + len(extra_docs) + 1:04d}"
            published_at = (base_date + timedelta(days=(len(documents) + len(extra_docs)) * 3)).strftime(
                "%Y-%m-%d"
            )
            extra_docs.append(
                Document(
                    doc_id=doc_id,
                    doc_type="glossary" if bridge_kind == "glossary" else "policy",
                    title=f"Bridge {slot} {len(extra_docs) + 1}",
                    published_at=published_at,
                    authority=bridge_clause.authority,
                    jurisdiction=list(regions),
                    applies_to={"products": list(products), "tiers": list(tiers)},
                    sections=[clause_id],
                )
            )
            bridge_clause.doc_id = doc_id
            bridge_clause.published_at = published_at
            extra_doc_ids.append(doc_id)

        # Add distractor bridge-like clauses.
        slot_list = list(SLOTS)
        for slot in SLOTS:
            other_slot = rng.choice([s for s in slot_list if s != slot])
            alias, _canonical = BRIDGE_TERMS.get(slot, (slot.replace("_", " "), slot.replace("_", " ")))
            _alias_other, canonical_other = BRIDGE_TERMS.get(
                other_slot, (other_slot.replace("_", " "), other_slot.replace("_", " "))
            )
            clause_id = _next_clause_id(clause_counter)
            clause_counter += 1
            distract_text = (
                f"For the purposes of this policy, '{canonical_other}' includes {alias} ('{alias}')."
            )
            distract_clause = Clause(
                clause_id=clause_id,
                doc_id="",
                published_at="",
                authority=rng.choice(authorities),
                text=distract_text,
                kind=bridge_kind,
                slot="definition",
                applies_if={},
                effect={"decision": "needs_more_info"},
                conditions=[],
                targets={"overrides": [], "revokes": [], "defines": []},
                terms_used=[],
                canonical_terms=[canonical_other],
                aliases=[alias],
                bridge_for_slot=other_slot,
                bridge_targets=[canonical_other],
                has_update_keywords=False,
                is_bridge_doc=True,
            )
            clauses.append(distract_clause)
            doc_id = f"D{len(documents) + len(extra_docs) + 1:04d}"
            published_at = (base_date + timedelta(days=(len(documents) + len(extra_docs)) * 3)).strftime(
                "%Y-%m-%d"
            )
            extra_docs.append(
                Document(
                    doc_id=doc_id,
                    doc_type="glossary" if bridge_kind == "glossary" else "policy",
                    title=f"Distractor {slot} {len(extra_docs) + 1}",
                    published_at=published_at,
                    authority=distract_clause.authority,
                    jurisdiction=list(regions),
                    applies_to={"products": list(products), "tiers": list(tiers)},
                    sections=[clause_id],
                )
            )
            distract_clause.doc_id = doc_id
            distract_clause.published_at = published_at
            extra_doc_ids.append(doc_id)

        documents.extend(extra_docs)

        for clause in clauses:
            if clause.slot in SLOTS:
                alias, canonical = BRIDGE_TERMS.get(clause.slot, (clause.slot, clause.slot))
                clause.canonical_terms = [canonical]
                clause.aliases = [alias]
                if rng.random() < canonical_density:
                    clause.text = _replace_slot_text(clause.text, clause.slot, canonical)
                elif rng.random() < alias_density:
                    clause.text = _replace_slot_text(clause.text, clause.slot, alias)
                clause.has_update_keywords = clause.kind == "update" or any(
                    kw in clause.text.lower() for kw in UPDATE_KEYWORDS
                )

    doc_lookup = {doc.doc_id: doc for doc in documents}
    doc_capacity = {doc.doc_id: clauses_per_doc for doc in documents}
    for doc_id in extra_doc_ids:
        doc_capacity[doc_id] = 0

    for clause in clauses:
        if clause.doc_id:
            doc = doc_lookup.get(clause.doc_id)
        else:
            doc = pick_doc(clause.kind)
            clause.doc_id = doc.doc_id
            clause.published_at = doc.published_at
            clause.authority = doc.authority
        if doc:
            doc.sections.append(clause.clause_id)
            if clause.kind == "update":
                doc.doc_type = rng.choice(UPDATE_DOC_TYPES)
            elif doc.doc_type == "policy":
                doc.doc_type = rng.choice(POLICY_DOC_TYPES)

    meta = {
        "term_definitions": term_definitions,
        "definition_dependency_depth": int(definition_dependency_depth),
        "definition_dependency_extra_terms": int(definition_dependency_extra_terms),
        "force_exception_chain_depth": int(force_exception_chain_depth),
        "force_exception_chain_all_apply": bool(force_exception_chain_all_apply),
        "priority_clause_ids": priority_clause_ids,
        "authority_priority": {"security": 0, "legal": 1, "product": 2, "support": 3},
        "scenario_mode": scenario_mode,
        "bridge_clause_by_slot": bridge_clause_by_slot,
        "bridge_terms": {slot: {"alias": BRIDGE_TERMS[slot][0], "canonical": BRIDGE_TERMS[slot][1]} for slot in SLOTS},
        "base_rule_by_slot": {slot: base_clauses[slot].clause_id for slot in SLOTS},
    }

    return World(documents=documents, clauses={c.clause_id: c for c in clauses}, meta=meta)


def generate_tasks(
    world: World,
    *,
    seed: int = 0,
    n_tasks: int = 200,
    scenario_mode: str = "v0",
    bridge_prob: float = 0.8,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    exclusive_core_evidence: bool = False,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
) -> List[Task]:
    rng = random.Random(seed + 1000)
    products = products or DEFAULT_PRODUCTS
    tiers = tiers or DEFAULT_TIERS
    regions = regions or DEFAULT_REGIONS
    data_types = data_types or DEFAULT_DATA_TYPES
    purposes = purposes or DEFAULT_PURPOSES

    tasks: List[Task] = []
    base_date = datetime(2025, 6, 1)

    attempts = 0
    max_attempts = max(50, n_tasks * 20)
    while len(tasks) < n_tasks:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to generate enough tasks with exclusive_core_evidence constraints. "
                f"Generated={len(tasks)} target={n_tasks}"
            )
        slot = rng.choice(SLOTS)
        context = {
            "slot": slot,
            "product": rng.choice(products),
            "tier": rng.choice(tiers),
            "region": rng.choice(regions),
            "data_type": rng.choice(data_types),
            "purpose": rng.choice(purposes),
        }
        slot_text = SLOT_LABELS.get(slot, slot)
        slot_hint_alias = None
        canonical_slot_term = None
        bridge_clause_id = None
        needs_update_resolution = False
        if scenario_mode == "bridged_v1_1":
            alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))
            canonical_slot_term = canonical
            if rng.random() < bridged_mix_canonical_in_ticket_rate:
                slot_text = canonical
                slot_hint_alias = None
                bridge_clause_id = None
            elif rng.random() < alias_density:
                slot_text = alias
                slot_hint_alias = alias
                bridge_clause_id = world.meta.get("bridge_clause_by_slot", {}).get(slot)
            else:
                slot_text = canonical
                bridge_clause_id = world.meta.get("bridge_clause_by_slot", {}).get(slot)
        ticket = (
            f"Customer asks about {slot_text} for the {context['product']} "
            f"{context['tier']} plan in {context['region']}. "
            f"Data type: {context['data_type']}. Purpose: {context['purpose']}."
        )
        decision, conditions, evidence, debug = evaluate_context(world, context)
        needs_update_resolution = bool(debug.get("used_updates"))
        meta_ids = [
            cid
            for cid in evidence
            if (world.clauses.get(cid) and world.clauses[cid].kind == "priority")
        ]
        core_ids = [cid for cid in evidence if cid not in meta_ids]
        gold = Gold(
            decision=decision,
            conditions=conditions,
            gold_evidence=evidence,
            gold_evidence_core=core_ids,
            gold_evidence_meta=meta_ids,
        )
        if exclusive_core_evidence:
            core_ids_set = set(core_ids)
            if not core_ids_set:
                continue
            pruned_clauses = {
                cid: clause for cid, clause in world.clauses.items() if cid not in core_ids_set
            }
            pruned_world = World(
                documents=world.documents,
                clauses=pruned_clauses,
                meta=world.meta,
            )
            pruned_decision, _, _, _ = evaluate_context(pruned_world, context)
            if pruned_decision == decision:
                continue
            # Reject if any non-core clause alone can yield the gold decision.
            found_alt = False
            for clause in pruned_clauses.values():
                if clause.slot != slot:
                    continue
                solo_world = World(
                    documents=world.documents,
                    clauses={clause.clause_id: clause},
                    meta=world.meta,
                )
                solo_decision, _, _, _ = evaluate_context(solo_world, context)
                if solo_decision == decision:
                    found_alt = True
                    break
            if found_alt:
                continue

        task_index = len(tasks)
        task = Task(
            task_id=f"T{task_index + 1:04d}",
            timestamp=(base_date + timedelta(days=task_index)).strftime("%Y-%m-%d"),
            user_ticket=ticket,
            context=context,
            budgets={"tool_call_budget": 50, "open_budget": 5},
            gold=gold,
            scenario_mode=scenario_mode,
            slot_hint_alias=slot_hint_alias,
            canonical_slot_term=canonical_slot_term,
            bridge_clause_id=bridge_clause_id,
            needs_update_resolution=needs_update_resolution,
        )
        tasks.append(task)

    return tasks


def generate_threaded_tasks(
    world: World,
    *,
    seed: int = 0,
    n_threads: int = 100,
    open_budget_e1: int = 4,
    open_budget_e2: int = 4,
    open_budget_e3: int = 0,
    tool_budget_e1: int = 50,
    tool_budget_e2: int = 50,
    tool_budget_e3: int = 0,
    branch_distractor_rate: float = 0.5,
    exclusive_core_evidence: bool = False,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
) -> Tuple[List[Task], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    products = products or DEFAULT_PRODUCTS
    tiers = tiers or DEFAULT_TIERS
    regions = regions or DEFAULT_REGIONS
    data_types = data_types or DEFAULT_DATA_TYPES
    purposes = purposes or DEFAULT_PURPOSES

    tasks: List[Task] = []
    threads_meta: List[Dict[str, Any]] = []
    base_date = datetime(2025, 6, 1)

    def _choose_context() -> Dict[str, Any]:
        return {
            "slot": rng.choice(SLOTS),
            "product": rng.choice(products),
            "tier": rng.choice(tiers),
            "region": rng.choice(regions),
            "data_type": rng.choice(data_types),
            "purpose": rng.choice(purposes),
        }

    def _find_value(options: List[str], avoid: List[str]) -> str:
        candidates = [v for v in options if v not in avoid]
        if candidates:
            return rng.choice(candidates)
        return rng.choice(options)

    max_attempts = max(200, n_threads * 200)
    attempts = 0
    thread_index = 0
    while thread_index < n_threads:
        attempts += 1
        if attempts > max_attempts:
            if thread_index > 0:
                print(
                    "[threaded_v1_3_fu] Warning: unable to reach target threads. "
                    f"Generated={thread_index} target={n_threads}"
                )
                break
            raise RuntimeError(
                "Unable to generate enough threaded tasks with constraints. "
                f"Generated={thread_index} target={n_threads}"
            )
        context_base = _choose_context()
        slot = context_base["slot"]
        slot_text = SLOT_LABELS.get(slot, slot)
        alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))

        rule_candidates = [c for c in world.clauses.values() if c.slot == slot and c.kind == "rule"]
        exception_candidates = [
            c
            for c in world.clauses.values()
            if c.slot == slot and c.kind in {"exception", "update"}
        ]
        if not rule_candidates or not exception_candidates:
            continue
        gold_exception = rng.choice(exception_candidates)

        context_e2 = dict(context_base)
        for key, values in gold_exception.applies_if.items():
            if values:
                context_e2[key] = rng.choice(values)

        context_e1 = dict(context_base)
        if gold_exception.applies_if:
            keys = list(gold_exception.applies_if.keys())
            if "retention_bucket" in keys and len(keys) > 1:
                keys = [k for k in keys if k != "retention_bucket"]
            key = rng.choice(keys)
            options_map = {
                "product": products,
                "tier": tiers,
                "region": regions,
                "data_type": data_types,
                "purpose": purposes,
            }
            context_e1[key] = _find_value(options_map.get(key, [context_e1[key]]), gold_exception.applies_if[key])

        decision_e1, conditions_e1, evidence_e1, debug_e1 = evaluate_context(world, context_e1)
        decision_e2, conditions_e2, evidence_e2, debug_e2 = evaluate_context(world, context_e2)
        decision_e3, conditions_e3, evidence_e3, debug_e3 = evaluate_context(world, context_e2)

        if not evidence_e1:
            continue
        base_rule_id = evidence_e1[0]
        base_rule = world.clauses.get(base_rule_id)
        if not base_rule or base_rule.kind != "rule":
            continue
        if gold_exception.clause_id not in evidence_e2:
            continue

        def _split_evidence(evidence: List[str]) -> Tuple[List[str], List[str]]:
            meta_ids = [
                cid
                for cid in evidence
                if (world.clauses.get(cid) and world.clauses[cid].kind == "priority")
            ]
            core_ids = [cid for cid in evidence if cid not in meta_ids]
            return core_ids, meta_ids

        core_e1, meta_e1 = _split_evidence(evidence_e1)
        core_e2, meta_e2 = _split_evidence(evidence_e2)
        core_e3, meta_e3 = _split_evidence(evidence_e3)

        critical_clause_id_e1 = base_rule.clause_id
        critical_clause_id_e2 = gold_exception.clause_id
        critical_core_clause_ids = [critical_clause_id_e1, critical_clause_id_e2]
        core_e1 = list(dict.fromkeys(core_e1 + critical_core_clause_ids))
        core_e2 = list(dict.fromkeys(core_e2 + critical_core_clause_ids))
        core_e3 = list(dict.fromkeys(core_e3 + critical_core_clause_ids))

        # Enforce exclusive core evidence if requested.
        if exclusive_core_evidence:
            for ctx, decision, core_ids in [
                (context_e1, decision_e1, core_e1),
                (context_e2, decision_e2, core_e2),
                (context_e2, decision_e3, core_e3),
            ]:
                core_ids_set = set(core_ids)
                if not core_ids_set:
                    break
                pruned_clauses = {
                    cid: clause for cid, clause in world.clauses.items() if cid not in core_ids_set
                }
                pruned_world = World(
                    documents=world.documents,
                    clauses=pruned_clauses,
                    meta=world.meta,
                )
                pruned_decision, _, _, _ = evaluate_context(pruned_world, ctx)
                if pruned_decision == decision:
                    core_ids_set = set()
                    break
            if not core_ids_set:
                continue

        distractor_id = None
        if rng.random() < branch_distractor_rate:
            distractors = [c for c in exception_candidates if c.clause_id != gold_exception.clause_id]
            if distractors:
                distractor_id = rng.choice(distractors).clause_id

        if rng.random() < bridged_mix_canonical_in_ticket_rate:
            e1_slot_text = canonical
            slot_hint_alias = None
        elif rng.random() < alias_density:
            e1_slot_text = alias
            slot_hint_alias = alias
        else:
            e1_slot_text = canonical
            slot_hint_alias = None

        ticket_e1 = (
            f"Episode 1: Customer asks about {e1_slot_text} for the {context_e1['product']} "
            f"{context_e1['tier']} plan in {context_e1['region']}. "
            f"Data type: {context_e1['data_type']}. Purpose: {context_e1['purpose']}."
        )
        ticket_e2 = (
            f"Episode 2 follow-up about {canonical}: additional constraints apply. "
            f"Refer to commit1.fact1 for the canonical term."
        )
        if distractor_id:
            ticket_e2 += " Note: a related exception may also apply."
        ticket_e3 = (
            "Episode 3 final decision: use commit1.fact1 and commit2.fact2 to resolve the policy."
        )

        thread_id = f"TH{thread_index + 1:04d}"
        thread_config = {
            "open_budget_e1": open_budget_e1,
            "open_budget_e2": open_budget_e2,
            "open_budget_e3": open_budget_e3,
            "tool_budget_e1": tool_budget_e1,
            "tool_budget_e2": tool_budget_e2,
            "tool_budget_e3": tool_budget_e3,
            "branch_distractor_rate": branch_distractor_rate,
        }

        def _mk_task(episode_id: int, episode_kind: str, ticket: str, ctx: Dict[str, Any], decision: str, conditions: List[str], evidence: List[str], core_ids: List[str], meta_ids: List[str], needs_update: bool) -> Task:
            task_index = len(tasks)
            return Task(
                task_id=f"T{task_index + 1:04d}",
                timestamp=(base_date + timedelta(days=task_index)).strftime("%Y-%m-%d"),
                user_ticket=ticket,
                context=ctx,
                budgets={
                    "tool_call_budget": tool_budget_e1 if episode_id == 1 else tool_budget_e2 if episode_id == 2 else tool_budget_e3,
                    "open_budget": open_budget_e1 if episode_id == 1 else open_budget_e2 if episode_id == 2 else open_budget_e3,
                },
                gold=Gold(
                    decision=decision,
                    conditions=conditions,
                    gold_evidence=evidence,
                    gold_evidence_core=core_ids,
                    gold_evidence_meta=meta_ids,
                ),
                scenario_mode="threaded_v1_2",
                slot_hint_alias=slot_hint_alias if episode_id == 1 else None,
                canonical_slot_term=canonical,
                bridge_clause_id=world.meta.get("bridge_clause_by_slot", {}).get(slot),
                needs_update_resolution=needs_update,
                thread_id=thread_id,
                episode_id=episode_id,
                episode_kind=episode_kind,
                thread_config=thread_config,
                branch_distractor_clause_id=distractor_id,
            )

        tasks.append(
            _mk_task(
                1,
                "e1_retrieve_rule",
                ticket_e1,
                context_e1,
                decision_e1,
                conditions_e1,
                evidence_e1,
                core_e1,
                meta_e1,
                bool(debug_e1.get("used_updates")),
            )
        )
        tasks.append(
            _mk_task(
                2,
                "e2_exception_update",
                ticket_e2,
                context_e2,
                decision_e2,
                conditions_e2,
                evidence_e2,
                core_e2,
                meta_e2,
                bool(debug_e2.get("used_updates")),
            )
        )
        tasks.append(
            _mk_task(
                3,
                "e3_final_compose",
                ticket_e3,
                context_e2,
                decision_e3,
                conditions_e3,
                evidence_e3,
                core_e3,
                meta_e3,
                bool(debug_e3.get("used_updates")),
            )
        )
        threads_meta.append(
            {
                "thread_id": thread_id,
                "episode_task_ids": [tasks[-3].task_id, tasks[-2].task_id, tasks[-1].task_id],
                "slot": slot,
                "canonical_slot_term": canonical,
                "branch_distractor_clause_id": distractor_id,
                "thread_config": thread_config,
            }
        )
        thread_index += 1

    return tasks, threads_meta


def generate_threaded_tasks_v1_3_fu(
    world: World,
    *,
    seed: int = 0,
    n_threads: int = 100,
    open_budget_e1: int = 4,
    open_budget_e2: int = 4,
    open_budget_e3: int = 0,
    tool_budget_e1: int = 50,
    tool_budget_e2: int = 50,
    tool_budget_e3: int = 0,
    branch_distractor_rate: float = 0.5,
    exclusive_core_evidence: bool = False,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
    force_base_rule_e1: bool = False,
) -> Tuple[List[Task], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    products = products or DEFAULT_PRODUCTS
    tiers = tiers or DEFAULT_TIERS
    regions = regions or DEFAULT_REGIONS
    data_types = data_types or DEFAULT_DATA_TYPES
    purposes = purposes or DEFAULT_PURPOSES

    def _retention_days_for_bucket(bucket: str) -> int:
        if bucket == "le_30":
            return rng.randint(7, 30)
        return rng.randint(31, 90)

    tasks: List[Task] = []
    threads_meta: List[Dict[str, Any]] = []
    base_date = datetime(2025, 6, 1)

    def _choose_context() -> Dict[str, Any]:
        bucket = rng.choice(RETENTION_BUCKETS)
        return {
            "slot": rng.choice(SLOTS),
            "product": rng.choice(products),
            "tier": rng.choice(tiers),
            "region": rng.choice(regions),
            "data_type": rng.choice(data_types),
            "purpose": rng.choice(purposes),
            "retention_bucket": bucket,
            "retention_days": _retention_days_for_bucket(bucket),
        }

    def _find_value(options: List[str], avoid: List[str]) -> str:
        candidates = [v for v in options if v not in avoid]
        if candidates:
            return rng.choice(candidates)
        return rng.choice(options)

    def _sync_retention_days(ctx: Dict[str, Any]) -> None:
        bucket = ctx.get("retention_bucket")
        if bucket:
            ctx["retention_days"] = _retention_days_for_bucket(bucket)

    max_attempts = max(200, n_threads * 200)
    attempts = 0
    thread_index = 0
    while thread_index < n_threads:
        attempts += 1
        if attempts > max_attempts:
            if thread_index > 0:
                print(
                    "[threaded_v1_3_fu] Warning: unable to reach target threads. "
                    f"Generated={thread_index} target={n_threads}"
                )
                break
            raise RuntimeError(
                "Unable to generate enough threaded tasks with constraints. "
                f"Generated={thread_index} target={n_threads}"
            )
        context_base = _choose_context()
        slot = context_base["slot"]
        slot_text = SLOT_LABELS.get(slot, slot)
        alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))

        rule_candidates = [c for c in world.clauses.values() if c.slot == slot and c.kind == "rule"]
        exception_candidates = [
            c
            for c in world.clauses.values()
            if c.slot == slot and c.kind in {"exception", "update"}
        ]
        if not rule_candidates or not exception_candidates:
            continue
        base_rule = rng.choice(rule_candidates)
        gold_exception = rng.choice(exception_candidates)

        context_e2 = dict(context_base)
        for key, values in gold_exception.applies_if.items():
            if values:
                context_e2[key] = rng.choice(values)
        _sync_retention_days(context_e2)

        context_e1 = dict(context_base)
        if gold_exception.applies_if:
            keys = list(gold_exception.applies_if.keys())
            if "retention_bucket" in keys and len(keys) > 1:
                keys = [k for k in keys if k != "retention_bucket"]
            key = rng.choice(keys)
            options_map = {
                "product": products,
                "tier": tiers,
                "region": regions,
                "data_type": data_types,
                "purpose": purposes,
                "retention_bucket": RETENTION_BUCKETS,
            }
            context_e1[key] = _find_value(options_map.get(key, [context_e1[key]]), gold_exception.applies_if[key])
            _sync_retention_days(context_e1)

        decision_e1, conditions_e1, evidence_e1, debug_e1 = evaluate_context(world, context_e1)
        decision_e2, conditions_e2, evidence_e2, debug_e2 = evaluate_context(world, context_e2)
        decision_e3, conditions_e3, evidence_e3, debug_e3 = evaluate_context(world, context_e2)

        if not evidence_e1:
            continue
        base_rule_id = evidence_e1[0]
        base_rule = world.clauses.get(base_rule_id)
        if not base_rule or base_rule.kind != "rule":
            continue
        if gold_exception.clause_id not in evidence_e2:
            continue

        def _split_evidence(evidence: List[str]) -> Tuple[List[str], List[str]]:
            meta_ids = [
                cid
                for cid in evidence
                if (world.clauses.get(cid) and world.clauses[cid].kind == "priority")
            ]
            core_ids = [cid for cid in evidence if cid not in meta_ids]
            return core_ids, meta_ids

        core_e1, meta_e1 = _split_evidence(evidence_e1)
        core_e2, meta_e2 = _split_evidence(evidence_e2)
        core_e3, meta_e3 = _split_evidence(evidence_e3)

        critical_clause_id_e1 = base_rule.clause_id
        critical_clause_id_e2 = gold_exception.clause_id
        critical_core_clause_ids = [critical_clause_id_e1, critical_clause_id_e2]
        core_e1 = list(dict.fromkeys(core_e1 + critical_core_clause_ids))
        core_e2 = list(dict.fromkeys(core_e2 + critical_core_clause_ids))
        core_e3 = list(dict.fromkeys(core_e3 + critical_core_clause_ids))

        # Enforce exclusive core evidence if requested.
        if exclusive_core_evidence:
            for ctx, decision, core_ids in [
                (context_e1, decision_e1, core_e1),
                (context_e2, decision_e2, core_e2),
                (context_e2, decision_e3, core_e3),
            ]:
                core_ids_set = set(core_ids)
                if not core_ids_set:
                    break
                pruned_clauses = {
                    cid: clause for cid, clause in world.clauses.items() if cid not in core_ids_set
                }
                pruned_world = World(
                    documents=world.documents,
                    clauses=pruned_clauses,
                    meta=world.meta,
                )
                pruned_decision, _, _, _ = evaluate_context(pruned_world, ctx)
                if pruned_decision == decision:
                    core_ids_set = set()
                    break
            if not core_ids_set:
                continue

        distractor_id = None
        if rng.random() < branch_distractor_rate:
            distractors = [c for c in exception_candidates if c.clause_id != gold_exception.clause_id]
            if distractors:
                distractor_id = rng.choice(distractors).clause_id

        if rng.random() < bridged_mix_canonical_in_ticket_rate:
            e1_slot_text = canonical
            slot_hint_alias = None
        elif rng.random() < alias_density:
            e1_slot_text = alias
            slot_hint_alias = alias
        else:
            e1_slot_text = canonical
            slot_hint_alias = None

        ticket_e1 = (
            f"Episode 1: Customer asks about {e1_slot_text} for the {context_e1['product']} "
            f"{context_e1['tier']} plan in {context_e1['region']}. "
            f"Data type: {context_e1['data_type']}. Purpose: {context_e1['purpose']}."
        )
        ticket_e2 = (
            f"Episode 2 follow-up about {canonical}: additional constraints apply. "
            f"Refer to commit1.fact1 for the canonical term."
        )
        if distractor_id:
            ticket_e2 += " Note: a related exception may also apply."
        ticket_e3 = (
            "Episode 3 final decision: resolve the policy using commit1.fact1 and commit2.fact2. "
            f"Retention days={context_e2['retention_days']}, region={context_e2['region']}, "
            f"product={context_e2['product']}, tier={context_e2['tier']}, "
            f"data_type={context_e2['data_type']}, purpose={context_e2['purpose']}."
        )

        thread_id = f"TH{thread_index + 1:04d}"
        thread_config = {
            "open_budget_e1": open_budget_e1,
            "open_budget_e2": open_budget_e2,
            "open_budget_e3": open_budget_e3,
            "tool_budget_e1": tool_budget_e1,
            "tool_budget_e2": tool_budget_e2,
            "tool_budget_e3": tool_budget_e3,
            "branch_distractor_rate": branch_distractor_rate,
        }

        def _mk_task(
            episode_id: int,
            episode_kind: str,
            ticket: str,
            ctx: Dict[str, Any],
            decision: str,
            conditions: List[str],
            evidence: List[str],
            core_ids: List[str],
            meta_ids: List[str],
            needs_update: bool,
        ) -> Task:
            task_index = len(tasks)
            return Task(
                task_id=f"T{task_index + 1:04d}",
                timestamp=(base_date + timedelta(days=task_index)).strftime("%Y-%m-%d"),
                user_ticket=ticket,
                context=ctx,
                budgets={
                    "tool_call_budget": tool_budget_e1
                    if episode_id == 1
                    else tool_budget_e2
                    if episode_id == 2
                    else tool_budget_e3,
                    "open_budget": open_budget_e1
                    if episode_id == 1
                    else open_budget_e2
                    if episode_id == 2
                    else open_budget_e3,
                },
                gold=Gold(
                    decision=decision,
                    conditions=conditions,
                    gold_evidence=evidence,
                    gold_evidence_core=core_ids,
                    gold_evidence_meta=meta_ids,
                ),
                scenario_mode="threaded_v1_3_fu",
                slot_hint_alias=slot_hint_alias if episode_id == 1 else None,
                canonical_slot_term=canonical,
                bridge_clause_id=world.meta.get("bridge_clause_by_slot", {}).get(slot),
                needs_update_resolution=needs_update,
                thread_id=thread_id,
                episode_id=episode_id,
                episode_kind=episode_kind,
                thread_config=thread_config,
                branch_distractor_clause_id=distractor_id,
                critical_clause_id_e1=critical_clause_id_e1,
                critical_clause_id_e2=critical_clause_id_e2,
                critical_core_clause_ids=critical_core_clause_ids,
            )

        tasks.append(
            _mk_task(
                1,
                "e1_retrieve_rule",
                ticket_e1,
                context_e1,
                decision_e1,
                conditions_e1,
                evidence_e1,
                core_e1,
                meta_e1,
                bool(debug_e1.get("used_updates")),
            )
        )
        tasks.append(
            _mk_task(
                2,
                "e2_exception_update",
                ticket_e2,
                context_e2,
                decision_e2,
                conditions_e2,
                evidence_e2,
                core_e2,
                meta_e2,
                bool(debug_e2.get("used_updates")),
            )
        )
        tasks.append(
            _mk_task(
                3,
                "e3_final_compose",
                ticket_e3,
                context_e2,
                decision_e3,
                conditions_e3,
                evidence_e3,
                core_e3,
                meta_e3,
                bool(debug_e3.get("used_updates")),
            )
        )
        threads_meta.append(
            {
                "thread_id": thread_id,
                "episode_task_ids": [tasks[-3].task_id, tasks[-2].task_id, tasks[-1].task_id],
                "slot": slot,
                "canonical_slot_term": canonical,
                "branch_distractor_clause_id": distractor_id,
                "critical_core_clause_ids": critical_core_clause_ids,
                "thread_config": thread_config,
            }
        )
        thread_index += 1

    return tasks, threads_meta


def generate_threaded_tasks_v1_3_fu_decoy(
    world: World,
    *,
    seed: int = 0,
    n_threads: int = 100,
    open_budget_e1: int = 4,
    open_budget_e2: int = 4,
    open_budget_e3: int = 0,
    tool_budget_e1: int = 50,
    tool_budget_e2: int = 50,
    tool_budget_e3: int = 0,
    branch_distractor_rate: float = 0.5,
    exclusive_core_evidence: bool = False,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
    decoy_count: int = DECOY_CLAUSE_COUNT,
    decoy_min_chars: int = DECOY_MIN_CHARS,
    decoy_max_chars: int = DECOY_MAX_CHARS,
    e3_litm_filler_count_min: int = 0,
    e3_litm_filler_count_max: int = 0,
    e3_litm_filler_len_jitter_max: int = 0,
    e3_clause_jitter_max_chars_critical: int = 0,
    e3_clause_jitter_max_chars_noncritical: int = 0,
    e3_clause_jitter_max_chars_decoy: int = 0,
    e3_clause_jitter_max_chars: int = 0,
    e3_clause_jitter_scope: str = "decoy_only",
) -> Tuple[List[Task], List[Dict[str, Any]]]:
    target_threads = int(n_threads)
    rng = random.Random(seed + 17)
    products = products or DEFAULT_PRODUCTS
    tiers = tiers or DEFAULT_TIERS
    regions = regions or DEFAULT_REGIONS
    data_types = data_types or DEFAULT_DATA_TYPES
    purposes = purposes or DEFAULT_PURPOSES
    jitter_max_chars = max(0, int(e3_clause_jitter_max_chars or 0))
    jitter_scope = str(e3_clause_jitter_scope or "decoy_only")
    if jitter_scope not in JITTER_SCOPE_VALUES:
        raise ValueError(
            f"Unsupported e3_clause_jitter_scope={jitter_scope!r}. "
            f"Choose one of {sorted(JITTER_SCOPE_VALUES)}."
        )
    max_critical = max(0, int(e3_clause_jitter_max_chars_critical or 0))
    max_noncritical = max(0, int(e3_clause_jitter_max_chars_noncritical or 0))
    max_decoy = max(0, int(e3_clause_jitter_max_chars_decoy or 0))
    litm_filler_count_min = max(0, int(e3_litm_filler_count_min or 0))
    litm_filler_count_max = max(0, int(e3_litm_filler_count_max or 0))
    litm_filler_len_jitter_max = max(0, int(e3_litm_filler_len_jitter_max or 0))
    if litm_filler_count_max < litm_filler_count_min:
        raise ValueError(
            "e3_litm_filler_count_max must be >= e3_litm_filler_count_min"
        )
    # Backward-compatible fallback from legacy (scope + single max) knobs.
    if max_critical == 0 and max_noncritical == 0 and max_decoy == 0 and jitter_max_chars > 0:
        if jitter_scope == "decoy_only":
            max_decoy = jitter_max_chars
        elif jitter_scope == "decoy_plus_noncritical":
            max_decoy = jitter_max_chars
            max_noncritical = jitter_max_chars
        else:
            max_decoy = jitter_max_chars
            max_noncritical = jitter_max_chars
            max_critical = jitter_max_chars

    existing_ids = [
        int(cid[1:])
        for cid in world.clauses.keys()
        if isinstance(cid, str) and cid.startswith("C") and cid[1:].isdigit()
    ]
    clause_counter = max(existing_ids) + 1 if existing_ids else 1
    doc_counter = len(world.documents) + 1

    def _next_doc_id() -> str:
        nonlocal doc_counter
        doc_id = f"DECOY{doc_counter:04d}"
        doc_counter += 1
        return doc_id

    def _next_litm_doc_id() -> str:
        nonlocal doc_counter
        doc_id = f"LITM{doc_counter:04d}"
        doc_counter += 1
        return doc_id

    def _next_decoy_clause_id() -> str:
        nonlocal clause_counter
        cid = _next_clause_id(clause_counter)
        clause_counter += 1
        return cid

    # Filtered sampling: keep generating batches until we have exactly target_threads.
    # This avoids dataset-size drift when rank-based validation is strict on some seeds.
    from .tools import ClauseIndex

    index = ClauseIndex(world)
    max_rank = max(1, int(open_budget_e1))
    max_attempts = max(20, target_threads)
    attempts = 0
    raw_generated_threads = 0
    estimated_keep_rate = 0.10
    selected_threads: List[Tuple[List[Task], Dict[str, Any]]] = []

    def _boost_threaded_ticket(task: Task) -> None:
        if task.episode_id not in {1, 2} or not task.thread_id:
            return
        slot = task.context.get("slot")
        if not slot:
            return
        slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
        alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))
        tag = f"{task.thread_id}_E{task.episode_id}"
        if tag not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} Case ref {tag}."
        if alias not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} Alias term: {alias}."
        if canonical not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} Canonical term: {canonical}."
        focus_phrase = f"Focus: {canonical} ({alias}) only."
        if focus_phrase not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} {focus_phrase}"
        alias_emphasis = f"Alias emphasis: {alias}; {alias}."
        if alias_emphasis not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} {alias_emphasis}"
        case_token = f"casecode_{task.thread_id}_E{task.episode_id}"
        if case_token not in task.user_ticket:
            task.user_ticket = f"{task.user_ticket} Casecode {case_token}."

    while len(selected_threads) < target_threads and attempts < max_attempts:
        attempts += 1
        remaining = target_threads - len(selected_threads)
        target_raw = int((remaining / max(estimated_keep_rate, 0.01)) * 1.25)
        batch_n_threads = max(remaining * 2, target_raw, target_threads)
        batch_n_threads = min(batch_n_threads, target_threads * 20)
        batch_seed = seed + attempts * 1009
        batch_tasks, batch_meta = generate_threaded_tasks_v1_3_fu(
            world,
            seed=batch_seed,
            n_threads=batch_n_threads,
            open_budget_e1=open_budget_e1,
            open_budget_e2=open_budget_e2,
            open_budget_e3=open_budget_e3,
            tool_budget_e1=tool_budget_e1,
            tool_budget_e2=tool_budget_e2,
            tool_budget_e3=tool_budget_e3,
            branch_distractor_rate=branch_distractor_rate,
            exclusive_core_evidence=exclusive_core_evidence,
            bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
            alias_density=alias_density,
            products=products,
            tiers=tiers,
            regions=regions,
            data_types=data_types,
            purposes=purposes,
        )
        raw_generated_threads += len(batch_meta)
        by_thread: Dict[str, List[Task]] = {}
        for task in batch_tasks:
            if task.thread_id:
                by_thread.setdefault(task.thread_id, []).append(task)
        kept_this_round = 0
        for meta in batch_meta:
            thread_id = meta.get("thread_id")
            if not thread_id:
                continue
            thread_tasks = sorted(
                by_thread.get(thread_id, []),
                key=lambda t: (t.episode_id or 0, t.task_id),
            )
            e1 = next((t for t in thread_tasks if t.episode_id == 1), None)
            if not e1 or not e1.critical_clause_id_e1:
                continue
            _boost_threaded_ticket(e1)
            results = index.search(
                query=e1.user_ticket,
                top_k=max(20, open_budget_e1 * 5),
                search_score_mode="bm25_plus_bridge_bonus",
                bridge_bonus=1.5,
            )
            rank = None
            for idx, item in enumerate(results, start=1):
                if item.get("clause_id") == e1.critical_clause_id_e1:
                    rank = idx
                    break
            if rank is not None and rank <= max_rank:
                for task in thread_tasks:
                    _boost_threaded_ticket(task)
                selected_threads.append((thread_tasks, dict(meta)))
                kept_this_round += 1
                if len(selected_threads) >= target_threads:
                    break
        if batch_meta:
            observed_keep_rate = kept_this_round / len(batch_meta)
            if observed_keep_rate > 0:
                estimated_keep_rate = 0.5 * estimated_keep_rate + 0.5 * observed_keep_rate

    if len(selected_threads) < target_threads:
        raise ValueError(
            "Unable to generate enough valid threaded_v1_3_fu_decoy threads. "
            f"requested={target_threads}, final={len(selected_threads)}, "
            f"generated_raw={raw_generated_threads}, attempts={attempts}, "
            f"estimated_keep_rate={estimated_keep_rate:.4f}"
        )

    # Reindex thread/task IDs for deterministic persisted dataset shape.
    tasks: List[Task] = []
    threads_meta: List[Dict[str, Any]] = []
    for idx, (thread_tasks, meta) in enumerate(selected_threads[:target_threads], start=1):
        new_thread_id = f"TH{idx:04d}"
        old_thread_id = thread_tasks[0].thread_id
        episode_task_ids: List[str] = []
        for task in sorted(thread_tasks, key=lambda t: (t.episode_id or 0, t.task_id)):
            if old_thread_id:
                task.user_ticket = task.user_ticket.replace(old_thread_id, new_thread_id)
            task.thread_id = new_thread_id
            task.task_id = f"T{len(tasks) + 1:04d}"
            task.scenario_mode = "threaded_v1_3_fu_decoy"
            episode_task_ids.append(task.task_id)
            tasks.append(task)
        meta["thread_id"] = new_thread_id
        meta["episode_task_ids"] = episode_task_ids
        threads_meta.append(meta)

    world.meta["n_threads_requested"] = target_threads
    world.meta["n_threads_generated_raw"] = raw_generated_threads
    world.meta["n_threads_generated_final"] = len(threads_meta)
    world.meta["n_threads_generation_attempts"] = attempts

    # Thread-level LITM depth/position jitter: add filler clauses used only at E3 packing.
    tasks_by_thread: Dict[str, List[Task]] = {}
    for task in tasks:
        if task.thread_id:
            tasks_by_thread.setdefault(task.thread_id, []).append(task)
    for meta in threads_meta:
        thread_id = str(meta.get("thread_id") or "")
        if not thread_id:
            continue
        slot = str(meta.get("slot") or "")
        slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
        alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))
        filler_count = _deterministic_int_in_range(
            seed,
            f"{thread_id}:litm_count",
            litm_filler_count_min,
            litm_filler_count_max,
        )
        position_idx = _deterministic_int_in_range(
            seed,
            f"{thread_id}:litm_position",
            0,
            len(LITM_FILLER_POSITIONS) - 1,
        )
        filler_position = LITM_FILLER_POSITIONS[position_idx]
        filler_clause_ids: List[str] = []
        for filler_idx in range(filler_count):
            doc_id = _next_litm_doc_id()
            litm_text = _make_litm_filler_text(
                slot=slot,
                alias=alias,
                canonical=canonical,
                thread_id=thread_id,
                filler_index=filler_idx + 1,
            )
            len_jitter = _deterministic_int_in_range(
                seed,
                f"{thread_id}:litm_len:{filler_idx}",
                0,
                litm_filler_len_jitter_max,
            )
            if len_jitter > 0:
                litm_text = apply_length_jitter(litm_text, len_jitter)
            doc = Document(
                doc_id=doc_id,
                doc_type="memo",
                title=f"LITM filler {slot_text} {thread_id}",
                published_at=tasks_by_thread.get(thread_id, [tasks[0]])[0].timestamp,
                authority="support",
                jurisdiction=[],
                applies_to={},
                sections=[],
            )
            world.documents.append(doc)
            clause_id = _next_decoy_clause_id()
            world.clauses[clause_id] = Clause(
                clause_id=clause_id,
                doc_id=doc_id,
                published_at=doc.published_at,
                authority="support",
                text=litm_text,
                kind="procedure",
                slot=slot,
                applies_if={"region": ["__LITM_NEVER__"]},
                effect={"decision": "allow"},
                conditions=[],
                targets={"overrides": [], "revokes": [], "defines": []},
                terms_used=[alias, canonical],
                canonical_terms=[canonical],
                aliases=[alias],
                bridge_for_slot=None,
                bridge_targets=[],
                has_update_keywords=False,
                is_bridge_doc=False,
            )
            filler_clause_ids.append(clause_id)
        meta["e3_litm_filler_count"] = filler_count
        meta["e3_litm_filler_clause_ids"] = list(filler_clause_ids)
        meta["e3_litm_filler_position"] = filler_position
        thread_cfg = dict(meta.get("thread_config") or {})
        thread_cfg["e3_litm_filler_count"] = filler_count
        thread_cfg["e3_litm_filler_clause_ids"] = list(filler_clause_ids)
        thread_cfg["e3_litm_filler_position"] = filler_position
        thread_cfg["e3_litm_filler_count_min"] = litm_filler_count_min
        thread_cfg["e3_litm_filler_count_max"] = litm_filler_count_max
        thread_cfg["e3_litm_filler_len_jitter_max"] = litm_filler_len_jitter_max
        meta["thread_config"] = thread_cfg
        for task in tasks_by_thread.get(thread_id, []):
            task_cfg = dict(task.thread_config or {})
            task_cfg.update(thread_cfg)
            task.thread_config = task_cfg

    # Recompute critical ids after filtering/reindexing and inject decoys.
    critical_clause_ids: set[str] = set()
    clause_thread_hint: Dict[str, str] = {}
    for task in tasks:
        if task.critical_clause_id_e1:
            critical_clause_ids.add(task.critical_clause_id_e1)
            if task.thread_id:
                clause_thread_hint.setdefault(task.critical_clause_id_e1, str(task.thread_id))
        if task.critical_clause_id_e2:
            critical_clause_ids.add(task.critical_clause_id_e2)
            if task.thread_id:
                clause_thread_hint.setdefault(task.critical_clause_id_e2, str(task.thread_id))
        if task.episode_id not in {1, 2} or not task.thread_id:
            continue
        slot = task.context.get("slot")
        if not slot:
            continue
        slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
        alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))
        tag = f"{task.thread_id}_E{task.episode_id}"
        case_token = f"casecode_{task.thread_id}_E{task.episode_id}"
        for idx in range(decoy_count):
            doc_id = _next_doc_id()
            decoy_text = _make_decoy_text(
                slot=slot,
                alias=alias,
                canonical=canonical,
                tag=tag,
                decoy_label=f"D{idx + 1}",
                case_token=case_token,
                min_chars=decoy_min_chars,
                max_chars=decoy_max_chars,
                context=task.context,
                rng=rng,
            )
            applies_if = _decoy_applies_if(
                task.context,
                products=products,
                tiers=tiers,
                regions=regions,
                data_types=data_types,
                purposes=purposes,
                rng=rng,
            )
            doc = Document(
                doc_id=doc_id,
                doc_type="memo",
                title=f"Decoy memo {slot_text} {tag}",
                published_at=task.timestamp,
                authority="support",
                jurisdiction=[],
                applies_to={},
                sections=[],
            )
            world.documents.append(doc)
            clause_id = _next_decoy_clause_id()
            world.clauses[clause_id] = Clause(
                clause_id=clause_id,
                doc_id=doc_id,
                published_at=task.timestamp,
                authority="support",
                text=decoy_text,
                kind="procedure",
                slot=slot,
                applies_if=applies_if,
                effect={"decision": "allow"},
                conditions=[],
                targets={"overrides": [], "revokes": [], "defines": []},
                terms_used=[alias, slot_text],
                canonical_terms=[],
                aliases=[alias],
                bridge_for_slot=None,
                bridge_targets=[],
                has_update_keywords=False,
                is_bridge_doc=False,
            )
            clause_thread_hint[clause_id] = str(task.thread_id)

    if critical_clause_ids:
        for cid in sorted(critical_clause_ids):
            clause = world.clauses.get(cid)
            if not clause:
                continue
            slot = clause.slot
            slot_text = SLOT_LABELS.get(slot, slot.replace("_", " "))
            _alias, canonical = BRIDGE_TERMS.get(slot, (slot_text, slot_text))
            alias_phrase = f"Alias term: {_alias}."
            canonical_phrase = f"Canonical term: {canonical}."
            lower_text = clause.text.lower()
            if _alias.lower() not in lower_text:
                clause.text = f"{clause.text} {alias_phrase}"
                lower_text = clause.text.lower()
            if canonical.lower() not in lower_text:
                clause.text = f"{clause.text} {canonical_phrase}"
            if canonical and canonical not in clause.canonical_terms:
                clause.canonical_terms.append(canonical)
            if _alias and _alias not in clause.aliases:
                clause.aliases.append(_alias)
            emphasis_phrase = f"Key clause for {canonical} ({_alias})."
            if emphasis_phrase.lower() not in clause.text.lower():
                clause.text = f"{clause.text} {emphasis_phrase}"
            alias_boost = f"Alias reminder: {_alias}; {_alias}."
            canonical_boost = f"Canonical reminder: {canonical}; {canonical}."
            if alias_boost.lower() not in clause.text.lower():
                clause.text = f"{clause.text} {alias_boost}"
            if canonical_boost.lower() not in clause.text.lower():
                clause.text = f"{clause.text} {canonical_boost}"

    if (max_critical > 0) or (max_noncritical > 0) or (max_decoy > 0):
        decoy_clause_ids = {
            cid for cid, clause in world.clauses.items() if str(clause.doc_id).startswith("DECOY")
        }
        all_core_clause_ids = {
            cid
            for task in tasks
            for cid in (getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or [])
        }
        noncritical_core_clause_ids = all_core_clause_ids - critical_clause_ids
        ordered_clause_ids: List[str] = []
        ordered_clause_ids.extend(sorted(decoy_clause_ids))
        ordered_clause_ids.extend(
            sorted(cid for cid in noncritical_core_clause_ids if cid not in ordered_clause_ids)
        )
        ordered_clause_ids.extend(
            sorted(cid for cid in critical_clause_ids if cid not in ordered_clause_ids)
        )

        for cid in ordered_clause_ids:
            clause = world.clauses.get(cid)
            if not clause:
                continue
            if cid in decoy_clause_ids:
                max_chars_for_clause = max_decoy
            elif cid in critical_clause_ids:
                max_chars_for_clause = max_critical
            else:
                max_chars_for_clause = max_noncritical
            thread_hint = clause_thread_hint.get(cid, "")
            jitter_chars = _deterministic_jitter_chars(
                seed,
                cid,
                max_chars_for_clause,
                thread_hint=thread_hint,
            )
            if jitter_chars > 0:
                clause.text = apply_length_jitter(clause.text, jitter_chars)

    for task in tasks:
        task.scenario_mode = "threaded_v1_3_fu_decoy"

    world.meta["e3_clause_jitter_max_chars"] = jitter_max_chars
    world.meta["e3_clause_jitter_max_chars_critical"] = max_critical
    world.meta["e3_clause_jitter_max_chars_noncritical"] = max_noncritical
    world.meta["e3_clause_jitter_max_chars_decoy"] = max_decoy
    world.meta["e3_clause_jitter_scope"] = jitter_scope
    world.meta["e3_litm_filler_count_min"] = litm_filler_count_min
    world.meta["e3_litm_filler_count_max"] = litm_filler_count_max
    world.meta["e3_litm_filler_len_jitter_max"] = litm_filler_len_jitter_max

    return tasks, threads_meta


def _write_jsonl(path: Path, items: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            if isinstance(item, dict):
                data = item
            elif hasattr(item, "to_dict"):
                data = item.to_dict()
            elif hasattr(item, "model_dump"):
                data = item.model_dump()
            else:
                data = asdict(item)
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def generate_world_and_tasks(
    *,
    out_dir: Optional[Path] = None,
    seed: int = 0,
    n_docs: int = 30,
    clauses_per_doc: int = 5,
    n_tasks: int = 200,
    exception_chain_depth: int = 2,
    definition_dependency_depth: int = 1,
    definition_dependency_extra_terms: int = 0,
    force_exception_chain_depth: int = 0,
    force_exception_chain_all_apply: bool = False,
    update_rate: float = 0.3,
    definition_density: float = 0.4,
    distractor_strength: float = 0.3,
    scenario_mode: str = "v0",
    bridge_prob: float = 0.8,
    bridged_mix_canonical_in_ticket_rate: float = 0.0,
    alias_density: float = 0.9,
    canonical_density: float = 0.95,
    bridge_kind: str = "definition",
    exclusive_core_evidence: bool = False,
    preset_name: Optional[str] = None,
    authorities: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
    n_threads: Optional[int] = None,
    open_budget_e1: int = 4,
    open_budget_e2: int = 4,
    open_budget_e3: int = 0,
    tool_budget_e1: int = 50,
    tool_budget_e2: int = 50,
    tool_budget_e3: int = 0,
    branch_distractor_rate: float = 0.5,
    e3_clause_jitter_max_chars_critical: int = 0,
    e3_clause_jitter_max_chars_noncritical: int = 0,
    e3_clause_jitter_max_chars_decoy: int = 0,
    e3_litm_filler_count_min: int = 0,
    e3_litm_filler_count_max: int = 0,
    e3_litm_filler_len_jitter_max: int = 0,
    e3_clause_jitter_max_chars: int = 0,
    e3_clause_jitter_scope: str = "decoy_only",
    pivot_rate: float = 0.0,
    pivot_type: str = "retention_flip",
) -> Tuple[World, List[Task], Path, Path]:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2]

    world = generate_world(
        seed=seed,
        n_docs=n_docs,
        clauses_per_doc=clauses_per_doc,
        exception_chain_depth=exception_chain_depth,
        definition_dependency_depth=definition_dependency_depth,
        definition_dependency_extra_terms=definition_dependency_extra_terms,
        force_exception_chain_depth=force_exception_chain_depth,
        force_exception_chain_all_apply=force_exception_chain_all_apply,
        update_rate=update_rate,
        definition_density=definition_density,
        distractor_strength=distractor_strength,
        scenario_mode=scenario_mode,
        bridge_prob=bridge_prob,
        bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
        alias_density=alias_density,
        canonical_density=canonical_density,
        bridge_kind=bridge_kind,
        authorities=authorities,
        products=products,
        tiers=tiers,
        regions=regions,
        data_types=data_types,
        purposes=purposes,
    )

    threads_meta: List[Dict[str, Any]] = []
    if scenario_mode == "threaded_v1_2":
        tasks, threads_meta = generate_threaded_tasks(
            world,
            seed=seed,
            n_threads=n_threads or n_tasks,
            open_budget_e1=open_budget_e1,
            open_budget_e2=open_budget_e2,
            open_budget_e3=open_budget_e3,
            tool_budget_e1=tool_budget_e1,
            tool_budget_e2=tool_budget_e2,
            tool_budget_e3=tool_budget_e3,
            branch_distractor_rate=branch_distractor_rate,
            exclusive_core_evidence=exclusive_core_evidence,
            bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
            alias_density=alias_density,
            products=products,
            tiers=tiers,
            regions=regions,
            data_types=data_types,
            purposes=purposes,
        )
    elif scenario_mode == "threaded_v1_3_fu":
        tasks, threads_meta = generate_threaded_tasks_v1_3_fu(
            world,
            seed=seed,
            n_threads=n_threads or n_tasks,
            open_budget_e1=open_budget_e1,
            open_budget_e2=open_budget_e2,
            open_budget_e3=open_budget_e3,
            tool_budget_e1=tool_budget_e1,
            tool_budget_e2=tool_budget_e2,
            tool_budget_e3=tool_budget_e3,
            branch_distractor_rate=branch_distractor_rate,
            exclusive_core_evidence=exclusive_core_evidence,
            bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
            alias_density=alias_density,
            products=products,
            tiers=tiers,
            regions=regions,
            data_types=data_types,
            purposes=purposes,
        )
    elif scenario_mode == "threaded_v1_3_fu_decoy":
        tasks, threads_meta = generate_threaded_tasks_v1_3_fu_decoy(
            world,
            seed=seed,
            n_threads=n_threads or n_tasks,
            open_budget_e1=open_budget_e1,
            open_budget_e2=open_budget_e2,
            open_budget_e3=open_budget_e3,
            tool_budget_e1=tool_budget_e1,
            tool_budget_e2=tool_budget_e2,
            tool_budget_e3=tool_budget_e3,
            branch_distractor_rate=branch_distractor_rate,
            exclusive_core_evidence=exclusive_core_evidence,
            bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
            alias_density=alias_density,
            products=products,
            tiers=tiers,
            regions=regions,
            data_types=data_types,
            purposes=purposes,
            e3_clause_jitter_max_chars_critical=e3_clause_jitter_max_chars_critical,
            e3_clause_jitter_max_chars_noncritical=e3_clause_jitter_max_chars_noncritical,
            e3_clause_jitter_max_chars_decoy=e3_clause_jitter_max_chars_decoy,
            e3_litm_filler_count_min=e3_litm_filler_count_min,
            e3_litm_filler_count_max=e3_litm_filler_count_max,
            e3_litm_filler_len_jitter_max=e3_litm_filler_len_jitter_max,
            e3_clause_jitter_max_chars=e3_clause_jitter_max_chars,
            e3_clause_jitter_scope=e3_clause_jitter_scope,
        )
    else:
        tasks = generate_tasks(
            world,
            seed=seed,
            n_tasks=n_tasks,
            scenario_mode=scenario_mode,
            bridge_prob=bridge_prob,
            bridged_mix_canonical_in_ticket_rate=bridged_mix_canonical_in_ticket_rate,
            alias_density=alias_density,
            exclusive_core_evidence=exclusive_core_evidence,
            products=products,
            tiers=tiers,
            regions=regions,
            data_types=data_types,
            purposes=purposes,
        )

    _apply_late_pivot_updates(
        tasks,
        pivot_rate=float(pivot_rate or 0.0),
        pivot_type=str(pivot_type or "retention_flip"),
        seed=seed,
    )

    world_dir = out_dir / "data" / "worlds"
    tasks_dir = out_dir / "data" / "tasks"
    _write_jsonl(world_dir / "documents.jsonl", world.documents)
    _write_jsonl(world_dir / "clauses.jsonl", list(world.clauses.values()))
    if scenario_mode in {"threaded_v1_2", "threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}:
        requested = int(n_threads or n_tasks)
        generated_final = len(threads_meta) if threads_meta else len(
            {t.thread_id for t in tasks if getattr(t, "thread_id", None)}
        )
        world.meta.setdefault("n_threads_requested", requested)
        world.meta.setdefault("n_threads_generated_raw", generated_final)
        world.meta["n_threads_generated_final"] = generated_final
    world.meta["e3_clause_jitter_max_chars"] = int(e3_clause_jitter_max_chars or 0)
    world.meta["e3_clause_jitter_max_chars_critical"] = int(
        e3_clause_jitter_max_chars_critical or 0
    )
    world.meta["e3_clause_jitter_max_chars_noncritical"] = int(
        e3_clause_jitter_max_chars_noncritical or 0
    )
    world.meta["e3_clause_jitter_max_chars_decoy"] = int(
        e3_clause_jitter_max_chars_decoy or 0
    )
    world.meta["e3_litm_filler_count_min"] = int(e3_litm_filler_count_min or 0)
    world.meta["e3_litm_filler_count_max"] = int(e3_litm_filler_count_max or 0)
    world.meta["e3_litm_filler_len_jitter_max"] = int(e3_litm_filler_len_jitter_max or 0)
    world.meta["e3_clause_jitter_scope"] = str(e3_clause_jitter_scope or "decoy_only")
    world.meta["exclusive_core_evidence"] = exclusive_core_evidence
    world.meta["pivot_rate_requested"] = float(pivot_rate or 0.0)
    world.meta["pivot_type"] = str(pivot_type or "retention_flip")
    world.meta["pivot_rate_actual"] = (
        sum(1 for t in tasks if getattr(t, "ticket_updated", None)) / max(1, len(tasks))
    )
    if preset_name:
        world.meta["preset"] = preset_name
    (world_dir / "meta.json").write_text(json.dumps(world.meta, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(tasks_dir / "tasks.jsonl", tasks)
    if scenario_mode in {"threaded_v1_2", "threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}:
        threads_path = out_dir / "data" / "threads.jsonl"
        episodes_path = out_dir / "data" / "episodes.jsonl"
        if threads_meta:
            _write_jsonl(threads_path, threads_meta)
        _write_jsonl(episodes_path, tasks)

    return world, tasks, world_dir, tasks_dir / "tasks.jsonl"
