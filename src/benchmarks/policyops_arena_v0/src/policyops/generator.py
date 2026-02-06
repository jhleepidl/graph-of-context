from __future__ import annotations

import json
import random
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


def _make_applies_if(
    rng: random.Random,
    regions: List[str],
    products: List[str],
    tiers: List[str],
    data_types: List[str],
    purposes: List[str],
    min_keys: int,
    max_keys: int,
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
        clause = _build_clause(
            clause_id=clause_id,
            kind="rule",
            slot=slot,
            applies_if={},
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
    max_chain = min(exception_chain_depth, max(0, available // max(1, len(SLOTS))))
    for slot in SLOTS:
        prev = base_clauses[slot]
        for _ in range(max_chain):
            clause_id = _next_clause_id(clause_counter)
            clause_counter += 1
            applies_if = _make_applies_if(
                rng, regions, products, tiers, data_types, purposes, min_keys=2, max_keys=3
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
    for idx in range(defs_to_add):
        term_id, label, definition = term_cycle[idx % len(term_cycle)]
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        clause = _build_clause(
            clause_id=clause_id,
            kind="definition",
            slot="definition",
            applies_if={},
            decision="needs_more_info",
            conditions=[],
            terms_used=[term_id],
            targets={"overrides": [], "revokes": [], "defines": [term_id]},
            term_label=label,
            term_definition=definition,
        )
        clauses.append(clause)
        term_definitions[term_id] = clause_id

    # Fill remaining slots.
    remaining = total_slots - len(clauses)
    for _ in range(remaining):
        clause_id = _next_clause_id(clause_counter)
        clause_counter += 1
        slot = rng.choice(SLOTS)
        roll = rng.random()
        if roll < distractor_strength:
            applies_if = _make_applies_if(
                rng, regions, products, tiers, data_types, purposes, min_keys=1, max_keys=2
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
                rng, regions, products, tiers, data_types, purposes, min_keys=1, max_keys=2
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
                rng, regions, products, tiers, data_types, purposes, min_keys=1, max_keys=2
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
            clause = _build_clause(
                clause_id=clause_id,
                kind="rule",
                slot=slot,
                applies_if={},
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
        "priority_clause_ids": priority_clause_ids,
        "authority_priority": {"security": 0, "legal": 1, "product": 2, "support": 3},
        "scenario_mode": scenario_mode,
        "bridge_clause_by_slot": bridge_clause_by_slot,
        "bridge_terms": {slot: {"alias": BRIDGE_TERMS[slot][0], "canonical": BRIDGE_TERMS[slot][1]} for slot in SLOTS},
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

    max_attempts = max(50, n_threads * 20)
    attempts = 0
    thread_index = 0
    while thread_index < n_threads:
        attempts += 1
        if attempts > max_attempts:
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

        context_e1 = dict(context_base)
        if gold_exception.applies_if:
            key = rng.choice(list(gold_exception.applies_if.keys()))
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

        if base_rule.clause_id not in evidence_e1:
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
) -> Tuple[World, List[Task], Path, Path]:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2]

    world = generate_world(
        seed=seed,
        n_docs=n_docs,
        clauses_per_doc=clauses_per_doc,
        exception_chain_depth=exception_chain_depth,
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

    world_dir = out_dir / "data" / "worlds"
    tasks_dir = out_dir / "data" / "tasks"
    _write_jsonl(world_dir / "documents.jsonl", world.documents)
    _write_jsonl(world_dir / "clauses.jsonl", list(world.clauses.values()))
    world.meta["exclusive_core_evidence"] = exclusive_core_evidence
    if preset_name:
        world.meta["preset"] = preset_name
    (world_dir / "meta.json").write_text(json.dumps(world.meta, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(tasks_dir / "tasks.jsonl", tasks)
    if scenario_mode == "threaded_v1_2":
        threads_path = out_dir / "data" / "threads.jsonl"
        episodes_path = out_dir / "data" / "episodes.jsonl"
        if threads_meta:
            _write_jsonl(threads_path, threads_meta)
        _write_jsonl(episodes_path, tasks)

    return world, tasks, world_dir, tasks_dir / "tasks.jsonl"
