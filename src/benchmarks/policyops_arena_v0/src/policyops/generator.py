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
    alias_density: float = 0.9,
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

    for idx in range(n_tasks):
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
            if rng.random() < alias_density:
                slot_text = alias
                slot_hint_alias = alias
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
        task = Task(
            task_id=f"T{idx + 1:04d}",
            timestamp=(base_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
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


def _write_jsonl(path: Path, items: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            if hasattr(item, "to_dict"):
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
    alias_density: float = 0.9,
    canonical_density: float = 0.95,
    bridge_kind: str = "definition",
    authorities: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    data_types: Optional[List[str]] = None,
    purposes: Optional[List[str]] = None,
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

    tasks = generate_tasks(
        world,
        seed=seed,
        n_tasks=n_tasks,
        scenario_mode=scenario_mode,
        bridge_prob=bridge_prob,
        alias_density=alias_density,
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
    (world_dir / "meta.json").write_text(json.dumps(world.meta, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(tasks_dir / "tasks.jsonl", tasks)

    return world, tasks, world_dir, tasks_dir / "tasks.jsonl"
