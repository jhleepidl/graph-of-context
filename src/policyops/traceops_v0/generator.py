from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .schema import TraceGold, TraceStep, TraceThread, TraceWorldClause

TRACEOPS_SCENARIOS = {"exception", "contradiction", "latent", "mixed", "indirect"}


def _unique_strs(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        sval = str(value).strip()
        if not sval or sval in seen:
            continue
        seen.add(sval)
        out.append(sval)
    return out


def _initial_state(rng: random.Random) -> Dict[str, Any]:
    return {
        "region": rng.choice(["us", "eu", "apac"]),
        "deadline": rng.choice(["tight", "normal", "flex"]),
        "budget": rng.choice(["low", "medium", "high"]),
        "residency": rng.choice(["global", "eu", "us"]),
        "retention_tier": rng.choice(["short", "standard", "long"]),
    }


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9_]+", str(text or "").lower()) if tok]


def _jaccard_text(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    denom = len(ta | tb)
    if denom <= 0:
        return 0.0
    return float(len(ta & tb)) / float(denom)


def _rand_handle(rng: random.Random) -> str:
    prefixes = ["HZ", "POL", "TAG", "IDX"]
    suffix = rng.choice(["RED", "BLUE", "AMBER", "13", "21", "42"])
    return f"{rng.choice(prefixes)}-{suffix}"


def _state_flip_specific(state: Dict[str, Any], key: str, rng: random.Random) -> Tuple[str, str]:
    choices: Dict[str, List[str]] = {
        "region": ["us", "eu", "apac"],
        "deadline": ["tight", "normal", "flex"],
        "budget": ["low", "medium", "high"],
        "residency": ["global", "eu", "us"],
        "retention_tier": ["short", "standard", "long"],
    }
    current = str(state.get(key, ""))
    options = [v for v in choices.get(key, []) if v != current]
    new_value = rng.choice(options) if options else current
    state[key] = new_value
    return key, new_value


def _clause_applicable_like_eval(clause: TraceWorldClause, state: Dict[str, Any]) -> bool:
    key = clause.state_key
    val = clause.state_value
    if str(clause.node_type or "") == "EXCEPTION" and (not key or val is None):
        text = str(clause.text or "").lower()
        m = re.search(r"if\s+(\w+)\s+is\s+(\w+)", text)
        if m:
            cond_key = str(m.group(1)).strip()
            cond_val = str(m.group(2)).strip().lower()
            return str(state.get(cond_key, "")).strip().lower() == cond_val
        if "residency mismatch" in text:
            if "residency" in state and "region" in state:
                return str(state.get("residency", "")).strip().lower() != str(state.get("region", "")).strip().lower()
        return True
    if not key or val is None:
        return True
    return str(state.get(key, "")) == str(val)


def _decision_from_state(state: Dict[str, Any], *, force_exception: bool = False) -> str:
    if force_exception:
        return "require_exception"
    if str(state.get("budget", "")) == "low":
        return "defer"
    if str(state.get("region", "")) == "eu" and str(state.get("residency", "")) != "eu":
        return "require_residency"
    if str(state.get("deadline", "")) == "tight" and str(state.get("retention_tier", "")) == "long":
        return "deny"
    return "allow"


def _flip_decision_label(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if normalized == "allow":
        return "deny"
    if normalized == "deny":
        return "allow"
    if normalized == "defer":
        return "allow"
    if normalized.startswith("require_") or normalized in {"override_invalidated", "needs_more_info", "unknown"}:
        return "allow"
    return "deny"


def _oracle_decision_from_evidence(
    clauses: Dict[str, TraceWorldClause],
    evidence_ids: Sequence[str],
    state: Dict[str, Any],
) -> str:
    selected = [clauses[cid] for cid in evidence_ids if cid in clauses]
    if not selected:
        return "unknown"

    decisions = [c for c in selected if str(c.node_type or "") == "DECISION"]
    updates = [c for c in selected if str(c.node_type or "") == "UPDATE"]
    assumptions = [c for c in selected if str(c.node_type or "") == "ASSUMPTION"]
    exceptions = [c for c in selected if str(c.node_type or "") == "EXCEPTION"]
    if not decisions or not updates or not assumptions:
        return "unknown"

    decisions_sorted = sorted(decisions, key=lambda c: (int(c.step_idx), str(c.clause_id)))
    latest_decision = decisions_sorted[-1]
    base = str((latest_decision.metadata or {}).get("decision_label", "") or "").strip().lower()
    if not base or base in {"handle_binding", "unknown"}:
        base = _decision_from_state(state)

    updates_sorted = sorted(updates, key=lambda c: (int(c.step_idx), str(c.clause_id)))
    latest_update = updates_sorted[-1]
    ukey = str(latest_update.state_key or "").strip().lower()
    uval = str(latest_update.state_value or "").strip().lower()
    if ukey == "budget" and uval == "low":
        base = "defer"
    elif ukey == "deadline" and uval == "tight":
        base = "deny"
    elif ukey == "region" and uval == "eu" and str(state.get("residency", "")).strip().lower() != "eu":
        base = "require_residency"
    elif ukey == "retention_tier" and uval == "long" and str(state.get("deadline", "")).strip().lower() == "tight":
        base = "deny"
    elif ukey and base in {"allow", "deny"}:
        # Generic update override: preserve compositional pressure in indirect pivots.
        if str(state.get(ukey, "")).strip().lower() == uval and uval:
            base = "deny" if base == "allow" else "allow"

    if any(_clause_applicable_like_eval(clause, state) for clause in exceptions):
        return "require_exception"
    return str(base or "unknown")


def _level_plan(
    level: int,
    rng: random.Random,
    *,
    trace_len_override: int | None,
) -> List[str]:
    if level <= 0:
        kinds = ["explore", "commit", "pivot_check"]
        return kinds if not trace_len_override else kinds[: max(1, trace_len_override)]
    if level == 1:
        kinds = ["explore", "commit", "explore", "commit", "pivot_check"]
        return kinds if not trace_len_override else kinds[: max(1, trace_len_override)]
    if level == 2:
        kinds = ["explore", "commit", "pivot_check", "explore", "commit", "pivot_check"]
        return kinds if not trace_len_override else kinds[: max(1, trace_len_override)]

    if level == 3:
        length = int(trace_len_override or rng.randint(9, 12))
        pivot_count = max(2, min(3, length // 4))
    else:
        length = int(trace_len_override or rng.randint(15, 24))
        pivot_count = max(3, min(5, length // 4))

    kinds = ["explore", "commit"]
    middle_kinds = ["explore", "commit", "update"]
    while len(kinds) < max(2, length):
        kinds.append(rng.choice(middle_kinds))

    pivot_positions = sorted(rng.sample(range(2, len(kinds)), k=min(pivot_count, max(1, len(kinds) - 2))))
    for pos in pivot_positions:
        kinds[pos] = "pivot_check"
    if "pivot_check" not in kinds:
        kinds[-1] = "pivot_check"
    if kinds[-1] != "pivot_check":
        kinds[-1] = "pivot_check"
    return kinds


def _choose_scenario(rng: random.Random, scenarios: Sequence[str]) -> str:
    cleaned = [s for s in scenarios if s in TRACEOPS_SCENARIOS]
    if not cleaned:
        cleaned = ["mixed"]
    return rng.choice(cleaned)


def _state_flip(state: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
    key = rng.choice(["region", "deadline", "budget", "residency", "retention_tier"])
    choices: Dict[str, List[str]] = {
        "region": ["us", "eu", "apac"],
        "deadline": ["tight", "normal", "flex"],
        "budget": ["low", "medium", "high"],
        "residency": ["global", "eu", "us"],
        "retention_tier": ["short", "standard", "long"],
    }
    current = str(state.get(key, ""))
    options = [v for v in choices.get(key, []) if v != current]
    new_value = rng.choice(options) if options else current
    state[key] = new_value
    return key, new_value


def generate_traceops_threads(
    *,
    level: int = 1,
    scenarios: Sequence[str] | None = None,
    seed: int = 0,
    threads: int = 32,
    trace_len: int | None = None,
    delay_to_relevance: int | None = None,
    distractor_branching: int = 2,
    contradiction_rate: float = 0.35,
    exception_density: float = 0.35,
    state_flip_count: int = 1,
    indirection_rate: float = 0.4,
    trap_distractor_count: int = 4,
    trap_similarity_boost: float = 0.7,
    core_size_min: int = 2,
    core_size_max: int = 4,
    alias_chain_len: int = 2,
    indirect_pivot_style: str = "blended",
    core_necessity_enable: bool = False,
    core_necessity_require_all: bool = True,
    trap_decision_flip_enable: bool = False,
    hidden_core_enable: bool = False,
    hidden_core_kind: str = "low_overlap_clause",
    hidden_core_link_mode: str = "depends_on",
) -> tuple[List[TraceThread], Dict[str, Any]]:
    rng = random.Random(int(seed))
    levelsafe = max(0, int(level))
    thread_count = max(1, int(threads))
    delay_default = 1 if levelsafe <= 1 else 3 if levelsafe == 2 else 6
    delay = max(1, int(delay_to_relevance if delay_to_relevance is not None else delay_default))
    indirection_rate = max(0.0, min(1.0, float(indirection_rate)))
    trap_similarity_boost = max(0.0, min(1.0, float(trap_similarity_boost)))
    trap_distractor_count = max(0, int(trap_distractor_count))
    core_size_min = max(1, int(core_size_min))
    core_size_max = max(core_size_min, int(core_size_max))
    alias_chain_len = max(1, int(alias_chain_len))
    core_necessity_enable = bool(core_necessity_enable)
    core_necessity_require_all = bool(core_necessity_require_all)
    trap_decision_flip_enable = bool(trap_decision_flip_enable)
    hidden_core_enable = bool(hidden_core_enable)
    hidden_core_kind = str(hidden_core_kind or "low_overlap_clause").strip().lower() or "low_overlap_clause"
    hidden_core_link_mode = str(hidden_core_link_mode or "depends_on").strip().lower() or "depends_on"
    indirect_pivot_style = str(indirect_pivot_style or "blended").strip().lower()
    if indirect_pivot_style not in {"ordinal_ref", "alias_handle", "blended"}:
        indirect_pivot_style = "blended"
    scenarios_norm = [str(s).strip() for s in (scenarios or ["mixed"]) if str(s).strip()]

    result: List[TraceThread] = []
    for tidx in range(thread_count):
        trng = random.Random((int(seed) * 100003) + tidx)
        thread_id = f"TR{tidx + 1:04d}"
        scenario = _choose_scenario(trng, scenarios_norm)
        kinds = _level_plan(levelsafe, trng, trace_len_override=trace_len)
        if scenario in {"contradiction", "mixed"} and "update" not in kinds:
            for ridx in range(len(kinds) - 2, 0, -1):
                if kinds[ridx] != "pivot_check":
                    kinds[ridx] = "update"
                    break

        state = _initial_state(trng)
        initial_state = dict(state)
        clauses: Dict[str, TraceWorldClause] = {}
        steps: List[TraceStep] = []
        clause_counter = 0

        decision_ids: List[str] = []
        assumption_ids: List[str] = []
        option_ids: List[str] = []
        update_ids: List[str] = []
        update_ids_by_key: Dict[str, List[str]] = {}
        active_invalidated: set[str] = set()

        latent_clause_id: str | None = None
        latent_intro_step: int | None = None
        handle_token = _rand_handle(trng)
        ordinal_key = trng.choice(["region", "deadline", "budget", "residency", "retention_tier"])
        handle_binding_ids: List[str] = []

        def _new_clause(
            *,
            step_idx: int,
            node_type: str,
            text: str,
            state_key: str | None = None,
            state_value: str | None = None,
            depends_on: Sequence[str] | None = None,
            tags: Sequence[str] | None = None,
            branch_id: str | None = None,
            metadata: Dict[str, Any] | None = None,
        ) -> str:
            nonlocal clause_counter
            clause_counter += 1
            cid = f"{thread_id}-C{clause_counter:04d}"
            clauses[cid] = TraceWorldClause(
                clause_id=cid,
                thread_id=thread_id,
                step_idx=int(step_idx),
                node_type=str(node_type),
                text=str(text),
                state_key=state_key,
                state_value=state_value,
                depends_on=list(depends_on or []),
                tags=list(tags or []),
                branch_id=branch_id,
                metadata=dict(metadata or {}),
            )
            return cid

        for step_idx, kind in enumerate(kinds):
            introduced: List[str] = []
            avoid_targets: List[str] = []
            message = ""
            if kind == "explore":
                message = "Let's explore options and gather evidence before deciding."
                branch_count = max(1, int(distractor_branching))
                for bidx in range(branch_count):
                    bid = f"B{bidx + 1}"
                    opt_id = _new_clause(
                        step_idx=step_idx,
                        node_type="OPTION",
                        text=f"Option {bid} for {state['region']} with {state['retention_tier']} retention.",
                        state_key="region",
                        state_value=str(state.get("region")),
                        tags=["option", "branch"],
                        branch_id=bid,
                    )
                    introduced.append(opt_id)
                    option_ids.append(opt_id)
                ev_id = _new_clause(
                    step_idx=step_idx,
                    node_type="EVIDENCE",
                    text=f"Evidence snapshot: deadline={state['deadline']}, budget={state['budget']}.",
                    tags=["evidence", "stable"],
                )
                introduced.append(ev_id)

                indirect_active_step = (
                    scenario == "indirect"
                    or (scenario == "mixed" and trng.random() < float(indirection_rate))
                )
                if indirect_active_step and len(handle_binding_ids) < alias_chain_len:
                    bind_id = _new_clause(
                        step_idx=step_idx,
                        node_type="ASSUMPTION",
                        text=(
                            f"Handle {handle_token} currently binds to policy gate "
                            f"{ordinal_key}={state.get(ordinal_key)}."
                        ),
                        tags=["indirect", "handle_binding", "alias"],
                        metadata={
                            "handle": handle_token,
                            "binding_key": ordinal_key,
                            "binding_value": str(state.get(ordinal_key, "")),
                        },
                    )
                    introduced.append(bind_id)
                    assumption_ids.append(bind_id)
                    handle_binding_ids.append(bind_id)

                wants_exception = scenario in {"exception", "latent"} or (
                    scenario == "mixed" and trng.random() < float(exception_density)
                )
                if wants_exception and latent_clause_id is None and step_idx <= 1:
                    latent_clause_id = _new_clause(
                        step_idx=step_idx,
                        node_type="EXCEPTION",
                        text="Footnote exception: residency mismatch can be tolerated with manual review.",
                        state_key=None,
                        state_value=None,
                        tags=["exception", "latent"],
                        metadata={"salience": "low"},
                    )
                    latent_intro_step = int(step_idx)
                    introduced.append(latent_clause_id)

            elif kind == "commit":
                message = "Let's commit to the best current plan and assumptions."
                chosen_option = option_ids[-1] if option_ids else ""
                asm_id = _new_clause(
                    step_idx=step_idx,
                    node_type="ASSUMPTION",
                    text=f"Assume region={state['region']} and budget={state['budget']} remain stable.",
                    state_key="region",
                    state_value=str(state.get("region")),
                    tags=["assumption", "state"],
                )
                introduced.append(asm_id)
                assumption_ids.append(asm_id)
                decision_label = _decision_from_state(state)
                dep = [cid for cid in [chosen_option, asm_id] if cid]
                dec_id = _new_clause(
                    step_idx=step_idx,
                    node_type="DECISION",
                    text=f"Decision checkpoint: {decision_label}",
                    depends_on=dep,
                    tags=["decision"],
                    metadata={"decision_label": decision_label},
                )
                introduced.append(dec_id)
                decision_ids.append(dec_id)

                indirect_active_step = (
                    scenario == "indirect"
                    or (scenario == "mixed" and trng.random() < float(indirection_rate))
                )
                if indirect_active_step and len(handle_binding_ids) < alias_chain_len:
                    bind_decision_id = _new_clause(
                        step_idx=step_idx,
                        node_type="DECISION",
                        text=(
                            f"Handle {handle_token} binding revision {len(handle_binding_ids)+1}: "
                            f"use latest {ordinal_key} update for final decision."
                        ),
                        depends_on=[dec_id] if dec_id else [],
                        tags=["indirect", "handle_binding", "alias", "revision"],
                        metadata={
                            "decision_label": "handle_binding",
                            "handle": handle_token,
                            "binding_key": ordinal_key,
                        },
                    )
                    introduced.append(bind_decision_id)
                    decision_ids.append(bind_decision_id)
                    handle_binding_ids.append(bind_decision_id)

            elif kind == "update":
                message = "User update arrived; prior assumptions may be invalid."
                flips = max(1, int(state_flip_count))
                update_step_ids: List[str] = []
                indirect_active_step = (
                    scenario == "indirect"
                    or (scenario == "mixed" and trng.random() < float(indirection_rate))
                )
                forced_updates = 1 if indirect_active_step else 0
                for _ in range(forced_updates):
                    key, value = _state_flip_specific(state, ordinal_key, trng)
                    upd_id = _new_clause(
                        step_idx=step_idx,
                        node_type="UPDATE",
                        text=f"Update: {key} changed to {value}.",
                        state_key=key,
                        state_value=value,
                        tags=["update", "state_flip", "ordinal_chain"],
                        metadata={"ordinal_key": ordinal_key, "indirect": True},
                    )
                    introduced.append(upd_id)
                    update_ids.append(upd_id)
                    update_ids_by_key.setdefault(str(key), []).append(upd_id)
                    update_step_ids.append(upd_id)
                for _ in range(max(0, flips - forced_updates)):
                    key, value = _state_flip(state, trng)
                    upd_id = _new_clause(
                        step_idx=step_idx,
                        node_type="UPDATE",
                        text=f"Update: {key} changed to {value}.",
                        state_key=key,
                        state_value=value,
                        tags=["update", "state_flip"],
                    )
                    introduced.append(upd_id)
                    update_ids.append(upd_id)
                    update_ids_by_key.setdefault(str(key), []).append(upd_id)
                    update_step_ids.append(upd_id)

                contradiction_mode = scenario == "contradiction" or (
                    scenario == "mixed" and trng.random() < float(contradiction_rate)
                )
                if contradiction_mode:
                    # Prefer invalidating stale decisions/assumptions first, then options.
                    # Keep the most recent decision checkpoint out of avoid targets to reduce
                    # accidental invalidation of pivot-critical context.
                    decision_pool = list(decision_ids[:-1]) if len(decision_ids) > 1 else []
                    assumption_pool = list(assumption_ids)
                    option_pool = list(option_ids)
                    pool = (
                        list(reversed(decision_pool[-3:]))
                        + list(reversed(assumption_pool[-3:]))
                        + list(reversed(option_pool[-2:]))
                    )
                    avoid_targets = _unique_strs(pool[: max(1, min(2, len(pool)))])
                    active_invalidated.update(avoid_targets)
                    for upd_id in update_step_ids:
                        clause = clauses.get(upd_id)
                        if clause:
                            clause.tags = _unique_strs(list(clause.tags) + ["contradiction"])
                            clause.metadata["invalidates"] = list(avoid_targets)

                if scenario in {"exception", "mixed"} and trng.random() < float(exception_density):
                    deadline_now = str(state.get("deadline", "")).strip().lower()
                    activated = deadline_now == "tight"
                    ex_text = (
                        "Exception activated: manual override allowed if deadline is tight."
                        if activated
                        else "Exception latent: manual override allowed if deadline is tight."
                    )
                    ex_tags = ["exception", "activation"] if activated else ["exception", "latent"]
                    ex_id = _new_clause(
                        step_idx=step_idx,
                        node_type="EXCEPTION",
                        text=ex_text,
                        state_key=None,
                        state_value=None,
                        depends_on=update_step_ids[:1],
                        tags=ex_tags,
                    )
                    introduced.append(ex_id)
                    if latent_clause_id is None:
                        latent_clause_id = ex_id
                        latent_intro_step = int(step_idx)

                if indirect_active_step and len(handle_binding_ids) < alias_chain_len and update_step_ids:
                    upd_ref = update_step_ids[-1]
                    bind_id = _new_clause(
                        step_idx=step_idx,
                        node_type="ASSUMPTION",
                        text=(
                            f"{handle_token} revision {len(handle_binding_ids)+1} now follows "
                            f"the latest {ordinal_key} change."
                        ),
                        depends_on=[upd_ref],
                        tags=["indirect", "handle_binding", "alias", "revision"],
                        metadata={
                            "handle": handle_token,
                            "binding_key": ordinal_key,
                            "source_update_id": upd_ref,
                        },
                    )
                    introduced.append(bind_id)
                    assumption_ids.append(bind_id)
                    handle_binding_ids.append(bind_id)

            elif kind == "pivot_check":
                message = "Given all updates so far, what is the final decision now?"
            else:
                message = "Continue processing the trace."

            step_gold: TraceGold | None = None
            pivot_required_ids: List[str] = []
            pivot_meta: Dict[str, Any] = {
                "core_necessity_enable": bool(core_necessity_enable),
                "core_necessity_flip_count": 0,
                "core_necessity_all_required": False,
                "core_necessity_failed": False,
                "trap_decision_label": "",
                "trap_decision_flip": False,
                "hidden_core_ids": [],
                "hidden_core_parent_ids": [],
            }
            if kind == "pivot_check":
                all_prior_ids = [
                    cid
                    for cid, clause in clauses.items()
                    if int(clause.step_idx) < int(step_idx)
                ]
                latest_decision = ""
                latest_decision_id = ""
                for cid in reversed(all_prior_ids):
                    clause = clauses[cid]
                    if clause.node_type != "DECISION":
                        continue
                    latest_decision = str(clause.metadata.get("decision_label", "allow"))
                    latest_decision_id = cid
                    break

                indirect_pivot_active = (
                    scenario == "indirect"
                    or (scenario == "mixed" and trng.random() < float(indirection_rate))
                )
                style_choice = indirect_pivot_style
                if style_choice == "blended":
                    style_choice = trng.choice(["ordinal_ref", "alias_handle", "blended"])

                if indirect_pivot_active:
                    # Ensure a handle binding exists before pivot for alias-style references.
                    if (style_choice in {"alias_handle", "blended"}) and not handle_binding_ids and steps:
                        prev_step = steps[-1]
                        bind_id = _new_clause(
                            step_idx=int(prev_step.step_idx),
                            node_type="ASSUMPTION",
                            text=f"Handle {handle_token} binds to {ordinal_key}={state.get(ordinal_key)}.",
                            tags=["indirect", "handle_binding", "alias"],
                            metadata={"handle": handle_token, "binding_key": ordinal_key},
                        )
                        prev_step.introduced_clause_ids = _unique_strs(
                            list(prev_step.introduced_clause_ids) + [bind_id]
                        )
                        assumption_ids.append(bind_id)
                        handle_binding_ids.append(bind_id)

                    update_chain = list(update_ids_by_key.get(str(ordinal_key), []))
                    ordinal_phrase = "the latest change"
                    if len(update_chain) >= 2:
                        ordinal_phrase = trng.choice(
                            [
                                "the latest change",
                                "the second latest change",
                                "the second change",
                            ]
                        )
                    if style_choice == "ordinal_ref":
                        message = f"For {ordinal_phrase} to {ordinal_key}, what should we decide now?"
                    elif style_choice == "alias_handle":
                        message = f"Under {handle_token}, with current state, is it allowed?"
                    else:
                        message = (
                            f"Under {handle_token}, considering {ordinal_phrase} to {ordinal_key}, "
                            "what is the final decision?"
                        )

                    trap_clause_ids: List[str] = []
                    if steps and trap_distractor_count > 0:
                        prev_step = steps[-1]
                        prev_step_idx = int(prev_step.step_idx)
                        stale_val = str(state.get(ordinal_key, "unknown"))
                        if len(update_chain) >= 2:
                            prev_upd = clauses.get(update_chain[-2])
                            if prev_upd is not None and prev_upd.state_value is not None:
                                stale_val = str(prev_upd.state_value)
                        mismatch_value = "tight" if str(state.get("deadline", "")).lower() != "tight" else "flex"
                        for didx in range(int(trap_distractor_count)):
                            if didx % 3 == 0:
                                text = (
                                    f"{message} Historical note: {ordinal_key}={stale_val} remains the reference policy."
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="UPDATE",
                                    text=text,
                                    state_key=str(ordinal_key),
                                    state_value=str(stale_val),
                                    tags=["distractor", "trap", "stale"],
                                    metadata={"trap": True, "trap_kind": "stale"},
                                )
                            elif didx % 3 == 1:
                                text = (
                                    f"{message} Exception note: allow override if deadline is {mismatch_value}."
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="EXCEPTION",
                                    text=text,
                                    state_key="deadline",
                                    state_value=str(mismatch_value),
                                    tags=["distractor", "trap", "inapplicable"],
                                    metadata={"trap": True, "trap_kind": "inapplicable"},
                                )
                            else:
                                text = (
                                    f"{message} Archived binding says {handle_token} keeps prior rule without update."
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="ASSUMPTION",
                                    text=text,
                                    tags=["distractor", "trap", "avoided"],
                                    metadata={"trap": True, "trap_kind": "avoided"},
                                )
                                prev_step.avoid_target_ids = _unique_strs(
                                    list(prev_step.avoid_target_ids) + [cid]
                                )
                                active_invalidated.add(cid)
                            prev_step.introduced_clause_ids = _unique_strs(
                                list(prev_step.introduced_clause_ids) + [cid]
                            )
                            trap_clause_ids.append(cid)

                    hidden_core_ids: List[str] = []
                    hidden_core_parent_ids: List[str] = []
                    if hidden_core_enable and steps:
                        prev_step = steps[-1]
                        parent_id = ""
                        if update_chain:
                            parent_id = str(update_chain[-1])
                        elif handle_binding_ids:
                            parent_id = str(handle_binding_ids[-1])
                        elif latest_decision_id:
                            parent_id = str(latest_decision_id)
                        if parent_id:
                            hidden_text = (
                                f"Ledger correlation marker retains {handle_token} continuity index."
                                if hidden_core_kind == "low_overlap_clause"
                                else f"{handle_token} alias revision applies without textual restatement."
                            )
                            hidden_dep = [parent_id] if hidden_core_link_mode == "depends_on" else []
                            hidden_id = _new_clause(
                                step_idx=int(prev_step.step_idx),
                                node_type="ASSUMPTION",
                                text=hidden_text,
                                depends_on=hidden_dep,
                                tags=["indirect", "hidden_core", hidden_core_kind],
                                metadata={
                                    "hidden_core": True,
                                    "hidden_core_kind": str(hidden_core_kind),
                                    "hidden_core_link_mode": str(hidden_core_link_mode),
                                },
                            )
                            prev_step.introduced_clause_ids = _unique_strs(
                                list(prev_step.introduced_clause_ids) + [hidden_id]
                            )
                            assumption_ids.append(hidden_id)
                            hidden_core_ids = [hidden_id]
                            hidden_core_parent_ids = [parent_id]

                    base_candidate_core = _unique_strs(
                        list(update_chain[-2:])
                        + list(assumption_ids[-max(1, alias_chain_len):])
                        + list(handle_binding_ids[-max(1, alias_chain_len):])
                        + ([latest_decision_id] if latest_decision_id else [])
                        + list(hidden_core_ids)
                    )
                    if core_necessity_enable and core_necessity_require_all:
                        has_update = any(
                            clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "UPDATE"
                            for cid in base_candidate_core
                        )
                        if not has_update:
                            for cid in reversed(all_prior_ids):
                                clause = clauses.get(cid)
                                if clause is None or str(clause.node_type or "") != "UPDATE":
                                    continue
                                base_candidate_core.append(cid)
                                break
                        has_assumption = any(
                            clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "ASSUMPTION"
                            for cid in base_candidate_core
                        )
                        if not has_assumption:
                            for cid in reversed(all_prior_ids):
                                clause = clauses.get(cid)
                                if clause is None or str(clause.node_type or "") != "ASSUMPTION":
                                    continue
                                base_candidate_core.append(cid)
                                break
                    # Add one applicable exception when available.
                    for cid in reversed(all_prior_ids):
                        clause = clauses.get(cid)
                        if clause is None or str(clause.node_type or "") != "EXCEPTION":
                            continue
                        if _clause_applicable_like_eval(clause, state):
                            base_candidate_core.append(cid)
                            break
                    base_candidate_core = [cid for cid in _unique_strs(base_candidate_core) if cid in clauses]
                    if not base_candidate_core and latest_decision_id:
                        base_candidate_core = [latest_decision_id]

                    max_attempts = 6 if (core_necessity_enable and core_necessity_require_all) else 1
                    accepted_core: List[str] = []
                    accepted_decision = latest_decision or _decision_from_state(state)
                    accepted_conditions: List[str] = [f"region={state['region']}", f"budget={state['budget']}"]
                    accepted_flip_count = 0
                    accepted_all_required = False
                    accepted_exception = ""

                    for _attempt in range(max_attempts):
                        candidate_core = list(base_candidate_core)
                        desired_core = min(
                            len(candidate_core),
                            trng.randint(core_size_min, core_size_max) if candidate_core else 0,
                        )
                        if desired_core < min(core_size_min, len(candidate_core)):
                            desired_core = min(core_size_min, len(candidate_core))
                        if core_necessity_enable and core_necessity_require_all:
                            desired_core = max(desired_core, min(3, len(candidate_core)))
                        if desired_core <= 0 and latest_decision_id:
                            candidate_core = [latest_decision_id]
                            desired_core = 1

                        chosen_core: List[str] = []
                        if core_necessity_enable and core_necessity_require_all:
                            updates_recent = [
                                cid
                                for cid in reversed(candidate_core)
                                if clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "UPDATE"
                            ]
                            decisions_recent = [
                                cid
                                for cid in reversed(candidate_core)
                                if clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "DECISION"
                            ]
                            assumptions_recent = [
                                cid
                                for cid in reversed(candidate_core)
                                if clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "ASSUMPTION"
                            ]
                            exceptions_recent = [
                                cid
                                for cid in reversed(candidate_core)
                                if clauses.get(cid) is not None and str(clauses[cid].node_type or "") == "EXCEPTION"
                            ]
                            preferred_assumption = ""
                            for hid in hidden_core_ids:
                                if hid in assumptions_recent:
                                    preferred_assumption = hid
                                    break
                            if not preferred_assumption and assumptions_recent:
                                preferred_assumption = assumptions_recent[0]
                            mandatory = []
                            if updates_recent:
                                mandatory.append(updates_recent[0])
                            if decisions_recent:
                                mandatory.append(decisions_recent[0])
                            if preferred_assumption:
                                mandatory.append(preferred_assumption)
                            if exceptions_recent:
                                mandatory.append(exceptions_recent[0])
                            chosen_core = [cid for cid in _unique_strs(mandatory) if cid in clauses]
                            # Keep compositional core compact to preserve "all pieces required".
                            if len(chosen_core) > int(core_size_max):
                                chosen_core = chosen_core[: int(core_size_max)]
                            desired_core = max(len(chosen_core), min(int(core_size_min), len(candidate_core)))
                            desired_core = min(desired_core, int(core_size_max), len(candidate_core))
                            if len(chosen_core) < desired_core:
                                recency_sorted = sorted(
                                    candidate_core,
                                    key=lambda cid: (
                                        int(clauses[cid].step_idx),
                                        str(cid),
                                    ),
                                    reverse=True,
                                )
                                for cid in recency_sorted:
                                    if len(chosen_core) >= desired_core:
                                        break
                                    if cid not in chosen_core:
                                        chosen_core.append(cid)
                            chosen_core = [cid for cid in _unique_strs(chosen_core) if cid in clauses][:desired_core]
                        else:
                            low_overlap_sorted = sorted(
                                candidate_core,
                                key=lambda cid: (
                                    _jaccard_text(message, str(clauses[cid].text or "")),
                                    int(clauses[cid].step_idx),
                                    str(cid),
                                ),
                            )
                            if low_overlap_sorted:
                                chosen_core.append(low_overlap_sorted[0])
                            for hid in hidden_core_ids:
                                if hid in candidate_core and hid not in chosen_core:
                                    chosen_core.append(hid)
                            if core_necessity_enable:
                                for node_type in ("UPDATE", "DECISION", "ASSUMPTION", "EXCEPTION"):
                                    for cid in reversed(candidate_core):
                                        clause = clauses.get(cid)
                                        if clause is None or str(clause.node_type or "") != node_type:
                                            continue
                                        if cid not in chosen_core:
                                            chosen_core.append(cid)
                                        break

                            recency_sorted = sorted(
                                candidate_core,
                                key=lambda cid: (
                                    int(clauses[cid].step_idx),
                                    trng.random(),
                                    str(cid),
                                ),
                                reverse=True,
                            )
                            for cid in recency_sorted:
                                if len(chosen_core) >= desired_core:
                                    break
                                if cid not in chosen_core:
                                    chosen_core.append(cid)
                            chosen_core = [cid for cid in _unique_strs(chosen_core) if cid in clauses][:desired_core]

                        chosen_exception = ""
                        chosen_updates = []
                        for cid in chosen_core:
                            clause = clauses.get(cid)
                            if clause is None:
                                continue
                            if str(clause.node_type or "") == "EXCEPTION" and not chosen_exception:
                                chosen_exception = cid
                            if str(clause.node_type or "") == "UPDATE":
                                chosen_updates.append(cid)

                        fallback_decision = latest_decision or _decision_from_state(state)
                        fallback_conditions = [f"region={state['region']}", f"budget={state['budget']}"]
                        if chosen_exception:
                            fallback_decision = "require_exception"
                            fallback_conditions = ["apply_latest_update", f"exception={chosen_exception}"]
                        elif len(chosen_updates) >= 2 or active_invalidated:
                            fallback_decision = "override_invalidated"
                            fallback_conditions = ["apply_latest_update"]

                        if core_necessity_enable:
                            oracle_decision = _oracle_decision_from_evidence(clauses, chosen_core, state)
                            decision = oracle_decision if oracle_decision else "unknown"
                            if decision == "require_exception" and chosen_exception:
                                conditions = ["apply_latest_update", f"exception={chosen_exception}"]
                            elif decision in {"override_invalidated", "defer", "needs_more_info", "unknown"}:
                                conditions = ["apply_latest_update"]
                            else:
                                conditions = fallback_conditions
                        else:
                            decision = fallback_decision
                            conditions = fallback_conditions

                        flip_count = 0
                        all_required = False
                        if core_necessity_enable and chosen_core:
                            base_decision = _oracle_decision_from_evidence(clauses, chosen_core, state)
                            for removed_id in chosen_core:
                                reduced = [cid for cid in chosen_core if cid != removed_id]
                                reduced_decision = _oracle_decision_from_evidence(clauses, reduced, state)
                                if reduced_decision == "unknown" or reduced_decision != base_decision:
                                    flip_count += 1
                            all_required = bool(base_decision != "unknown" and flip_count == len(chosen_core))

                        accepted_core = chosen_core
                        accepted_decision = decision
                        accepted_conditions = list(conditions)
                        accepted_flip_count = int(flip_count)
                        accepted_all_required = bool(all_required)
                        accepted_exception = str(chosen_exception or "")
                        if not (core_necessity_enable and core_necessity_require_all) or accepted_all_required:
                            break

                    pivot_required_ids = [cid for cid in _unique_strs(accepted_core) if cid in clauses]
                    decision = accepted_decision
                    conditions = list(accepted_conditions)
                    chosen_exception = accepted_exception

                    trap_decision_label = ""
                    trap_decision_flip = False
                    if trap_decision_flip_enable:
                        trap_decision_label = _flip_decision_label(decision)
                        trap_decision_flip = bool(trap_decision_label and trap_decision_label != str(decision))
                        if trap_clause_ids:
                            lead_clause = clauses.get(trap_clause_ids[0])
                            if lead_clause is not None:
                                lead_clause.text = (
                                    f"{lead_clause.text} Recommended outcome: {trap_decision_label}."
                                )
                            for cid in trap_clause_ids:
                                clause = clauses.get(cid)
                                if clause is None:
                                    continue
                                clause.metadata["trap_decision_label"] = trap_decision_label
                                clause.metadata["trap_decision_flip"] = trap_decision_flip

                    gold_text = " ".join(
                        str(clauses[cid].text or "") for cid in pivot_required_ids if cid in clauses
                    )
                    indirection_overlap = _jaccard_text(message, gold_text)
                    best_gold_sim = 0.0
                    for cid in pivot_required_ids:
                        clause = clauses.get(cid)
                        if clause is None:
                            continue
                        best_gold_sim = max(best_gold_sim, _jaccard_text(message, str(clause.text or "")))
                    best_distractor_sim = 0.0
                    for cid in trap_clause_ids:
                        clause = clauses.get(cid)
                        if clause is None:
                            continue
                        best_distractor_sim = max(
                            best_distractor_sim,
                            _jaccard_text(message, str(clause.text or "")),
                        )
                    trap_gap = float(best_distractor_sim - best_gold_sim)
                    trap_required = bool(trap_clause_ids) and trng.random() < float(trap_similarity_boost)
                    if trap_required and trap_gap <= 0.0 and trap_clause_ids:
                        boosted_id = trap_clause_ids[0]
                        boosted_clause = clauses.get(boosted_id)
                        if boosted_clause is not None:
                            boosted_clause.text = (
                                f"{message} {message} stale stale legacy legacy binding."
                            )
                        best_distractor_sim = max(
                            best_distractor_sim,
                            _jaccard_text(message, str(clauses[boosted_id].text or "")),
                        )
                        trap_gap = float(best_distractor_sim - best_gold_sim)
                    pivot_meta = {
                        "indirect_active": True,
                        "pivot_style": str(style_choice),
                        "indirection_overlap_gold": float(indirection_overlap),
                        "best_gold_sim": float(best_gold_sim),
                        "best_distractor_sim": float(best_distractor_sim),
                        "trap_gap": float(trap_gap),
                        "trap_present": bool(best_distractor_sim > best_gold_sim and len(trap_clause_ids) > 0),
                        "core_size": int(len(pivot_required_ids)),
                        "trap_distractor_ids": list(trap_clause_ids),
                        "core_necessity_enable": bool(core_necessity_enable),
                        "core_necessity_flip_count": int(accepted_flip_count if core_necessity_enable else 0),
                        "core_necessity_all_required": bool(
                            accepted_all_required if core_necessity_enable else False
                        ),
                        "core_necessity_failed": bool(
                            core_necessity_enable and core_necessity_require_all and not accepted_all_required
                        ),
                        "trap_decision_label": str(trap_decision_label or ""),
                        "trap_decision_flip": bool(trap_decision_flip),
                        "hidden_core_ids": list(hidden_core_ids),
                        "hidden_core_parent_ids": list(hidden_core_parent_ids),
                        "hidden_core_kind": str(hidden_core_kind),
                        "hidden_core_link_mode": str(hidden_core_link_mode),
                        "ordinal_key": str(ordinal_key),
                        "handle_token": str(handle_token),
                    }
                else:
                    use_latent = (
                        latent_clause_id is not None
                        and latent_intro_step is not None
                        and (int(step_idx) - int(latent_intro_step)) >= int(delay)
                    )

                    if use_latent:
                        decision = "require_exception"
                        conditions = [f"exception={latent_clause_id}"]
                        pivot_required_ids = [latent_clause_id]
                        if latest_decision_id:
                            pivot_required_ids.append(latest_decision_id)
                    elif active_invalidated:
                        decision = "override_invalidated"
                        conditions = ["apply_latest_update"]
                        pivot_required_ids = []
                        if update_ids:
                            pivot_required_ids.append(update_ids[-1])
                        if latest_decision_id:
                            pivot_required_ids.append(latest_decision_id)
                    else:
                        decision = latest_decision or _decision_from_state(state)
                        conditions = [
                            f"region={state['region']}",
                            f"budget={state['budget']}",
                        ]
                        pivot_required_ids = [latest_decision_id] if latest_decision_id else []
                        if latent_clause_id and trng.random() < float(exception_density) * 0.5:
                            pivot_required_ids.append(latent_clause_id)

                pivot_required_ids = [cid for cid in _unique_strs(pivot_required_ids) if cid in clauses]
                step_gold = TraceGold(
                    decision=decision,
                    conditions=conditions,
                    evidence_ids=list(pivot_required_ids),
                    evidence_core_ids=list(pivot_required_ids),
                    evidence_meta_ids=[],
                )

            step = TraceStep(
                step_id=f"{thread_id}-S{step_idx + 1:03d}",
                thread_id=thread_id,
                step_idx=int(step_idx),
                kind=str(kind),
                message=message,
                state=dict(state),
                introduced_clause_ids=list(_unique_strs(introduced)),
                avoid_target_ids=list(_unique_strs(avoid_targets)),
                pivot_required_ids=list(_unique_strs(pivot_required_ids)),
                gold=step_gold,
                metadata={
                    "delay_to_relevance": int(delay),
                    "active_invalidated_count": int(len(active_invalidated)),
                    **dict(pivot_meta),
                },
            )
            steps.append(step)

        pivot_steps = [s for s in steps if str(s.kind) == "pivot_check"]
        trap_present_count = sum(
            1
            for s in pivot_steps
            if isinstance(s.metadata, dict) and bool(s.metadata.get("trap_present", False))
        )
        indirect_pivot_count = sum(
            1
            for s in pivot_steps
            if isinstance(s.metadata, dict) and bool(s.metadata.get("indirect_active", False))
        )
        thread = TraceThread(
            thread_id=thread_id,
            level=int(levelsafe),
            scenario=scenario,
            initial_state=dict(initial_state),
            steps=steps,
            clauses=clauses,
            meta={
                "trace_len": len(steps),
                "pivot_count": len(pivot_steps),
                "delay_to_relevance": int(delay),
                "distractor_branching": int(max(1, distractor_branching)),
                "contradiction_rate": float(contradiction_rate),
                "exception_density": float(exception_density),
                "state_flip_count": int(max(1, state_flip_count)),
                "indirect_pivot_count": int(indirect_pivot_count),
                "trap_present_count": int(trap_present_count),
                "traceops_indirection_rate": float(indirection_rate),
                "traceops_trap_distractor_count": int(trap_distractor_count),
                "traceops_trap_similarity_boost": float(trap_similarity_boost),
                "traceops_core_size_min": int(core_size_min),
                "traceops_core_size_max": int(core_size_max),
                "traceops_alias_chain_len": int(alias_chain_len),
                "traceops_indirect_pivot_style": str(indirect_pivot_style),
                "traceops_core_necessity_enable": bool(core_necessity_enable),
                "traceops_core_necessity_require_all": bool(core_necessity_require_all),
                "traceops_trap_decision_flip_enable": bool(trap_decision_flip_enable),
                "traceops_hidden_core_enable": bool(hidden_core_enable),
                "traceops_hidden_core_kind": str(hidden_core_kind),
                "traceops_hidden_core_link_mode": str(hidden_core_link_mode),
            },
        )
        result.append(thread)

    meta = {
        "benchmark": "traceops_v0",
        "traceops_level": int(levelsafe),
        "traceops_scenarios": list(scenarios_norm),
        "traceops_seed": int(seed),
        "traceops_threads": int(thread_count),
        "traceops_trace_len": int(trace_len) if trace_len else None,
        "traceops_delay_to_relevance": int(delay),
        "traceops_distractor_branching": int(max(1, distractor_branching)),
        "traceops_contradiction_rate": float(contradiction_rate),
        "traceops_exception_density": float(exception_density),
        "traceops_state_flip_count": int(max(1, state_flip_count)),
        "traceops_indirection_rate": float(indirection_rate),
        "traceops_trap_distractor_count": int(trap_distractor_count),
        "traceops_trap_similarity_boost": float(trap_similarity_boost),
        "traceops_core_size_min": int(core_size_min),
        "traceops_core_size_max": int(core_size_max),
        "traceops_alias_chain_len": int(alias_chain_len),
        "traceops_indirect_pivot_style": str(indirect_pivot_style),
        "traceops_core_necessity_enable": bool(core_necessity_enable),
        "traceops_core_necessity_require_all": bool(core_necessity_require_all),
        "traceops_trap_decision_flip_enable": bool(trap_decision_flip_enable),
        "traceops_hidden_core_enable": bool(hidden_core_enable),
        "traceops_hidden_core_kind": str(hidden_core_kind),
        "traceops_hidden_core_link_mode": str(hidden_core_link_mode),
        "pivots_available_total": int(
            sum(1 for thread in result for step in thread.steps if str(step.kind) == "pivot_check")
        ),
    }
    return result, meta


def save_traceops_dataset(out_dir: Path, threads: Sequence[TraceThread], meta: Dict[str, Any]) -> Path:
    thread_list = list(threads)
    base = Path(out_dir)
    data_dir = base / "data" / "traceops"
    data_dir.mkdir(parents=True, exist_ok=True)

    threads_path = data_dir / "threads.jsonl"
    with threads_path.open("w", encoding="utf-8") as handle:
        for thread in thread_list:
            handle.write(json.dumps(thread.to_dict(), ensure_ascii=True) + "\n")

    meta_payload = dict(meta)
    meta_payload["thread_count"] = int(len(thread_list))
    meta_payload["total_steps"] = int(sum(len(thread.steps) for thread in thread_list))
    meta_payload["total_clauses"] = int(sum(len(thread.clauses) for thread in thread_list))
    (data_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return data_dir


def load_traceops_dataset(base_dir: Path) -> tuple[List[TraceThread], Dict[str, Any]]:
    data_dir = Path(base_dir) / "data" / "traceops"
    threads_path = data_dir / "threads.jsonl"
    meta_path = data_dir / "meta.json"
    if not threads_path.exists():
        raise FileNotFoundError(f"TraceOps dataset not found: {threads_path}")

    threads: List[TraceThread] = []
    with threads_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, dict):
                threads.append(TraceThread.from_dict(raw))

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return threads, meta


__all__ = [
    "TRACEOPS_SCENARIOS",
    "generate_traceops_threads",
    "save_traceops_dataset",
    "load_traceops_dataset",
]
