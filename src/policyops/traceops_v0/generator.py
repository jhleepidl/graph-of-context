from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .schema import TraceGold, TraceStep, TraceThread, TraceWorldClause, normalize_decision

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
    trap_flip_salience: float = 0.25,
    trap_flip_attach_kind: str = "avoided",
    trap_graph_excludable_rate: float = 0.7,
    trap_graph_excludable_kinds: str = "stale,inapplicable,avoided,decision_checkpoint",
    trap_graph_force_topk: int = 1,
    trap_graph_force_include_flip_target: bool = True,
    trap_graph_force_include_decision_checkpoint: bool = True,
    trap_invalidation_text_strength: float = 0.6,
    defer_budget_rate: float = 0.15,
    hidden_core_enable: bool = False,
    hidden_core_kind: str = "low_overlap_clause",
    hidden_core_link_mode: str = "depends_on",
    trap_require_avoided: bool = False,
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
    trap_flip_salience = max(0.0, min(1.0, float(trap_flip_salience)))
    trap_flip_attach_kind = str(trap_flip_attach_kind or "avoided").strip().lower() or "avoided"
    if trap_flip_attach_kind not in {"stale", "inapplicable", "avoided", "random", "none"}:
        trap_flip_attach_kind = "avoided"
    trap_graph_excludable_rate = max(0.0, min(1.0, float(trap_graph_excludable_rate)))
    trap_graph_excludable_kinds_raw = [
        str(part).strip().lower()
        for part in str(
            trap_graph_excludable_kinds or "stale,inapplicable,avoided,decision_checkpoint"
        ).split(",")
    ]
    allowed_trap_excludable_kinds = {"stale", "inapplicable", "avoided", "decision_checkpoint"}
    trap_graph_excludable_kind_set = {
        part for part in trap_graph_excludable_kinds_raw if part in allowed_trap_excludable_kinds
    }
    trap_graph_excludable_unknown_tokens = sorted(
        {
            part
            for part in trap_graph_excludable_kinds_raw
            if part and part not in allowed_trap_excludable_kinds
        }
    )
    if not trap_graph_excludable_kind_set:
        trap_graph_excludable_kind_set = {"stale", "inapplicable", "avoided", "decision_checkpoint"}
    trap_graph_excludable_kinds = ",".join(sorted(trap_graph_excludable_kind_set))
    trap_graph_force_topk = max(0, int(trap_graph_force_topk))
    trap_graph_force_include_flip_target = bool(trap_graph_force_include_flip_target)
    trap_graph_force_include_decision_checkpoint = bool(
        trap_graph_force_include_decision_checkpoint
    )
    trap_invalidation_text_strength = max(0.0, min(1.0, float(trap_invalidation_text_strength)))
    defer_budget_rate = max(0.0, min(1.0, float(defer_budget_rate)))
    hidden_core_enable = bool(hidden_core_enable)
    hidden_core_kind = str(hidden_core_kind or "low_overlap_clause").strip().lower() or "low_overlap_clause"
    hidden_core_link_mode = str(hidden_core_link_mode or "depends_on").strip().lower() or "depends_on"
    trap_require_avoided = bool(trap_require_avoided)
    core_necessity_max_tries = 16
    trap_gap_max_tries = 8
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
            clause_obj = TraceWorldClause(
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
            # Single source of truth for decision-checkpoint trap semantics.
            if (
                str(clause_obj.node_type or "") == "DECISION"
                and str(clause_obj.text or "").startswith("Decision checkpoint:")
            ):
                clause_obj.metadata["trap"] = True
                clause_obj.metadata["trap_kind"] = "decision_checkpoint"
                clause_obj.metadata["trap_decision_checkpoint"] = True
                clause_obj.metadata.setdefault("decision_checkpoint_controlling", False)
                clause_obj.tags = _unique_strs(list(clause_obj.tags) + ["decision_checkpoint"])
            clauses[cid] = clause_obj
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
                "core_necessity_tries_used": 0,
                "trap_decision_label": "",
                "trap_decision_flip": False,
                "trap_flip_target_id": "",
                "trap_flip_target_kind": "",
                "trap_graph_excludable_count": 0,
                "trap_graph_excludable_ids": [],
                "trap_graph_excludable_forced_ids": [],
                "trap_graph_excludable_forced_reasons": [],
                "decision_checkpoint_trap_count": 0,
                "decision_checkpoint_trap_ids": [],
                "decision_checkpoint_trap_excludable_ids": [],
                "trap_invalidation_attached_to_update": False,
                "invalidation_update_injected": False,
                "invalidation_update_step_idx": None,
                "defer_budget_rate": float(defer_budget_rate),
                "trap_gap_failed": False,
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

                def _maybe_calibrate_needs_more_info(
                    decision_value: str, conditions_value: Sequence[str]
                ) -> tuple[str, List[str]]:
                    if normalize_decision(decision_value) != "needs_more_info":
                        return str(decision_value), list(conditions_value)
                    if str(state.get("budget", "")).strip().lower() != "low":
                        return str(decision_value), list(conditions_value)
                    if trng.random() <= float(defer_budget_rate):
                        return str(decision_value), list(conditions_value)
                    if str(state.get("region", "")).strip().lower() == "eu" and str(
                        state.get("residency", "")
                    ).strip().lower() != "eu":
                        return "require_residency", [
                            f"region={state.get('region')}",
                            f"residency={state.get('residency')}",
                        ]
                    if str(state.get("deadline", "")).strip().lower() == "tight":
                        return "deny", []
                    return "allow", []

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
                    prev_step_for_trap = next(
                        (s for s in reversed(steps) if str(s.kind) != "pivot_check"),
                        None,
                    )

                    def _cleanup_trap_clause_ids(ids: Sequence[str]) -> None:
                        if prev_step_for_trap is None:
                            return
                        id_set = {str(cid) for cid in ids if str(cid).strip()}
                        if not id_set:
                            return
                        prev_step_for_trap.introduced_clause_ids = [
                            cid for cid in list(prev_step_for_trap.introduced_clause_ids) if cid not in id_set
                        ]
                        prev_step_for_trap.avoid_target_ids = [
                            cid for cid in list(prev_step_for_trap.avoid_target_ids) if cid not in id_set
                        ]
                        for cid in id_set:
                            active_invalidated.discard(cid)
                            clauses.pop(cid, None)

                    def _make_trap_clause_set(attempt_idx: int) -> List[str]:
                        if prev_step_for_trap is None or int(trap_distractor_count) <= 0:
                            return []
                        prev_step_idx = int(prev_step_for_trap.step_idx)
                        stale_val = str(state.get(ordinal_key, "unknown"))
                        if len(update_chain) >= 2:
                            prev_upd = clauses.get(update_chain[-2])
                            if prev_upd is not None and prev_upd.state_value is not None:
                                stale_val = str(prev_upd.state_value)
                        mismatch_value = "tight" if str(state.get("deadline", "")).lower() != "tight" else "flex"

                        out_ids: List[str] = []
                        trap_count = max(0, int(trap_distractor_count))
                        forced_avoided_idx = trap_count - 1 if (trap_require_avoided and trap_count > 0) else -1
                        for didx in range(trap_count):
                            mode = didx % 3
                            if didx == forced_avoided_idx:
                                mode = 2
                            lexical_boost = ""
                            if int(attempt_idx) > 0:
                                lexical_boost = " Final policy note: latest change reference."
                            if mode == 0:
                                text = (
                                    f"{message} Historical note: {ordinal_key}={stale_val} remains the reference policy."
                                    f"{lexical_boost}"
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="UPDATE",
                                    text=text,
                                    state_key=str(ordinal_key),
                                    state_value=str(stale_val),
                                    tags=["distractor", "trap", "stale"],
                                    metadata={"trap": True, "trap_kind": "stale", "trap_attempt": int(attempt_idx)},
                                )
                            elif mode == 1:
                                text = (
                                    f"{message} Exception note: allow override if deadline is {mismatch_value}."
                                    f"{lexical_boost}"
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="EXCEPTION",
                                    text=text,
                                    state_key="deadline",
                                    state_value=str(mismatch_value),
                                    tags=["distractor", "trap", "inapplicable"],
                                    metadata={"trap": True, "trap_kind": "inapplicable", "trap_attempt": int(attempt_idx)},
                                )
                            else:
                                text = (
                                    f"{message} Archived binding says {handle_token} keeps prior rule without update."
                                    f"{lexical_boost}"
                                )
                                cid = _new_clause(
                                    step_idx=prev_step_idx,
                                    node_type="ASSUMPTION",
                                    text=text,
                                    tags=["distractor", "trap", "avoided"],
                                    metadata={"trap": True, "trap_kind": "avoided", "trap_attempt": int(attempt_idx)},
                                )
                                prev_step_for_trap.avoid_target_ids = _unique_strs(
                                    list(prev_step_for_trap.avoid_target_ids) + [cid]
                                )
                                active_invalidated.add(cid)
                            prev_step_for_trap.introduced_clause_ids = _unique_strs(
                                list(prev_step_for_trap.introduced_clause_ids) + [cid]
                            )
                            out_ids.append(cid)
                        return out_ids

                    trap_clause_ids = _make_trap_clause_set(0)

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

                    max_attempts = core_necessity_max_tries if core_necessity_enable else 1
                    accepted_core_required_ids: List[str] = []
                    accepted_decision = latest_decision or _decision_from_state(state)
                    accepted_conditions: List[str] = [f"region={state['region']}", f"budget={state['budget']}"]
                    accepted_flip_count = 0
                    accepted_all_required = False
                    accepted_exception = ""
                    accepted_tries_used = 0

                    for attempt_idx in range(1, max_attempts + 1):
                        candidate_core = list(base_candidate_core)
                        trng.shuffle(candidate_core)
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

                        core_required_ids: List[str] = []
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
                            core_required_ids = [cid for cid in _unique_strs(mandatory) if cid in clauses]
                            # Keep compositional core compact to preserve "all pieces required".
                            if len(core_required_ids) > int(core_size_max):
                                core_required_ids = core_required_ids[: int(core_size_max)]
                            desired_core = max(len(core_required_ids), min(int(core_size_min), len(candidate_core)))
                            desired_core = min(desired_core, int(core_size_max), len(candidate_core))
                            if len(core_required_ids) < desired_core:
                                recency_sorted = sorted(
                                    candidate_core,
                                    key=lambda cid: (
                                        int(clauses[cid].step_idx),
                                        str(cid),
                                    ),
                                    reverse=True,
                                )
                                for cid in recency_sorted:
                                    if len(core_required_ids) >= desired_core:
                                        break
                                    if cid not in core_required_ids:
                                        core_required_ids.append(cid)
                            core_required_ids = [
                                cid for cid in _unique_strs(core_required_ids) if cid in clauses
                            ][:desired_core]
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
                                core_required_ids.append(low_overlap_sorted[0])
                            for hid in hidden_core_ids:
                                if hid in candidate_core and hid not in core_required_ids:
                                    core_required_ids.append(hid)
                            if core_necessity_enable:
                                for node_type in ("UPDATE", "DECISION", "ASSUMPTION", "EXCEPTION"):
                                    for cid in reversed(candidate_core):
                                        clause = clauses.get(cid)
                                        if clause is None or str(clause.node_type or "") != node_type:
                                            continue
                                        if cid not in core_required_ids:
                                            core_required_ids.append(cid)
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
                                if len(core_required_ids) >= desired_core:
                                    break
                                if cid not in core_required_ids:
                                    core_required_ids.append(cid)
                            core_required_ids = [
                                cid for cid in _unique_strs(core_required_ids) if cid in clauses
                            ][:desired_core]

                        chosen_exception = ""
                        chosen_updates = []
                        for cid in core_required_ids:
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
                            oracle_decision = _oracle_decision_from_evidence(clauses, core_required_ids, state)
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
                        if core_necessity_enable and core_required_ids:
                            base_decision = _oracle_decision_from_evidence(clauses, core_required_ids, state)
                            for removed_id in core_required_ids:
                                reduced = [cid for cid in core_required_ids if cid != removed_id]
                                reduced_decision = _oracle_decision_from_evidence(clauses, reduced, state)
                                if reduced_decision == "unknown" or reduced_decision != base_decision:
                                    flip_count += 1
                            all_required = bool(flip_count == len(core_required_ids))

                        accepted_core_required_ids = list(core_required_ids)
                        accepted_decision = decision
                        accepted_conditions = list(conditions)
                        accepted_flip_count = int(flip_count)
                        accepted_all_required = bool(all_required)
                        accepted_exception = str(chosen_exception or "")
                        accepted_tries_used = int(attempt_idx)
                        if not core_necessity_enable:
                            break
                        if core_necessity_require_all:
                            if accepted_all_required:
                                break
                        else:
                            break

                    pivot_required_ids = [
                        cid for cid in _unique_strs(accepted_core_required_ids) if cid in clauses
                    ]
                    decision = accepted_decision
                    conditions = list(accepted_conditions)
                    chosen_exception = accepted_exception
                    decision, conditions = _maybe_calibrate_needs_more_info(decision, conditions)

                    trap_decision_label = ""
                    trap_decision_flip = False
                    if trap_decision_flip_enable:
                        trap_decision_label = _flip_decision_label(decision)
                        trap_decision_flip = bool(trap_decision_label and trap_decision_label != str(decision))

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

                    def _best_distractor_similarity(ids: Sequence[str]) -> float:
                        best = 0.0
                        for dcid in ids:
                            dclause = clauses.get(dcid)
                            if dclause is None:
                                continue
                            best = max(best, _jaccard_text(message, str(dclause.text or "")))
                        return float(best)

                    def _rank_distractor_similarity(ids: Sequence[str]) -> List[Tuple[float, str]]:
                        ranked: List[Tuple[float, str]] = []
                        for dcid in ids:
                            dclause = clauses.get(str(dcid))
                            if dclause is None:
                                continue
                            sim = float(_jaccard_text(message, str(dclause.text or "")))
                            ranked.append((sim, str(dcid)))
                        ranked.sort(key=lambda item: (-item[0], item[1]))
                        return ranked

                    def _trap_kind(cid: str) -> str:
                        clause = clauses.get(str(cid))
                        if clause is None:
                            return ""
                        return str((clause.metadata or {}).get("trap_kind", "") or "").strip().lower()

                    def _pick_trap_flip_target(ids: Sequence[str]) -> tuple[str, str]:
                        ordered_ids = [str(cid) for cid in ids if str(cid).strip()]
                        if not ordered_ids or trap_flip_attach_kind == "none":
                            return "", ""
                        if trap_flip_attach_kind == "random":
                            chosen = str(trng.choice(ordered_ids))
                            return chosen, _trap_kind(chosen)
                        preferred = [
                            cid
                            for cid in ordered_ids
                            if _trap_kind(cid) == str(trap_flip_attach_kind)
                        ]
                        target_id = str(preferred[0] if preferred else ordered_ids[0])
                        return target_id, _trap_kind(target_id)

                    def _flip_suffix(decision_label: str) -> str:
                        if not decision_label:
                            return ""
                        if trap_flip_salience >= 0.75:
                            return f"Recommended outcome: {decision_label}."
                        if trap_flip_salience >= 0.35:
                            return f"Some notes suggest outcome: {decision_label}."
                        if trap_flip_salience > 0.0:
                            return f"A minority view once favored: {decision_label}."
                        return ""

                    def _annotate_trap_flip(
                        ids: Sequence[str], decision_label: str, flip_flag: bool
                    ) -> tuple[str, str]:
                        target_id, target_kind = _pick_trap_flip_target(ids)
                        suffix = _flip_suffix(decision_label)
                        if target_id and suffix:
                            tclause = clauses.get(str(target_id))
                            if tclause is not None:
                                tclause.text = f"{tclause.text} {suffix}"
                        for dcid in ids:
                            dclause = clauses.get(str(dcid))
                            if dclause is None:
                                continue
                            dclause.metadata["trap_decision_label"] = decision_label
                            dclause.metadata["trap_decision_flip"] = bool(flip_flag)
                            if target_id and str(dcid) == str(target_id):
                                dclause.metadata["trap_flip_target"] = True
                        return str(target_id), str(target_kind)

                    trap_flip_target_id = ""
                    trap_flip_target_kind = ""
                    if trap_decision_flip_enable:
                        trap_flip_target_id, trap_flip_target_kind = _annotate_trap_flip(
                            trap_clause_ids, trap_decision_label, trap_decision_flip
                        )

                    trap_gap_failed = False
                    best_distractor_sim = _best_distractor_similarity(trap_clause_ids)
                    trap_gap = float(best_distractor_sim - best_gold_sim)
                    if trap_clause_ids and trap_gap <= 0.0:
                        for trap_try in range(1, int(trap_gap_max_tries) + 1):
                            if trap_gap > 0.0:
                                break
                            _cleanup_trap_clause_ids(trap_clause_ids)
                            trap_clause_ids = _make_trap_clause_set(trap_try)
                            if trap_decision_flip_enable:
                                trap_flip_target_id, trap_flip_target_kind = _annotate_trap_flip(
                                    trap_clause_ids, trap_decision_label, trap_decision_flip
                                )
                            best_distractor_sim = _best_distractor_similarity(trap_clause_ids)
                            trap_gap = float(best_distractor_sim - best_gold_sim)
                        if trap_gap <= 0.0 and trap_clause_ids:
                            trap_gap_failed = True

                    decision_checkpoint_trap_ids: List[str] = []
                    checkpoint_ids: List[str] = []
                    for cid in all_prior_ids:
                        clause = clauses.get(str(cid))
                        if clause is None:
                            continue
                        if str(clause.node_type or "") != "DECISION":
                            continue
                        if not str(clause.text or "").startswith("Decision checkpoint:"):
                            continue
                        checkpoint_ids.append(str(cid))

                    controlling_id = ""
                    if checkpoint_ids:
                        controlling_id = max(
                            checkpoint_ids,
                            key=lambda cid: (int(clauses[cid].step_idx), str(cid)),
                        )
                    for cid in checkpoint_ids:
                        clause = clauses.get(str(cid))
                        if clause is None:
                            continue
                        clause.metadata["trap"] = True
                        clause.metadata["trap_kind"] = "decision_checkpoint"
                        clause.metadata["trap_decision_checkpoint"] = True
                        clause.metadata["decision_checkpoint_controlling"] = bool(
                            controlling_id and str(cid) == str(controlling_id)
                        )
                        clause.metadata["trap_decision_label"] = str(trap_decision_label or "")
                        clause.metadata["trap_decision_flip"] = bool(trap_decision_flip)
                        clause.tags = _unique_strs(list(clause.tags) + ["decision_checkpoint"])
                        decision_checkpoint_trap_ids.append(str(cid))

                    trap_clause_ids = _unique_strs(list(trap_clause_ids) + list(decision_checkpoint_trap_ids))
                    if trap_clause_ids:
                        best_distractor_sim = _best_distractor_similarity(trap_clause_ids)
                        trap_gap = float(best_distractor_sim - best_gold_sim)
                        trap_gap_failed = bool(trap_gap <= 0.0)

                    trap_graph_excludable_ids: List[str] = []
                    trap_graph_excludable_forced_ids: List[str] = []
                    trap_graph_excludable_forced_reasons: List[Dict[str, str]] = []
                    trap_invalidation_attached_to_update = False
                    if trap_clause_ids:
                        eligible = [
                            str(cid)
                            for cid in trap_clause_ids
                            if _trap_kind(str(cid)) in trap_graph_excludable_kind_set
                        ]
                        eligible_set = set(eligible)
                        forced_reason_map: Dict[str, List[str]] = {}

                        def _force_add(cid: str, reason: str) -> None:
                            sid = str(cid or "").strip()
                            if not sid or sid not in eligible_set:
                                return
                            reasons = forced_reason_map.setdefault(sid, [])
                            if reason not in reasons:
                                reasons.append(reason)

                        if trap_graph_force_include_flip_target and trap_flip_target_id:
                            _force_add(str(trap_flip_target_id), "flip_target")

                        if trap_graph_force_topk > 0 and eligible:
                            ranked_all = _rank_distractor_similarity(eligible)
                            for _, rcid in ranked_all[: int(trap_graph_force_topk)]:
                                _force_add(str(rcid), "topk_sim")

                        if trap_graph_force_include_decision_checkpoint and eligible:
                            checkpoint_eligible = [
                                str(cid)
                                for cid in eligible
                                if _trap_kind(str(cid)) == "decision_checkpoint"
                            ]
                            if checkpoint_eligible:
                                for checkpoint_id in checkpoint_eligible:
                                    _force_add(str(checkpoint_id), "decision_checkpoint_all")

                        sampled_ids: List[str] = []
                        for cid in eligible:
                            if trng.random() <= float(trap_graph_excludable_rate):
                                sampled_ids.append(str(cid))

                        trap_graph_excludable_ids = _unique_strs(
                            list(sampled_ids) + list(forced_reason_map.keys())
                        )
                        if eligible and not trap_graph_excludable_ids:
                            fallback_id = str(eligible[0])
                            trap_graph_excludable_ids = [fallback_id]
                            _force_add(fallback_id, "fallback_first")

                        trap_graph_excludable_forced_ids = [
                            cid for cid in trap_graph_excludable_ids if cid in forced_reason_map
                        ]
                        trap_graph_excludable_forced_reasons = [
                            {"id": cid, "reason": "+".join(forced_reason_map.get(cid, []))}
                            for cid in trap_graph_excludable_forced_ids
                        ]
                    decision_checkpoint_trap_excludable_ids = [
                        cid
                        for cid in trap_graph_excludable_ids
                        if _trap_kind(str(cid)) == "decision_checkpoint"
                    ]

                    def _ensure_latest_update_step_for_trap(
                        *,
                        required_ids: Sequence[str] | None = None,
                    ) -> tuple[TraceStep | None, bool]:
                        latest_update_step: TraceStep | None = None
                        for s in reversed(steps):
                            if str(s.kind) == "update":
                                latest_update_step = s
                                break
                        required = [str(cid) for cid in (required_ids or []) if str(cid).strip()]
                        requires_injected_update = latest_update_step is None
                        if latest_update_step is not None and required:
                            latest_update_idx = int(latest_update_step.step_idx)
                            late_required = [
                                cid
                                for cid in required
                                if clauses.get(str(cid)) is not None
                                and int(clauses[str(cid)].step_idx) > latest_update_idx
                            ]
                            requires_injected_update = bool(late_required)
                        if not requires_injected_update and latest_update_step is not None:
                            return latest_update_step, False
                        if not steps:
                            return None, False
                        fallback_step = steps[-1]
                        fallback_step.kind = "update"
                        if "update" not in str(fallback_step.message or "").lower():
                            base_message = str(fallback_step.message or "").strip()
                            fallback_step.message = (
                                f"{base_message} User update reconciles earlier conflicting notes.".strip()
                            )
                        noop_key = str(ordinal_key)
                        noop_val = str(state.get(noop_key, ""))
                        noop_id = _new_clause(
                            step_idx=int(fallback_step.step_idx),
                            node_type="UPDATE",
                            text=f"Update: {noop_key} remains {noop_val}.",
                            state_key=noop_key,
                            state_value=noop_val,
                            tags=["update", "graph_excludable_noop"],
                            metadata={"noop_update": True},
                        )
                        fallback_step.introduced_clause_ids = _unique_strs(
                            list(fallback_step.introduced_clause_ids) + [noop_id]
                        )
                        update_ids.append(noop_id)
                        update_ids_by_key.setdefault(noop_key, []).append(noop_id)
                        return fallback_step, True

                    invalidation_update_injected = False
                    invalidation_update_step_idx: int | None = None
                    if trap_graph_excludable_ids:
                        update_step_for_trap, invalidation_update_injected = _ensure_latest_update_step_for_trap(
                            required_ids=decision_checkpoint_trap_excludable_ids,
                        )
                        if update_step_for_trap is not None:
                            update_step_for_trap.avoid_target_ids = _unique_strs(
                                list(update_step_for_trap.avoid_target_ids) + list(trap_graph_excludable_ids)
                            )
                            invalidation_update_step_idx = int(update_step_for_trap.step_idx)
                            trap_invalidation_attached_to_update = True
                            active_invalidated.update(trap_graph_excludable_ids)
                            hint_ids = ",".join(list(trap_graph_excludable_ids)[:3])
                            hint_suffix = ""
                            if trap_invalidation_text_strength >= 0.75:
                                hint_suffix = (
                                    f"This update supersedes earlier notes, including: {hint_ids}."
                                )
                            elif trap_invalidation_text_strength >= 0.35:
                                hint_suffix = (
                                    f"Earlier notes ({hint_ids}) are not controlling under current state."
                                )
                            elif trap_invalidation_text_strength > 0.0:
                                hint_suffix = (
                                    f"Cross-check: prior note IDs ({hint_ids}) may no longer control."
                                )
                            if hint_suffix:
                                for uid in reversed(list(update_step_for_trap.introduced_clause_ids or [])):
                                    uclause = clauses.get(str(uid))
                                    if uclause is None or str(uclause.node_type or "") != "UPDATE":
                                        continue
                                    if hint_suffix not in str(uclause.text or ""):
                                        uclause.text = f"{uclause.text} {hint_suffix}"
                                    uclause.metadata["trap_graph_excludable_ids"] = list(
                                        trap_graph_excludable_ids
                                    )
                                    break

                    pivot_meta = {
                        "indirect_active": True,
                        "pivot_style": str(style_choice),
                        "indirection_overlap_gold": float(indirection_overlap),
                        "best_gold_sim": float(best_gold_sim),
                        "best_distractor_sim": float(best_distractor_sim),
                        "trap_gap": float(trap_gap),
                        "trap_present": bool(trap_gap > 0.0 and len(trap_clause_ids) > 0),
                        "trap_gap_failed": bool(trap_gap_failed),
                        "core_size": int(len(pivot_required_ids)),
                        "trap_distractor_ids": list(trap_clause_ids),
                        "trap_flip_target_id": str(trap_flip_target_id or ""),
                        "trap_flip_target_kind": str(trap_flip_target_kind or ""),
                        "trap_graph_excludable_count": int(len(trap_graph_excludable_ids)),
                        "trap_graph_excludable_ids": list(trap_graph_excludable_ids),
                        "trap_graph_excludable_forced_ids": list(
                            trap_graph_excludable_forced_ids
                        ),
                        "trap_graph_excludable_forced_reasons": list(
                            trap_graph_excludable_forced_reasons
                        ),
                        "decision_checkpoint_trap_count": int(len(decision_checkpoint_trap_ids)),
                        "decision_checkpoint_trap_ids": list(decision_checkpoint_trap_ids),
                        "decision_checkpoint_trap_excludable_ids": list(
                            decision_checkpoint_trap_excludable_ids
                        ),
                        "trap_invalidation_attached_to_update": bool(trap_invalidation_attached_to_update),
                        "invalidation_update_injected": bool(invalidation_update_injected),
                        "invalidation_update_step_idx": (
                            int(invalidation_update_step_idx)
                            if isinstance(invalidation_update_step_idx, int)
                            else None
                        ),
                        "trap_flip_salience": float(trap_flip_salience),
                        "trap_flip_attach_kind": str(trap_flip_attach_kind),
                        "trap_graph_excludable_rate": float(trap_graph_excludable_rate),
                        "trap_graph_excludable_kinds": str(trap_graph_excludable_kinds),
                        "trap_graph_force_topk": int(trap_graph_force_topk),
                        "trap_graph_force_include_flip_target": bool(
                            trap_graph_force_include_flip_target
                        ),
                        "trap_graph_force_include_decision_checkpoint": bool(
                            trap_graph_force_include_decision_checkpoint
                        ),
                        "trap_invalidation_text_strength": float(trap_invalidation_text_strength),
                        "defer_budget_rate": float(defer_budget_rate),
                        "core_necessity_enable": bool(core_necessity_enable),
                        "core_necessity_flip_count": int(accepted_flip_count if core_necessity_enable else 0),
                        "core_necessity_all_required": bool(
                            accepted_all_required if core_necessity_enable else False
                        ),
                        "core_necessity_failed": bool(
                            core_necessity_enable and core_necessity_require_all and not accepted_all_required
                        ),
                        "core_necessity_tries_used": int(accepted_tries_used if core_necessity_enable else 0),
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

                    decision, conditions = _maybe_calibrate_needs_more_info(decision, conditions)

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
                "traceops_trap_flip_salience": float(trap_flip_salience),
                "traceops_trap_flip_attach_kind": str(trap_flip_attach_kind),
                "traceops_trap_graph_excludable_rate": float(trap_graph_excludable_rate),
                "traceops_trap_graph_excludable_kinds": str(trap_graph_excludable_kinds),
                "traceops_trap_graph_excludable_unknown_tokens": list(
                    trap_graph_excludable_unknown_tokens
                ),
                "traceops_trap_graph_force_topk": int(trap_graph_force_topk),
                "traceops_trap_graph_force_include_flip_target": bool(
                    trap_graph_force_include_flip_target
                ),
                "traceops_trap_graph_force_include_decision_checkpoint": bool(
                    trap_graph_force_include_decision_checkpoint
                ),
                "traceops_trap_invalidation_text_strength": float(trap_invalidation_text_strength),
                "traceops_defer_budget_rate": float(defer_budget_rate),
                "traceops_hidden_core_enable": bool(hidden_core_enable),
                "traceops_hidden_core_kind": str(hidden_core_kind),
                "traceops_hidden_core_link_mode": str(hidden_core_link_mode),
                "traceops_trap_require_avoided": bool(trap_require_avoided),
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
        "traceops_trap_flip_salience": float(trap_flip_salience),
        "traceops_trap_flip_attach_kind": str(trap_flip_attach_kind),
        "traceops_trap_graph_excludable_rate": float(trap_graph_excludable_rate),
        "traceops_trap_graph_excludable_kinds": str(trap_graph_excludable_kinds),
        "traceops_trap_graph_excludable_unknown_tokens": list(
            trap_graph_excludable_unknown_tokens
        ),
        "traceops_trap_graph_force_topk": int(trap_graph_force_topk),
        "traceops_trap_graph_force_include_flip_target": bool(
            trap_graph_force_include_flip_target
        ),
        "traceops_trap_graph_force_include_decision_checkpoint": bool(
            trap_graph_force_include_decision_checkpoint
        ),
        "traceops_trap_invalidation_text_strength": float(trap_invalidation_text_strength),
        "traceops_defer_budget_rate": float(defer_budget_rate),
        "traceops_hidden_core_enable": bool(hidden_core_enable),
        "traceops_hidden_core_kind": str(hidden_core_kind),
        "traceops_hidden_core_link_mode": str(hidden_core_link_mode),
        "traceops_trap_require_avoided": bool(trap_require_avoided),
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
