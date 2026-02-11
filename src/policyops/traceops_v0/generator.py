from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .schema import TraceGold, TraceStep, TraceThread, TraceWorldClause

TRACEOPS_SCENARIOS = {"exception", "contradiction", "latent", "mixed"}


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
) -> tuple[List[TraceThread], Dict[str, Any]]:
    rng = random.Random(int(seed))
    levelsafe = max(0, int(level))
    thread_count = max(1, int(threads))
    delay_default = 1 if levelsafe <= 1 else 3 if levelsafe == 2 else 6
    delay = max(1, int(delay_to_relevance if delay_to_relevance is not None else delay_default))
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
        active_invalidated: set[str] = set()

        latent_clause_id: str | None = None
        latent_intro_step: int | None = None

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

            elif kind == "update":
                message = "User update arrived; prior assumptions may be invalid."
                flips = max(1, int(state_flip_count))
                update_step_ids: List[str] = []
                for _ in range(flips):
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
                    ex_id = _new_clause(
                        step_idx=step_idx,
                        node_type="EXCEPTION",
                        text="Exception activated: manual override allowed if deadline is tight.",
                        state_key=None,
                        state_value=None,
                        depends_on=update_step_ids[:1],
                        tags=["exception", "activation"],
                    )
                    introduced.append(ex_id)
                    if latent_clause_id is None:
                        latent_clause_id = ex_id
                        latent_intro_step = int(step_idx)

            elif kind == "pivot_check":
                message = "Given all updates so far, what is the final decision now?"
            else:
                message = "Continue processing the trace."

            step_gold: TraceGold | None = None
            pivot_required_ids: List[str] = []
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
                },
            )
            steps.append(step)

        thread = TraceThread(
            thread_id=thread_id,
            level=int(levelsafe),
            scenario=scenario,
            initial_state=dict(initial_state),
            steps=steps,
            clauses=clauses,
            meta={
                "trace_len": len(steps),
                "pivot_count": sum(1 for s in steps if s.kind == "pivot_check"),
                "delay_to_relevance": int(delay),
                "distractor_branching": int(max(1, distractor_branching)),
                "contradiction_rate": float(contradiction_rate),
                "exception_density": float(exception_density),
                "state_flip_count": int(max(1, state_flip_count)),
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
