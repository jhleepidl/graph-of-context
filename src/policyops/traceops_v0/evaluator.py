from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Set, Tuple

from .schema import TraceGold, TraceStep, TraceThread, TraceWorldClause

if TYPE_CHECKING:
    from ..baselines import LLMClient


def _unique_strs(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        sval = str(value).strip()
        if not sval or sval in seen:
            continue
        seen.add(sval)
        out.append(sval)
    return out


def _tokenize(text: str) -> Set[str]:
    toks = []
    token = []
    for ch in str(text).lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
        elif token:
            toks.append("".join(token))
            token = []
    if token:
        toks.append("".join(token))
    return set(toks)


def _build_traceops_llm_prompt(
    step: TraceStep,
    thread: TraceThread,
    context_ids: Sequence[str],
) -> str:
    allowed_ids = _unique_strs(list(context_ids))
    allowed_conditions: List[str] = ["apply_latest_update"]
    state = step.state if isinstance(step.state, dict) else {}
    for key, value in state.items():
        k = str(key).strip()
        v = str(value).strip()
        if not k or not v:
            continue
        allowed_conditions.append(f"{k}={v}")
    for cid in allowed_ids:
        clause = thread.clauses.get(cid)
        if clause is None:
            continue
        if str(clause.node_type or "") == "EXCEPTION":
            allowed_conditions.append(f"exception={cid}")
    allowed_conditions = _unique_strs(allowed_conditions)

    lines: List[str] = []
    lines.append("Use ONLY the provided CONTEXT. Output STRICT JSON only.")
    lines.append(
        'JSON schema: {"decision":"allow|deny|require_condition|needs_more_info","conditions":["<condition_string>"],"evidence":["<clause_id>"]}'
    )
    lines.append("You MUST choose one of the 4 decision labels exactly as written.")
    lines.append(
        "For conditions: choose strings ONLY from ALLOWED_CONDITIONS (copy exact); no free-form text."
    )
    lines.append("Return only a JSON object; no markdown, no explanations.")
    lines.append("")
    lines.append(f"Pivot question: {step.message}")
    lines.append("")
    lines.append("Allowed evidence clause IDs:")
    lines.append(", ".join(allowed_ids) if allowed_ids else "(none)")
    lines.append("")
    lines.append("ALLOWED_CONDITIONS:")
    if allowed_conditions:
        for cond in allowed_conditions:
            lines.append(f"- {cond}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("CONTEXT:")
    for cid in allowed_ids:
        clause = thread.clauses.get(cid)
        if clause is None:
            continue
        ctype = str(clause.node_type or "UNKNOWN")
        ctext = str(clause.text or "").replace("\n", " ").strip()
        lines.append(f"CLAUSE {cid} ({ctype}): {ctext}")
    return "\n".join(lines)


def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_llm_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    parsed: Dict[str, Any] | None = None
    if raw:
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None
        if parsed is None:
            parsed = _extract_first_json_object(raw)
        if parsed is None:
            brace_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if brace_match:
                try:
                    candidate = json.loads(brace_match.group(0))
                    if isinstance(candidate, dict):
                        parsed = candidate
                except Exception:
                    parsed = None
    if parsed is None:
        return {
            "decision": "needs_more_info",
            "conditions": [],
            "evidence": [],
            "parse_error": True,
        }
    decision = str(parsed.get("decision", "needs_more_info") or "needs_more_info")
    if decision not in {"allow", "deny", "require_condition", "needs_more_info"}:
        decision = "needs_more_info"
    conditions_raw = parsed.get("conditions")
    conditions_items = list(conditions_raw) if isinstance(conditions_raw, list) else []
    conditions = _unique_strs([str(item).strip() for item in conditions_items])
    evidence_raw = parsed.get("evidence")
    evidence = _unique_strs(list(evidence_raw) if isinstance(evidence_raw, list) else [])
    return {
        "decision": decision,
        "conditions": conditions,
        "evidence": evidence,
        "parse_error": False,
    }


def _clause_applicable(clause: TraceWorldClause, state: Dict[str, Any]) -> bool:
    key = clause.state_key
    val = clause.state_value
    if not key or val is None:
        return True
    return str(state.get(key)) == str(val)


def _gold_decision_family(gold_decision: str) -> str:
    decision = str(gold_decision or "")
    if decision == "allow":
        return "allow"
    if decision == "deny":
        return "deny"
    if decision == "defer":
        return "needs_more_info"
    if decision.startswith("require_"):
        return "require_condition"
    if decision == "override_invalidated":
        return "require_condition"
    return decision


def _dependency_closure(
    seed_ids: Sequence[str],
    clauses: Dict[str, TraceWorldClause],
    *,
    max_hops: int,
    universe_ids: Sequence[str],
) -> List[str]:
    ordered_universe = _unique_strs(universe_ids)
    universe_set = set(ordered_universe)
    seeds = [cid for cid in _unique_strs(seed_ids) if cid in universe_set]
    if not seeds:
        return []
    if int(max_hops) <= 0:
        return seeds

    neighbors: Dict[str, Set[str]] = {cid: set() for cid in ordered_universe}
    for cid in ordered_universe:
        clause = clauses.get(cid)
        if clause is None:
            continue
        for dep in clause.depends_on:
            dep_id = str(dep)
            if dep_id in universe_set and dep_id != cid:
                neighbors[cid].add(dep_id)
                neighbors.setdefault(dep_id, set()).add(cid)

    visited: Set[str] = set(seeds)
    frontier = list(seeds)
    for _ in range(int(max_hops)):
        if not frontier:
            break
        next_frontier: List[str] = []
        for cid in frontier:
            for nid in neighbors.get(cid, set()):
                if nid in visited:
                    continue
                visited.add(nid)
                next_frontier.append(nid)
        frontier = next_frontier

    uni_rank = {cid: idx for idx, cid in enumerate(ordered_universe)}
    return sorted(list(visited), key=lambda cid: (uni_rank.get(cid, 10**9), cid))


def _choose_traceops_context(
    method: str,
    *,
    thread: TraceThread,
    step: TraceStep,
    history_ids: Sequence[str],
    invalidated_ids: Sequence[str],
    args: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    clauses = thread.clauses
    ordered_history = _unique_strs(history_ids)
    avoid_set = set(_unique_strs(invalidated_ids))

    context_ids: List[str] = []
    debug: Dict[str, Any] = {}

    if method in {"full", "full_history", "oracle"}:
        context_ids = list(ordered_history)
    elif method in {"similarity_only", "topk", "engine"}:
        q_tokens = _tokenize(step.message)
        scored: List[tuple[int, int, str]] = []
        for idx, cid in enumerate(ordered_history):
            clause = clauses.get(cid)
            if clause is None:
                continue
            score = len(q_tokens & _tokenize(clause.text))
            scored.append((score, -idx, cid))
        scored.sort(reverse=True)
        k = max(4, int(getattr(args, "traceops_similarity_topk", 8) or 8))
        context_ids = [cid for _, _, cid in scored[:k]]
    elif method in {"agent_fold", "lossy_summary"}:
        # Intentionally lossy: keep only recent decision/update style notes.
        recent = list(reversed(ordered_history))
        keep: List[str] = []
        for cid in recent:
            clause = clauses.get(cid)
            if clause is None:
                continue
            if clause.node_type in {"DECISION", "UPDATE", "ASSUMPTION"}:
                keep.append(cid)
            if len(keep) >= 6:
                break
        context_ids = list(reversed(keep))
    elif method in {"goc", "goc_oracle", "goc_force_required"}:
        use_avoids = bool(getattr(args, "goc_enable_avoids", True))
        candidates = [cid for cid in ordered_history if (cid not in avoid_set or not use_avoids)]
        if not candidates:
            candidates = list(ordered_history)
        context_ordered_history = list(ordered_history)
        should_force_include_required = bool(
            method in {"goc_oracle", "goc_force_required"}
            or bool(getattr(args, "traceops_force_include_required", False))
        )
        universe_ids: List[str] = []
        universe_cap = max(1, int(getattr(args, "traceops_universe_debug_cap", 200) or 200))

        def _applicable_rate(ids: Sequence[str]) -> float:
            id_list = list(_unique_strs(ids))
            if not id_list:
                return float("nan")
            applicable_count = 0
            for cid in id_list:
                clause = clauses.get(cid)
                if clause and _clause_applicable(clause, step.state):
                    applicable_count += 1
            return float(applicable_count) / float(len(id_list))

        def _latest_of_type(node_type: str) -> str | None:
            for cid in reversed(context_ordered_history):
                clause = clauses.get(cid)
                if clause is None or clause.node_type != node_type:
                    continue
                if use_avoids and cid in avoid_set:
                    continue
                return cid
            return None

        anchor_ids: List[str] = []
        for anchor_type in ("DECISION", "UPDATE", "EXCEPTION"):
            aid = _latest_of_type(anchor_type)
            if aid:
                anchor_ids.append(aid)
        anchor_ids = _unique_strs(anchor_ids)

        seed_ids: List[str] = []
        seed_topk = max(1, int(getattr(args, "goc_applicability_seed_topk", 8) or 8))
        if bool(getattr(args, "goc_applicability_seed_enable", False)):
            scored: List[tuple[float, int, str]] = []
            for idx, cid in enumerate(candidates):
                clause = clauses.get(cid)
                if clause is None:
                    continue
                if use_avoids and cid in avoid_set:
                    continue
                if not _clause_applicable(clause, step.state):
                    continue
                score = 1.0
                if clause.node_type in {"EXCEPTION", "UPDATE"}:
                    score += 0.3
                if clause.node_type == "DECISION":
                    score += 0.2
                scored.append((score, idx, cid))
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            seed_ids = [cid for _, _, cid in scored if cid not in set(anchor_ids)][:seed_topk]

        closure_added: List[str] = []
        selected_before_closure = _unique_strs(anchor_ids + seed_ids)
        if bool(getattr(args, "goc_dependency_closure_enable", False)):
            hops = max(0, int(getattr(args, "goc_dependency_closure_hops", 1) or 1))
            topk = max(0, int(getattr(args, "goc_dependency_closure_topk", 12) or 12))
            universe_mode = str(getattr(args, "goc_dependency_closure_universe", "candidates") or "candidates")
            if universe_mode == "world":
                raw_universe_ids = list(thread.clauses.keys())
            elif universe_mode == "memory_opened":
                raw_universe_ids = list(ordered_history)
            else:
                raw_universe_ids = list(candidates)
            # Never let closure peek into future steps.
            universe_ids = [
                cid
                for cid in _unique_strs(raw_universe_ids)
                if cid in clauses and int(clauses[cid].step_idx) < int(step.step_idx)
            ]
            closure = _dependency_closure(selected_before_closure, clauses, max_hops=hops, universe_ids=universe_ids)
            base_set = set(selected_before_closure)
            for cid in closure:
                if cid in base_set:
                    continue
                if use_avoids and cid in avoid_set:
                    continue
                clause = clauses.get(cid)
                if clause is None:
                    continue
                if _clause_applicable(clause, step.state) or clause.node_type in {"EXCEPTION", "UPDATE"}:
                    closure_added.append(cid)
                if len(closure_added) >= topk:
                    break
        context_ids = _unique_strs(anchor_ids + seed_ids + closure_added)

        base_max = int(getattr(args, "goc_unfold_max_nodes", 0) or 0)
        if base_max <= 0:
            base_max = 999

        force_ids: List[str] = []
        force_but_avoided: List[str] = []
        force_but_inapplicable: List[str] = []
        if should_force_include_required:
            required_ids = _unique_strs(
                list(step.pivot_required_ids or [])
                or (list(step.gold.evidence_core_ids) if step.gold is not None else [])
            )
            opened_history_set = set(ordered_history)
            force_ids = [cid for cid in required_ids if cid in opened_history_set]
            force_but_avoided = [cid for cid in force_ids if cid in avoid_set]
            for cid in force_ids:
                clause = clauses.get(cid)
                if clause and (not _clause_applicable(clause, step.state)):
                    force_but_inapplicable.append(cid)
            context_ids = _unique_strs(context_ids + force_ids)

        if len(context_ids) > base_max:
            context_ids = context_ids[:base_max]

        debug = {
            "goc_anchor_ids": list(anchor_ids),
            "goc_applicability_seed_ids": list(seed_ids),
            "goc_applicability_seed_applicable_rate": _applicable_rate(seed_ids),
            "goc_dependency_closure_added_ids": list(closure_added),
            "goc_dependency_closure_added_applicable_rate": _applicable_rate(closure_added),
            "goc_dependency_closure_universe_effective": (
                list(universe_ids[:universe_cap])
                if bool(getattr(args, "goc_dependency_closure_enable", False))
                else []
            ),
            "goc_dependency_closure_universe_effective_size": int(len(universe_ids)) if bool(getattr(args, "goc_dependency_closure_enable", False)) else 0,
            "goc_avoid_target_clause_ids": list(_unique_strs(invalidated_ids)),
            "goc_required_force_included_ids": list(force_ids),
            "goc_required_force_included_but_avoided_ids": list(force_but_avoided),
            "goc_required_force_included_but_inapplicable_ids": list(force_but_inapplicable),
        }
    else:
        context_ids = list(ordered_history)

    return _unique_strs(context_ids), debug


def _infer_prediction(
    *,
    step: TraceStep,
    context_ids: Sequence[str],
    invalidated_ids: Sequence[str],
) -> Dict[str, Any]:
    gold = step.gold or TraceGold(
        decision="needs_more_info",
        conditions=[],
        evidence_ids=[],
        evidence_core_ids=[],
        evidence_meta_ids=[],
    )
    context_set = set(_unique_strs(context_ids))
    required = set(_unique_strs(step.pivot_required_ids or gold.evidence_core_ids or gold.evidence_ids))

    if (not required) or required.issubset(context_set):
        decision = gold.decision
        conditions = list(gold.conditions)
        evidence = list(_unique_strs(list(required) or list(gold.evidence_core_ids) or list(gold.evidence_ids)))
    else:
        decision = "needs_more_info"
        conditions = []
        evidence = list(context_ids)[-1:] if context_ids else []

    return {
        "decision": decision,
        "conditions": conditions,
        "evidence": _unique_strs(evidence),
    }


def _score_step(
    *,
    thread: TraceThread,
    step: TraceStep,
    prediction: Dict[str, Any],
    context_ids: Sequence[str],
    invalidated_ids: Sequence[str],
    eval_mode: str = "deterministic",
) -> Dict[str, Any]:
    gold = step.gold
    if gold is None:
        return {}

    context_set = set(_unique_strs(context_ids))
    required_ids = _unique_strs(step.pivot_required_ids or gold.evidence_core_ids or gold.evidence_ids)
    required_set = set(required_ids)

    pred_decision = str(prediction.get("decision", ""))
    pred_conditions = _unique_strs(prediction.get("conditions") or [])
    pred_evidence = _unique_strs(prediction.get("evidence") or [])

    gold_decision = str(gold.decision)
    gold_decision_family = _gold_decision_family(gold_decision)
    decision_correct_exact = bool(pred_decision == gold_decision)
    decision_correct_family = bool(pred_decision == gold_decision_family)
    decision_correct = decision_correct_family if str(eval_mode) == "llm" else decision_correct_exact
    gold_conditions = _unique_strs(gold.conditions)
    conditions_correct_exact = bool(set(pred_conditions) == set(gold_conditions))
    conditions_correct_subset = bool(set(gold_conditions).issubset(set(pred_conditions)))
    if str(eval_mode) == "llm":
        conditions_correct = conditions_correct_subset
    else:
        conditions_correct = conditions_correct_exact
    answer_correct = bool(decision_correct and conditions_correct)
    evidence_valid_in_context = bool(all(cid in context_set for cid in pred_evidence))
    critical_coverage = (
        float(len(required_set & context_set)) / float(len(required_set))
        if required_set
        else float("nan")
    )
    revive_success = bool(required_set.issubset(context_set)) if required_set else True

    avoid_set = set(_unique_strs(invalidated_ids))
    stale_set = context_set & avoid_set
    avoided_injected = bool(stale_set)
    inapplicable = 0
    for cid in context_set:
        clause = thread.clauses.get(cid)
        if clause and not _clause_applicable(clause, step.state):
            inapplicable += 1
    inapplicable_rate = float(inapplicable) / float(len(context_set)) if context_set else 0.0

    token_est = int(math.ceil(max(0, len(step.message) + sum(len(thread.clauses[cid].text) for cid in context_set if cid in thread.clauses)) / 4.0))

    return {
        "decision_correct": decision_correct,
        "gold_decision_family": gold_decision_family,
        "decision_correct_exact": decision_correct_exact,
        "decision_correct_family": decision_correct_family,
        "conditions_correct": conditions_correct,
        "conditions_correct_exact": conditions_correct_exact,
        "conditions_correct_subset": conditions_correct_subset,
        "answer_correct": answer_correct,
        "evidence_valid_in_context": evidence_valid_in_context,
        "critical_coverage": critical_coverage,
        "revive_success": revive_success,
        "avoided_injected": avoided_injected,
        "stale_present": bool(stale_set),
        "stale_count": int(len(stale_set)),
        "inapplicable_injected_rate": inapplicable_rate,
        "pred_decision": pred_decision,
        "pred_conditions": pred_conditions,
        "pred_evidence": pred_evidence,
        "token_est": token_est,
    }


def evaluate_traceops_method(
    method: str,
    threads: Sequence[TraceThread],
    *,
    args: Any,
    client: "LLMClient | None" = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    thread_records: List[Dict[str, Any]] = []

    eval_mode = str(getattr(args, "traceops_eval_mode", "deterministic") or "deterministic").strip().lower()
    if eval_mode not in {"deterministic", "llm"}:
        eval_mode = "deterministic"
    use_llm = eval_mode == "llm"
    if use_llm and client is None:
        raise RuntimeError("traceops_eval_mode=llm requires a non-null client")

    llm_temperature = float(getattr(args, "traceops_llm_temperature", 0.0) or 0.0)
    llm_max_output_tokens = int(getattr(args, "traceops_llm_max_output_tokens", 256) or 256)
    llm_max_pivots = int(getattr(args, "traceops_llm_max_pivots", 0) or 0)
    llm_pivots_seen = 0

    cache_dir = Path(str(getattr(args, "traceops_llm_cache_dir", ".cache/traceops_llm") or ".cache/traceops_llm"))
    if use_llm:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _usage_int(usage: Dict[str, Any], key: str) -> int | None:
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _mean(values: Sequence[float]) -> float:
        clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not clean:
            return float("nan")
        return float(sum(clean) / len(clean))

    max_steps = int(getattr(args, "traceops_max_steps", 0) or 0)
    stop_all = False
    for thread in threads:
        if stop_all:
            break
        history_ids: List[str] = []
        invalidated_ids: List[str] = []
        pivot_step_records: List[Dict[str, Any]] = []

        for step in thread.steps:
            if stop_all:
                break
            if max_steps > 0 and int(step.step_idx) >= max_steps:
                break
            history_ids = _unique_strs(history_ids + list(step.introduced_clause_ids))
            if step.kind == "update" and step.avoid_target_ids:
                invalidated_ids = _unique_strs(invalidated_ids + list(step.avoid_target_ids))

            if step.kind != "pivot_check" or step.gold is None:
                continue
            if use_llm and llm_max_pivots > 0 and llm_pivots_seen >= llm_max_pivots:
                stop_all = True
                break

            context_ids, debug = _choose_traceops_context(
                method,
                thread=thread,
                step=step,
                history_ids=history_ids,
                invalidated_ids=invalidated_ids,
                args=args,
            )
            llm_output_text = ""
            llm_parse_error = False
            llm_usage: Dict[str, Any] = {}
            if use_llm:
                prompt = _build_traceops_llm_prompt(step, thread, context_ids)
                task_id = f"{thread.thread_id}:{step.step_id}"
                model_name = str(getattr(args, "model", "") or "")
                cache_key = hashlib.sha256(
                    f"{model_name}\n{method}\n{task_id}\n{prompt}".encode("utf-8")
                ).hexdigest()
                cache_path = cache_dir / f"{cache_key}.json"
                cached_payload: Dict[str, Any] = {}
                if cache_path.exists():
                    try:
                        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    except Exception:
                        cached_payload = {}
                if isinstance(cached_payload, dict) and isinstance(cached_payload.get("output_text"), str):
                    llm_output_text = str(cached_payload.get("output_text") or "")
                    usage_obj = cached_payload.get("usage")
                    llm_usage = usage_obj if isinstance(usage_obj, dict) else {}
                else:
                    if hasattr(client, "generate_with_usage"):
                        llm_output_text, llm_usage = client.generate_with_usage(  # type: ignore[attr-defined]
                            prompt,
                            temperature=llm_temperature,
                            max_output_tokens=llm_max_output_tokens,
                        )
                    else:
                        llm_output_text = client.generate(prompt)  # type: ignore[union-attr]
                        llm_usage = {}
                    cache_payload = {
                        "output_text": llm_output_text,
                        "usage": llm_usage,
                        "raw": None,
                    }
                    cache_path.write_text(
                        json.dumps(cache_payload, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                pred = _parse_llm_json(llm_output_text)
                llm_parse_error = bool(pred.get("parse_error"))
                llm_pivots_seen += 1
            else:
                pred = _infer_prediction(
                    step=step,
                    context_ids=context_ids,
                    invalidated_ids=invalidated_ids,
                )

            score = _score_step(
                thread=thread,
                step=step,
                prediction=pred,
                context_ids=context_ids,
                invalidated_ids=invalidated_ids,
                eval_mode=eval_mode,
            )

            prompt_tokens_actual = _usage_int(llm_usage, "input_tokens")
            completion_tokens_actual = _usage_int(llm_usage, "output_tokens")
            total_tokens_actual = _usage_int(llm_usage, "total_tokens")
            if (
                total_tokens_actual is None
                and prompt_tokens_actual is not None
                and completion_tokens_actual is not None
            ):
                total_tokens_actual = int(prompt_tokens_actual + completion_tokens_actual)

            rec = {
                "task_id": f"{thread.thread_id}:{step.step_id}",
                "thread_id": thread.thread_id,
                "episode_id": int(step.step_idx) + 1,
                "step_idx": int(step.step_idx),
                "step_kind": step.kind,
                "is_pivot_task": True,
                "traceops_level": int(thread.level),
                "traceops_scenario": str(thread.scenario),
                "traceops_eval_mode": eval_mode,
                "gold_decision": str(step.gold.decision),
                "gold_decision_family": str(score.get("gold_decision_family", "")),
                "gold_conditions": list(step.gold.conditions),
                "gold_evidence_ids": list(step.gold.evidence_ids),
                "pivot_required_clause_ids": list(step.pivot_required_ids),
                "pred_decision": score.get("pred_decision"),
                "pred_conditions": score.get("pred_conditions"),
                "pred_evidence": score.get("pred_evidence"),
                "decision_correct": bool(score.get("decision_correct", False)),
                "decision_correct_exact": bool(score.get("decision_correct_exact", False)),
                "decision_correct_family": bool(score.get("decision_correct_family", False)),
                "conditions_correct": bool(score.get("conditions_correct", False)),
                "conditions_correct_exact": bool(score.get("conditions_correct_exact", False)),
                "conditions_correct_subset": bool(score.get("conditions_correct_subset", False)),
                "e3_answer_correct": bool(score.get("answer_correct", False)),
                "e3_evidence_valid_in_context": bool(score.get("evidence_valid_in_context", False)),
                "critical_coverage_e3": score.get("critical_coverage"),
                "inapplicable_injected_rate_e3": score.get("inapplicable_injected_rate"),
                "stale_present": bool(score.get("stale_present", False)),
                "stale_count": int(score.get("stale_count", 0) or 0),
                "goc_avoid_target_clause_ids": list(_unique_strs(invalidated_ids)),
                "goc_avoided_node_injected": bool(score.get("avoided_injected", False)),
                "goc_applicability_seed_ids": list(debug.get("goc_applicability_seed_ids") or []),
                "goc_applicability_seed_used": len(list(debug.get("goc_applicability_seed_ids") or [])),
                "goc_applicability_seed_applicable_rate": debug.get(
                    "goc_applicability_seed_applicable_rate", float("nan")
                ),
                "goc_dependency_closure_added_ids": list(debug.get("goc_dependency_closure_added_ids") or []),
                "goc_dependency_closure_added_used": len(list(debug.get("goc_dependency_closure_added_ids") or [])),
                "goc_dependency_closure_added_applicable_rate": debug.get(
                    "goc_dependency_closure_added_applicable_rate", float("nan")
                ),
                "goc_dependency_closure_universe_effective": list(
                    debug.get("goc_dependency_closure_universe_effective") or []
                ),
                "goc_dependency_closure_universe_effective_size": int(
                    debug.get("goc_dependency_closure_universe_effective_size", 0) or 0
                ),
                "goc_required_force_included_ids": list(
                    debug.get("goc_required_force_included_ids") or []
                ),
                "goc_required_force_included_but_avoided_ids": list(
                    debug.get("goc_required_force_included_but_avoided_ids") or []
                ),
                "goc_required_force_included_but_inapplicable_ids": list(
                    debug.get("goc_required_force_included_but_inapplicable_ids") or []
                ),
                "e3_context_clause_ids": list(context_ids),
                "prompt_tokens_est": int(score.get("token_est", 0) or 0),
                "prompt_tokens": int(score.get("token_est", 0) or 0),
                "prompt_tokens_actual": prompt_tokens_actual,
                "completion_tokens_actual": completion_tokens_actual,
                "total_tokens_actual": total_tokens_actual,
                "llm_output_text": str(llm_output_text or "")[:2000],
                "llm_parse_error": bool(llm_parse_error),
                "revive_success": bool(score.get("revive_success", False)),
            }
            records.append(rec)
            pivot_step_records.append(rec)

        if not pivot_step_records:
            continue

        strict_thread = all(
            bool(r.get("decision_correct"))
            and bool(r.get("e3_evidence_valid_in_context"))
            and (not bool(r.get("goc_avoided_node_injected")))
            for r in pivot_step_records
        )
        last = pivot_step_records[-1]
        e3_only = bool(last.get("e3_answer_correct")) and bool(last.get("e3_evidence_valid_in_context"))
        thread_records.append(
            {
                "thread_id": thread.thread_id,
                "thread_strict_correct": bool(strict_thread),
                "thread_e3_only_correct": bool(e3_only),
                "thread_judge_correct": bool(strict_thread),
                "pivot_checks": int(len(pivot_step_records)),
                "pivot_decision_correct_count": int(sum(1 for r in pivot_step_records if r.get("decision_correct"))),
                "pivot_token_total_est": int(
                    sum(
                        int(r.get("prompt_tokens_est", r.get("prompt_tokens", 0)) or 0)
                        for r in pivot_step_records
                    )
                ),
                "pivot_token_total_actual": (
                    int(sum(int(r.get("total_tokens_actual", 0) or 0) for r in pivot_step_records))
                    if pivot_step_records
                    and all(isinstance(r.get("total_tokens_actual"), (int, float)) for r in pivot_step_records)
                    else float("nan")
                ),
            }
        )

    pivot_records = list(records)
    decision_vals = [1.0 if r.get("decision_correct") else 0.0 for r in pivot_records]
    e3_only_vals = [
        1.0 if (r.get("e3_answer_correct") and r.get("e3_evidence_valid_in_context")) else 0.0
        for r in pivot_records
    ]
    strict_vals = [1.0 if tr.get("thread_strict_correct") else 0.0 for tr in thread_records]
    pivot_tokens_est = [float(r.get("prompt_tokens_est", r.get("prompt_tokens", 0)) or 0) for r in pivot_records]
    pivot_tokens_actual = [
        float(r.get("total_tokens_actual"))
        for r in pivot_records
        if isinstance(r.get("total_tokens_actual"), (int, float))
    ]
    total_tokens_by_thread_est = [float(tr.get("pivot_token_total_est", 0) or 0) for tr in thread_records]
    total_tokens_by_thread_actual = [
        float(tr.get("pivot_token_total_actual"))
        for tr in thread_records
        if isinstance(tr.get("pivot_token_total_actual"), (int, float))
    ]
    avoid_counts = [len(list(r.get("goc_avoid_target_clause_ids") or [])) for r in pivot_records]
    avoided_injected_vals = [1.0 if r.get("goc_avoided_node_injected") else 0.0 for r in pivot_records]
    revive_vals = [1.0 if r.get("revive_success") else 0.0 for r in pivot_records]

    tokens_pivot_mean_est = _mean(pivot_tokens_est)
    tokens_total_mean_est = _mean(total_tokens_by_thread_est)
    tokens_pivot_mean_actual = _mean(pivot_tokens_actual)
    tokens_total_mean_actual = _mean(total_tokens_by_thread_actual)
    tokens_pivot_mean = tokens_pivot_mean_actual if math.isfinite(tokens_pivot_mean_actual) else tokens_pivot_mean_est
    tokens_total_mean = tokens_total_mean_actual if math.isfinite(tokens_total_mean_actual) else tokens_total_mean_est

    metrics = {
        "decision_accuracy": _mean(decision_vals),
        "judge_accuracy": _mean(e3_only_vals),
        "pivot_decision_accuracy": _mean(decision_vals),
        "pivot_e3_only_accuracy": _mean(e3_only_vals),
        "strict_pivot_accuracy": _mean(strict_vals),
        "tokens_pivot_mean": tokens_pivot_mean,
        "tokens_total_mean": tokens_total_mean,
        "tokens_pivot_mean_est": tokens_pivot_mean_est,
        "tokens_total_mean_est": tokens_total_mean_est,
        "tokens_pivot_mean_actual": tokens_pivot_mean_actual,
        "tokens_total_mean_actual": tokens_total_mean_actual,
        "mean_avoid_targets_per_pivot": _mean([float(v) for v in avoid_counts]),
        "avoided_injected_rate": _mean(avoided_injected_vals),
        "revive_success_rate": _mean(revive_vals),
        "pivot_records": int(len(pivot_records)),
        "thread_records": int(len(thread_records)),
        "traceops_eval_mode": eval_mode,
        "llm_pivots_evaluated": int(llm_pivots_seen) if use_llm else 0,
    }

    return {
        "method": method,
        "metrics": metrics,
        "records": records,
        "thread_records": thread_records,
    }


__all__ = [
    "_build_traceops_llm_prompt",
    "_parse_llm_json",
    "evaluate_traceops_method",
]
