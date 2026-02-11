from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Set, Tuple

from .schema import TraceGold, TraceStep, TraceThread, TraceWorldClause, normalize_decision

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


def _normalize_exception_text(text: str) -> str:
    raw = str(text or "").lower()
    raw = re.sub(r"\b[a-z]{1,4}\d{2,}\b", " ", raw)
    raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _hash_fraction(text: str) -> float:
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    # 64-bit mantissa-style stable fraction in [0,1).
    intval = int(digest[:16], 16)
    return float(intval) / float(16 ** 16)


def _build_exception_equiv_map(
    clauses: Dict[str, TraceWorldClause],
    allowed_clause_ids: Sequence[str],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cid in _unique_strs(allowed_clause_ids):
        clause = clauses.get(cid)
        if clause is None:
            continue
        if str(clause.node_type or "") != "EXCEPTION":
            continue
        out[str(cid)] = _normalize_exception_text(str(clause.text or ""))
    return out


def _canonicalize_condition_tag(tag: str, ex_equiv_map: Dict[str, str]) -> str:
    raw = str(tag or "").strip()
    if not raw.lower().startswith("exception="):
        return raw
    _, _, cid = raw.partition("=")
    cid = str(cid).strip()
    ex_key = str(ex_equiv_map.get(cid, "") or "")
    if ex_key:
        return f"exception_key={ex_key}"
    return raw


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

    schema_line = (
        'JSON schema: {"decision":"allow|deny|require_condition|needs_more_info",'
        '"conditions":["<condition_string>"],"evidence":["<clause_id>"]}'
    )
    lines: List[str] = []
    lines.append("Use ONLY the provided CONTEXT. Output STRICT JSON only.")
    lines.append(schema_line)
    lines.append("You MUST choose one of the 4 decision labels exactly as written.")
    lines.append(
        "For decision=allow|deny|require_condition: `evidence` MUST contain at least 2 DISTINCT clause ids."
    )
    lines.append(
        "Every evidence id MUST be selected from Allowed evidence clause IDs."
    )
    lines.append(
        "If the above evidence requirement is not satisfied, decision MUST be `needs_more_info`, "
        "conditions MUST be [], and evidence should contain at most one most relevant allowed clause id (or [])."
    )
    lines.append(
        "For conditions: choose strings ONLY from ALLOWED_CONDITIONS (copy exact); no free-form text."
    )
    lines.append(
        "Output `needs_more_info` if: (a) decisive evidence is missing, "
        "(b) UPDATE/EXCEPTION/DECISION conflicts are unresolved, or "
        "(c) required condition tags cannot be copied exactly from ALLOWED_CONDITIONS."
    )
    lines.append(
        "If you include an EXCEPTION clause id in `evidence`, you MUST include "
        "`exception=<that_clause_id>` in `conditions`."
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
    if (
        str(getattr(clause, "node_type", "") or "") == "EXCEPTION"
        and (not key or val is None)
    ):
        text = str(getattr(clause, "text", "") or "").lower()
        m = re.search(r"if\s+(\w+)\s+is\s+(\w+)", text)
        if m:
            cond_key = str(m.group(1)).strip()
            cond_val = str(m.group(2)).strip()
            state_val = state.get(cond_key) if isinstance(state, dict) else None
            return str(state_val).strip().lower() == cond_val.lower()
        if "residency mismatch" in text:
            if isinstance(state, dict) and "residency" in state and "region" in state:
                return str(state.get("residency")).strip().lower() != str(state.get("region")).strip().lower()
        return True
    if not key or val is None:
        return True
    return str(state.get(key)) == str(val)


def _gold_decision_family(gold_decision: str) -> str:
    return normalize_decision(gold_decision)


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

        def _extract_update_key(clause: TraceWorldClause | None) -> str:
            if clause is None or str(getattr(clause, "node_type", "") or "") != "UPDATE":
                return ""
            key = str(getattr(clause, "state_key", "") or "").strip().lower()
            if key:
                return key
            text = str(getattr(clause, "text", "") or "").strip().lower()
            m = re.search(r"update:\s*([a-z_][a-z0-9_]*)\s+changed\s+to", text)
            if m:
                return str(m.group(1)).strip().lower()
            return ""

        state_keys = {str(k).strip().lower() for k in (step.state or {}).keys()}
        state_vals = {
            str(v).strip().lower()
            for v in (step.state or {}).values()
            if str(v).strip()
        }

        def _extract_trigger_keys(clause: TraceWorldClause | None) -> List[str]:
            if clause is None:
                return []
            keys: List[str] = []
            skey = str(getattr(clause, "state_key", "") or "").strip().lower()
            if skey:
                keys.append(skey)
            text = str(getattr(clause, "text", "") or "").lower()
            for m in re.finditer(r"if\s+([a-z_][a-z0-9_]*)\s+is\s+([a-z0-9_\-]+)", text):
                keys.append(str(m.group(1)).strip().lower())
            for k in state_keys:
                if k and re.search(rf"\b{re.escape(k)}\b", text):
                    keys.append(k)
            return _unique_strs(keys)

        anchor_ids: List[str] = []
        for anchor_type in ("DECISION", "UPDATE"):
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
                if clause.node_type == "UPDATE":
                    score += 0.3
                if clause.node_type == "EXCEPTION":
                    score += 0.05
                    if str(getattr(clause, "metadata", {}).get("salience", "") or "") == "low":
                        score -= 0.5
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
                if _clause_applicable(clause, step.state):
                    closure_added.append(cid)
                if len(closure_added) >= topk:
                    break
        context_ids = _unique_strs(anchor_ids + seed_ids + closure_added)

        depwalk_added: List[str] = []
        depwalk_enable = bool(getattr(args, "goc_depwalk_enable", False))
        depwalk_hops = max(0, int(getattr(args, "goc_depwalk_hops", 2) or 2))
        depwalk_topk_per_hop = max(
            0,
            int(getattr(args, "goc_depwalk_topk_per_hop", 6) or 6),
        )
        if depwalk_enable and depwalk_hops > 0 and depwalk_topk_per_hop > 0:
            depwalk_history = [
                cid
                for cid in context_ordered_history
                if cid in clauses and int(clauses[cid].step_idx) < int(step.step_idx)
            ]
            depwalk_history = [cid for cid in depwalk_history if (not use_avoids or cid not in avoid_set)]
            depwalk_set = set(depwalk_history)
            history_rank = {cid: idx for idx, cid in enumerate(depwalk_history)}
            neighbors: Dict[str, Set[str]] = {cid: set() for cid in depwalk_history}

            def _connect(aid: str, bid: str) -> None:
                if aid == bid or aid not in depwalk_set or bid not in depwalk_set:
                    return
                neighbors.setdefault(aid, set()).add(bid)
                neighbors.setdefault(bid, set()).add(aid)

            for cid in depwalk_history:
                clause = clauses.get(cid)
                if clause is None:
                    continue
                for dep in clause.depends_on:
                    did = str(dep)
                    if did in depwalk_set:
                        _connect(cid, did)

            last_update_by_key: Dict[str, str] = {}
            for cid in depwalk_history:
                clause = clauses.get(cid)
                key = _extract_update_key(clause)
                if not key:
                    continue
                prev = last_update_by_key.get(key)
                if prev:
                    _connect(cid, prev)
                last_update_by_key[key] = cid

            for idx, cid in enumerate(depwalk_history):
                clause = clauses.get(cid)
                if clause is None:
                    continue
                ctype = str(clause.node_type or "")
                if ctype == "EXCEPTION":
                    keys = _extract_trigger_keys(clause)
                    if not keys:
                        continue
                    priors = depwalk_history[:idx]
                    for key in keys:
                        linked = 0
                        for pid in reversed(priors):
                            pclause = clauses.get(pid)
                            if pclause is None or str(pclause.node_type or "") not in {"UPDATE", "EVIDENCE"}:
                                continue
                            pkey = str(getattr(pclause, "state_key", "") or "").strip().lower()
                            ptext = str(getattr(pclause, "text", "") or "").lower()
                            if pkey == key or re.search(rf"\b{re.escape(key)}\b", ptext):
                                _connect(cid, pid)
                                linked += 1
                            if linked >= 2:
                                break
                elif ctype == "DECISION":
                    dtoks = _tokenize(clause.text)
                    priors = depwalk_history[:idx]
                    linked = 0
                    for pid in reversed(priors):
                        pclause = clauses.get(pid)
                        if pclause is None or str(pclause.node_type or "") not in {"UPDATE", "EVIDENCE"}:
                            continue
                        ptoks = _tokenize(pclause.text)
                        pkey = str(getattr(pclause, "state_key", "") or "").strip().lower()
                        pval = str(getattr(pclause, "state_value", "") or "").strip().lower()
                        key_overlap = bool(pkey and pkey in dtoks)
                        val_overlap = bool(pval and pval in dtoks)
                        state_token_overlap = bool((dtoks & state_keys) and (ptoks & state_keys))
                        state_val_overlap = bool((dtoks & state_vals) and (ptoks & state_vals))
                        if key_overlap or val_overlap or state_token_overlap or state_val_overlap:
                            _connect(cid, pid)
                            linked += 1
                        if linked >= 3:
                            break

            selected_set = set(_unique_strs(context_ids))
            frontier = [cid for cid in _unique_strs(anchor_ids + seed_ids) if cid in depwalk_set]
            if not frontier:
                frontier = [cid for cid in _unique_strs(context_ids) if cid in depwalk_set]
            selected_exception_texts = {
                _normalize_exception_text(str(clauses[cid].text or ""))
                for cid in selected_set
                if cid in clauses and str(clauses[cid].node_type or "") == "EXCEPTION"
            }

            for _ in range(depwalk_hops):
                if not frontier:
                    break
                frontier_set = set(frontier)
                cand_ids: List[str] = []
                for fid in frontier:
                    for nid in neighbors.get(fid, set()):
                        if nid in frontier_set or nid in selected_set:
                            continue
                        if nid in avoid_set:
                            continue
                        cand_ids.append(nid)
                cand_ids = _unique_strs(cand_ids)
                scored_neighbors: List[Tuple[int, int, int, str]] = []
                for nid in cand_ids:
                    clause = clauses.get(nid)
                    if clause is None:
                        continue
                    if not _clause_applicable(clause, step.state):
                        continue
                    novelty_ok = 1
                    if str(clause.node_type or "") == "EXCEPTION":
                        norm_text = _normalize_exception_text(str(clause.text or ""))
                        if norm_text and norm_text in selected_exception_texts:
                            novelty_ok = 0
                    if novelty_ok == 0:
                        continue
                    scored_neighbors.append(
                        (
                            novelty_ok,
                            int(clause.step_idx),
                            -int(history_rank.get(nid, 10**9)),
                            nid,
                        )
                    )
                scored_neighbors.sort(reverse=True)
                hop_added: List[str] = []
                for _, _, _, nid in scored_neighbors:
                    if len(hop_added) >= depwalk_topk_per_hop:
                        break
                    clause = clauses.get(nid)
                    if clause is None:
                        continue
                    hop_added.append(nid)
                    selected_set.add(nid)
                    depwalk_added.append(nid)
                    if str(clause.node_type or "") == "EXCEPTION":
                        norm_text = _normalize_exception_text(str(clause.text or ""))
                        if norm_text:
                            selected_exception_texts.add(norm_text)
                frontier = hop_added

            context_ids = _unique_strs(context_ids + depwalk_added)

        update_key_cap = max(
            1,
            int(getattr(args, "goc_exception_rescue_recent_update_keys", 4) or 4),
        )
        update_key_trace: List[str] = []
        for cid in context_ordered_history:
            clause = clauses.get(cid)
            key = _extract_update_key(clause)
            if key:
                update_key_trace.append(key)
        recent_update_keys = set(_unique_strs(update_key_trace[-update_key_cap:]))

        exception_rescue_ids: List[str] = []
        exception_rescue_reasons: List[str] = []
        update_history_rescue_ids: List[str] = []
        rescue_ran = False
        rescue_reason_short = "not_needed"

        history_exception_ids = [
            cid
            for cid in context_ordered_history
            if cid in clauses
            and str(clauses[cid].node_type or "") == "EXCEPTION"
            and (not use_avoids or cid not in avoid_set)
        ]
        included_exception_ids = [
            cid
            for cid in context_ids
            if cid in clauses and str(clauses[cid].node_type or "") == "EXCEPTION"
        ]
        proxy_poor = bool(len(depwalk_added) == 0 and len(seed_ids) == 0 and len(closure_added) == 0)
        active_exceptions_missing = bool(history_exception_ids and not included_exception_ids)
        if active_exceptions_missing:
            rescue_ran = True
            rescue_reason_short = "active_exception_missing"
        elif proxy_poor:
            rescue_ran = True
            rescue_reason_short = "weak_depwalk"

        raw_exception_rescue_topk = getattr(args, "goc_exception_rescue_topk", None)
        exception_rescue_topk = (
            2 if raw_exception_rescue_topk is None else max(0, int(raw_exception_rescue_topk))
        )
        if rescue_ran and exception_rescue_topk > 0:
            mismatch = False
            if isinstance(step.state, dict) and "residency" in step.state and "region" in step.state:
                mismatch = (
                    str(step.state.get("residency", "")).strip().lower()
                    != str(step.state.get("region", "")).strip().lower()
                )

            def _primary_reason(reasons: Sequence[str]) -> str:
                ranked = [
                    "activation",
                    "latent_mismatch",
                    "update_key_match",
                    "state_text_match",
                    "applicable_hint",
                ]
                for label in ranked:
                    if label in reasons:
                        return label
                return str(reasons[0]) if reasons else ""

            scored_exceptions: List[Dict[str, Any]] = []
            context_set = set(context_ids)
            for idx, cid in enumerate(context_ordered_history):
                if cid in context_set:
                    continue
                if cid in avoid_set:
                    continue
                clause = clauses.get(cid)
                if clause is None or str(clause.node_type or "") != "EXCEPTION":
                    continue
                text = str(clause.text or "").lower()
                tags = {str(tag).strip().lower() for tag in (clause.tags or [])}
                reasons: List[str] = []
                score = 0.0

                if "activation" in tags:
                    score += 2.0
                    reasons.append("activation")
                if "latent" in tags and mismatch:
                    score += 1.8
                    reasons.append("latent_mismatch")

                key_matched = False
                for key in recent_update_keys:
                    if key and re.search(rf"\b{re.escape(key)}\b", text):
                        key_matched = True
                        break
                if key_matched:
                    score += 1.0
                    reasons.append("update_key_match")

                text_hint = False
                for key in [
                    "deadline",
                    "residency",
                    "mismatch",
                    "region",
                    "budget",
                    "retention",
                    "tier",
                ]:
                    if key in text and isinstance(step.state, dict) and key in step.state:
                        text_hint = True
                        break
                if text_hint:
                    score += 0.2
                    reasons.append("state_text_match")

                if _clause_applicable(clause, step.state):
                    score += 0.2
                    reasons.append("applicable_hint")

                if (
                    str(getattr(clause, "metadata", {}).get("salience", "") or "").strip().lower()
                    == "low"
                ):
                    score -= 0.25

                if not reasons:
                    continue
                scored_exceptions.append(
                    {
                        "cid": str(cid),
                        "score": float(score),
                        "idx": int(idx),
                        "reasons": list(_unique_strs(reasons)),
                    }
                )

            scored_exceptions.sort(
                key=lambda item: (-float(item["score"]), -int(item["idx"]), str(item["cid"]))
            )
            activation_cap = min(2, exception_rescue_topk)
            selected_exception_ids: List[str] = []
            selected_exception_reasons: List[str] = []

            for item in scored_exceptions:
                if len(selected_exception_ids) >= activation_cap:
                    break
                reasons = set(item.get("reasons") or [])
                if "activation" not in reasons:
                    continue
                cid = str(item.get("cid"))
                if cid in selected_exception_ids:
                    continue
                selected_exception_ids.append(cid)
                selected_exception_reasons.append(_primary_reason(list(reasons)))

            for item in scored_exceptions:
                if len(selected_exception_ids) >= exception_rescue_topk:
                    break
                cid = str(item.get("cid"))
                if cid in selected_exception_ids:
                    continue
                reasons = list(item.get("reasons") or [])
                selected_exception_ids.append(cid)
                selected_exception_reasons.append(_primary_reason(reasons))

            exception_rescue_ids = list(selected_exception_ids)
            exception_rescue_reasons = list(selected_exception_reasons)
            context_ids = _unique_strs(context_ids + exception_rescue_ids)

        raw_update_history_rescue_topk = getattr(args, "goc_update_history_rescue_topk", None)
        update_history_rescue_topk = (
            4 if raw_update_history_rescue_topk is None else max(0, int(raw_update_history_rescue_topk))
        )
        if rescue_ran and update_history_rescue_topk > 0:
            updates_by_key: Dict[str, List[str]] = {}
            latest_rank: Dict[str, int] = {}
            for idx, cid in enumerate(context_ordered_history):
                clause = clauses.get(cid)
                key = _extract_update_key(clause)
                if not key:
                    continue
                updates_by_key.setdefault(key, []).append(str(cid))
                latest_rank[key] = int(idx)

            context_set = set(context_ids)
            ranked_keys = sorted(
                [key for key, values in updates_by_key.items() if len(values) >= 2],
                key=lambda key: (-int(latest_rank.get(key, -1)), str(key)),
            )
            for key in ranked_keys:
                candidate_ids = list(updates_by_key.get(key, []))[-2:]
                for cid in candidate_ids:
                    if len(update_history_rescue_ids) >= update_history_rescue_topk:
                        break
                    if cid in context_set:
                        continue
                    if cid in avoid_set:
                        continue
                    update_history_rescue_ids.append(cid)
                    context_set.add(cid)
                if len(update_history_rescue_ids) >= update_history_rescue_topk:
                    break
            context_ids = _unique_strs(context_ids + update_history_rescue_ids)

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

        def _goc_sort_key(cid: str) -> Tuple[int, int, str]:
            clause = clauses.get(cid)
            if clause is None:
                return (1, 10**9, str(cid))
            is_exception = 1 if str(clause.node_type or "") == "EXCEPTION" else 0
            return (is_exception, int(clause.step_idx), str(cid))

        context_ids = sorted(_unique_strs(context_ids), key=_goc_sort_key)

        debug = {
            "goc_anchor_ids": list(anchor_ids),
            "goc_applicability_seed_ids": list(seed_ids),
            "goc_applicability_seed_applicable_rate": _applicable_rate(seed_ids),
            "goc_depwalk_added_ids": list(depwalk_added),
            "goc_depwalk_added_count": int(len(depwalk_added)),
            "goc_depwalk_added_applicable_rate": _applicable_rate(depwalk_added),
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
            "goc_exception_rescue_ids": list(exception_rescue_ids),
            "goc_exception_rescue_reason": list(exception_rescue_reasons),
            "goc_rescue_ran": bool(rescue_ran),
            "goc_rescue_reason_short": str(rescue_reason_short),
            "goc_update_history_rescue_ids": list(update_history_rescue_ids),
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

    context_list = _unique_strs(context_ids)
    context_set = set(context_list)
    required_ids = _unique_strs(step.pivot_required_ids or gold.evidence_core_ids or gold.evidence_ids)
    required_set = set(required_ids)

    pred_decision_raw = str(prediction.get("decision", ""))
    pred_decision = normalize_decision(pred_decision_raw)
    pred_conditions = _unique_strs(prediction.get("conditions") or [])
    pred_evidence = _unique_strs(prediction.get("evidence") or [])

    def _extract_exception_ids(tags: Sequence[str]) -> List[str]:
        out: List[str] = []
        for tag in tags:
            raw = str(tag or "").strip()
            if not raw.lower().startswith("exception="):
                continue
            _, _, ex_id = raw.partition("=")
            ex_id = str(ex_id).strip()
            if ex_id:
                out.append(ex_id)
        return _unique_strs(out)

    gold_conditions = _unique_strs(gold.conditions)
    history_clause_ids = [
        cid
        for cid, clause in thread.clauses.items()
        if int(clause.step_idx) < int(step.step_idx)
    ]
    ex_map_inputs = _unique_strs(
        list(history_clause_ids)
        + list(context_list)
        + list(required_ids)
        + list(pred_evidence)
        + _extract_exception_ids(gold_conditions)
        + _extract_exception_ids(pred_conditions)
    )
    ex_equiv_map = _build_exception_equiv_map(thread.clauses, ex_map_inputs)
    context_exception_keys = {
        str(ex_equiv_map.get(cid, "") or "")
        for cid in context_list
        if cid in ex_equiv_map and str(ex_equiv_map.get(cid, "") or "")
    }

    gold_decision_raw = str(gold.decision)
    gold_decision = normalize_decision(gold_decision_raw)
    gold_decision_family = _gold_decision_family(gold_decision_raw)
    decision_correct_strict_raw = bool(pred_decision_raw == gold_decision_raw)
    decision_correct_exact = bool(decision_correct_strict_raw)
    decision_correct_family = bool(pred_decision == gold_decision_family)
    decision_correct = bool(pred_decision == gold_decision)
    conditions_correct_exact = bool(set(pred_conditions) == set(gold_conditions))
    conditions_correct_subset = bool(set(gold_conditions).issubset(set(pred_conditions)))
    pred_conditions_equiv = _unique_strs(
        [_canonicalize_condition_tag(tag, ex_equiv_map) for tag in pred_conditions]
    )
    gold_conditions_equiv = _unique_strs(
        [_canonicalize_condition_tag(tag, ex_equiv_map) for tag in gold_conditions]
    )
    conditions_correct_exact_equiv = bool(
        set(pred_conditions_equiv) == set(gold_conditions_equiv)
    )
    conditions_correct_subset_equiv = bool(
        set(gold_conditions_equiv).issubset(set(pred_conditions_equiv))
    )
    if str(eval_mode) == "llm":
        conditions_correct = conditions_correct_subset_equiv
    else:
        conditions_correct = conditions_correct_exact
    answer_correct = bool(decision_correct and conditions_correct)
    evidence_valid_in_context = bool(all(cid in context_set for cid in pred_evidence))

    required_covered_strict: Set[str] = set()
    required_covered_equiv: Set[str] = set()
    evidence_core_missing_ids_strict: List[str] = []
    evidence_core_missing_equiv_keys: List[str] = []
    id_mismatch_but_equiv = False
    for cid in required_ids:
        clause = thread.clauses.get(cid)
        if cid in context_set:
            required_covered_strict.add(cid)
            required_covered_equiv.add(cid)
            continue
        evidence_core_missing_ids_strict.append(cid)
        if clause is None or str(clause.node_type or "") != "EXCEPTION":
            continue
        ex_key = str(ex_equiv_map.get(cid, "") or "")
        if ex_key and ex_key in context_exception_keys:
            required_covered_equiv.add(cid)
            id_mismatch_but_equiv = True
        elif ex_key:
            evidence_core_missing_equiv_keys.append(ex_key)

    evidence_core_covered_strict = bool(len(required_covered_strict) == len(required_ids))
    evidence_core_covered_equiv = bool(len(required_covered_equiv) == len(required_ids))

    if required_set:
        critical_coverage_strict = float(len(required_covered_strict)) / float(len(required_ids))
        critical_coverage_equiv = float(len(required_covered_equiv)) / float(len(required_ids))
    else:
        critical_coverage_strict = float("nan")
        critical_coverage_equiv = float("nan")
    critical_coverage = (
        critical_coverage_equiv if str(eval_mode) == "llm" else critical_coverage_strict
    )
    revive_success_strict = evidence_core_covered_strict if required_ids else True
    revive_success_equiv = evidence_core_covered_equiv if required_ids else True
    revive_success = revive_success_equiv if str(eval_mode) == "llm" else revive_success_strict

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
        "gold_decision": gold_decision,
        "gold_decision_raw": gold_decision_raw,
        "gold_decision_family": gold_decision_family,
        "decision_correct_strict_raw": decision_correct_strict_raw,
        "decision_correct_exact": decision_correct_exact,
        "decision_correct_family": decision_correct_family,
        "conditions_correct": conditions_correct,
        "conditions_correct_exact": conditions_correct_exact,
        "conditions_correct_subset": conditions_correct_subset,
        "conditions_correct_exact_equiv": conditions_correct_exact_equiv,
        "conditions_correct_subset_equiv": conditions_correct_subset_equiv,
        "answer_correct": answer_correct,
        "evidence_valid_in_context": evidence_valid_in_context,
        "critical_coverage": critical_coverage,
        "critical_coverage_strict": critical_coverage_strict,
        "critical_coverage_equiv": critical_coverage_equiv,
        "evidence_core_covered_strict": evidence_core_covered_strict,
        "evidence_core_covered_equiv": evidence_core_covered_equiv,
        "evidence_core_missing_ids_strict": list(_unique_strs(evidence_core_missing_ids_strict)),
        "evidence_core_missing_equiv_keys": list(_unique_strs(evidence_core_missing_equiv_keys)),
        "evidence_core_id_mismatch_but_equiv_present": bool(id_mismatch_but_equiv),
        "revive_success": revive_success,
        "revive_success_strict": revive_success_strict,
        "revive_success_equiv": revive_success_equiv,
        "avoided_injected": avoided_injected,
        "stale_present": bool(stale_set),
        "stale_count": int(len(stale_set)),
        "inapplicable_injected_rate": inapplicable_rate,
        "pred_decision": pred_decision,
        "pred_decision_raw": pred_decision_raw,
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
    llm_eval_scope = str(getattr(args, "traceops_llm_eval_scope", "pivots") or "pivots").strip().lower()
    if llm_eval_scope not in {"pivots", "all", "sample"}:
        llm_eval_scope = "pivots"
    llm_sample_rate = float(getattr(args, "traceops_llm_sample_rate", 0.2) or 0.2)
    llm_sample_rate = max(0.0, min(1.0, llm_sample_rate))
    llm_seed = int(getattr(args, "traceops_llm_seed", 0) or 0)
    llm_pivots_seen = 0
    llm_steps_seen = 0
    pivots_available_total = 0
    steps_available_total = 0
    sampled_steps_evaluated = 0

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
    for thread in threads:
        history_ids: List[str] = []
        invalidated_ids: List[str] = []
        pivot_step_records: List[Dict[str, Any]] = []

        for step in thread.steps:
            if max_steps > 0 and int(step.step_idx) >= max_steps:
                break
            steps_available_total += 1
            history_ids = _unique_strs(history_ids + list(step.introduced_clause_ids))
            if step.kind == "update" and step.avoid_target_ids:
                invalidated_ids = _unique_strs(invalidated_ids + list(step.avoid_target_ids))

            is_scored_pivot = bool(step.kind == "pivot_check" and step.gold is not None)
            if is_scored_pivot:
                pivots_available_total += 1
            sampled_step = False
            if use_llm:
                if llm_eval_scope == "all":
                    sampled_step = True
                elif llm_eval_scope == "pivots":
                    sampled_step = is_scored_pivot
                else:
                    sample_key = f"{llm_seed}:{thread.thread_id}:{step.step_id}:{step.step_idx}"
                    sampled_step = _hash_fraction(sample_key) < llm_sample_rate

                if sampled_step and llm_max_pivots > 0 and llm_steps_seen >= llm_max_pivots:
                    sampled_step = False

            if use_llm and not sampled_step:
                continue
            if not is_scored_pivot and not sampled_step:
                continue

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
            pred: Dict[str, Any] | None = None
            if use_llm and sampled_step:
                sampled_steps_evaluated += 1
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
                llm_steps_seen += 1
                if is_scored_pivot:
                    llm_pivots_seen += 1
            elif is_scored_pivot and not use_llm:
                pred = _infer_prediction(
                    step=step,
                    context_ids=context_ids,
                    invalidated_ids=invalidated_ids,
                )

            if not is_scored_pivot:
                continue
            if pred is None:
                continue

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

            exception_injected_ids: List[str] = []
            exception_applicable_count = 0
            for cid in list(context_ids):
                clause = thread.clauses.get(cid)
                if clause is None:
                    continue
                if str(clause.node_type or "") != "EXCEPTION":
                    continue
                exception_injected_ids.append(str(cid))
                if _clause_applicable(clause, step.state):
                    exception_applicable_count += 1
            step_meta = dict(step.metadata or {})
            hidden_core_ids = _unique_strs(step_meta.get("hidden_core_ids") or [])
            hidden_core_parent_ids = _unique_strs(step_meta.get("hidden_core_parent_ids") or [])
            trap_distractor_ids = _unique_strs(step_meta.get("trap_distractor_ids") or [])
            trap_distractor_set = {str(cid) for cid in trap_distractor_ids if str(cid).strip()}
            trap_injected_ids = [
                str(cid) for cid in list(context_ids) if str(cid) in trap_distractor_set
            ]
            trap_injected_count = int(len(trap_injected_ids))
            trap_injected_rate = (
                float(trap_injected_count) / float(len(trap_distractor_set))
                if trap_distractor_set
                else float("nan")
            )
            decision_checkpoint_trap_ids = _unique_strs(
                step_meta.get("decision_checkpoint_trap_ids") or []
            )
            decision_checkpoint_trap_set = {
                str(cid) for cid in decision_checkpoint_trap_ids if str(cid).strip()
            }
            decision_checkpoint_trap_excludable_ids = _unique_strs(
                step_meta.get("decision_checkpoint_trap_excludable_ids") or []
            )
            decision_checkpoint_trap_excludable_set = {
                str(cid)
                for cid in decision_checkpoint_trap_excludable_ids
                if str(cid).strip()
            }
            decision_checkpoint_trap_injected_ids = [
                str(cid) for cid in list(context_ids) if str(cid) in decision_checkpoint_trap_set
            ]
            decision_checkpoint_trap_injected_count = int(
                len(decision_checkpoint_trap_injected_ids)
            )
            decision_checkpoint_trap_injected_rate = (
                float(decision_checkpoint_trap_injected_count)
                / float(len(decision_checkpoint_trap_set))
                if decision_checkpoint_trap_set
                else float("nan")
            )
            raw_flip_count = step_meta.get("core_necessity_flip_count")
            core_necessity_flip_count = (
                int(raw_flip_count)
                if isinstance(raw_flip_count, (int, float))
                else float("nan")
            )

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
                "traceops_llm_eval_scope": llm_eval_scope,
                "sampled_step": bool(sampled_step),
                "gold_decision_raw": str(score.get("gold_decision_raw", str(step.gold.decision))),
                "gold_decision": str(score.get("gold_decision", normalize_decision(step.gold.decision))),
                "gold_decision_family": str(score.get("gold_decision_family", "")),
                "gold_conditions": list(step.gold.conditions),
                "gold_evidence_ids": list(step.gold.evidence_ids),
                "pivot_required_clause_ids": list(step.pivot_required_ids),
                "pred_decision_raw": str(score.get("pred_decision_raw", pred.get("decision", ""))),
                "pred_decision": str(score.get("pred_decision", normalize_decision(pred.get("decision", "")))),
                "pred_conditions": score.get("pred_conditions"),
                "pred_evidence": score.get("pred_evidence"),
                "decision_correct": bool(score.get("decision_correct", False)),
                "decision_correct_strict_raw": bool(score.get("decision_correct_strict_raw", False)),
                "decision_correct_exact": bool(score.get("decision_correct_exact", False)),
                "decision_correct_family": bool(score.get("decision_correct_family", False)),
                "conditions_correct": bool(score.get("conditions_correct", False)),
                "conditions_correct_exact": bool(score.get("conditions_correct_exact", False)),
                "conditions_correct_subset": bool(score.get("conditions_correct_subset", False)),
                "conditions_correct_exact_equiv": bool(
                    score.get("conditions_correct_exact_equiv", False)
                ),
                "conditions_correct_subset_equiv": bool(
                    score.get("conditions_correct_subset_equiv", False)
                ),
                "e3_answer_correct": bool(score.get("answer_correct", False)),
                "e3_evidence_valid_in_context": bool(score.get("evidence_valid_in_context", False)),
                "critical_coverage_e3": score.get("critical_coverage"),
                "critical_coverage_strict_e3": score.get("critical_coverage_strict"),
                "critical_coverage_equiv_e3": score.get("critical_coverage_equiv"),
                "evidence_core_covered_strict": bool(
                    score.get("evidence_core_covered_strict", False)
                ),
                "evidence_core_covered_equiv": bool(
                    score.get("evidence_core_covered_equiv", False)
                ),
                "evidence_core_missing_ids_strict": list(
                    score.get("evidence_core_missing_ids_strict") or []
                ),
                "evidence_core_missing_equiv_keys": list(
                    score.get("evidence_core_missing_equiv_keys") or []
                ),
                "evidence_core_id_mismatch_but_equiv_present": bool(
                    score.get("evidence_core_id_mismatch_but_equiv_present", False)
                ),
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
                "goc_depwalk_added_ids": list(debug.get("goc_depwalk_added_ids") or []),
                "goc_depwalk_added_count": int(debug.get("goc_depwalk_added_count", 0) or 0),
                "goc_depwalk_added_applicable_rate": debug.get(
                    "goc_depwalk_added_applicable_rate", float("nan")
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
                "goc_exception_rescue_ids": list(debug.get("goc_exception_rescue_ids") or []),
                "goc_exception_rescue_reason": list(debug.get("goc_exception_rescue_reason") or []),
                "goc_exception_rescue_count": len(
                    list(debug.get("goc_exception_rescue_ids") or [])
                ),
                "goc_rescue_ran": bool(debug.get("goc_rescue_ran", False)),
                "goc_rescue_reason_short": str(debug.get("goc_rescue_reason_short", "not_needed") or "not_needed"),
                "goc_update_history_rescue_ids": list(
                    debug.get("goc_update_history_rescue_ids") or []
                ),
                "goc_update_history_rescue_count": len(
                    list(debug.get("goc_update_history_rescue_ids") or [])
                ),
                "e3_context_clause_ids": list(context_ids),
                "goc_exception_injected_ids": list(exception_injected_ids),
                "goc_exception_injected_count": int(len(exception_injected_ids)),
                "goc_exception_applicable_count": int(exception_applicable_count),
                "prompt_tokens_est": int(score.get("token_est", 0) or 0),
                "prompt_tokens": int(score.get("token_est", 0) or 0),
                "prompt_tokens_actual": prompt_tokens_actual,
                "completion_tokens_actual": completion_tokens_actual,
                "total_tokens_actual": total_tokens_actual,
                "llm_output_text": str(llm_output_text or "")[:2000],
                "llm_parse_error": bool(llm_parse_error),
                "revive_success": bool(score.get("revive_success", False)),
                "indirection_overlap_gold": (
                    float(step.metadata.get("indirection_overlap_gold"))
                    if isinstance(step.metadata.get("indirection_overlap_gold"), (int, float))
                    else float("nan")
                ),
                "best_gold_sim": (
                    float(step.metadata.get("best_gold_sim"))
                    if isinstance(step.metadata.get("best_gold_sim"), (int, float))
                    else float("nan")
                ),
                "best_distractor_sim": (
                    float(step.metadata.get("best_distractor_sim"))
                    if isinstance(step.metadata.get("best_distractor_sim"), (int, float))
                    else float("nan")
                ),
                "trap_gap": (
                    float(step.metadata.get("trap_gap"))
                    if isinstance(step.metadata.get("trap_gap"), (int, float))
                    else float("nan")
                ),
                "trap_present": bool(step.metadata.get("trap_present", False)),
                "core_size": int(step.metadata.get("core_size", len(step.pivot_required_ids)) or 0),
                "pivot_style": str(step.metadata.get("pivot_style", "")),
                "trap_distractor_ids": list(trap_distractor_ids),
                "trap_injected_ids": list(trap_injected_ids),
                "trap_injected_count": int(trap_injected_count),
                "trap_injected_rate": trap_injected_rate,
                "core_necessity_flip_count": core_necessity_flip_count,
                "core_necessity_all_required": bool(step_meta.get("core_necessity_all_required", False)),
                "core_necessity_failed": bool(step_meta.get("core_necessity_failed", False)),
                "trap_decision_label": str(step_meta.get("trap_decision_label", "") or ""),
                "trap_decision_flip": bool(step_meta.get("trap_decision_flip", False)),
                "trap_flip_target_id": str(step_meta.get("trap_flip_target_id", "") or ""),
                "trap_flip_target_kind": str(step_meta.get("trap_flip_target_kind", "") or ""),
                "trap_graph_excludable_count": int(step_meta.get("trap_graph_excludable_count", 0) or 0),
                "trap_graph_excludable_ids": list(_unique_strs(step_meta.get("trap_graph_excludable_ids") or [])),
                "decision_checkpoint_trap_count": int(
                    step_meta.get("decision_checkpoint_trap_count", len(decision_checkpoint_trap_ids))
                    or 0
                ),
                "decision_checkpoint_trap_ids": list(decision_checkpoint_trap_ids),
                "decision_checkpoint_trap_excludable_ids": list(
                    decision_checkpoint_trap_excludable_ids
                ),
                "decision_checkpoint_trap_excludable_count": int(
                    len(decision_checkpoint_trap_excludable_set)
                ),
                "decision_checkpoint_trap_injected_ids": list(
                    decision_checkpoint_trap_injected_ids
                ),
                "decision_checkpoint_trap_injected_count": int(
                    decision_checkpoint_trap_injected_count
                ),
                "decision_checkpoint_trap_injected_rate": decision_checkpoint_trap_injected_rate,
                "trap_invalidation_attached_to_update": bool(
                    step_meta.get("trap_invalidation_attached_to_update", False)
                ),
                "trap_flip_salience": (
                    float(step_meta.get("trap_flip_salience"))
                    if isinstance(step_meta.get("trap_flip_salience"), (int, float))
                    else float("nan")
                ),
                "trap_flip_attach_kind": str(step_meta.get("trap_flip_attach_kind", "") or ""),
                "trap_graph_excludable_rate": (
                    float(step_meta.get("trap_graph_excludable_rate"))
                    if isinstance(step_meta.get("trap_graph_excludable_rate"), (int, float))
                    else float("nan")
                ),
                "trap_graph_excludable_kinds": str(step_meta.get("trap_graph_excludable_kinds", "") or ""),
                "trap_invalidation_text_strength": (
                    float(step_meta.get("trap_invalidation_text_strength"))
                    if isinstance(step_meta.get("trap_invalidation_text_strength"), (int, float))
                    else float("nan")
                ),
                "hidden_core_ids": list(hidden_core_ids),
                "hidden_core_parent_ids": list(hidden_core_parent_ids),
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
    decision_strict_raw_vals = [
        1.0 if r.get("decision_correct_strict_raw") else 0.0 for r in pivot_records
    ]
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
    exception_injected_counts = [
        int(r.get("goc_exception_injected_count", 0) or 0) for r in pivot_records
    ]
    exception_injected_vals = [1.0 if cnt > 0 else 0.0 for cnt in exception_injected_counts]
    indirection_vals = [
        float(r.get("indirection_overlap_gold"))
        for r in pivot_records
        if isinstance(r.get("indirection_overlap_gold"), (int, float)) and math.isfinite(float(r.get("indirection_overlap_gold")))
    ]
    trap_gap_vals = [
        float(r.get("trap_gap"))
        for r in pivot_records
        if isinstance(r.get("trap_gap"), (int, float)) and math.isfinite(float(r.get("trap_gap")))
    ]
    trap_injected_count_vals = [
        float(int(r.get("trap_injected_count", 0) or 0)) for r in pivot_records
    ]
    trap_injected_rate_vals = [
        float(r.get("trap_injected_rate"))
        for r in pivot_records
        if isinstance(r.get("trap_injected_rate"), (int, float))
        and math.isfinite(float(r.get("trap_injected_rate")))
    ]
    trap_injected_any_vals = [
        1.0 if int(r.get("trap_injected_count", 0) or 0) > 0 else 0.0 for r in pivot_records
    ]
    decision_checkpoint_trap_injected_count_vals = [
        float(int(r.get("decision_checkpoint_trap_injected_count", 0) or 0))
        for r in pivot_records
    ]
    decision_checkpoint_trap_injected_rate_vals = [
        float(r.get("decision_checkpoint_trap_injected_rate"))
        for r in pivot_records
        if isinstance(r.get("decision_checkpoint_trap_injected_rate"), (int, float))
        and math.isfinite(float(r.get("decision_checkpoint_trap_injected_rate")))
    ]
    core_size_vals = [float(int(r.get("core_size", 0) or 0)) for r in pivot_records]
    trap_present_vals = [1.0 if bool(r.get("trap_present", False)) else 0.0 for r in pivot_records]
    core_need_all_required_vals = [
        1.0 if bool(r.get("core_necessity_all_required", False)) else 0.0 for r in pivot_records
    ]
    core_need_failed_vals = [1.0 if bool(r.get("core_necessity_failed", False)) else 0.0 for r in pivot_records]
    core_need_flip_vals = [
        float(r.get("core_necessity_flip_count"))
        for r in pivot_records
        if isinstance(r.get("core_necessity_flip_count"), (int, float))
        and math.isfinite(float(r.get("core_necessity_flip_count")))
    ]
    trap_decision_flip_vals = [1.0 if bool(r.get("trap_decision_flip", False)) else 0.0 for r in pivot_records]
    hidden_core_present_vals = [
        1.0 if len(list(r.get("hidden_core_ids") or [])) > 0 else 0.0 for r in pivot_records
    ]

    method_name = str(method or "")
    depwalk_enabled = bool(getattr(args, "goc_depwalk_enable", False))
    hidden_pivot_records = [
        r for r in pivot_records if len(list(r.get("hidden_core_ids") or [])) > 0
    ]
    hidden_core_rescued_by_depwalk_rate = float("nan")
    hidden_core_missing_without_depwalk_rate = float("nan")
    if method_name == "goc" and depwalk_enabled:
        rescued_vals = []
        for rec in hidden_pivot_records:
            hidden_set = {str(cid) for cid in (rec.get("hidden_core_ids") or []) if str(cid).strip()}
            depwalk_set = {str(cid) for cid in (rec.get("goc_depwalk_added_ids") or []) if str(cid).strip()}
            rescued_vals.append(1.0 if bool(hidden_set & depwalk_set) else 0.0)
        hidden_core_rescued_by_depwalk_rate = _mean(rescued_vals)
    elif method_name == "goc" and not depwalk_enabled:
        missing_vals = []
        for rec in hidden_pivot_records:
            hidden_set = {str(cid) for cid in (rec.get("hidden_core_ids") or []) if str(cid).strip()}
            context_set = {str(cid) for cid in (rec.get("e3_context_clause_ids") or []) if str(cid).strip()}
            missing_vals.append(1.0 if bool(hidden_set - context_set) else 0.0)
        hidden_core_missing_without_depwalk_rate = _mean(missing_vals)

    tokens_pivot_mean_est = _mean(pivot_tokens_est)
    tokens_total_mean_est = _mean(total_tokens_by_thread_est)
    tokens_pivot_mean_actual = _mean(pivot_tokens_actual)
    tokens_total_mean_actual = _mean(total_tokens_by_thread_actual)
    tokens_pivot_mean = tokens_pivot_mean_actual if math.isfinite(tokens_pivot_mean_actual) else tokens_pivot_mean_est
    tokens_total_mean = tokens_total_mean_actual if math.isfinite(tokens_total_mean_actual) else tokens_total_mean_est

    metrics = {
        "decision_accuracy": _mean(decision_vals),
        "decision_accuracy_strict_raw": _mean(decision_strict_raw_vals),
        "judge_accuracy": _mean(e3_only_vals),
        "pivot_decision_accuracy": _mean(decision_vals),
        "pivot_decision_accuracy_strict_raw": _mean(decision_strict_raw_vals),
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
        "exception_injected_rate": _mean(exception_injected_vals),
        "mean_exception_injected_count": _mean([float(v) for v in exception_injected_counts]),
        "revive_success_rate": _mean(revive_vals),
        "mean_indirection_overlap_gold": _mean(indirection_vals),
        "mean_trap_gap": _mean(trap_gap_vals),
        "trap_present_rate": _mean(trap_present_vals),
        "mean_trap_injected_count": _mean(trap_injected_count_vals),
        "mean_trap_injected_rate": _mean(trap_injected_rate_vals),
        "trap_injected_any_rate": _mean(trap_injected_any_vals),
        "mean_decision_checkpoint_trap_injected_count": _mean(
            decision_checkpoint_trap_injected_count_vals
        ),
        "decision_checkpoint_trap_injected_rate": _mean(
            decision_checkpoint_trap_injected_rate_vals
        ),
        "mean_core_size": _mean(core_size_vals),
        "core_necessity_all_required_rate": _mean(core_need_all_required_vals),
        "mean_core_necessity_flip_count": _mean(core_need_flip_vals),
        "core_necessity_failed_rate": _mean(core_need_failed_vals),
        "trap_decision_flip_rate": _mean(trap_decision_flip_vals),
        "hidden_core_present_rate": _mean(hidden_core_present_vals),
        "hidden_core_rescued_by_depwalk_rate": hidden_core_rescued_by_depwalk_rate,
        "hidden_core_missing_without_depwalk_rate": hidden_core_missing_without_depwalk_rate,
        "pivot_records": int(len(pivot_records)),
        "thread_records": int(len(thread_records)),
        "pivots_available_total": int(pivots_available_total),
        "pivots_evaluated": int(len(pivot_records)),
        "steps_available_total": int(steps_available_total),
        "traceops_eval_mode": eval_mode,
        "traceops_llm_eval_scope": llm_eval_scope,
        "traceops_llm_sample_rate": llm_sample_rate,
        "sampled_steps_evaluated": int(sampled_steps_evaluated) if use_llm else 0,
        "sampled_step_rate": (
            float(sampled_steps_evaluated) / float(max(1, steps_available_total))
            if use_llm and llm_eval_scope == "sample"
            else float("nan")
        ),
        "llm_pivots_evaluated": int(llm_pivots_seen) if use_llm else 0,
        "llm_steps_evaluated": int(llm_steps_seen) if use_llm else 0,
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
