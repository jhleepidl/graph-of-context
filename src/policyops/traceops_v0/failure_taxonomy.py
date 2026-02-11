from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


def normalize_exception_text(text: str) -> str:
    raw = str(text or "").lower().strip()
    raw = re.sub(r"\b[a-z]{1,4}\d{2,}\b", " ", raw)
    raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def build_exception_equiv_map(context_clauses: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for clause in list(context_clauses or []):
        if not isinstance(clause, dict):
            continue
        ctype = str(clause.get("type", clause.get("node_type", "")) or "")
        if ctype != "EXCEPTION":
            continue
        cid = str(clause.get("id", clause.get("clause_id", "")) or "").strip()
        if not cid:
            continue
        out[cid] = normalize_exception_text(str(clause.get("text", "") or ""))
    return out


def _event_key(event: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(event.get("task_id", "") or ""),
        str(event.get("thread_id", "") or ""),
        int(event.get("step_idx", -1) or -1),
    )


def _canonicalize_tag(tag: str, ex_map: Dict[str, str]) -> str:
    raw = str(tag or "").strip()
    if not raw.lower().startswith("exception="):
        return raw
    _, _, cid = raw.partition("=")
    cid = str(cid).strip()
    ex_key = str(ex_map.get(cid, "") or "")
    return f"exception_key={ex_key}" if ex_key else raw


def _exception_applicable_from_event(clause: Dict[str, Any], state: Dict[str, Any]) -> bool:
    text = str(clause.get("text", "") or "").lower()
    state_key = str(clause.get("state_key", "") or "").strip()
    state_value = str(clause.get("state_value", "") or "").strip()
    if state_key and state_value:
        return str(state.get(state_key, "")).strip().lower() == state_value.lower()
    m = re.search(r"if\s+(\w+)\s+is\s+(\w+)", text)
    if m:
        key = str(m.group(1)).strip()
        value = str(m.group(2)).strip().lower()
        return str(state.get(key, "")).strip().lower() == value
    if "residency mismatch" in text:
        if "residency" in state and "region" in state:
            return str(state.get("residency", "")).strip().lower() != str(state.get("region", "")).strip().lower()
    return True


def _is_correct(record: Dict[str, Any]) -> bool:
    if isinstance(record.get("e3_answer_correct"), bool):
        return bool(record.get("e3_answer_correct"))
    if isinstance(record.get("decision_correct"), bool):
        return bool(record.get("decision_correct"))
    return False


def detect_categories(
    winner_record: Dict[str, Any],
    loser_record: Dict[str, Any],
) -> List[str]:
    _ = winner_record
    categories: List[str] = []
    diag = loser_record.get("diag") if isinstance(loser_record.get("diag"), dict) else {}
    scenario_diag = diag.get("scenario_metrics") if isinstance(diag.get("scenario_metrics"), dict) else {}

    if not bool(loser_record.get("evidence_core_covered_strict", True)):
        categories.append("gold_core_missing_in_context_strict")
    if not bool(loser_record.get("evidence_core_covered_equiv", True)):
        categories.append("gold_core_missing_in_context_equiv")
    if bool(loser_record.get("evidence_core_id_mismatch_but_equiv_present", False)):
        categories.append("gold_core_id_mismatch_but_equiv_present")

    gold = loser_record.get("gold") if isinstance(loser_record.get("gold"), dict) else {}
    pred = loser_record.get("pred") if isinstance(loser_record.get("pred"), dict) else {}
    gold_conditions = [str(x) for x in (gold.get("conditions") or [])]
    pred_conditions = [str(x) for x in (pred.get("conditions") or [])]

    if "apply_latest_update" in gold_conditions and "apply_latest_update" not in pred_conditions:
        categories.append("missing_apply_latest_update_tag")

    gold_has_exception = any(tag.startswith("exception=") for tag in gold_conditions)
    if gold_has_exception:
        ex_map = build_exception_equiv_map(loser_record.get("context_clauses") or [])
        canon_gold = {_canonicalize_tag(tag, ex_map) for tag in gold_conditions if tag.startswith("exception=")}
        canon_pred = {_canonicalize_tag(tag, ex_map) for tag in pred_conditions if tag.startswith("exception=")}
        non_exception_gold = {tag for tag in gold_conditions if not tag.startswith("exception=")}
        non_exception_pred = set(pred_conditions)
        if non_exception_gold.issubset(non_exception_pred) and (not canon_gold.issubset(canon_pred)):
            categories.append("missing_exception_tag_when_gold_has_exception_equiv")

    context_clauses = [
        c for c in (loser_record.get("context_clauses") or []) if isinstance(c, dict)
    ]
    ex_texts = [
        normalize_exception_text(str(c.get("text", "") or ""))
        for c in context_clauses
        if str(c.get("type", c.get("node_type", "")) or "") == "EXCEPTION"
    ]
    ex_texts = [t for t in ex_texts if t]
    if len(ex_texts) > len(set(ex_texts)):
        categories.append("duplicate_exception_text_in_context")

    state = loser_record.get("state") if isinstance(loser_record.get("state"), dict) else {}
    inapplicable_exception = False
    for clause in context_clauses:
        if str(clause.get("type", clause.get("node_type", "")) or "") != "EXCEPTION":
            continue
        if not _exception_applicable_from_event(clause, state):
            inapplicable_exception = True
            break
    if inapplicable_exception:
        categories.append("exception_present_but_inapplicable")

    avoid_ids = {
        str(cid)
        for cid in (loser_record.get("goc_avoid_target_clause_ids") or [])
        if str(cid).strip()
    }
    context_ids = {
        str(cid) for cid in (loser_record.get("context_clause_ids") or []) if str(cid).strip()
    }
    if avoid_ids and (avoid_ids & context_ids):
        categories.append("avoid_target_injected")

    indirection_overlap = loser_record.get("indirection_overlap_gold")
    if not isinstance(indirection_overlap, (int, float)):
        indirection_overlap = scenario_diag.get("indirection_overlap_gold")
    if isinstance(indirection_overlap, (int, float)) and float(indirection_overlap) < 0.18:
        categories.append("high_indirection_failure")

    core_size = loser_record.get("core_size")
    if not isinstance(core_size, (int, float)):
        core_size = scenario_diag.get("core_size")
    if isinstance(core_size, (int, float)) and int(core_size) >= 3:
        if not bool(loser_record.get("evidence_core_covered_equiv", True)):
            categories.append("compositional_core_partial")

    trap_present = loser_record.get("trap_present")
    if not isinstance(trap_present, bool):
        trap_present = bool(scenario_diag.get("trap_present", False))
    trap_ids = loser_record.get("trap_distractor_ids")
    if trap_ids is None:
        trap_ids = scenario_diag.get("trap_distractor_ids")
    trap_set = {str(cid) for cid in (trap_ids or []) if str(cid).strip()}
    pred_evidence = {str(cid) for cid in (pred.get("evidence") or []) if str(cid).strip()}
    if trap_present and trap_set:
        if (trap_set & context_ids) or (trap_set & pred_evidence):
            categories.append("distractor_trap_failure")

    return categories


def compare_pair(
    event_traces_a: Sequence[Dict[str, Any]],
    event_traces_b: Sequence[Dict[str, Any]],
    *,
    scenario: str,
    variant_a: str,
    variant_b: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    amap = {_event_key(ev): ev for ev in event_traces_a if isinstance(ev, dict)}
    bmap = {_event_key(ev): ev for ev in event_traces_b if isinstance(ev, dict)}
    common_keys = sorted(set(amap.keys()) & set(bmap.keys()))

    rows: List[Dict[str, Any]] = []
    examples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "A_correct_B_wrong": {},
        "B_correct_A_wrong": {},
    }
    dir_totals = {"A_correct_B_wrong": 0, "B_correct_A_wrong": 0}
    bucket: Dict[Tuple[str, str], List[str]] = {}

    for key in common_keys:
        a = amap[key]
        b = bmap[key]
        a_ok = _is_correct(a)
        b_ok = _is_correct(b)
        if a_ok and not b_ok:
            direction = "A_correct_B_wrong"
            winner, loser = a, b
        elif b_ok and not a_ok:
            direction = "B_correct_A_wrong"
            winner, loser = b, a
        else:
            continue

        dir_totals[direction] += 1
        cats = detect_categories(winner, loser)
        if not cats:
            cats = ["other_error"]
        key_str = f"{key[0]}|{key[1]}|{key[2]}"
        for cat in cats:
            bucket.setdefault((direction, cat), []).append(key_str)
            examples.setdefault(direction, {}).setdefault(cat, []).append(
                {
                    "key": key_str,
                    "winner_gold": (winner.get("gold") or {}),
                    "winner_pred": (winner.get("pred") or {}),
                    "winner_diag": (winner.get("diag") or {}),
                    "loser_gold": (loser.get("gold") or {}),
                    "loser_pred": (loser.get("pred") or {}),
                    "loser_diag": (loser.get("diag") or {}),
                    "loser_evidence_core_covered_strict": loser.get("evidence_core_covered_strict"),
                    "loser_evidence_core_covered_equiv": loser.get("evidence_core_covered_equiv"),
                    "loser_evidence_core_id_mismatch_but_equiv_present": loser.get(
                        "evidence_core_id_mismatch_but_equiv_present"
                    ),
                    "loser_context_clauses": list(loser.get("context_clauses") or [])[:8],
                }
            )

    for (direction, cat), keys in sorted(bucket.items(), key=lambda item: (item[0][0], item[0][1])):
        unique_keys = list(dict.fromkeys(keys))
        denom = max(1, int(dir_totals.get(direction, 0)))
        rows.append(
            {
                "scenario": str(scenario),
                "variant_a": str(variant_a),
                "variant_b": str(variant_b),
                "direction": str(direction),
                "category": str(cat),
                "count": int(len(keys)),
                "rate_over_direction": float(len(keys)) / float(denom),
                "example_keys": ";".join(unique_keys[:3]),
            }
        )

    return rows, examples


__all__ = [
    "normalize_exception_text",
    "build_exception_equiv_map",
    "detect_categories",
    "compare_pair",
]
