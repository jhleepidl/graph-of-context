from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import hashlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Set, Tuple

from .baselines import (
    DummyClient,
    LLMClient,
    OpenAIClient,
    _ensure_min_evidence,
    _build_prompt,
    _enrich_search_results,
    run_engine_oracle,
    run_full_history,
    run_goc_heuristic,
    run_oracle,
    run_topk_rag,
    run_similarity_only,
    run_agent_fold,
    summarize_clause_history,
    _parse_prediction,
)
from .controller import Controller, RerankController
from .diagnostics import compute_retrieval_diagnostics, merge_search_results_union
from .symbolic_judge import judge_from_opened_clauses, judge_threaded_final
from .analysis import (
    analyze_failure_slice,
    analyze_bridged_ab,
    analyze_selection_triage,
    analyze_slot_breakdown,
    analyze_split_sweep_ab,
    analyze_bundle,
)
from .bridged_ab import compute_bridged_ab_slices
from .schemas import Gold


def _find_src_dir_for_import(start: Path) -> Path | None:
    """Find a src root that contains goc_logger for fallback imports."""
    resolved = start.resolve()
    for parent in [resolved.parent] + list(resolved.parents):
        cand = parent / "src" / "goc_logger"
        if cand.exists():
            return parent / "src"
        cand_alt = parent / "goc_logger"
        if cand_alt.exists():
            return parent
    return None


def _import_goc_policy_module() -> Any:
    try:
        import goc_policy as _gp  # type: ignore

        return _gp
    except ModuleNotFoundError:
        _root_src = _find_src_dir_for_import(Path(__file__))
        if _root_src is not None and str(_root_src) not in sys.path:
            sys.path.append(str(_root_src))
        import goc_policy as _gp  # type: ignore

        return _gp


try:
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
except ModuleNotFoundError:
    _root_src = _find_src_dir_for_import(Path(__file__))
    if _root_src is not None and str(_root_src) not in sys.path:
        sys.path.append(str(_root_src))
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
from .env import PolicyOpsEnv
from .eval import aggregate_metrics, evaluate_prediction, gold_decision_distribution, save_report
from .generator import generate_world_and_tasks
from .world import evaluate_context, load_tasks, load_world

CALIB_PRESET_N8 = "bridged_v1_1_calib_n8_exclcore"
CALIB_PRESET_N10 = "bridged_v1_1_calib_n10_exclcore"
CALIBRATED_PRESET = "bridged_v1_1_calibrated"
THREADED_PRESET_N8 = "threaded_v1_2_calib_n8_exclcore"
THREADED_PRESET_N10 = "threaded_v1_2_calib_n10_exclcore"
THREADED_FU_PRESET_N8 = "threaded_v1_3_fu_calib_n8"
THREADED_FU_PRESET_N10 = "threaded_v1_3_fu_calib_n10"
THREADED_FU_DECOY_JITTER_PRESET_N10 = "threaded_v1_3_fu_decoy_calib_jitter_n10"
THREADED_FU_DECOY_DEPTHJITTER_MODE = "threaded_v1_3_fu_decoy_depthjitter"
PRESET_CONFIGS = {
    CALIB_PRESET_N8: {
        "scenario_mode": "bridged_v1_1",
        "n_docs": 8,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
    },
    CALIB_PRESET_N10: {
        "scenario_mode": "bridged_v1_1",
        "n_docs": 10,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
    },
    # Backward-compatible alias; prefer CALIB_PRESET_N10/CALIB_PRESET_N8 for research runs.
    CALIBRATED_PRESET: {
        "scenario_mode": "bridged_v1_1",
        "n_docs": 10,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
    },
    THREADED_PRESET_N8: {
        "scenario_mode": "threaded_v1_2",
        "n_docs": 8,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
        "n_threads": 100,
        "open_budget_e1": 4,
        "open_budget_e2": 4,
        "open_budget_e3": 0,
        "tool_budget_e1": 50,
        "tool_budget_e2": 50,
        "tool_budget_e3": 0,
    },
    THREADED_PRESET_N10: {
        "scenario_mode": "threaded_v1_2",
        "n_docs": 10,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
        "n_threads": 100,
        "open_budget_e1": 4,
        "open_budget_e2": 4,
        "open_budget_e3": 0,
        "tool_budget_e1": 50,
        "tool_budget_e2": 50,
        "tool_budget_e3": 0,
    },
    THREADED_FU_PRESET_N8: {
        "scenario_mode": "threaded_v1_3_fu",
        "n_docs": 8,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
        "n_threads": 100,
        "open_budget_e1": 4,
        "open_budget_e2": 4,
        "open_budget_e3": 0,
        "tool_budget_e1": 50,
        "tool_budget_e2": 50,
        "tool_budget_e3": 0,
    },
    THREADED_FU_PRESET_N10: {
        "scenario_mode": "threaded_v1_3_fu",
        "n_docs": 10,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
        "n_threads": 100,
        "open_budget_e1": 4,
        "open_budget_e2": 4,
        "open_budget_e3": 0,
        "tool_budget_e1": 50,
        "tool_budget_e2": 50,
        "tool_budget_e3": 0,
    },
    THREADED_FU_DECOY_JITTER_PRESET_N10: {
        "scenario_mode": "threaded_v1_3_fu_decoy",
        "n_docs": 10,
        "clauses_per_doc": 5,
        "alias_density": 0.9,
        "canonical_density": 0.95,
        "exclusive_core_evidence": True,
        "n_threads": 100,
        "open_budget_e1": 4,
        "open_budget_e2": 4,
        "open_budget_e3": 0,
        "tool_budget_e1": 50,
        "tool_budget_e2": 50,
        "tool_budget_e3": 0,
        "e3_clause_jitter_max_chars_critical": 200,
        "e3_clause_jitter_max_chars_noncritical": 400,
        "e3_clause_jitter_max_chars_decoy": 400,
        "e3_clause_jitter_scope": "decoy_plus_noncritical",
    },
}
DEFAULT_GENERATION = {
    "n_docs": 30,
    "clauses_per_doc": 5,
    "alias_density": 0.9,
    "canonical_density": 0.95,
}
DEEP_RANK_CORE_THRESHOLD = 10
COST_BLOWUP_TOKENS = 2000
THREADED_FU_RECOMMENDED_BUDGET = 1350
SCENARIO_MODE_ALIASES = {
    THREADED_FU_DECOY_DEPTHJITTER_MODE: "threaded_v1_3_fu_decoy",
}


def _canonical_scenario_mode(mode: Any) -> Any:
    if not isinstance(mode, str):
        return mode
    return SCENARIO_MODE_ALIASES.get(mode, mode)


def _normalize_scenario_mode_arg(args: argparse.Namespace) -> None:
    if not hasattr(args, "scenario_mode"):
        return
    args.scenario_mode = _canonical_scenario_mode(getattr(args, "scenario_mode", None))


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _avg_p90(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p90": 0.0}
    avg = sum(values) / len(values)
    ordered = sorted(values)
    idx = int(0.9 * (len(ordered) - 1))
    p90 = ordered[idx]
    return {"avg": avg, "p90": float(p90)}


def _task_metrics_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    def _as_float(key: str, default: float = 0.0) -> float:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    return {
        "decision_accuracy": 1.0 if record.get("decision_correct") else 0.0,
        "condition_f1": _as_float("condition_f1"),
        "evidence_precision": _as_float("evidence_precision"),
        "evidence_recall": _as_float("evidence_recall"),
        "evidence_precision_core": _as_float("evidence_precision_core"),
        "evidence_recall_core": _as_float("evidence_recall_core"),
        "critical_evidence_hit": _as_float("critical_evidence_hit"),
    }


def _extract_clause_ids(results: Any) -> List[str]:
    if not isinstance(results, list):
        return []
    ids: List[str] = []
    for item in results:
        if isinstance(item, dict):
            cid = item.get("clause_id")
            if cid:
                ids.append(str(cid))
    return ids


def _unique_strs(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for val in values:
        if val is None:
            continue
        sval = str(val).strip()
        if not sval or sval in seen:
            continue
        seen.add(sval)
        out.append(sval)
    return out


# --- Phase 12: commit anchor by model evidence (canonicalize clause IDs) ---
_CLAUSE_ID_CANON_RE = re.compile(r"\bC[-_\s]?0*(\d+)\b", re.IGNORECASE)

def _canonicalize_clause_id(value: Any) -> str | None:
    """Normalize clause identifiers to generator format C0001.

    Accepts: 'C0001', 'C-001', 'c 1', '1', etc.
    Returns None if it cannot be parsed.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Already in Cdddd
    if len(s) >= 2 and (s[0] in {"C", "c"}) and s[1:].isdigit():
        try:
            return f"C{int(s[1:]):04d}"
        except Exception:
            return None
    m = _CLAUSE_ID_CANON_RE.search(s)
    if m:
        try:
            return f"C{int(m.group(1)):04d}"
        except Exception:
            return None
    if s.isdigit():
        try:
            return f"C{int(s):04d}"
        except Exception:
            return None
    return None


def _canonicalize_clause_ids(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for item in values:
        cid = _canonicalize_clause_id(item)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _extract_model_evidence_clause_ids(
    pred: Any,
    *,
    opened_ids: List[str],
    max_ids: int,
    require_opened: bool,
) -> List[str]:
    if not isinstance(pred, dict):
        return []
    evidence_raw = pred.get("evidence") or []
    model_ids = _canonicalize_clause_ids(evidence_raw)
    if require_opened:
        opened_set = set(opened_ids or [])
        model_ids = [cid for cid in model_ids if cid in opened_set]
    if max_ids and max_ids > 0:
        model_ids = model_ids[:max_ids]
    return model_ids


def _choose_commit_anchor_ids(
    *,
    opened_supporting: List[str],
    model_evidence: List[str],
    source: str,
    max_ids: int,
    fallback_opened: bool,
) -> Tuple[List[str], str]:
    """Return (anchor_ids, source_used)."""
    opened_supporting = list(opened_supporting or [])
    model_evidence = list(model_evidence or [])
    source_used = source
    if source == "opened_supporting":
        anchors = opened_supporting
    elif source == "model_evidence":
        anchors = model_evidence
        if not anchors and fallback_opened:
            anchors = opened_supporting
            source_used = "opened_supporting_fallback"
    elif source == "hybrid":
        anchors = _unique_strs(list(model_evidence) + list(opened_supporting))
        if not anchors and fallback_opened:
            anchors = opened_supporting
            source_used = "opened_supporting_fallback"
    else:
        anchors = opened_supporting
        source_used = "opened_supporting"
    if max_ids and max_ids > 0:
        anchors = list(anchors)[:max_ids]
    return list(anchors), source_used

def _clause_docid_map(world: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    clauses = getattr(world, "clauses", {}) if world is not None else {}
    if not isinstance(clauses, dict):
        return out
    for cid, clause in clauses.items():
        doc_id = str(getattr(clause, "doc_id", "") or "").strip()
        if doc_id:
            out[str(cid)] = doc_id
    return out


def _world_docid_set(world: Any) -> Set[str]:
    return set(_clause_docid_map(world).values())


def _clause_ids_to_doc_ids(clause_ids: List[str], world: Any) -> List[str]:
    cmap = _clause_docid_map(world)
    out: List[str] = []
    seen: Set[str] = set()
    for cid in clause_ids:
        did = cmap.get(str(cid))
        if not did or did in seen:
            continue
        seen.add(did)
        out.append(did)
    return out


def _pick_final_snapshot(snapshots: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not snapshots:
        return None
    for snap in reversed(snapshots):
        if not isinstance(snap, dict):
            continue
        if str(snap.get("snapshot_kind", "")).lower() == "final":
            return snap
    for snap in reversed(snapshots):
        if isinstance(snap, dict):
            return snap
    return None


def _extract_active_from_snapshot(
    snapshot: Dict[str, Any] | None,
    world: Any,
) -> Tuple[List[str], List[str], List[str]]:
    if not isinstance(snapshot, dict):
        return [], [], []
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, dict):
        nodes = {}
    active_node_ids = _unique_strs(list(snapshot.get("active") or []))

    clause_ids: List[str] = []
    doc_ids: List[str] = []
    seen_clause: Set[str] = set()
    seen_doc: Set[str] = set()
    world_clause_ids = (
        {str(k) for k in getattr(world, "clauses", {}).keys()}
        if world is not None
        else set()
    )
    world_doc_ids = _world_docid_set(world)

    for nid in active_node_ids:
        node = nodes.get(nid)
        if not isinstance(node, dict):
            continue
        for raw in list(node.get("docids") or []):
            d = str(raw).strip()
            if not d:
                continue
            if d in world_clause_ids and d not in seen_clause:
                seen_clause.add(d)
                clause_ids.append(d)
            if d in world_doc_ids and d not in seen_doc:
                seen_doc.add(d)
                doc_ids.append(d)
        if node.get("docids"):
            continue
        text_preview = str(node.get("text_preview", "") or "")
        m = re.search(r"CLAUSE_ID:\s*([^\s|]+)", text_preview)
        if m:
            cid = str(m.group(1)).strip()
            if cid in world_clause_ids and cid not in seen_clause:
                seen_clause.add(cid)
                clause_ids.append(cid)

    # Ensure doc ids include clause->doc mapping for recalled clause ids.
    for did in _clause_ids_to_doc_ids(clause_ids, world):
        if did not in seen_doc:
            seen_doc.add(did)
            doc_ids.append(did)
    return active_node_ids, clause_ids, doc_ids


def _match_snapshot_nodes_for_clause_ids(
    snapshot: Dict[str, Any] | None,
    clause_ids: List[str],
) -> Tuple[List[str], List[str]]:
    if not isinstance(snapshot, dict):
        return [], []
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, dict):
        return [], []
    clause_set = set(str(cid) for cid in clause_ids if cid)
    if not clause_set:
        return [], []
    matched_nodes: List[str] = []
    matched_docids: List[str] = []
    seen_nodes: Set[str] = set()
    seen_docids: Set[str] = set()
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        node_docids = [str(d) for d in (node.get("docids") or []) if d]
        if not (clause_set & set(node_docids)):
            continue
        nid_s = str(nid)
        if nid_s not in seen_nodes:
            seen_nodes.add(nid_s)
            matched_nodes.append(nid_s)
        for d in node_docids:
            if d not in seen_docids:
                seen_docids.add(d)
                matched_docids.append(d)
    return matched_nodes, matched_docids


def _is_threaded_or_bridged(mode: str | None) -> bool:
    if not isinstance(mode, str):
        return False
    return mode.startswith("threaded_") or mode.startswith("bridged_")


def _is_non_winning_branch_clause(task: Any, world: Any, clause_id: str) -> bool:
    clause = world.clauses.get(clause_id) if world and getattr(world, "clauses", None) else None
    if clause is None:
        return False

    distractor = getattr(task, "branch_distractor_clause_id", None)
    if distractor and str(clause_id) == str(distractor):
        return True

    slot = task.context.get("slot") if isinstance(getattr(task, "context", None), dict) else None
    bridge_for_slot = getattr(clause, "bridge_for_slot", None)
    if slot and bridge_for_slot and str(bridge_for_slot) != str(slot):
        return True

    doc_id = str(getattr(clause, "doc_id", "") or "")
    if doc_id.startswith("DECOY"):
        return True

    return False


def _min_rank(results: List[Dict[str, Any]], target_ids: List[str]) -> int | None:
    if not results or not target_ids:
        return None
    target_set = set(target_ids)
    ordered = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for idx, item in enumerate(ordered, start=1):
        cid = item.get("clause_id")
        if cid in target_set:
            return idx
    return None


def _rank_summary(values: List[int]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    mean_val = sum(ordered) / len(ordered)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median_val = (ordered[mid - 1] + ordered[mid]) / 2
    else:
        median_val = ordered[mid]
    idx = int(0.9 * (len(ordered) - 1))
    p90_val = ordered[idx]
    return {"mean": float(mean_val), "median": float(median_val), "p90": float(p90_val)}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def _lexical_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _select_similarity_clause_ids(
    ticket: str,
    clause_ids: List[str],
    world: Any,
    top_k: int = 4,
) -> List[str]:
    if not clause_ids:
        return []
    ticket_tokens = set(_tokenize(ticket))
    scored: List[tuple[str, int]] = []
    for cid in clause_ids:
        clause = world.clauses.get(cid)
        if not clause:
            continue
        tokens = set(_tokenize(clause.text))
        score = len(ticket_tokens & tokens)
        scored.append((cid, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def extract_activity(ticket_text: str) -> Set[str]:
    text = (ticket_text or "").lower()
    acts: Set[str] = set()
    if "retain" in text or "retaining" in text:
        acts.add("retain")
    if "export" in text or "exporting" in text:
        acts.add("export")
    if "share" in text or "sharing" in text:
        acts.add("share")
    if "logs" in text:
        acts.add("logs")
    if "telemetry" in text:
        acts.add("telemetry")
    if "health" in text:
        acts.add("health")
    return acts


def clause_activity(clause_text: str) -> Set[str]:
    return extract_activity(clause_text)


def _extract_retention_days_from_text(text: str) -> int | None:
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


_CONTEXT_OVERRIDE_KEYS: Set[str] = {
    "slot",
    "region",
    "product",
    "tier",
    "purpose",
    "data_type",
    "retention_days",
    "retention_bucket",
}


def _normalize_goc_avoids_mode(mode: Any) -> str:
    value = str(mode or "").strip().lower()
    return value if value in {"legacy_commit", "applicability", "off"} else ""


def _resolve_goc_avoids_mode(args: Any) -> str:
    explicit = _normalize_goc_avoids_mode(getattr(args, "goc_avoids_mode", None))
    if explicit:
        return explicit
    return "applicability" if bool(getattr(args, "goc_enable_avoids", True)) else "off"


def _normalize_pivot_gold_mode(mode: Any) -> str:
    value = str(mode or "respect_ticket_updated").strip().lower()
    if value in {"original", "respect_ticket_updated", "both"}:
        return value
    return "respect_ticket_updated"


def _build_gold_from_context(world: Any, context: Dict[str, Any]) -> Gold:
    decision, conditions, evidence, _ = evaluate_context(world, context)
    evidence_ids = _unique_strs(list(evidence or []))
    meta_ids = [
        cid
        for cid in evidence_ids
        if world is not None
        and getattr(getattr(world, "clauses", {}), "get", None)
        and world.clauses.get(cid) is not None
        and str(getattr(world.clauses.get(cid), "kind", "") or "") == "priority"
    ]
    meta_set = set(meta_ids)
    core_ids = [cid for cid in evidence_ids if cid not in meta_set]
    return Gold(
        decision=str(decision or "needs_more_info"),
        conditions=list(conditions or []),
        gold_evidence=evidence_ids,
        gold_evidence_core=core_ids,
        gold_evidence_meta=meta_ids,
    )


def _apply_constraint_updates_to_gold(
    gold: Gold,
    *,
    pivot_type: str,
    constraint_updates: Dict[str, Any] | None,
) -> tuple[Gold, bool]:
    updates = constraint_updates if isinstance(constraint_updates, dict) else {}
    require_condition_value = str(updates.get("require_condition") or "").strip()
    if str(pivot_type or "") != "constraint_add" or not require_condition_value:
        return gold, False
    overridden = Gold(
        decision="require_condition",
        conditions=[require_condition_value],
        gold_evidence=list(gold.gold_evidence or []),
        gold_evidence_core=list(getattr(gold, "gold_evidence_core", None) or gold.gold_evidence or []),
        gold_evidence_meta=list(getattr(gold, "gold_evidence_meta", None) or []),
    )
    return overridden, True


def _compute_commit_quality(
    *,
    anchor_clause_ids: List[str] | None,
    supporting_clause_ids: List[str] | None,
    eval_core_clause_ids: List[str] | None,
) -> Dict[str, Any]:
    core_set = set(_unique_strs(list(eval_core_clause_ids or [])))
    anchor_set = set(_unique_strs(list(anchor_clause_ids or [])))
    supporting_set = set(_unique_strs(list(supporting_clause_ids or [])))
    union_set = anchor_set | supporting_set
    supporting_core_hits = sorted(core_set & supporting_set)
    if not core_set:
        return {
            "correct_anchor": None,
            "correct_union": None,
            "anchor_promotion_needed": None,
            "supporting_core_hits": supporting_core_hits,
        }
    correct_anchor = bool(core_set & anchor_set)
    correct_union = bool(core_set & union_set)
    anchor_promotion_needed = bool(
        correct_union and not correct_anchor and bool(supporting_core_hits)
    )
    return {
        "correct_anchor": correct_anchor,
        "correct_union": correct_union,
        "anchor_promotion_needed": anchor_promotion_needed,
        "supporting_core_hits": supporting_core_hits,
    }


def _compute_thread_e3_correctness(
    *,
    judge_correct: bool | None,
    commit1_correct: bool | None,
    commit2_correct: bool | None,
    evidence_after: List[str] | None,
    opened_ids: List[str] | None,
    commit_clause_ids: List[str] | None,
    e3_answer_correct: bool | None,
) -> Dict[str, Any]:
    evidence_list = list(evidence_after or [])
    opened_set = set(_unique_strs(list(opened_ids or [])))
    commit_set = set(_unique_strs(list(commit_clause_ids or [])))
    evidence_valid_in_context = all(cid in opened_set for cid in evidence_list)
    evidence_valid_in_commits = all(cid in commit_set for cid in evidence_list)
    strict_correct: bool | None = None
    if isinstance(judge_correct, bool):
        strict_correct = bool(
            judge_correct
            and commit1_correct is True
            and commit2_correct is True
            and evidence_valid_in_context
        )
    e3_only_correct: bool | None = None
    if isinstance(e3_answer_correct, bool):
        e3_only_correct = bool(e3_answer_correct and evidence_valid_in_context)
    return {
        "evidence_valid_in_context": evidence_valid_in_context,
        "evidence_valid_in_commits": evidence_valid_in_commits,
        "strict_correct": strict_correct,
        "e3_only_correct": e3_only_correct,
    }


def _parse_context_overrides_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    overrides: Dict[str, Any] = {}
    for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^\s,;()]+)", str(text)):
        key = str(match.group(1) or "").strip().lower()
        if key not in _CONTEXT_OVERRIDE_KEYS:
            continue
        value_raw = str(match.group(2) or "").strip().strip(".,;:!?)\"'")
        if not value_raw:
            continue
        if key == "retention_days":
            try:
                overrides[key] = int(value_raw)
            except Exception:
                continue
        else:
            overrides[key] = value_raw
    if isinstance(overrides.get("retention_days"), int) and not overrides.get("retention_bucket"):
        days = int(overrides["retention_days"])
        overrides["retention_bucket"] = "le_30" if days <= 30 else "gt_30"
    return overrides


def _parse_constraint_updates(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    match = re.search(
        r"\brequire_condition\s*[:=]\s*([^\s,;()]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    value = str(match.group(1) or "").strip().strip(".,;:!?)\"'")
    if not value:
        return {}
    return {"require_condition": value}


def _constraint_key_value(
    constraint_updates: Dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    updates = constraint_updates if isinstance(constraint_updates, dict) else {}
    value = str(updates.get("require_condition") or "").strip()
    if not value:
        return None, None
    return "require_condition", value


def _compute_effective_context_with_constraint_updates(
    task: Any,
    episode_id: int,
    *,
    threaded_mode: bool,
    thread_state: Dict[str, Any] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    effective_context = _compute_effective_context(
        task,
        episode_id,
        threaded_mode=threaded_mode,
        thread_state=thread_state,
    )
    if int(episode_id or 0) != 3:
        return effective_context, {}
    ticket_updated = str(getattr(task, "ticket_updated", None) or "").strip()
    if not ticket_updated:
        return effective_context, {}
    return effective_context, _parse_constraint_updates(ticket_updated)


def _compute_effective_context(
    task: Any,
    episode_id: int,
    *,
    threaded_mode: bool,
    thread_state: Dict[str, Any] | None,
) -> Dict[str, Any]:
    del threaded_mode, thread_state
    base = dict(getattr(task, "context", None) or {})
    if int(episode_id or 0) != 3:
        return base
    ticket_updated = str(getattr(task, "ticket_updated", None) or "").strip()
    if not ticket_updated:
        return base
    updated_days = _extract_retention_days_from_text(ticket_updated)
    if updated_days is not None:
        base["retention_days"] = int(updated_days)
        base["retention_bucket"] = "le_30" if updated_days <= 30 else "gt_30"
    kv_overrides = _parse_context_overrides_from_text(ticket_updated)
    if kv_overrides:
        base.update(kv_overrides)
        if isinstance(base.get("retention_days"), int):
            days = int(base.get("retention_days"))
            base["retention_bucket"] = "le_30" if days <= 30 else "gt_30"
    return base


def _compute_context_delta(
    initial_context: Dict[str, Any],
    effective_context: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    delta: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(initial_context.keys()) | set(effective_context.keys()))
    for key in keys:
        before = initial_context.get(key)
        after = effective_context.get(key)
        if before != after:
            delta[str(key)] = {"before": before, "after": after}
    return delta


def _state_node_id(episode: int, key: str, value: Any) -> str:
    raw = str(value if value is not None else "null")
    safe = re.sub(r"\s+", "_", raw)
    safe = re.sub(r"[^a-zA-Z0-9_.:/=-]", "_", safe)
    return f"state:e{int(episode)}:{key}={safe}"


def _build_state_node_maps(context: Dict[str, Any], episode: int) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    by_key: Dict[str, str] = {}
    node_meta: Dict[str, Dict[str, Any]] = {}
    for key in sorted(context.keys()):
        value = context.get(key)
        if isinstance(value, (dict, list, tuple, set)):
            continue
        skey = str(key)
        node_id = _state_node_id(episode, skey, value)
        by_key[skey] = node_id
        node_meta[node_id] = {
            "kind": "state",
            "key": skey,
            "value": value,
            "episode": int(episode),
        }
    return by_key, node_meta


def _append_unique_edge(edges: List[Dict[str, str]], edge_type: str, u: str, v: str) -> None:
    if not edge_type or not u or not v:
        return
    for edge in edges:
        if edge.get("type") == edge_type and edge.get("u") == u and edge.get("v") == v:
            return
    edges.append({"type": edge_type, "u": u, "v": v})


def _clause_applies_to_context(clause: Any, ctx: Dict[str, Any]) -> bool:
    applies_if = getattr(clause, "applies_if", None) or {}
    if not isinstance(applies_if, dict):
        return True
    for key, values in applies_if.items():
        if not isinstance(values, list) or not values:
            continue
        if ctx.get(key) not in values:
            return False
    return True


def _compute_goc_avoid_clause_ids(
    *,
    mode: str,
    is_pivot_task: bool,
    opened_history_ids: List[str],
    world: Any,
    effective_context: Dict[str, Any],
    commit1: Dict[str, Any] | None = None,
    commit2: Dict[str, Any] | None = None,
) -> tuple[List[str], List[str]]:
    normalized_mode = _normalize_goc_avoids_mode(mode) or "applicability"
    opened_ids = _unique_strs(opened_history_ids)
    inapplicable: List[str] = []
    for cid in opened_ids:
        clause = world.clauses.get(cid) if world is not None else None
        if clause is None:
            continue
        if not _clause_applies_to_context(clause, effective_context):
            inapplicable.append(cid)
    inapplicable = _unique_strs(inapplicable)
    if not is_pivot_task or normalized_mode == "off":
        return [], inapplicable
    if normalized_mode == "legacy_commit":
        c1 = commit1 or {}
        c2 = commit2 or {}
        legacy = _unique_strs(
            list(c1.get("anchor_clause_ids") or [])
            + list(c1.get("supporting_clause_ids") or [])
            + list(c2.get("anchor_clause_ids") or [])
            + list(c2.get("supporting_clause_ids") or [])
        )
        return legacy, inapplicable
    return list(inapplicable), inapplicable


def _is_unconstrained_clause(clause: Any) -> bool:
    applies_if = getattr(clause, "applies_if", None) or {}
    if not isinstance(applies_if, dict) or not applies_if:
        return True
    for values in applies_if.values():
        if isinstance(values, list) and values:
            return False
    return True


def _compute_phase14_applicability_seed_ids(
    *,
    candidate_clause_ids: List[str],
    world: Any,
    effective_context: Dict[str, Any],
    avoid_clause_ids: List[str] | None = None,
    top_k: int = 8,
) -> tuple[List[str], float]:
    """Pick top-K applicable seed clauses with stable ordering tie-breaks."""
    ordered = _unique_strs(candidate_clause_ids)
    avoid_set = set(_unique_strs(avoid_clause_ids or []))
    if not ordered:
        return [], float("nan")
    max_k = max(0, int(top_k or 0))
    if max_k <= 0:
        return [], float("nan")

    scored: List[tuple[float, int, str]] = []
    for idx, cid in enumerate(ordered):
        if cid in avoid_set:
            continue
        clause = world.clauses.get(cid) if world is not None else None
        if clause is None:
            continue
        if not _clause_applies_to_context(clause, effective_context):
            continue
        score = 1.0
        if str(getattr(clause, "kind", "") or "") == "priority":
            score += 0.05
        scored.append((score, idx, cid))

    if not scored:
        return [], float("nan")

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    seed_ids = [cid for _, _, cid in scored[:max_k]]
    applicable_rate = 1.0 if seed_ids else float("nan")
    return seed_ids, applicable_rate


def _compute_phase14_dependency_closure_ids(
    *,
    seed_clause_ids: List[str],
    unfold_candidate_ids: List[str],
    opened_history_ids: List[str],
    avoid_clause_ids: List[str],
    world: Any,
    effective_context: Dict[str, Any],
    top_k: int,
    hops: int,
    universe_mode: str,
) -> tuple[List[str], float]:
    """Compute closure-added clause IDs under applicability/avoid constraints."""
    seed_ids = _unique_strs(seed_clause_ids)
    if not seed_ids:
        return [], float("nan")
    max_top_k = max(0, int(top_k or 0))
    if max_top_k <= 0:
        return [], float("nan")

    mode = str(universe_mode or "candidates")
    if mode not in {"candidates", "world", "memory_opened"}:
        mode = "candidates"

    if mode == "world":
        clauses_map = getattr(world, "clauses", None) if world is not None else None
        if isinstance(clauses_map, dict):
            universe_clause_ids = list(clauses_map.keys())
        else:
            universe_clause_ids = list(_unique_strs(unfold_candidate_ids))
    elif mode == "memory_opened":
        universe_clause_ids = list(_unique_strs(opened_history_ids))
    else:
        universe_clause_ids = list(_unique_strs(unfold_candidate_ids))

    closure_full = _expand_clause_dependency_closure(
        seed_clause_ids=seed_ids,
        candidate_clause_ids=list(_unique_strs(unfold_candidate_ids)),
        world=world,
        max_hops=max(0, int(hops or 0)),
        universe_clause_ids=universe_clause_ids,
    )
    if not closure_full:
        return [], float("nan")

    selected_set = set(seed_ids)
    avoid_set = set(_unique_strs(avoid_clause_ids))
    closure_candidates: List[str] = []
    for cid in closure_full:
        if cid in selected_set or cid in avoid_set:
            continue
        clause = world.clauses.get(cid) if world is not None else None
        if clause is None:
            continue
        applicable = _clause_applies_to_context(clause, effective_context)
        keep = (
            applicable
            or str(getattr(clause, "kind", "") or "") == "priority"
            or _is_unconstrained_clause(clause)
        )
        if keep:
            closure_candidates.append(cid)
    closure_added = closure_candidates[:max_top_k]
    if not closure_added:
        return [], float("nan")
    applicable_count = 0
    for cid in closure_added:
        clause = world.clauses.get(cid) if world is not None else None
        if clause is not None and _clause_applies_to_context(clause, effective_context):
            applicable_count += 1
    rate = float(applicable_count) / float(len(closure_added))
    return closure_added, rate


def _compute_pivot_compliance(
    *,
    ticket_initial: str,
    ticket_updated: str,
    pivot_type: str,
    raw_output: str | None,
) -> tuple[bool | None, bool | None, int | None, int | None]:
    if not ticket_updated:
        return None, None, None, None
    kind = str(pivot_type or "retention_flip")
    text = str(raw_output or "").lower()
    if kind != "retention_flip":
        return None, None, None, None
    old_days = _extract_retention_days_from_text(ticket_initial)
    updated_days = _extract_retention_days_from_text(ticket_updated)
    if updated_days is None:
        return None, None, old_days, updated_days
    has_updated = str(updated_days) in text
    opposite_days = 90 if updated_days == 30 else 30 if updated_days == 90 else None
    has_opposite = str(opposite_days) in text if opposite_days is not None else False
    has_old = str(old_days) in text if old_days is not None else False
    pivot_compliant = bool(has_updated and not has_opposite and not (has_old and old_days != updated_days))
    stale_evidence = bool(has_old and old_days != updated_days) or bool(has_opposite)
    return pivot_compliant, stale_evidence, old_days, updated_days


def _token_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def _build_clause_dependency_neighbors(
    clause_ids: List[str],
    world: Any,
) -> Dict[str, Set[str]]:
    ordered_ids = _unique_strs(clause_ids)
    neighbors: Dict[str, Set[str]] = {cid: set() for cid in ordered_ids}
    if not ordered_ids or world is None:
        return neighbors

    clause_set = set(ordered_ids)
    clauses = getattr(world, "clauses", {}) if world is not None else {}
    if not isinstance(clauses, dict):
        clauses = {}
    world_meta = getattr(world, "meta", {}) if world is not None else {}
    term_definitions = (
        world_meta.get("term_definitions", {})
        if isinstance(world_meta, dict)
        else {}
    )
    if not isinstance(term_definitions, dict):
        term_definitions = {}

    doc_to_clause_ids: Dict[str, List[str]] = {}
    for cid in ordered_ids:
        clause = clauses.get(cid)
        if clause is None:
            continue
        doc_id = str(getattr(clause, "doc_id", "") or "").strip()
        if doc_id:
            doc_to_clause_ids.setdefault(doc_id, []).append(cid)

        targets = getattr(clause, "targets", None)
        if isinstance(targets, dict):
            for key in ("overrides", "revokes", "defines"):
                raw_vals = targets.get(key) or []
                if not isinstance(raw_vals, list):
                    continue
                for raw in raw_vals:
                    target_id = str(raw).strip()
                    if not target_id or target_id == cid or target_id not in clause_set:
                        continue
                    neighbors.setdefault(cid, set()).add(target_id)
                    neighbors.setdefault(target_id, set()).add(cid)

        terms_used = getattr(clause, "terms_used", None)
        if isinstance(terms_used, list):
            for term in terms_used:
                term_key = str(term).strip()
                if not term_key:
                    continue
                def_clause_id = str(term_definitions.get(term_key, "") or "").strip()
                if not def_clause_id or def_clause_id == cid or def_clause_id not in clause_set:
                    continue
                neighbors.setdefault(cid, set()).add(def_clause_id)
                neighbors.setdefault(def_clause_id, set()).add(cid)
    # Keep same-document clauses lightly connected to avoid disconnected closures when
    # explicit dependency links are sparse.
    # NOTE: only enable this fallback when explicit dependency edges are genuinely sparse;
    # otherwise it can dominate multi-hop closure expansion and wash out hop controls.
    explicit_pairs = sum(len(v) for v in neighbors.values()) // 2
    if explicit_pairs < max(1, len(ordered_ids) // 50):
        for doc_clause_ids in doc_to_clause_ids.values():
            if len(doc_clause_ids) <= 1:
                continue
            for idx in range(1, len(doc_clause_ids)):
                left = doc_clause_ids[idx - 1]
                right = doc_clause_ids[idx]
                neighbors.setdefault(left, set()).add(right)
                neighbors.setdefault(right, set()).add(left)

    return neighbors


def _expand_clause_dependency_closure(
    seed_clause_ids: List[str],
    candidate_clause_ids: List[str],
    world: Any,
    max_hops: int,
    *,
    universe_clause_ids: List[str] | None = None,
) -> List[str]:
    """Expand a dependency closure up to `max_hops` starting from `seed_clause_ids`.

    Key behaviors:
    - Traversal can optionally use a broader `universe_clause_ids` (e.g., all world clauses),
      rather than being restricted to the initial retrieval candidate pool. This makes hop controls
      meaningful even when dependency nodes are not retrieved directly.
    - Output ordering prioritizes: smaller hop distance -> earlier, then original candidate order,
      then stable universe order.
    """
    preferred_order = _unique_strs(candidate_clause_ids)
    ordered_universe = _unique_strs(universe_clause_ids) if universe_clause_ids else list(preferred_order)
    if not ordered_universe:
        return []
    universe_set = set(ordered_universe)

    seeds = [cid for cid in _unique_strs(seed_clause_ids) if cid in universe_set]
    if not seeds:
        return []

    max_hops = max(0, int(max_hops or 0))
    if max_hops <= 0:
        return list(seeds)

    neighbors = _build_clause_dependency_neighbors(ordered_universe, world)

    # BFS with hop distance tracking.
    visited: Set[str] = set(seeds)
    dist: Dict[str, int] = {cid: 0 for cid in seeds}
    frontier: List[str] = list(seeds)
    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        next_frontier: List[str] = []
        for cid in frontier:
            for nid in neighbors.get(cid, set()):
                if nid in visited or nid not in universe_set:
                    continue
                visited.add(nid)
                dist[nid] = hop
                next_frontier.append(nid)
        frontier = next_frontier

    # Stable ordering: by hop distance, then by preferred candidate rank, then by universe rank.
    pref_rank: Dict[str, int] = {cid: idx for idx, cid in enumerate(preferred_order)}
    uni_rank: Dict[str, int] = {cid: idx for idx, cid in enumerate(ordered_universe)}
    closure_ids = sorted(
        (cid for cid in visited),
        key=lambda cid: (
            dist.get(cid, 10**9),
            pref_rank.get(cid, 10**9),
            uni_rank.get(cid, 10**9),
            cid,
        ),
    )
    return closure_ids

def _extract_commit_supporting(
    task: Any,
    opened_ids: List[str],
    world: Any,
    episode_kind: str | None,
) -> List[str]:
    supporting: List[str] = []
    core_ids = list(getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or [])
    opened_set = set(opened_ids or [])
    if episode_kind == "e1_retrieve_rule":
        bridge_id = getattr(task, "bridge_clause_id", None)
        if bridge_id and bridge_id in opened_set:
            supporting.append(bridge_id)
        for cid in core_ids:
            if cid in opened_set and cid not in supporting:
                supporting.append(cid)
                break
    elif episode_kind == "e2_exception_update":
        for cid in core_ids:
            if cid in opened_set and cid not in supporting:
                supporting.append(cid)
    else:
        for cid in core_ids:
            if cid in opened_set and cid not in supporting:
                supporting.append(cid)
    return supporting


def _extract_commit_short_fact(
    task: Any,
    supporting_ids: List[str],
    world: Any,
    episode_kind: str | None,
) -> Dict[str, Any]:
    if not supporting_ids:
        return {}
    if getattr(task, "scenario_mode", "") in {"threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}:
        if episode_kind == "e1_retrieve_rule":
            bridge_id = getattr(task, "bridge_clause_id", None)
            canonical_term = None
            if bridge_id and bridge_id in supporting_ids:
                clause = world.clauses.get(bridge_id)
                if clause and clause.canonical_terms:
                    canonical_term = clause.canonical_terms[0]
            return {"canonical_term": canonical_term} if canonical_term else {}
        if episode_kind == "e2_exception_update":
            return {"exception_exists": True}
        return {}
    if episode_kind == "e1_retrieve_rule":
        bridge_id = getattr(task, "bridge_clause_id", None)
        canonical_term = None
        if bridge_id and bridge_id in supporting_ids:
            clause = world.clauses.get(bridge_id)
            if clause and clause.canonical_terms:
                canonical_term = clause.canonical_terms[0]
        short_fact = {}
        if canonical_term:
            short_fact["canonical_term"] = canonical_term
        core_ids = list(getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or [])
        for cid in supporting_ids:
            if cid in core_ids:
                clause = world.clauses.get(cid)
                if clause and clause.effect:
                    short_fact["base_decision"] = clause.effect.get("decision")
                break
        return short_fact
    if episode_kind == "e2_exception_update":
        for cid in supporting_ids:
            clause = world.clauses.get(cid)
            if clause and clause.applies_if:
                key = next(iter(clause.applies_if.keys()))
                val = clause.applies_if.get(key, [""])[0]
                return {"exception_condition": f"{key}={val}"}
        return {}
    return {}


def _format_commit_refs(commit1: Dict[str, Any] | None, commit2: Dict[str, Any] | None, episode_id: int) -> str:
    lines = []
    if episode_id >= 2:
        lines.append("commit1.fact1")
    if episode_id >= 3:
        lines.append("commit2.fact2")
    if not lines:
        return ""
    return "Commit refs: " + ", ".join(lines)


def _normalize_pivot_message_style(style: Any) -> str:
    val = str(style or "transcript").strip().lower()
    return val if val in {"banner", "transcript"} else "transcript"


def _format_pivot_ticket_message(
    *,
    ticket_initial: str,
    ticket_updated: str,
    assistant_summary: str | None,
    pivot_message_style: str,
) -> str:
    style = _normalize_pivot_message_style(pivot_message_style)
    initial = str(ticket_initial or "").strip()
    updated = str(ticket_updated or "").strip()
    summary = str(assistant_summary or "").strip()

    if style == "banner":
        base = initial
        if summary:
            base = f"{base}\n\n{summary}" if base else summary
        if base:
            return f"{base}\n\nTicket Update (latest user instruction):\n{updated}".strip()
        return f"Ticket Update (latest user instruction):\n{updated}".strip()

    lines = [
        "Conversation so far:",
        f"User: {initial}",
    ]
    if summary:
        lines.append(f"Assistant: {summary}")
    lines.extend(
        [
            f"User: {updated}",
            "Now: produce the final decision.",
        ]
    )
    return "\n".join(lines).strip()


def _build_threaded_prompt(
    ticket: str,
    commit_facts: Dict[str, Any],
    commit_clause_ids: List[str],
    clauses: List[Any] | None = None,
    summary_text: str | None = None,
) -> str:
    header = (
        "You are a policy assistant. Use commit memory to answer the ticket.\n"
        "Use only commit anchor clause_ids as evidence.\n"
        'Return JSON: {"decision":"allow|deny|require_condition|needs_more_info",'
        '"conditions":[...], "evidence":[...], "customer_message":"..."}\n'
    )
    body = ["Ticket:", ticket, ""]
    body.append("Commit memory (masked refs):")
    for key in sorted(commit_facts.keys()):
        val = commit_facts.get(key)
        body.append(f"- {key}: {val if val is not None else 'unknown'}")
    body.append("")
    body.append("Allowed evidence IDs: " + (", ".join(commit_clause_ids) if commit_clause_ids else "none"))
    if summary_text:
        body.append("")
        body.append("Folded summary:")
        body.append(summary_text)
    if clauses:
        body.append("")
        body.append("Context clauses:")
        for clause in clauses:
            body.append(f"[{clause['clause_id']}] {clause['text']}")
    body.append("")
    body.append("Return JSON only.")
    return header + "\n".join(body)


def _apply_context_budget(
    summary_text: str | None,
    clause_ids: List[str],
    world: Any,
    budget_chars: int,
) -> tuple[str | None, List[Dict[str, str]], int, bool, int, int, int]:
    used = 0
    summary = summary_text
    available_rendered: List[tuple[str, str]] = []
    for cid in clause_ids:
        clause = world.clauses.get(cid) if world else None
        if not clause:
            continue
        prefix = f"[{cid}] "
        available_rendered.append((cid, f"{prefix}{clause.text}"))

    total_before = len(summary_text or "") + sum(len(text) for _, text in available_rendered)
    after_chars = 0
    dropped_clause_count = 0
    content_dropped = False
    if summary:
        if len(summary) > budget_chars:
            # Summary itself can be truncated by budget.
            summary = summary[:budget_chars]
            used = len(summary)
            after_chars = used
            dropped_clause_count = len(available_rendered)
            content_dropped = True
            truncated = bool(total_before > budget_chars and content_dropped and after_chars <= budget_chars)
            return summary, [], used, truncated, total_before, after_chars, dropped_clause_count
        used += len(summary)
    clauses: List[Dict[str, str]] = []
    fully_included = 0
    for cid, rendered in available_rendered:
        prefix = f"[{cid}] "
        clause = world.clauses.get(cid) if world else None
        if not clause:
            continue
        if used + len(rendered) > budget_chars:
            remaining = budget_chars - used - len(prefix)
            if remaining > 0 and not clauses:
                trimmed_text = clause.text[:remaining]
                clauses.append({"clause_id": cid, "text": trimmed_text})
                used += len(prefix) + len(trimmed_text)
                content_dropped = True
            content_dropped = True
            break
        clauses.append({"clause_id": cid, "text": clause.text})
        used += len(rendered)
        fully_included += 1
    dropped_clause_count = max(0, len(available_rendered) - fully_included)
    after_chars = len(summary or "") + sum(len(f"[{c['clause_id']}] ") + len(c["text"]) for c in clauses)
    if after_chars < total_before:
        content_dropped = True
    truncated = bool(total_before > budget_chars and content_dropped and after_chars <= budget_chars)
    return summary, clauses, used, truncated, total_before, after_chars, dropped_clause_count


def _inject_litm_filler_clause_ids(
    opened_history_ids: List[str],
    filler_clause_ids: List[str],
    *,
    position: str = "between",
    critical0_id: str | None = None,
    critical1_id: str | None = None,
) -> List[str]:
    base = list(dict.fromkeys(opened_history_ids))
    fillers = [cid for cid in filler_clause_ids if cid and cid not in base]
    if not fillers:
        return base
    pos = str(position or "between")
    if pos == "pre":
        return fillers + base
    if pos == "post":
        return base + fillers
    insert_idx = len(base) // 2
    if critical1_id and critical1_id in base:
        insert_idx = base.index(critical1_id)
    elif critical0_id and critical0_id in base:
        insert_idx = base.index(critical0_id) + 1
    return base[:insert_idx] + fillers + base[insert_idx:]


def _select_goc_unfold_clause_ids(
    opened_history_ids: List[str],
    commit_clause_ids: List[str],
    critical_clause_ids: List[str],
    ticket: str,
    world: Any,
    *,
    activity_filter: bool = False,
    activity_filter_fallback: bool = True,
    mmr_lambda: float = 0.35,
    anchor_top1_lexical: bool = True,
    max_selected: int = 4,
    avoid_clause_ids: List[str] | None = None,
    include_debug: bool = False,
) -> tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    avoid_set = set(_unique_strs(avoid_clause_ids or []))
    opened_history_ids = [cid for cid in _unique_strs(opened_history_ids) if cid not in avoid_set]
    commit_clause_ids = [cid for cid in _unique_strs(commit_clause_ids) if cid not in avoid_set]
    critical_clause_ids = [cid for cid in _unique_strs(critical_clause_ids) if cid not in avoid_set]

    ticket_tokens = set(_tokenize(ticket))
    ticket_lex_tokens = _lexical_tokens(ticket)
    ticket_activity = extract_activity(ticket)
    action_activity = {"retain", "export", "share"}
    activity_filter = bool(activity_filter)
    activity_filter_fallback = bool(activity_filter_fallback)
    mmr_lambda = float(mmr_lambda or 0.35)
    anchor_top1_lexical = bool(anchor_top1_lexical)
    max_selected = int(max_selected or 4)
    max_selected = max(1, max_selected)
    include_debug = bool(include_debug)

    scored: List[tuple[str, float]] = []
    reasons: Dict[str, List[str]] = {}
    candidates: List[Dict[str, Any]] = []
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    for cid in opened_history_ids:
        clause = world.clauses.get(cid) if world else None
        if not clause:
            continue
        score = 0.0
        cid_reasons: List[str] = []
        if cid in commit_clause_ids:
            score += 10.0
            cid_reasons.append("commit_anchor")
        kind = getattr(clause, "kind", None)
        if kind in {"rule", "exception", "update", "procedure"}:
            score += 3.0
            cid_reasons.append("core_kind")
        elif kind in {"priority"}:
            score -= 1.0
            cid_reasons.append("meta")
        elif kind in {"definition", "glossary"}:
            score -= 0.5
            cid_reasons.append("definition")
        tokens = set(_tokenize(clause.text))
        if ticket_tokens & tokens:
            score += 2.0
            cid_reasons.append("ticket_match")
        lexical_score = float(len(ticket_lex_tokens & _lexical_tokens(clause.text)))
        acts = clause_activity(clause.text)
        cand = {
            "cid": cid,
            "base_score": float(score),
            "lexical_score": lexical_score,
            "activity": acts,
            "token_set": _lexical_tokens(clause.text),
        }
        candidates.append(cand)
        candidate_by_id[cid] = cand
        scored.append((cid, score))
        reasons[cid] = cid_reasons

    def _score_key(c: Dict[str, Any]) -> tuple[float, float, str]:
        return (float(c.get("base_score", 0.0)), float(c.get("lexical_score", 0.0)), str(c.get("cid", "")))

    scored.sort(key=lambda item: (-item[1], item[0]))
    ordered = [cid for cid, _ in scored]
    filter_fallback_used = False

    if not activity_filter:
        # Backward-compatible ordering when activity filtering is disabled.
        critical_first = [cid for cid in critical_clause_ids if cid in ordered]
        commit_first = [cid for cid in commit_clause_ids if cid in ordered and cid not in critical_first]
        remaining = [cid for cid in ordered if cid not in critical_first and cid not in commit_first]
        legacy_ordered = critical_first + commit_first + remaining
        debug_payload: Dict[str, Any] = {}
        if include_debug:
            top10 = sorted(candidates, key=lambda c: _score_key(c), reverse=True)[:10]
            debug_payload = {
                "ticket_activity": sorted(ticket_activity),
                "avoid_clause_ids": sorted(avoid_set),
                "selected_activity_summary": [
                    {
                        "clause_id": cid,
                        "activity": sorted(candidate_by_id.get(cid, {}).get("activity", set())),
                    }
                    for cid in legacy_ordered[:max_selected]
                ],
                "top10_candidates_by_base_score": [
                    {"clause_id": c["cid"], "score": float(c["base_score"])}
                    for c in top10
                ],
                "filter_fallback_used": False,
            }
        return legacy_ordered, reasons, debug_payload

    filtered_candidates: List[Dict[str, Any]] = []
    ticket_action = ticket_activity & action_activity
    if ticket_action:
        for cand in candidates:
            if cand["activity"] & ticket_action:
                filtered_candidates.append(cand)
                reasons[cand["cid"]].append("activity_match")
    if not filtered_candidates and activity_filter_fallback:
        filtered_candidates = list(candidates)
        filter_fallback_used = True

    selected_ids: List[str] = []
    working = list(filtered_candidates)
    if anchor_top1_lexical and working:
        anchor = sorted(
            working,
            key=lambda c: (
                float(c.get("lexical_score", 0.0)),
                float(c.get("base_score", 0.0)),
                str(c.get("cid", "")),
            ),
            reverse=True,
        )[0]
        anchor_cid = str(anchor["cid"])
        selected_ids.append(anchor_cid)
        reasons[anchor_cid].append("anchor_top1_lexical")

    while len(selected_ids) < max_selected:
        remaining = [cand for cand in working if cand["cid"] not in selected_ids]
        if not remaining:
            break
        best_cid = None
        best_val = None
        for cand in remaining:
            redundancy = 0.0
            if selected_ids:
                redundancy = max(
                    _token_jaccard(
                        cand.get("token_set", set()),
                        candidate_by_id.get(sel_id, {}).get("token_set", set()),
                    )
                    for sel_id in selected_ids
                    if sel_id in candidate_by_id
                )
            mmr_value = float(cand.get("base_score", 0.0)) - float(mmr_lambda) * redundancy
            tie = (
                mmr_value,
                float(cand.get("lexical_score", 0.0)),
                float(cand.get("base_score", 0.0)),
                str(cand.get("cid", "")),
            )
            if best_val is None or tie > best_val:
                best_val = tie
                best_cid = str(cand["cid"])
        if not best_cid:
            break
        selected_ids.append(best_cid)
        reasons[best_cid].append("mmr_selected")

    if not selected_ids:
        selected_ids = [cid for cid in ordered[:max_selected]]

    debug_payload = {}
    if include_debug:
        top10 = sorted(candidates, key=lambda c: _score_key(c), reverse=True)[:10]
        debug_payload = {
            "ticket_activity": sorted(ticket_activity),
            "avoid_clause_ids": sorted(avoid_set),
            "selected_activity_summary": [
                {
                    "clause_id": cid,
                    "activity": sorted(candidate_by_id.get(cid, {}).get("activity", set())),
                }
                for cid in selected_ids
            ],
            "top10_candidates_by_base_score": [
                {"clause_id": c["cid"], "score": float(c["base_score"])}
                for c in top10
            ],
            "filter_fallback_used": bool(filter_fallback_used),
        }
    return selected_ids, reasons, debug_payload


def _run_shared_topk(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    *,
    primary_top_k: int = 20,
) -> tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    results = env.search(task.user_ticket, top_k=primary_top_k)
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    for item in results:
        if len(opened_ids) >= env.open_budget:
            break
        clause_id = item.get("clause_id")
        if not clause_id:
            continue
        try:
            clause = env.open(clause_id)
        except RuntimeError as exc:
            if "budget" in str(exc):
                break
            continue
        except Exception:
            continue
        opened.append(clause)
        opened_ids.append(clause_id)
    prompt = _build_prompt(task.user_ticket, opened)
    raw_output = client.generate(prompt)
    prediction = _parse_prediction(raw_output)
    diag = {
        "primary_search_results": _enrich_search_results(env, results),
        "primary_search_top_k": primary_top_k,
        "primary_search_query": task.user_ticket,
        "rewrite_used": False,
        "rewrite_queries": [],
        "opened_total_clause_ids": list(opened_ids),
    }
    return prediction, opened_ids, prompt, raw_output, diag


def _is_bridge_clause(clause: Any) -> bool:
    if not clause:
        return False
    return bool(
        getattr(clause, "is_bridge_doc", False)
        or getattr(clause, "bridge_for_slot", None)
        or getattr(clause, "kind", None) in {"definition", "glossary"}
    )


def _is_meta_clause(clause: Any) -> bool:
    if not clause:
        return False
    return getattr(clause, "kind", None) == "priority" or getattr(clause, "slot", None) == "meta"


def _is_rule_clause(clause: Any) -> bool:
    if not clause:
        return False
    return getattr(clause, "kind", None) in {"rule", "exception", "update", "procedure"}


def _apply_preset(args: argparse.Namespace) -> None:
    preset = getattr(args, "preset", None)
    if not preset:
        return
    config = PRESET_CONFIGS.get(preset)
    if not config:
        return
    if hasattr(args, "scenario_mode") and config.get("scenario_mode"):
        current_mode = getattr(args, "scenario_mode", None)
        if current_mode in {None, "", "v0"} or current_mode == config["scenario_mode"]:
            args.scenario_mode = config["scenario_mode"]
    for key in (
        "n_docs",
        "clauses_per_doc",
        "alias_density",
        "canonical_density",
        "exclusive_core_evidence",
        "n_threads",
        "open_budget_e1",
        "open_budget_e2",
        "open_budget_e3",
        "tool_budget_e1",
        "tool_budget_e2",
        "tool_budget_e3",
        "branch_distractor_rate",
        "e3_clause_jitter_max_chars",
        "e3_clause_jitter_max_chars_critical",
        "e3_clause_jitter_max_chars_noncritical",
        "e3_clause_jitter_max_chars_decoy",
        "e3_clause_jitter_scope",
        "e3_litm_filler_count_min",
        "e3_litm_filler_count_max",
        "e3_litm_filler_len_jitter_max",
    ):
        if hasattr(args, key) and config.get(key) is not None:
            current = getattr(args, key)
            should_apply = current is None or isinstance(current, bool)
            if key == "n_threads":
                # For threaded generation, unset n_threads falls back to n_tasks.
                # If caller explicitly set n_tasks (e.g. --n_tasks 60), do not
                # force preset n_threads and unexpectedly inflate run size.
                n_tasks_current = getattr(args, "n_tasks", None) if hasattr(args, "n_tasks") else None
                n_tasks_explicit = n_tasks_current not in {None, 200}
                should_apply = current is None and not n_tasks_explicit
            if key == "e3_clause_jitter_max_chars":
                should_apply = current in {None, 0}
            if key == "e3_clause_jitter_scope":
                should_apply = current in {None, "", "decoy_only"}
            if key in {
                "e3_clause_jitter_max_chars_critical",
                "e3_clause_jitter_max_chars_noncritical",
                "e3_clause_jitter_max_chars_decoy",
                "e3_litm_filler_count_min",
                "e3_litm_filler_count_max",
                "e3_litm_filler_len_jitter_max",
            }:
                should_apply = current in {None, 0}
            if should_apply:
                setattr(args, key, config[key])


def _apply_generation_defaults(args: argparse.Namespace) -> None:
    for key, value in DEFAULT_GENERATION.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)


def _apply_threaded_budget_default(args: argparse.Namespace) -> None:
    scenario_mode = getattr(args, "scenario_mode", "")
    if not (isinstance(scenario_mode, str) and scenario_mode.startswith("threaded_v1_3_fu")):
        return
    if getattr(args, "thread_context_budget_sweep", ""):
        return
    current = getattr(args, "thread_context_budget_chars", None)
    if current is None or current == 8000:
        setattr(args, "thread_context_budget_chars", THREADED_FU_RECOMMENDED_BUDGET)


def _resolve_goc_internal_budgets(args: argparse.Namespace) -> Tuple[int, int, int]:
    gp = _import_goc_policy_module()
    cfg = gp.resolve_internal_budgets_from_namespace(args)
    return int(cfg.active_tokens), int(cfg.unfold_tokens), int(cfg.unfold_k)


def _resolve_goc_unfold_controls(args: argparse.Namespace) -> Tuple[int, int, str]:
    gp = _import_goc_policy_module()
    cfg = gp.resolve_unfold_controls_from_namespace(args)
    return int(cfg.max_nodes), int(cfg.hops), str(cfg.budget_mode)


def _resolve_goc_unfold_policy(args: argparse.Namespace) -> Tuple[str, Tuple[int, int], Tuple[int, int], str]:
    gp = _import_goc_policy_module()
    cfg = gp.resolve_unfold_policy_from_namespace(args)
    return str(cfg.policy), tuple(cfg.default_knobs), tuple(cfg.pivot_knobs), str(cfg.budget_mode)


def quickcheck_compare_report(
    report: Dict[str, Any],
    *,
    label: str = "compare",
    print_fn: Any = print,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def _add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": passed, "details": details})
        if print_fn:
            status = "PASS" if passed else "FAIL"
            print_fn(f"[quickcheck:{label}] {name}: {status} ({details})")

    method_reports = report.get("method_reports", {}) or {}

    # goc_base invariants
    goc_base_report = method_reports.get("goc_base")
    if not goc_base_report:
        _add_check("goc_base_invariants", True, "no goc_base report found")
    else:
        records = goc_base_report.get("records", []) or []
        hop2_nonempty = 0
        union_mismatch = 0
        for rec in records:
            hop2 = rec.get("hop2_search_results")
            if not isinstance(hop2, list) or len(hop2) != 0:
                hop2_nonempty += 1
            hop1_ids = set(_extract_clause_ids(rec.get("hop1_search_results")))
            union_ids = set(_extract_clause_ids(rec.get("search_results_union")))
            if hop1_ids != union_ids:
                union_mismatch += 1
        metrics = goc_base_report.get("metrics", {}) or {}
        hop1_rate = metrics.get("gold_in_search_topk_rate_hop1")
        union_rate = metrics.get("gold_in_search_topk_rate_union")
        rate_ok = (
            isinstance(hop1_rate, (int, float))
            and isinstance(union_rate, (int, float))
            and abs(float(hop1_rate) - float(union_rate)) <= 1e-9
        )
        passed = hop2_nonempty == 0 and union_mismatch == 0 and rate_ok
        details = (
            f"records={len(records)}, hop2_nonempty={hop2_nonempty}, "
            f"union_mismatch={union_mismatch}, rate_match={rate_ok}"
        )
        _add_check("goc_base_invariants", passed, details)

    # Record-level invariant: gold_in_search_topk => gold_in_search_topk_union
    violations = 0
    total_gold = 0
    for report_obj in method_reports.values():
        for rec in report_obj.get("records", []) or []:
            if rec.get("gold_in_search_topk") is True:
                total_gold += 1
                if rec.get("gold_in_search_topk_union") is not True:
                    violations += 1
    _add_check(
        "gold_in_search_topk_implies_union",
        violations == 0,
        f"violations={violations}, checked={total_gold}",
    )

    # Selection metrics sanity
    sel_gap_bad = 0
    rate_bad = 0
    eff_bad = 0
    checked = 0
    for report_obj in method_reports.values():
        for rec in report_obj.get("records", []) or []:
            checked += 1
            gap = rec.get("selection_gap")
            feasible = rec.get("feasible_open_rate")
            realized = rec.get("realized_open_rate")
            eff = rec.get("selection_efficiency")
            if not isinstance(gap, (int, float)) or gap < 0:
                sel_gap_bad += 1
            if not isinstance(feasible, (int, float)) or feasible < 0 or feasible > 1:
                rate_bad += 1
            if not isinstance(realized, (int, float)) or realized < 0 or realized > 1:
                rate_bad += 1
            if isinstance(feasible, (int, float)) and feasible == 0:
                eff_ok = eff is None or (isinstance(eff, float) and math.isnan(eff))
                if not eff_ok:
                    eff_bad += 1
            else:
                if not isinstance(eff, (int, float)) or (isinstance(eff, float) and math.isnan(eff)):
                    eff_bad += 1
    sel_pass = sel_gap_bad == 0 and rate_bad == 0 and eff_bad == 0
    sel_details = (
        f"records={checked}, gap_bad={sel_gap_bad}, rate_bad={rate_bad}, eff_bad={eff_bad}"
    )
    _add_check("selection_metrics_sanity", sel_pass, sel_details)

    # Judge metrics present
    if report.get("judge") in {
        "symbolic",
        "symbolic_packed",
        "symbolic_packed_allcritical",
        "symbolic_full_episode",
    }:
        missing_record = 0
        missing_metric = 0
        for method, report_obj in method_reports.items():
            metrics = report_obj.get("metrics", {}) or {}
            if not isinstance(metrics.get("judge_accuracy"), (int, float)):
                missing_metric += 1
            for rec in report_obj.get("records", []) or []:
                if rec.get("judge_decision") is None or rec.get("judge_correct") is None:
                    missing_record += 1
        passed = missing_record == 0 and missing_metric == 0
        details = f"missing_record={missing_record}, missing_metric={missing_metric}"
        _add_check("judge_metrics_present", passed, details)
    else:
        _add_check("judge_metrics_present", True, "judge not symbolic")

    # Acc-no-core and judge-supporting metrics present (summary)
    missing_acc = 0
    missing_support = 0
    acc_details = []
    support_details = []
    for method, report_obj in method_reports.items():
        metrics = report_obj.get("metrics", {}) or {}
        if "acc_no_core_evidence_rate" not in metrics:
            missing_acc += 1
        acc_details.append(f"{method}={metrics.get('acc_no_core_evidence_rate')}")
        support_keys = [
            "judge_supporting_count_mean",
            "judge_used_any_core_rate",
            "judge_used_any_bridge_rate",
            "judge_used_any_critical_core_rate",
            "judge_used_any_decoy_rate",
        ]
        if any(k not in metrics for k in support_keys):
            missing_support += 1
        support_details.append(
            f"{method}:count_mean={metrics.get('judge_supporting_count_mean')},"
            f"core_rate={metrics.get('judge_used_any_core_rate')},"
            f"bridge_rate={metrics.get('judge_used_any_bridge_rate')},"
            f"critical_rate={metrics.get('judge_used_any_critical_core_rate')},"
            f"decoy_rate={metrics.get('judge_used_any_decoy_rate')}"
        )
    _add_check(
        "acc_no_core_evidence_rate_present",
        missing_acc == 0,
        "; ".join(acc_details),
    )
    _add_check(
        "judge_supporting_metrics_present",
        missing_support == 0,
        "; ".join(support_details),
    )

    deep_missing = 0
    open_missing = 0
    deep_details = []
    open_count_details = []
    for method, report_obj in method_reports.items():
        metrics = report_obj.get("metrics", {}) or {}
        deep_val = metrics.get("deep_rank_core_rate")
        if not isinstance(deep_val, (int, float)):
            deep_missing += 1
        deep_details.append(f"{method}={deep_val}")
        bridge_mean = metrics.get("opened_bridge_count_mean")
        rule_mean = metrics.get("opened_rule_count_mean")
        if not isinstance(bridge_mean, (int, float)) or not isinstance(rule_mean, (int, float)):
            open_missing += 1
        open_count_details.append(
            f"{method}:bridge_mean={bridge_mean},rule_mean={rule_mean}"
        )
    _add_check(
        "deep_rank_core_rate_present",
        deep_missing == 0,
        "; ".join(deep_details),
    )
    _add_check(
        "opened_counts_present",
        open_missing == 0,
        "; ".join(open_count_details),
    )

    if report.get("judge") == "symbolic":
        ub_details = []
        for method, report_obj in method_reports.items():
            metrics = report_obj.get("metrics", {}) or {}
            ub_details.append(
                f"{method}={metrics.get('selection_upper_bound_judge_acc')}"
            )
        _add_check(
            "selection_upper_bound_present",
            True,
            "; ".join(ub_details),
        )

    # Threaded metrics (v1.2 / v1.3)
    scenario_mode_param = report.get("scenario_params", {}).get("scenario_mode")
    threaded_mode = isinstance(scenario_mode_param, str) and scenario_mode_param.startswith(
        "threaded_v1_"
    )
    if not threaded_mode:
        for report_obj in method_reports.values():
            if any(
                isinstance(rec.get("scenario_mode"), str)
                and rec.get("scenario_mode").startswith("threaded_v1_")
                for rec in report_obj.get("records", []) or []
            ):
                threaded_mode = True
                break
    if threaded_mode:
        missing_thread_metrics = 0
        missing_episode_metrics = 0
        missing_thread_records = 0
        for method, report_obj in method_reports.items():
            metrics = report_obj.get("metrics", {}) or {}
            if metrics.get("thread_judge_accuracy") is None:
                missing_thread_metrics += 1
            required_episode = [
                "episode_judge_accuracy_e1",
                "episode_judge_accuracy_e2",
                "episode_judge_accuracy_e3",
            ]
            if any(k not in metrics for k in required_episode):
                missing_episode_metrics += 1
            if not report_obj.get("thread_records"):
                missing_thread_records += 1
        _add_check(
            "thread_metrics_present",
            missing_thread_metrics == 0,
            f"missing={missing_thread_metrics}",
        )
        _add_check(
            "episode_metrics_present",
            missing_episode_metrics == 0,
            f"missing={missing_episode_metrics}",
        )
        _add_check(
            "thread_records_present",
            missing_thread_records == 0,
            f"missing={missing_thread_records}",
        )

    # Threaded v1.3 FU context budget fields
    threaded_fu = isinstance(scenario_mode_param, str) and scenario_mode_param.startswith(
        "threaded_v1_3_fu"
    )
    if not threaded_fu:
        for report_obj in method_reports.values():
            if any(
                isinstance(rec.get("scenario_mode"), str)
                and rec.get("scenario_mode").startswith("threaded_v1_3_fu")
                for rec in report_obj.get("records", []) or []
            ):
                threaded_fu = True
                break
    if threaded_fu:
        missing_metric = 0
        missing_record = 0
        for method, report_obj in method_reports.items():
            metrics = report_obj.get("metrics", {}) or {}
            if "e3_packed_all_critical_rate" not in metrics:
                missing_metric += 1
            for rec in report_obj.get("records", []) or []:
                if rec.get("episode_id") != 3:
                    continue
                for key in [
                    "e3_context_budget_chars",
                    "e3_context_chars_used",
                    "e3_context_clause_count",
                    "e3_packed_clause_count",
                    "e3_context_token_est",
                    "e3_packed_token_est",
                    "e3_context_truncated",
                    "e3_packed_total_chars_before",
                    "e3_packed_total_chars_after",
                    "e3_packed_truncated",
                    "e3_packed_dropped_clause_count",
                    "e3_packed_clause_lens",
                    "e3_packed_clause_is_critical",
                    "e12_opened_clause_ids",
                    "full_episode_clause_ids",
                    "e3_packed_all_critical",
                    "e3_packed_any_critical",
                    "e3_packed_all_critical_full_episode",
                    "e3_packed_any_critical_full_episode",
                    "e3_litm_filler_count",
                    "e3_litm_filler_position",
                ]:
                    if key not in rec:
                        missing_record += 1
                        break
        _add_check(
            "threaded_fu_context_present",
            missing_metric == 0 and missing_record == 0,
            f"missing_metric={missing_metric}, missing_record={missing_record}",
        )
        truncation_bad = 0
        dropped_bad = 0
        checked_records = 0
        for report_obj in method_reports.values():
            for rec in report_obj.get("records", []) or []:
                if rec.get("episode_id") != 3:
                    continue
                before = rec.get("e3_packed_total_chars_before")
                truncated = rec.get("e3_packed_truncated")
                dropped = rec.get("e3_packed_dropped_clause_count")
                if not isinstance(before, (int, float)) or not isinstance(truncated, bool):
                    continue
                checked_records += 1
                if before <= (rec.get("e3_context_budget_chars") or 0) and truncated:
                    truncation_bad += 1
                if truncated is False and isinstance(dropped, (int, float)) and int(dropped) != 0:
                    dropped_bad += 1
        _add_check(
            "e3_truncation_metric_sanity",
            truncation_bad == 0 and dropped_bad == 0,
            f"checked={checked_records}, truncation_bad={truncation_bad}, dropped_bad={dropped_bad}",
        )
        full_episode_list_bad = 0
        full_episode_subset_bad = 0
        full_episode_checked = 0
        for report_obj in method_reports.values():
            for rec in report_obj.get("records", []) or []:
                if rec.get("episode_id") != 3:
                    continue
                packed_ids = rec.get("e3_packed_clause_ids")
                full_ids = rec.get("full_episode_clause_ids")
                if not isinstance(packed_ids, list) or not isinstance(full_ids, list):
                    full_episode_list_bad += 1
                    continue
                full_episode_checked += 1
                if not set(packed_ids).issubset(set(full_ids)):
                    full_episode_subset_bad += 1
        _add_check(
            "full_episode_clause_ids_sane",
            full_episode_list_bad == 0 and full_episode_subset_bad == 0,
            (
                f"checked={full_episode_checked}, "
                f"list_bad={full_episode_list_bad}, subset_bad={full_episode_subset_bad}"
            ),
        )
        if report.get("judge") == "symbolic_full_episode":
            compare_non_decreasing = 0
            compare_checked = 0
            for method, report_obj in method_reports.items():
                metrics = report_obj.get("metrics", {}) or {}
                full_acc = metrics.get("e3_judge_accuracy_full_episode")
                packed_acc = metrics.get("e3_judge_accuracy_packed")
                if not isinstance(full_acc, (int, float)) or not isinstance(packed_acc, (int, float)):
                    continue
                compare_checked += 1
                if float(full_acc) + 1e-9 >= float(packed_acc):
                    compare_non_decreasing += 1
            if compare_checked == 0:
                _add_check(
                    "full_episode_acc_vs_packed",
                    True,
                    "no paired metrics to compare in this run",
                )
            else:
                ratio = compare_non_decreasing / compare_checked
                _add_check(
                    "full_episode_acc_vs_packed",
                    ratio >= 0.8,
                    (
                        f"paired={compare_checked}, non_decreasing={compare_non_decreasing}, "
                        f"ratio={ratio:.3f}"
                    ),
                )
        jitter_max = report.get("scenario_params", {}).get("e3_clause_jitter_max_chars")
        jitter_critical = report.get("scenario_params", {}).get(
            "e3_clause_jitter_max_chars_critical"
        )
        jitter_noncritical = report.get("scenario_params", {}).get(
            "e3_clause_jitter_max_chars_noncritical"
        )
        jitter_decoy = report.get("scenario_params", {}).get(
            "e3_clause_jitter_max_chars_decoy"
        )
        jitter_scope = report.get("scenario_params", {}).get("e3_clause_jitter_scope")
        litm_filler_min = report.get("scenario_params", {}).get("e3_litm_filler_count_min")
        litm_filler_max = report.get("scenario_params", {}).get("e3_litm_filler_count_max")
        litm_filler_len_jitter = report.get("scenario_params", {}).get(
            "e3_litm_filler_len_jitter_max"
        )
        jitter_ok = (
            isinstance(jitter_max, int)
            and isinstance(jitter_critical, int)
            and isinstance(jitter_noncritical, int)
            and isinstance(jitter_decoy, int)
            and jitter_scope in {
            "decoy_only",
            "decoy_plus_noncritical",
            "all",
            }
        )
        _add_check(
            "jitter_params_present",
            jitter_ok,
            (
                f"e3_clause_jitter_max_chars={jitter_max}, "
                f"e3_clause_jitter_max_chars_critical={jitter_critical}, "
                f"e3_clause_jitter_max_chars_noncritical={jitter_noncritical}, "
                f"e3_clause_jitter_max_chars_decoy={jitter_decoy}, "
                f"e3_clause_jitter_scope={jitter_scope}"
            ),
        )
        litm_ok = (
            isinstance(litm_filler_min, int)
            and isinstance(litm_filler_max, int)
            and isinstance(litm_filler_len_jitter, int)
            and litm_filler_min >= 0
            and litm_filler_max >= litm_filler_min
        )
        _add_check(
            "litm_filler_params_present",
            litm_ok,
            (
                f"e3_litm_filler_count_min={litm_filler_min}, "
                f"e3_litm_filler_count_max={litm_filler_max}, "
                f"e3_litm_filler_len_jitter_max={litm_filler_len_jitter}"
            ),
        )
        if litm_ok and litm_filler_max > litm_filler_min:
            filler_vals: List[int] = []
            for report_obj in method_reports.values():
                for rec in report_obj.get("records", []) or []:
                    if rec.get("episode_id") != 3:
                        continue
                    val = rec.get("e3_litm_filler_count")
                    if isinstance(val, int):
                        filler_vals.append(val)
            unique_vals = sorted(set(filler_vals))
            in_range = all(litm_filler_min <= v <= litm_filler_max for v in unique_vals)
            diverse = len(unique_vals) >= 3
            _add_check(
                "litm_filler_distribution",
                in_range and diverse,
                f"n_unique={len(unique_vals)}, unique_values={unique_vals[:12]}",
            )

    if isinstance(scenario_mode_param, str) and scenario_mode_param.startswith(
        "threaded_v1_3_fu_decoy"
    ):
        requested = report.get("scenario_params", {}).get("n_threads_requested")
        final = report.get("scenario_params", {}).get("n_threads_generated_final")
        if isinstance(requested, int) and isinstance(final, int):
            _add_check(
                "thread_count_exact_generation",
                requested == final,
                f"requested={requested}, final={final}",
            )

    # Shared open policy sanity (threaded)
    if report.get("scenario_params", {}).get("thread_open_policy") == "shared_topk":
        mismatches = 0
        total = 0
        ref_by_key: Dict[tuple[str, int], List[str]] = {}
        for report_obj in method_reports.values():
            for rec in report_obj.get("records", []) or []:
                if rec.get("episode_id") not in {1, 2}:
                    continue
                thread_id = rec.get("thread_id")
                if not thread_id:
                    continue
                key = (thread_id, int(rec.get("episode_id")))
                opened = sorted(rec.get("opened_clause_ids") or [])
                if key not in ref_by_key:
                    ref_by_key[key] = opened
                else:
                    total += 1
                    if opened != ref_by_key[key]:
                        mismatches += 1
        _add_check(
            "shared_open_policy_consistency",
            mismatches == 0,
            f"checked={total}, mismatches={mismatches}",
        )

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def quickcheck_sweep_summary(
    rows: List[Dict[str, Any]],
    *,
    judge_mode: str = "llm",
    label: str = "sweep",
    print_fn: Any = print,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def _add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": passed, "details": details})
        if print_fn:
            status = "PASS" if passed else "FAIL"
            print_fn(f"[quickcheck:{label}] {name}: {status} ({details})")

    if not rows:
        _add_check("summary_rows_present", False, "no summary rows")
        return {"passed": False, "checks": checks}
    _add_check("summary_rows_present", True, f"rows={len(rows)}")

    required = [
        "feasible_open_rate",
        "realized_open_rate",
        "selection_gap",
        "selection_efficiency",
    ]
    if isinstance(judge_mode, str) and judge_mode.startswith("symbolic"):
        required.append("judge_accuracy")

    missing = 0
    for row in rows:
        for key in required:
            if key not in row:
                missing += 1
                break
    _add_check("summary_required_keys", missing == 0, f"missing_rows={missing}")

    # Reuse selection sanity checks on aggregated rows.
    sel_gap_bad = 0
    rate_bad = 0
    eff_bad = 0
    for row in rows:
        gap = row.get("selection_gap")
        feasible = row.get("feasible_open_rate")
        realized = row.get("realized_open_rate")
        eff = row.get("selection_efficiency")
        if not isinstance(gap, (int, float)) or gap < 0:
            sel_gap_bad += 1
        if not isinstance(feasible, (int, float)) or feasible < 0 or feasible > 1:
            rate_bad += 1
        if not isinstance(realized, (int, float)) or realized < 0 or realized > 1:
            rate_bad += 1
        if isinstance(feasible, (int, float)) and feasible == 0:
            eff_ok = eff is None or (isinstance(eff, float) and math.isnan(eff))
            if not eff_ok:
                eff_bad += 1
        else:
            if not isinstance(eff, (int, float)) or (isinstance(eff, float) and math.isnan(eff)):
                eff_bad += 1
    sel_pass = sel_gap_bad == 0 and rate_bad == 0 and eff_bad == 0
    sel_details = (
        f"rows={len(rows)}, gap_bad={sel_gap_bad}, rate_bad={rate_bad}, eff_bad={eff_bad}"
    )
    _add_check("summary_selection_metrics_sanity", sel_pass, sel_details)

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _resolve_controller_state_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_rerank_weights_path(base_dir: Path, path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        return path if path.is_absolute() else base_dir / path
    return base_dir / "runs" / "controller" / "weights.json"


def _apply_open_budget(tasks: List[Any], open_budget: int) -> List[Any]:
    cloned = copy.deepcopy(tasks)
    for task in cloned:
        task.budgets["open_budget"] = open_budget
    return cloned


def _split_tasks(
    tasks: List[Any],
    task_split: str,
    train_ratio: float,
    split_seed: int,
) -> tuple[List[Any], List[Any]]:
    if task_split != "holdout":
        return [], list(tasks)
    rng = __import__("random").Random(split_seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    candidates: List[str] = [raw]
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        end_idx: int | None = None
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
        if end_idx is not None and end_idx > 0:
            inner = "\n".join(lines[1:end_idx]).strip()
            if inner:
                candidates.append(inner)
    no_fence_lines = [line for line in lines if not line.strip().startswith("```")]
    if no_fence_lines:
        without_fences = "\n".join(no_fence_lines).strip()
        if without_fences:
            candidates.append(without_fences)

    decoder = json.JSONDecoder()
    seen: Set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _parse_prediction_for_answer_metrics(
    raw_output: str,
) -> tuple[Dict[str, Any], bool]:
    extracted = _extract_first_json_object(raw_output)
    if isinstance(extracted, dict):
        return _parse_prediction(json.dumps(extracted, ensure_ascii=False)), False

    raw = str(raw_output or "").strip()
    fallback_obj: Dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            fallback_obj = parsed
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end >= start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    fallback_obj = parsed
            except Exception:
                fallback_obj = None

    if isinstance(fallback_obj, dict):
        return _parse_prediction(json.dumps(fallback_obj, ensure_ascii=False)), False

    # Preserve legacy behavior for downstream fields while exposing parse failure.
    return _parse_prediction(raw_output), True

def apply_evidence_padding(
    pred: Dict[str, Any],
    opened_ids: List[str],
    mode: str,
    min_count: int,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    raw_evidence = list(pred.get("evidence", []) or [])
    if opened_ids:
        filtered = []
        seen: set[str] = set()
        for cid in raw_evidence:
            if cid in opened_ids and cid not in seen:
                filtered.append(cid)
                seen.add(cid)
    else:
        filtered = raw_evidence

    if mode == "none":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = filtered
        return pred_for_eval, dict(pred_for_eval), filtered, filtered

    evidence_after = list(filtered)
    if opened_ids:
        evidence_after = _ensure_min_evidence(evidence_after, opened_ids, min_count=min_count)

    if mode == "schema_only":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = filtered
        pred_for_record = dict(pred)
        pred_for_record["evidence"] = evidence_after
        return pred_for_eval, pred_for_record, filtered, evidence_after

    if mode == "global":
        pred_for_eval = dict(pred)
        pred_for_eval["evidence"] = evidence_after
        return pred_for_eval, dict(pred_for_eval), filtered, evidence_after

    raise ValueError(f"Unknown evidence padding mode: {mode}")


def normalize_prediction_schema(
    prediction: Dict[str, Any],
) -> tuple[Dict[str, Any], str | None, str | None, str | None]:
    allowed = {"allow", "deny", "require_condition", "needs_more_info"}
    decision_before = prediction.get("decision")
    conditions = prediction.get("conditions")
    if conditions is None:
        conditions = prediction.get("required_conditions") or []
    if not isinstance(conditions, list):
        conditions = []
    decision = decision_before if decision_before in allowed else "needs_more_info"
    normalize_reason = None
    if conditions and decision in {"allow", "deny", "needs_more_info"}:
        decision = "require_condition"
        normalize_reason = "reqconds_nonempty_force_require_condition"
    elif decision == "require_condition" and not conditions:
        decision = "needs_more_info"
        normalize_reason = "reqconds_empty_force_needs_more_info"
        if not prediction.get("customer_message"):
            prediction["customer_message"] = "Need required conditions."
    prediction["decision"] = decision
    prediction["conditions"] = conditions
    decision_after = decision
    return prediction, normalize_reason, decision_before, decision_after


def _evaluate_method(
    method: str,
    world: Any,
    tasks: List[Any],
    args: argparse.Namespace,
    client: LLMClient,
    run_dir: Path,
    run_id: str | None = None,
    controller: Controller | RerankController | None = None,
    controller_mode: str = "off",
    controller_policy: str = "bandit",
    llm_backend: str = "dummy",
    client_class: str = "DummyClient",
    resolved_model: str = "dummy",
) -> Dict[str, Any]:
    metrics: List[Dict[str, float]] = []
    tool_calls: List[int] = []
    open_calls: List[int] = []
    prompt_tokens_list: List[int] = []
    records: List[Dict[str, Any]] = []
    controller_actions: Dict[str, int] = {}
    thread_state_by_id: Dict[str, Dict[str, Any]] = {}
    thread_records: List[Dict[str, Any]] = []
    episode_judge: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    episode_commit: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    episode_cov_core: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    (
        goc_internal_budget_active_tokens,
        goc_internal_budget_unfold_tokens,
        goc_internal_unfold_k,
    ) = _resolve_goc_internal_budgets(args)
    (
        _goc_unfold_max_nodes_fixed,
        _goc_unfold_hops_fixed,
        _goc_unfold_budget_mode_fixed,
    ) = _resolve_goc_unfold_controls(args)
    (
        goc_unfold_policy,
        (goc_unfold_default_max_nodes_cfg, goc_unfold_default_hops_cfg),
        (goc_unfold_pivot_max_nodes_cfg, goc_unfold_pivot_hops_cfg),
        goc_unfold_budget_mode_cfg,
    ) = _resolve_goc_unfold_policy(args)
    # For backward compatibility in debug/diag, keep 'cfg' as the default knobs.
    goc_unfold_max_nodes_cfg = int(goc_unfold_default_max_nodes_cfg)
    goc_unfold_hops_cfg = int(goc_unfold_default_hops_cfg)
    save_event_trace = bool(getattr(args, "save_event_trace", False))
    event_trace_sample_rate = float(getattr(args, "event_trace_sample_rate", 1.0) or 1.0)
    event_trace_sample_rate = max(0.0, min(1.0, event_trace_sample_rate))
    event_trace_dir_arg = str(getattr(args, "event_trace_dir", "") or "").strip()
    event_trace_root = Path(event_trace_dir_arg) if event_trace_dir_arg else (run_dir / "event_traces")
    event_trace_run_id = run_id or run_dir.name
    task_iterable: List[Any] = list(tasks)
    parallel_workers = max(1, int(getattr(args, "parallel_workers", 1) or 1))
    threaded_modes = {"threaded_v1_2", "threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}
    tasks_are_threaded = bool(
        any(getattr(t, "thread_id", None) for t in tasks)
        or any(getattr(t, "scenario_mode", getattr(args, "scenario_mode", "v0")) in threaded_modes for t in tasks)
    )
    if parallel_workers > 1 and len(tasks) > 1:
        if controller_mode == "train":
            raise ValueError(
                "parallel_workers > 1 is not supported with controller_mode=train "
                "(controller updates must run sequentially)."
            )
        if (
            method == "goc"
            and bool(getattr(args, "save_goc_graph", False))
            and str(getattr(args, "goc_graph_dir", "") or "").endswith(".jsonl")
        ):
            raise ValueError(
                "parallel_workers > 1 is not supported with --goc_graph_dir ending in .jsonl "
                "(shared graph jsonl file would race). Use per-task graph output directory."
            )

        grouped_tasks: List[Tuple[str, List[Any]]] = []
        if tasks_are_threaded:
            grouped: Dict[str, List[Any]] = {}
            order: List[str] = []
            for task in tasks:
                key = str(getattr(task, "thread_id", None) or f"__task__:{task.task_id}")
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(task)
            for key in order:
                grouped_tasks.append(
                    (
                        key,
                        sorted(
                            grouped[key],
                            key=lambda t: (
                                int(getattr(t, "episode_id", 0) or 0),
                                str(getattr(t, "task_id", "")),
                            ),
                        ),
                    )
                )
        else:
            grouped_tasks = [(str(task.task_id), [task]) for task in tasks]

        def _run_group(task_group: List[Any]) -> Dict[str, Any]:
            local_args = argparse.Namespace(**vars(args))
            local_args.parallel_workers = 1
            return _evaluate_method(
                method,
                world,
                task_group,
                local_args,
                client,
                run_dir,
                run_id=run_id,
                controller=controller,
                controller_mode=controller_mode,
                controller_policy=controller_policy,
                llm_backend=llm_backend,
                client_class=client_class,
                resolved_model=resolved_model,
            )

        ordered_reports: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            fut_to_idx = {
                executor.submit(_run_group, task_group): idx
                for idx, (_key, task_group) in enumerate(grouped_tasks)
            }
            tmp_reports: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                tmp_reports[idx] = fut.result()
            ordered_reports = [tmp_reports[idx] for idx in range(len(grouped_tasks))]

        merged_records: List[Dict[str, Any]] = []
        merged_thread_records: List[Dict[str, Any]] = []
        merged_controller_actions: Dict[str, int] = {}
        for rep in ordered_reports:
            merged_records.extend(list(rep.get("records") or []))
            merged_thread_records.extend(list(rep.get("thread_records") or []))
            for action, count in (rep.get("controller_actions_distribution") or {}).items():
                merged_controller_actions[action] = merged_controller_actions.get(action, 0) + int(count or 0)

        task_pos = {str(task.task_id): idx for idx, task in enumerate(tasks)}
        merged_records.sort(key=lambda r: task_pos.get(str(r.get("task_id", "")), 10**9))
        merged_thread_records.sort(
            key=lambda r: (
                str(r.get("thread_id") or ""),
                int(r.get("episode_id") or 0),
                task_pos.get(str(r.get("task_id", "")), 10**9),
            )
        )

        records = merged_records
        thread_records = merged_thread_records
        controller_actions = merged_controller_actions
        for rec in records:
            episode_id = rec.get("episode_id")
            if not rec.get("thread_id") or episode_id not in {1, 2, 3}:
                continue
            if isinstance(rec.get("judge_correct"), bool):
                episode_judge[int(episode_id)].append(1.0 if rec.get("judge_correct") else 0.0)
            if isinstance(rec.get("commit_correct"), bool):
                episode_commit[int(episode_id)].append(1.0 if rec.get("commit_correct") else 0.0)
            if isinstance(rec.get("opened_gold_coverage_core"), (int, float)):
                episode_cov_core[int(episode_id)].append(float(rec.get("opened_gold_coverage_core") or 0.0))
        metrics = [_task_metrics_from_record(rec) for rec in records]
        tool_calls = [int(rec.get("tool_calls") or 0) for rec in records]
        open_calls = [int(rec.get("open_calls") or 0) for rec in records]
        prompt_tokens_list = [int(rec.get("prompt_tokens") or 0) for rec in records]
        task_iterable = []

    total_tasks = len(task_iterable)
    for task_idx, task in enumerate(task_iterable, start=1):
        if total_tasks and (task_idx == 1 or task_idx % 25 == 0 or task_idx == total_tasks):
            print(
                f"[eval:{method}] task {task_idx}/{total_tasks} ({task.task_id})",
                flush=True,
            )
        scenario_mode = getattr(task, "scenario_mode", getattr(args, "scenario_mode", "v0"))
        threaded_mode = scenario_mode in {"threaded_v1_2", "threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}
        thread_id = getattr(task, "thread_id", None)
        episode_id = getattr(task, "episode_id", None)
        episode_kind = getattr(task, "episode_kind", None)
        thread_state: Dict[str, Any] | None = None
        if threaded_mode and thread_id:
            thread_state = thread_state_by_id.setdefault(
                thread_id,
                {
                    "commit1": None,
                    "commit2": None,
                    "opened_history_ids": [],
                    "episode_records": {},
                    "initial_ticket_node_id": None,
                    "pivot_ticket_node_ids": [],
                    "avoids_edges": [],
                    "initial_context": None,
                    "state_nodes": {},
                    "state_edges": [],
                    "pivot_context_delta": {},
                    "pivot_override_keys": [],
                },
            )
        search_score_mode = getattr(args, "search_score_mode", "bm25_plus_bridge_bonus")
        bridge_bonus = float(getattr(args, "bridge_bonus", 1.5))
        env = PolicyOpsEnv(
            world,
            tool_call_budget=task.budgets.get("tool_call_budget", 50),
            open_budget=task.budgets.get("open_budget", 5),
            search_score_mode=search_score_mode,
            bridge_bonus=bridge_bonus,
        )
        error: str | None = None
        raw_output: str | None = None
        controller_action: str | None = None
        diag: Dict[str, Any] = {}
        oracle_meta: Dict[str, Any] = {}
        goc_graph: GoCGraph | None = None
        goc_graph_path: Path | None = None
        goc_graph_task_path: Path | None = None
        goc_graph_dot_path: Path | None = None
        goc_internal_graph_task_path: Path | None = None
        goc_internal_snapshots: List[Dict[str, Any]] = []
        log_graph = False
        if method == "goc" and args.save_goc_graph:
            if args.goc_graph_sample_rate >= 1.0:
                log_graph = True
            elif args.goc_graph_sample_rate <= 0.0:
                log_graph = False
            else:
                digest = hashlib.md5(task.task_id.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
                log_graph = bucket <= float(args.goc_graph_sample_rate)
        if log_graph:
            goc_graph = GoCGraph(goc_graph_version=args.goc_graph_schema)
            goc_internal_graph_task_path = run_dir / "graphs_internal" / f"{task.task_id}.jsonl"
            if args.goc_graph_dir and args.goc_graph_dir.endswith(".jsonl"):
                goc_graph_path = Path(args.goc_graph_dir)
                goc_graph_task_path = run_dir / "graphs" / f"{task.task_id}.jsonl"
            else:
                goc_graph_task_path = (
                    Path(args.goc_graph_dir)
                    if args.goc_graph_dir
                    else (run_dir / "graphs")
                ) / f"{task.task_id}.jsonl"
                goc_graph_path = goc_graph_task_path
        agent_query_policy = getattr(args, "agent_query_policy", "single_hop")
        shared_open_policy = getattr(args, "thread_open_policy", "current")
        task_for_run = task
        ticket_initial = str(getattr(task, "ticket_initial", None) or task.user_ticket or "")
        ticket_updated = str(getattr(task, "ticket_updated", None) or "").strip()
        pivot_type = str(getattr(task, "pivot_type", None) or "")
        is_pivot_task = bool(ticket_updated)
        pivot_gold_mode = _normalize_pivot_gold_mode(
            getattr(args, "pivot_gold_mode", "respect_ticket_updated")
        )
        pivot_message_style = _normalize_pivot_message_style(
            getattr(args, "pivot_message_style", "transcript")
        )
        goc_enable_avoids = bool(getattr(args, "goc_enable_avoids", True))
        goc_avoids_mode = _resolve_goc_avoids_mode(args) if method == "goc" else "off"
        goc_avoids_edge_injected = False
        goc_initial_ticket_node_id: str | None = None
        goc_pivot_ticket_node_id: str | None = None
        goc_avoid_target_clause_ids: List[str] = []
        goc_avoided_node_injected: bool | None = None
        goc_inapplicable_clause_ids: List[str] = []
        pivot_override_keys: List[str] = []
        pivot_context_delta: Dict[str, Dict[str, Any]] = {}
        critical_coverage_e3: float | None = None
        critical_missing_ids: List[str] = []
        eval_critical_core_clause_ids: List[str] = []
        inapplicable_injected_rate_e3: float | None = None
        inapplicable_injected_clause_ids: List[str] = []
        goc_effective_context, pivot_constraint_updates = _compute_effective_context_with_constraint_updates(
            task,
            int(episode_id or 0),
            threaded_mode=bool(threaded_mode),
            thread_state=thread_state,
        )
        pivot_constraint_key, pivot_constraint_value = _constraint_key_value(
            pivot_constraint_updates
        )
        pivot_constraint_keys = sorted(
            [str(k) for k in (pivot_constraint_updates or {}).keys()]
        )
        goc_initial_context = dict(getattr(task, "context", None) or {})
        eval_gold: Gold = task.gold
        pivot_gold: Gold | None = None
        pivot_gold_constraint_applied = False
        eval_gold_is_pivot = False
        use_pivot_gold_eval = bool(
            int(episode_id or 0) == 3
            and bool(is_pivot_task)
            and pivot_gold_mode in {"respect_ticket_updated", "both"}
        )
        eval_context_for_judge = (
            dict(goc_effective_context)
            if use_pivot_gold_eval and isinstance(goc_effective_context, dict)
            else (dict(getattr(task, "context", None) or {}) if isinstance(getattr(task, "context", None), dict) else {})
        )
        orig_task_metrics: Dict[str, float] | None = None
        e3_evidence_valid_in_context: bool | None = None
        e3_evidence_valid_in_commits: bool | None = None
        state_nodes_e1_by_key: Dict[str, str] = {}
        state_node_meta_e1: Dict[str, Dict[str, Any]] = {}
        state_nodes_e3_by_key: Dict[str, str] = {}
        state_node_meta_e3: Dict[str, Dict[str, Any]] = {}
        if threaded_mode and thread_state is not None:
            if not isinstance(thread_state.get("initial_context"), dict):
                thread_state["initial_context"] = dict(getattr(task, "context", None) or {})
            goc_initial_context = dict(thread_state.get("initial_context") or {})
            if int(episode_id or 0) == 1:
                state_nodes_e1_by_key, state_node_meta_e1 = _build_state_node_maps(goc_initial_context, 1)
                ts_nodes = thread_state.setdefault("state_nodes", {})
                if isinstance(ts_nodes, dict):
                    ts_nodes.update(state_node_meta_e1)
            if int(episode_id or 0) == 3 and is_pivot_task:
                state_nodes_e3_by_key, state_node_meta_e3 = _build_state_node_maps(goc_effective_context, 3)
                ts_nodes = thread_state.setdefault("state_nodes", {})
                if isinstance(ts_nodes, dict):
                    ts_nodes.update(state_node_meta_e3)
                pivot_context_delta = _compute_context_delta(goc_initial_context, goc_effective_context)
                pivot_override_keys = sorted(pivot_context_delta.keys())
                thread_state["pivot_context_delta"] = dict(pivot_context_delta)
                thread_state["pivot_override_keys"] = list(pivot_override_keys)
                thread_state["pivot_constraint_updates"] = dict(pivot_constraint_updates)
                thread_state["pivot_constraint_keys"] = list(pivot_constraint_keys)
                thread_state["pivot_constraint_key"] = pivot_constraint_key
                thread_state["pivot_constraint_value"] = pivot_constraint_value
        if ticket_initial and ticket_initial != str(getattr(task, "user_ticket", "") or ""):
            task_for_run = copy.deepcopy(task)
            task_for_run.user_ticket = ticket_initial
        e3_context_budget_chars: int | None = None
        e3_context_chars_used: int | None = None
        e3_context_clause_count: int | None = None
        e3_packed_clause_count: int | None = None
        e3_context_token_est: int | None = None
        e3_packed_token_est: int | None = None
        e3_context_truncated: bool | None = None
        e3_context_clause_ids: List[str] = []
        e3_packed_total_chars_before: int | None = None
        e3_packed_total_chars_after: int | None = None
        e3_packed_truncated: bool | None = None
        e3_packed_dropped_clause_count: int | None = None
        e3_packed_clause_ids: List[str] = []
        e3_packed_clause_lens: List[int] = []
        e3_packed_clause_is_critical: List[bool] = []
        e12_opened_clause_ids: List[str] = []
        full_episode_clause_ids: List[str] = []
        e3_packed_contains_critical: bool | None = None
        e3_packed_contains_critical0: bool | None = None
        e3_packed_contains_critical1: bool | None = None
        e3_packed_critical_count: int | None = None
        e3_packed_all_critical: bool | None = None
        e3_packed_any_critical: bool | None = None
        e3_packed_all_critical_full_episode: bool | None = None
        e3_packed_any_critical_full_episode: bool | None = None
        e3_packed_critical_count_full_episode: int | None = None
        e3_decoy_clause_count: int | None = None
        e3_litm_filler_count: int | None = None
        e3_litm_filler_position: str | None = None
        e3_litm_filler_clause_ids: List[str] = []
        e3_prompt_includes_required_core: bool | None = None
        e3_prompt_includes_critical_core: bool | None = None
        e3_truncation_loss_estimate: bool | None = None
        goc_folded_episode_count: int | None = None
        goc_unfolded_clause_count: int | None = None
        goc_unfolded_critical_clause_count: int | None = None
        goc_unfold_selected_clause_ids: List[str] = []
        goc_unfold_reason: str | None = None
        # Phase 8: per-task unfold knobs (always defined for GoC to avoid UnboundLocalError
        # on non-final episodes / early exits).
        goc_unfold_max_nodes_used: int | None = None
        goc_unfold_hops_used: int | None = None
        if method == "goc":
            goc_unfold_max_nodes_used = int(_goc_unfold_max_nodes_fixed or 0)
            goc_unfold_hops_used = int(_goc_unfold_hops_fixed or 0)
        goc_candidate_pool_count: int | None = None
        goc_candidate_pool_added_count: int | None = None
        goc_candidate_pool_ids_preclosure: List[str] | None = None
        goc_policyops_dep_graph_snapshot: Dict[str, Any] | None = None
        goc_closure_candidate_count: int | None = None
        goc_unfolded_node_count: int | None = None
        goc_selected_node_ids: List[str] = []
        goc_unfolded_node_ids: List[str] = []
        goc_applicability_seed_enable_cfg = bool(
            getattr(args, "goc_applicability_seed_enable", False)
        )
        goc_applicability_seed_topk_cfg = int(
            getattr(args, "goc_applicability_seed_topk", 8) or 0
        )
        goc_applicability_seed_ids: List[str] = []
        goc_applicability_seed_used: int | None = None
        goc_applicability_seed_applicable_rate: float | None = None
        goc_dependency_closure_enable_cfg = bool(
            getattr(args, "goc_dependency_closure_enable", False)
        )
        goc_dependency_closure_topk_cfg = int(
            getattr(args, "goc_dependency_closure_topk", 12) or 0
        )
        goc_dependency_closure_hops_cfg = int(
            getattr(args, "goc_dependency_closure_hops", 1) or 0
        )
        goc_dependency_closure_universe_cfg = str(
            getattr(args, "goc_dependency_closure_universe", "candidates") or "candidates"
        )
        if goc_dependency_closure_universe_cfg not in {"candidates", "world", "memory_opened"}:
            goc_dependency_closure_universe_cfg = "candidates"
        goc_dependency_closure_added_ids: List[str] = []
        goc_dependency_closure_added_used: int | None = None
        goc_dependency_closure_added_applicable_rate: float | None = None
        judge_correct_full_episode: bool | None = None
        full_episode_supporting_count: int | None = None
        commit_refs = ""
        if threaded_mode and thread_state and episode_id:
            if task_for_run is task:
                task_for_run = copy.deepcopy(task)
            task_for_run.user_ticket = ticket_initial or task_for_run.user_ticket
            commit_refs = _format_commit_refs(
                (thread_state.get("commit1") or {}).get("short_fact"),
                (thread_state.get("commit2") or {}).get("short_fact"),
                episode_id,
            )
            if commit_refs:
                task_for_run.user_ticket = f"{task_for_run.user_ticket}\n\n{commit_refs}"
            # Track per-thread initial ticket node id for pivot-time avoids edge injection.
            if not thread_state.get("initial_ticket_node_id"):
                thread_state["initial_ticket_node_id"] = f"ticket_initial:{thread_id}"
            goc_initial_ticket_node_id = str(thread_state.get("initial_ticket_node_id") or "")
            if int(episode_id or 0) == 1 and state_nodes_e1_by_key:
                state_edges = thread_state.setdefault("state_edges", [])
                if isinstance(state_edges, list):
                    for node_id in state_nodes_e1_by_key.values():
                        _append_unique_edge(state_edges, "ctx_dep", goc_initial_ticket_node_id, str(node_id))

        if threaded_mode and episode_id and int(episode_id) >= 3 and is_pivot_task:
            if task_for_run is task:
                task_for_run = copy.deepcopy(task)
            task_for_run.user_ticket = _format_pivot_ticket_message(
                ticket_initial=ticket_initial or str(task_for_run.user_ticket or ""),
                ticket_updated=ticket_updated,
                assistant_summary=commit_refs or None,
                pivot_message_style=pivot_message_style,
            )
            if thread_state is not None and goc_avoids_mode != "off":
                if not thread_state.get("initial_ticket_node_id"):
                    thread_state["initial_ticket_node_id"] = f"ticket_initial:{thread_id}"
                goc_initial_ticket_node_id = str(thread_state.get("initial_ticket_node_id") or "")
                goc_pivot_ticket_node_id = f"ticket_pivot:{task.task_id}"
                if goc_initial_ticket_node_id:
                    edge = {
                        "type": "avoids",
                        "u": goc_pivot_ticket_node_id,
                        "v": goc_initial_ticket_node_id,
                    }
                    thread_state.setdefault("pivot_ticket_node_ids", []).append(goc_pivot_ticket_node_id)
                    thread_state.setdefault("avoids_edges", []).append(edge)
                    goc_avoids_edge_injected = True
            if thread_state is not None:
                if not goc_pivot_ticket_node_id:
                    goc_pivot_ticket_node_id = f"ticket_pivot:{task.task_id}"
                if state_nodes_e3_by_key:
                    state_edges = thread_state.setdefault("state_edges", [])
                    if isinstance(state_edges, list):
                        for node_id in state_nodes_e3_by_key.values():
                            _append_unique_edge(state_edges, "ctx_dep", goc_pivot_ticket_node_id, str(node_id))
                        for key in pivot_override_keys:
                            new_node = state_nodes_e3_by_key.get(key)
                            old_node = state_nodes_e1_by_key.get(key)
                            if not old_node and isinstance(thread_state.get("state_nodes"), dict):
                                for node_id, meta in thread_state.get("state_nodes", {}).items():
                                    if (
                                        isinstance(meta, dict)
                                        and str(meta.get("kind")) == "state"
                                        and int(meta.get("episode", 0) or 0) == 1
                                        and str(meta.get("key", "")) == str(key)
                                    ):
                                        old_node = str(node_id)
                                        break
                            if new_node and old_node and new_node != old_node:
                                _append_unique_edge(state_edges, "overrides", str(new_node), str(old_node))
        elif (not threaded_mode) and is_pivot_task:
            if task_for_run is task:
                task_for_run = copy.deepcopy(task)
            task_for_run.user_ticket = ticket_updated
        if threaded_mode and episode_id in {1, 2} and shared_open_policy == "shared_topk":
            pred, opened_ids, prompt, raw_output, diag = _run_shared_topk(
                task_for_run,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
            )
            opened_doc_ids_shared = []
            for cid in opened_ids:
                clause = world.clauses.get(cid)
                if clause:
                    opened_doc_ids_shared.append(clause.doc_id)
            diag["shared_open_policy_applied"] = True
            diag["opened_clause_ids_shared"] = list(opened_ids)
            diag["opened_doc_ids_shared"] = opened_doc_ids_shared
            diag["hop1_search_results"] = list(diag.get("primary_search_results") or [])
            diag["hop2_search_results"] = []
            diag.setdefault("hop2_executed", False)
            diag.setdefault("hop2_skip_reason", "shared_open_policy")
            diag.setdefault("hop2_query", "")
            diag.setdefault("hop2_candidate_query", "")
            diag.setdefault("hop2_query_contains_canonical", False)
        elif threaded_mode and episode_id == 3 and thread_state:
            commit1 = thread_state.get("commit1") or {}
            commit2 = thread_state.get("commit2") or {}
            commit_clause_ids = list(
                dict.fromkeys(
                    (commit1.get("supporting_clause_ids") or [])
                    + (commit2.get("supporting_clause_ids") or [])
                )
            )
            commit_facts: Dict[str, Any] = {}
            if scenario_mode == "threaded_v1_2":
                commit_facts = {
                    "commit1.fact1": (commit1.get("short_fact") or {}).get("canonical_term"),
                    "commit2.fact2": (commit2.get("short_fact") or {}).get("exception_condition"),
                }
            else:
                if commit1.get("short_fact") is not None:
                    commit_facts["commit1.fact1"] = commit1.get("short_fact")
                if commit2.get("short_fact") is not None:
                    commit_facts["commit2.fact2"] = commit2.get("short_fact")
            opened_history_ids = list(dict.fromkeys(thread_state.get("opened_history_ids", [])))
            e12_opened_clause_ids = list(opened_history_ids)
            thread_cfg = dict(getattr(task, "thread_config", None) or {})
            e3_litm_filler_clause_ids = list(
                dict.fromkeys(thread_cfg.get("e3_litm_filler_clause_ids") or [])
            )
            e3_litm_filler_position = str(
                thread_cfg.get("e3_litm_filler_position") or "between"
            )
            if thread_cfg.get("e3_litm_filler_count") is not None:
                e3_litm_filler_count = int(thread_cfg.get("e3_litm_filler_count") or 0)
            else:
                e3_litm_filler_count = len(e3_litm_filler_clause_ids)
            opened_history_ids = _inject_litm_filler_clause_ids(
                opened_history_ids,
                e3_litm_filler_clause_ids,
                position=e3_litm_filler_position,
                critical0_id=getattr(task, "critical_clause_id_e1", None),
                critical1_id=getattr(task, "critical_clause_id_e2", None),
            )
            compose_strategy = "commit_only"
            context_clause_ids: List[str] = []
            summary_text = None
            reason_map: Dict[str, List[str]] = {}
            goc_selection_debug: Dict[str, Any] = {}
            if method in {"full", "full_history"}:
                compose_strategy = "full_history"
                context_clause_ids = list(dict.fromkeys(opened_history_ids))
            elif method == "similarity_only":
                compose_strategy = "similarity_only"
                context_clause_ids = _select_similarity_clause_ids(
                    task_for_run.user_ticket,
                    opened_history_ids,
                    world,
                    top_k=max(1, len(opened_history_ids)),
                )
            elif method == "agent_fold":
                compose_strategy = "agent_fold"
                clause_objs = [world.clauses.get(cid) for cid in opened_history_ids if world.clauses.get(cid)]
                summary_text = summarize_clause_history(clause_objs, max_items=6)
                context_clause_ids = _select_similarity_clause_ids(
                    task_for_run.user_ticket,
                    opened_history_ids,
                    world,
                    top_k=max(1, min(6, len(opened_history_ids))),
                )
            elif method == "goc":
                compose_strategy = "goc_fold_unfold"
                avoid_clause_ids, goc_inapplicable_clause_ids = _compute_goc_avoid_clause_ids(
                    mode=goc_avoids_mode,
                    is_pivot_task=bool(is_pivot_task),
                    opened_history_ids=opened_history_ids,
                    world=world,
                    effective_context=goc_effective_context,
                    commit1=commit1,
                    commit2=commit2,
                )
                goc_avoid_target_clause_ids = list(avoid_clause_ids)
                selection_commit_clause_ids = (
                    list(commit_clause_ids)
                    if _normalize_goc_avoids_mode(goc_avoids_mode) == "legacy_commit"
                    else []
                )
                if avoid_clause_ids:
                    avoid_set = set(avoid_clause_ids)
                    clause_objs_for_summary = [
                        world.clauses.get(cid)
                        for cid in opened_history_ids
                        if world.clauses.get(cid) and cid not in avoid_set
                    ]
                else:
                    clause_objs_for_summary = [
                        world.clauses.get(cid) for cid in opened_history_ids if world.clauses.get(cid)
                    ]
                summary_text = (
                    summarize_clause_history(clause_objs_for_summary, max_items=6)
                    if clause_objs_for_summary
                    else ""
                )
                critical_clause_ids = list(getattr(task, "critical_core_clause_ids", None) or [])
                activity_filter = bool(getattr(args, "goc_activity_filter", False))
                activity_filter_fallback = bool(
                    getattr(args, "goc_activity_filter_fallback", True)
                )
                mmr_lambda = float(getattr(args, "goc_mmr_lambda", 0.35) or 0.35)
                anchor_top1 = bool(getattr(args, "goc_anchor_top1_lexical", True))
                include_activity_debug = bool(
                    getattr(args, "goc_activity_debug_in_snapshot", False)
                )

                # First pass chooses anchor seeds from full opened history.
                anchor_ids, _anchor_reason_map, _anchor_debug = _select_goc_unfold_clause_ids(
                    opened_history_ids,
                    selection_commit_clause_ids,
                    critical_clause_ids,
                    task_for_run.user_ticket,
                    world,
                    activity_filter=activity_filter,
                    activity_filter_fallback=activity_filter_fallback,
                    mmr_lambda=mmr_lambda,
                    anchor_top1_lexical=anchor_top1,
                    max_selected=max(1, min(4, len(opened_history_ids))),
                    avoid_clause_ids=avoid_clause_ids,
                    include_debug=False,
                )
                fallback_seed_ids = [cid for cid in opened_history_ids if cid not in set(avoid_clause_ids)]
                anchor_ids = _unique_strs(
                    list(anchor_ids[: max(1, min(4, len(anchor_ids)))])
                    if anchor_ids
                    else list(fallback_seed_ids[: max(1, min(4, len(fallback_seed_ids)))])
                )

                candidate_pool_size = int(getattr(args, "goc_candidate_pool_size", 0) or 0)
                goc_closedbook_universe = str(getattr(args, "goc_closedbook_universe", "memory") or "memory")
                if goc_closedbook_universe not in {"memory", "world"}:
                    goc_closedbook_universe = "memory"
                # In strict closed-book final (E3), do NOT expand candidate pool beyond observed memory.
                if goc_closedbook_universe == "memory":
                    candidate_pool_size = 0
                unfold_candidate_ids = [cid for cid in opened_history_ids if cid not in set(avoid_clause_ids)]
                if candidate_pool_size > 0:
                    try:
                        pool_results = env.search(task_for_run.user_ticket, top_k=candidate_pool_size)
                    except Exception:
                        pool_results = []
                    if pool_results:
                        seen = set(unfold_candidate_ids)
                        pool_results_sorted = sorted(
                            pool_results, key=lambda it: it.get("score", 0.0), reverse=True
                        )
                        for item in pool_results_sorted:
                            cid = item.get("clause_id")
                            if cid and cid not in seen:
                                unfold_candidate_ids.append(cid)
                                seen.add(cid)
                goc_candidate_pool_count = int(len(unfold_candidate_ids))
                goc_candidate_pool_added_count = int(
                    max(0, len(unfold_candidate_ids) - len(opened_history_ids))
                )
                goc_candidate_pool_ids_preclosure = list(unfold_candidate_ids)
                # Resolve per-task unfold knobs via generic GoC policy utility.
                gp = _import_goc_policy_module()
                unfold_policy_cfg = gp.GoCUnfoldPolicy(
                    policy=str(goc_unfold_policy),
                    default_knobs=(
                        int(goc_unfold_default_max_nodes_cfg),
                        int(goc_unfold_default_hops_cfg),
                    ),
                    pivot_knobs=(
                        int(goc_unfold_pivot_max_nodes_cfg),
                        int(goc_unfold_pivot_hops_cfg),
                    ),
                    budget_mode=str(goc_unfold_budget_mode_cfg),
                )
                goc_unfold_max_nodes_used, goc_unfold_hops_used = gp.choose_unfold_knobs(
                    unfold_policy_cfg,
                    is_pivot_task=bool(is_pivot_task),
                    is_update_episode=bool(is_pivot_task),
                )
                if (
                    goc_unfold_budget_mode_cfg in {"hops_only", "nodes_and_hops"}
                    and goc_unfold_hops_used > 0
                    and anchor_ids
                ):
                    # Universe for closure traversal: default to candidate memory; optionally allow full world.
                    if goc_closedbook_universe == "world":
                        universe_clause_ids: List[str] | None = None
                        try:
                            clauses_map = getattr(world, "clauses", None)
                            if isinstance(clauses_map, dict) and clauses_map:
                                universe_clause_ids = list(clauses_map.keys())
                        except Exception:
                            universe_clause_ids = None
                    else:
                        universe_clause_ids = list(_unique_strs(unfold_candidate_ids))
                    closure_ids = _expand_clause_dependency_closure(
                        anchor_ids,
                        unfold_candidate_ids,
                        world,
                        max_hops=goc_unfold_hops_used,
                        universe_clause_ids=universe_clause_ids,
                    )
                    if closure_ids:
                        unfold_candidate_ids = closure_ids
                    else:
                        unfold_candidate_ids = list(anchor_ids)
                if avoid_clause_ids:
                    avoid_set = set(avoid_clause_ids)
                    unfold_candidate_ids = [cid for cid in unfold_candidate_ids if cid not in avoid_set]

                if goc_unfold_budget_mode_cfg == "hops_only":
                    max_selected = max(1, len(unfold_candidate_ids))
                else:
                    max_selected = int(goc_unfold_max_nodes_used) if goc_unfold_max_nodes_used > 0 else 4

                goc_applicability_seed_ids = []
                goc_applicability_seed_applicable_rate = float("nan")
                is_phase14_pivot_e3 = bool(
                    int(episode_id or 0) == 3 and bool(is_pivot_task)
                )
                if (
                    is_phase14_pivot_e3
                    and goc_applicability_seed_enable_cfg
                    and goc_applicability_seed_topk_cfg > 0
                ):
                    seed_candidates = list(_unique_strs(unfold_candidate_ids))
                    if not seed_candidates:
                        seed_candidates = [
                            cid
                            for cid in _unique_strs(opened_history_ids)
                            if cid not in set(_unique_strs(avoid_clause_ids))
                        ]
                    goc_applicability_seed_ids, goc_applicability_seed_applicable_rate = (
                        _compute_phase14_applicability_seed_ids(
                            candidate_clause_ids=seed_candidates,
                            world=world,
                            effective_context=goc_effective_context,
                            avoid_clause_ids=avoid_clause_ids,
                            top_k=goc_applicability_seed_topk_cfg,
                        )
                    )
                goc_applicability_seed_used = int(len(goc_applicability_seed_ids))

                context_clause_ids, reason_map, goc_selection_debug = _select_goc_unfold_clause_ids(
                    unfold_candidate_ids,
                    selection_commit_clause_ids,
                    critical_clause_ids,
                    task_for_run.user_ticket,
                    world,
                    activity_filter=activity_filter,
                    activity_filter_fallback=activity_filter_fallback,
                    mmr_lambda=mmr_lambda,
                    anchor_top1_lexical=anchor_top1,
                    max_selected=max_selected,
                    avoid_clause_ids=avoid_clause_ids,
                    include_debug=include_activity_debug,
                )
                if goc_applicability_seed_ids:
                    merged = _unique_strs(list(goc_applicability_seed_ids) + list(context_clause_ids))
                    context_clause_ids = merged[:max_selected]
                    for cid in goc_applicability_seed_ids:
                        reason_map.setdefault(cid, [])
                        if "applicability_seed" not in reason_map[cid]:
                            reason_map[cid].append("applicability_seed")

                goc_dependency_closure_added_ids = []
                goc_dependency_closure_added_applicable_rate = float("nan")
                if (
                    is_phase14_pivot_e3
                    and goc_dependency_closure_enable_cfg
                    and goc_dependency_closure_topk_cfg > 0
                ):
                    closure_added_ids, closure_added_rate = _compute_phase14_dependency_closure_ids(
                        seed_clause_ids=context_clause_ids,
                        unfold_candidate_ids=unfold_candidate_ids,
                        opened_history_ids=opened_history_ids,
                        avoid_clause_ids=avoid_clause_ids,
                        world=world,
                        effective_context=goc_effective_context,
                        top_k=goc_dependency_closure_topk_cfg,
                        hops=goc_dependency_closure_hops_cfg,
                        universe_mode=goc_dependency_closure_universe_cfg,
                    )
                    if closure_added_ids:
                        base_ids = list(_unique_strs(context_clause_ids))
                        merged = _unique_strs(base_ids + closure_added_ids)
                        if len(merged) > max_selected:
                            merged = merged[:max_selected]
                        base_set = set(base_ids)
                        goc_dependency_closure_added_ids = [
                            cid for cid in merged if cid not in base_set
                        ]
                        context_clause_ids = merged
                    goc_dependency_closure_added_applicable_rate = closure_added_rate
                goc_dependency_closure_added_used = int(len(goc_dependency_closure_added_ids))

                if include_activity_debug:
                    goc_selection_debug = dict(goc_selection_debug or {})
                    goc_selection_debug["phase14_applicability_seed_ids"] = list(
                        goc_applicability_seed_ids
                    )
                    goc_selection_debug["phase14_dependency_closure_added_ids"] = list(
                        goc_dependency_closure_added_ids
                    )
                goc_closure_candidate_count = int(len(unfold_candidate_ids))
                # closure candidates (after hop expansion) vs unfolded nodes (actually opened/used)
                goc_unfolded_node_count = int(len(context_clause_ids))
                goc_unfolded_node_ids = list(context_clause_ids)
                # Optional: emit a task-local dependency graph snapshot (for debugging & analysis).
                if log_graph and world is not None and episode_id == 3:
                    try:
                        clauses_map = getattr(world, "clauses", None)
                        clause_lookup = clauses_map if isinstance(clauses_map, dict) else {}
                        # Use the closure candidate set as the node set for the dependency snapshot.
                        dep_nodes = _unique_strs(unfold_candidate_ids)
                        dep_node_set = set(dep_nodes)
                        dep_neighbors = _build_clause_dependency_neighbors(dep_nodes, world)
                        dep_edges: Dict[str, List[str]] = {}
                        dep_edge_pairs = 0
                        for u in dep_nodes:
                            vs = sorted(v for v in dep_neighbors.get(u, set()) if v in dep_node_set and v != u)
                            if vs:
                                dep_edges[str(u)] = [str(v) for v in vs]
                                dep_edge_pairs += len(vs)

                        dep_node_meta: Dict[str, Dict[str, Any]] = {}
                        for cid in dep_nodes:
                            clause = clause_lookup.get(cid)
                            if clause is None:
                                continue
                            dep_node_meta[str(cid)] = {
                                "doc_id": str(getattr(clause, "doc_id", "") or ""),
                                "kind": str(getattr(clause, "kind", "") or ""),
                                "slot": str(getattr(clause, "slot", "") or ""),
                                "published_at": str(getattr(clause, "published_at", "") or ""),
                                "text_preview": str(getattr(clause, "text", "") or "")[:220],
                            }
                        state_node_meta: Dict[str, Dict[str, Any]] = {}
                        if isinstance(thread_state, dict):
                            ts_nodes = thread_state.get("state_nodes")
                            if isinstance(ts_nodes, dict):
                                for node_id, meta in ts_nodes.items():
                                    if isinstance(meta, dict) and str(meta.get("kind", "")) == "state":
                                        state_node_meta[str(node_id)] = dict(meta)
                        if state_node_meta:
                            dep_node_meta.update(state_node_meta)
                        if goc_initial_ticket_node_id:
                            dep_node_meta.setdefault(
                                str(goc_initial_ticket_node_id),
                                {"kind": "ticket_initial", "thread_id": thread_id},
                            )
                        if goc_pivot_ticket_node_id:
                            dep_node_meta.setdefault(
                                str(goc_pivot_ticket_node_id),
                                {"kind": "ticket_pivot", "thread_id": thread_id},
                            )
                        ctx_dep_edges: Dict[str, List[str]] = {}
                        override_edges: Dict[str, List[str]] = {}
                        if isinstance(thread_state, dict):
                            ts_edges = thread_state.get("state_edges")
                            if isinstance(ts_edges, list):
                                for edge in ts_edges:
                                    if not isinstance(edge, dict):
                                        continue
                                    edge_type = str(edge.get("type") or "")
                                    u = str(edge.get("u") or "")
                                    v = str(edge.get("v") or "")
                                    if not u or not v:
                                        continue
                                    if edge_type == "ctx_dep":
                                        ctx_dep_edges.setdefault(u, [])
                                        if v not in ctx_dep_edges[u]:
                                            ctx_dep_edges[u].append(v)
                                    elif edge_type == "overrides":
                                        override_edges.setdefault(u, [])
                                        if v not in override_edges[u]:
                                            override_edges[u].append(v)

                        def _find_state_node(key: str, value: Any) -> str:
                            preferred = ""
                            for node_id, meta in state_node_meta.items():
                                if not isinstance(meta, dict):
                                    continue
                                if str(meta.get("key", "")) != str(key):
                                    continue
                                if str(meta.get("value", "")) != str(value):
                                    continue
                                if int(meta.get("episode", 0) or 0) == 3:
                                    return str(node_id)
                                if not preferred:
                                    preferred = str(node_id)
                            return preferred

                        for cid in dep_nodes:
                            clause = clause_lookup.get(cid)
                            if clause is None:
                                continue
                            applies_if = getattr(clause, "applies_if", None) or {}
                            if not isinstance(applies_if, dict):
                                continue
                            for key, values in applies_if.items():
                                if not isinstance(values, list) or not values:
                                    continue
                                for value in values:
                                    state_node_id = _find_state_node(str(key), value)
                                    if not state_node_id:
                                        state_node_id = _state_node_id(int(episode_id), str(key), value)
                                        state_node_meta[state_node_id] = {
                                            "kind": "state",
                                            "key": str(key),
                                            "value": value,
                                            "episode": int(episode_id),
                                        }
                                        dep_node_meta[state_node_id] = dict(state_node_meta[state_node_id])
                                    ctx_dep_edges.setdefault(str(cid), [])
                                    if state_node_id not in ctx_dep_edges[str(cid)]:
                                        ctx_dep_edges[str(cid)].append(state_node_id)
                        for edge_map in (dep_edges, ctx_dep_edges, override_edges):
                            for key in list(edge_map.keys()):
                                edge_map[key] = sorted(set(edge_map[key]))
                        ctx_dep_pairs = sum(len(vs) for vs in ctx_dep_edges.values())
                        override_pairs = sum(len(vs) for vs in override_edges.values())

                        goc_policyops_dep_graph_snapshot = {
                            "snapshot_kind": "policyops_dependency_graph",
                            "episode_id": int(episode_id),
                            "pivot_type": str(pivot_type or ""),
                            "goc_K": int(goc_unfold_max_nodes_used) if goc_unfold_max_nodes_used is not None else None,
                            "goc_H": int(goc_unfold_hops_used) if goc_unfold_hops_used is not None else None,
                            "candidate_pool_clause_ids": list(goc_candidate_pool_ids_preclosure or []),
                            "closure_clause_ids": dep_nodes,
                            "unfolded_clause_ids": list(context_clause_ids),
                            "anchor_clause_ids": list(anchor_ids or []),
                            "applicability_seed_clause_ids": list(goc_applicability_seed_ids or []),
                            "dependency_closure_added_clause_ids": list(goc_dependency_closure_added_ids or []),
                            "depends_edge_pairs": int(dep_edge_pairs),
                            "ctx_dep_edge_pairs": int(ctx_dep_pairs),
                            "override_edge_pairs": int(override_pairs),
                            "effective_context": dict(goc_effective_context),
                            "initial_context": dict(goc_initial_context),
                            "pivot_override_keys": list(pivot_override_keys),
                            "pivot_context_delta": dict(pivot_context_delta),
                            "nodes": dep_node_meta,
                            "edges": {
                                "depends": dep_edges,
                                "ctx_dep": ctx_dep_edges,
                                "overrides": override_edges,
                                "depends_llm": {},
                                "doc_ref": {},
                                "seq": {},
                            },
                        }
                    except Exception:
                        goc_policyops_dep_graph_snapshot = None

                goc_selected_node_ids = list(context_clause_ids)
                goc_folded_episode_count = 2
            context_budget = int(getattr(args, "thread_context_budget_chars", 8000))
            e3_context_budget_chars = context_budget
            (
                summary_text,
                context_clauses,
                used_chars,
                truncated,
                total_before_chars,
                total_after_chars,
                dropped_clause_count,
            ) = _apply_context_budget(
                summary_text,
                context_clause_ids,
                world,
                context_budget,
            )
            e3_context_chars_used = used_chars
            e3_context_clause_ids = [c["clause_id"] for c in context_clauses]
            e3_context_clause_count = len(context_clauses)
            e3_packed_clause_count = e3_context_clause_count
            e3_packed_total_chars_before = total_before_chars
            e3_packed_total_chars_after = total_after_chars
            e3_context_token_est = int(math.ceil(max(0, e3_context_chars_used) / 4.0))
            e3_packed_token_est = int(math.ceil(max(0, e3_packed_total_chars_after) / 4.0))
            e3_packed_truncated = truncated
            e3_packed_dropped_clause_count = dropped_clause_count
            e3_context_truncated = e3_packed_truncated
            e3_packed_clause_ids = list(e3_context_clause_ids)
            if goc_avoid_target_clause_ids:
                avoid_set = set(_unique_strs(goc_avoid_target_clause_ids))
                goc_avoided_node_injected = bool(avoid_set & set(e3_context_clause_ids))
            if method == "goc":
                critical_ids = set(_unique_strs(getattr(task, "critical_core_clause_ids", None) or []))
                if critical_ids:
                    critical_present = critical_ids & set(e3_context_clause_ids)
                    critical_missing_ids = sorted(critical_ids - critical_present)
                    critical_coverage_e3 = float(len(critical_present)) / float(len(critical_ids))
                injected_ids_for_applicability: List[str] = []
                for cid in e3_context_clause_ids:
                    clause = world.clauses.get(cid) if world else None
                    if clause is None:
                        continue
                    if not _clause_applies_to_context(clause, goc_effective_context):
                        injected_ids_for_applicability.append(cid)
                inapplicable_injected_clause_ids = list(_unique_strs(injected_ids_for_applicability))
                if e3_context_clause_ids:
                    inapplicable_injected_rate_e3 = float(len(inapplicable_injected_clause_ids)) / float(
                        len(e3_context_clause_ids)
                    )
                else:
                    inapplicable_injected_rate_e3 = None
            full_episode_clause_ids = list(
                dict.fromkeys(e12_opened_clause_ids + e3_packed_clause_ids)
            )
            e3_packed_clause_lens = [len(c.get("text", "")) for c in context_clauses]
            e3_decoy_clause_count = 0
            for cid in e3_context_clause_ids:
                clause = world.clauses.get(cid) if world else None
                if clause and str(clause.doc_id).startswith("DECOY"):
                    e3_decoy_clause_count += 1
            if method == "goc":
                goc_unfolded_clause_count = len(context_clauses)
                goc_unfold_selected_clause_ids = list(e3_context_clause_ids[:10])
                goc_selected_node_ids = list(e3_context_clause_ids)
                if not goc_unfolded_node_ids:
                    goc_unfolded_node_ids = list(e3_context_clause_ids)
                    goc_unfolded_node_count = len(goc_unfolded_node_ids)
                    goc_closure_candidate_count = len(goc_unfolded_node_ids)
                reasons: List[str] = []
                for cid in e3_context_clause_ids:
                    reasons.extend(reason_map.get(cid, []))
                if reasons:
                    goc_unfold_reason = "|".join(sorted(set(reasons)))
            if scenario_mode in {"threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}:
                required_core_ids = list(
                    getattr(task.gold, "gold_evidence_core", None)
                    or task.gold.gold_evidence
                    or []
                )
                if required_core_ids:
                    e3_prompt_includes_required_core = any(
                        cid in e3_packed_clause_ids for cid in required_core_ids
                    )
                critical_core_ids = list(getattr(task, "critical_core_clause_ids", None) or [])
                if critical_core_ids:
                    critical0_id = critical_core_ids[0] if len(critical_core_ids) > 0 else None
                    critical1_id = critical_core_ids[1] if len(critical_core_ids) > 1 else None
                    e3_packed_clause_is_critical = [
                        cid in set(critical_core_ids) for cid in e3_packed_clause_ids
                    ]
                    e3_prompt_includes_critical_core = all(
                        cid in e3_packed_clause_ids for cid in critical_core_ids
                    )
                    e3_packed_critical_count = sum(
                        1 for cid in critical_core_ids if cid in e3_packed_clause_ids
                    )
                    e3_packed_contains_critical = e3_packed_critical_count > 0
                    e3_packed_any_critical = e3_packed_contains_critical
                    e3_packed_all_critical = e3_packed_critical_count == len(critical_core_ids)
                    e3_packed_critical_count_full_episode = sum(
                        1 for cid in critical_core_ids if cid in full_episode_clause_ids
                    )
                    e3_packed_any_critical_full_episode = (
                        e3_packed_critical_count_full_episode > 0
                    )
                    e3_packed_all_critical_full_episode = (
                        e3_packed_critical_count_full_episode == len(critical_core_ids)
                    )
                    e3_packed_contains_critical0 = (
                        critical0_id in e3_packed_clause_ids if critical0_id else None
                    )
                    e3_packed_contains_critical1 = (
                        critical1_id in e3_packed_clause_ids if critical1_id else None
                    )
                else:
                    e3_packed_clause_is_critical = [False] * len(e3_packed_clause_ids)
                if critical_core_ids and method == "goc":
                    goc_unfolded_critical_clause_count = sum(
                        1 for cid in critical_core_ids if cid in e3_packed_clause_ids
                    )
                if critical_core_ids:
                    critical_in_history = all(
                        cid in opened_history_ids for cid in critical_core_ids
                    )
                    if e3_context_truncated is not None:
                        e3_truncation_loss_estimate = bool(
                            critical_in_history
                            and e3_context_truncated
                            and not e3_prompt_includes_critical_core
                        )
            if goc_graph and method == "goc" and scenario_mode in {"threaded_v1_3_fu", "threaded_v1_3_fu_decoy"}:
                rid = run_id or run_dir.name
                fold_node_id = f"fold:{task.task_id}:{goc_graph.step}"
                goc_graph.add_node(
                    fold_node_id,
                    "fold",
                    thread_id=thread_id,
                    episode_id=episode_id,
                    folded_episode_count=goc_folded_episode_count,
                    step=goc_graph.step,
                )
                if goc_graph_path:
                    append_event(
                        goc_graph_path,
                        build_event(
                            rid,
                            task.task_id,
                            method,
                            goc_graph.step,
                            "FOLD",
                            {
                                "thread_id": thread_id,
                                "episode_id": episode_id,
                                "folded_episode_count": goc_folded_episode_count,
                            },
                        ),
                    )
                goc_graph.step += 1
                unfold_node_id = f"unfold:{task.task_id}:{goc_graph.step}"
                goc_graph.add_node(
                    unfold_node_id,
                    "unfold",
                    thread_id=thread_id,
                    episode_id=episode_id,
                    clause_ids=e3_context_clause_ids,
                    reason=goc_unfold_reason,
                    step=goc_graph.step,
                )
                if goc_graph_path:
                    append_event(
                        goc_graph_path,
                        build_event(
                            rid,
                            task.task_id,
                            method,
                            goc_graph.step,
                            "UNFOLD",
                            {
                                "thread_id": thread_id,
                                "episode_id": episode_id,
                                "clause_ids": e3_context_clause_ids,
                                "reason": goc_unfold_reason,
                                "goc_unfold_max_nodes": goc_unfold_max_nodes_used,
                                "goc_unfold_hops": goc_unfold_hops_used,
                                "closure_candidate_count": goc_closure_candidate_count,
                                "candidate_pool_count": goc_candidate_pool_count,
                                "candidate_pool_added_count": goc_candidate_pool_added_count,
                                "candidate_pool_size": int(getattr(args, "goc_candidate_pool_size", 0) or 0),
                                "unfolded_node_count": goc_unfolded_node_count,
                                "selected_node_ids": goc_selected_node_ids,
                                "unfolded_node_ids": goc_unfolded_node_ids,
                            },
                        ),
                    )
                goc_graph.step += 1
            prompt = _build_threaded_prompt(
                task_for_run.user_ticket,
                commit_facts,
                commit_clause_ids,
                clauses=context_clauses,
                summary_text=summary_text,
            )
            raw_output = client.generate(prompt)
            pred = _parse_prediction(raw_output)
            opened_ids = list(e3_context_clause_ids)
            diag = {
                "compose_strategy": compose_strategy,
                "commit_clause_ids": commit_clause_ids,
                "commit_facts": commit_facts,
                "compose_context_clause_ids": context_clause_ids,
                "compose_summary_used": bool(summary_text),
                "e3_context_budget_chars": e3_context_budget_chars,
                "e3_context_chars_used": e3_context_chars_used,
                "e3_context_clause_count": e3_context_clause_count,
                "e3_packed_clause_count": e3_packed_clause_count,
                "e3_context_token_est": e3_context_token_est,
                "e3_packed_token_est": e3_packed_token_est,
                "e3_context_truncated": e3_context_truncated,
                "e3_context_clause_ids": e3_context_clause_ids,
                "e3_packed_clause_lens": e3_packed_clause_lens,
                "e3_packed_clause_is_critical": e3_packed_clause_is_critical,
                "e3_packed_total_chars_before": e3_packed_total_chars_before,
                "e3_packed_total_chars_after": e3_packed_total_chars_after,
                "e3_packed_truncated": e3_packed_truncated,
                "e3_packed_dropped_clause_count": e3_packed_dropped_clause_count,
                "e3_prompt_includes_required_core": e3_prompt_includes_required_core,
                "e3_prompt_includes_critical_core": e3_prompt_includes_critical_core,
                "e3_truncation_loss_estimate": e3_truncation_loss_estimate,
                "goc_folded_episode_count": goc_folded_episode_count,
                "goc_unfolded_clause_count": goc_unfolded_clause_count,
                "goc_unfolded_critical_clause_count": goc_unfolded_critical_clause_count,
                "goc_unfold_selected_clause_ids": goc_unfold_selected_clause_ids,
                "goc_unfold_reason": goc_unfold_reason,
                "goc_candidate_pool_count": goc_candidate_pool_count,
                "goc_candidate_pool_added_count": goc_candidate_pool_added_count,
                "goc_candidate_pool_size": (
                    int(getattr(args, "goc_candidate_pool_size", 0) or 0)
                    if method == "goc"
                    else None
                ),
                "goc_unfold_max_nodes": goc_unfold_max_nodes_used if method == "goc" else None,
                "goc_unfold_hops": goc_unfold_hops_used if method == "goc" else None,
                "goc_unfold_budget_mode": (
                    goc_unfold_budget_mode_cfg if method == "goc" else None
                ),
                "closure_candidate_count": goc_closure_candidate_count,
                "unfolded_node_count": goc_unfolded_node_count,
                "selected_node_ids": list(goc_selected_node_ids),
                "unfolded_node_ids": list(goc_unfolded_node_ids),
                "pivot_message_style": pivot_message_style,
                "pivot_gold_mode": pivot_gold_mode,
                "goc_enable_avoids": bool(goc_enable_avoids),
                "goc_avoids_mode": str(goc_avoids_mode),
                "goc_initial_ticket_node_id": goc_initial_ticket_node_id,
                "goc_pivot_ticket_node_id": goc_pivot_ticket_node_id,
                "goc_avoids_edge_injected": bool(goc_avoids_edge_injected),
                "goc_avoid_target_clause_ids": list(goc_avoid_target_clause_ids),
                "goc_avoided_node_injected": goc_avoided_node_injected,
                "goc_inapplicable_clause_ids": list(goc_inapplicable_clause_ids),
                "goc_applicability_seed_enable": bool(goc_applicability_seed_enable_cfg),
                "goc_applicability_seed_topk": int(goc_applicability_seed_topk_cfg),
                "goc_applicability_seed_ids": list(goc_applicability_seed_ids),
                "goc_applicability_seed_used": goc_applicability_seed_used,
                "goc_applicability_seed_applicable_rate": goc_applicability_seed_applicable_rate,
                "goc_dependency_closure_enable": bool(goc_dependency_closure_enable_cfg),
                "goc_dependency_closure_topk": int(goc_dependency_closure_topk_cfg),
                "goc_dependency_closure_hops": int(goc_dependency_closure_hops_cfg),
                "goc_dependency_closure_universe": str(goc_dependency_closure_universe_cfg),
                "goc_dependency_closure_added_ids": list(goc_dependency_closure_added_ids),
                "goc_dependency_closure_added_used": goc_dependency_closure_added_used,
                "goc_dependency_closure_added_applicable_rate": goc_dependency_closure_added_applicable_rate,
                "goc_effective_context": dict(goc_effective_context),
                "goc_initial_context": dict(goc_initial_context),
                "pivot_override_keys": list(pivot_override_keys),
                "pivot_context_delta": dict(pivot_context_delta),
                "pivot_constraint_updates": dict(pivot_constraint_updates),
                "pivot_constraint_keys": list(pivot_constraint_keys),
                "pivot_constraint_key": pivot_constraint_key,
                "pivot_constraint_value": pivot_constraint_value,
            }
            if bool(getattr(args, "goc_activity_debug_in_snapshot", False)) and goc_selection_debug:
                diag["goc_activity_debug"] = goc_selection_debug
        elif method == "topk":
            pred, opened_ids, prompt, raw_output, diag = run_topk_rag(
                task_for_run,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
        elif method in {"full", "full_history"}:
            pred, opened_ids, prompt, raw_output, diag = run_full_history(
                task_for_run,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
        elif method in {"goc", "goc_base"}:
            if method == "goc_base":
                agent_query_policy = "single_hop"
            history_seed_clause_ids_for_goc: list[str] | None = None
            if threaded_mode and thread_state:
                try:
                    _hs: list[str] = []
                    _hs.extend(list(thread_state.get("opened_history_ids", []) or []))
                    if episode_id and int(episode_id) >= 2:
                        _c1 = (thread_state.get("commit1") or {}).get("supporting_clause_ids") or []
                        _hs = list(dict.fromkeys(list(_c1) + _hs))
                    history_seed_clause_ids_for_goc = _hs if _hs else None
                except Exception:
                    history_seed_clause_ids_for_goc = None
            history_seed_clause_ids = None
            if threaded_mode and thread_state:
                _hs: list[str] = []
                try:
                    _hs.extend(list(thread_state.get("opened_history_ids", []) or []))
                except Exception:
                    pass
                # Include prior commit anchors as additional seeds when available.
                if episode_id and int(episode_id) >= 2:
                    try:
                        _hs = list(dict.fromkeys(list((thread_state.get("commit1") or {}).get("supporting_clause_ids") or []) + _hs))
                    except Exception:
                        pass
                history_seed_clause_ids = _hs if _hs else None
            pred, opened_ids, prompt, raw_output, error, controller_action, diag = run_goc_heuristic(
                task_for_run,
                env,
                client,
                controller=controller,
                controller_mode=controller_mode,
                primary_top_k=args.primary_search_top_k,
                force_open_top_n=args.force_open_top_n,
                force_open_source=args.force_open_source,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
                agent_query_policy=agent_query_policy,
                hop1_query_mode=getattr(args, "hop1_query_mode", "stripped"),
                bridge_reward_bonus=float(getattr(args, "bridge_reward_bonus", 0.0)),
                controller_policy=controller_policy,
                open_split_mode=getattr(args, "open_split_mode", "all_union_rank"),
                open_split_hop1=int(getattr(args, "open_split_hop1", 0)),
                open_policy=getattr(args, "open_policy", "current"),
                save_internal_graph=bool(log_graph),
                internal_budget_active=int(goc_internal_budget_active_tokens),
                internal_budget_unfold=int(goc_internal_budget_unfold_tokens),
                internal_unfold_k=int(goc_internal_unfold_k),
                goc_activity_filter=bool(getattr(args, "goc_activity_filter", False)),
                goc_activity_filter_fallback=bool(
                    getattr(args, "goc_activity_filter_fallback", True)
                ),
                goc_mmr_lambda=float(getattr(args, "goc_mmr_lambda", 0.35) or 0.35),
                goc_anchor_top1_lexical=bool(
                    getattr(args, "goc_anchor_top1_lexical", True)
                ),
                goc_activity_debug_in_snapshot=bool(
                    getattr(args, "goc_activity_debug_in_snapshot", False)
                ),
                history_seed_clause_ids=history_seed_clause_ids_for_goc,
                goc_graph_frontier_enabled=bool(getattr(args, "goc_graph_frontier", True)),
                goc_graph_frontier_hops=int(getattr(args, "goc_graph_frontier_hops", 2) or 2),
                goc_graph_frontier_max_nodes=int(getattr(args, "goc_graph_frontier_max_nodes", 50) or 50),
                goc_graph_frontier_seed_top_n=int(getattr(args, "goc_graph_frontier_seed_top_n", 6) or 6),
                goc_graph_frontier_score_frac=float(getattr(args, "goc_graph_frontier_score_frac", 0.7) or 0.7),
            )
            if isinstance(diag, dict):
                maybe_snaps = diag.get("goc_internal_snapshots")
                if isinstance(maybe_snaps, list):
                    goc_internal_snapshots = [s for s in maybe_snaps if isinstance(s, dict)]
            if controller_action:
                controller_actions[controller_action] = controller_actions.get(controller_action, 0) + 1
        elif method == "similarity_only":
            pred, opened_ids, prompt, raw_output, diag = run_similarity_only(
                task_for_run,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
        elif method == "agent_fold":
            pred, opened_ids, prompt, raw_output, diag = run_agent_fold(
                task_for_run,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
        elif method == "oracle":
            pred, opened_ids, prompt, raw_output, oracle_meta = run_oracle(
                task,
                env,
                client,
                primary_top_k=args.primary_search_top_k,
                use_query_rewrite=args.use_query_rewrite,
                rewrite_queries_count=args.rewrite_queries,
                rewrite_mode=args.query_rewrite_mode,
                merge_rank_fusion=args.merge_rank_fusion,
            )
            diag = oracle_meta
        elif method == "engine":
            pred = run_engine_oracle(task, env)
            opened_ids = []
            prompt = ""
            raw_output = None
            diag = {"primary_search_results": [], "primary_search_top_k": 0, "primary_search_query": ""}
        else:
            raise ValueError(f"Unknown method: {method}")

        answer_parse_failed: bool | None = None
        if isinstance(raw_output, str) and method != "engine":
            pred, answer_parse_failed = _parse_prediction_for_answer_metrics(raw_output)

        commit_supporting_clause_ids: List[str] | None = None
        commit_short_fact: Dict[str, Any] | None = None
        commit_correct: bool | None = None
        commit_correct_anchor: bool | None = None
        commit_correct_union: bool | None = None
        commit_anchor_promotion_needed: bool | None = None
        commit_supporting_core_hits: List[str] = []
        commit1_correct_anchor: bool | None = None
        commit1_correct_union: bool | None = None
        commit1_anchor_promotion_needed: bool | None = None
        commit1_supporting_core_hits: List[str] = []
        commit2_correct_anchor: bool | None = None
        commit2_correct_union: bool | None = None
        commit2_anchor_promotion_needed: bool | None = None
        commit2_supporting_core_hits: List[str] = []
        thread_strict_correct: bool | None = None
        thread_e3_only_correct: bool | None = None
        thread_judge_correct: bool | None = None
        commit_clause_ids: List[str] | None = None
        commit_anchor_clause_ids: List[str] | None = None
        commit_anchor_source_used: str | None = None
        commit_model_evidence_clause_ids: List[str] | None = None
        if threaded_mode and thread_state and episode_id in {1, 2}:
            # Commit stage (episodes 1/2): store anchors for episode-3 "allowed evidence".
            # Phase 12 adds an option to derive commit anchors from model-provided evidence IDs
            # (lossless + recoverable, but correctness depends on citing the right clause IDs).
            commit_supporting_clause_ids = _extract_commit_supporting(
                task,
                opened_ids,
                world,
                episode_kind,
            )

            commit_anchor_source = getattr(args, "commit_anchor_source", "opened_supporting")
            commit_anchor_max_ids = int(getattr(args, "commit_anchor_max_ids", 4) or 0)
            commit_anchor_require_opened = bool(getattr(args, "commit_anchor_require_opened", True))
            commit_anchor_fallback_opened = bool(getattr(args, "commit_anchor_fallback_opened", True))

            model_evidence_clause_ids = _extract_model_evidence_clause_ids(
                pred,
                opened_ids=opened_ids,
                max_ids=0,
                require_opened=commit_anchor_require_opened,
            )
            commit_model_evidence_clause_ids = list(model_evidence_clause_ids)

            commit_anchor_clause_ids, commit_anchor_source_used_local = _choose_commit_anchor_ids(
                opened_supporting=commit_supporting_clause_ids,
                model_evidence=model_evidence_clause_ids,
                source=str(commit_anchor_source),
                max_ids=commit_anchor_max_ids,
                fallback_opened=commit_anchor_fallback_opened,
            )
            commit_anchor_source_used = commit_anchor_source_used_local
            commit_clause_ids = list(commit_anchor_clause_ids)

            commit_short_fact = _extract_commit_short_fact(
                task,
                commit_anchor_clause_ids,
                world,
                episode_kind,
            )
            core_ids = list(
                getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
            )
            commit_quality = _compute_commit_quality(
                anchor_clause_ids=commit_anchor_clause_ids,
                supporting_clause_ids=commit_supporting_clause_ids,
                eval_core_clause_ids=core_ids,
            )
            commit_correct_anchor = commit_quality.get("correct_anchor")
            commit_correct_union = commit_quality.get("correct_union")
            commit_anchor_promotion_needed = commit_quality.get("anchor_promotion_needed")
            commit_supporting_core_hits = list(commit_quality.get("supporting_core_hits") or [])
            commit_correct = commit_correct_anchor
            if episode_id == 1:
                thread_state["commit1"] = {
                    "supporting_clause_ids": commit_supporting_clause_ids,
                    "anchor_clause_ids": commit_anchor_clause_ids,
                    "model_evidence_clause_ids": model_evidence_clause_ids,
                    "anchor_source": commit_anchor_source_used_local,
                    "short_fact": commit_short_fact or {},
                    "commit_correct": commit_correct_anchor,
                    "commit_correct_anchor": commit_correct_anchor,
                    "commit_correct_union": commit_correct_union,
                    "commit_anchor_promotion_needed": commit_anchor_promotion_needed,
                    "commit_supporting_core_hits": list(commit_supporting_core_hits),
                }
            elif episode_id == 2:
                thread_state["commit2"] = {
                    "supporting_clause_ids": commit_supporting_clause_ids,
                    "anchor_clause_ids": commit_anchor_clause_ids,
                    "model_evidence_clause_ids": model_evidence_clause_ids,
                    "anchor_source": commit_anchor_source_used_local,
                    "short_fact": commit_short_fact or {},
                    "commit_correct": commit_correct_anchor,
                    "commit_correct_anchor": commit_correct_anchor,
                    "commit_correct_union": commit_correct_union,
                    "commit_anchor_promotion_needed": commit_anchor_promotion_needed,
                    "commit_supporting_core_hits": list(commit_supporting_core_hits),
                }
            history_source = opened_ids
            if isinstance(diag, dict) and diag.get("opened_total_clause_ids"):
                history_source = list(diag.get("opened_total_clause_ids") or [])
            opened_history_ids = list(
                dict.fromkeys(thread_state.get("opened_history_ids", []) + history_source)
            )
            thread_state["opened_history_ids"] = opened_history_ids
            e12_opened_clause_ids = list(opened_history_ids)
        if threaded_mode and thread_state and episode_id == 3:
            commit1 = thread_state.get("commit1") or {}
            commit2 = thread_state.get("commit2") or {}
            commit_clause_ids = list(
                dict.fromkeys(
                    (commit1.get("anchor_clause_ids") or [])
                    + (commit2.get("anchor_clause_ids") or [])
                )
            )
            commit_supporting_clause_ids = list(
                dict.fromkeys(
                    (commit1.get("supporting_clause_ids") or [])
                    + (commit2.get("supporting_clause_ids") or [])
                )
            )
            commit_anchor_clause_ids = commit_clause_ids
            if not e12_opened_clause_ids:
                e12_opened_clause_ids = list(
                    dict.fromkeys(thread_state.get("opened_history_ids", []))
                )
            if not full_episode_clause_ids:
                full_episode_clause_ids = list(
                    dict.fromkeys(e12_opened_clause_ids + e3_packed_clause_ids)
                )
        if use_pivot_gold_eval:
            try:
                pivot_gold = _build_gold_from_context(world, eval_context_for_judge)
                pivot_gold, pivot_gold_constraint_applied = _apply_constraint_updates_to_gold(
                    pivot_gold,
                    pivot_type=pivot_type,
                    constraint_updates=pivot_constraint_updates,
                )
                eval_gold = pivot_gold
                eval_gold_is_pivot = True
            except Exception:
                pivot_gold = None
                eval_gold = task.gold
                eval_gold_is_pivot = False
                pivot_gold_constraint_applied = False
        else:
            eval_gold = task.gold
            eval_gold_is_pivot = False
            pivot_gold_constraint_applied = False
        primary_results = diag.get("primary_search_results", []) if isinstance(diag, dict) else []
        gold_core_ids = list(
            getattr(eval_gold, "gold_evidence_core", None) or eval_gold.gold_evidence or []
        )
        retrieval_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(eval_gold.gold_evidence or []),
            search_results=primary_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=args.save_search_snapshot,
            snapshot_k=args.search_snapshot_k,
        )
        retrieval_diag_core = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=gold_core_ids,
            search_results=primary_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
            snapshot_k=args.search_snapshot_k,
        )
        is_two_hop = method == "goc" and agent_query_policy == "two_hop_bridge"
        if is_two_hop:
            hop1_results = (
                diag.get("hop1_search_results")
                if isinstance(diag, dict) and "hop1_search_results" in diag
                else []
            )
            hop2_results = (
                diag.get("hop2_search_results")
                if isinstance(diag, dict) and "hop2_search_results" in diag
                else []
            )
            hop1_results = hop1_results or []
            hop2_results = hop2_results or []
        else:
            hop1_results = primary_results
            hop2_results = []
        union_results = merge_search_results_union(hop1_results, hop2_results)
        hop1_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(eval_gold.gold_evidence or []),
            search_results=hop1_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        hop2_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(eval_gold.gold_evidence or []),
            search_results=hop2_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        union_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(eval_gold.gold_evidence or []),
            search_results=union_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        winning_clause_id = (eval_gold.gold_evidence or [None])[0]
        eval_critical_core_clause_ids = list(
            getattr(eval_gold, "gold_evidence_core", None) or eval_gold.gold_evidence or []
        )
        if not eval_critical_core_clause_ids:
            eval_critical_core_clause_ids = list(getattr(task, "critical_core_clause_ids", None) or [])
        if method == "goc" and threaded_mode and episode_id == 3:
            eval_critical_ids_set = set(_unique_strs(eval_critical_core_clause_ids))
            if eval_critical_ids_set:
                eval_present = eval_critical_ids_set & set(e3_context_clause_ids)
                critical_missing_ids = sorted(eval_critical_ids_set - eval_present)
                critical_coverage_e3 = float(len(eval_present)) / float(len(eval_critical_ids_set))
                goc_unfolded_critical_clause_count = sum(
                    1 for cid in eval_critical_ids_set if cid in set(e3_packed_clause_ids)
                )
        if threaded_mode and thread_state:
            _c1 = thread_state.get("commit1") or {}
            _c2 = thread_state.get("commit2") or {}
            commit1_quality = _compute_commit_quality(
                anchor_clause_ids=list(_c1.get("anchor_clause_ids") or []),
                supporting_clause_ids=list(_c1.get("supporting_clause_ids") or []),
                eval_core_clause_ids=eval_critical_core_clause_ids,
            )
            commit2_quality = _compute_commit_quality(
                anchor_clause_ids=list(_c2.get("anchor_clause_ids") or []),
                supporting_clause_ids=list(_c2.get("supporting_clause_ids") or []),
                eval_core_clause_ids=eval_critical_core_clause_ids,
            )
            commit1_correct_anchor = commit1_quality.get("correct_anchor")
            commit1_correct_union = commit1_quality.get("correct_union")
            commit1_anchor_promotion_needed = commit1_quality.get("anchor_promotion_needed")
            commit1_supporting_core_hits = list(commit1_quality.get("supporting_core_hits") or [])
            commit2_correct_anchor = commit2_quality.get("correct_anchor")
            commit2_correct_union = commit2_quality.get("correct_union")
            commit2_anchor_promotion_needed = commit2_quality.get("anchor_promotion_needed")
            commit2_supporting_core_hits = list(commit2_quality.get("supporting_core_hits") or [])
        min_gold_core_rank_hop2 = _min_rank(hop2_results, gold_core_ids)
        min_gold_core_rank_union = _min_rank(union_results, gold_core_ids)
        min_gold_winning_rank_hop2 = _min_rank(
            hop2_results,
            [winning_clause_id] if winning_clause_id else [],
        )
        min_gold_winning_rank_union = _min_rank(
            union_results,
            [winning_clause_id] if winning_clause_id else [],
        )
        deep_rank_core_flag = bool(
            min_gold_core_rank_union is not None
            and min_gold_core_rank_union > DEEP_RANK_CORE_THRESHOLD
        )

        judge_decision = None
        judge_correct = None
        judge_correct_packed_allcritical: bool | None = None
        judge_supporting_clause_ids: List[str] | None = None
        judge_supporting_count: int | None = None
        pivot_compliant: bool | None = None
        stale_evidence: bool | None = None
        pivot_old_days: int | None = None
        pivot_updated_days: int | None = None
        judge_correct_with_pivot_compliance: bool | None = None
        judge_mode = getattr(args, "judge", "llm")
        judge_context_override = (
            dict(eval_context_for_judge)
            if int(episode_id or 0) == 3 and bool(is_pivot_task) and eval_gold_is_pivot
            else None
        )
        judge_constraint_updates = (
            dict(pivot_constraint_updates)
            if int(episode_id or 0) == 3 and bool(is_pivot_task) and eval_gold_is_pivot
            else None
        )
        if judge_mode in {
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        }:
            if judge_mode in {"symbolic_packed", "symbolic_packed_allcritical"} and threaded_mode and episode_id == 3:
                judge_pred = judge_from_opened_clauses(
                    task,
                    e3_packed_clause_ids,
                    world,
                    context=judge_context_override,
                    constraint_updates=judge_constraint_updates,
                )
            elif judge_mode == "symbolic_full_episode" and threaded_mode and episode_id == 3:
                judge_pred = judge_from_opened_clauses(
                    task,
                    full_episode_clause_ids,
                    world,
                    context=judge_context_override,
                    constraint_updates=judge_constraint_updates,
                )
            elif judge_mode == "symbolic" and threaded_mode and episode_id == 3 and commit_clause_ids is not None:
                judge_pred = judge_threaded_final(
                    task,
                    commit_clause_ids,
                    world,
                    context=judge_context_override,
                    constraint_updates=judge_constraint_updates,
                )
            else:
                judge_pred = judge_from_opened_clauses(
                    task,
                    opened_ids,
                    world,
                    context=judge_context_override,
                    constraint_updates=judge_constraint_updates,
                )
            judge_decision = judge_pred.get("decision")
            judge_correct = judge_decision == eval_gold.decision
            judge_supporting_clause_ids = list(
                judge_pred.get("supporting_clause_ids") or []
            )
            judge_supporting_count = len(judge_supporting_clause_ids)
            if judge_mode == "symbolic_full_episode":
                judge_correct_full_episode = judge_correct
                full_episode_supporting_count = judge_supporting_count
        if method == "engine":
            normalize_reason = None
            decision_before_norm = pred.get("decision")
            decision_after_norm = pred.get("decision")
        else:
            pred, normalize_reason, decision_before_norm, decision_after_norm = normalize_prediction_schema(pred)
        pred_for_eval, pred_for_record, evidence_before, evidence_after = apply_evidence_padding(
            pred,
            opened_ids,
            mode=args.evidence_padding_mode,
            min_count=args.min_evidence_count,
        )
        pred_decision_eval = pred_for_eval.get("decision")
        episode_decision_correct = (
            bool(pred_decision_eval == eval_gold.decision)
            if eval_gold is not None
            else None
        )
        pred_conditions_eval_raw = pred_for_eval.get("conditions", [])
        pred_conditions_eval = (
            list(pred_conditions_eval_raw) if isinstance(pred_conditions_eval_raw, list) else []
        )
        gold_conditions_eval = list(getattr(eval_gold, "conditions", None) or [])
        episode_conditions_correct = bool(
            set(map(str, pred_conditions_eval)) == set(map(str, gold_conditions_eval))
        )
        episode_answer_correct = bool(
            bool(episode_decision_correct)
            and (bool(episode_conditions_correct) if episode_conditions_correct is not None else True)
        )
        gold_mode_used = "original"
        if int(episode_id or 0) == 3 and bool(is_pivot_task):
            if pivot_gold_mode == "respect_ticket_updated":
                gold_mode_used = "pivot"
            elif pivot_gold_mode == "both":
                gold_mode_used = "both"
            else:
                gold_mode_used = "original"
        commit1_correct = None
        commit2_correct = None
        if threaded_mode and thread_state:
            commit1 = thread_state.get("commit1") or {}
            commit2 = thread_state.get("commit2") or {}
            commit1_correct = commit1.get("commit_correct")
            commit2_correct = commit2.get("commit_correct")
            if episode_id == 3 and commit_clause_ids is not None:
                commit_correct = bool(commit1_correct is True and commit2_correct is True)
                thread_e3_payload = _compute_thread_e3_correctness(
                    judge_correct=judge_correct,
                    commit1_correct=commit1_correct,
                    commit2_correct=commit2_correct,
                    evidence_after=list(evidence_after or []),
                    opened_ids=list(opened_ids or []),
                    commit_clause_ids=list(commit_clause_ids or []),
                    e3_answer_correct=episode_answer_correct,
                )
                e3_evidence_valid_in_context = thread_e3_payload.get("evidence_valid_in_context")
                e3_evidence_valid_in_commits = thread_e3_payload.get("evidence_valid_in_commits")
                thread_strict_correct = thread_e3_payload.get("strict_correct")
                thread_e3_only_correct = thread_e3_payload.get("e3_only_correct")
                thread_judge_correct = thread_strict_correct
                if isinstance(thread_strict_correct, bool):
                    judge_correct = thread_strict_correct
                if judge_mode == "symbolic_full_episode":
                    judge_correct_full_episode = judge_correct
                if isinstance(judge_correct, bool):
                    judge_correct_packed_allcritical = bool(
                        judge_correct and e3_packed_all_critical is True
                    )
                if judge_mode == "symbolic_packed_allcritical":
                    judge_correct = judge_correct_packed_allcritical
        elif threaded_mode and episode_id == 3 and isinstance(judge_correct, bool):
            judge_correct_packed_allcritical = bool(
                judge_correct and e3_packed_all_critical is True
            )
            if judge_mode == "symbolic_full_episode":
                judge_correct_full_episode = judge_correct
        if threaded_mode and episode_id == 3:
            if e3_evidence_valid_in_context is None:
                e3_evidence_valid_in_context = all(cid in set(opened_ids or []) for cid in (evidence_after or []))
            if e3_evidence_valid_in_commits is None and commit_clause_ids is not None:
                e3_evidence_valid_in_commits = all(
                    cid in set(commit_clause_ids or []) for cid in (evidence_after or [])
                )
            if thread_e3_only_correct is None and isinstance(episode_answer_correct, bool):
                thread_e3_only_correct = bool(episode_answer_correct and bool(e3_evidence_valid_in_context))
        if threaded_mode:
            if int(episode_id or 0) == 1:
                commit_correct_anchor = commit1_correct_anchor
                commit_correct_union = commit1_correct_union
                commit_anchor_promotion_needed = commit1_anchor_promotion_needed
                commit_supporting_core_hits = list(commit1_supporting_core_hits)
                if isinstance(commit_correct_anchor, bool):
                    commit_correct = commit_correct_anchor
            elif int(episode_id or 0) == 2:
                commit_correct_anchor = commit2_correct_anchor
                commit_correct_union = commit2_correct_union
                commit_anchor_promotion_needed = commit2_anchor_promotion_needed
                commit_supporting_core_hits = list(commit2_supporting_core_hits)
                if isinstance(commit_correct_anchor, bool):
                    commit_correct = commit_correct_anchor
            elif int(episode_id or 0) == 3:
                if commit1_correct_anchor is not None and commit2_correct_anchor is not None:
                    commit_correct_anchor = bool(
                        commit1_correct_anchor is True and commit2_correct_anchor is True
                    )
                if commit1_correct_union is not None and commit2_correct_union is not None:
                    commit_correct_union = bool(
                        commit1_correct_union is True and commit2_correct_union is True
                    )
                if commit1_anchor_promotion_needed is not None or commit2_anchor_promotion_needed is not None:
                    commit_anchor_promotion_needed = bool(
                        bool(commit1_anchor_promotion_needed)
                        or bool(commit2_anchor_promotion_needed)
                    )
                commit_supporting_core_hits = sorted(
                    set(commit1_supporting_core_hits) | set(commit2_supporting_core_hits)
                )
                if isinstance(commit_correct_anchor, bool):
                    commit_correct = commit_correct_anchor
        if threaded_mode and episode_id == 3:
            if e3_context_chars_used is None:
                e3_context_chars_used = 0
            if e3_packed_total_chars_before is None:
                e3_packed_total_chars_before = 0
            if e3_packed_total_chars_after is None:
                e3_packed_total_chars_after = 0
            if e3_packed_truncated is None:
                e3_packed_truncated = False
            if e3_packed_dropped_clause_count is None:
                e3_packed_dropped_clause_count = 0
            if e3_packed_clause_count is None:
                e3_packed_clause_count = len(e3_packed_clause_ids or [])
            if e3_context_token_est is None:
                e3_context_token_est = int(math.ceil(max(0, e3_context_chars_used) / 4.0))
            if e3_packed_token_est is None:
                e3_packed_token_est = int(math.ceil(max(0, e3_packed_total_chars_after) / 4.0))
        if threaded_mode:
            if not e12_opened_clause_ids and thread_state:
                e12_opened_clause_ids = list(
                    dict.fromkeys(thread_state.get("opened_history_ids", []))
                )
            if not full_episode_clause_ids:
                full_episode_clause_ids = list(e12_opened_clause_ids)
        if is_pivot_task and (not threaded_mode or episode_id == 3):
            (
                pivot_compliant,
                stale_evidence,
                pivot_old_days,
                pivot_updated_days,
            ) = _compute_pivot_compliance(
                ticket_initial=ticket_initial,
                ticket_updated=ticket_updated,
                pivot_type=pivot_type,
                raw_output=raw_output,
            )
            if isinstance(judge_correct, bool) and isinstance(pivot_compliant, bool):
                # Keep pivot_compliant as a sanity diagnostic; do not override main correctness.
                judge_correct_with_pivot_compliance = bool(judge_correct and pivot_compliant)
        task_metrics = evaluate_prediction(pred_for_eval, eval_gold, world)
        if pivot_gold_mode == "both":
            orig_task_metrics = evaluate_prediction(pred_for_eval, task.gold, world)
        metrics.append(task_metrics)
        tool_calls.append(env.tool_call_count)
        open_calls.append(env.open_count)
        prompt_tokens = len(prompt.split()) if prompt else 0
        prompt_tokens_list.append(prompt_tokens)
        pred_decision = pred_for_record.get("decision")
        bridge_clause_id = getattr(task, "bridge_clause_id", None)
        bridge_found = bool(bridge_clause_id and bridge_clause_id in opened_ids)
        update_found_when_needed = False
        if getattr(task, "needs_update_resolution", False):
            update_found_when_needed = any(
                world.clauses.get(cid) and world.clauses[cid].kind == "update" for cid in opened_ids
            )
        winning_rank_exists = retrieval_diag.get("winning_clause_rank") is not None
        action_reward = None
        action_opened_has_winning = None
        action_opened_gold_coverage = None
        reward_breakdown = None
        if method == "goc" and controller_mode == "train":
            winning_clause = eval_gold.gold_evidence[0] if eval_gold.gold_evidence else None
            action_opened_has_winning = (
                bool(winning_clause and winning_clause in opened_ids)
                if opened_ids
                else False
            )
            action_opened_gold_coverage = retrieval_diag.get("opened_gold_coverage")
            r_win = 1.0 if action_opened_has_winning else 0.0
            r_cov = 0.5 * float(action_opened_gold_coverage or 0.0)
            r_open_penalty = -0.01 * float(env.open_count)
            action_reward = r_win + r_cov + r_open_penalty
            reward_breakdown = {
                "r_win": r_win,
                "r_cov": r_cov,
                "r_open_penalty": r_open_penalty,
                "r_total": action_reward,
            }
        opened_decoy_clause_count = 0
        if opened_ids:
            for cid in opened_ids:
                clause = world.clauses.get(cid)
                if clause and str(clause.doc_id).startswith("DECOY"):
                    opened_decoy_clause_count += 1
        capture_goc_advantages = method in {"goc", "similarity_only"}
        opened_for_prompt_clause_ids = (
            list(diag.get("opened_for_prompt_clause_ids") or [])
            if isinstance(diag, dict) and isinstance(diag.get("opened_for_prompt_clause_ids"), list)
            else []
        )
        opened_total_clause_ids = (
            list(diag.get("opened_total_clause_ids") or [])
            if isinstance(diag, dict) and isinstance(diag.get("opened_total_clause_ids"), list)
            else []
        )
        opened_evidence_clause_ids = _unique_strs(
            opened_total_clause_ids or opened_for_prompt_clause_ids or list(opened_ids)
        )
        opened_evidence_doc_ids = _clause_ids_to_doc_ids(opened_evidence_clause_ids, world)
        active_context_node_ids: List[str] = []
        active_context_clause_ids: List[str] = []
        active_context_doc_ids: List[str] = []
        unfolded_activated_node_ids: List[str] = []
        unfolded_activated_clause_ids: List[str] = []
        unfolded_activated_doc_ids: List[str] = []
        closure_recalled_clause_ids: List[str] = []
        closure_recall_core: float | None = None
        wrong_branch_recall_rate: float | None = None
        if capture_goc_advantages:
            final_internal_snapshot = _pick_final_snapshot(goc_internal_snapshots)
            (
                active_context_node_ids,
                active_context_clause_ids,
                active_context_doc_ids,
            ) = _extract_active_from_snapshot(final_internal_snapshot, world)
            if not active_context_clause_ids:
                if threaded_mode and episode_id == 3 and e3_context_clause_ids:
                    active_context_clause_ids = _unique_strs(list(e3_context_clause_ids))
                elif opened_for_prompt_clause_ids:
                    active_context_clause_ids = _unique_strs(opened_for_prompt_clause_ids)
                else:
                    active_context_clause_ids = _unique_strs(opened_evidence_clause_ids)
            if not active_context_doc_ids:
                active_context_doc_ids = _clause_ids_to_doc_ids(active_context_clause_ids, world)

            if method == "goc":
                unfolded_from_diag = (
                    list(diag.get("goc_unfold_selected_clause_ids") or [])
                    if isinstance(diag, dict)
                    and isinstance(diag.get("goc_unfold_selected_clause_ids"), list)
                    else []
                )
                unfolded_activated_clause_ids = _unique_strs(
                    unfolded_from_diag or list(goc_unfold_selected_clause_ids)
                )
                (
                    unfolded_activated_node_ids,
                    unfolded_docids_from_nodes,
                ) = _match_snapshot_nodes_for_clause_ids(
                    final_internal_snapshot,
                    unfolded_activated_clause_ids,
                )
                if not unfolded_activated_node_ids and unfolded_activated_clause_ids:
                    unfolded_activated_node_ids = list(unfolded_activated_clause_ids)
                world_doc_ids = _world_docid_set(world)
                unfolded_activated_doc_ids = _unique_strs(
                    _clause_ids_to_doc_ids(unfolded_activated_clause_ids, world)
                    + [d for d in unfolded_docids_from_nodes if d in world_doc_ids]
                )

            closure_recalled_clause_ids = _unique_strs(
                unfolded_activated_clause_ids + active_context_clause_ids
            )
            if gold_core_ids:
                core_set = set(_unique_strs(gold_core_ids))
                closure_recall_core = (
                    len(core_set & set(closure_recalled_clause_ids)) / float(len(core_set))
                    if core_set
                    else None
                )
            if _is_threaded_or_bridged(scenario_mode):
                recall_ids = closure_recalled_clause_ids or unfolded_activated_clause_ids
                if recall_ids:
                    wrong_branch_hits = sum(
                        1
                        for cid in recall_ids
                        if _is_non_winning_branch_clause(task, world, str(cid))
                    )
                    wrong_branch_recall_rate = wrong_branch_hits / float(len(recall_ids))
                else:
                    wrong_branch_recall_rate = 0.0
        record: Dict[str, Any] = {
            "task_id": task.task_id,
            "method": method,
            "ticket_initial": ticket_initial or None,
            "ticket_updated": ticket_updated or None,
            "pivot_type": pivot_type or None,
            "is_pivot_task": bool(is_pivot_task),
            "pivot_message_style": pivot_message_style,
            "pivot_compliant": pivot_compliant,
            "stale_evidence": stale_evidence,
            "pivot_old_days": pivot_old_days,
            "pivot_updated_days": pivot_updated_days,
            "opened_clause_ids": opened_ids,
            "opened_decoy_clause_count": opened_decoy_clause_count,
            "opened_decoy_present": opened_decoy_clause_count > 0,
            "tool_calls": env.tool_call_count,
            "open_calls": env.open_count,
            "open_budget": env.open_budget,
            "prompt_tokens": prompt_tokens,
            "pred_decision": pred_decision,
            "decision_before_normalize": decision_before_norm,
            "decision_after_normalize": decision_after_norm,
            "normalize_reason": normalize_reason,
            "gold_decision": eval_gold.decision,
            "decision_correct": pred_decision == eval_gold.decision,
            "conditions_correct": episode_conditions_correct,
            "answer_correct": episode_answer_correct,
            "gold_mode_used": gold_mode_used,
            "gold_decision_original": task.gold.decision,
            "decision_correct_original": pred_decision == task.gold.decision,
            "pivot_gold_mode": pivot_gold_mode,
            "eval_gold_is_pivot": bool(eval_gold_is_pivot),
            "pivot_gold_decision": pivot_gold.decision if pivot_gold is not None else None,
            "pivot_gold_conditions": (
                list(getattr(pivot_gold, "conditions", None) or [])
                if pivot_gold is not None
                else []
            ),
            "pivot_gold_evidence_ids": list(pivot_gold.gold_evidence or []) if pivot_gold is not None else [],
            "pivot_gold_evidence_core_ids": (
                list(getattr(pivot_gold, "gold_evidence_core", None) or pivot_gold.gold_evidence or [])
                if pivot_gold is not None
                else []
            ),
            "pivot_gold_evidence_meta_ids": (
                list(getattr(pivot_gold, "gold_evidence_meta", None) or [])
                if pivot_gold is not None
                else []
            ),
            "pivot_gold_constraint_applied": bool(pivot_gold_constraint_applied),
            "pivot_constraint_updates": dict(pivot_constraint_updates),
            "pivot_constraint_keys": list(pivot_constraint_keys),
            "pivot_constraint_key": pivot_constraint_key,
            "pivot_constraint_value": pivot_constraint_value,
            "answer_parse_failed": answer_parse_failed,
            "judge_decision": judge_decision,
            "judge_correct": judge_correct,
            "judge_correct_with_pivot_compliance": judge_correct_with_pivot_compliance,
            "judge_correct_packed_allcritical": judge_correct_packed_allcritical,
            "judge_correct_full_episode": judge_correct_full_episode,
            "judge_supporting_clause_ids": judge_supporting_clause_ids,
            "judge_supporting_count": judge_supporting_count,
            "full_episode_supporting_count": full_episode_supporting_count,
            "scenario_mode": scenario_mode,
            "thread_id": thread_id,
            "episode_id": episode_id,
            "episode_kind": episode_kind,
            "thread_config": getattr(task, "thread_config", None),
            "agent_query_policy": agent_query_policy,
            "open_policy": getattr(args, "open_policy", "current"),
            "slot": task.context.get("slot") if isinstance(getattr(task, "context", None), dict) else None,
            "slot_term": (
                getattr(task, "canonical_slot_term", None)
                or (task.context.get("slot") if isinstance(getattr(task, "context", None), dict) else None)
            ),
            "slot_hint_alias": getattr(task, "slot_hint_alias", None),
            "canonical_slot_term": getattr(task, "canonical_slot_term", None),
            "bridge_clause_id": bridge_clause_id,
            "needs_update_resolution": getattr(task, "needs_update_resolution", False),
            "bridge_found": bridge_found,
            "canonical_used_in_query2": bool(diag.get("hop2_query_contains_canonical"))
            if isinstance(diag, dict)
            else False,
            "update_found_when_needed": update_found_when_needed,
            "winning_rank_exists": winning_rank_exists,
            "decision_accuracy": task_metrics.get("decision_accuracy"),
            "condition_f1": task_metrics.get("condition_f1"),
            "evidence_precision": task_metrics.get("evidence_precision"),
            "evidence_recall": task_metrics.get("evidence_recall"),
            "evidence_precision_core": task_metrics.get("evidence_precision_core"),
            "evidence_recall_core": task_metrics.get("evidence_recall_core"),
            "critical_evidence_hit": task_metrics.get("critical_evidence_hit"),
            "orig_decision_accuracy": (
                orig_task_metrics.get("decision_accuracy") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_condition_f1": (
                orig_task_metrics.get("condition_f1") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_evidence_precision": (
                orig_task_metrics.get("evidence_precision") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_evidence_recall": (
                orig_task_metrics.get("evidence_recall") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_evidence_precision_core": (
                orig_task_metrics.get("evidence_precision_core") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_evidence_recall_core": (
                orig_task_metrics.get("evidence_recall_core") if isinstance(orig_task_metrics, dict) else None
            ),
            "orig_critical_evidence_hit": (
                orig_task_metrics.get("critical_evidence_hit") if isinstance(orig_task_metrics, dict) else None
            ),
            "pred_evidence_count": len(pred_for_record.get("evidence", []) or []),
            "gold_evidence_count": len(eval_gold.gold_evidence or []),
            "gold_evidence_count_original": len(task.gold.gold_evidence or []),
            "gold_evidence_ids": list(eval_gold.gold_evidence or []),
            "gold_evidence_core_ids": list(
                getattr(eval_gold, "gold_evidence_core", None) or eval_gold.gold_evidence or []
            ),
            "gold_evidence_meta_ids": list(getattr(eval_gold, "gold_evidence_meta", None) or []),
            "gold_evidence_ids_original": list(task.gold.gold_evidence or []),
            "gold_evidence_core_ids_original": list(
                getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
            ),
            "gold_evidence_meta_ids_original": list(getattr(task.gold, "gold_evidence_meta", None) or []),
            "meta_evidence_present": bool(getattr(eval_gold, "gold_evidence_meta", None)),
            "core_evidence_size": len(
                getattr(eval_gold, "gold_evidence_core", None) or eval_gold.gold_evidence or []
            ),
            "evidence_before_pad": evidence_before,
            "evidence_after_pad": evidence_after,
            "error": error,
            "evidence_padding_mode": args.evidence_padding_mode,
            "min_evidence_count": args.min_evidence_count,
            "controller_action": controller_action,
            "controller_action_reward": action_reward,
            "controller_action_opened_gold_coverage": action_opened_gold_coverage,
            "controller_action_opened_has_winning_clause": action_opened_has_winning,
            "controller_reward_breakdown": reward_breakdown,
            "commit_supporting_clause_ids": commit_supporting_clause_ids,
            "commit_anchor_clause_ids": commit_anchor_clause_ids,
            "commit_anchor_source_used": commit_anchor_source_used,
            "commit_model_evidence_clause_ids": commit_model_evidence_clause_ids,
            "commit_short_fact": commit_short_fact,
            "commit_correct": commit_correct,
            "commit_correct_anchor": commit_correct_anchor,
            "commit_correct_union": commit_correct_union,
            "commit_anchor_promotion_needed": commit_anchor_promotion_needed,
            "commit_supporting_core_hits": list(commit_supporting_core_hits),
            "commit1_correct_anchor": commit1_correct_anchor,
            "commit1_correct_union": commit1_correct_union,
            "commit1_anchor_promotion_needed": commit1_anchor_promotion_needed,
            "commit1_supporting_core_hits": list(commit1_supporting_core_hits),
            "commit2_correct_anchor": commit2_correct_anchor,
            "commit2_correct_union": commit2_correct_union,
            "commit2_anchor_promotion_needed": commit2_anchor_promotion_needed,
            "commit2_supporting_core_hits": list(commit2_supporting_core_hits),
            "commit_clause_ids": commit_clause_ids,
            "e3_evidence_valid": e3_evidence_valid_in_context,
            "e3_evidence_valid_in_context": e3_evidence_valid_in_context,
            "e3_evidence_valid_in_commits": e3_evidence_valid_in_commits,
            "thread_strict_correct": thread_strict_correct,
            "thread_e3_only_correct": thread_e3_only_correct,
            "thread_judge_correct": thread_judge_correct,
            "critical_clause_id_e1": getattr(task, "critical_clause_id_e1", None),
            "critical_clause_id_e2": getattr(task, "critical_clause_id_e2", None),
            "critical_core_clause_ids": list(getattr(task, "critical_core_clause_ids", None) or []),
            "eval_critical_core_clause_ids": list(eval_critical_core_clause_ids),
            "e3_context_budget_chars": e3_context_budget_chars,
            "e3_context_chars_used": e3_context_chars_used,
            "e3_context_clause_count": e3_context_clause_count,
            "e3_packed_clause_count": e3_packed_clause_count,
            "e3_context_token_est": e3_context_token_est,
            "e3_packed_token_est": e3_packed_token_est,
            "e3_context_truncated": e3_context_truncated,
            "e3_context_clause_ids": e3_context_clause_ids,
            "e12_opened_clause_ids": list(e12_opened_clause_ids),
            "full_episode_clause_ids": list(full_episode_clause_ids),
            "e3_packed_clause_lens": list(e3_packed_clause_lens),
            "e3_packed_clause_is_critical": list(e3_packed_clause_is_critical),
            "e3_packed_total_chars_before": e3_packed_total_chars_before,
            "e3_packed_total_chars_after": e3_packed_total_chars_after,
            "e3_packed_truncated": e3_packed_truncated,
            "e3_packed_dropped_clause_count": e3_packed_dropped_clause_count,
            "e3_decoy_clause_count": e3_decoy_clause_count,
            "e3_prompt_includes_required_core": e3_prompt_includes_required_core,
            "e3_prompt_includes_critical_core": e3_prompt_includes_critical_core,
            "e3_truncation_loss_estimate": e3_truncation_loss_estimate,
            "e3_packed_clause_ids": list(e3_packed_clause_ids),
            "e3_packed_contains_critical": e3_packed_contains_critical,
            "e3_packed_contains_critical0": e3_packed_contains_critical0,
            "e3_packed_contains_critical1": e3_packed_contains_critical1,
            "e3_packed_critical_count": e3_packed_critical_count,
            "e3_packed_any_critical": e3_packed_any_critical,
            "e3_packed_all_critical": e3_packed_all_critical,
            "e3_packed_critical_count_full_episode": e3_packed_critical_count_full_episode,
            "e3_packed_any_critical_full_episode": e3_packed_any_critical_full_episode,
            "e3_packed_all_critical_full_episode": e3_packed_all_critical_full_episode,
            "e3_litm_filler_count": e3_litm_filler_count,
            "e3_litm_filler_position": e3_litm_filler_position,
            "e3_litm_filler_clause_ids": list(e3_litm_filler_clause_ids),
            "goc_folded_episode_count": goc_folded_episode_count,
            "goc_unfolded_clause_count": goc_unfolded_clause_count,
            "goc_unfolded_critical_clause_count": goc_unfolded_critical_clause_count,
            "goc_unfold_selected_clause_ids": goc_unfold_selected_clause_ids,
            "goc_unfold_reason": goc_unfold_reason,
            "goc_unfold_max_nodes": goc_unfold_max_nodes_used if method == "goc" else None,
            "goc_unfold_hops": goc_unfold_hops_used if method == "goc" else None,
            "goc_unfold_budget_mode": goc_unfold_budget_mode_cfg if method == "goc" else None,
            "goc_closure_candidate_count": goc_closure_candidate_count,
            "goc_unfolded_node_count": goc_unfolded_node_count,
            "goc_selected_node_ids": list(goc_selected_node_ids),
            "goc_unfolded_node_ids": list(goc_unfolded_node_ids),
            "goc_enable_avoids": bool(goc_enable_avoids) if method == "goc" else None,
            "goc_avoids_mode": str(goc_avoids_mode) if method == "goc" else None,
            "goc_initial_ticket_node_id": goc_initial_ticket_node_id,
            "goc_pivot_ticket_node_id": goc_pivot_ticket_node_id,
            "goc_avoids_edge_injected": bool(goc_avoids_edge_injected) if method == "goc" else None,
            "goc_avoid_target_clause_ids": list(goc_avoid_target_clause_ids),
            "goc_avoided_node_injected": goc_avoided_node_injected,
            "goc_inapplicable_clause_ids": list(goc_inapplicable_clause_ids),
            "goc_applicability_seed_enable": bool(goc_applicability_seed_enable_cfg),
            "goc_applicability_seed_topk": int(goc_applicability_seed_topk_cfg),
            "goc_applicability_seed_ids": list(goc_applicability_seed_ids),
            "goc_applicability_seed_used": goc_applicability_seed_used,
            "goc_applicability_seed_applicable_rate": goc_applicability_seed_applicable_rate,
            "goc_dependency_closure_enable": bool(goc_dependency_closure_enable_cfg),
            "goc_dependency_closure_topk": int(goc_dependency_closure_topk_cfg),
            "goc_dependency_closure_hops": int(goc_dependency_closure_hops_cfg),
            "goc_dependency_closure_universe": str(goc_dependency_closure_universe_cfg),
            "goc_dependency_closure_added_ids": list(goc_dependency_closure_added_ids),
            "goc_dependency_closure_added_used": goc_dependency_closure_added_used,
            "goc_dependency_closure_added_applicable_rate": goc_dependency_closure_added_applicable_rate,
            "goc_effective_context": dict(goc_effective_context),
            "goc_initial_context": dict(goc_initial_context),
            "pivot_override_keys": list(pivot_override_keys),
            "pivot_context_delta": dict(pivot_context_delta),
            "critical_coverage_e3": critical_coverage_e3,
            "critical_missing_ids": list(critical_missing_ids),
            "inapplicable_injected_rate_e3": inapplicable_injected_rate_e3,
            "inapplicable_injected_clause_ids_e3": list(inapplicable_injected_clause_ids),
            "opened_evidence_clause_ids": opened_evidence_clause_ids,
            "opened_evidence_doc_ids": opened_evidence_doc_ids,
            "active_context_node_ids": active_context_node_ids,
            "active_context_clause_ids": active_context_clause_ids,
            "active_context_doc_ids": active_context_doc_ids,
            "unfolded_activated_node_ids": unfolded_activated_node_ids,
            "unfolded_activated_clause_ids": unfolded_activated_clause_ids,
            "unfolded_activated_doc_ids": unfolded_activated_doc_ids,
            "closure_recalled_clause_ids": closure_recalled_clause_ids,
            "closure_recall_core": closure_recall_core,
            "wrong_branch_recall_rate": wrong_branch_recall_rate,
            "min_gold_core_rank_hop2": min_gold_core_rank_hop2,
            "min_gold_core_rank_union": min_gold_core_rank_union,
            "min_gold_winning_rank_hop2": min_gold_winning_rank_hop2,
            "min_gold_winning_rank_union": min_gold_winning_rank_union,
            "deep_rank_core_flag": deep_rank_core_flag,
        }
        episode_num = int(episode_id or 0)
        if episode_num in {1, 2, 3}:
            ep_key = f"e{episode_num}"
            record[f"{ep_key}_decision_correct"] = episode_decision_correct
            record[f"{ep_key}_conditions_correct"] = episode_conditions_correct
            record[f"{ep_key}_answer_correct"] = episode_answer_correct
            if episode_num == 3:
                record["e3_answer_correct"] = episode_answer_correct
        if isinstance(diag, dict):
            record.update(diag)
        record.setdefault("shared_open_policy_applied", False)
        record.setdefault("opened_clause_ids_shared", [])
        record.setdefault("opened_doc_ids_shared", [])
        if threaded_mode:
            distractor_id = getattr(task, "branch_distractor_clause_id", None)
            record["branch_distractor_opened"] = bool(distractor_id and distractor_id in opened_ids)
            record["branch_trap"] = bool(
                episode_id == 2
                and record.get("branch_distractor_opened")
                and record.get("commit_correct") is False
            )
            record["fold_drift"] = bool(
                episode_id == 3
                and record.get("compose_strategy") in {"full_history", "agent_fold"}
                and record.get("commit_correct") is True
                and record.get("thread_judge_correct") is False
            )
            record["cost_blowup"] = bool(prompt_tokens > COST_BLOWUP_TOKENS)
        record["hop1_search_results"] = hop1_results
        record["hop2_search_results"] = hop2_results
        record["search_results_union"] = union_results
        if method != "goc":
            record.setdefault("hop2_executed", False)
            record.setdefault("hop2_skip_reason", "no_hop2_method")
            record.setdefault("hop2_query", "")
            record.setdefault("hop2_candidate_query", "")
            record.setdefault("hop2_query_contains_canonical", False)
            record.setdefault("hop2_query_contains_gold_canonical", False)
        if agent_query_policy != "two_hop_bridge":
            record.setdefault("hop2_executed", False)
            record.setdefault("hop2_skip_reason", record.get("hop2_skip_reason") or "no_hop2_method")
            record.setdefault("hop2_query", record.get("hop2_query") or "")
            record.setdefault("hop2_candidate_query", record.get("hop2_candidate_query") or "")
            record.setdefault("hop2_query_contains_canonical", False)
        # Gold-canonical diagnostic (analysis only; no policy impact)
        gold_canonical = (getattr(task, "canonical_slot_term", None) or "").lower()
        hop2_q = (record.get("hop2_query") or record.get("hop2_candidate_query") or "")
        record["hop2_query_contains_gold_canonical"] = (
            bool(gold_canonical) and gold_canonical in hop2_q.lower()
        )
        # Bridge canonical diagnostics (analysis-only)
        probe_id = record.get("bridge_probe_clause_id")
        opened_ids_set = set(record.get("opened_total_clause_ids") or record.get("opened_clause_ids") or [])
        bridge_probe_text = ""
        if probe_id and probe_id in world.clauses:
            bridge_probe_text = world.clauses[probe_id].text.lower()
        record["bridge_probe_contains_gold_canonical"] = (
            bool(gold_canonical) and gold_canonical in bridge_probe_text
        )
        record["bridge_opened_contains_gold_canonical"] = (
            bool(gold_canonical)
            and any(
                (world.clauses.get(cid) and gold_canonical in world.clauses[cid].text.lower())
                for cid in opened_ids_set
                if cid in world.clauses
            )
        )
        if not args.save_search_snapshot:
            record.pop("rewrite_queries", None)
            record.pop("rewrite_used", None)
            record.pop("hop1_query_text", None)
        if scenario_mode == "bridged_v1_1":
            for key, default in [
                ("bridge_needed", bool(bridge_clause_id)),
                ("bridge_opened_any", False),
                ("bridge_opened_gold", False),
                ("bridge_probe_clause_id", None),
                ("bridge_probe_is_slot_specific", None),
                ("bridge_gold_clause_id", getattr(task, "bridge_clause_id", None)),
                ("hop2_candidate_query", ""),
                ("hop2_executed", False),
                ("hop2_query", ""),
                ("hop2_query_contains_canonical", False),
                ("hop2_skip_reason", None),
                ("open_from_hop1_count", 0),
                ("open_from_hop2_count", 0),
                ("prompt_includes_from_hop2", False),
            ]:
                record.setdefault(key, default)
            opened_ids_set = set(record.get("opened_total_clause_ids") or record.get("opened_clause_ids") or [])
            if record.get("bridge_opened_any") is False:
                record["bridge_opened_any"] = any(
                    (cid in world.clauses)
                    and (
                        bool(getattr(world.clauses[cid], "bridge_for_slot", None))
                        or bool(getattr(world.clauses[cid], "is_bridge_doc", False))
                    )
                    for cid in opened_ids_set
                )
            if record.get("bridge_opened_gold") is False and bridge_clause_id:
                record["bridge_opened_gold"] = bridge_clause_id in opened_ids_set
        record.update(retrieval_diag)
        opened_ids_set = set(record.get("opened_total_clause_ids") or record.get("opened_clause_ids") or [])
        opened_bridge_count = sum(
            1 for cid in opened_ids_set if _is_bridge_clause(world.clauses.get(cid))
        )
        opened_meta_count = sum(
            1 for cid in opened_ids_set if _is_meta_clause(world.clauses.get(cid))
        )
        opened_rule_count = sum(
            1 for cid in opened_ids_set if _is_rule_clause(world.clauses.get(cid))
        )
        record.setdefault("opened_bridge_count", opened_bridge_count)
        record.setdefault("opened_meta_count", opened_meta_count)
        record.setdefault("opened_rule_count", opened_rule_count)
        record.setdefault("bridge_open_cap_hit", False)
        record.setdefault("meta_avoided_count", 0)
        record.setdefault(
            "hop2_pool_used_count",
            record.get("open_from_hop2_count", 0),
        )
        record.setdefault("fallback_reason", None)
        hit_hop1 = (
            hop1_diag.get("winning_clause_rank") is not None
            and hop1_diag.get("winning_clause_rank") <= (record.get("open_budget") or 0)
        )
        hit_hop2 = (
            hop2_diag.get("winning_clause_rank") is not None
            and hop2_diag.get("winning_clause_rank") <= (record.get("open_budget") or 0)
        )
        hit_union = (
            union_diag.get("winning_clause_rank") is not None
            and union_diag.get("winning_clause_rank") <= (record.get("open_budget") or 0)
        )
        union_ids = {
            item.get("clause_id") for item in (union_results or []) if item.get("clause_id")
        }
        winning_clause = ((record.get("gold_evidence_ids") or [None])[0] if isinstance(record, dict) else None)
        core_id_set = set(record.get("gold_evidence_core_ids") or [])
        winning_in_union = bool(
            (winning_clause and winning_clause in union_ids) or (union_ids & core_id_set)
        )
        record.update(
            {
                "gold_in_search_topk_hop1": hop1_diag.get("gold_in_search_topk"),
                "gold_in_search_topk_hop2": hop2_diag.get("gold_in_search_topk"),
                "gold_in_search_topk_union": union_diag.get("gold_in_search_topk"),
                "winning_clause_rank_hop1": hop1_diag.get("winning_clause_rank"),
                "winning_clause_rank_hop2": hop2_diag.get("winning_clause_rank"),
                "winning_clause_rank_union": union_diag.get("winning_clause_rank"),
                "opened_has_winning_clause_union": union_diag.get("opened_has_winning_clause"),
                "winning_in_union": winning_in_union,
                "hit_at_open_budget_hop1": hit_hop1,
                "hit_at_open_budget_hop2": hit_hop2,
                "hit_at_open_budget_union": hit_union,
            }
        )
        opened_has_union = bool(record.get("opened_has_winning_clause_union"))
        record["rank_success"] = bool(hit_union)
        record["policy_gain_over_rank"] = (1.0 if opened_has_union else 0.0) - (
            1.0 if hit_union else 0.0
        )
        record["rank_gap"] = (1.0 if hit_union else 0.0) - (1.0 if opened_has_union else 0.0)
        feasible_open_rate = 1.0 if record.get("hit_at_open_budget_union") else 0.0
        realized_open_rate = (
            1.0 if record.get("opened_has_winning_clause_union") else 0.0
        )
        record["feasible_open_rate"] = feasible_open_rate
        record["realized_open_rate"] = realized_open_rate
        record["selection_gap"] = max(0.0, feasible_open_rate - realized_open_rate)
        record["selection_efficiency"] = (
            realized_open_rate / feasible_open_rate if feasible_open_rate > 0 else None
        )
        record.update(
            {
                "opened_gold_count_core": retrieval_diag_core.get("opened_gold_count"),
                "opened_gold_coverage_core": retrieval_diag_core.get("opened_gold_coverage"),
                "opened_has_winning_clause_core": retrieval_diag_core.get(
                    "opened_has_winning_clause"
                ),
                "gold_in_search_topk_core": retrieval_diag_core.get("gold_in_search_topk"),
                "min_gold_rank_core": retrieval_diag_core.get("min_gold_rank"),
                "winning_clause_rank_core": retrieval_diag_core.get("winning_clause_rank"),
                "best_gold_score_core": retrieval_diag_core.get("best_gold_score"),
                "best_non_gold_score_core": retrieval_diag_core.get("best_non_gold_score"),
                "gold_score_gap_core": retrieval_diag_core.get("gold_score_gap"),
            }
        )
        judge_used_any_core = None
        judge_used_any_bridge = None
        judge_used_any_critical_core = None
        judge_used_any_decoy = None
        if isinstance(record.get("judge_supporting_clause_ids"), list):
            supporting = set(record.get("judge_supporting_clause_ids") or [])
            core_ids = set(record.get("gold_evidence_core_ids") or [])
            judge_used_any_core = bool(supporting & core_ids)
            bridge_id = record.get("bridge_clause_id")
            judge_used_any_bridge = bool(bridge_id and bridge_id in supporting)
            critical_ids = set(record.get("critical_core_clause_ids") or [])
            if critical_ids:
                judge_used_any_critical_core = bool(supporting & critical_ids)
            decoy_ids = {
                cid
                for cid in supporting
                if (world.clauses.get(cid) and str(world.clauses[cid].doc_id).startswith("DECOY"))
            }
            judge_used_any_decoy = bool(decoy_ids)
        record["judge_used_any_core"] = judge_used_any_core
        record["judge_used_any_bridge"] = judge_used_any_bridge
        record["judge_used_any_critical_core"] = judge_used_any_critical_core
        record["judge_used_any_decoy"] = judge_used_any_decoy
        record["acc_no_core_evidence"] = bool(
            record.get("judge_correct") is True
            and (record.get("opened_gold_coverage_core") in {0, 0.0})
        )
        if args.evidence_padding_mode in {"schema_only", "global"}:
            record["evidence_before_pad"] = evidence_before
            record["evidence_after_pad"] = evidence_after
        include_raw_output = bool(
            getattr(args, "debug_n", 0) or getattr(args, "debug_task_ids", "")
        )
        if include_raw_output:
            record["raw_output"] = raw_output or ""
        prompt_path: Path | None = None
        raw_path: Path | None = None
        if args.save_prompts:
            prompt_dir = run_dir / "prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / f"{task.task_id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            record["prompt_path"] = str(prompt_path)
        if args.save_raw:
            raw_dir = run_dir / "raw_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{task.task_id}.txt"
            raw_path.write_text(raw_output or "", encoding="utf-8")
            record["raw_path"] = str(raw_path)
        record["goc_graph_jsonl_path"] = str(goc_graph_task_path) if goc_graph_task_path else None
        record["goc_graph_dot_path"] = str(goc_graph_dot_path) if goc_graph_dot_path else None
        record["goc_internal_graph_jsonl_path"] = (
            str(goc_internal_graph_task_path) if goc_internal_graph_task_path else None
        )
        if save_event_trace:
            trace_this_task = False
            if event_trace_sample_rate >= 1.0:
                trace_this_task = True
            elif event_trace_sample_rate > 0.0:
                digest = hashlib.md5(f"{method}:{task.task_id}".encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
                trace_this_task = bucket <= event_trace_sample_rate
            if trace_this_task:
                budget_chars = (
                    record.get("e3_context_budget_chars")
                    if record.get("e3_context_budget_chars") is not None
                    else getattr(args, "thread_context_budget_chars", None)
                )
                budget_label = str(budget_chars) if budget_chars is not None else "na"
                trace_path = (
                    event_trace_root
                    / method
                    / f"budget_{budget_label}"
                    / f"{task.task_id}.jsonl"
                )
                prompt_text = prompt or ""
                prompt_payload: Dict[str, Any]
                if prompt_path is not None:
                    prompt_payload = {"prompt_path": str(prompt_path)}
                else:
                    prompt_payload = {
                        "prompt_text": prompt_text[:20000],
                        "prompt_text_truncated": len(prompt_text) > 20000,
                    }
                raw_text = raw_output or ""
                prediction_payload: Dict[str, Any]
                if raw_path is not None:
                    prediction_payload = {"raw_path": str(raw_path)}
                else:
                    prediction_payload = {
                        "raw_text": raw_text[:20000],
                        "raw_text_truncated": len(raw_text) > 20000,
                    }
                prediction_payload["judge_correct"] = record.get("judge_correct")
                prediction_payload["packed_all_critical"] = record.get(
                    "judge_correct_packed_allcritical"
                )
                prediction_payload["judge_correct_packed_allcritical"] = record.get(
                    "judge_correct_packed_allcritical"
                )
                snapshot_selected_clause_ids = _unique_strs(
                    list(record.get("opened_for_prompt_clause_ids") or [])
                    or list(record.get("e3_context_clause_ids") or [])
                    or list(record.get("opened_clause_ids") or [])
                )
                snapshot_unfolded_clause_ids = _unique_strs(
                    list(record.get("unfolded_activated_clause_ids") or [])
                    or list(record.get("goc_unfold_selected_clause_ids") or [])
                )
                append_event(
                    trace_path,
                    build_event(
                        event_trace_run_id,
                        task.task_id,
                        method,
                        0,
                        "INIT",
                        {
                            "thread_id": thread_id,
                            "episode_id": episode_id,
                            "budget_chars": budget_chars,
                            "method": method,
                            "model": resolved_model,
                            "is_pivot_task": bool(record.get("is_pivot_task")),
                            "pivot_type": record.get("pivot_type"),
                            "pivot_old_days": record.get("pivot_old_days"),
                            "pivot_updated_days": record.get("pivot_updated_days"),
                        },
                    ),
                )
                append_event(
                    trace_path,
                    build_event(
                        event_trace_run_id,
                        task.task_id,
                        method,
                        1,
                        "PROMPT",
                        prompt_payload,
                    ),
                )
                append_event(
                    trace_path,
                    build_event(
                        event_trace_run_id,
                        task.task_id,
                        method,
                        2,
                        "PREDICTION",
                        prediction_payload,
                    ),
                )
                snapshot_payload: Dict[str, Any] = {
                    "selected_clause_ids": snapshot_selected_clause_ids,
                    "unfolded_clause_ids": snapshot_unfolded_clause_ids,
                    "active_context_clause_ids": _unique_strs(
                        list(record.get("active_context_clause_ids") or [])
                    ),
                    "prompt_tokens": record.get("prompt_tokens"),
                    "e3_context_chars_used": record.get("e3_context_chars_used"),
                    "e3_context_token_est": record.get("e3_context_token_est"),
                    "e3_packed_token_est": record.get("e3_packed_token_est"),
                    "pivot_compliant": record.get("pivot_compliant"),
                    "stale_evidence": record.get("stale_evidence"),
                    "goc_unfold_max_nodes": record.get("goc_unfold_max_nodes"),
                    "goc_unfold_hops": record.get("goc_unfold_hops"),
                    "goc_unfold_budget_mode": record.get("goc_unfold_budget_mode"),
                    "closure_candidate_count": record.get("goc_closure_candidate_count"),
                    "unfolded_node_count": record.get("goc_unfolded_node_count"),
                    "goc_avoids_edge_injected": record.get("goc_avoids_edge_injected"),
                    "goc_avoid_target_clause_ids": list(record.get("goc_avoid_target_clause_ids") or []),
                    "goc_avoided_node_injected": record.get("goc_avoided_node_injected"),
                    "selected_node_ids": _unique_strs(
                        list(record.get("goc_selected_node_ids") or [])
                    ),
                    "unfolded_node_ids": _unique_strs(
                        list(record.get("goc_unfolded_node_ids") or [])
                    ),
                }
                if bool(getattr(args, "goc_activity_debug_in_snapshot", False)):
                    debug_payload = record.get("goc_activity_debug")
                    if isinstance(debug_payload, dict):
                        if isinstance(debug_payload.get("ticket_activity"), list):
                            snapshot_payload["ticket_activity"] = list(
                                debug_payload.get("ticket_activity") or []
                            )
                        if debug_payload.get("selected_activity_summary") is not None:
                            snapshot_payload["selected_activity_summary"] = debug_payload.get(
                                "selected_activity_summary"
                            )
                        if debug_payload.get("top10_candidates_by_base_score") is not None:
                            snapshot_payload["top10_candidates_by_base_score"] = debug_payload.get(
                                "top10_candidates_by_base_score"
                            )
                        snapshot_payload["filter_fallback_used"] = bool(
                            debug_payload.get("filter_fallback_used", False)
                        )
                append_event(
                    trace_path,
                    build_event(
                        event_trace_run_id,
                        task.task_id,
                        method,
                        3,
                        "SNAPSHOT",
                        snapshot_payload,
                    ),
                )
        if method == "goc" and getattr(args, "goc_graph_sample_rate", 1.0) <= 0.0:
            record["goc_graph_jsonl_path"] = None
            record["goc_graph_dot_path"] = None
            record["goc_internal_graph_jsonl_path"] = None
        records.append(record)

        if threaded_mode and thread_state and episode_id:
            thread_state["episode_records"][episode_id] = record
            if isinstance(record.get("judge_correct"), bool):
                episode_judge[episode_id].append(1.0 if record.get("judge_correct") else 0.0)
            if isinstance(record.get("commit_correct"), bool):
                episode_commit[episode_id].append(1.0 if record.get("commit_correct") else 0.0)
            if isinstance(record.get("opened_gold_coverage_core"), (int, float)):
                episode_cov_core[episode_id].append(float(record.get("opened_gold_coverage_core") or 0.0))
            if episode_id == 3:
                commit1 = thread_state.get("commit1") or {}
                commit2 = thread_state.get("commit2") or {}
                thread_records.append(
                    {
                        "thread_id": thread_id,
                        "method": method,
                        "is_pivot_task": bool(record.get("is_pivot_task")),
                        "thread_strict_correct": record.get("thread_strict_correct"),
                        "thread_e3_only_correct": record.get("thread_e3_only_correct"),
                        "thread_judge_correct": record.get("thread_judge_correct"),
                        "thread_judge_correct_pivot": record.get("thread_judge_correct"),
                        "thread_decision_correct": record.get("decision_correct"),
                        "commit1_correct": commit1.get("commit_correct"),
                        "commit2_correct": commit2.get("commit_correct"),
                        "commit1_correct_anchor": commit1.get("commit_correct_anchor"),
                        "commit1_correct_union": commit1.get("commit_correct_union"),
                        "commit1_anchor_promotion_needed": commit1.get("commit_anchor_promotion_needed"),
                        "commit2_correct_anchor": commit2.get("commit_correct_anchor"),
                        "commit2_correct_union": commit2.get("commit_correct_union"),
                        "commit2_anchor_promotion_needed": commit2.get("commit_anchor_promotion_needed"),
                        "e3_answer_correct": record.get("e3_answer_correct"),
                        "e3_decision_correct": record.get("e3_decision_correct"),
                        "e3_evidence_valid": record.get("e3_evidence_valid"),
                        "e3_evidence_valid_in_context": record.get("e3_evidence_valid_in_context"),
                        "e3_evidence_valid_in_commits": record.get("e3_evidence_valid_in_commits"),
                        "pivot_gold_mode": record.get("pivot_gold_mode"),
                        "open_calls_total": sum(
                            int(r.get("open_calls") or 0)
                            for r in thread_state.get("episode_records", {}).values()
                        ),
                        "tool_calls_total": sum(
                            int(r.get("tool_calls") or 0)
                            for r in thread_state.get("episode_records", {}).values()
                        ),
                    }
                )

        if goc_graph and goc_graph_path:
            rid = run_id or run_dir.name
            log_events = "events" in args.goc_graph_mode
            last_event_node: str | None = None
            def _log(event_type: str, payload: Dict[str, Any]) -> None:
                if not log_events:
                    return
                if thread_id:
                    payload.setdefault("thread_id", thread_id)
                if episode_id:
                    payload.setdefault("episode_id", episode_id)
                if episode_kind:
                    payload.setdefault("episode_kind", episode_kind)
                append_event(
                    goc_graph_path,
                    build_event(
                        rid,
                        task.task_id,
                        method,
                        goc_graph.step,
                        event_type,
                        payload,
                    ),
                )
            def _add_event_node(node_id: str, node_type: str, **attrs: Any) -> None:
                nonlocal last_event_node
                if args.goc_graph_schema != "v1":
                    return
                goc_graph.add_node(node_id, node_type, step=goc_graph.step, **attrs)
                if last_event_node:
                    goc_graph.add_edge(
                        f"next:{last_event_node}:{node_id}",
                        last_event_node,
                        node_id,
                        "next",
                    )
                last_event_node = node_id
            episode_node_id = f"episode:{task.task_id}"
            ticket_id = f"ticket:{task.task_id}"
            goc_graph.add_node(episode_node_id, "episode", step=goc_graph.step)
            goc_graph.add_node(ticket_id, "ticket", text=task_for_run.user_ticket, step=goc_graph.step)
            _log("INIT", {"episode_id": episode_node_id, "ticket_id": ticket_id})
            goc_graph.step += 1

            if bool(record.get("goc_avoids_edge_injected")):
                init_ticket_node_id = str(
                    record.get("goc_initial_ticket_node_id")
                    or f"ticket_initial:{thread_id}"
                )
                pivot_ticket_node_id = str(
                    record.get("goc_pivot_ticket_node_id")
                    or f"ticket_pivot:{task.task_id}"
                )
                goc_graph.add_node(
                    init_ticket_node_id,
                    "ticket_initial",
                    thread_id=thread_id,
                    text=ticket_initial,
                    step=goc_graph.step,
                )
                goc_graph.add_node(
                    pivot_ticket_node_id,
                    "ticket_pivot",
                    thread_id=thread_id,
                    text=ticket_updated,
                    step=goc_graph.step,
                )
                goc_graph.add_edge(
                    f"avoids:{pivot_ticket_node_id}:{init_ticket_node_id}",
                    pivot_ticket_node_id,
                    init_ticket_node_id,
                    "avoids",
                )
                _log(
                    "AVOIDS_EDGE",
                    {
                        "pivot_ticket_node_id": pivot_ticket_node_id,
                        "initial_ticket_node_id": init_ticket_node_id,
                        "avoid_target_clause_ids": list(record.get("goc_avoid_target_clause_ids") or []),
                    },
                )
                goc_graph.step += 1

            if threaded_mode and episode_id and (record.get("commit_anchor_clause_ids") or record.get("commit_supporting_clause_ids")):
                commit_node_id = f"commit:{thread_id}:C{episode_id}"
                goc_graph.add_node(
                    commit_node_id,
                    "commit_anchor",
                    thread_id=thread_id,
                    episode_id=episode_id,
                    short_fact=record.get("commit_short_fact"),
                    step=goc_graph.step,
                )
                goc_graph.add_edge(
                    f"commit_from_ticket:{ticket_id}:{commit_node_id}",
                    ticket_id,
                    commit_node_id,
                    "commit_from_ticket",
                )
                _log(
                    "COMMIT_ANCHOR",
                    {
                        "commit_id": commit_node_id,
                        "anchor_clause_ids": record.get("commit_anchor_clause_ids") or record.get("commit_clause_ids"),
                        "supporting_clause_ids": record.get("commit_supporting_clause_ids"),
                        "anchor_source_used": record.get("commit_anchor_source_used"),
                        "model_evidence_clause_ids": record.get("commit_model_evidence_clause_ids"),
                    },
                )

            gold_node = f"gold:{task.task_id}"
            goc_graph.add_node(
                gold_node,
                "gold",
                decision=record.get("gold_decision"),
                decision_original=record.get("gold_decision_original"),
                pivot_gold_mode=record.get("pivot_gold_mode"),
                step=goc_graph.step,
            )
            for cid in record.get("gold_evidence_ids") or []:
                doc_id = f"doc:{cid}"
                goc_graph.add_node(doc_id, "doc_ref", clause_id=cid)
                goc_graph.add_edge(f"gold:{task.task_id}:{cid}", gold_node, doc_id, "gold_evidence")

            primary_results = diag.get("primary_search_results", [])
            rewrite_queries = diag.get("rewrite_queries") or [task.user_ticket]
            first_seen: Dict[str, Dict[str, Any]] = {}
            stages = ["primary", "rewrite1", "rewrite2"]
            rerank_scores: Dict[str, float] = {}
            if controller_policy == "rerank" and isinstance(controller, RerankController):
                secondary_results = diag.get("secondary_search_results", [])
                rerank_scores = controller.score_candidates(
                    primary_results,
                    secondary_results,
                    world,
                    task.context,
                )
            variants = []
            if isinstance(diag, dict) and diag.get("primary_search_variants"):
                variants = list(diag.get("primary_search_variants") or [])
            else:
                for idx, query in enumerate(rewrite_queries[:3]):
                    stage = stages[idx]
                    results = primary_results if idx == 0 else []
                    variants.append({"stage": stage, "query": query, "results": results})

            if agent_query_policy == "two_hop_bridge":
                variants = [v for v in variants if v.get("stage") != "hybrid_merged"]
            for variant in variants:
                stage = variant.get("stage") or "primary"
                query = variant.get("query") or ""
                qid = f"query:{task.task_id}:{stage}"
                query_meta = variant.get("query_meta") or {}
                node_attrs = {"text": query, "step": goc_graph.step, "stage": stage}
                if isinstance(query_meta, dict):
                    node_attrs.update(query_meta)
                goc_graph.add_node(qid, "query", **node_attrs)
                tool_call_id = f"tool_call:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(tool_call_id, "tool_call", tool="search", query_id=qid)
                depends_on = variant.get("depends_on")
                if depends_on:
                    goc_graph.add_edge(
                        f"depends:{qid}:{depends_on}",
                        qid,
                        depends_on,
                        "depends_on",
                    )
                results = variant.get("results") or []
                retrieved = []
                for rank, item in enumerate(results[: args.search_snapshot_k], start=1):
                    cid = item.get("clause_id")
                    if not cid:
                        continue
                    doc_id = f"doc:{cid}"
                    clause = world.clauses.get(cid)
                    node_attrs = {
                        "clause_id": cid,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "first_seen_query_id": qid,
                        "first_seen_rank": rank,
                        "first_seen_score": item.get("score"),
                        "step": goc_graph.step,
                    }
                    if args.goc_graph_include_clause_text and cid in world.clauses:
                        node_attrs["text"] = world.clauses[cid].text[:2000]
                    goc_graph.add_node(doc_id, "doc_ref", **node_attrs)
                    if cid not in first_seen:
                        first_seen[cid] = {
                            "query_id": qid,
                            "rank": rank,
                            "score": item.get("score"),
                        }
                    edge_id = f"retrieved:{task.task_id}:{stage}:{cid}"
                    goc_graph.add_edge(
                        edge_id,
                        qid,
                        doc_id,
                        "retrieved",
                        rank=rank,
                        score=item.get("score"),
                        source=stage,
                    )
                    payload = {
                        "doc_id": doc_id,
                        "clause_id": cid,
                        "rank": rank,
                        "score": item.get("score"),
                        "source": stage,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "snippet": (clause.text[:200] if clause else ""),
                    }
                    for key in [
                        "in_base_topk",
                        "base_rank",
                        "base_score",
                        "in_struct_topk",
                        "struct_rank",
                        "struct_score",
                        "merge_score",
                    ]:
                        if key in item:
                            payload[key] = item.get(key)
                    retrieved.append(payload)
                obs_id = f"observation:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    obs_id,
                    "observation",
                    observation_type="search_results_summary",
                    text=f"{len(retrieved)} results",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("SEARCH", {"query_id": qid, "query": query, "results": retrieved})
                goc_graph.step += 1

            secondary_results = diag.get("secondary_search_results", [])
            if secondary_results:
                qid = f"query:{task.task_id}:secondary"
                query = diag.get("secondary_search_query", "")
                goc_graph.add_node(qid, "query", text=query, step=goc_graph.step)
                tool_call_id = f"tool_call:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(tool_call_id, "tool_call", tool="search", query_id=qid)
                retrieved = []
                for rank, item in enumerate(secondary_results[: args.search_snapshot_k], start=1):
                    cid = item.get("clause_id")
                    if not cid:
                        continue
                    doc_id = f"doc:{cid}"
                    clause = world.clauses.get(cid)
                    node_attrs = {
                        "clause_id": cid,
                        "kind": getattr(clause, "kind", None),
                        "slot": getattr(clause, "slot", None),
                        "published_at": getattr(clause, "published_at", None),
                        "authority": getattr(clause, "authority", None),
                        "step": goc_graph.step,
                    }
                    if args.goc_graph_include_clause_text and cid in world.clauses:
                        node_attrs["text"] = world.clauses[cid].text[:2000]
                    goc_graph.add_node(doc_id, "doc_ref", **node_attrs)
                    if cid not in first_seen:
                        first_seen[cid] = {
                            "query_id": qid,
                            "rank": rank,
                            "score": item.get("score"),
                        }
                    edge_id = f"retrieved:{task.task_id}:secondary:{cid}"
                    goc_graph.add_edge(
                        edge_id,
                        qid,
                        doc_id,
                        "retrieved",
                        rank=rank,
                        score=item.get("score"),
                        source="secondary",
                    )
                    retrieved.append(
                        {
                            "doc_id": doc_id,
                            "clause_id": cid,
                            "rank": rank,
                            "score": item.get("score"),
                            "source": "secondary",
                            "kind": getattr(clause, "kind", None),
                            "slot": getattr(clause, "slot", None),
                            "published_at": getattr(clause, "published_at", None),
                            "authority": getattr(clause, "authority", None),
                            "snippet": (clause.text[:200] if clause else ""),
                        }
                    )
                obs_id = f"observation:search:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    obs_id,
                    "observation",
                    observation_type="search_results_summary",
                    text=f"{len(retrieved)} results",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("SEARCH", {"query_id": qid, "query": query, "results": retrieved})
                goc_graph.step += 1

            forced = set(diag.get("forced_open_ids") or [])
            for idx, cid in enumerate(opened_ids, start=1):
                selected_by = "forced_open" if cid in forced else "heuristic_fallback"
                if controller_policy == "rerank":
                    selected_by = "controller_rerank"
                elif controller_action:
                    selected_by = "controller_bandit"
                reason = selected_by
                probe_ids = set(diag.get("opened_probe_clause_ids", []) if isinstance(diag, dict) else [])
                open_stage = "probe" if cid in probe_ids else "prompt"
                bm25_meta = first_seen.get(cid, {})
                rerank_score = rerank_scores.get(cid) if rerank_scores else None
                controller_context_key = None
                if controller_action and hasattr(controller, "_context_key"):
                    try:
                        search_stats = controller.compute_search_stats(primary_results)  # type: ignore[attr-defined]
                        features = controller.build_context_features(task.context, env.open_budget, search_stats)  # type: ignore[attr-defined]
                        controller_context_key = controller._context_key(features)  # type: ignore[attr-defined]
                    except Exception:
                        controller_context_key = None
                tool_call_id = f"tool_call:open:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    tool_call_id,
                    "tool_call",
                    tool="open",
                    clause_id=cid,
                )
                if args.goc_graph_schema == "v1" and cid in world.clauses:
                    obs_id = f"observation:open:{task.task_id}:{goc_graph.step}"
                    _add_event_node(
                        obs_id,
                        "observation",
                        observation_type="opened_clause_excerpt",
                        text=world.clauses[cid].text[:200],
                    )
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                goc_graph.add_edge(
                    f"opened:{task.task_id}:{cid}",
                    episode_id,
                    f"doc:{cid}",
                    "opened",
                    open_step=idx,
                    reason=reason,
                    open_stage=open_stage,
                    selected_for_prompt=open_stage == "prompt",
                    selected_by=selected_by,
                    from_query_id=bm25_meta.get("query_id"),
                    bm25_rank=bm25_meta.get("rank"),
                    bm25_score=bm25_meta.get("score"),
                    rerank_score=rerank_score,
                    controller_policy=controller_policy,
                    controller_mode=controller_mode,
                    controller_action=controller_action,
                    controller_context_key=controller_context_key,
                )
                _log(
                    "OPEN",
                    {
                        "clause_id": cid,
                        "reason": reason,
                        "open_stage": open_stage,
                        "selected_for_prompt": open_stage == "prompt",
                        "open_index": idx,
                        "selected_by": selected_by,
                        "from_query_id": bm25_meta.get("query_id"),
                        "bm25_rank": bm25_meta.get("rank"),
                        "bm25_score": bm25_meta.get("score"),
                        "rerank_score": rerank_score,
                        "controller_policy": controller_policy,
                        "controller_mode": controller_mode,
                        "controller_action": controller_action,
                        "controller_context_key": controller_context_key,
                    },
                )
                goc_graph.step += 1

            prompt_id = f"prompt:{task.task_id}:{goc_graph.step}"
            controller_info = {"controller_policy": controller_policy, "controller_action": controller_action}
            if rerank_scores:
                top_candidates = []
                items = sorted(primary_results, key=lambda it: it.get("score", 0.0), reverse=True)[:5]
                dates = [world.clauses[it["clause_id"]].published_at for it in items if it.get("clause_id") in world.clauses]
                min_date = min(dates) if dates else None
                max_date = max(dates) if dates else None
                for item in items:
                    cid = item.get("clause_id")
                    clause = world.clauses.get(cid) if cid else None
                    recency_score = 0.0
                    if clause and clause.published_at and min_date and max_date and max_date != min_date:
                        recency_score = (clause.published_at > min_date) * 1.0
                    text = clause.text.lower() if clause else ""
                    top_candidates.append(
                        {
                            "clause_id": cid,
                            "total_score": rerank_scores.get(cid),
                            "bm25_score": item.get("score"),
                            "recency_score": recency_score,
                            "slot_match": 1.0 if clause and clause.slot == task.context.get("slot") else 0.0,
                            "is_update": bool(clause and clause.kind == "update"),
                            "is_exception": bool(clause and clause.kind == "exception"),
                            "is_definition": bool(clause and clause.kind == "definition"),
                            "has_conditions_kw": any(k in text for k in ["conditions:", "must", "require", "provided that"]),
                            "kind": clause.kind if clause else None,
                            "slot": clause.slot if clause else None,
                            "published_at": clause.published_at if clause else None,
                        }
                    )
                controller_info["top_candidates"] = top_candidates
            controller_node_id = f"controller:{task.task_id}"
            goc_graph.add_node(controller_node_id, "controller", **controller_info, step=goc_graph.step)
            goc_graph.add_edge(
                f"controls:{task.task_id}",
                episode_id,
                controller_node_id,
                "controller",
            )
            _add_event_node(
                f"summary:{task.task_id}:{goc_graph.step}",
                "summary",
                text="prompt_materialization",
            )
            prompt_clause_ids = (
                diag.get("opened_for_prompt_clause_ids", [])
                if isinstance(diag, dict) and diag.get("opened_for_prompt_clause_ids") is not None
                else opened_ids
            )
            prompt_payload = goc_graph.materialize_prompt(
                prompt_id,
                [f"doc:{cid}" for cid in prompt_clause_ids],
                budget_info={"open_budget": env.open_budget},
                controller_info=controller_info,
            )
            _log("PROMPT", prompt_payload)
            for node_id in prompt_payload.get("included_node_ids", []):
                goc_graph.add_edge(
                    f"selected:{prompt_id}:{node_id}",
                    episode_id,
                    node_id,
                    "selected_for_prompt",
                    prompt_id=prompt_id,
                    created_step=goc_graph.step,
                    budget_info=prompt_payload.get("budget_info"),
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"summarized:{prompt_id}:{node_id}",
                        f"summary:{task.task_id}:{goc_graph.step}",
                        node_id,
                        "summarized",
                    )
            goc_graph.step += 1

            if record.get("raw_path"):
                tool_call_id = f"tool_call:llm:{task.task_id}:{goc_graph.step}"
                _add_event_node(
                    tool_call_id,
                    "tool_call",
                    tool="llm_generate",
                )
                if args.goc_graph_schema == "v1":
                    obs_id = f"observation:llm:{task.task_id}:{goc_graph.step}"
                    _add_event_node(
                        obs_id,
                        "observation",
                        observation_type="llm_raw",
                        text="raw_saved",
                    )
                    goc_graph.add_edge(
                        f"depends:{obs_id}:{tool_call_id}",
                        obs_id,
                        tool_call_id,
                        "depends_on",
                    )
                _log("LLM_RAW", {"raw_path": record.get("raw_path")})
                goc_graph.step += 1

            answer_id = f"answer:{task.task_id}"
            goc_graph.add_node(answer_id, "answer", decision=pred_for_record.get("decision"), step=goc_graph.step)
            for cid in evidence_before:
                goc_graph.add_edge(
                    f"cites_raw:{task.task_id}:{cid}",
                    answer_id,
                    f"doc:{cid}",
                    "cites_evidence_raw",
                )
            for cid in evidence_after:
                goc_graph.add_edge(
                    f"cites:{task.task_id}:{cid}",
                    answer_id,
                    f"doc:{cid}",
                    "cites_evidence",
                )
                if args.goc_graph_schema == "v1":
                    goc_graph.add_edge(
                        f"depends:{answer_id}:{cid}",
                        answer_id,
                        f"doc:{cid}",
                        "depends_on",
                    )
            _log(
                "PREDICTION",
                goc_graph.record_prediction(
                    pred_for_record.get("decision", ""),
                    pred_for_record.get("conditions", []),
                    evidence_before,
                    evidence_after,
                    parse_error=error,
                    raw_path=record.get("raw_path"),
                    prompt_path=record.get("prompt_path"),
                ),
            )
            goc_graph.step += 1

            _log("DONE", {})
            goc_graph.step += 1

            if "final" in args.goc_graph_mode:
                snapshot_payload = goc_graph.to_snapshot_dict()
                append_event(
                    goc_graph_path,
                    build_event(
                        rid,
                        task.task_id,
                        method,
                        goc_graph.step,
                        "SNAPSHOT",
                        snapshot_payload,
                    ),
                )
                goc_graph.step += 1
            if goc_graph_task_path and goc_graph_path != goc_graph_task_path:
                goc_graph_task_path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for line in goc_graph_path.read_text(encoding="utf-8").splitlines():
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if data.get("task_id") == task.task_id:
                        lines.append(line)
                goc_graph_task_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if args.save_goc_dot and goc_graph_task_path:
                from goc_logger.export import export_dot

                goc_graph_dot_path = goc_graph_task_path.with_suffix(".dot")
                export_dot(goc_graph_task_path, task.task_id, goc_graph_dot_path)
                record["goc_graph_dot_path"] = str(goc_graph_dot_path)
            if goc_graph_task_path:
                record["goc_graph_jsonl_path"] = str(goc_graph_task_path)
            if method == "goc" and goc_internal_graph_task_path:
                if isinstance(goc_policyops_dep_graph_snapshot, dict):
                    goc_internal_snapshots.append(goc_policyops_dep_graph_snapshot)
                snapshots = [s for s in (goc_internal_snapshots or []) if isinstance(s, dict)]
                if not snapshots:
                    try:
                        from src.memory import GoCMemory

                        _tmp_mem = GoCMemory(budget_active=1200, budget_unfold=650, unfold_k=8)
                        _snap0 = _tmp_mem.snapshot()
                        _snap0["snapshot_kind"] = "step"
                        _snap0["snapshot_idx"] = 1
                        _snapf = _tmp_mem.snapshot()
                        _snapf["snapshot_kind"] = "final"
                        _snapf["snapshot_idx"] = 2
                        snapshots = [_snap0, _snapf]
                    except Exception:
                        snapshots = []
                if snapshots:
                    goc_internal_graph_task_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(goc_internal_graph_task_path, "w", encoding="utf-8") as fp:
                        for snap in snapshots:
                            fp.write(json.dumps(snap, ensure_ascii=False) + "\n")
                    record["goc_internal_graph_jsonl_path"] = str(goc_internal_graph_task_path)
                else:
                    record["goc_internal_graph_jsonl_path"] = None

    aggregate = aggregate_metrics(metrics)
    aggregate["gold_decision_distribution"] = gold_decision_distribution(tasks)
    if records:
        aggregate["opened_gold_coverage_core_mean"] = sum(
            r.get("opened_gold_coverage_core", 0.0) for r in records
        ) / len(records)
    else:
        aggregate["opened_gold_coverage_core_mean"] = 0.0
    pred_dist = {"allow": 0, "deny": 0, "require_condition": 0, "needs_more_info": 0}
    for rec in records:
        pred = rec.get("pred_decision")
        if pred in pred_dist:
            pred_dist[pred] += 1
        else:
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
    aggregate["pred_decision_distribution"] = pred_dist
    non_rc = [r for r in records if r.get("gold_decision") != "require_condition"]
    if non_rc:
        aggregate["spurious_require_condition_rate"] = sum(
            1 for r in non_rc if r.get("pred_decision") == "require_condition"
        ) / len(non_rc)
    else:
        aggregate["spurious_require_condition_rate"] = 0.0
    gold_rc = [r for r in records if r.get("gold_decision") == "require_condition"]
    if gold_rc:
        aggregate["missed_require_condition_rate"] = sum(
            1 for r in gold_rc if r.get("pred_decision") != "require_condition"
        ) / len(gold_rc)
    else:
        aggregate["missed_require_condition_rate"] = 0.0
    bridge_records = [r for r in records if r.get("bridge_clause_id")]
    if bridge_records:
        aggregate["bridge_found_rate"] = sum(1 for r in bridge_records if r.get("bridge_found")) / len(
            bridge_records
        )
    else:
        aggregate["bridge_found_rate"] = 0.0
    two_hop_records = [r for r in records if r.get("agent_query_policy") == "two_hop_bridge"]
    if two_hop_records:
        aggregate["canonical_used_in_query2_rate"] = sum(
            1 for r in two_hop_records if r.get("canonical_used_in_query2")
        ) / len(two_hop_records)
    else:
        aggregate["canonical_used_in_query2_rate"] = 0.0
    update_needed_records = [r for r in records if r.get("needs_update_resolution")]
    if update_needed_records:
        aggregate["update_found_when_needed_rate"] = sum(
            1 for r in update_needed_records if r.get("update_found_when_needed")
        ) / len(update_needed_records)
    else:
        aggregate["update_found_when_needed_rate"] = 0.0
    aggregate["winning_rank_exists_rate"] = sum(
        1 for r in records if r.get("winning_rank_exists")
    ) / len(records) if records else 0.0
    records_with_rank = [r for r in records if r.get("winning_rank_exists")]
    if records_with_rank:
        aggregate["decision_accuracy_when_winning_rank_exists"] = sum(
            1 for r in records_with_rank if r.get("decision_correct")
        ) / len(records_with_rank)
    else:
        aggregate["decision_accuracy_when_winning_rank_exists"] = 0.0
    def _mean_metric(key: str, default: float | None = 0.0) -> float | None:
        vals: List[float] = []
        for rec in records:
            val = rec.get(key)
            if isinstance(val, (int, float)):
                fval = float(val)
                if math.isnan(fval):
                    continue
                vals.append(fval)
        return sum(vals) / len(vals) if vals else default
    e3_records = [r for r in records if r.get("episode_id") == 3]
    def _mean_metric_e3(key: str, default: float | None = None) -> float | None:
        vals: List[float] = []
        for rec in e3_records:
            val = rec.get(key)
            if isinstance(val, (int, float)):
                fval = float(val)
                if math.isnan(fval):
                    continue
                vals.append(fval)
        return sum(vals) / len(vals) if vals else default
    aggregate["gold_in_search_topk_rate_hop1"] = _mean_metric("gold_in_search_topk_hop1")
    aggregate["gold_in_search_topk_rate_hop2"] = _mean_metric("gold_in_search_topk_hop2")
    aggregate["gold_in_search_topk_rate_union"] = _mean_metric("gold_in_search_topk_union")
    aggregate["winning_clause_rank_mean_hop1"] = _mean_metric("winning_clause_rank_hop1")
    aggregate["winning_clause_rank_mean_hop2"] = _mean_metric("winning_clause_rank_hop2")
    aggregate["winning_clause_rank_mean_union"] = _mean_metric("winning_clause_rank_union")
    aggregate["hit_at_open_budget_union"] = _mean_metric("hit_at_open_budget_union")
    aggregate["rank_success_rate"] = _mean_metric("rank_success")
    aggregate["winning_in_union_rate"] = _mean_metric("winning_in_union")
    aggregate["opened_has_winning_clause_rate_union"] = _mean_metric(
        "opened_has_winning_clause_union"
    )
    aggregate["policy_gain_over_rank"] = _mean_metric("policy_gain_over_rank", default=None)
    aggregate["rank_gap"] = _mean_metric("rank_gap", default=None)
    aggregate["feasible_open_rate"] = _mean_metric("feasible_open_rate")
    aggregate["realized_open_rate"] = _mean_metric("realized_open_rate")
    aggregate["selection_gap"] = _mean_metric("selection_gap")
    aggregate["selection_efficiency"] = _mean_metric("selection_efficiency", default=None)
    aggregate["judge_accuracy"] = _mean_metric("judge_correct", default=None)
    aggregate["judge_accuracy_packed"] = (
        aggregate.get("judge_accuracy")
        if getattr(args, "judge", "llm") == "symbolic_packed"
        else None
    )
    aggregate["judge_accuracy_full_episode"] = (
        aggregate.get("judge_accuracy")
        if getattr(args, "judge", "llm") == "symbolic_full_episode"
        else None
    )
    aggregate["acc_no_core_evidence_rate"] = _mean_metric("acc_no_core_evidence", default=0.0)
    aggregate["judge_used_any_core_rate"] = _mean_metric("judge_used_any_core", default=None)
    aggregate["judge_used_any_bridge_rate"] = _mean_metric("judge_used_any_bridge", default=None)
    aggregate["judge_used_any_critical_core_rate"] = _mean_metric(
        "judge_used_any_critical_core",
        default=None,
    )
    aggregate["judge_used_any_decoy_rate"] = _mean_metric(
        "judge_used_any_decoy",
        default=None,
    )
    aggregate["judge_supporting_count_mean"] = _mean_metric("judge_supporting_count", default=None)
    aggregate["full_episode_supporting_count_mean"] = _mean_metric(
        "full_episode_supporting_count",
        default=None,
    )
    aggregate["opened_bridge_count_mean"] = _mean_metric("opened_bridge_count", default=0.0)
    aggregate["opened_meta_count_mean"] = _mean_metric("opened_meta_count", default=0.0)
    aggregate["opened_rule_count_mean"] = _mean_metric("opened_rule_count", default=0.0)
    aggregate["deep_rank_core_rate"] = _mean_metric("deep_rank_core_flag", default=0.0)
    aggregate["e3_prompt_includes_required_core_rate"] = _mean_metric(
        "e3_prompt_includes_required_core",
        default=None,
    )
    aggregate["e3_prompt_includes_critical_core_rate"] = _mean_metric(
        "e3_prompt_includes_critical_core",
        default=None,
    )
    aggregate["e3_packed_contains_critical_rate"] = _mean_metric(
        "e3_packed_contains_critical",
        default=None,
    )
    aggregate["e3_packed_contains_critical0_rate"] = _mean_metric(
        "e3_packed_contains_critical0",
        default=None,
    )
    aggregate["e3_packed_contains_critical1_rate"] = _mean_metric(
        "e3_packed_contains_critical1",
        default=None,
    )
    aggregate["e3_packed_any_critical_rate"] = _mean_metric(
        "e3_packed_any_critical",
        default=None,
    )
    aggregate["e3_packed_all_critical_rate"] = _mean_metric(
        "e3_packed_all_critical",
        default=None,
    )
    aggregate["e3_packed_any_critical_rate_full_episode"] = _mean_metric(
        "e3_packed_any_critical_full_episode",
        default=None,
    )
    aggregate["e3_packed_all_critical_rate_full_episode"] = _mean_metric(
        "e3_packed_all_critical_full_episode",
        default=None,
    )
    aggregate["e3_context_truncated_rate"] = _mean_metric("e3_packed_truncated", default=None)
    if aggregate["e3_context_truncated_rate"] is None:
        aggregate["e3_context_truncated_rate"] = _mean_metric("e3_context_truncated", default=None)
    aggregate["e3_context_chars_used_mean"] = _mean_metric("e3_context_chars_used", default=None)
    aggregate["e3_context_token_est_mean"] = _mean_metric("e3_context_token_est", default=None)
    aggregate["e3_packed_token_est_mean"] = _mean_metric("e3_packed_token_est", default=None)
    aggregate["e3_context_clause_count_mean"] = _mean_metric("e3_context_clause_count", default=None)
    aggregate["e3_packed_clause_count_mean"] = _mean_metric("e3_packed_clause_count", default=None)
    aggregate["e3_packed_total_chars_before_mean"] = _mean_metric(
        "e3_packed_total_chars_before",
        default=None,
    )
    aggregate["e3_packed_total_chars_after_mean"] = _mean_metric(
        "e3_packed_total_chars_after",
        default=None,
    )
    aggregate["e3_packed_dropped_clause_count_mean"] = _mean_metric(
        "e3_packed_dropped_clause_count",
        default=None,
    )
    aggregate["e3_decoy_clause_count_mean"] = _mean_metric(
        "e3_decoy_clause_count",
        default=None,
    )
    aggregate["e3_litm_filler_count_mean"] = _mean_metric(
        "e3_litm_filler_count",
        default=None,
    )
    e3_chars_total = sum(
        int(rec.get("e3_context_chars_used") or 0)
        for rec in e3_records
        if isinstance(rec.get("e3_context_chars_used"), (int, float))
    )
    e3_tokens_total = sum(
        int(rec.get("e3_context_token_est") or 0)
        for rec in e3_records
        if isinstance(rec.get("e3_context_token_est"), (int, float))
    )
    aggregate["e3_context_chars_used_total"] = float(e3_chars_total)
    aggregate["e3_context_token_est_total"] = float(e3_tokens_total)
    packed_all_correct_count = sum(
        1
        for rec in e3_records
        if rec.get("e3_packed_all_critical") is True
        or rec.get("judge_correct_packed_allcritical") is True
    )
    aggregate["e3_packed_all_critical_correct_count"] = float(packed_all_correct_count)
    denom_correct = max(packed_all_correct_count, 1)
    aggregate["cost_per_correct_chars"] = (
        float(e3_chars_total) / float(denom_correct)
        if e3_records
        else None
    )
    aggregate["cost_per_correct_token_est"] = (
        float(e3_tokens_total) / float(denom_correct)
        if e3_records
        else None
    )
    all_critical_rate = aggregate.get("e3_packed_all_critical_rate")
    token_mean = aggregate.get("e3_context_token_est_mean")
    if isinstance(all_critical_rate, (int, float)):
        denom_token_mean = float(token_mean) if isinstance(token_mean, (int, float)) else 0.0
        aggregate["acc_per_1k_tokens"] = 1000.0 * float(all_critical_rate) / max(denom_token_mean, 1e-9)
    else:
        aggregate["acc_per_1k_tokens"] = None
    aggregate["e3_truncation_loss_estimate_rate"] = _mean_metric(
        "e3_truncation_loss_estimate",
        default=None,
    )
    aggregate["e3_packed_critical_count_mean"] = _mean_metric(
        "e3_packed_critical_count",
        default=None,
    )
    aggregate["opened_decoy_clause_count_mean"] = _mean_metric(
        "opened_decoy_clause_count",
        default=None,
    )
    aggregate["goc_unfolded_clause_count_mean"] = _mean_metric(
        "goc_unfolded_clause_count",
        default=None,
    )
    aggregate["goc_closure_candidate_count_mean"] = _mean_metric(
        "goc_closure_candidate_count",
        default=None,
    )
    aggregate["goc_unfolded_node_count_mean"] = _mean_metric(
        "goc_unfolded_node_count",
        default=None,
    )
    aggregate["goc_unfolded_critical_clause_count_mean"] = _mean_metric(
        "goc_unfolded_critical_clause_count",
        default=None,
    )
    aggregate["goc_folded_episode_count_mean"] = _mean_metric(
        "goc_folded_episode_count",
        default=None,
    )
    aggregate["closure_recall_core_mean"] = _mean_metric(
        "closure_recall_core",
        default=None,
    )
    aggregate["wrong_branch_recall_rate_mean"] = _mean_metric(
        "wrong_branch_recall_rate",
        default=None,
    )
    pivot_records = [r for r in records if r.get("is_pivot_task")]
    nonpivot_records = [r for r in records if not r.get("is_pivot_task")]
    aggregate["pivot_rate_actual"] = (
        float(len(pivot_records)) / float(len(records)) if records else 0.0
    )
    pivot_compliance_vals = [
        1.0 if r.get("pivot_compliant") else 0.0
        for r in pivot_records
        if isinstance(r.get("pivot_compliant"), bool)
    ]
    aggregate["pivot_compliance_rate"] = (
        sum(pivot_compliance_vals) / len(pivot_compliance_vals)
        if pivot_compliance_vals
        else None
    )
    stale_evidence_vals = [
        1.0 if r.get("stale_evidence") else 0.0
        for r in pivot_records
        if isinstance(r.get("stale_evidence"), bool)
    ]
    aggregate["stale_evidence_rate"] = (
        sum(stale_evidence_vals) / len(stale_evidence_vals)
        if stale_evidence_vals
        else None
    )
    avoids_eval_records = [
        r
        for r in e3_records
        if isinstance(r.get("goc_avoid_target_clause_ids"), list)
        and len(r.get("goc_avoid_target_clause_ids")) > 0
    ]
    avoids_injected_vals = [
        1.0 if r.get("goc_avoided_node_injected") else 0.0
        for r in avoids_eval_records
        if isinstance(r.get("goc_avoided_node_injected"), bool)
    ]
    aggregate["avoids_edge_injected_rate"] = (
        float(len(avoids_eval_records)) / float(len(e3_records))
        if e3_records
        else None
    )
    aggregate["avoided_node_injected_rate"] = (
        sum(avoids_injected_vals) / len(avoids_injected_vals)
        if avoids_injected_vals
        else None
    )
    aggregate["avoided_node_eval_count"] = len(avoids_injected_vals)
    pivot_acc_vals = [
        1.0 if r.get("judge_correct") else 0.0
        for r in pivot_records
        if isinstance(r.get("judge_correct"), bool)
    ]
    nonpivot_acc_vals = [
        1.0 if r.get("judge_correct") else 0.0
        for r in nonpivot_records
        if isinstance(r.get("judge_correct"), bool)
    ]
    pivot_acc = (sum(pivot_acc_vals) / len(pivot_acc_vals)) if pivot_acc_vals else None
    nonpivot_acc = (sum(nonpivot_acc_vals) / len(nonpivot_acc_vals)) if nonpivot_acc_vals else None
    aggregate["pivot_delta_accuracy"] = (
        float(pivot_acc - nonpivot_acc)
        if (pivot_acc is not None and nonpivot_acc is not None)
        else None
    )
    for ep_id in (1, 2, 3):
        ep_records = [r for r in records if int(r.get("episode_id") or 0) == ep_id]
        ep_decision_vals = [
            1.0 if r.get("decision_correct") else 0.0
            for r in ep_records
            if isinstance(r.get("decision_correct"), bool)
        ]
        ep_conditions_vals = [
            1.0 if r.get("conditions_correct") else 0.0
            for r in ep_records
            if isinstance(r.get("conditions_correct"), bool)
        ]
        ep_answer_vals = [
            1.0 if r.get("answer_correct") else 0.0
            for r in ep_records
            if isinstance(r.get("answer_correct"), bool)
        ]
        aggregate[f"e{ep_id}_decision_accuracy"] = (
            sum(ep_decision_vals) / len(ep_decision_vals) if ep_decision_vals else None
        )
        aggregate[f"e{ep_id}_conditions_accuracy"] = (
            sum(ep_conditions_vals) / len(ep_conditions_vals) if ep_conditions_vals else None
        )
        aggregate[f"e{ep_id}_answer_accuracy"] = (
            sum(ep_answer_vals) / len(ep_answer_vals) if ep_answer_vals else None
        )
    e3_pivot_records = [
        r for r in records if int(r.get("episode_id") or 0) == 3 and bool(r.get("is_pivot_task"))
    ]
    e3_nonpivot_records = [
        r for r in records if int(r.get("episode_id") or 0) == 3 and not bool(r.get("is_pivot_task"))
    ]
    e3_pivot_decision_vals = [
        1.0 if r.get("decision_correct") else 0.0
        for r in e3_pivot_records
        if isinstance(r.get("decision_correct"), bool)
    ]
    e3_nonpivot_decision_vals = [
        1.0 if r.get("decision_correct") else 0.0
        for r in e3_nonpivot_records
        if isinstance(r.get("decision_correct"), bool)
    ]
    aggregate["e3_pivot_decision_accuracy"] = (
        sum(e3_pivot_decision_vals) / len(e3_pivot_decision_vals)
        if e3_pivot_decision_vals
        else None
    )
    aggregate["e3_nonpivot_decision_accuracy"] = (
        sum(e3_nonpivot_decision_vals) / len(e3_nonpivot_decision_vals)
        if e3_nonpivot_decision_vals
        else None
    )
    aggregate["judge_accuracy_packed_allcritical"] = _mean_metric(
        "judge_correct_packed_allcritical",
        default=None,
    )
    core_rank_union_vals = [
        int(r.get("min_gold_core_rank_union"))
        for r in records
        if isinstance(r.get("min_gold_core_rank_union"), int)
    ]
    core_rank_summary = _rank_summary(core_rank_union_vals)
    aggregate["min_gold_core_rank_union_mean"] = core_rank_summary.get("mean")
    aggregate["min_gold_core_rank_union_median"] = core_rank_summary.get("median")
    aggregate["min_gold_core_rank_union_p90"] = core_rank_summary.get("p90")
    if thread_records:
        def _mean_thread_bool(key: str, rows: List[Dict[str, Any]]) -> float | None:
            vals = [
                1.0 if rec.get(key) else 0.0
                for rec in rows
                if isinstance(rec.get(key), bool)
            ]
            return (sum(vals) / len(vals)) if vals else None

        thread_judge_vals = [
            1.0 if r.get("thread_judge_correct") else 0.0
            for r in thread_records
            if r.get("thread_judge_correct") is not None
        ]
        thread_decision_vals = [
            1.0 if r.get("thread_decision_correct") else 0.0
            for r in thread_records
            if r.get("thread_decision_correct") is not None
        ]
        aggregate["thread_judge_accuracy"] = (
            sum(thread_judge_vals) / len(thread_judge_vals) if thread_judge_vals else None
        )
        aggregate["thread_decision_accuracy"] = (
            sum(thread_decision_vals) / len(thread_decision_vals) if thread_decision_vals else None
        )
        aggregate["thread_strict_accuracy"] = _mean_thread_bool(
            "thread_strict_correct",
            thread_records,
        )
        aggregate["thread_e3_only_accuracy"] = _mean_thread_bool(
            "thread_e3_only_correct",
            thread_records,
        )
        pivot_thread_records = [r for r in thread_records if bool(r.get("is_pivot_task"))]
        nonpivot_thread_records = [r for r in thread_records if not bool(r.get("is_pivot_task"))]
        aggregate["strict_final_pivot_accuracy"] = _mean_thread_bool(
            "thread_strict_correct",
            pivot_thread_records,
        )
        aggregate["strict_final_nonpivot_accuracy"] = _mean_thread_bool(
            "thread_strict_correct",
            nonpivot_thread_records,
        )
        aggregate["e3_pivot_e3_only_accuracy"] = _mean_thread_bool(
            "thread_e3_only_correct",
            pivot_thread_records,
        )
        aggregate["e3_nonpivot_e3_only_accuracy"] = _mean_thread_bool(
            "thread_e3_only_correct",
            nonpivot_thread_records,
        )
        aggregate["commit1_anchor_accuracy"] = _mean_thread_bool(
            "commit1_correct_anchor",
            thread_records,
        )
        aggregate["commit1_union_accuracy"] = _mean_thread_bool(
            "commit1_correct_union",
            thread_records,
        )
        aggregate["commit1_anchor_promotion_needed_rate"] = _mean_thread_bool(
            "commit1_anchor_promotion_needed",
            thread_records,
        )
        aggregate["commit2_anchor_accuracy"] = _mean_thread_bool(
            "commit2_correct_anchor",
            thread_records,
        )
        aggregate["commit2_union_accuracy"] = _mean_thread_bool(
            "commit2_correct_union",
            thread_records,
        )
        aggregate["commit2_anchor_promotion_needed_rate"] = _mean_thread_bool(
            "commit2_anchor_promotion_needed",
            thread_records,
        )
        aggregate["commit1_anchor_accuracy_pivot"] = _mean_thread_bool(
            "commit1_correct_anchor",
            pivot_thread_records,
        )
        aggregate["commit1_union_accuracy_pivot"] = _mean_thread_bool(
            "commit1_correct_union",
            pivot_thread_records,
        )
        aggregate["commit1_anchor_promotion_needed_rate_pivot"] = _mean_thread_bool(
            "commit1_anchor_promotion_needed",
            pivot_thread_records,
        )
        aggregate["commit2_anchor_accuracy_pivot"] = _mean_thread_bool(
            "commit2_correct_anchor",
            pivot_thread_records,
        )
        aggregate["commit2_union_accuracy_pivot"] = _mean_thread_bool(
            "commit2_correct_union",
            pivot_thread_records,
        )
        aggregate["commit2_anchor_promotion_needed_rate_pivot"] = _mean_thread_bool(
            "commit2_anchor_promotion_needed",
            pivot_thread_records,
        )
        e3_valid_ctx_vals = [
            1.0 if r.get("e3_evidence_valid_in_context") else 0.0
            for r in thread_records
            if r.get("e3_evidence_valid_in_context") is not None
        ]
        e3_valid_commits_vals = [
            1.0 if r.get("e3_evidence_valid_in_commits") else 0.0
            for r in thread_records
            if r.get("e3_evidence_valid_in_commits") is not None
        ]
        aggregate["e3_evidence_valid_in_context_rate"] = (
            sum(e3_valid_ctx_vals) / len(e3_valid_ctx_vals) if e3_valid_ctx_vals else None
        )
        aggregate["e3_evidence_valid_in_commits_rate"] = (
            sum(e3_valid_commits_vals) / len(e3_valid_commits_vals)
            if e3_valid_commits_vals
            else None
        )
        for ep_id in (1, 2, 3):
            judge_vals = episode_judge.get(ep_id, [])
            commit_vals = episode_commit.get(ep_id, [])
            cov_vals = episode_cov_core.get(ep_id, [])
            aggregate[f"episode_judge_accuracy_e{ep_id}"] = (
                sum(judge_vals) / len(judge_vals) if judge_vals else None
            )
            aggregate[f"episode_commit_success_e{ep_id}"] = (
                sum(commit_vals) / len(commit_vals) if commit_vals else None
            )
            aggregate[f"episode_opened_gold_coverage_core_mean_e{ep_id}"] = (
                sum(cov_vals) / len(cov_vals) if cov_vals else None
            )
        if getattr(args, "judge", "llm") in {"symbolic_packed", "symbolic_packed_allcritical"}:
            aggregate["e3_judge_accuracy_packed"] = aggregate.get("episode_judge_accuracy_e3")
        else:
            aggregate["e3_judge_accuracy_packed"] = None
        if getattr(args, "judge", "llm") == "symbolic_full_episode":
            aggregate["e3_judge_accuracy_full_episode"] = aggregate.get(
                "episode_judge_accuracy_e3"
            )
        else:
            aggregate["e3_judge_accuracy_full_episode"] = None
        if getattr(args, "judge", "llm") == "symbolic_packed_allcritical":
            aggregate["e3_judge_accuracy_packed_allcritical"] = aggregate.get("episode_judge_accuracy_e3")
        else:
            e3_allcritical_vals = [
                1.0 if rec.get("judge_correct_packed_allcritical") else 0.0
                for rec in records
                if rec.get("episode_id") == 3
                and rec.get("judge_correct_packed_allcritical") is not None
            ]
            aggregate["e3_judge_accuracy_packed_allcritical"] = (
                sum(e3_allcritical_vals) / len(e3_allcritical_vals)
                if e3_allcritical_vals
                else None
            )
        if getattr(args, "judge", "llm") != "symbolic_full_episode":
            e3_full_episode_vals = [
                1.0 if rec.get("judge_correct_full_episode") else 0.0
                for rec in records
                if rec.get("episode_id") == 3
                and rec.get("judge_correct_full_episode") is not None
            ]
            if e3_full_episode_vals:
                aggregate["e3_judge_accuracy_full_episode"] = (
                    sum(e3_full_episode_vals) / len(e3_full_episode_vals)
                )
    if getattr(args, "open_policy", "current") == "oracle_open_if_in_union":
        aggregate["selection_upper_bound_judge_acc"] = aggregate.get("judge_accuracy")
    else:
        aggregate["selection_upper_bound_judge_acc"] = None
    aggregate["bridged_ab_slices"] = compute_bridged_ab_slices(records)
    action_reward_mean = {}
    action_counts = {}
    action_cov_mean = {}
    action_winning_rate = {}
    if controller_actions:
        action_rewards: Dict[str, List[float]] = {}
        action_covs: Dict[str, List[float]] = {}
        action_wins: Dict[str, List[float]] = {}
        for rec in records:
            action = rec.get("controller_action")
            if not action:
                continue
            reward = rec.get("controller_action_reward")
            cov = rec.get("controller_action_opened_gold_coverage")
            win = rec.get("controller_action_opened_has_winning_clause")
            if isinstance(reward, (int, float)):
                action_rewards.setdefault(action, []).append(float(reward))
            if isinstance(cov, (int, float)):
                action_covs.setdefault(action, []).append(float(cov))
            if win is not None:
                action_wins.setdefault(action, []).append(1.0 if win else 0.0)
        for action, count in controller_actions.items():
            action_counts[action] = count
            rewards = action_rewards.get(action, [])
            covs = action_covs.get(action, [])
            wins = action_wins.get(action, [])
            action_reward_mean[action] = sum(rewards) / len(rewards) if rewards else 0.0
            action_cov_mean[action] = sum(covs) / len(covs) if covs else 0.0
            action_winning_rate[action] = sum(wins) / len(wins) if wins else 0.0
    report = {
        "method": method,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "metrics": aggregate,
        "counts": {"tasks": len(tasks)},
        "usage": {
            "tool_calls_avg": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
            "open_calls_avg": sum(open_calls) / len(open_calls) if open_calls else 0.0,
            "prompt_tokens_avg": sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0.0,
        },
        "tool_calls": sum(tool_calls),
        "open_calls": sum(open_calls),
        "records": records,
        "thread_records": thread_records,
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "controller_policy": controller_policy,
        "controller_actions_distribution": controller_actions,
        "controller_action_reward_mean": action_reward_mean,
        "controller_action_counts": action_counts,
        "controller_action_opened_gold_coverage_mean": action_cov_mean,
        "controller_action_opened_has_winning_clause_rate": action_winning_rate,
        "goc_internal_budget_active_tokens": (
            int(goc_internal_budget_active_tokens) if method in {"goc", "goc_base"} else None
        ),
        "goc_internal_budget_unfold_tokens": (
            int(goc_internal_budget_unfold_tokens) if method in {"goc", "goc_base"} else None
        ),
        "goc_internal_unfold_k": int(goc_internal_unfold_k) if method in {"goc", "goc_base"} else None,
        "goc_unfold_max_nodes": int(goc_unfold_max_nodes_cfg) if method in {"goc", "goc_base"} else None,
        "goc_unfold_hops": int(goc_unfold_hops_cfg) if method in {"goc", "goc_base"} else None,
        "goc_unfold_budget_mode": (
            str(goc_unfold_budget_mode_cfg) if method in {"goc", "goc_base"} else None
        ),

        "goc_unfold_policy": str(goc_unfold_policy) if method in {"goc", "goc_base"} else None,
        "goc_unfold_default_max_nodes": int(goc_unfold_default_max_nodes_cfg)
        if method in {"goc", "goc_base"}
        else None,
        "goc_unfold_default_hops": int(goc_unfold_default_hops_cfg) if method in {"goc", "goc_base"} else None,
        "goc_unfold_pivot_max_nodes": int(goc_unfold_pivot_max_nodes_cfg)
        if method in {"goc", "goc_base"}
        else None,
        "goc_unfold_pivot_hops": int(goc_unfold_pivot_hops_cfg) if method in {"goc", "goc_base"} else None,
        "pivot_message_style": _normalize_pivot_message_style(
            getattr(args, "pivot_message_style", "transcript")
        ),
        "pivot_gold_mode": _normalize_pivot_gold_mode(
            getattr(args, "pivot_gold_mode", "respect_ticket_updated")
        ),
        "goc_enable_avoids": bool(getattr(args, "goc_enable_avoids", True))
        if method in {"goc", "goc_base"}
        else None,
        "goc_avoids_mode": _resolve_goc_avoids_mode(args)
        if method in {"goc", "goc_base"}
        else None,
    }
    if controller_policy == "rerank" and isinstance(controller, RerankController):
        report["rerank_weights"] = dict(controller.weights)
        report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    return report

def cmd_generate(args: argparse.Namespace) -> None:
    if _is_traceops_benchmark(args):
        _cmd_generate_traceops(args)
        return
    _normalize_scenario_mode_arg(args)
    _apply_preset(args)
    _apply_generation_defaults(args)
    _, tasks, _, _ = generate_world_and_tasks(
        out_dir=args.out_dir,
        seed=args.seed,
        n_docs=args.n_docs,
        clauses_per_doc=args.clauses_per_doc,
        n_tasks=args.n_tasks,
        n_threads=getattr(args, "n_threads", None),
        exception_chain_depth=args.exception_chain_depth,
        definition_dependency_depth=getattr(args, "definition_dependency_depth", 1),
        definition_dependency_extra_terms=getattr(args, "definition_dependency_extra_terms", 0),
        force_exception_chain_depth=getattr(args, "force_exception_chain_depth", 0),
        force_exception_chain_all_apply=bool(getattr(args, "force_exception_chain_all_apply", False)),
        update_rate=args.update_rate,
        definition_density=args.definition_density,
        distractor_strength=args.distractor_strength,
        scenario_mode=args.scenario_mode,
        bridge_prob=args.bridge_prob,
        bridged_mix_canonical_in_ticket_rate=getattr(args, "bridged_mix_canonical_in_ticket_rate", 0.0),
        alias_density=args.alias_density,
        canonical_density=args.canonical_density,
        bridge_kind=args.bridge_kind,
        exclusive_core_evidence=getattr(args, "exclusive_core_evidence", False),
        open_budget_e1=getattr(args, "open_budget_e1", 4),
        open_budget_e2=getattr(args, "open_budget_e2", 4),
        open_budget_e3=getattr(args, "open_budget_e3", 0),
        tool_budget_e1=getattr(args, "tool_budget_e1", 50),
        tool_budget_e2=getattr(args, "tool_budget_e2", 50),
        tool_budget_e3=getattr(args, "tool_budget_e3", 0),
        branch_distractor_rate=getattr(args, "branch_distractor_rate", 0.5),
        e3_clause_jitter_max_chars_critical=int(
            getattr(args, "e3_clause_jitter_max_chars_critical", 0) or 0
        ),
        e3_clause_jitter_max_chars_noncritical=int(
            getattr(args, "e3_clause_jitter_max_chars_noncritical", 0) or 0
        ),
        e3_clause_jitter_max_chars_decoy=int(
            getattr(args, "e3_clause_jitter_max_chars_decoy", 0) or 0
        ),
        e3_litm_filler_count_min=int(getattr(args, "e3_litm_filler_count_min", 0) or 0),
        e3_litm_filler_count_max=int(getattr(args, "e3_litm_filler_count_max", 0) or 0),
        e3_litm_filler_len_jitter_max=int(
            getattr(args, "e3_litm_filler_len_jitter_max", 0) or 0
        ),
        e3_clause_jitter_max_chars=int(getattr(args, "e3_clause_jitter_max_chars", 0) or 0),
        e3_clause_jitter_scope=str(getattr(args, "e3_clause_jitter_scope", "decoy_only") or "decoy_only"),
        pivot_rate=float(getattr(args, "pivot_rate", 0.0) or 0.0),
        pivot_type=str(getattr(args, "pivot_type", "retention_flip") or "retention_flip"),
        preset_name=getattr(args, "preset", None),
    )
    thread_ids = {t.thread_id for t in tasks if getattr(t, "thread_id", None)}
    if thread_ids:
        print(
            f"Generated PolicyOps Arena v0 data. tasks={len(tasks)} threads={len(thread_ids)}",
            flush=True,
        )
    else:
        print(f"Generated PolicyOps Arena v0 data. tasks={len(tasks)}", flush=True)


def cmd_eval(args: argparse.Namespace) -> None:
    if _is_traceops_benchmark(args):
        _cmd_eval_traceops(args)
        return
    _normalize_scenario_mode_arg(args)
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"

    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)
    if getattr(args, "n_threads", None):
        thread_ids = [t.thread_id for t in tasks if getattr(t, "thread_id", None)]
        if thread_ids:
            unique = sorted(dict.fromkeys(thread_ids))
            selected = set(unique[: int(args.n_threads)])
            tasks = [t for t in tasks if t.thread_id in selected]
    if any(getattr(t, "thread_id", None) for t in tasks):
        tasks = sorted(
            tasks,
            key=lambda t: (
                t.thread_id or "",
                t.episode_id or 0,
                t.task_id,
            ),
        )

    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")

    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)

    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller and args.method == "goc":
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="rerank",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="bandit",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(state_path)

    run_dir = Path(base_dir) / "runs" / args.method
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    report = _evaluate_method(
        args.method,
        world,
        eval_tasks,
        args,
        client,
        run_dir,
        run_id=timestamp,
        controller=controller,
        controller_mode=controller_mode,
        controller_policy=args.controller_policy,
        llm_backend=llm_backend,
        client_class=client_class,
        resolved_model=resolved_model,
    )

    report["task_split"] = args.task_split
    report["train_ratio"] = args.train_ratio
    report["split_seed"] = args.split_seed
    report["num_train_tasks"] = len(train_tasks)
    report["num_eval_tasks"] = len(eval_tasks)

    out_path = run_dir / f"{timestamp}.json"
    save_report(out_path, report)

    print("Evaluation complete.")
    for key, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
    print(f"Report saved to {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    if _is_traceops_benchmark(args):
        _cmd_compare_traceops(args)
        return
    _normalize_scenario_mode_arg(args)
    if getattr(args, "thread_context_budget_sweep", ""):
        sweep_values = [
            int(v.strip())
            for v in str(args.thread_context_budget_sweep).split(",")
            if v.strip()
        ]
        if not sweep_values:
            raise ValueError("thread_context_budget_sweep provided but no budgets parsed")
        base_dir = args.out_dir or _default_base_dir()
        compare_dir = Path(base_dir) / "runs" / "compare"
        sweep_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sweep_dir = Path(base_dir) / "runs" / "context_budget_sweep" / sweep_stamp
        sweep_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, Any]] = []
        total_budgets = len(sweep_values)
        for budget_idx, budget in enumerate(sweep_values, start=1):
            print(
                f"[compare-sweep] starting budget={budget} ({budget_idx}/{total_budgets})",
                flush=True,
            )
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.thread_context_budget_sweep = ""
            sweep_args.thread_context_budget_chars = budget
            before = set(compare_dir.glob("*.json")) if compare_dir.exists() else set()
            cmd_compare(sweep_args)
            after = set(compare_dir.glob("*.json")) if compare_dir.exists() else set()
            new_files = sorted(list(after - before), key=lambda p: p.stat().st_mtime)
            report_path = new_files[-1] if new_files else None
            if report_path is None:
                continue
            print(
                f"[compare-sweep] finished budget={budget} report={report_path}",
                flush=True,
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            for method, report_obj in payload.get("method_reports", {}).items():
                metrics = report_obj.get("metrics", {}) or {}
                summary_rows.append(
                    {
                        "budget": budget,
                        "method": method,
                        "report_json": str(report_path),
                        "goc_internal_budget_active_tokens": (
                            report_obj.get("goc_internal_budget_active_tokens")
                            if method == "goc"
                            else None
                        ),
                        "goc_internal_budget_unfold_tokens": (
                            report_obj.get("goc_internal_budget_unfold_tokens")
                            if method == "goc"
                            else None
                        ),
                        "goc_internal_unfold_k": (
                            report_obj.get("goc_internal_unfold_k")
                            if method == "goc"
                            else None
                        ),
                        "goc_unfold_max_nodes": (
                            report_obj.get("goc_unfold_max_nodes")
                            if method == "goc"
                            else None
                        ),
                        "goc_unfold_hops": (
                            report_obj.get("goc_unfold_hops")
                            if method == "goc"
                            else None
                        ),
                        "goc_unfold_budget_mode": (
                            report_obj.get("goc_unfold_budget_mode")
                            if method == "goc"
                            else None
                        ),
                        "judge_accuracy_packed": metrics.get("judge_accuracy_packed"),
                        "judge_accuracy_packed_allcritical": metrics.get(
                            "judge_accuracy_packed_allcritical"
                        ),
                        "rank_success_rate": metrics.get("rank_success_rate"),
                        "winning_in_union_rate": metrics.get("winning_in_union_rate"),
                        "selection_efficiency": metrics.get("selection_efficiency"),
                        "e3_context_token_est_mean": metrics.get("e3_context_token_est_mean"),
                        "e3_packed_token_est_mean": metrics.get("e3_packed_token_est_mean"),
                        "acc_per_1k_tokens": metrics.get("acc_per_1k_tokens"),
                        "cost_per_correct_token_est": metrics.get("cost_per_correct_token_est"),
                        "e3_context_truncated_rate": metrics.get("e3_context_truncated_rate"),
                        "e3_packed_all_critical_rate": metrics.get(
                            "e3_packed_all_critical_rate"
                        ),
                        "e3_packed_any_critical_rate": metrics.get(
                            "e3_packed_any_critical_rate"
                        ),
                        "e3_packed_critical_count_mean": metrics.get(
                            "e3_packed_critical_count_mean"
                        ),
                        "e3_decoy_clause_count_mean": metrics.get(
                            "e3_decoy_clause_count_mean"
                        ),
                        "e3_context_clause_count_mean": metrics.get(
                            "e3_context_clause_count_mean"
                        ),
                        "e3_context_chars_used_mean": metrics.get(
                            "e3_context_chars_used_mean"
                        ),
                        "goc_unfolded_clause_count_mean": metrics.get(
                            "goc_unfolded_clause_count_mean"
                        ),
                        "goc_closure_candidate_count_mean": metrics.get(
                            "goc_closure_candidate_count_mean"
                        ),
                        "goc_unfolded_node_count_mean": metrics.get(
                            "goc_unfolded_node_count_mean"
                        ),
                        "goc_unfolded_critical_clause_count_mean": metrics.get(
                            "goc_unfolded_critical_clause_count_mean"
                        ),
                        "goc_folded_episode_count_mean": metrics.get(
                            "goc_folded_episode_count_mean"
                        ),
                        "closure_recall_core_mean": metrics.get(
                            "closure_recall_core_mean"
                        ),
                        "avoided_node_injected_rate": metrics.get(
                            "avoided_node_injected_rate"
                        ),
                        "wrong_branch_recall_rate_mean": metrics.get(
                            "wrong_branch_recall_rate_mean"
                        ),
                    }
                )
        if summary_rows:
            csv_path = sweep_dir / "results_context_budget_sweep.csv"
            md_path = sweep_dir / "results_context_budget_sweep.md"
            fieldnames = [
                "budget",
                "method",
                "report_json",
                "goc_internal_budget_active_tokens",
                "goc_internal_budget_unfold_tokens",
                "goc_internal_unfold_k",
                "goc_unfold_max_nodes",
                "goc_unfold_hops",
                "goc_unfold_budget_mode",
                "judge_accuracy_packed",
                "judge_accuracy_packed_allcritical",
                "rank_success_rate",
                "winning_in_union_rate",
                "selection_efficiency",
                "e3_context_token_est_mean",
                "e3_packed_token_est_mean",
                "acc_per_1k_tokens",
                "cost_per_correct_token_est",
                "e3_context_truncated_rate",
                "e3_packed_all_critical_rate",
                "e3_packed_any_critical_rate",
                "e3_packed_critical_count_mean",
                "e3_decoy_clause_count_mean",
                "e3_context_clause_count_mean",
                "e3_context_chars_used_mean",
                "goc_unfolded_clause_count_mean",
                "goc_closure_candidate_count_mean",
                "goc_unfolded_node_count_mean",
                "goc_unfolded_critical_clause_count_mean",
                "goc_folded_episode_count_mean",
                "closure_recall_core_mean",
                "avoided_node_injected_rate",
                "wrong_branch_recall_rate_mean",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            lines = []
            lines.append("# Context Budget Sweep Summary")
            lines.append("")
            lines.append("|" + "|".join(fieldnames) + "|")
            lines.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
            for row in summary_rows:
                lines.append(
                    "|"
                    + "|".join(
                        f"{row.get(k):.4f}" if isinstance(row.get(k), (int, float)) else str(row.get(k))
                        for k in fieldnames
                    )
                    + "|"
                )
            md_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Context budget sweep summary saved to {csv_path}")
        return
    _apply_preset(args)
    _apply_threaded_budget_default(args)
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"

    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)
    if getattr(args, "n_threads", None):
        thread_ids = [t.thread_id for t in tasks if getattr(t, "thread_id", None)]
        if thread_ids:
            unique = sorted(dict.fromkeys(thread_ids))
            selected = set(unique[: int(args.n_threads)])
            tasks = [t for t in tasks if t.thread_id in selected]
    if any(getattr(t, "thread_id", None) for t in tasks):
        tasks = sorted(
            tasks,
            key=lambda t: (
                t.thread_id or "",
                t.episode_id or 0,
                t.task_id,
            ),
        )

    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")
    (
        goc_internal_budget_active_tokens_eff,
        goc_internal_budget_unfold_tokens_eff,
        goc_internal_unfold_k_eff,
    ) = _resolve_goc_internal_budgets(args)

    methods = args.methods or ["topk", "full", "goc", "oracle", "engine"]
    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)
    if args.controller_mode == "train" and not train_tasks:
        train_tasks = eval_tasks
    if llm_backend == "openai":
        llm_methods = [m for m in methods if m != "engine"]
        est_calls = len(eval_tasks) * len(llm_methods)
        print(
            f"[compare] eval_tasks={len(eval_tasks)} methods={methods} "
            f"llm_methods={llm_methods} est_llm_calls~{est_calls}",
            flush=True,
        )
    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="rerank",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_report = _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    Path(base_dir) / "runs" / "controller_train",
                    controller=controller,
                    controller_mode="train",
                    controller_policy="bandit",
                )
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                train_out_dir.mkdir(parents=True, exist_ok=True)
                train_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_report(train_out_dir / f"{train_stamp}.json", train_report)
                controller.save(state_path)

    compare_dir = Path(base_dir) / "runs" / "compare"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    compare_run_dir = compare_dir / timestamp
    compare_run_dir.mkdir(parents=True, exist_ok=True)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for method in methods:
        method_dir = compare_run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_report = _evaluate_method(
            method,
            world,
            eval_tasks,
            args,
            client,
            method_dir,
            run_id=timestamp,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            controller_policy=args.controller_policy,
            llm_backend=llm_backend,
            client_class=client_class,
            resolved_model=resolved_model,
        )
        method_reports[method] = method_report

        records = method_report.get("records", [])
        prompt_tokens_vals = [int(r.get("prompt_tokens", 0)) for r in records]
        open_calls_vals = [int(r.get("open_calls", 0)) for r in records]
        tool_calls_vals = [int(r.get("tool_calls", 0)) for r in records]
        opened_gold_cov_vals = [
            float(r.get("opened_gold_coverage"))
            for r in records
            if isinstance(r.get("opened_gold_coverage"), (int, float))
        ]
        opened_has_win_vals = [
            1.0 if r.get("opened_has_winning_clause") else 0.0
            for r in records
            if r.get("opened_has_winning_clause") is not None
        ]
        gold_in_topk_vals = [
            1.0 if r.get("gold_in_search_topk") else 0.0
            for r in records
            if r.get("gold_in_search_topk") is not None
        ]
        winning_rank_vals = [
            int(r.get("winning_clause_rank"))
            for r in records
            if isinstance(r.get("winning_clause_rank"), int)
        ]
        min_gold_rank_vals = [
            int(r.get("min_gold_rank"))
            for r in records
            if isinstance(r.get("min_gold_rank"), int)
        ]
        gold_score_gap_vals = [
            float(r.get("gold_score_gap"))
            for r in records
            if isinstance(r.get("gold_score_gap"), (int, float))
        ]
        summary[method] = {
            "decision_accuracy": method_report["metrics"].get("decision_accuracy"),
            "judge_accuracy": method_report["metrics"].get("judge_accuracy"),
            "judge_accuracy_packed": method_report["metrics"].get("judge_accuracy_packed"),
            "judge_accuracy_full_episode": method_report["metrics"].get(
                "judge_accuracy_full_episode"
            ),
            "judge_accuracy_packed_allcritical": method_report["metrics"].get(
                "judge_accuracy_packed_allcritical"
            ),
            "acc_no_core_evidence_rate": method_report["metrics"].get("acc_no_core_evidence_rate"),
            "judge_used_any_core_rate": method_report["metrics"].get("judge_used_any_core_rate"),
            "judge_used_any_bridge_rate": method_report["metrics"].get("judge_used_any_bridge_rate"),
            "judge_used_any_critical_core_rate": method_report["metrics"].get(
                "judge_used_any_critical_core_rate"
            ),
            "judge_used_any_decoy_rate": method_report["metrics"].get(
                "judge_used_any_decoy_rate"
            ),
            "judge_supporting_count_mean": method_report["metrics"].get(
                "judge_supporting_count_mean"
            ),
            "full_episode_supporting_count_mean": method_report["metrics"].get(
                "full_episode_supporting_count_mean"
            ),
            "selection_upper_bound_judge_acc": method_report["metrics"].get(
                "selection_upper_bound_judge_acc"
            ),
            "thread_judge_accuracy": method_report["metrics"].get("thread_judge_accuracy"),
            "thread_decision_accuracy": method_report["metrics"].get("thread_decision_accuracy"),
            "episode_judge_accuracy_e1": method_report["metrics"].get("episode_judge_accuracy_e1"),
            "episode_judge_accuracy_e2": method_report["metrics"].get("episode_judge_accuracy_e2"),
            "episode_judge_accuracy_e3": method_report["metrics"].get("episode_judge_accuracy_e3"),
            "episode_commit_success_e1": method_report["metrics"].get("episode_commit_success_e1"),
            "episode_commit_success_e2": method_report["metrics"].get("episode_commit_success_e2"),
            "episode_commit_success_e3": method_report["metrics"].get("episode_commit_success_e3"),
            "deep_rank_core_rate": method_report["metrics"].get("deep_rank_core_rate"),
            "closure_recall_core_mean": method_report["metrics"].get(
                "closure_recall_core_mean"
            ),
            "avoided_node_injected_rate": method_report["metrics"].get(
                "avoided_node_injected_rate"
            ),
            "wrong_branch_recall_rate_mean": method_report["metrics"].get(
                "wrong_branch_recall_rate_mean"
            ),
            "min_gold_core_rank_union_mean": method_report["metrics"].get(
                "min_gold_core_rank_union_mean"
            ),
            "min_gold_core_rank_union_median": method_report["metrics"].get(
                "min_gold_core_rank_union_median"
            ),
            "min_gold_core_rank_union_p90": method_report["metrics"].get(
                "min_gold_core_rank_union_p90"
            ),
            "opened_bridge_count_mean": method_report["metrics"].get("opened_bridge_count_mean"),
            "opened_meta_count_mean": method_report["metrics"].get("opened_meta_count_mean"),
            "opened_rule_count_mean": method_report["metrics"].get("opened_rule_count_mean"),
            "condition_f1": method_report["metrics"].get("condition_f1"),
            "evidence_recall": method_report["metrics"].get("evidence_recall"),
            "critical_evidence_hit": method_report["metrics"].get("critical_evidence_hit"),
            "bridge_found_rate": method_report["metrics"].get("bridge_found_rate"),
            "canonical_used_in_query2_rate": method_report["metrics"].get("canonical_used_in_query2_rate"),
            "update_found_when_needed_rate": method_report["metrics"].get("update_found_when_needed_rate"),
            "winning_rank_exists_rate": method_report["metrics"].get("winning_rank_exists_rate"),
            "decision_accuracy_when_winning_rank_exists": method_report["metrics"].get(
                "decision_accuracy_when_winning_rank_exists"
            ),
            "prompt_tokens": _avg_p90(prompt_tokens_vals),
            "open_calls": _avg_p90(open_calls_vals),
            "tool_calls": _avg_p90(tool_calls_vals),
            "opened_gold_coverage_mean": sum(opened_gold_cov_vals) / len(opened_gold_cov_vals)
            if opened_gold_cov_vals
            else 0.0,
            "opened_has_winning_clause_rate": sum(opened_has_win_vals) / len(opened_has_win_vals)
            if opened_has_win_vals
            else 0.0,
            "gold_in_search_topk_rate": sum(gold_in_topk_vals) / len(gold_in_topk_vals)
            if gold_in_topk_vals
            else 0.0,
            "winning_clause_rank_mean": sum(winning_rank_vals) / len(winning_rank_vals)
            if winning_rank_vals
            else None,
            "min_gold_rank_mean": sum(min_gold_rank_vals) / len(min_gold_rank_vals)
            if min_gold_rank_vals
            else None,
            "gold_score_gap_mean": sum(gold_score_gap_vals) / len(gold_score_gap_vals)
            if gold_score_gap_vals
            else None,
            "rank_success_rate": method_report["metrics"].get("rank_success_rate"),
            "winning_in_union_rate": method_report["metrics"].get("winning_in_union_rate"),
            "policy_gain_over_rank": method_report["metrics"].get("policy_gain_over_rank"),
            "rank_gap": method_report["metrics"].get("rank_gap"),
            "feasible_open_rate": method_report["metrics"].get("feasible_open_rate"),
            "realized_open_rate": method_report["metrics"].get("realized_open_rate"),
            "selection_gap": method_report["metrics"].get("selection_gap"),
            "selection_efficiency": method_report["metrics"].get("selection_efficiency"),
            "e3_judge_accuracy_packed": method_report["metrics"].get("e3_judge_accuracy_packed"),
            "e3_judge_accuracy_full_episode": method_report["metrics"].get(
                "e3_judge_accuracy_full_episode"
            ),
            "e3_judge_accuracy_packed_allcritical": method_report["metrics"].get(
                "e3_judge_accuracy_packed_allcritical"
            ),
            "e3_packed_any_critical_rate_full_episode": method_report["metrics"].get(
                "e3_packed_any_critical_rate_full_episode"
            ),
            "e3_packed_all_critical_rate_full_episode": method_report["metrics"].get(
                "e3_packed_all_critical_rate_full_episode"
            ),
            "e3_packed_total_chars_before_mean": method_report["metrics"].get(
                "e3_packed_total_chars_before_mean"
            ),
            "e3_packed_total_chars_after_mean": method_report["metrics"].get(
                "e3_packed_total_chars_after_mean"
            ),
            "e3_context_chars_used_mean": method_report["metrics"].get(
                "e3_context_chars_used_mean"
            ),
            "e3_context_token_est_mean": method_report["metrics"].get(
                "e3_context_token_est_mean"
            ),
            "e3_packed_token_est_mean": method_report["metrics"].get(
                "e3_packed_token_est_mean"
            ),
            "e3_context_chars_used_total": method_report["metrics"].get(
                "e3_context_chars_used_total"
            ),
            "e3_context_token_est_total": method_report["metrics"].get(
                "e3_context_token_est_total"
            ),
            "e3_packed_dropped_clause_count_mean": method_report["metrics"].get(
                "e3_packed_dropped_clause_count_mean"
            ),
            "cost_per_correct_chars": method_report["metrics"].get(
                "cost_per_correct_chars"
            ),
            "cost_per_correct_token_est": method_report["metrics"].get(
                "cost_per_correct_token_est"
            ),
            "acc_per_1k_tokens": method_report["metrics"].get("acc_per_1k_tokens"),
        }
        if method == "goc":
            summary[method]["goc_internal_budget_active_tokens"] = method_report.get(
                "goc_internal_budget_active_tokens"
            )
            summary[method]["goc_internal_budget_unfold_tokens"] = method_report.get(
                "goc_internal_budget_unfold_tokens"
            )
            summary[method]["goc_internal_unfold_k"] = method_report.get(
                "goc_internal_unfold_k"
            )
            summary[method]["goc_unfold_max_nodes"] = method_report.get(
                "goc_unfold_max_nodes"
            )
            summary[method]["goc_unfold_hops"] = method_report.get(
                "goc_unfold_hops"
            )
            summary[method]["goc_unfold_budget_mode"] = method_report.get(
                "goc_unfold_budget_mode"
            )
        if method == "oracle":
            oracle_cov = [
                r.get("oracle_gold_coverage")
                for r in records
                if isinstance(r.get("oracle_gold_coverage"), (int, float))
            ]
            summary[method]["oracle_gold_coverage"] = (
                sum(oracle_cov) / len(oracle_cov) if oracle_cov else 0.0
            )

    git_sha = None
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_sha = None
    task_open_budgets = {t.budgets.get("open_budget", 5) for t in eval_tasks}
    open_budget = list(task_open_budgets)[0] if len(task_open_budgets) == 1 else None
    mix_rate = getattr(args, "bridged_mix_canonical_in_ticket_rate", None)
    if mix_rate is None:
        mix_rate = sum(1 for t in eval_tasks if not getattr(t, "bridge_clause_id", None)) / max(
            1, len(eval_tasks)
        )
    scenario_params = {
        "preset": getattr(args, "preset", None),
        "scenario_mode": getattr(args, "scenario_mode", "v0"),
        "seed": getattr(args, "seed", None),
        "bridge_bonus": getattr(args, "bridge_bonus", 1.5),
        "bridged_mix_canonical_in_ticket_rate": mix_rate,
        "open_budget": open_budget,
        "open_split_mode": getattr(args, "open_split_mode", "all_union_rank"),
        "open_split_hop1": getattr(args, "open_split_hop1", 0),
        "open_policy": getattr(args, "open_policy", "current"),
        "hop1_query_mode": getattr(args, "hop1_query_mode", "stripped"),
        "agent_query_policy": getattr(args, "agent_query_policy", "single_hop"),
        "thread_context_budget_chars": getattr(args, "thread_context_budget_chars", None),
        "thread_open_policy": getattr(args, "thread_open_policy", None),
        "thread_context_budget_sweep": getattr(args, "thread_context_budget_sweep", None),
        "goc_internal_budget_active_tokens": int(goc_internal_budget_active_tokens_eff),
        "goc_internal_budget_unfold_tokens": int(goc_internal_budget_unfold_tokens_eff),
        "goc_internal_unfold_k": int(goc_internal_unfold_k_eff),
        "goc_unfold_max_nodes": int(getattr(args, "goc_unfold_max_nodes", 0) or 0),
        "goc_unfold_hops": int(getattr(args, "goc_unfold_hops", 0) or 0),
        "goc_unfold_budget_mode": str(
            getattr(args, "goc_unfold_budget_mode", "nodes_and_hops") or "nodes_and_hops"
        ),

        "goc_unfold_policy": str(getattr(args, "goc_unfold_policy", "fixed") or "fixed"),
        "goc_unfold_default_max_nodes": int(getattr(args, "goc_unfold_default_max_nodes", 0) or 0),
        "goc_unfold_default_hops": int(getattr(args, "goc_unfold_default_hops", 0) or 0),
        "goc_unfold_pivot_max_nodes": int(getattr(args, "goc_unfold_pivot_max_nodes", 0) or 0),
        "goc_unfold_pivot_hops": int(getattr(args, "goc_unfold_pivot_hops", 0) or 0),
        "pivot_message_style": _normalize_pivot_message_style(
            getattr(args, "pivot_message_style", "transcript")
        ),
        "pivot_gold_mode": _normalize_pivot_gold_mode(
            getattr(args, "pivot_gold_mode", "respect_ticket_updated")
        ),
        "goc_enable_avoids": bool(getattr(args, "goc_enable_avoids", True)),
        "goc_avoids_mode": _resolve_goc_avoids_mode(args),
        "goc_applicability_seed_enable": bool(
            getattr(args, "goc_applicability_seed_enable", False)
        ),
        "goc_applicability_seed_topk": int(
            getattr(args, "goc_applicability_seed_topk", 8) or 0
        ),
        "goc_dependency_closure_enable": bool(
            getattr(args, "goc_dependency_closure_enable", False)
        ),
        "goc_dependency_closure_topk": int(
            getattr(args, "goc_dependency_closure_topk", 12) or 0
        ),
        "goc_dependency_closure_hops": int(
            getattr(args, "goc_dependency_closure_hops", 1) or 0
        ),
        "goc_dependency_closure_universe": str(
            getattr(args, "goc_dependency_closure_universe", "candidates")
            or "candidates"
        ),
        "e3_clause_jitter_max_chars": world.meta.get("e3_clause_jitter_max_chars"),
        "e3_clause_jitter_max_chars_critical": world.meta.get("e3_clause_jitter_max_chars_critical"),
        "e3_clause_jitter_max_chars_noncritical": world.meta.get("e3_clause_jitter_max_chars_noncritical"),
        "e3_clause_jitter_max_chars_decoy": world.meta.get("e3_clause_jitter_max_chars_decoy"),
        "e3_clause_jitter_scope": world.meta.get("e3_clause_jitter_scope"),
        "e3_litm_filler_count_min": world.meta.get("e3_litm_filler_count_min"),
        "e3_litm_filler_count_max": world.meta.get("e3_litm_filler_count_max"),
        "e3_litm_filler_len_jitter_max": world.meta.get("e3_litm_filler_len_jitter_max"),
        "pivot_rate_requested": world.meta.get("pivot_rate_requested"),
        "pivot_rate_actual": world.meta.get("pivot_rate_actual"),
        "pivot_type": world.meta.get("pivot_type"),
    }
    if any(getattr(t, "thread_id", None) for t in eval_tasks):
        scenario_params["n_threads"] = len({t.thread_id for t in eval_tasks if t.thread_id})
        scenario_params["n_threads_requested"] = world.meta.get("n_threads_requested")
        scenario_params["n_threads_generated_raw"] = world.meta.get("n_threads_generated_raw")
        scenario_params["n_threads_generated_final"] = world.meta.get("n_threads_generated_final")
    scenario_params["exclusive_core_evidence"] = world.meta.get("exclusive_core_evidence")
    compare_report = {
        "methods": methods,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": timestamp,
        "git_sha": git_sha,
        "invoked_cmdline": " ".join(sys.argv),
        "judge": getattr(args, "judge", "llm"),
        "scenario_params": scenario_params,
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "gold_decision_distribution": gold_decision_distribution(eval_tasks),
        "summary": summary,
        "method_reports": method_reports,
        "goc_internal_budget_active_tokens": int(goc_internal_budget_active_tokens_eff),
        "goc_internal_budget_unfold_tokens": int(goc_internal_budget_unfold_tokens_eff),
        "goc_internal_unfold_k": int(goc_internal_unfold_k_eff),
        "goc_unfold_max_nodes": int(getattr(args, "goc_unfold_max_nodes", 0) or 0),
        "goc_unfold_hops": int(getattr(args, "goc_unfold_hops", 0) or 0),
        "goc_unfold_budget_mode": str(
            getattr(args, "goc_unfold_budget_mode", "nodes_and_hops") or "nodes_and_hops"
        ),
        "goc_applicability_seed_enable": bool(
            getattr(args, "goc_applicability_seed_enable", False)
        ),
        "goc_applicability_seed_topk": int(
            getattr(args, "goc_applicability_seed_topk", 8) or 0
        ),
        "goc_dependency_closure_enable": bool(
            getattr(args, "goc_dependency_closure_enable", False)
        ),
        "goc_dependency_closure_topk": int(
            getattr(args, "goc_dependency_closure_topk", 12) or 0
        ),
        "goc_dependency_closure_hops": int(
            getattr(args, "goc_dependency_closure_hops", 1) or 0
        ),
        "goc_dependency_closure_universe": str(
            getattr(args, "goc_dependency_closure_universe", "candidates")
            or "candidates"
        ),
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "controller_policy": args.controller_policy,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    if args.controller_policy == "rerank" and isinstance(controller, RerankController):
        compare_report["rerank_weights"] = dict(controller.weights)
        compare_report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

    out_path = compare_dir / f"{timestamp}.json"
    save_report(out_path, compare_report)
    quickcheck_compare_report(compare_report, label=timestamp)
    if args.save_goc_graph or args.save_goc_dot:
        readme_path = compare_run_dir / "README_debug.md"
        readme_path.write_text(
            "\n".join(
                [
                    "# Debug Run Commands",
                    "",
                    "## Compare command",
                    f"PYTHONPATH=src python -m policyops.run compare "
                    f"--llm {args.llm} --model {args.model} --methods {' '.join(methods)} "
                    f"--save_goc_graph --save_goc_dot",
                    "",
                    "## Triage command",
                    f"PYTHONPATH=src python -m policyops.triage "
                    f"--compare_report {out_path} --method goc --max_per_bucket 20",
                    "",
                    "## Bucket checklist",
                    "- A_open_selection_fail: gold in search but not opened",
                    "- B_reasoning_fail: winning clause opened but wrong decision",
                    "- C_retrieval_fail: gold not in search",
                    "- D_decision_confusion: require_condition mispredicted",
                    "- E_budget_edge_fail: winning clause beyond open_budget",
                    "- F_evidence_padding_artifact: evidence_before empty, after non-empty",
                ]
            ),
            encoding="utf-8",
        )
    print("Compare complete.")
    print(f"Report saved to {out_path}")

    if args.debug_n or args.debug_task_ids:
        debug_ids = []
        if args.debug_task_ids:
            debug_ids = [t.strip() for t in args.debug_task_ids.split(",") if t.strip()]
        for method, report in method_reports.items():
            print(f"[debug] method={method}")
            records = report.get("records", [])
            if debug_ids:
                selected = [r for r in records if r.get("task_id") in debug_ids]
            else:
                selected = records[: args.debug_n]
            for rec in selected:
                raw_preview = ""
                if args.save_raw and rec.get("raw_path"):
                    try:
                        raw_preview = Path(rec["raw_path"]).read_text(encoding="utf-8")[:300]
                    except Exception:
                        raw_preview = ""
                elif rec.get("raw_output"):
                    raw_preview = str(rec.get("raw_output", ""))[:300]
                raw_preview = " ".join(raw_preview.split())
                print(
                    f"task_id={rec.get('task_id')} gold={rec.get('gold_decision')} "
                    f"pred={rec.get('pred_decision')} opened={rec.get('opened_clause_ids')} "
                    f"error={rec.get('error')}"
                )
                print(f"raw_preview={raw_preview}")
                if rec.get("prompt_path"):
                    print(f"prompt_path={rec.get('prompt_path')}")
                if rec.get("raw_path"):
                    print(f"raw_path={rec.get('raw_path')}")


def _run_compare_with_tasks(
    world: Any,
    tasks: List[Any],
    args: argparse.Namespace,
    base_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    if args.llm == "dummy" and args.model != "dummy":
        raise RuntimeError(
            f"Refusing to run with --llm dummy while --model={args.model}. "
            "Use --llm openai or set --model dummy."
        )
    if args.llm == "openai":
        client = OpenAIClient(model=args.model, dotenv_path=args.dotenv)
        llm_backend = "openai"
    else:
        client = DummyClient()
        llm_backend = "dummy"
    client_class = client.__class__.__name__
    resolved_model = getattr(client, "model", "dummy")

    methods = args.methods or ["topk", "full", "goc", "oracle", "engine"]
    train_tasks, eval_tasks = _split_tasks(tasks, args.task_split, args.train_ratio, args.split_seed)
    controller: Controller | RerankController | None = None
    controller_mode = "off"
    state_path = None
    if args.use_controller:
        controller_mode = "eval" if args.controller_mode == "train" else args.controller_mode
        if args.controller_policy == "rerank":
            weights_path = _resolve_rerank_weights_path(base_dir, args.controller_weights_path)
            controller = RerankController.load(weights_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    train_out_dir,
                    controller=controller,
                    controller_mode="train",
                    controller_policy="rerank",
                )
                controller.save(weights_path)
        else:
            state_path = _resolve_controller_state_path(base_dir, args.controller_state_path)
            controller = Controller.load(state_path)
            if args.controller_mode == "train" and train_tasks:
                train_args = argparse.Namespace(**vars(args))
                train_args.save_prompts = False
                train_args.save_raw = False
                train_out_dir = Path(base_dir) / "runs" / "controller_train"
                _evaluate_method(
                    "goc",
                    world,
                    train_tasks,
                    train_args,
                    DummyClient(),
                    train_out_dir,
                    controller=controller,
                    controller_mode="train",
                    controller_policy="bandit",
                )
                controller.save(state_path)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    run_id = output_dir.name
    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_report = _evaluate_method(
            method,
            world,
            eval_tasks,
            args,
            client,
            method_dir,
            run_id=run_id,
            controller=controller if method == "goc" else None,
            controller_mode=controller_mode if method == "goc" else "off",
            controller_policy=args.controller_policy,
            llm_backend=llm_backend,
            client_class=client_class,
            resolved_model=resolved_model,
        )
        method_reports[method] = method_report

        records = method_report.get("records", [])
        prompt_tokens_vals = [int(r.get("prompt_tokens", 0)) for r in records]
        open_calls_vals = [int(r.get("open_calls", 0)) for r in records]
        tool_calls_vals = [int(r.get("tool_calls", 0)) for r in records]
        opened_gold_cov_vals = [
            float(r.get("opened_gold_coverage"))
            for r in records
            if isinstance(r.get("opened_gold_coverage"), (int, float))
        ]
        opened_has_win_vals = [
            1.0 if r.get("opened_has_winning_clause") else 0.0
            for r in records
            if r.get("opened_has_winning_clause") is not None
        ]
        gold_in_topk_vals = [
            1.0 if r.get("gold_in_search_topk") else 0.0
            for r in records
            if r.get("gold_in_search_topk") is not None
        ]
        winning_rank_vals = [
            int(r.get("winning_clause_rank"))
            for r in records
            if isinstance(r.get("winning_clause_rank"), int)
        ]
        min_gold_rank_vals = [
            int(r.get("min_gold_rank"))
            for r in records
            if isinstance(r.get("min_gold_rank"), int)
        ]
        gold_score_gap_vals = [
            float(r.get("gold_score_gap"))
            for r in records
            if isinstance(r.get("gold_score_gap"), (int, float))
        ]
        summary[method] = {
            "decision_accuracy": method_report["metrics"].get("decision_accuracy"),
            "judge_accuracy": method_report["metrics"].get("judge_accuracy"),
            "judge_accuracy_packed": method_report["metrics"].get("judge_accuracy_packed"),
            "judge_accuracy_full_episode": method_report["metrics"].get(
                "judge_accuracy_full_episode"
            ),
            "acc_no_core_evidence_rate": method_report["metrics"].get("acc_no_core_evidence_rate"),
            "judge_used_any_core_rate": method_report["metrics"].get("judge_used_any_core_rate"),
            "judge_used_any_bridge_rate": method_report["metrics"].get("judge_used_any_bridge_rate"),
            "judge_used_any_critical_core_rate": method_report["metrics"].get(
                "judge_used_any_critical_core_rate"
            ),
            "judge_used_any_decoy_rate": method_report["metrics"].get(
                "judge_used_any_decoy_rate"
            ),
            "judge_supporting_count_mean": method_report["metrics"].get(
                "judge_supporting_count_mean"
            ),
            "full_episode_supporting_count_mean": method_report["metrics"].get(
                "full_episode_supporting_count_mean"
            ),
            "selection_upper_bound_judge_acc": method_report["metrics"].get(
                "selection_upper_bound_judge_acc"
            ),
            "thread_judge_accuracy": method_report["metrics"].get("thread_judge_accuracy"),
            "thread_decision_accuracy": method_report["metrics"].get("thread_decision_accuracy"),
            "episode_judge_accuracy_e1": method_report["metrics"].get("episode_judge_accuracy_e1"),
            "episode_judge_accuracy_e2": method_report["metrics"].get("episode_judge_accuracy_e2"),
            "episode_judge_accuracy_e3": method_report["metrics"].get("episode_judge_accuracy_e3"),
            "episode_commit_success_e1": method_report["metrics"].get("episode_commit_success_e1"),
            "episode_commit_success_e2": method_report["metrics"].get("episode_commit_success_e2"),
            "episode_commit_success_e3": method_report["metrics"].get("episode_commit_success_e3"),
            "deep_rank_core_rate": method_report["metrics"].get("deep_rank_core_rate"),
            "closure_recall_core_mean": method_report["metrics"].get(
                "closure_recall_core_mean"
            ),
            "avoided_node_injected_rate": method_report["metrics"].get(
                "avoided_node_injected_rate"
            ),
            "wrong_branch_recall_rate_mean": method_report["metrics"].get(
                "wrong_branch_recall_rate_mean"
            ),
            "min_gold_core_rank_union_mean": method_report["metrics"].get(
                "min_gold_core_rank_union_mean"
            ),
            "min_gold_core_rank_union_median": method_report["metrics"].get(
                "min_gold_core_rank_union_median"
            ),
            "min_gold_core_rank_union_p90": method_report["metrics"].get(
                "min_gold_core_rank_union_p90"
            ),
            "opened_bridge_count_mean": method_report["metrics"].get("opened_bridge_count_mean"),
            "opened_meta_count_mean": method_report["metrics"].get("opened_meta_count_mean"),
            "opened_rule_count_mean": method_report["metrics"].get("opened_rule_count_mean"),
            "condition_f1": method_report["metrics"].get("condition_f1"),
            "evidence_recall": method_report["metrics"].get("evidence_recall"),
            "critical_evidence_hit": method_report["metrics"].get("critical_evidence_hit"),
            "prompt_tokens": _avg_p90(prompt_tokens_vals),
            "open_calls": _avg_p90(open_calls_vals),
            "tool_calls": _avg_p90(tool_calls_vals),
            "opened_gold_coverage_mean": sum(opened_gold_cov_vals) / len(opened_gold_cov_vals)
            if opened_gold_cov_vals
            else 0.0,
            "opened_has_winning_clause_rate": sum(opened_has_win_vals) / len(opened_has_win_vals)
            if opened_has_win_vals
            else 0.0,
            "gold_in_search_topk_rate": sum(gold_in_topk_vals) / len(gold_in_topk_vals)
            if gold_in_topk_vals
            else 0.0,
            "winning_clause_rank_mean": sum(winning_rank_vals) / len(winning_rank_vals)
            if winning_rank_vals
            else None,
            "min_gold_rank_mean": sum(min_gold_rank_vals) / len(min_gold_rank_vals)
            if min_gold_rank_vals
            else None,
            "gold_score_gap_mean": sum(gold_score_gap_vals) / len(gold_score_gap_vals)
            if gold_score_gap_vals
            else None,
            "gold_in_search_topk_rate_hop1": method_report["metrics"].get(
                "gold_in_search_topk_rate_hop1"
            ),
            "gold_in_search_topk_rate_hop2": method_report["metrics"].get(
                "gold_in_search_topk_rate_hop2"
            ),
            "gold_in_search_topk_rate_union": method_report["metrics"].get(
                "gold_in_search_topk_rate_union"
            ),
            "winning_clause_rank_mean_hop1": method_report["metrics"].get(
                "winning_clause_rank_mean_hop1"
            ),
            "winning_clause_rank_mean_hop2": method_report["metrics"].get(
                "winning_clause_rank_mean_hop2"
            ),
            "winning_clause_rank_mean_union": method_report["metrics"].get(
                "winning_clause_rank_mean_union"
            ),
            "hit_at_open_budget_union": method_report["metrics"].get(
                "hit_at_open_budget_union"
            ),
            "rank_success_rate": method_report["metrics"].get("rank_success_rate"),
            "winning_in_union_rate": method_report["metrics"].get("winning_in_union_rate"),
            "policy_gain_over_rank": method_report["metrics"].get("policy_gain_over_rank"),
            "rank_gap": method_report["metrics"].get("rank_gap"),
            "opened_has_winning_clause_rate_union": method_report["metrics"].get(
                "opened_has_winning_clause_rate_union"
            ),
            "feasible_open_rate": method_report["metrics"].get("feasible_open_rate"),
            "realized_open_rate": method_report["metrics"].get("realized_open_rate"),
            "selection_gap": method_report["metrics"].get("selection_gap"),
            "selection_efficiency": method_report["metrics"].get("selection_efficiency"),
        }
        if method == "oracle":
            oracle_cov = [
                r.get("oracle_gold_coverage")
                for r in records
                if isinstance(r.get("oracle_gold_coverage"), (int, float))
            ]
            summary[method]["oracle_gold_coverage"] = (
                sum(oracle_cov) / len(oracle_cov) if oracle_cov else 0.0
            )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    git_sha = None
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_sha = None
    task_open_budgets = {t.budgets.get("open_budget", 5) for t in eval_tasks}
    open_budget = list(task_open_budgets)[0] if len(task_open_budgets) == 1 else None
    mix_rate = getattr(args, "bridged_mix_canonical_in_ticket_rate", None)
    if mix_rate is None:
        mix_rate = sum(1 for t in eval_tasks if not getattr(t, "bridge_clause_id", None)) / max(
            1, len(eval_tasks)
        )
    scenario_params = {
        "preset": getattr(args, "preset", None),
        "scenario_mode": getattr(args, "scenario_mode", "v0"),
        "bridge_bonus": getattr(args, "bridge_bonus", 1.5),
        "bridged_mix_canonical_in_ticket_rate": mix_rate,
        "open_budget": open_budget,
        "open_split_mode": getattr(args, "open_split_mode", "all_union_rank"),
        "open_split_hop1": getattr(args, "open_split_hop1", 0),
        "open_policy": getattr(args, "open_policy", "current"),
        "hop1_query_mode": getattr(args, "hop1_query_mode", "stripped"),
        "agent_query_policy": getattr(args, "agent_query_policy", "single_hop"),
        "thread_context_budget_chars": getattr(args, "thread_context_budget_chars", None),
        "thread_open_policy": getattr(args, "thread_open_policy", None),
        "thread_context_budget_sweep": getattr(args, "thread_context_budget_sweep", None),
        "e3_clause_jitter_max_chars": world.meta.get("e3_clause_jitter_max_chars"),
        "e3_clause_jitter_max_chars_critical": world.meta.get("e3_clause_jitter_max_chars_critical"),
        "e3_clause_jitter_max_chars_noncritical": world.meta.get("e3_clause_jitter_max_chars_noncritical"),
        "e3_clause_jitter_max_chars_decoy": world.meta.get("e3_clause_jitter_max_chars_decoy"),
        "e3_clause_jitter_scope": world.meta.get("e3_clause_jitter_scope"),
        "e3_litm_filler_count_min": world.meta.get("e3_litm_filler_count_min"),
        "e3_litm_filler_count_max": world.meta.get("e3_litm_filler_count_max"),
        "e3_litm_filler_len_jitter_max": world.meta.get("e3_litm_filler_len_jitter_max"),
    }
    scenario_params["exclusive_core_evidence"] = world.meta.get("exclusive_core_evidence")
    compare_report = {
        "methods": methods,
        "model": args.model,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "git_sha": git_sha,
        "invoked_cmdline": " ".join(sys.argv),
        "judge": getattr(args, "judge", "llm"),
        "scenario_params": scenario_params,
        "llm_backend": llm_backend,
        "client_class": client_class,
        "resolved_model": resolved_model,
        "gold_decision_distribution": gold_decision_distribution(eval_tasks),
        "summary": summary,
        "method_reports": method_reports,
        "controller_enabled": bool(controller),
        "controller_mode": controller_mode,
        "controller_policy": args.controller_policy,
        "task_split": args.task_split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "num_train_tasks": len(train_tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    if args.controller_policy == "rerank" and isinstance(controller, RerankController):
        compare_report["rerank_weights"] = dict(controller.weights)
        compare_report["feature_importance"] = sorted(
            controller.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    return compare_report


def _is_traceops_benchmark(args: argparse.Namespace) -> bool:
    return str(getattr(args, "benchmark", "policyops_v0") or "policyops_v0") == "traceops_v0"


def _parse_traceops_scenarios_arg(raw: Any) -> List[str]:
    parts = [str(x).strip() for x in str(raw or "mixed").split(",")]
    out = [p for p in parts if p]
    return out or ["mixed"]


def _cmd_generate_traceops(args: argparse.Namespace) -> None:
    from .traceops_v0.generator import generate_traceops_threads, save_traceops_dataset

    base_dir = Path(args.out_dir) if args.out_dir else _default_base_dir()
    scenarios = _parse_traceops_scenarios_arg(getattr(args, "traceops_scenarios", "mixed"))
    threads, meta = generate_traceops_threads(
        level=int(getattr(args, "traceops_level", 1) or 1),
        scenarios=scenarios,
        seed=int(getattr(args, "traceops_seed", getattr(args, "seed", 0)) or 0),
        threads=int(getattr(args, "traceops_threads", getattr(args, "n_tasks", 1)) or 1),
        trace_len=(
            int(getattr(args, "traceops_trace_len", 0) or 0)
            if int(getattr(args, "traceops_trace_len", 0) or 0) > 0
            else None
        ),
        delay_to_relevance=(
            int(getattr(args, "traceops_delay_to_relevance", 0) or 0)
            if int(getattr(args, "traceops_delay_to_relevance", 0) or 0) > 0
            else None
        ),
        distractor_branching=int(getattr(args, "traceops_distractor_branching", 2) or 2),
        contradiction_rate=float(getattr(args, "traceops_contradiction_rate", 0.35) or 0.35),
        exception_density=float(getattr(args, "traceops_exception_density", 0.35) or 0.35),
        state_flip_count=int(getattr(args, "traceops_state_flip_count", 1) or 1),
        indirection_rate=float(getattr(args, "traceops_indirection_rate", 0.4) or 0.4),
        trap_distractor_count=int(getattr(args, "traceops_trap_distractor_count", 4) or 4),
        trap_similarity_boost=float(getattr(args, "traceops_trap_similarity_boost", 0.7) or 0.7),
        core_size_min=int(getattr(args, "traceops_core_size_min", 2) or 2),
        core_size_max=int(getattr(args, "traceops_core_size_max", 4) or 4),
        alias_chain_len=int(getattr(args, "traceops_alias_chain_len", 2) or 2),
        indirect_pivot_style=str(getattr(args, "traceops_indirect_pivot_style", "blended") or "blended"),
        core_necessity_enable=bool(getattr(args, "traceops_core_necessity_enable", False)),
        core_necessity_require_all=bool(getattr(args, "traceops_core_necessity_require_all", True)),
        trap_decision_flip_enable=bool(getattr(args, "traceops_trap_decision_flip_enable", False)),
        trap_flip_salience=float(getattr(args, "traceops_trap_flip_salience", 0.25) or 0.25),
        trap_flip_attach_kind=str(
            getattr(args, "traceops_trap_flip_attach_kind", "avoided") or "avoided"
        ),
        trap_graph_excludable_rate=float(
            getattr(args, "traceops_trap_graph_excludable_rate", 0.7) or 0.7
        ),
        trap_graph_excludable_kinds=str(
            getattr(args, "traceops_trap_graph_excludable_kinds", "stale,inapplicable,avoided")
            or "stale,inapplicable,avoided"
        ),
        trap_invalidation_text_strength=float(
            getattr(args, "traceops_trap_invalidation_text_strength", 0.6) or 0.6
        ),
        hidden_core_enable=bool(getattr(args, "traceops_hidden_core_enable", False)),
        hidden_core_kind=str(getattr(args, "traceops_hidden_core_kind", "low_overlap_clause") or "low_overlap_clause"),
        hidden_core_link_mode=str(
            getattr(args, "traceops_hidden_core_link_mode", "depends_on") or "depends_on"
        ),
    )
    data_dir = save_traceops_dataset(base_dir, threads, meta)
    total_steps = sum(len(t.steps) for t in threads)
    print(
        f"Generated TraceOps v0 data. threads={len(threads)} steps={total_steps} data_dir={data_dir}",
        flush=True,
    )


def _cmd_eval_traceops(args: argparse.Namespace) -> None:
    from .traceops_v0.evaluator import evaluate_traceops_method
    from .traceops_v0.generator import load_traceops_dataset

    base_dir = Path(args.out_dir) if args.out_dir else _default_base_dir()
    threads, meta = load_traceops_dataset(base_dir)
    if getattr(args, "n_threads", None):
        threads = list(threads[: max(1, int(args.n_threads))])
    method = str(getattr(args, "method", "goc") or "goc")
    run_dir = base_dir / "runs" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    client = None
    if str(getattr(args, "traceops_eval_mode", "deterministic") or "deterministic") == "llm":
        if str(getattr(args, "llm", "openai") or "openai") != "openai":
            raise RuntimeError("traceops_eval_mode=llm requires --llm openai")
        client = OpenAIClient(
            model=getattr(args, "model", "gpt-4.1-mini"),
            dotenv_path=getattr(args, "dotenv", ".env"),
        )
    report = evaluate_traceops_method(method, threads, args=args, client=client)
    payload = {
        "benchmark": "traceops_v0",
        "method": method,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": timestamp,
        "model": getattr(args, "model", ""),
        "scenario_params": {
            **dict(meta),
            "traceops_eval_mode": str(getattr(args, "traceops_eval_mode", "deterministic") or "deterministic"),
            "traceops_llm_max_pivots": int(getattr(args, "traceops_llm_max_pivots", 0) or 0),
            "traceops_llm_eval_scope": str(getattr(args, "traceops_llm_eval_scope", "pivots") or "pivots"),
            "traceops_llm_sample_rate": float(getattr(args, "traceops_llm_sample_rate", 0.2) or 0.2),
            "traceops_indirection_rate": float(getattr(args, "traceops_indirection_rate", 0.4) or 0.4),
            "traceops_trap_distractor_count": int(getattr(args, "traceops_trap_distractor_count", 4) or 4),
            "traceops_trap_similarity_boost": float(getattr(args, "traceops_trap_similarity_boost", 0.7) or 0.7),
            "traceops_core_size_min": int(getattr(args, "traceops_core_size_min", 2) or 2),
            "traceops_core_size_max": int(getattr(args, "traceops_core_size_max", 4) or 4),
            "traceops_alias_chain_len": int(getattr(args, "traceops_alias_chain_len", 2) or 2),
            "traceops_indirect_pivot_style": str(
                getattr(args, "traceops_indirect_pivot_style", "blended") or "blended"
            ),
            "traceops_core_necessity_enable": bool(
                getattr(args, "traceops_core_necessity_enable", False)
            ),
            "traceops_core_necessity_require_all": bool(
                getattr(args, "traceops_core_necessity_require_all", True)
            ),
            "traceops_trap_decision_flip_enable": bool(
                getattr(args, "traceops_trap_decision_flip_enable", False)
            ),
            "traceops_trap_flip_salience": float(
                getattr(args, "traceops_trap_flip_salience", 0.25) or 0.25
            ),
            "traceops_trap_flip_attach_kind": str(
                getattr(args, "traceops_trap_flip_attach_kind", "avoided") or "avoided"
            ),
            "traceops_trap_graph_excludable_rate": float(
                getattr(args, "traceops_trap_graph_excludable_rate", 0.7) or 0.7
            ),
            "traceops_trap_graph_excludable_kinds": str(
                getattr(args, "traceops_trap_graph_excludable_kinds", "stale,inapplicable,avoided")
                or "stale,inapplicable,avoided"
            ),
            "traceops_trap_invalidation_text_strength": float(
                getattr(args, "traceops_trap_invalidation_text_strength", 0.6) or 0.6
            ),
            "traceops_hidden_core_enable": bool(getattr(args, "traceops_hidden_core_enable", False)),
            "traceops_hidden_core_kind": str(
                getattr(args, "traceops_hidden_core_kind", "low_overlap_clause") or "low_overlap_clause"
            ),
            "traceops_hidden_core_link_mode": str(
                getattr(args, "traceops_hidden_core_link_mode", "depends_on") or "depends_on"
            ),
            "goc_depwalk_enable": bool(getattr(args, "goc_depwalk_enable", False)),
            "goc_depwalk_hops": int(getattr(args, "goc_depwalk_hops", 2) or 2),
            "goc_depwalk_topk_per_hop": int(getattr(args, "goc_depwalk_topk_per_hop", 6) or 6),
        },
        "metrics": dict(report.get("metrics") or {}),
        "records": list(report.get("records") or []),
        "thread_records": list(report.get("thread_records") or []),
    }
    out_path = run_dir / f"{timestamp}.json"
    save_report(out_path, payload)
    print("TraceOps evaluation complete.")
    print(f"Report saved to {out_path}")


def _cmd_compare_traceops(args: argparse.Namespace) -> None:
    from .traceops_v0.evaluator import evaluate_traceops_method
    from .traceops_v0.generator import load_traceops_dataset

    base_dir = Path(args.out_dir) if args.out_dir else _default_base_dir()
    threads, meta = load_traceops_dataset(base_dir)
    if getattr(args, "n_threads", None):
        threads = list(threads[: max(1, int(args.n_threads))])

    methods_raw = getattr(args, "methods", None) or ["full", "similarity_only", "agent_fold", "goc"]
    methods: List[str] = []
    for item in list(methods_raw):
        text = str(item or "").strip()
        if not text:
            continue
        if "," in text:
            methods.extend([part.strip() for part in text.split(",") if part.strip()])
        else:
            methods.append(text)
    methods = methods or ["full", "similarity_only", "agent_fold", "goc"]
    client = None
    if str(getattr(args, "traceops_eval_mode", "deterministic") or "deterministic") == "llm":
        if str(getattr(args, "llm", "openai") or "openai") != "openai":
            raise RuntimeError("traceops_eval_mode=llm requires --llm openai")
        client = OpenAIClient(
            model=getattr(args, "model", "gpt-4.1-mini"),
            dotenv_path=getattr(args, "dotenv", ".env"),
        )
    compare_dir = base_dir / "runs" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    compare_run_dir = compare_dir / timestamp
    compare_run_dir.mkdir(parents=True, exist_ok=True)

    method_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for method in methods:
        method_dir = compare_run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        report = evaluate_traceops_method(method, threads, args=args, client=client)
        method_reports[method] = report
        metrics = dict(report.get("metrics") or {})
        summary[method] = {
            "pivot_decision_accuracy": metrics.get("pivot_decision_accuracy"),
            "pivot_e3_only_accuracy": metrics.get("pivot_e3_only_accuracy"),
            "strict_pivot_accuracy": metrics.get("strict_pivot_accuracy"),
            "pivots_available_total": metrics.get("pivots_available_total"),
            "pivots_evaluated": metrics.get("pivots_evaluated"),
            "steps_available_total": metrics.get("steps_available_total"),
            "traceops_llm_eval_scope": metrics.get("traceops_llm_eval_scope"),
            "sampled_step_rate": metrics.get("sampled_step_rate"),
            "sampled_steps_evaluated": metrics.get("sampled_steps_evaluated"),
            "tokens_pivot_mean": metrics.get("tokens_pivot_mean"),
            "tokens_total_mean": metrics.get("tokens_total_mean"),
            "tokens_pivot_mean_est": metrics.get("tokens_pivot_mean_est"),
            "tokens_total_mean_est": metrics.get("tokens_total_mean_est"),
            "tokens_pivot_mean_actual": metrics.get("tokens_pivot_mean_actual"),
            "tokens_total_mean_actual": metrics.get("tokens_total_mean_actual"),
            "mean_avoid_targets_per_pivot": metrics.get("mean_avoid_targets_per_pivot"),
            "avoided_injected_rate": metrics.get("avoided_injected_rate"),
            "revive_success_rate": metrics.get("revive_success_rate"),
            "mean_indirection_overlap_gold": metrics.get("mean_indirection_overlap_gold"),
            "mean_trap_gap": metrics.get("mean_trap_gap"),
            "trap_present_rate": metrics.get("trap_present_rate"),
            "mean_trap_injected_count": metrics.get("mean_trap_injected_count"),
            "mean_trap_injected_rate": metrics.get("mean_trap_injected_rate"),
            "trap_injected_any_rate": metrics.get("trap_injected_any_rate"),
            "mean_core_size": metrics.get("mean_core_size"),
            "decision_accuracy": metrics.get("decision_accuracy"),
            "judge_accuracy": metrics.get("judge_accuracy"),
        }

    compare_report = {
        "benchmark": "traceops_v0",
        "methods": methods,
        "model": getattr(args, "model", ""),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": timestamp,
        "invoked_cmdline": " ".join(sys.argv),
        "judge": getattr(args, "judge", "symbolic"),
        "scenario_params": {
            **dict(meta),
            "traceops_max_steps": int(getattr(args, "traceops_max_steps", 0) or 0),
            "traceops_eval_mode": str(getattr(args, "traceops_eval_mode", "deterministic") or "deterministic"),
            "traceops_llm_max_pivots": int(getattr(args, "traceops_llm_max_pivots", 0) or 0),
            "traceops_llm_eval_scope": str(getattr(args, "traceops_llm_eval_scope", "pivots") or "pivots"),
            "traceops_llm_sample_rate": float(getattr(args, "traceops_llm_sample_rate", 0.2) or 0.2),
            "traceops_indirection_rate": float(getattr(args, "traceops_indirection_rate", 0.4) or 0.4),
            "traceops_trap_distractor_count": int(getattr(args, "traceops_trap_distractor_count", 4) or 4),
            "traceops_trap_similarity_boost": float(getattr(args, "traceops_trap_similarity_boost", 0.7) or 0.7),
            "traceops_core_size_min": int(getattr(args, "traceops_core_size_min", 2) or 2),
            "traceops_core_size_max": int(getattr(args, "traceops_core_size_max", 4) or 4),
            "traceops_alias_chain_len": int(getattr(args, "traceops_alias_chain_len", 2) or 2),
            "traceops_indirect_pivot_style": str(
                getattr(args, "traceops_indirect_pivot_style", "blended") or "blended"
            ),
            "traceops_core_necessity_enable": bool(
                getattr(args, "traceops_core_necessity_enable", False)
            ),
            "traceops_core_necessity_require_all": bool(
                getattr(args, "traceops_core_necessity_require_all", True)
            ),
            "traceops_trap_decision_flip_enable": bool(
                getattr(args, "traceops_trap_decision_flip_enable", False)
            ),
            "traceops_trap_flip_salience": float(
                getattr(args, "traceops_trap_flip_salience", 0.25) or 0.25
            ),
            "traceops_trap_flip_attach_kind": str(
                getattr(args, "traceops_trap_flip_attach_kind", "avoided") or "avoided"
            ),
            "traceops_trap_graph_excludable_rate": float(
                getattr(args, "traceops_trap_graph_excludable_rate", 0.7) or 0.7
            ),
            "traceops_trap_graph_excludable_kinds": str(
                getattr(args, "traceops_trap_graph_excludable_kinds", "stale,inapplicable,avoided")
                or "stale,inapplicable,avoided"
            ),
            "traceops_trap_invalidation_text_strength": float(
                getattr(args, "traceops_trap_invalidation_text_strength", 0.6) or 0.6
            ),
            "traceops_hidden_core_enable": bool(getattr(args, "traceops_hidden_core_enable", False)),
            "traceops_hidden_core_kind": str(
                getattr(args, "traceops_hidden_core_kind", "low_overlap_clause") or "low_overlap_clause"
            ),
            "traceops_hidden_core_link_mode": str(
                getattr(args, "traceops_hidden_core_link_mode", "depends_on") or "depends_on"
            ),
            "goc_depwalk_enable": bool(getattr(args, "goc_depwalk_enable", False)),
            "goc_depwalk_hops": int(getattr(args, "goc_depwalk_hops", 2) or 2),
            "goc_depwalk_topk_per_hop": int(getattr(args, "goc_depwalk_topk_per_hop", 6) or 6),
        },
        "summary": summary,
        "method_reports": method_reports,
    }
    out_path = compare_dir / f"{timestamp}.json"
    save_report(out_path, compare_report)
    nested_out_path = compare_run_dir / f"{timestamp}.json"
    save_report(nested_out_path, compare_report)
    print("TraceOps compare complete.")
    print(f"Report saved to {out_path}")

def cmd_sweep(args: argparse.Namespace) -> None:
    _normalize_scenario_mode_arg(args)
    _apply_preset(args)
    _apply_generation_defaults(args)
    base_dir = args.out_dir or _default_base_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(base_dir) / "runs" / "sweeps" / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = sweep_dir / f"seed={seed}"
        data_ready = (seed_dir / "data" / "worlds" / "documents.jsonl").exists()
        if not args.reuse_data or not data_ready:
            generate_world_and_tasks(
                out_dir=seed_dir,
                seed=seed,
                n_docs=args.n_docs,
                n_tasks=args.n_tasks,
                scenario_mode=getattr(args, "scenario_mode", "v0"),
                bridge_prob=getattr(args, "bridge_prob", 0.8),
                alias_density=getattr(args, "alias_density", 0.9),
                canonical_density=getattr(args, "canonical_density", 0.95),
                bridge_kind=getattr(args, "bridge_kind", "definition"),
                bridged_mix_canonical_in_ticket_rate=getattr(
                    args, "bridged_mix_canonical_in_ticket_rate", 0.0
                ),
                exclusive_core_evidence=getattr(args, "exclusive_core_evidence", False),
                e3_clause_jitter_max_chars_critical=int(
                    getattr(args, "e3_clause_jitter_max_chars_critical", 0) or 0
                ),
                e3_clause_jitter_max_chars_noncritical=int(
                    getattr(args, "e3_clause_jitter_max_chars_noncritical", 0) or 0
                ),
                e3_clause_jitter_max_chars_decoy=int(
                    getattr(args, "e3_clause_jitter_max_chars_decoy", 0) or 0
                ),
                e3_litm_filler_count_min=int(
                    getattr(args, "e3_litm_filler_count_min", 0) or 0
                ),
                e3_litm_filler_count_max=int(
                    getattr(args, "e3_litm_filler_count_max", 0) or 0
                ),
                e3_litm_filler_len_jitter_max=int(
                    getattr(args, "e3_litm_filler_len_jitter_max", 0) or 0
                ),
                e3_clause_jitter_max_chars=int(getattr(args, "e3_clause_jitter_max_chars", 0) or 0),
                e3_clause_jitter_scope=str(getattr(args, "e3_clause_jitter_scope", "decoy_only") or "decoy_only"),
                preset_name=getattr(args, "preset", None),
            )

        world = load_world(seed_dir / "data" / "worlds")
        tasks = load_tasks(seed_dir / "data" / "tasks" / "tasks.jsonl")

        for open_budget in args.open_budgets:
            open_dir = seed_dir / f"open={open_budget}"
            open_dir.mkdir(parents=True, exist_ok=True)
            tasks_budget = _apply_open_budget(tasks, open_budget)

            compare_report = _run_compare_with_tasks(
                world,
                tasks_budget,
                args,
                base_dir=seed_dir,
                output_dir=open_dir,
            )
            compare_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            compare_path = open_dir / f"{compare_stamp}.json"
            save_report(compare_path, compare_report)

            for method, metrics in compare_report.get("summary", {}).items():
                row = {
                    "seed": seed,
                    "open_budget": open_budget,
                    "method": method,
                    "decision_accuracy": metrics.get("decision_accuracy"),
                    "judge_accuracy": metrics.get("judge_accuracy"),
                    "judge_accuracy_full_episode": metrics.get("judge_accuracy_full_episode"),
                    "acc_no_core_evidence_rate": metrics.get("acc_no_core_evidence_rate"),
                    "judge_used_any_core_rate": metrics.get("judge_used_any_core_rate"),
                    "judge_used_any_bridge_rate": metrics.get("judge_used_any_bridge_rate"),
                    "judge_used_any_critical_core_rate": metrics.get(
                        "judge_used_any_critical_core_rate"
                    ),
                    "judge_used_any_decoy_rate": metrics.get("judge_used_any_decoy_rate"),
                    "judge_supporting_count_mean": metrics.get("judge_supporting_count_mean"),
                    "full_episode_supporting_count_mean": metrics.get(
                        "full_episode_supporting_count_mean"
                    ),
                    "selection_upper_bound_judge_acc": metrics.get("selection_upper_bound_judge_acc"),
                    "deep_rank_core_rate": metrics.get("deep_rank_core_rate"),
                    "closure_recall_core_mean": metrics.get("closure_recall_core_mean"),
                    "avoided_node_injected_rate": metrics.get("avoided_node_injected_rate"),
                    "wrong_branch_recall_rate_mean": metrics.get(
                        "wrong_branch_recall_rate_mean"
                    ),
                    "min_gold_core_rank_union_mean": metrics.get("min_gold_core_rank_union_mean"),
                    "min_gold_core_rank_union_median": metrics.get("min_gold_core_rank_union_median"),
                    "min_gold_core_rank_union_p90": metrics.get("min_gold_core_rank_union_p90"),
                    "opened_bridge_count_mean": metrics.get("opened_bridge_count_mean"),
                    "opened_meta_count_mean": metrics.get("opened_meta_count_mean"),
                    "opened_rule_count_mean": metrics.get("opened_rule_count_mean"),
                    "condition_f1": metrics.get("condition_f1"),
                    "evidence_recall": metrics.get("evidence_recall"),
                    "critical_evidence_hit": metrics.get("critical_evidence_hit"),
                    "avg_open_calls": metrics.get("open_calls", {}).get("avg"),
                    "p90_open_calls": metrics.get("open_calls", {}).get("p90"),
                    "avg_prompt_tokens": metrics.get("prompt_tokens", {}).get("avg"),
                    "p90_prompt_tokens": metrics.get("prompt_tokens", {}).get("p90"),
                    "oracle_gold_coverage": metrics.get("oracle_gold_coverage"),
                    "opened_gold_coverage_mean": metrics.get("opened_gold_coverage_mean"),
                    "gold_in_search_topk_rate": metrics.get("gold_in_search_topk_rate"),
                    "winning_clause_rank_mean": metrics.get("winning_clause_rank_mean"),
                    "min_gold_rank_mean": metrics.get("min_gold_rank_mean"),
                    "gold_score_gap_mean": metrics.get("gold_score_gap_mean"),
                    "gold_in_search_topk_rate_hop1": metrics.get("gold_in_search_topk_rate_hop1"),
                    "gold_in_search_topk_rate_hop2": metrics.get("gold_in_search_topk_rate_hop2"),
                    "gold_in_search_topk_rate_union": metrics.get("gold_in_search_topk_rate_union"),
                    "winning_clause_rank_mean_hop1": metrics.get("winning_clause_rank_mean_hop1"),
                    "winning_clause_rank_mean_hop2": metrics.get("winning_clause_rank_mean_hop2"),
                    "winning_clause_rank_mean_union": metrics.get("winning_clause_rank_mean_union"),
                    "hit_at_open_budget_union": metrics.get("hit_at_open_budget_union"),
                    "rank_success_rate": metrics.get("rank_success_rate"),
                    "winning_in_union_rate": metrics.get("winning_in_union_rate"),
                    "policy_gain_over_rank": metrics.get("policy_gain_over_rank"),
                    "rank_gap": metrics.get("rank_gap"),
                    "opened_has_winning_clause_rate_union": metrics.get(
                        "opened_has_winning_clause_rate_union"
                    ),
                    "feasible_open_rate": metrics.get("feasible_open_rate"),
                    "realized_open_rate": metrics.get("realized_open_rate"),
                    "selection_gap": metrics.get("selection_gap"),
                    "selection_efficiency": metrics.get("selection_efficiency"),
                }
                summary_rows.append(row)

    summary_json_path = sweep_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    summary_csv_path = sweep_dir / "summary.csv"
    fieldnames = [
        "seed",
        "open_budget",
        "method",
        "decision_accuracy",
        "judge_accuracy",
        "judge_accuracy_full_episode",
        "acc_no_core_evidence_rate",
        "judge_used_any_core_rate",
        "judge_used_any_bridge_rate",
        "judge_used_any_critical_core_rate",
        "judge_used_any_decoy_rate",
        "judge_supporting_count_mean",
        "full_episode_supporting_count_mean",
        "selection_upper_bound_judge_acc",
        "deep_rank_core_rate",
        "closure_recall_core_mean",
        "avoided_node_injected_rate",
        "wrong_branch_recall_rate_mean",
        "min_gold_core_rank_union_mean",
        "min_gold_core_rank_union_median",
        "min_gold_core_rank_union_p90",
        "opened_bridge_count_mean",
        "opened_meta_count_mean",
        "opened_rule_count_mean",
        "condition_f1",
        "evidence_recall",
        "critical_evidence_hit",
        "avg_open_calls",
        "p90_open_calls",
        "avg_prompt_tokens",
        "p90_prompt_tokens",
        "oracle_gold_coverage",
        "opened_gold_coverage_mean",
        "gold_in_search_topk_rate",
        "winning_clause_rank_mean",
        "min_gold_rank_mean",
        "gold_score_gap_mean",
        "gold_in_search_topk_rate_hop1",
        "gold_in_search_topk_rate_hop2",
        "gold_in_search_topk_rate_union",
        "winning_clause_rank_mean_hop1",
        "winning_clause_rank_mean_hop2",
        "winning_clause_rank_mean_union",
        "hit_at_open_budget_union",
        "rank_success_rate",
        "winning_in_union_rate",
        "policy_gain_over_rank",
        "rank_gap",
        "opened_has_winning_clause_rate_union",
        "feasible_open_rate",
        "realized_open_rate",
        "selection_gap",
        "selection_efficiency",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    quickcheck_sweep_summary(summary_rows, judge_mode=getattr(args, "judge", "llm"), label=timestamp)

    print("Sweep complete.")
    print(f"Summary saved to {summary_csv_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    mode = getattr(args, "mode", "failure_slice")
    report_path = Path(args.report) if getattr(args, "report", "") else None
    if mode in {"failure_slice", "bridged_ab", "selection_triage", "slot_breakdown"} and not report_path:
        raise ValueError("--report is required for this analyze mode")
    if mode == "bridged_ab":
        output_path = analyze_bridged_ab(report_path, method=args.method)
        print(f"Bridged A×B report saved to {output_path}")
    elif mode == "selection_triage":
        csv_path, md_path = analyze_selection_triage(
            report_path,
            method=args.method,
            max_per_bucket=int(getattr(args, "max_per_bucket", 20)),
        )
        print(f"Selection triage CSV saved to {csv_path}")
        print(f"Pattern summary saved to {md_path}")
    elif mode == "slot_breakdown":
        csv_path, md_path = analyze_slot_breakdown(report_path, method=args.method)
        print(f"Slot breakdown CSV saved to {csv_path}")
        print(f"Slot breakdown MD saved to {md_path}")
    elif mode == "analysis_bundle":
        run_dir = Path(getattr(args, "run_dir", ""))
        if not run_dir:
            raise ValueError("--run_dir is required for analysis_bundle mode")
        outputs = analyze_bundle(run_dir)
        print(f"Analysis bundle saved to {outputs['analysis_bundle_zip']}")
        print(f"Share bundle saved to {outputs['share_bundle_zip']}")
    elif mode == "split_sweep_ab":
        sweep_dir = Path(getattr(args, "sweep_dir", "")) if getattr(args, "sweep_dir", "") else None
        if not sweep_dir:
            raise ValueError("--sweep_dir is required for split_sweep_ab mode")
        out_path = analyze_split_sweep_ab(
            sweep_dir,
            results_md=Path(getattr(args, "results_md", ""))
            if getattr(args, "results_md", "")
            else None,
            method=args.method,
        )
        print(f"Split sweep A3×B2 summary saved to {out_path}")
    else:
        output_path = analyze_failure_slice(report_path, top_k=args.k)
        print(f"Failure slice report saved to {output_path}")


def cmd_ablate_controller(args: argparse.Namespace) -> None:
    base_dir = args.out_dir or _default_base_dir()
    world_dir = Path(base_dir) / "data" / "worlds"
    tasks_path = Path(base_dir) / "data" / "tasks" / "tasks.jsonl"
    world = load_world(world_dir)
    tasks = load_tasks(tasks_path)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ablate_dir = Path(base_dir) / "runs" / "ablate" / timestamp
    ablate_dir.mkdir(parents=True, exist_ok=True)

    off_args = argparse.Namespace(**vars(args))
    off_args.use_controller = False
    off_args.controller_mode = "off"

    on_args = argparse.Namespace(**vars(args))
    on_args.use_controller = True
    on_args.controller_policy = "rerank"
    on_args.controller_mode = args.controller_mode

    report_off = _run_compare_with_tasks(
        world,
        tasks,
        off_args,
        base_dir=Path(base_dir),
        output_dir=ablate_dir / "off",
    )
    report_on = _run_compare_with_tasks(
        world,
        tasks,
        on_args,
        base_dir=Path(base_dir),
        output_dir=ablate_dir / "on",
    )

    def _get_metric(report: Dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(report.get("summary", {}).get("goc", {}).get(key, default) or 0.0)

    def _get_prompt_avg(report: Dict[str, Any]) -> float:
        prompt = report.get("summary", {}).get("goc", {}).get("prompt_tokens", {})
        return float(prompt.get("avg") or 0.0) if isinstance(prompt, dict) else 0.0

    delta = {
        "decision_accuracy": _get_metric(report_on, "decision_accuracy")
        - _get_metric(report_off, "decision_accuracy"),
        "critical_evidence_hit": _get_metric(report_on, "critical_evidence_hit")
        - _get_metric(report_off, "critical_evidence_hit"),
        "opened_gold_coverage_mean": _get_metric(report_on, "opened_gold_coverage_mean")
        - _get_metric(report_off, "opened_gold_coverage_mean"),
        "gold_in_search_topk_rate": _get_metric(report_on, "gold_in_search_topk_rate")
        - _get_metric(report_off, "gold_in_search_topk_rate"),
        "prompt_tokens_avg": _get_prompt_avg(report_on) - _get_prompt_avg(report_off),
    }

    out_payload = {
        "timestamp": timestamp,
        "controller_policy": "rerank",
        "off_report": report_off,
        "on_report": report_on,
        "delta": delta,
    }
    out_path = ablate_dir / "ablate_summary.json"
    save_report(out_path, out_payload)
    print("Ablation complete.")
    for key, value in delta.items():
        print(f"{key}: {value:.4f}")
    print(f"Ablation report saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolicyOps Arena v0 runner")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic world and tasks")
    gen.add_argument(
        "--benchmark",
        choices=["policyops_v0", "traceops_v0"],
        default="policyops_v0",
    )
    gen.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default=None)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--n_docs", type=int, default=None)
    gen.add_argument("--clauses_per_doc", type=int, default=None)
    gen.add_argument("--n_tasks", type=int, default=200)
    gen.add_argument("--exception_chain_depth", type=int, default=2)
    gen.add_argument(
        "--definition_dependency_depth",
        type=int,
        default=1,
        help="Max hop depth for recursive definition evidence used by symbolic judge (default=1)",
    )
    gen.add_argument(
        "--definition_dependency_extra_terms",
        type=int,
        default=0,
        help="Extra dependency terms added to each definition clause (creates multi-hop definition chains)",
    )
    gen.add_argument(
        "--force_exception_chain_depth",
        type=int,
        default=0,
        help="If >0, overrides exception_chain_depth to force a longer exception override chain",
    )
    gen.add_argument(
        "--force_exception_chain_all_apply",
        action="store_true",
        help="Force all exception-chain clauses to share the base rule's applies_if so full chain applies",
    )
    gen.add_argument("--update_rate", type=float, default=0.3)
    gen.add_argument("--definition_density", type=float, default=0.4)
    gen.add_argument("--distractor_strength", type=float, default=0.3)
    gen.add_argument(
        "--scenario_mode",
        choices=[
            "v0",
            "bridged_v1_1",
            "threaded_v1_2",
            "threaded_v1_3_fu",
            "threaded_v1_3_fu_decoy",
            THREADED_FU_DECOY_DEPTHJITTER_MODE,
        ],
        default="v0",
    )
    gen.add_argument("--bridge_prob", type=float, default=0.8)
    gen.add_argument("--bridged_mix_canonical_in_ticket_rate", type=float, default=0.0)
    gen.add_argument("--alias_density", type=float, default=None)
    gen.add_argument("--canonical_density", type=float, default=None)
    gen.add_argument("--bridge_kind", choices=["definition", "glossary"], default="definition")
    gen.add_argument("--exclusive_core_evidence", action="store_true", default=False)
    gen.add_argument("--n_threads", type=int, default=None)
    gen.add_argument("--open_budget_e1", type=int, default=4)
    gen.add_argument("--open_budget_e2", type=int, default=4)
    gen.add_argument("--open_budget_e3", type=int, default=0)
    gen.add_argument("--tool_budget_e1", type=int, default=50)
    gen.add_argument("--tool_budget_e2", type=int, default=50)
    gen.add_argument("--tool_budget_e3", type=int, default=0)
    gen.add_argument("--branch_distractor_rate", type=float, default=0.5)
    gen.add_argument("--e3_clause_jitter_max_chars", type=int, default=0)
    gen.add_argument("--e3_clause_jitter_max_chars_critical", type=int, default=0)
    gen.add_argument("--e3_clause_jitter_max_chars_noncritical", type=int, default=0)
    gen.add_argument("--e3_clause_jitter_max_chars_decoy", type=int, default=0)
    gen.add_argument("--e3_litm_filler_count_min", type=int, default=0)
    gen.add_argument("--e3_litm_filler_count_max", type=int, default=0)
    gen.add_argument("--e3_litm_filler_len_jitter_max", type=int, default=0)
    gen.add_argument(
        "--e3_clause_jitter_scope",
        choices=["decoy_only", "decoy_plus_noncritical", "all"],
        default="decoy_only",
    )
    gen.add_argument("--pivot_rate", type=float, default=0.0)
    gen.add_argument(
        "--pivot_type",
        choices=[
            "retention_flip",
            "entity_switch",
            "constraint_add",
            # Backward-compatible aliases
            "action_flip",
            "exception_add",
        ],
        default="retention_flip",
    )
    gen.add_argument("--out_dir", type=Path, default=None)
    gen.add_argument("--traceops_level", type=int, default=1)
    gen.add_argument("--traceops_scenarios", type=str, default="mixed")
    gen.add_argument("--traceops_seed", type=int, default=0)
    gen.add_argument("--traceops_threads", type=int, default=32)
    gen.add_argument("--traceops_trace_len", type=int, default=0)
    gen.add_argument("--traceops_delay_to_relevance", type=int, default=0)
    gen.add_argument("--traceops_distractor_branching", type=int, default=2)
    gen.add_argument("--traceops_contradiction_rate", type=float, default=0.35)
    gen.add_argument("--traceops_exception_density", type=float, default=0.35)
    gen.add_argument("--traceops_state_flip_count", type=int, default=1)
    gen.add_argument("--traceops_indirection_rate", type=float, default=0.4)
    gen.add_argument("--traceops_trap_distractor_count", type=int, default=4)
    gen.add_argument("--traceops_trap_similarity_boost", type=float, default=0.7)
    gen.add_argument("--traceops_core_size_min", type=int, default=2)
    gen.add_argument("--traceops_core_size_max", type=int, default=4)
    gen.add_argument("--traceops_alias_chain_len", type=int, default=2)
    gen.add_argument(
        "--traceops_indirect_pivot_style",
        choices=["ordinal_ref", "alias_handle", "blended"],
        default="blended",
    )
    gen.add_argument("--traceops_core_necessity_enable", action="store_true", default=False)
    gen.add_argument("--traceops_core_necessity_require_all", action="store_true", default=True)
    gen.add_argument("--no_traceops_core_necessity_require_all", action="store_false", dest="traceops_core_necessity_require_all")
    gen.add_argument("--traceops_trap_decision_flip_enable", action="store_true", default=False)
    gen.add_argument("--traceops_trap_flip_salience", type=float, default=0.25)
    gen.add_argument(
        "--traceops_trap_flip_attach_kind",
        choices=["stale", "inapplicable", "avoided", "random", "none"],
        default="avoided",
    )
    gen.add_argument("--traceops_trap_graph_excludable_rate", type=float, default=0.7)
    gen.add_argument(
        "--traceops_trap_graph_excludable_kinds",
        type=str,
        default="stale,inapplicable,avoided",
    )
    gen.add_argument("--traceops_trap_invalidation_text_strength", type=float, default=0.6)
    gen.add_argument("--traceops_hidden_core_enable", action="store_true", default=False)
    gen.add_argument(
        "--traceops_hidden_core_kind",
        choices=["low_overlap_clause", "alias_only_update"],
        default="low_overlap_clause",
    )
    gen.add_argument(
        "--traceops_hidden_core_link_mode",
        choices=["depends_on", "none"],
        default="depends_on",
    )
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="Evaluate baselines")
    ev.add_argument(
        "--benchmark",
        choices=["policyops_v0", "traceops_v0"],
        default="policyops_v0",
    )
    ev.add_argument(
        "--method",
        choices=["topk", "full", "full_history", "goc", "goc_base", "oracle", "engine", "similarity_only", "agent_fold"],
        required=True,
    )
    ev.add_argument("--model", type=str, default="gpt-4.1-mini")
    ev.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    ev.add_argument(
        "--judge",
        choices=[
            "llm",
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        ],
        default="llm",
    )
    ev.add_argument("--dotenv", type=str, default=".env")
    ev.add_argument("--out_dir", type=Path, default=None)
    ev.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    ev.add_argument("--min_evidence_count", type=int, default=2)
    ev.add_argument("--save_prompts", action="store_true", help="Save per-task prompts to files")
    ev.add_argument("--save_raw", action="store_true", help="Save per-task raw outputs to files")
    ev.add_argument("--save_search_snapshot", action="store_true")
    ev.add_argument("--search_snapshot_k", type=int, default=20)
    ev.add_argument("--primary_search_top_k", type=int, default=20)
    ev.add_argument("--save_goc_graph", action="store_true")
    ev.add_argument("--save_goc_dot", action="store_true")
    ev.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    ev.add_argument("--goc_graph_include_clause_text", action="store_true")
    ev.add_argument("--goc_graph_dir", type=str, default="")
    ev.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    ev.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    ev.add_argument(
        "--scenario_mode",
        choices=[
            "v0",
            "bridged_v1_1",
            "threaded_v1_2",
            "threaded_v1_3_fu",
            "threaded_v1_3_fu_decoy",
            THREADED_FU_DECOY_DEPTHJITTER_MODE,
        ],
        default="v0",
    )
    ev.add_argument(
        "--pivot_message_style",
        choices=["banner", "transcript"],
        default="transcript",
        help="How pivot updates are rendered for episode-3 tasks.",
    )
    ev.add_argument(
        "--pivot_gold_mode",
        choices=["original", "respect_ticket_updated", "both"],
        default="respect_ticket_updated",
        help=(
            "Gold evaluation mode for episode-3 pivot tasks: "
            "original=task.context/task.gold, "
            "respect_ticket_updated=effective_context-derived pivot gold, "
            "both=compute both metrics but final correctness uses pivot gold."
        ),
    )
    ev.add_argument("--goc_enable_avoids", action="store_true", default=True)
    ev.add_argument("--no_goc_enable_avoids", action="store_false", dest="goc_enable_avoids")
    ev.add_argument(
        "--goc_avoids_mode",
        choices=["legacy_commit", "applicability", "off"],
        default=None,
        help=(
            "GoC E3 avoid filtering mode: "
            "legacy_commit=commit-anchor based, applicability=applies_if mismatch based, off=disabled. "
            "When unset, default behavior is applicability via --goc_enable_avoids alias."
        ),
    )
    ev.add_argument("--thread_context_budget_chars", type=int, default=8000)
    ev.add_argument("--thread_open_policy", choices=["current", "shared_topk"], default="current")
    ev.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    ev.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    ev.add_argument("--bridge_bonus", type=float, default=1.5)
    ev.add_argument("--hop1_query_mode", choices=["raw", "stripped", "llm_keywords"], default="stripped")
    ev.add_argument("--open_split_mode", choices=["all_union_rank", "split_hop1_hop2"], default="all_union_rank")
    ev.add_argument("--open_split_hop1", type=int, default=0)
    ev.add_argument(
        "--open_policy",
        choices=[
            "current",
            "oracle_open_if_in_union",
            "soft_core_rerank",
            "hop2_priority",
            "bridge_one_only",
            "core_first_heuristic",
        ],
        default="current",
    )
    ev.add_argument("--bridge_reward_bonus", type=float, default=0.0)
    ev.add_argument("--use_query_rewrite", action="store_true", default=True)
    ev.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    ev.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    ev.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    ev.add_argument("--rewrite_queries", type=int, default=3)
    ev.add_argument("--force_open_top_n", type=int, default=1)
    ev.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    ev.add_argument("--use_controller", action="store_true")
    ev.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    ev.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    ev.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    ev.add_argument("--controller_weights_path", type=str, default="")
    ev.add_argument("--task_split", choices=["none", "holdout"], default="none")
    ev.add_argument("--train_ratio", type=float, default=0.7)
    ev.add_argument("--split_seed", type=int, default=0)
    ev.add_argument("--traceops_max_steps", type=int, default=0)
    ev.add_argument(
        "--traceops_eval_mode",
        choices=["deterministic", "llm"],
        default="deterministic",
    )
    ev.add_argument("--traceops_llm_temperature", type=float, default=0.0)
    ev.add_argument(
        "--traceops_llm_max_pivots",
        type=int,
        default=0,
        help="Cap number of pivot_check steps evaluated in llm mode (0 = no cap).",
    )
    ev.add_argument("--traceops_llm_cache_dir", type=str, default=".cache/traceops_llm")
    ev.add_argument("--traceops_llm_seed", type=int, default=0)
    ev.add_argument("--traceops_llm_max_output_tokens", type=int, default=256)
    ev.add_argument(
        "--traceops_llm_eval_scope",
        choices=["pivots", "all", "sample"],
        default="pivots",
        help="LLM evaluation scope for TraceOps: pivots only, all steps, or deterministic sampled steps.",
    )
    ev.add_argument(
        "--traceops_llm_sample_rate",
        type=float,
        default=0.2,
        help="Sampling rate for TraceOps llm scope=sample (0..1).",
    )
    ev.add_argument(
        "--goc_depwalk_enable",
        action="store_true",
        default=False,
        help="Enable Phase16 dependency-walk expansion in TraceOps GoC context selection.",
    )
    ev.add_argument("--goc_depwalk_hops", type=int, default=2)
    ev.add_argument("--goc_depwalk_topk_per_hop", type=int, default=6)
    ev.add_argument("--traceops_indirection_rate", type=float, default=0.4)
    ev.add_argument("--traceops_trap_distractor_count", type=int, default=4)
    ev.add_argument("--traceops_trap_similarity_boost", type=float, default=0.7)
    ev.add_argument("--traceops_core_size_min", type=int, default=2)
    ev.add_argument("--traceops_core_size_max", type=int, default=4)
    ev.add_argument("--traceops_alias_chain_len", type=int, default=2)
    ev.add_argument(
        "--traceops_indirect_pivot_style",
        choices=["ordinal_ref", "alias_handle", "blended"],
        default="blended",
    )
    ev.add_argument("--traceops_core_necessity_enable", action="store_true", default=False)
    ev.add_argument("--traceops_core_necessity_require_all", action="store_true", default=True)
    ev.add_argument("--no_traceops_core_necessity_require_all", action="store_false", dest="traceops_core_necessity_require_all")
    ev.add_argument("--traceops_trap_decision_flip_enable", action="store_true", default=False)
    ev.add_argument("--traceops_trap_flip_salience", type=float, default=0.25)
    ev.add_argument(
        "--traceops_trap_flip_attach_kind",
        choices=["stale", "inapplicable", "avoided", "random", "none"],
        default="avoided",
    )
    ev.add_argument("--traceops_trap_graph_excludable_rate", type=float, default=0.7)
    ev.add_argument(
        "--traceops_trap_graph_excludable_kinds",
        type=str,
        default="stale,inapplicable,avoided",
    )
    ev.add_argument("--traceops_trap_invalidation_text_strength", type=float, default=0.6)
    ev.add_argument("--traceops_hidden_core_enable", action="store_true", default=False)
    ev.add_argument(
        "--traceops_hidden_core_kind",
        choices=["low_overlap_clause", "alias_only_update"],
        default="low_overlap_clause",
    )
    ev.add_argument(
        "--traceops_hidden_core_link_mode",
        choices=["depends_on", "none"],
        default="depends_on",
    )
    ev.add_argument(
        "--traceops_force_include_required",
        action="store_true",
        default=False,
        help="TraceOps debug-only oracle mode: force include required/pivot evidence IDs in GoC context.",
    )
    ev.set_defaults(func=cmd_eval)

    cmp = sub.add_parser("compare", help="Compare methods in one run")
    cmp.add_argument(
        "--benchmark",
        choices=["policyops_v0", "traceops_v0"],
        default="policyops_v0",
    )
    cmp.add_argument("--methods", nargs="+", default=["topk", "full", "goc", "oracle", "engine"])
    cmp.add_argument("--model", type=str, default="gpt-4.1-mini")
    cmp.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    cmp.add_argument(
        "--judge",
        choices=[
            "llm",
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        ],
        default="llm",
    )
    cmp.add_argument("--dotenv", type=str, default=".env")
    cmp.add_argument("--out_dir", type=Path, default=None)
    cmp.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default=None)
    cmp.add_argument("--seed", type=int, default=0)
    cmp.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    cmp.add_argument("--min_evidence_count", type=int, default=2)
    cmp.add_argument("--save_prompts", action="store_true", help="Save per-task prompts to files")
    cmp.add_argument("--save_raw", action="store_true", help="Save per-task raw outputs to files")
    cmp.add_argument("--save_event_trace", action="store_true", help="Save per-task JSONL event traces")
    cmp.add_argument(
        "--event_trace_dir",
        type=str,
        default="",
        help="Per-task event trace directory (default: <run_dir>/event_traces)",
    )
    cmp.add_argument("--event_trace_sample_rate", type=float, default=1.0)
    cmp.add_argument("--save_search_snapshot", action="store_true")
    cmp.add_argument("--search_snapshot_k", type=int, default=20)
    cmp.add_argument("--primary_search_top_k", type=int, default=20)
    cmp.add_argument("--save_goc_graph", action="store_true")
    cmp.add_argument("--save_goc_dot", action="store_true")
    cmp.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    cmp.add_argument("--goc_graph_include_clause_text", action="store_true")
    cmp.add_argument("--goc_graph_dir", type=str, default="")
    cmp.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    cmp.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    cmp.add_argument("--goc_internal_budget_active_tokens", type=int, default=1200)
    cmp.add_argument("--goc_internal_budget_unfold_tokens", type=int, default=650)
    cmp.add_argument("--goc_internal_unfold_k", type=int, default=8)
    cmp.add_argument("--goc_budget_follow_sweep", action="store_true")
    cmp.add_argument("--goc_budget_chars_to_tokens_divisor", type=int, default=4)
    cmp.add_argument(
        "--goc_unfold_max_nodes",
        type=int,
        default=0,
        help="Max unfolded nodes in final active context (0 keeps legacy selection cap)",
    )
    cmp.add_argument(
        "--goc_unfold_hops",
        type=int,
        default=0,
        help="Dependency-closure hop depth from selected anchors (0 disables hop expansion)",
    )
    cmp.add_argument(
        "--goc_unfold_budget_mode",
        choices=["nodes_only", "hops_only", "nodes_and_hops"],
        default="nodes_and_hops",
    )

    cmp.add_argument(
        "--goc_unfold_policy",
        choices=["fixed", "adaptive_pivot", "adaptive_heuristic"],
        default="fixed",
        help="GoC unfold control policy. 'fixed' uses (--goc_unfold_max_nodes, --goc_unfold_hops). "
             "'adaptive_pivot' switches between default/pivot knobs when a ticket update is present.",
    )
    cmp.add_argument(
        "--goc_unfold_default_max_nodes",
        type=int,
        default=0,
        help="(Adaptive) default max nodes for non-pivot tasks (0 falls back to --goc_unfold_max_nodes).",
    )
    cmp.add_argument(
        "--goc_unfold_default_hops",
        type=int,
        default=0,
        help="(Adaptive) default hop depth for non-pivot tasks (0 falls back to --goc_unfold_hops).",
    )
    cmp.add_argument(
        "--goc_unfold_pivot_max_nodes",
        type=int,
        default=0,
        help="(Adaptive) max nodes for pivot/update tasks (0 falls back to --goc_unfold_max_nodes).",
    )
    cmp.add_argument(
        "--goc_unfold_pivot_hops",
        type=int,
        default=0,
        help="(Adaptive) hop depth for pivot/update tasks (0 falls back to --goc_unfold_hops).",
    )
    cmp.add_argument(
        "--goc_candidate_pool_size",
        type=int,
        default=0,
        help="If >0, expands GoC unfold candidate pool with top-N clause search results (improves K/H meaning)",
    )
    cmp.add_argument(
        "--goc_closedbook_universe",
        choices=["memory", "world"],
        default="memory",
        help="E3 closed-book final: restrict closure universe to observed memory (memory) vs allow full world (world)",
    )
    cmp.add_argument("--goc_graph_frontier", action="store_true", default=True)
    cmp.add_argument("--no_goc_graph_frontier", action="store_false", dest="goc_graph_frontier")
    cmp.add_argument("--goc_graph_frontier_hops", type=int, default=2)
    cmp.add_argument("--goc_graph_frontier_max_nodes", type=int, default=50)
    cmp.add_argument("--goc_graph_frontier_seed_top_n", type=int, default=6)
    cmp.add_argument("--goc_graph_frontier_score_frac", type=float, default=0.7)
    cmp.add_argument("--goc_activity_filter", action="store_true", default=False)
    cmp.add_argument("--goc_activity_filter_fallback", action="store_true", default=True)
    cmp.add_argument(
        "--no_goc_activity_filter_fallback",
        action="store_false",
        dest="goc_activity_filter_fallback",
    )
    cmp.add_argument("--goc_mmr_lambda", type=float, default=0.35)
    cmp.add_argument("--goc_anchor_top1_lexical", action="store_true", default=False)
    cmp.add_argument(
        "--no_goc_anchor_top1_lexical",
        action="store_false",
        dest="goc_anchor_top1_lexical",
    )
    cmp.add_argument("--goc_activity_debug_in_snapshot", action="store_true", default=False)
    cmp.add_argument(
        "--scenario_mode",
        choices=[
            "v0",
            "bridged_v1_1",
            "threaded_v1_2",
            "threaded_v1_3_fu",
            "threaded_v1_3_fu_decoy",
            THREADED_FU_DECOY_DEPTHJITTER_MODE,
        ],
        default="v0",
    )
    cmp.add_argument(
        "--pivot_message_style",
        choices=["banner", "transcript"],
        default="transcript",
        help="How pivot updates are rendered for episode-3 tasks.",
    )
    cmp.add_argument(
        "--pivot_gold_mode",
        choices=["original", "respect_ticket_updated", "both"],
        default="respect_ticket_updated",
        help=(
            "Gold evaluation mode for episode-3 pivot tasks: "
            "original=task.context/task.gold, "
            "respect_ticket_updated=effective_context-derived pivot gold, "
            "both=compute both metrics but final correctness uses pivot gold."
        ),
    )
    cmp.add_argument("--goc_enable_avoids", action="store_true", default=True)
    cmp.add_argument("--no_goc_enable_avoids", action="store_false", dest="goc_enable_avoids")
    cmp.add_argument(
        "--goc_avoids_mode",
        choices=["legacy_commit", "applicability", "off"],
        default=None,
        help=(
            "GoC E3 avoid filtering mode: "
            "legacy_commit=commit-anchor based, applicability=applies_if mismatch based, off=disabled. "
            "When unset, default behavior is applicability via --goc_enable_avoids alias."
        ),
    )
    cmp.add_argument(
        "--goc_applicability_seed_enable",
        action="store_true",
        default=False,
        help="Enable Phase14 applicability seeding for E3 pivot GoC selection.",
    )
    cmp.add_argument(
        "--goc_applicability_seed_topk",
        type=int,
        default=8,
        help="Top-K applicable clauses used as pre-seeds for E3 GoC selection.",
    )
    cmp.add_argument(
        "--goc_dependency_closure_enable",
        action="store_true",
        default=False,
        help="Enable budgeted dependency closure add-on after GoC E3 clause selection.",
    )
    cmp.add_argument(
        "--goc_dependency_closure_topk",
        type=int,
        default=12,
        help="Maximum number of closure-added clause IDs in E3 GoC selection.",
    )
    cmp.add_argument(
        "--goc_dependency_closure_hops",
        type=int,
        default=1,
        help="Hop depth for dependency closure add-on in E3 GoC selection.",
    )
    cmp.add_argument(
        "--goc_dependency_closure_universe",
        choices=["candidates", "world", "memory_opened"],
        default="candidates",
        help="Universe used for dependency closure expansion in E3 GoC selection.",
    )
    cmp.add_argument("--thread_context_budget_chars", type=int, default=8000)
    cmp.add_argument("--thread_open_policy", choices=["current", "shared_topk"], default="current")
    cmp.add_argument("--thread_context_budget_sweep", type=str, default="")
    # Phase 12: commit anchor selection for threaded mode (episodes 1/2).
    # These anchors define the "allowed evidence" for episode-3 final answering in threaded settings.
    cmp.add_argument(
        "--commit_anchor_source",
        choices=["opened_supporting", "model_evidence", "hybrid"],
        default="opened_supporting",
        help=(
            "Commit anchor source for threaded mode: "
            "opened_supporting=heuristic supporting clauses from opened evidence; "
            "model_evidence=LLM output 'evidence' field; "
            "hybrid=union(model_evidence, opened_supporting)."
        ),
    )
    cmp.add_argument(
        "--commit_anchor_max_ids",
        type=int,
        default=4,
        help="Max anchor clause_ids per commit episode (0 disables cap).",
    )
    cmp.add_argument(
        "--commit_anchor_require_opened",
        action="store_true",
        default=True,
        help="When using model_evidence/hybrid, keep only evidence IDs that were opened in that episode.",
    )
    cmp.add_argument(
        "--no_commit_anchor_require_opened",
        action="store_false",
        dest="commit_anchor_require_opened",
    )
    cmp.add_argument(
        "--commit_anchor_fallback_opened",
        action="store_true",
        default=True,
        help="If model_evidence/hybrid yields zero anchors, fall back to opened_supporting anchors.",
    )
    cmp.add_argument(
        "--no_commit_anchor_fallback_opened",
        action="store_false",
        dest="commit_anchor_fallback_opened",
    )

    cmp.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    cmp.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    cmp.add_argument("--bridge_bonus", type=float, default=1.5)
    cmp.add_argument("--hop1_query_mode", choices=["raw", "stripped", "llm_keywords"], default="stripped")
    cmp.add_argument("--open_split_mode", choices=["all_union_rank", "split_hop1_hop2"], default="all_union_rank")
    cmp.add_argument("--open_split_hop1", type=int, default=0)
    cmp.add_argument(
        "--open_policy",
        choices=[
            "current",
            "oracle_open_if_in_union",
            "soft_core_rerank",
            "hop2_priority",
            "bridge_one_only",
            "core_first_heuristic",
        ],
        default="current",
    )
    cmp.add_argument("--bridge_reward_bonus", type=float, default=0.0)
    cmp.add_argument("--use_query_rewrite", action="store_true", default=True)
    cmp.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    cmp.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    cmp.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    cmp.add_argument("--rewrite_queries", type=int, default=3)
    cmp.add_argument("--force_open_top_n", type=int, default=1)
    cmp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    cmp.add_argument("--use_controller", action="store_true")
    cmp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    cmp.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    cmp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    cmp.add_argument("--controller_weights_path", type=str, default="")
    cmp.add_argument("--task_split", choices=["none", "holdout"], default="none")
    cmp.add_argument("--train_ratio", type=float, default=0.7)
    cmp.add_argument("--split_seed", type=int, default=0)
    cmp.add_argument("--debug_n", type=int, default=0)
    cmp.add_argument("--debug_task_ids", type=str, default="")
    cmp.add_argument("--n_threads", type=int, default=None)
    cmp.add_argument("--traceops_max_steps", type=int, default=0)
    cmp.add_argument(
        "--traceops_eval_mode",
        choices=["deterministic", "llm"],
        default="deterministic",
    )
    cmp.add_argument("--traceops_llm_temperature", type=float, default=0.0)
    cmp.add_argument(
        "--traceops_llm_max_pivots",
        type=int,
        default=0,
        help="Cap number of pivot_check steps evaluated in llm mode (0 = no cap).",
    )
    cmp.add_argument("--traceops_llm_cache_dir", type=str, default=".cache/traceops_llm")
    cmp.add_argument("--traceops_llm_seed", type=int, default=0)
    cmp.add_argument("--traceops_llm_max_output_tokens", type=int, default=256)
    cmp.add_argument(
        "--traceops_llm_eval_scope",
        choices=["pivots", "all", "sample"],
        default="pivots",
        help="LLM evaluation scope for TraceOps: pivots only, all steps, or deterministic sampled steps.",
    )
    cmp.add_argument(
        "--traceops_llm_sample_rate",
        type=float,
        default=0.2,
        help="Sampling rate for TraceOps llm scope=sample (0..1).",
    )
    cmp.add_argument(
        "--goc_depwalk_enable",
        action="store_true",
        default=False,
        help="Enable Phase16 dependency-walk expansion in TraceOps GoC context selection.",
    )
    cmp.add_argument("--goc_depwalk_hops", type=int, default=2)
    cmp.add_argument("--goc_depwalk_topk_per_hop", type=int, default=6)
    cmp.add_argument("--traceops_indirection_rate", type=float, default=0.4)
    cmp.add_argument("--traceops_trap_distractor_count", type=int, default=4)
    cmp.add_argument("--traceops_trap_similarity_boost", type=float, default=0.7)
    cmp.add_argument("--traceops_core_size_min", type=int, default=2)
    cmp.add_argument("--traceops_core_size_max", type=int, default=4)
    cmp.add_argument("--traceops_alias_chain_len", type=int, default=2)
    cmp.add_argument(
        "--traceops_indirect_pivot_style",
        choices=["ordinal_ref", "alias_handle", "blended"],
        default="blended",
    )
    cmp.add_argument("--traceops_core_necessity_enable", action="store_true", default=False)
    cmp.add_argument("--traceops_core_necessity_require_all", action="store_true", default=True)
    cmp.add_argument("--no_traceops_core_necessity_require_all", action="store_false", dest="traceops_core_necessity_require_all")
    cmp.add_argument("--traceops_trap_decision_flip_enable", action="store_true", default=False)
    cmp.add_argument("--traceops_trap_flip_salience", type=float, default=0.25)
    cmp.add_argument(
        "--traceops_trap_flip_attach_kind",
        choices=["stale", "inapplicable", "avoided", "random", "none"],
        default="avoided",
    )
    cmp.add_argument("--traceops_trap_graph_excludable_rate", type=float, default=0.7)
    cmp.add_argument(
        "--traceops_trap_graph_excludable_kinds",
        type=str,
        default="stale,inapplicable,avoided",
    )
    cmp.add_argument("--traceops_trap_invalidation_text_strength", type=float, default=0.6)
    cmp.add_argument("--traceops_hidden_core_enable", action="store_true", default=False)
    cmp.add_argument(
        "--traceops_hidden_core_kind",
        choices=["low_overlap_clause", "alias_only_update"],
        default="low_overlap_clause",
    )
    cmp.add_argument(
        "--traceops_hidden_core_link_mode",
        choices=["depends_on", "none"],
        default="depends_on",
    )
    cmp.add_argument(
        "--traceops_force_include_required",
        action="store_true",
        default=False,
        help="TraceOps debug-only oracle mode: force include required/pivot evidence IDs in GoC context.",
    )
    cmp.add_argument("--parallel_workers", type=int, default=1)
    cmp.set_defaults(func=cmd_compare)

    swp = sub.add_parser("sweep", help="Run multi-seed/open_budget sweeps")
    swp.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default=None)
    swp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    swp.add_argument("--open_budgets", nargs="+", type=int, default=[3, 5, 8])
    swp.add_argument("--n_docs", type=int, default=None)
    swp.add_argument("--n_tasks", type=int, default=200)
    swp.add_argument("--methods", nargs="+", default=["topk", "full", "goc", "oracle", "engine"])
    swp.add_argument("--model", type=str, default="gpt-4.1-mini")
    swp.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    swp.add_argument(
        "--judge",
        choices=[
            "llm",
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        ],
        default="llm",
    )
    swp.add_argument(
        "--scenario_mode",
        choices=[
            "v0",
            "bridged_v1_1",
            "threaded_v1_2",
            "threaded_v1_3_fu",
            "threaded_v1_3_fu_decoy",
            THREADED_FU_DECOY_DEPTHJITTER_MODE,
        ],
        default="v0",
    )
    swp.add_argument(
        "--pivot_message_style",
        choices=["banner", "transcript"],
        default="transcript",
        help="How pivot updates are rendered for episode-3 tasks.",
    )
    swp.add_argument(
        "--pivot_gold_mode",
        choices=["original", "respect_ticket_updated", "both"],
        default="respect_ticket_updated",
        help=(
            "Gold evaluation mode for episode-3 pivot tasks: "
            "original=task.context/task.gold, "
            "respect_ticket_updated=effective_context-derived pivot gold, "
            "both=compute both metrics but final correctness uses pivot gold."
        ),
    )
    swp.add_argument("--goc_enable_avoids", action="store_true", default=True)
    swp.add_argument("--no_goc_enable_avoids", action="store_false", dest="goc_enable_avoids")
    swp.add_argument(
        "--goc_avoids_mode",
        choices=["legacy_commit", "applicability", "off"],
        default=None,
        help=(
            "GoC E3 avoid filtering mode: "
            "legacy_commit=commit-anchor based, applicability=applies_if mismatch based, off=disabled. "
            "When unset, default behavior is applicability via --goc_enable_avoids alias."
        ),
    )
    swp.add_argument("--thread_context_budget_chars", type=int, default=8000)
    swp.add_argument("--thread_open_policy", choices=["current", "shared_topk"], default="current")
    swp.add_argument("--e3_clause_jitter_max_chars", type=int, default=0)
    swp.add_argument("--e3_clause_jitter_max_chars_critical", type=int, default=0)
    swp.add_argument("--e3_clause_jitter_max_chars_noncritical", type=int, default=0)
    swp.add_argument("--e3_clause_jitter_max_chars_decoy", type=int, default=0)
    swp.add_argument("--e3_litm_filler_count_min", type=int, default=0)
    swp.add_argument("--e3_litm_filler_count_max", type=int, default=0)
    swp.add_argument("--e3_litm_filler_len_jitter_max", type=int, default=0)
    swp.add_argument(
        "--e3_clause_jitter_scope",
        choices=["decoy_only", "decoy_plus_noncritical", "all"],
        default="decoy_only",
    )
    swp.add_argument("--bridge_prob", type=float, default=0.8)
    swp.add_argument("--alias_density", type=float, default=None)
    swp.add_argument("--canonical_density", type=float, default=None)
    swp.add_argument("--bridge_kind", choices=["definition", "glossary"], default="definition")
    swp.add_argument("--exclusive_core_evidence", action="store_true", default=False)
    swp.add_argument("--bridged_mix_canonical_in_ticket_rate", type=float, default=0.0)
    swp.add_argument("--agent_query_policy", choices=["single_hop", "two_hop_bridge"], default="single_hop")
    swp.add_argument(
        "--search_score_mode",
        choices=["bm25", "bm25_plus_bridge_bonus"],
        default="bm25_plus_bridge_bonus",
    )
    swp.add_argument("--bridge_bonus", type=float, default=1.5)
    swp.add_argument("--hop1_query_mode", choices=["raw", "stripped", "llm_keywords"], default="stripped")
    swp.add_argument("--open_split_mode", choices=["all_union_rank", "split_hop1_hop2"], default="all_union_rank")
    swp.add_argument("--open_split_hop1", type=int, default=0)
    swp.add_argument(
        "--open_policy",
        choices=[
            "current",
            "oracle_open_if_in_union",
            "soft_core_rerank",
            "hop2_priority",
            "bridge_one_only",
            "core_first_heuristic",
        ],
        default="current",
    )
    swp.add_argument("--bridge_reward_bonus", type=float, default=0.0)
    swp.add_argument("--dotenv", type=str, default=".env")
    swp.add_argument("--out_dir", type=Path, default=None)
    swp.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
        help="Evidence padding mode for evaluation and reporting",
    )
    swp.add_argument("--min_evidence_count", type=int, default=2)
    swp.add_argument("--save_prompts", action="store_true")
    swp.add_argument("--save_raw", action="store_true")
    swp.add_argument("--save_search_snapshot", action="store_true")
    swp.add_argument("--search_snapshot_k", type=int, default=20)
    swp.add_argument("--primary_search_top_k", type=int, default=20)
    swp.add_argument("--save_goc_graph", action="store_true")
    swp.add_argument("--save_goc_dot", action="store_true")
    swp.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    swp.add_argument("--goc_graph_include_clause_text", action="store_true")
    swp.add_argument("--goc_graph_dir", type=str, default="")
    swp.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    swp.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    swp.add_argument("--goc_closedbook_universe", choices=["memory", "world"], default="memory")
    swp.add_argument("--goc_graph_frontier", action="store_true", default=True)
    swp.add_argument("--no_goc_graph_frontier", action="store_false", dest="goc_graph_frontier")
    swp.add_argument("--goc_graph_frontier_hops", type=int, default=2)
    swp.add_argument("--goc_graph_frontier_max_nodes", type=int, default=50)
    swp.add_argument("--goc_graph_frontier_seed_top_n", type=int, default=6)
    swp.add_argument("--goc_graph_frontier_score_frac", type=float, default=0.7)
    swp.add_argument("--use_query_rewrite", action="store_true", default=True)
    swp.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    swp.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    swp.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    swp.add_argument("--rewrite_queries", type=int, default=3)
    swp.add_argument("--force_open_top_n", type=int, default=1)
    swp.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    swp.add_argument("--use_controller", action="store_true")
    swp.add_argument("--controller_mode", choices=["off", "eval", "train"], default="off")
    swp.add_argument("--controller_policy", choices=["bandit", "rerank"], default="bandit")
    swp.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    swp.add_argument("--controller_weights_path", type=str, default="")
    swp.add_argument("--task_split", choices=["none", "holdout"], default="none")
    swp.add_argument("--train_ratio", type=float, default=0.7)
    swp.add_argument("--split_seed", type=int, default=0)
    swp.add_argument("--reuse_data", action="store_true")
    swp.set_defaults(func=cmd_sweep)

    ana = sub.add_parser("analyze", help="Analyze compare report failure slices")
    ana.add_argument("--report", type=str, default="")
    ana.add_argument("--k", type=int, default=20)
    ana.add_argument(
        "--mode",
        choices=["failure_slice", "bridged_ab", "selection_triage", "slot_breakdown", "analysis_bundle", "split_sweep_ab"],
        default="failure_slice",
    )
    ana.add_argument("--method", type=str, default="goc")
    ana.add_argument("--max_per_bucket", type=int, default=20)
    ana.add_argument("--run_dir", type=str, default="")
    ana.add_argument("--sweep_dir", type=str, default="")
    ana.add_argument("--results_md", type=str, default="")
    ana.set_defaults(func=cmd_analyze)

    abl = sub.add_parser("ablate_controller", help="Compare controller off vs rerank on")
    abl.add_argument("--methods", nargs="+", default=["goc"])
    abl.add_argument("--model", type=str, default="gpt-4.1-mini")
    abl.add_argument("--llm", choices=["dummy", "openai"], default="openai")
    abl.add_argument(
        "--judge",
        choices=[
            "llm",
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        ],
        default="llm",
    )
    abl.add_argument(
        "--pivot_message_style",
        choices=["banner", "transcript"],
        default="transcript",
        help="How pivot updates are rendered for episode-3 tasks.",
    )
    abl.add_argument(
        "--pivot_gold_mode",
        choices=["original", "respect_ticket_updated", "both"],
        default="respect_ticket_updated",
        help=(
            "Gold evaluation mode for episode-3 pivot tasks: "
            "original=task.context/task.gold, "
            "respect_ticket_updated=effective_context-derived pivot gold, "
            "both=compute both metrics but final correctness uses pivot gold."
        ),
    )
    abl.add_argument("--goc_enable_avoids", action="store_true", default=True)
    abl.add_argument("--no_goc_enable_avoids", action="store_false", dest="goc_enable_avoids")
    abl.add_argument(
        "--goc_avoids_mode",
        choices=["legacy_commit", "applicability", "off"],
        default=None,
        help=(
            "GoC E3 avoid filtering mode: "
            "legacy_commit=commit-anchor based, applicability=applies_if mismatch based, off=disabled. "
            "When unset, default behavior is applicability via --goc_enable_avoids alias."
        ),
    )
    abl.add_argument("--dotenv", type=str, default=".env")
    abl.add_argument("--out_dir", type=Path, default=None)
    abl.add_argument(
        "--evidence_padding_mode",
        choices=["none", "schema_only", "global"],
        default="schema_only",
    )
    abl.add_argument("--min_evidence_count", type=int, default=2)
    abl.add_argument("--save_prompts", action="store_true")
    abl.add_argument("--save_raw", action="store_true")
    abl.add_argument("--save_search_snapshot", action="store_true")
    abl.add_argument("--search_snapshot_k", type=int, default=20)
    abl.add_argument("--primary_search_top_k", type=int, default=20)
    abl.add_argument("--save_goc_graph", action="store_true")
    abl.add_argument("--save_goc_dot", action="store_true")
    abl.add_argument("--goc_graph_mode", choices=["events", "final", "events+final"], default="events")
    abl.add_argument("--goc_graph_include_clause_text", action="store_true")
    abl.add_argument("--goc_graph_dir", type=str, default="")
    abl.add_argument("--goc_graph_schema", choices=["v0", "v1"], default="v0")
    abl.add_argument("--goc_graph_sample_rate", type=float, default=1.0)
    abl.add_argument("--goc_closedbook_universe", choices=["memory", "world"], default="memory")
    abl.add_argument("--goc_graph_frontier", action="store_true", default=True)
    abl.add_argument("--no_goc_graph_frontier", action="store_false", dest="goc_graph_frontier")
    abl.add_argument("--goc_graph_frontier_hops", type=int, default=2)
    abl.add_argument("--goc_graph_frontier_max_nodes", type=int, default=50)
    abl.add_argument("--goc_graph_frontier_seed_top_n", type=int, default=6)
    abl.add_argument("--goc_graph_frontier_score_frac", type=float, default=0.7)
    abl.add_argument("--use_query_rewrite", action="store_true", default=True)
    abl.add_argument("--no_query_rewrite", action="store_false", dest="use_query_rewrite")
    abl.add_argument(
        "--query_rewrite_mode",
        choices=["base", "structured", "expanded", "hybrid"],
        default="expanded",
    )
    abl.add_argument("--merge_rank_fusion", choices=["max", "rrf"], default="max")
    abl.add_argument("--rewrite_queries", type=int, default=3)
    abl.add_argument("--force_open_top_n", type=int, default=1)
    abl.add_argument("--force_open_source", choices=["primary", "merged"], default="primary")
    abl.add_argument("--controller_mode", choices=["off", "eval", "train"], default="train")
    abl.add_argument("--controller_policy", choices=["bandit", "rerank"], default="rerank")
    abl.add_argument("--controller_state_path", type=str, default="runs/controller_state.json")
    abl.add_argument("--controller_weights_path", type=str, default="")
    abl.add_argument("--task_split", choices=["none", "holdout"], default="holdout")
    abl.add_argument("--train_ratio", type=float, default=0.7)
    abl.add_argument("--split_seed", type=int, default=0)
    abl.set_defaults(func=cmd_ablate_controller)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
