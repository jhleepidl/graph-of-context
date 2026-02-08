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
try:
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    _root_src = Path(__file__).resolve().parents[4]
    if str(_root_src) not in sys.path:
        sys.path.append(str(_root_src))
    from goc_logger.graph import GoCGraph
    from goc_logger.log import append_event, build_event
from .env import PolicyOpsEnv
from .eval import aggregate_metrics, evaluate_prediction, gold_decision_distribution, save_report
from .generator import generate_world_and_tasks
from .world import load_tasks, load_world

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
) -> tuple[List[str], Dict[str, List[str]]]:
    ticket_tokens = set(_tokenize(ticket))
    scored: List[tuple[str, float]] = []
    reasons: Dict[str, List[str]] = {}
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
        scored.append((cid, score))
        reasons[cid] = cid_reasons
    scored.sort(key=lambda item: (-item[1], item[0]))
    ordered = [cid for cid, _ in scored]
    # Ensure commit anchors are first in stable order.
    critical_first = [cid for cid in critical_clause_ids if cid in ordered]
    commit_first = [cid for cid in commit_clause_ids if cid in ordered and cid not in critical_first]
    remaining = [cid for cid in ordered if cid not in critical_first and cid not in commit_first]
    return critical_first + commit_first + remaining, reasons


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

    total_tasks = len(tasks)
    for task_idx, task in enumerate(tasks, start=1):
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
        judge_correct_full_episode: bool | None = None
        full_episode_supporting_count: int | None = None
        if threaded_mode and thread_state and episode_id:
            task_for_run = copy.deepcopy(task)
            commit_refs = _format_commit_refs(
                (thread_state.get("commit1") or {}).get("short_fact"),
                (thread_state.get("commit2") or {}).get("short_fact"),
                episode_id,
            )
            if commit_refs:
                task_for_run.user_ticket = f"{task_for_run.user_ticket}\n\n{commit_refs}"
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
                clause_objs = [world.clauses.get(cid) for cid in opened_history_ids if world.clauses.get(cid)]
                summary_text = summarize_clause_history(clause_objs, max_items=6)
                critical_clause_ids = list(getattr(task, "critical_core_clause_ids", None) or [])
                context_clause_ids, reason_map = _select_goc_unfold_clause_ids(
                    opened_history_ids,
                    commit_clause_ids,
                    critical_clause_ids,
                    task_for_run.user_ticket,
                    world,
                )
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
            opened_ids = []
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
            }
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
                internal_budget_active=1200,
                internal_budget_unfold=650,
                internal_unfold_k=8,
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

        commit_supporting_clause_ids: List[str] | None = None
        commit_short_fact: Dict[str, Any] | None = None
        commit_correct: bool | None = None
        e3_evidence_valid: bool | None = None
        thread_judge_correct: bool | None = None
        commit_clause_ids: List[str] | None = None
        if threaded_mode and thread_state and episode_id in {1, 2}:
            commit_supporting_clause_ids = _extract_commit_supporting(
                task,
                opened_ids,
                world,
                episode_kind,
            )
            commit_short_fact = _extract_commit_short_fact(
                task,
                commit_supporting_clause_ids,
                world,
                episode_kind,
            )
            core_ids = list(
                getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
            )
            commit_correct = bool(
                commit_supporting_clause_ids
                and any(cid in commit_supporting_clause_ids for cid in core_ids)
            )
            if episode_id == 1:
                thread_state["commit1"] = {
                    "supporting_clause_ids": commit_supporting_clause_ids,
                    "short_fact": commit_short_fact or {},
                    "commit_correct": commit_correct,
                }
            elif episode_id == 2:
                thread_state["commit2"] = {
                    "supporting_clause_ids": commit_supporting_clause_ids,
                    "short_fact": commit_short_fact or {},
                    "commit_correct": commit_correct,
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
                    (commit1.get("supporting_clause_ids") or [])
                    + (commit2.get("supporting_clause_ids") or [])
                )
            )
            commit_supporting_clause_ids = commit_clause_ids
            if not e12_opened_clause_ids:
                e12_opened_clause_ids = list(
                    dict.fromkeys(thread_state.get("opened_history_ids", []))
                )
            if not full_episode_clause_ids:
                full_episode_clause_ids = list(
                    dict.fromkeys(e12_opened_clause_ids + e3_packed_clause_ids)
                )
        primary_results = diag.get("primary_search_results", []) if isinstance(diag, dict) else []
        gold_core_ids = list(
            getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
        )
        retrieval_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(task.gold.gold_evidence or []),
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
            gold_ids=list(task.gold.gold_evidence or []),
            search_results=hop1_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        hop2_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(task.gold.gold_evidence or []),
            search_results=hop2_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        union_diag = compute_retrieval_diagnostics(
            opened_ids=opened_ids,
            gold_ids=list(task.gold.gold_evidence or []),
            search_results=union_results,
            top_k_used=diag.get("primary_search_top_k", args.primary_search_top_k)
            if isinstance(diag, dict)
            else args.primary_search_top_k,
            save_snapshot=False,
        )
        winning_clause_id = (task.gold.gold_evidence or [None])[0]
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
        judge_mode = getattr(args, "judge", "llm")
        if judge_mode in {
            "symbolic",
            "symbolic_packed",
            "symbolic_packed_allcritical",
            "symbolic_full_episode",
        }:
            if judge_mode in {"symbolic_packed", "symbolic_packed_allcritical"} and threaded_mode and episode_id == 3:
                judge_pred = judge_from_opened_clauses(task, e3_packed_clause_ids, world)
            elif judge_mode == "symbolic_full_episode" and threaded_mode and episode_id == 3:
                judge_pred = judge_from_opened_clauses(task, full_episode_clause_ids, world)
            elif judge_mode == "symbolic" and threaded_mode and episode_id == 3 and commit_clause_ids is not None:
                judge_pred = judge_threaded_final(task, commit_clause_ids, world)
            else:
                judge_pred = judge_from_opened_clauses(task, opened_ids, world)
            judge_decision = judge_pred.get("decision")
            judge_correct = judge_decision == task.gold.decision
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
        commit1_correct = None
        commit2_correct = None
        if threaded_mode and thread_state:
            commit1 = thread_state.get("commit1") or {}
            commit2 = thread_state.get("commit2") or {}
            commit1_correct = commit1.get("commit_correct")
            commit2_correct = commit2.get("commit_correct")
            if episode_id == 3 and commit_clause_ids is not None:
                commit_correct = bool(commit1_correct is True and commit2_correct is True)
                e3_evidence_valid = all(
                    cid in set(commit_clause_ids) for cid in (evidence_after or [])
                )
                thread_judge_correct = bool(
                    judge_correct
                    and commit1_correct is True
                    and commit2_correct is True
                    and e3_evidence_valid
                )
                judge_correct = thread_judge_correct
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
        task_metrics = evaluate_prediction(pred_for_eval, task.gold, world)
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
            winning_clause = task.gold.gold_evidence[0] if task.gold.gold_evidence else None
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
            "gold_decision": task.gold.decision,
            "decision_correct": pred_decision == task.gold.decision,
            "judge_decision": judge_decision,
            "judge_correct": judge_correct,
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
            "evidence_precision": task_metrics.get("evidence_precision"),
            "evidence_recall": task_metrics.get("evidence_recall"),
            "evidence_precision_core": task_metrics.get("evidence_precision_core"),
            "evidence_recall_core": task_metrics.get("evidence_recall_core"),
            "critical_evidence_hit": task_metrics.get("critical_evidence_hit"),
            "pred_evidence_count": len(pred_for_record.get("evidence", []) or []),
            "gold_evidence_count": len(task.gold.gold_evidence or []),
            "gold_evidence_ids": list(task.gold.gold_evidence or []),
            "gold_evidence_core_ids": list(
                getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
            ),
            "gold_evidence_meta_ids": list(getattr(task.gold, "gold_evidence_meta", None) or []),
            "meta_evidence_present": bool(getattr(task.gold, "gold_evidence_meta", None)),
            "core_evidence_size": len(
                getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or []
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
            "commit_short_fact": commit_short_fact,
            "commit_correct": commit_correct,
            "commit_clause_ids": commit_clause_ids,
            "e3_evidence_valid": e3_evidence_valid,
            "thread_judge_correct": thread_judge_correct,
            "critical_clause_id_e1": getattr(task, "critical_clause_id_e1", None),
            "critical_clause_id_e2": getattr(task, "critical_clause_id_e2", None),
            "critical_core_clause_ids": list(getattr(task, "critical_core_clause_ids", None) or []),
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
        winning_clause = (task.gold.gold_evidence or [None])[0]
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
                        "thread_judge_correct": record.get("thread_judge_correct"),
                        "thread_decision_correct": record.get("decision_correct"),
                        "commit1_correct": commit1.get("commit_correct"),
                        "commit2_correct": commit2.get("commit_correct"),
                        "e3_evidence_valid": record.get("e3_evidence_valid"),
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
            goc_graph.add_node(ticket_id, "ticket", text=task.user_ticket, step=goc_graph.step)
            _log("INIT", {"episode_id": episode_node_id, "ticket_id": ticket_id})
            goc_graph.step += 1

            if threaded_mode and episode_id and record.get("commit_supporting_clause_ids"):
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
                        "supporting_clause_ids": record.get("commit_supporting_clause_ids"),
                    },
                )

            gold_node = f"gold:{task.task_id}"
            goc_graph.add_node(gold_node, "gold", decision=task.gold.decision, step=goc_graph.step)
            for cid in task.gold.gold_evidence or []:
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
                        "goc_unfolded_critical_clause_count_mean": metrics.get(
                            "goc_unfolded_critical_clause_count_mean"
                        ),
                        "goc_folded_episode_count_mean": metrics.get(
                            "goc_folded_episode_count_mean"
                        ),
                        "closure_recall_core_mean": metrics.get(
                            "closure_recall_core_mean"
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
                "e3_context_truncated_rate",
                "e3_packed_all_critical_rate",
                "e3_packed_any_critical_rate",
                "e3_packed_critical_count_mean",
                "e3_decoy_clause_count_mean",
                "e3_context_clause_count_mean",
                "e3_context_chars_used_mean",
                "goc_unfolded_clause_count_mean",
                "goc_unfolded_critical_clause_count_mean",
                "goc_folded_episode_count_mean",
                "closure_recall_core_mean",
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
        "e3_clause_jitter_max_chars": world.meta.get("e3_clause_jitter_max_chars"),
        "e3_clause_jitter_max_chars_critical": world.meta.get("e3_clause_jitter_max_chars_critical"),
        "e3_clause_jitter_max_chars_noncritical": world.meta.get("e3_clause_jitter_max_chars_noncritical"),
        "e3_clause_jitter_max_chars_decoy": world.meta.get("e3_clause_jitter_max_chars_decoy"),
        "e3_clause_jitter_scope": world.meta.get("e3_clause_jitter_scope"),
        "e3_litm_filler_count_min": world.meta.get("e3_litm_filler_count_min"),
        "e3_litm_filler_count_max": world.meta.get("e3_litm_filler_count_max"),
        "e3_litm_filler_len_jitter_max": world.meta.get("e3_litm_filler_len_jitter_max"),
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
                    f"PYTHONPATH=src/benchmarks/policyops_arena_v0/src:src python -m policyops.run compare "
                    f"--llm {args.llm} --model {args.model} --methods {' '.join(methods)} "
                    f"--save_goc_graph --save_goc_dot",
                    "",
                    "## Triage command",
                    f"PYTHONPATH=src/benchmarks/policyops_arena_v0/src:src python -m policyops.triage "
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
    gen.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default=None)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--n_docs", type=int, default=None)
    gen.add_argument("--clauses_per_doc", type=int, default=None)
    gen.add_argument("--n_tasks", type=int, default=200)
    gen.add_argument("--exception_chain_depth", type=int, default=2)
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
    gen.add_argument("--out_dir", type=Path, default=None)
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="Evaluate baselines")
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
    ev.set_defaults(func=cmd_eval)

    cmp = sub.add_parser("compare", help="Compare methods in one run")
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
    cmp.add_argument("--thread_context_budget_chars", type=int, default=8000)
    cmp.add_argument("--thread_open_policy", choices=["current", "shared_topk"], default="current")
    cmp.add_argument("--thread_context_budget_sweep", type=str, default="")
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
