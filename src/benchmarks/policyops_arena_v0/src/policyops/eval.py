from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from .schemas import Gold, Task, World
from .world import CRITICAL_KINDS


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    precision = len(pred & gold) / len(pred)
    recall = len(pred & gold) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _precision_recall(pred: set[str], gold: set[str]) -> Tuple[float, float]:
    if not pred and not gold:
        return 1.0, 1.0
    if not pred:
        return 0.0, 0.0
    precision = len(pred & gold) / len(pred)
    recall = len(pred & gold) / len(gold) if gold else 1.0
    return precision, recall


def evaluate_prediction(pred: Dict[str, Any], gold: Gold, world: World) -> Dict[str, float]:
    decision_acc = 1.0 if pred.get("decision") == gold.decision else 0.0
    pred_conditions = set(pred.get("conditions", []) or [])
    gold_conditions = set(gold.conditions or [])
    cond_f1 = _f1(pred_conditions, gold_conditions)

    pred_evidence = set(pred.get("evidence", []) or [])
    gold_evidence = set(gold.gold_evidence or [])
    evidence_precision, evidence_recall = _precision_recall(pred_evidence, gold_evidence)
    core_ids = set(getattr(gold, "gold_evidence_core", []) or gold.gold_evidence or [])
    evidence_precision_core, evidence_recall_core = _precision_recall(pred_evidence, core_ids)

    # Critical evidence hit ratio.
    required_kinds = set()
    for clause_id in gold.gold_evidence:
        clause = world.clauses.get(clause_id)
        if clause and clause.kind in CRITICAL_KINDS:
            required_kinds.add(clause.kind)
    if not required_kinds:
        critical_hit = 1.0
    else:
        hits = 0
        for kind in required_kinds:
            if any(world.clauses.get(cid) and world.clauses[cid].kind == kind for cid in pred_evidence):
                hits += 1
        critical_hit = hits / len(required_kinds)

    return {
        "decision_accuracy": decision_acc,
        "condition_f1": cond_f1,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_precision_core": evidence_precision_core,
        "evidence_recall_core": evidence_recall_core,
        "critical_evidence_hit": critical_hit,
    }


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: mean([m[key] for m in metrics]) for key in keys}


def gold_decision_distribution(tasks: List[Task]) -> Dict[str, int]:
    dist: Dict[str, int] = {
        "allow": 0,
        "deny": 0,
        "require_condition": 0,
        "needs_more_info": 0,
    }
    for task in tasks:
        decision = task.gold.decision
        if decision in dist:
            dist[decision] += 1
        else:
            dist[decision] = dist.get(decision, 0) + 1
    return dist


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
