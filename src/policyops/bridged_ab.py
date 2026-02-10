from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def compute_bridged_ab_slices(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    filtered = [r for r in records if r.get("scenario_mode") == "bridged_v1_1"]
    axes = {
        "A": [
            "A0_no_bridge_needed",
            "A1_needed_not_opened",
            "A2_opened_wrong_bridge",
            "A3_opened_gold_bridge",
            "A_unknown",
        ],
        "B": [
            "B0_no_hop2",
            "B1_hop2_no_gold_canonical",
            "B2_hop2_with_gold_canonical",
        ],
    }
    cells: Dict[str, Dict[str, Any]] = {}

    def bucket_a(rec: Dict[str, Any]) -> str:
        if rec.get("bridge_needed") is False:
            return "A0_no_bridge_needed"
        if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is False:
            return "A1_needed_not_opened"
        if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is True:
            if rec.get("bridge_opened_gold") is True:
                return "A3_opened_gold_bridge"
            if rec.get("bridge_opened_gold") is False:
                return "A2_opened_wrong_bridge"
        return "A_unknown"

    def bucket_b(rec: Dict[str, Any]) -> str:
        hop2_executed = rec.get("hop2_executed") is True
        hop2_gold = rec.get("hop2_query_contains_gold_canonical") is True
        if not hop2_executed:
            return "B0_no_hop2"
        if hop2_gold:
            return "B2_hop2_with_gold_canonical"
        return "B1_hop2_no_gold_canonical"

    for rec in filtered:
        a = bucket_a(rec)
        b = bucket_b(rec)
        cells.setdefault(a, {}).setdefault(
            b,
            {
                "decision_acc": [],
                "judge_acc": [],
                "opened_gold_coverage_core": [],
                "opened_has_winning_clause_core": [],
                "gold_in_search_topk_core": [],
                "min_gold_core_rank_union": [],
                "deep_rank_core_flag": [],
                "open_calls": [],
                "tool_calls": [],
                "winning_clause_rank_core": [],
                "bridge_probe_contains_gold_canonical": [],
                "bridge_opened_contains_gold_canonical": [],
            },
        )
        cell = cells[a][b]
        cell["decision_acc"].append(1.0 if rec.get("decision_correct") else 0.0)
        if rec.get("judge_correct") is not None:
            cell["judge_acc"].append(1.0 if rec.get("judge_correct") else 0.0)
        cell["opened_gold_coverage_core"].append(float(rec.get("opened_gold_coverage_core") or 0.0))
        cell["opened_has_winning_clause_core"].append(
            1.0 if rec.get("opened_has_winning_clause_core") else 0.0
        )
        cell["gold_in_search_topk_core"].append(1.0 if rec.get("gold_in_search_topk_core") else 0.0)
        if rec.get("min_gold_core_rank_union") is not None:
            cell["min_gold_core_rank_union"].append(float(rec.get("min_gold_core_rank_union")))
        cell["deep_rank_core_flag"].append(1.0 if rec.get("deep_rank_core_flag") else 0.0)
        cell["open_calls"].append(float(rec.get("open_calls") or 0.0))
        cell["tool_calls"].append(float(rec.get("tool_calls") or 0.0))
        if rec.get("winning_clause_rank_core") is not None:
            cell["winning_clause_rank_core"].append(float(rec.get("winning_clause_rank_core")))
        cell["bridge_probe_contains_gold_canonical"].append(
            1.0 if rec.get("bridge_probe_contains_gold_canonical") else 0.0
        )
        cell["bridge_opened_contains_gold_canonical"].append(
            1.0 if rec.get("bridge_opened_contains_gold_canonical") else 0.0
        )

    rendered: Dict[str, Dict[str, Any]] = {}
    for a_key, b_map in cells.items():
        rendered[a_key] = {}
        for b_key, values in b_map.items():
            rendered[a_key][b_key] = {
                "n": len(values["decision_acc"]),
                "decision_acc": _safe_mean(values["decision_acc"]),
                "judge_acc": _safe_mean(values["judge_acc"]) if values["judge_acc"] else None,
                "opened_gold_coverage_core_mean": _safe_mean(values["opened_gold_coverage_core"]),
                "opened_has_winning_clause_core_rate": _safe_mean(
                    values["opened_has_winning_clause_core"]
                ),
                "gold_in_search_topk_core_rate": _safe_mean(values["gold_in_search_topk_core"]),
                "core_min_rank_union_mean": _safe_mean(values["min_gold_core_rank_union"])
                if values["min_gold_core_rank_union"]
                else None,
                "deep_rank_core_rate": _safe_mean(values["deep_rank_core_flag"]),
                "open_calls_avg": _safe_mean(values["open_calls"]),
                "tool_calls_avg": _safe_mean(values["tool_calls"]),
                "winning_clause_rank_core_mean": _safe_mean(values["winning_clause_rank_core"]),
                "bridge_probe_contains_gold_canonical_rate": _safe_mean(
                    values["bridge_probe_contains_gold_canonical"]
                ),
                "bridge_opened_contains_gold_canonical_rate": _safe_mean(
                    values["bridge_opened_contains_gold_canonical"]
                ),
            }

    return {"axes": axes, "cells": rendered, "n_records": len(filtered)}
