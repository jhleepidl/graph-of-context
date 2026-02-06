from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare, quickcheck_compare_report


def _clause_ids(results):
    if not isinstance(results, list):
        return []
    return [item.get("clause_id") for item in results if isinstance(item, dict) and item.get("clause_id")]


def test_metrics_quickcheck_symbolic(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=11,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=6,
    )
    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        judge="symbolic",
        dotenv=".env",
        methods=["goc", "goc_base"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=False,
        save_goc_dot=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir=str(tmp_path / "goc_graph.jsonl"),
        goc_graph_schema="v1",
        goc_graph_sample_rate=1.0,
        scenario_mode="v0",
        agent_query_policy="two_hop_bridge",
        use_query_rewrite=True,
        rewrite_queries=3,
        query_rewrite_mode="expanded",
        merge_rank_fusion="max",
        force_open_top_n=1,
        force_open_source="primary",
        use_controller=False,
        controller_mode="off",
        controller_policy="bandit",
        controller_state_path=str(tmp_path / "controller_state.json"),
        controller_weights_path="",
        task_split="none",
        train_ratio=0.7,
        split_seed=0,
        debug_n=0,
        debug_task_ids="",
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    report = json.loads(compare_reports[-1].read_text(encoding="utf-8"))

    qc = quickcheck_compare_report(report, print_fn=None)
    assert qc["passed"]

    goc_base_records = report.get("method_reports", {}).get("goc_base", {}).get("records", [])
    assert goc_base_records
    for rec in goc_base_records:
        assert rec.get("hop2_search_results") == []
        hop1_ids = set(_clause_ids(rec.get("hop1_search_results")))
        union_ids = set(_clause_ids(rec.get("search_results_union")))
        assert hop1_ids == union_ids
        if rec.get("gold_in_search_topk") is True:
            assert rec.get("gold_in_search_topk_union") is True

    goc_metrics = report.get("method_reports", {}).get("goc", {}).get("metrics", {})
    assert isinstance(goc_metrics.get("judge_accuracy"), (int, float))
    assert "acc_no_core_evidence_rate" in goc_metrics
    goc_records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert goc_records
    assert all(r.get("judge_decision") is not None for r in goc_records)
    assert all(r.get("judge_correct") in {True, False} for r in goc_records)
    assert any((r.get("judge_supporting_clause_ids") or []) for r in goc_records)
    required_fields = [
        "min_gold_core_rank_hop2",
        "min_gold_core_rank_union",
        "min_gold_winning_rank_hop2",
        "min_gold_winning_rank_union",
        "deep_rank_core_flag",
        "opened_bridge_count",
        "opened_meta_count",
        "opened_rule_count",
    ]
    for rec in goc_records:
        for field in required_fields:
            assert field in rec
