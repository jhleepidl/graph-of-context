from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.analysis import analyze_bundle
from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare


def _run_tiny_threaded_compare(tmp_path: Path) -> dict:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu",
        n_threads=1,
        open_budget_e1=3,
        open_budget_e2=3,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        judge="symbolic_packed",
        dotenv=".env",
        methods=["goc", "similarity_only"],
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
        scenario_mode="threaded_v1_3_fu",
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
        open_policy="current",
        open_split_mode="all_union_rank",
        open_split_hop1=0,
        bridge_bonus=1.5,
        search_score_mode="bm25_plus_bridge_bonus",
        bridge_reward_bonus=0.0,
        n_threads=None,
        thread_context_budget_chars=600,
        thread_context_budget_sweep="",
        thread_open_policy="current",
        preset="threaded_v1_3_fu_calib_n10",
    )
    cmd_compare(compare_args)
    compare_reports = sorted((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    return json.loads(compare_reports[-1].read_text(encoding="utf-8"))


def test_goc_advantage_metrics_exist_in_report_and_bundle(tmp_path: Path) -> None:
    report = _run_tiny_threaded_compare(tmp_path)

    for method in ["goc", "similarity_only"]:
        method_report = report.get("method_reports", {}).get(method, {})
        assert method_report
        metrics = method_report.get("metrics", {}) or {}
        assert "closure_recall_core_mean" in metrics
        assert "wrong_branch_recall_rate_mean" in metrics

        records = method_report.get("records", []) or []
        assert records
        e3_records = [r for r in records if r.get("episode_id") == 3]
        assert e3_records
        for rec in e3_records:
            for key in [
                "opened_evidence_clause_ids",
                "opened_evidence_doc_ids",
                "active_context_clause_ids",
                "active_context_doc_ids",
                "unfolded_activated_clause_ids",
                "unfolded_activated_doc_ids",
                "closure_recalled_clause_ids",
                "closure_recall_core",
                "wrong_branch_recall_rate",
            ]:
                assert key in rec

    analyze_bundle(tmp_path)
    sweep_csv = tmp_path / "analysis_bundle" / "results_context_budget_sweep.csv"
    assert sweep_csv.exists()
    header = sweep_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "closure_recall_core_mean" in header
    assert "wrong_branch_recall_rate_mean" in header
