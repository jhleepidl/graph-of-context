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
        n_docs=8,
        clauses_per_doc=5,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=5,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        seed=0,
        n_docs=8,
        clauses_per_doc=5,
        n_tasks=0,
        n_threads=5,
        exception_chain_depth=1,
        update_rate=0.4,
        definition_density=0.5,
        distractor_strength=0.2,
        scenario_mode="threaded_v1_3_fu_decoy_depthjitter",
        preset="threaded_v1_3_fu_calib_n10",
        bridge_prob=0.8,
        bridged_mix_canonical_in_ticket_rate=0.0,
        alias_density=0.9,
        canonical_density=0.95,
        bridge_kind="definition",
        exclusive_core_evidence=True,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        branch_distractor_rate=0.5,
        llm="dummy",
        model="dummy",
        dotenv="",
        save_raw=False,
        save_prompts=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=False,
        save_goc_dot=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        goc_graph_schema="v1",
        goc_graph_sample_rate=1.0,
        scenario_params=None,
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
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
        judge="symbolic_packed",
        thread_open_policy="shared_topk",
        thread_context_budget_chars=1800,
        thread_context_budget_sweep="",
        evidence_padding_mode="schema_only",
        min_evidence_count=1,
        goc_rerank=False,
        goc_rerank_weights_path="",
        reuse_data=False,
        oracle_rank_override=None,
        controller_resume=False,
        controller_state_path_out="",
        controller_save_path="",
    )
    cmd_compare(compare_args)
    compare_reports = sorted((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    return json.loads(compare_reports[-1].read_text(encoding="utf-8"))


def test_threaded_cost_fields_present(tmp_path: Path) -> None:
    report = _run_tiny_threaded_compare(tmp_path)
    for method, report_obj in report.get("method_reports", {}).items():
        records = report_obj.get("records", []) or []
        assert records
        e3_seen = False
        for rec in records:
            if rec.get("episode_id") != 3:
                continue
            e3_seen = True
            assert "e3_context_token_est" in rec
            assert "e3_packed_token_est" in rec
            assert isinstance(rec.get("e3_context_token_est"), int)
            assert isinstance(rec.get("e3_packed_token_est"), int)
        assert e3_seen, f"no E3 record for method={method}"
        metrics = report_obj.get("metrics", {}) or {}
        assert "cost_per_correct_token_est" in metrics
        assert "e3_context_token_est_mean" in metrics
        assert "e3_context_token_est_total" in metrics


def test_analysis_bundle_cost_reports(tmp_path: Path) -> None:
    _run_tiny_threaded_compare(tmp_path)
    result = analyze_bundle(tmp_path)
    assert "cost_pareto_md" in result
    assert "accuracy_matched_cost_md" in result
    cost_pareto_md = tmp_path / "analysis_bundle" / "cost_pareto.md"
    accuracy_md = tmp_path / "analysis_bundle" / "accuracy_matched_cost.md"
    assert cost_pareto_md.exists()
    assert accuracy_md.exists()
    sweep_csv = tmp_path / "analysis_bundle" / "results_context_budget_sweep.csv"
    assert sweep_csv.exists()
    header = sweep_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "e3_context_token_est_mean" in header
    assert "cost_per_correct_token_est" in header
    assert "acc_per_1k_tokens" in header
