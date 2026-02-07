from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare


def _base_compare_args(tmp_path: Path, *, budget: int, methods: list[str], judge: str) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=str(tmp_path),
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=0,
        n_threads=4,
        exception_chain_depth=1,
        update_rate=0.4,
        definition_density=0.5,
        distractor_strength=0.2,
        scenario_mode="threaded_v1_3_fu",
        preset="threaded_v1_3_fu_calib_n10",
        bridge_prob=0.8,
        bridged_mix_canonical_in_ticket_rate=0.0,
        alias_density=0.9,
        canonical_density=0.95,
        bridge_kind="definition",
        exclusive_core_evidence=True,
        open_budget_e1=3,
        open_budget_e2=3,
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
        methods=methods,
        judge=judge,
        thread_open_policy="shared_topk",
        thread_context_budget_chars=budget,
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


def _run_compare(tmp_path: Path, *, budget: int, methods: list[str], judge: str) -> dict:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu",
        n_threads=4,
        open_budget_e1=3,
        open_budget_e2=3,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    compare_args = _base_compare_args(tmp_path, budget=budget, methods=methods, judge=judge)
    cmd_compare(compare_args)
    compare_reports = sorted((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    return json.loads(compare_reports[-1].read_text(encoding="utf-8"))


def test_symbolic_packed_uses_packed_only(tmp_path: Path) -> None:
    report = _run_compare(tmp_path, budget=0, methods=["full_history"], judge="symbolic_packed")
    records = report.get("method_reports", {}).get("full_history", {}).get("records", [])
    assert records
    found = False
    for rec in records:
        if rec.get("episode_id") != 3:
            continue
        if rec.get("gold_decision") == "needs_more_info":
            continue
        if (rec.get("e3_packed_clause_ids") or []) == []:
            found = True
            assert rec.get("judge_correct") is False
            break
    assert found, "No E3 record with empty packed clauses and non-NMI gold decision."


def test_packed_clause_ids_recorded(tmp_path: Path) -> None:
    report = _run_compare(
        tmp_path,
        budget=500,
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
        judge="symbolic_packed",
    )
    for method, report_obj in report.get("method_reports", {}).items():
        records = report_obj.get("records", [])
        assert records
        any_with_context = False
        for rec in records:
            assert isinstance(rec.get("e3_packed_clause_ids"), list)
            count = rec.get("e3_context_clause_count") or 0
            if rec.get("episode_id") == 3 and count > 0:
                any_with_context = True
                assert len(rec.get("e3_packed_clause_ids") or []) > 0
        assert any_with_context


def test_packed_contains_critical_aligns_with_judge(tmp_path: Path) -> None:
    report = _run_compare(
        tmp_path,
        budget=500,
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
        judge="symbolic_packed",
    )
    for report_obj in report.get("method_reports", {}).values():
        records = report_obj.get("records", [])
        assert records
        for rec in records:
            if rec.get("episode_id") != 3:
                continue
            critical_ids = rec.get("critical_core_clause_ids") or []
            if not critical_ids:
                continue
            if rec.get("judge_correct") is True:
                assert rec.get("e3_packed_any_critical") is True


def test_prompt_includes_critical_matches_packed_ids(tmp_path: Path) -> None:
    report = _run_compare(
        tmp_path,
        budget=500,
        methods=["full_history"],
        judge="symbolic_packed",
    )
    records = report.get("method_reports", {}).get("full_history", {}).get("records", [])
    assert records
    for rec in records:
        if rec.get("episode_id") != 3:
            continue
        critical_ids = rec.get("critical_core_clause_ids") or []
        if not critical_ids:
            continue
        expected_all = (rec.get("e3_packed_critical_count") or 0) == len(critical_ids)
        expected_any = (rec.get("e3_packed_critical_count") or 0) > 0
        assert rec.get("e3_prompt_includes_critical_core") == expected_all
        assert rec.get("e3_packed_all_critical") == expected_all
        assert rec.get("e3_packed_any_critical") == expected_any


def test_e3_truncation_metrics_consistent(tmp_path: Path) -> None:
    report = _run_compare(
        tmp_path,
        budget=300,
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
        judge="symbolic_packed",
    )
    for report_obj in report.get("method_reports", {}).values():
        for rec in report_obj.get("records", []):
            if rec.get("episode_id") != 3:
                continue
            before = rec.get("e3_packed_total_chars_before")
            budget = rec.get("e3_context_budget_chars")
            truncated = rec.get("e3_packed_truncated")
            dropped = rec.get("e3_packed_dropped_clause_count")
            assert isinstance(before, int)
            assert isinstance(budget, int)
            assert isinstance(truncated, bool)
            assert isinstance(dropped, int)
            if before <= budget:
                assert truncated is False
            if truncated is False:
                assert dropped == 0


def test_symbolic_packed_allcritical_requires_all_critical(tmp_path: Path) -> None:
    report = _run_compare(
        tmp_path,
        budget=500,
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
        judge="symbolic_packed_allcritical",
    )
    for report_obj in report.get("method_reports", {}).values():
        metrics = report_obj.get("metrics", {}) or {}
        assert "e3_judge_accuracy_packed_allcritical" in metrics
        for rec in report_obj.get("records", []):
            if rec.get("episode_id") != 3:
                continue
            critical_ids = rec.get("critical_core_clause_ids") or []
            if not critical_ids:
                continue
            if rec.get("e3_packed_all_critical") is False:
                assert rec.get("judge_correct") is False
