from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare
from policyops.world import load_tasks


def test_threaded_v1_3_fu_shared_open_and_budget(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu",
        n_threads=2,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    assert tasks
    for task in tasks:
        assert task.critical_core_clause_ids

    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        judge="symbolic",
        dotenv=".env",
        methods=["goc", "full_history", "similarity_only", "agent_fold"],
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
        thread_context_budget_chars=200,
        thread_open_policy="shared_topk",
        thread_context_budget_sweep="",
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    report = json.loads(compare_reports[-1].read_text(encoding="utf-8"))

    # Shared open policy: identical E1/E2 opened clauses across methods.
    ref_by_key = {}
    for method in compare_args.methods:
        records = report["method_reports"][method]["records"]
        for rec in records:
            if rec.get("episode_id") not in {1, 2}:
                continue
            key = (rec.get("thread_id"), rec.get("episode_id"))
            opened = sorted(rec.get("opened_clause_ids") or [])
            if key not in ref_by_key:
                ref_by_key[key] = opened
            else:
                assert opened == ref_by_key[key]

    # Budget enforcement: full_history truncated in at least one E3.
    truncated_any = False
    for rec in report["method_reports"]["full_history"]["records"]:
        if rec.get("episode_id") == 3 and rec.get("e3_context_truncated") is True:
            truncated_any = True
            break
    assert truncated_any

    # Critical core metrics exist.
    metrics = report["method_reports"]["goc"]["metrics"]
    assert "e3_packed_all_critical_rate" in metrics
    assert "e3_packed_any_critical_rate" in metrics
    assert "e3_packed_critical_count_mean" in metrics

    # Commit facts should not contain retention detail.
    for rec in report["method_reports"]["goc"]["records"]:
        if rec.get("episode_id") in {1, 2} and rec.get("commit_short_fact"):
            assert "retention" not in str(rec.get("commit_short_fact")).lower()
