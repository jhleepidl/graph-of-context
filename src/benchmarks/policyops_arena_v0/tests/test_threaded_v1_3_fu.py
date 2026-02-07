from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare
from policyops.world import load_tasks, load_world


def test_threaded_v1_3_fu_generation_and_budget(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu",
        n_threads=2,
        open_budget_e1=3,
        open_budget_e2=3,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    world = load_world(tmp_path / "data" / "worlds")
    assert len(tasks) == 6
    by_thread = {}
    for task in tasks:
        assert task.thread_id
        by_thread.setdefault(task.thread_id, []).append(task)
    for episodes in by_thread.values():
        episode_ids = sorted([t.episode_id for t in episodes])
        assert episode_ids == [1, 2, 3]
        e3 = [t for t in episodes if t.episode_id == 3][0]
        assert e3.budgets["open_budget"] == 0
        assert e3.budgets["tool_call_budget"] == 0
        assert "Retention days=" in e3.user_ticket
        for cid in e3.gold.gold_evidence_core or []:
            clause = world.clauses.get(cid)
            assert clause is not None
            assert "Retention" in clause.text

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
        thread_context_budget_chars=500,
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    report = json.loads(compare_reports[-1].read_text(encoding="utf-8"))
    goc_unfolded_any = False
    for method in compare_args.methods:
        method_report = report.get("method_reports", {}).get(method, {})
        records = method_report.get("records", [])
        assert records
        for rec in records:
            if rec.get("episode_id") == 3:
                assert rec.get("e3_context_budget_chars") == 500
                assert rec.get("e3_context_chars_used") <= 500
                assert rec.get("e3_context_clause_count") is not None
                assert rec.get("e3_context_truncated") is not None
                assert "e3_prompt_includes_required_core" in rec
            if rec.get("episode_id") in {1, 2} and rec.get("commit_short_fact"):
                assert "retention" not in str(rec.get("commit_short_fact")).lower()
            if method == "goc" and rec.get("episode_id") == 3:
                if (rec.get("goc_unfolded_clause_count") or 0) > 0:
                    goc_unfolded_any = True
    assert goc_unfolded_any
