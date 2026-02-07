from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import (
    generate_world_and_tasks,
    DECOY_MIN_CHARS,
    JITTER_FILLER_FRAGMENT,
)
from policyops.run import cmd_compare
from policyops.world import load_tasks, load_world, evaluate_context, World


def test_threaded_generation_exact_n_threads(tmp_path: Path) -> None:
    requested_threads = 50
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=requested_threads,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    world = load_world(tmp_path / "data" / "worlds")
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    thread_ids = sorted({t.thread_id for t in tasks if t.thread_id})
    assert len(thread_ids) == requested_threads
    assert len(tasks) == requested_threads * 3
    assert world.meta.get("n_threads_requested") == requested_threads
    assert world.meta.get("n_threads_generated_final") == requested_threads
    assert int(world.meta.get("n_threads_generated_raw") or 0) >= requested_threads


def test_critical_jitter_changes_len_distribution(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=30,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
        e3_clause_jitter_max_chars_critical=200,
        e3_clause_jitter_max_chars_noncritical=0,
        e3_clause_jitter_max_chars_decoy=0,
    )
    world = load_world(tmp_path / "data" / "worlds")
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    critical_ids = set()
    for task in tasks:
        if task.critical_clause_id_e1:
            critical_ids.add(task.critical_clause_id_e1)
        if task.critical_clause_id_e2:
            critical_ids.add(task.critical_clause_id_e2)
    lengths = [len(world.clauses[cid].text) for cid in critical_ids if cid in world.clauses]
    assert lengths
    assert max(lengths) - min(lengths) > 0
    assert any(
        JITTER_FILLER_FRAGMENT in world.clauses[cid].text
        for cid in critical_ids
        if cid in world.clauses
    )


def test_defaults_no_change(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=20,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
        e3_clause_jitter_max_chars_critical=0,
        e3_clause_jitter_max_chars_noncritical=0,
        e3_clause_jitter_max_chars_decoy=0,
    )
    world = load_world(tmp_path / "data" / "worlds")
    assert all(JITTER_FILLER_FRAGMENT not in clause.text for clause in world.clauses.values())


def test_depth_jitter_params_saved(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=20,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
        e3_litm_filler_count_min=0,
        e3_litm_filler_count_max=6,
        e3_litm_filler_len_jitter_max=20,
    )
    world = load_world(tmp_path / "data" / "worlds")
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    assert world.meta.get("e3_litm_filler_count_min") == 0
    assert world.meta.get("e3_litm_filler_count_max") == 6
    assert world.meta.get("e3_litm_filler_len_jitter_max") == 20
    counts = []
    for task in tasks:
        if task.episode_id != 3:
            continue
        cfg = task.thread_config or {}
        assert "e3_litm_filler_count" in cfg
        assert "e3_litm_filler_clause_ids" in cfg
        counts.append(int(cfg.get("e3_litm_filler_count")))
    assert counts
    assert min(counts) >= 0
    assert max(counts) <= 6


def test_depth_jitter_filler_count_distribution(tmp_path: Path) -> None:
    varied_dir = tmp_path / "varied"
    fixed_dir = tmp_path / "fixed"
    generate_world_and_tasks(
        out_dir=varied_dir,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=120,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
        e3_litm_filler_count_min=0,
        e3_litm_filler_count_max=6,
    )
    varied_tasks = load_tasks(varied_dir / "data" / "tasks" / "tasks.jsonl")
    varied_counts = sorted(
        {
            int((task.thread_config or {}).get("e3_litm_filler_count", -1))
            for task in varied_tasks
            if task.episode_id == 3
        }
    )
    assert len(varied_counts) >= 3
    assert varied_counts[0] >= 0
    assert varied_counts[-1] <= 6

    generate_world_and_tasks(
        out_dir=fixed_dir,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=40,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
        e3_litm_filler_count_min=2,
        e3_litm_filler_count_max=2,
    )
    fixed_tasks = load_tasks(fixed_dir / "data" / "tasks" / "tasks.jsonl")
    fixed_counts = {
        int((task.thread_config or {}).get("e3_litm_filler_count", -1))
        for task in fixed_tasks
        if task.episode_id == 3
    }
    assert fixed_counts == {2}


def test_decoys_irrelevant_and_long(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=1,
        open_budget_e1=4,
        open_budget_e2=4,
        open_budget_e3=0,
        tool_budget_e1=10,
        tool_budget_e2=10,
        tool_budget_e3=0,
        exclusive_core_evidence=True,
    )
    world = load_world(tmp_path / "data" / "worlds")
    tasks = load_tasks(tmp_path / "data" / "tasks" / "tasks.jsonl")
    decoy_ids = [cid for cid, clause in world.clauses.items() if str(clause.doc_id).startswith("DECOY")]
    assert decoy_ids
    for cid in decoy_ids:
        assert len(world.clauses[cid].text) >= DECOY_MIN_CHARS

    pruned_clauses = {
        cid: clause for cid, clause in world.clauses.items() if cid not in decoy_ids
    }
    pruned_world = World(
        documents=world.documents,
        clauses=pruned_clauses,
        meta=world.meta,
    )
    for task in tasks:
        decision, _, _, _ = evaluate_context(pruned_world, task.context)
        assert decision == task.gold.decision


def test_decoy_shared_topk_opens_critical(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        scenario_mode="threaded_v1_3_fu_decoy",
        n_threads=1,
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
        model="dummy",
        llm="dummy",
        judge="symbolic_packed",
        dotenv=".env",
        methods=["goc"],
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
        scenario_mode="threaded_v1_3_fu_decoy",
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
        thread_context_budget_sweep="",
        thread_open_policy="shared_topk",
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    report = json.loads(compare_reports[-1].read_text(encoding="utf-8"))
    records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert records
    seen_e1 = False
    seen_e2 = False
    for rec in records:
        if rec.get("episode_id") == 1:
            seen_e1 = True
            assert rec.get("critical_clause_id_e1") in (rec.get("opened_clause_ids") or [])
        if rec.get("episode_id") == 2:
            seen_e2 = True
            assert rec.get("critical_clause_id_e2") in (rec.get("opened_clause_ids") or [])
    assert seen_e1 and seen_e2
