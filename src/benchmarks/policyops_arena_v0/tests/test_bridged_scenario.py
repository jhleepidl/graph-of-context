from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.generator import generate_world_and_tasks
from policyops.env import PolicyOpsEnv
from policyops.run import cmd_compare


def test_bridged_scenario_two_hop_and_scoring(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=0,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=5,
        scenario_mode="bridged_v1_1",
        bridge_prob=1.0,
        alias_density=1.0,
        canonical_density=1.0,
        bridge_kind="definition",
    )
    task = next(t for t in tasks if t.bridge_clause_id)
    assert task.slot_hint_alias
    assert task.canonical_slot_term
    bridge_clause = world.clauses.get(task.bridge_clause_id)
    assert bridge_clause is not None
    assert bridge_clause.is_bridge_doc is True
    assert bridge_clause.bridge_for_slot == task.context.get("slot")
    assert bridge_clause.canonical_terms
    core_counts = [len(getattr(t.gold, "gold_evidence_core", []) or []) for t in tasks]
    assert sum(1 for c in core_counts if c >= 1) >= 3

    env = PolicyOpsEnv(
        world,
        tool_call_budget=20,
        open_budget=5,
        search_score_mode="bm25_plus_bridge_bonus",
        bridge_bonus=1.5,
    )
    hop1_results = env.search(task.slot_hint_alias, top_k=5)
    hop1_ids = [item.get("clause_id") for item in hop1_results]
    assert task.bridge_clause_id in hop1_ids

    hop2_results = env.search(task.canonical_slot_term, top_k=5)
    assert any(
        (item.get("bonus_applied", 0.0) or 0.0) > 0.0 and item.get("query_contains_canonical")
        for item in hop2_results
    )

    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        dotenv=".env",
        methods=["goc"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=True,
        save_goc_dot=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir=str(tmp_path / "goc_graph.jsonl"),
        goc_graph_schema="v1",
        goc_graph_sample_rate=1.0,
        scenario_mode="bridged_v1_1",
        agent_query_policy="two_hop_bridge",
        search_score_mode="bm25_plus_bridge_bonus",
        bridge_bonus=1.5,
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
    goc_records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert goc_records
    assert all(r.get("open_calls") <= r.get("open_budget") for r in goc_records)
    assert any(r.get("bridge_opened_in_probe") for r in goc_records)
    assert any(r.get("hop2_executed") and r.get("hop2_query_contains_canonical") for r in goc_records)

    graph_path = tmp_path / "goc_graph.jsonl"
    task_id = goc_records[0].get("task_id")
    lines = []
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if data.get("task_id") == task_id:
            lines.append(data)
    search_events = [line for line in lines if line.get("event_type") == "SEARCH"]
    assert len(search_events) >= 2
    assert all(
        "secondary" not in (evt.get("payload", {}).get("query_id") or "")
        for evt in search_events
    )
    assert any(
        task.canonical_slot_term in evt.get("payload", {}).get("query", "")
        for evt in search_events
        if isinstance(evt.get("payload", {}).get("query"), str)
    )
    open_events = [line for line in lines if line.get("event_type") == "OPEN"]
    assert open_events
    assert all(evt["payload"].get("open_stage") in {"probe", "prompt"} for evt in open_events)
    assert any(evt["payload"].get("open_stage") == "probe" for evt in open_events)
    opened_ids = [evt["payload"].get("clause_id") for evt in open_events if evt["payload"].get("clause_id")]
    assert len(opened_ids) == len(set(opened_ids))

    # reward shaping flag
    reward_args = argparse.Namespace(**vars(compare_args))
    reward_args.use_controller = True
    reward_args.controller_mode = "train"
    reward_args.controller_policy = "bandit"
    reward_args.bridge_reward_bonus = 0.2
    cmd_compare(reward_args)
    train_reports = list((tmp_path / "runs" / "controller_train").glob("*.json"))
    assert train_reports
    reward_report = json.loads(train_reports[-1].read_text())
    reward_records = reward_report.get("records", [])
    assert any(
        rec.get("controller_reward_breakdown") and "r_bridge" in rec["controller_reward_breakdown"]
        for rec in reward_records
    )
