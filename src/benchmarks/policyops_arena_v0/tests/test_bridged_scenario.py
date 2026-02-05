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
    world, tasks, _, tasks_path = generate_world_and_tasks(
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

    # Force at least one hop2 skip by removing probe budget in the first task.
    task_lines = tasks_path.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(task_lines[0])
    first["budgets"]["open_budget"] = 0
    task_lines[0] = json.dumps(first)
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")

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
        methods=["goc", "goc_base"],
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
    assert any(r.get("extracted_facets") for r in goc_records)
    assert all("bridge_needed" in r for r in goc_records)
    assert all("bridge_opened_any" in r for r in goc_records)
    assert all("bridge_opened_gold" in r for r in goc_records)
    assert all("hop2_candidate_query" in r for r in goc_records)
    assert all("hop2_skip_reason" in r for r in goc_records)
    zero_budget = [r for r in goc_records if r.get("open_budget") == 0]
    assert zero_budget
    assert all(r.get("hop2_executed") in {False, 0} for r in zero_budget)
    assert all(r.get("hop2_skip_reason") for r in zero_budget)
    assert any(r.get("hop2_executed") and (r.get("open_from_hop2_count") or 0) > 0 for r in goc_records)

    # Regression guard: diag should not overwrite the gold bridge clause id
    rec_for_task = next((r for r in goc_records if r.get("task_id") == task.task_id), None)
    assert rec_for_task is not None
    assert rec_for_task.get("bridge_clause_id") == task.bridge_clause_id
    probe_id = rec_for_task.get("bridge_probe_clause_id")
    if probe_id:
        opened_total = rec_for_task.get("opened_total_clause_ids") or rec_for_task.get("opened_ids") or []
        assert probe_id in opened_total

    graph_path = tmp_path / "goc_graph.jsonl"
    graph_record = next((r for r in goc_records if (r.get("open_budget") or 0) > 0), goc_records[0])
    task_id = graph_record.get("task_id")
    lines = []
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if data.get("task_id") == task_id:
            lines.append(data)
    search_events = [line for line in lines if line.get("event_type") == "SEARCH"]
    rec_for_graph = next((r for r in goc_records if r.get("task_id") == task_id), None)
    assert rec_for_graph is not None
    if rec_for_graph.get("hop2_executed"):
        assert len(search_events) == 2
    else:
        assert len(search_events) == 1
    assert all(
        all(tag not in (evt.get("payload", {}).get("query_id") or "") for tag in ["secondary", "hybrid"])
        for evt in search_events
    )
    if rec_for_graph.get("hop2_executed"):
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

    metrics = report.get("method_reports", {}).get("goc", {}).get("metrics", {})
    bridged_ab = metrics.get("bridged_ab_slices")
    assert bridged_ab and bridged_ab.get("cells") is not None
    axes = bridged_ab.get("axes", {})
    assert "A" in axes and "B" in axes
    assert "A_unknown" in axes["A"]
    assert "B0_no_hop2" in axes["B"]
    unknown_n = sum(
        cell.get("n", 0)
        for cell in bridged_ab.get("cells", {}).get("A_unknown", {}).values()
    )
    assert unknown_n <= 2
    assert any("hop2_query_contains_gold_canonical" in r for r in goc_records)
    assert all("bridge_probe_contains_gold_canonical" in r for r in goc_records)
    assert all("bridge_opened_contains_gold_canonical" in r for r in goc_records)
    for rec in goc_records:
        if rec.get("hop2_executed"):
            union = rec.get("gold_in_search_topk_union")
            hop1 = rec.get("gold_in_search_topk_hop1")
            hop2 = rec.get("gold_in_search_topk_hop2")
            if isinstance(union, bool) and isinstance(hop1, bool) and isinstance(hop2, bool):
                assert union >= (hop1 or hop2)
    goc_base_report = report.get("method_reports", {}).get("goc_base", {})
    goc_base_records = goc_base_report.get("records", [])
    assert goc_base_records
    goc_base_metrics = goc_base_report.get("metrics", {})
    bridged_ab_base = goc_base_metrics.get("bridged_ab_slices", {})
    assert bridged_ab_base.get("n_records") == len(goc_base_records)
    assert all(r.get("hop2_skip_reason") for r in goc_base_records)
    assert all(r.get("gold_in_search_topk_hop2") in {False, 0} for r in goc_base_records)
    assert all(
        r.get("gold_in_search_topk_union") == r.get("gold_in_search_topk_hop1")
        for r in goc_base_records
    )
    assert all(
        r.get("winning_clause_rank_union") == r.get("winning_clause_rank_hop1")
        for r in goc_base_records
        if r.get("winning_clause_rank_hop1") is not None
    )


def test_bridged_mix_canonical_ticket_rate(tmp_path: Path) -> None:
    _, tasks, _, tasks_path = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=1,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=20,
        scenario_mode="bridged_v1_1",
        bridged_mix_canonical_in_ticket_rate=0.5,
    )
    assert sum(1 for t in tasks if not t.bridge_clause_id) >= 5

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
        save_goc_graph=False,
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
    report = json.loads(list((tmp_path / "runs" / "compare").glob("*.json"))[-1].read_text())
    goc_metrics = report.get("method_reports", {}).get("goc", {}).get("metrics", {})
    bridged_ab = goc_metrics.get("bridged_ab_slices", {})
    a_cells = bridged_ab.get("cells", {}).get("A0_no_bridge_needed", {})
    assert sum(cell.get("n", 0) for cell in a_cells.values()) > 0
    records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert any(r.get("hop2_executed") is False for r in records)


def test_open_split_mode_hop2(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=4,
        n_docs=6,
        clauses_per_doc=4,
        n_tasks=6,
        scenario_mode="bridged_v1_1",
        bridge_prob=1.0,
    )
    base_args = argparse.Namespace(
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
        save_goc_graph=False,
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
        force_open_top_n=0,
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
        hop1_query_mode="stripped",
        open_split_mode="split_hop1_hop2",
        open_split_hop1=0,
    )
    cmd_compare(base_args)
    report = json.loads(list((tmp_path / "runs" / "compare").glob("*.json"))[-1].read_text())
    goc_records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert any(
        r.get("hop2_executed") and (r.get("open_from_hop2_count") or 0) > 0 for r in goc_records
    )

    args_hop1 = argparse.Namespace(**vars(base_args))
    args_hop1.open_split_hop1 = 5
    cmd_compare(args_hop1)
    report2 = json.loads(list((tmp_path / "runs" / "compare").glob("*.json"))[-1].read_text())
    goc_records2 = report2.get("method_reports", {}).get("goc", {}).get("records", [])
    assert all(
        (r.get("open_from_hop2_count") or 0) == 0 for r in goc_records2 if r.get("hop2_executed")
    )
