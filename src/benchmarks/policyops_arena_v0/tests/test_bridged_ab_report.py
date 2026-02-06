from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.analysis import analyze_bridged_ab
from policyops.bridged_ab import compute_bridged_ab_slices
from policyops.generator import generate_world_and_tasks
from policyops.run import cmd_compare


def test_bridged_ab_report_has_decision_and_judge_acc(tmp_path: Path) -> None:
    generate_world_and_tasks(
        out_dir=tmp_path,
        seed=3,
        n_docs=8,
        clauses_per_doc=4,
        n_tasks=30,
        scenario_mode="bridged_v1_1",
        bridge_prob=0.8,
        canonical_density=0.95,
        alias_density=0.9,
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
        scenario_mode="bridged_v1_1",
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
    report_path = compare_reports[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    goc_records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert goc_records
    slices = compute_bridged_ab_slices(goc_records)
    judge_acc_values = []
    for a_map in slices.get("cells", {}).values():
        for cell in a_map.values():
            if isinstance(cell.get("judge_acc"), (int, float)):
                judge_acc_values.append(float(cell.get("judge_acc")))
    assert any(val > 0 for val in judge_acc_values)

    goc_md = analyze_bridged_ab(report_path, method="goc")
    goc_text = goc_md.read_text(encoding="utf-8")
    assert "decision_acc" in goc_text
    assert "judge_acc" in goc_text

    goc_base_md = analyze_bridged_ab(report_path, method="goc_base")
    goc_base_text = goc_base_md.read_text(encoding="utf-8")
    assert "decision_acc" in goc_base_text
    assert "judge_acc" in goc_base_text
