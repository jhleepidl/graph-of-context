from pathlib import Path
import argparse
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.baselines import DummyClient, run_goc_heuristic
from policyops.run import apply_evidence_padding, cmd_compare, cmd_sweep, cmd_analyze
from policyops.triage import triage_compare
from goc_logger.export import export_dot
from policyops.env import PolicyOpsEnv
from policyops.eval import evaluate_prediction
from policyops.generator import generate_world_and_tasks


def test_eval_smoke(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=2,
        n_docs=3,
        clauses_per_doc=3,
        n_tasks=2,
    )

    env = PolicyOpsEnv(world, tool_call_budget=10, open_budget=3)
    results = env.search("export identifiers", top_k=3)
    assert results
    opened = env.open(results[0]["clause_id"])
    assert "text" in opened

    task = tasks[0]
    pred = {
        "decision": task.gold.decision,
        "conditions": task.gold.conditions,
        "evidence": task.gold.gold_evidence,
    }
    metrics = evaluate_prediction(pred, task.gold, world)
    assert metrics["decision_accuracy"] == 1.0
    assert metrics["condition_f1"] == 1.0
    assert metrics["evidence_precision"] == 1.0
    assert metrics["evidence_recall"] == 1.0
    assert metrics["critical_evidence_hit"] == 1.0

    env_goc = PolicyOpsEnv(world, tool_call_budget=10, open_budget=3)
    pred_goc, opened_ids, prompt, raw, error, _, _ = run_goc_heuristic(task, env_goc, DummyClient())
    assert isinstance(pred_goc, dict)
    assert isinstance(opened_ids, list)
    assert isinstance(prompt, str)
    assert error is None or isinstance(error, str)
    assert pred_goc.get("decision") in {
        "allow",
        "deny",
        "require_condition",
        "needs_more_info",
    }
    if opened_ids:
        eval_pred, record_pred, before, after = apply_evidence_padding(
            pred_goc, opened_ids, mode="schema_only", min_count=2
        )
        assert eval_pred["evidence"] == before
        assert record_pred["evidence"] == after
        assert all(cid in opened_ids for cid in before)
        eval_pred_g, record_pred_g, before_g, after_g = apply_evidence_padding(
            pred_goc, opened_ids, mode="global", min_count=2
        )
        assert all(cid in opened_ids for cid in record_pred_g["evidence"])
        assert len(record_pred_g["evidence"]) >= min(2, len(opened_ids))

    compare_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        dotenv=".env",
        methods=["topk", "full", "goc", "oracle"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=True,
        save_goc_dot=True,
        goc_graph_mode="events+final",
        goc_graph_include_clause_text=False,
        goc_graph_dir=str(tmp_path / "goc_graph.jsonl"),
        goc_graph_schema="v1",
        goc_graph_sample_rate=1.0,
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
        debug_n=1,
        debug_task_ids="",
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports
    report = json.loads(compare_reports[-1].read_text(encoding="utf-8"))
    goc_records = report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert any(r.get("goc_graph_jsonl_path") for r in goc_records)
    oracle_records = report.get("method_reports", {}).get("oracle", {}).get("records", [])
    assert oracle_records
    coverage = oracle_records[0].get("oracle_gold_coverage")
    assert coverage is None or 0.0 <= coverage <= 1.0
    topk_records = report.get("method_reports", {}).get("topk", {}).get("records", [])
    assert topk_records
    diag_record = topk_records[0]
    assert "opened_gold_count" in diag_record
    assert "opened_gold_coverage" in diag_record
    assert 0.0 <= diag_record["opened_gold_coverage"] <= 1.0
    assert "num_search_results" in diag_record
    assert isinstance(diag_record["gold_in_search_topk"], bool)
    if goc_records:
        forced = goc_records[0].get("forced_open_ids")
        assert forced is not None
        assert len(forced) <= 1

    graph_path = tmp_path / "goc_graph.jsonl"
    assert graph_path.exists()
    graph_lines = [json.loads(line) for line in graph_path.read_text(encoding="utf-8").splitlines()]
    assert any(line.get("event_type") == "INIT" for line in graph_lines)
    assert any(line.get("event_type") == "PREDICTION" for line in graph_lines)
    assert any(line.get("event_type") == "DONE" for line in graph_lines)
    assert any(line.get("event_type") == "SNAPSHOT" for line in graph_lines)
    search_event = next(line for line in graph_lines if line.get("event_type") == "SEARCH")
    first_result = search_event["payload"]["results"][0]
    assert first_result.get("kind") is not None
    assert first_result.get("slot") is not None
    assert first_result.get("published_at") is not None
    assert first_result.get("snippet") is not None
    assert len(first_result.get("snippet", "")) <= 200
    open_event = next(line for line in graph_lines if line.get("event_type") == "OPEN")
    assert open_event["payload"].get("selected_by") is not None
    assert open_event["payload"].get("open_index") is not None
    snapshot = next(line for line in graph_lines if line.get("event_type") == "SNAPSHOT")
    nodes = snapshot["payload"]["nodes"]
    edges = snapshot["payload"]["edges"]
    doc_node = next(node for node in nodes if node.get("type") == "doc_ref")
    assert doc_node.get("kind") is not None
    assert doc_node.get("slot") is not None
    assert any(edge.get("type") == "selected_for_prompt" for edge in edges)
    assert any(node.get("type") == "tool_call" for node in nodes)
    assert any(edge.get("type") == "next" for edge in edges) or any(edge.get("type") == "depends_on" for edge in edges)
    dot_path = tmp_path / "goc_graph.dot"
    export_dot(graph_path, tasks[0].task_id, dot_path)
    assert dot_path.exists()
    dot_text = dot_path.read_text(encoding="utf-8").strip()
    assert dot_text
    assert doc_node.get("clause_id") in dot_text
    assert doc_node.get("kind") in dot_text
    assert doc_node.get("slot") in dot_text

    triage_dir = triage_compare(compare_reports[-1], method="goc", max_per_bucket=1)
    assert (triage_dir / "A_open_selection_fail").exists() or (triage_dir / "B_reasoning_fail").exists() or (triage_dir / "C_retrieval_fail").exists() or (triage_dir / "D_decision_confusion").exists()
    dot_files = list(triage_dir.rglob("graph.dot"))
    assert dot_files
    for dot in dot_files:
        text = dot.read_text(encoding="utf-8")
        assert "None/None" not in text
        assert "unknown/unknown" not in text
        assert "update/" in text or "rule/" in text or "exception/" in text
    jsonl_files = list(triage_dir.rglob("graph.jsonl"))
    assert jsonl_files
    for jsonl in jsonl_files:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        assert any('"event_type": "INIT"' in line for line in lines)
        assert any('"event_type": "DONE"' in line for line in lines)

    # Ensure new buckets can be created
    fake_report = {
        "method_reports": {
            "goc": {
                "records": [
                    {
                        "task_id": "T1001",
                        "gold_in_search_topk": True,
                        "opened_gold_coverage": 0.0,
                        "winning_clause_rank": 10,
                        "open_budget": 5,
                        "gold_decision": "allow",
                        "pred_decision": "deny",
                        "opened_has_winning_clause": False,
                        "decision_correct": False,
                        "evidence_before_pad": [],
                        "evidence_after_pad": ["C1"],
                        "gold_evidence_ids": ["C1"],
                    },
                    {
                        "task_id": "T1002",
                        "gold_in_search_topk": False,
                        "opened_gold_coverage": 0.0,
                        "gold_decision": "require_condition",
                        "pred_decision": "allow",
                        "opened_has_winning_clause": True,
                        "decision_correct": False,
                        "evidence_before_pad": [],
                        "evidence_after_pad": ["C2"],
                        "gold_evidence_ids": ["C2"],
                    },
                ]
            }
        }
    }
    fake_path = tmp_path / "fake_compare.json"
    fake_path.write_text(json.dumps(fake_report), encoding="utf-8")
    fake_out = triage_compare(fake_path, method="goc", max_per_bucket=1)
    assert (fake_out / "E_budget_edge_fail").exists()
    assert (fake_out / "F_evidence_padding_artifact").exists()

    controller_dir = tmp_path / "controller_eval"
    generate_world_and_tasks(
        out_dir=controller_dir,
        seed=7,
        n_docs=3,
        clauses_per_doc=3,
        n_tasks=3,
    )
    controller_weights = controller_dir / "runs" / "controller" / "weights.json"
    train_args = argparse.Namespace(
        out_dir=controller_dir,
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
        goc_graph_dir="",
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
        use_query_rewrite=True,
        rewrite_queries=3,
        query_rewrite_mode="expanded",
        merge_rank_fusion="max",
        force_open_top_n=1,
        force_open_source="primary",
        use_controller=True,
        controller_mode="train",
        controller_policy="rerank",
        controller_state_path=str(controller_dir / "controller_state.json"),
        controller_weights_path=str(controller_weights),
        task_split="holdout",
        train_ratio=0.7,
        split_seed=0,
        debug_n=0,
        debug_task_ids="",
    )
    cmd_compare(train_args)

    eval_args = argparse.Namespace(**vars(train_args))
    eval_args.controller_mode = "eval"
    eval_args.save_goc_graph = True
    eval_args.save_goc_dot = True
    eval_args.goc_graph_schema = "v1"
    eval_args.goc_graph_dir = str(controller_dir / "goc_graph_eval.jsonl")
    cmd_compare(eval_args)
    eval_report = json.loads(list((controller_dir / "runs" / "compare").glob("*.json"))[-1].read_text())
    eval_records = eval_report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert eval_records
    graph_path = Path(eval_records[0]["goc_graph_jsonl_path"])
    lines = [json.loads(line) for line in graph_path.read_text().splitlines()]
    search_event = next(line for line in lines if line.get("event_type") == "SEARCH")
    result_item = search_event["payload"]["results"][0]
    assert result_item.get("kind") is not None
    assert result_item.get("slot") is not None
    assert result_item.get("published_at") is not None
    open_event = next(line for line in lines if line.get("event_type") == "OPEN")
    assert open_event["payload"].get("selected_by") is not None
    assert open_event["payload"].get("open_index") is not None
    assert open_event["payload"].get("bm25_rank") is not None
    assert any(
        line.get("event_type") == "OPEN"
        and line["payload"].get("selected_by") in {"controller_rerank", "forced_open"}
        for line in lines
    )
    prompt_event = next(line for line in lines if line.get("event_type") == "PROMPT")
    assert prompt_event["payload"].get("controller_info") is not None

    dot_path = Path(eval_records[0]["goc_graph_dot_path"])
    dot_text = dot_path.read_text(encoding="utf-8")
    assert "rank:" in dot_text and "score:" in dot_text
    assert "None" not in dot_text

    compare_args_zero = argparse.Namespace(**vars(compare_args))
    compare_args_zero.goc_graph_sample_rate = 0.0
    compare_args_zero.goc_graph_dir = str(tmp_path / "goc_graph_zero.jsonl")
    cmd_compare(compare_args_zero)
    report_zero = json.loads(list((tmp_path / "runs" / "compare").glob("*.json"))[-1].read_text(encoding="utf-8"))
    goc_records_zero = report_zero.get("method_reports", {}).get("goc", {}).get("records", [])
    assert all(r.get("goc_graph_jsonl_path") is None for r in goc_records_zero)

    train_dir = tmp_path / "train"
    generate_world_and_tasks(
        out_dir=train_dir,
        seed=3,
        n_docs=3,
        clauses_per_doc=3,
        n_tasks=5,
    )
    controller_state = train_dir / "controller_state.json"
    train_args = argparse.Namespace(
        out_dir=train_dir,
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
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        save_goc_dot=False,
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
        use_query_rewrite=True,
        rewrite_queries=3,
        query_rewrite_mode="expanded",
        merge_rank_fusion="max",
        force_open_top_n=1,
        force_open_source="primary",
        use_controller=True,
        controller_mode="train",
        controller_policy="bandit",
        controller_state_path=str(controller_state),
        controller_weights_path="",
        task_split="holdout",
        train_ratio=0.7,
        split_seed=0,
        debug_n=0,
        debug_task_ids="",
    )
    cmd_compare(train_args)
    assert controller_state.exists()
    state = json.loads(controller_state.read_text(encoding="utf-8"))
    global_stats = state.get("stats", {}).get("global", {})
    assert len(global_stats.keys()) >= 5
    report_train = json.loads(list((train_dir / "runs" / "compare").glob("*.json"))[-1].read_text(encoding="utf-8"))
    assert report_train["num_train_tasks"] + report_train["num_eval_tasks"] == 5

    sweep_args = argparse.Namespace(
        out_dir=tmp_path,
        seeds=[0],
        open_budgets=[3],
        n_docs=3,
        n_tasks=5,
        methods=["topk", "goc"],
        model="dummy",
        llm="dummy",
        dotenv=".env",
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        save_goc_dot=False,
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
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
        task_split="holdout",
        train_ratio=0.7,
        split_seed=0,
        reuse_data=False,
        debug_n=0,
        debug_task_ids="",
    )
    cmd_sweep(sweep_args)
    sweeps = list((tmp_path / "runs" / "sweeps").glob("*"))
    assert sweeps
    summary_csv = list(sweeps[-1].glob("summary.csv"))
    assert summary_csv
    header = summary_csv[0].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "opened_gold_coverage_mean" in header
    assert "gold_in_search_topk_rate" in header
    assert "winning_clause_rank_mean" in header
    assert "min_gold_rank_mean" in header
    assert "gold_score_gap_mean" in header

    analyze_args = argparse.Namespace(
        report=str(compare_reports[-1]),
        k=20,
    )
    cmd_analyze(analyze_args)
    analysis_reports = list((tmp_path / "runs" / "analysis").glob("*_failure_slice.md"))
    assert analysis_reports

    engine_args = argparse.Namespace(
        out_dir=tmp_path,
        model="dummy",
        llm="dummy",
        dotenv=".env",
        methods=["engine"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        save_goc_dot=False,
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
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
    cmd_compare(engine_args)
    engine_report = json.loads(
        list((tmp_path / "runs" / "compare").glob("*.json"))[-1].read_text(encoding="utf-8")
    )
    engine_summary = engine_report.get("summary", {}).get("engine", {})
    assert engine_summary.get("decision_accuracy") == 1.0

    rerank_dir = tmp_path / "rerank"
    generate_world_and_tasks(
        out_dir=rerank_dir,
        seed=6,
        n_docs=3,
        clauses_per_doc=3,
        n_tasks=5,
    )
    rerank_weights = rerank_dir / "runs" / "controller" / "weights.json"
    rerank_args = argparse.Namespace(
        out_dir=rerank_dir,
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
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        save_goc_dot=False,
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
        use_query_rewrite=True,
        rewrite_queries=3,
        query_rewrite_mode="expanded",
        merge_rank_fusion="max",
        force_open_top_n=1,
        force_open_source="primary",
        use_controller=True,
        controller_mode="train",
        controller_policy="rerank",
        controller_state_path=str(rerank_dir / "controller_state.json"),
        controller_weights_path=str(rerank_weights),
        task_split="holdout",
        train_ratio=0.7,
        split_seed=0,
        debug_n=0,
        debug_task_ids="",
    )
    cmd_compare(rerank_args)
    assert rerank_weights.exists()
    rerank_report = json.loads(list((rerank_dir / "runs" / "compare").glob("*.json"))[-1].read_text())
    goc_records = rerank_report.get("method_reports", {}).get("goc", {}).get("records", [])
    assert any(r.get("rerank_used") for r in goc_records)


def test_dummy_model_mismatch_raises(tmp_path: Path) -> None:
    world, tasks, _, _ = generate_world_and_tasks(
        out_dir=tmp_path,
        seed=5,
        n_docs=3,
        clauses_per_doc=3,
        n_tasks=2,
    )
    assert world and tasks
    args = argparse.Namespace(
        out_dir=tmp_path,
        model="gpt-4.1-mini",
        llm="dummy",
        dotenv=".env",
        methods=["topk"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        save_search_snapshot=False,
        search_snapshot_k=20,
        primary_search_top_k=20,
        save_goc_graph=False,
        goc_graph_mode="events",
        goc_graph_include_clause_text=False,
        goc_graph_dir="",
        save_goc_dot=False,
        goc_graph_schema="v0",
        goc_graph_sample_rate=1.0,
        use_query_rewrite=True,
        rewrite_queries=3,
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
    with pytest.raises(RuntimeError):
        cmd_compare(args)
