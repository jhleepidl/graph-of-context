from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyops.baselines import DummyClient, run_goc_heuristic
from policyops.run import apply_evidence_padding, cmd_compare
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
    pred_goc, opened_ids, prompt, raw, error, _ = run_goc_heuristic(task, env_goc, DummyClient())
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
        methods=["topk", "full", "goc"],
        evidence_padding_mode="none",
        min_evidence_count=2,
        save_prompts=False,
        save_raw=False,
        use_controller=False,
        controller_mode="off",
        controller_state_path=str(tmp_path / "controller_state.json"),
    )
    cmd_compare(compare_args)
    compare_reports = list((tmp_path / "runs" / "compare").glob("*.json"))
    assert compare_reports

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
        use_controller=True,
        controller_mode="train",
        controller_state_path=str(controller_state),
    )
    cmd_compare(train_args)
    assert controller_state.exists()
    state = json.loads(controller_state.read_text(encoding="utf-8"))
    global_stats = state.get("stats", {}).get("global", {})
    assert len(global_stats.keys()) >= 5
