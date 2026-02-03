from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyops.baselines import DummyClient, run_goc_heuristic
from policyops.run import apply_evidence_padding
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

    pred_goc, opened_ids, prompt, raw, error = run_goc_heuristic(task, env, DummyClient())
    assert isinstance(pred_goc, dict)
    assert isinstance(opened_ids, list)
    assert isinstance(prompt, str)
    assert error is None or isinstance(error, str)
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
