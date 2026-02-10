from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.policyops.run import _build_gold_from_context, _compute_effective_context  # noqa: E402
from src.policyops.schemas import Clause, Document, Gold, Task, World  # noqa: E402
from src.policyops.symbolic_judge import judge_from_opened_clauses  # noqa: E402


def _toy_world_and_task() -> tuple[World, Task]:
    doc = Document(
        doc_id="DOC1",
        doc_type="policy",
        title="Retention",
        published_at="2026-01-01",
        authority="legal",
        jurisdiction=["global"],
        applies_to={},
        sections=[],
    )
    clause_gt = Clause(
        clause_id="C_GT30",
        doc_id="DOC1",
        published_at="2026-01-01",
        authority="legal",
        text="If retention bucket is gt_30, allow.",
        kind="rule",
        slot="retain_logs_90d",
        applies_if={"retention_bucket": ["gt_30"]},
        effect={"decision": "allow"},
        conditions=["retain_gt_30"],
        targets={},
        terms_used=[],
    )
    clause_le = Clause(
        clause_id="C_LE30",
        doc_id="DOC1",
        published_at="2026-01-01",
        authority="legal",
        text="If retention bucket is le_30, deny.",
        kind="rule",
        slot="retain_logs_90d",
        applies_if={"retention_bucket": ["le_30"]},
        effect={"decision": "deny"},
        conditions=["retain_le_30"],
        targets={},
        terms_used=[],
    )
    world = World(documents=[doc], clauses={"C_GT30": clause_gt, "C_LE30": clause_le}, meta={})
    task = Task(
        task_id="T_PIVOT",
        timestamp="2026-02-10T00:00:00Z",
        user_ticket="Please retain logs 90 days.",
        ticket_initial="Please retain logs 90 days.",
        ticket_updated="Correction: retention_days=30",
        pivot_type="retention_flip",
        context={
            "slot": "retain_logs_90d",
            "retention_days": 90,
            "retention_bucket": "gt_30",
        },
        budgets={"tool_call_budget": 1, "open_budget": 2},
        gold=Gold(
            decision="allow",
            conditions=["retain_gt_30"],
            gold_evidence=["C_GT30"],
            gold_evidence_core=["C_GT30"],
            gold_evidence_meta=[],
        ),
    )
    return world, task


def test_pivot_gold_is_computed_from_effective_context() -> None:
    world, task = _toy_world_and_task()
    effective_context = _compute_effective_context(
        task,
        episode_id=3,
        threaded_mode=True,
        thread_state={"initial_context": dict(task.context)},
    )
    assert effective_context["retention_days"] == 30
    assert effective_context["retention_bucket"] == "le_30"

    pivot_gold = _build_gold_from_context(world, effective_context)
    assert pivot_gold.decision == "deny"
    assert pivot_gold.decision != task.gold.decision
    assert "C_LE30" in set(pivot_gold.gold_evidence)


def test_symbolic_judge_context_override_changes_decision() -> None:
    world, task = _toy_world_and_task()
    effective_context = _compute_effective_context(
        task,
        episode_id=3,
        threaded_mode=True,
        thread_state={"initial_context": dict(task.context)},
    )
    opened = ["C_GT30", "C_LE30"]

    original_judge = judge_from_opened_clauses(task, opened, world)
    pivot_judge = judge_from_opened_clauses(task, opened, world, context=effective_context)

    assert original_judge["decision"] == "allow"
    assert pivot_judge["decision"] == "deny"
    assert original_judge["decision"] != pivot_judge["decision"]

