from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.memory import GoCMemory


def test_keep_score_summary_and_title_above_low() -> None:
    mem = GoCMemory(
        budget_active=10_000,
        fold_soft_anchor=True,
        fold_keep_recent_steps=0,
    )
    mem.record_msg("plain low utility node")
    low_id = mem.active[-1]
    mem.record_msg("title keyed node")
    title_id = mem.active[-1]
    mem.nodes[title_id].docids.append("TITLE:Alpha Node")
    mem.record_summary("compact important summary")
    summary_id = mem.active[-1]

    scores = mem._compute_keep_scores([low_id, title_id, summary_id], active_set=set(mem.active))
    assert scores[summary_id] > scores[title_id] > scores[low_id]


def test_keep_score_depends_from_recent_boosts_node() -> None:
    mem = GoCMemory(
        budget_active=10_000,
        fold_soft_anchor=True,
        fold_keep_recent_steps=1,
    )
    mem.record_msg("older low node")
    low_id = mem.active[-1]
    mem.record_msg("target node to be referenced")
    target_id = mem.active[-1]
    _filler = mem.record_msg("filler node")
    mem.record_msg("most recent node")
    recent_id = mem.active[-1]
    mem.add_edge("depends", recent_id, target_id)

    scores = mem._compute_keep_scores([low_id, target_id], active_set=set(mem.active))
    assert scores[target_id] > scores[low_id]


def test_soft_anchor_fold_protects_high_keep_nodes_and_traces_stats() -> None:
    mem = GoCMemory(
        budget_active=10_000,
        budget_unfold=300,
        fold_method="mincut",
        fold_soft_anchor=True,
        fold_keep_recent_steps=0,
        fold_keep_score_threshold=0.8,
        fold_window_max=20,
        fold_max_chunk=8,
        storage_segment_size=2,
    )

    mem.record_msg("old low utility details " * 90)
    low_id = mem.active[-1]
    mem.record_summary("important persistent summary facts " * 40)
    summary_id = mem.active[-1]
    mem.record_msg("new filler details " * 80)
    mem.drain_events()
    mem.budget_active = 800
    mem._maybe_fold_budget()

    fold_events = [ev for ev in mem.drain_events() if isinstance(ev, dict) and ev.get("type") == "fold"]
    assert fold_events, "expected at least one fold event"
    assert any(int(ev.get("high_keep_protected_count", 0)) >= 1 for ev in fold_events)
    assert all(bool(ev.get("fold_soft_anchor")) for ev in fold_events)
    assert all(summary_id not in (ev.get("chosen_chunk") or []) for ev in fold_events)
    assert low_id in mem.storage
