from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.memory import GoCMemory


def test_goc_snapshot_schema_and_fold_unfold_event_fields() -> None:
    mem = GoCMemory(
        budget_active=60,
        budget_unfold=400,
        unfold_k=2,
        fold_method="greedy",
        fold_max_chunk=4,
        fold_window_max=12,
        storage_segment_size=2,
    )

    mem.record_msg("alpha_anchor " * 80)
    mem.record_msg("rareterm_snapshot " * 80)
    mem.record_msg("beta_context " * 80)

    fold_events = [ev for ev in mem.drain_events() if isinstance(ev, dict) and ev.get("type") == "fold"]
    assert fold_events, "expected at least one fold event"
    fold_ev = fold_events[-1]
    for key in (
        "chosen_chunk",
        "proxy_id",
        "removed_token_est",
        "cut_edges_count",
        "fold_policy",
        "fold_method",
    ):
        assert key in fold_ev

    assert mem.storage, "expected folded nodes in storage"

    _ = mem.unfold("rareterm_snapshot", k=2)
    unfold_events = [ev for ev in mem.drain_events() if isinstance(ev, dict) and ev.get("type") == "unfold"]
    assert unfold_events, "expected an unfold event"
    unfold_ev = unfold_events[-1]
    for key in (
        "query",
        "chosen_seed_ids",
        "chosen_nodes_count",
        "activated",
        "used_tokens_est",
        "candidate_count",
    ):
        assert key in unfold_ev

    snap = mem.snapshot()
    for key in (
        "global_step",
        "budget_active",
        "budget_unfold",
        "active_tokens",
        "active",
        "storage",
        "nodes",
        "edges",
    ):
        assert key in snap

    assert isinstance(snap["active"], list)
    assert isinstance(snap["storage"], list)
    assert isinstance(snap["nodes"], dict)
    assert isinstance(snap["edges"], dict)
    for etype in ("depends", "depends_llm", "doc_ref", "seq"):
        assert etype in snap["edges"]

    assert snap["nodes"], "expected non-empty node snapshot"
    any_node = next(iter(snap["nodes"].values()))
    for key in (
        "kind",
        "thread",
        "step_idx",
        "token_len",
        "docids",
        "ttl",
        "text_preview",
        "has_storage_text",
    ):
        assert key in any_node
