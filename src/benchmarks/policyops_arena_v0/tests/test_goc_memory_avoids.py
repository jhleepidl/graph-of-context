from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.memory import GoCMemory


def test_dep_closure_excludes_avoided_node() -> None:
    mem = GoCMemory(budget_active=1024, budget_unfold=1024, unfold_k=4, max_dep_depth=3)
    old_id = mem.add_node(thread="main", kind="msg", text="A old requirement")
    pivot_id = mem.add_node(thread="main", kind="msg", text="B pivot requirement")
    assert mem.add_edge("depends", pivot_id, old_id)
    assert mem.add_avoids_edge(pivot_id, old_id)

    closure = mem._dep_closure([pivot_id], depth=3)  # noqa: SLF001 - intentional white-box test
    assert pivot_id in closure
    assert old_id not in closure


def test_unfold_with_seed_ids_respects_avoids_filter() -> None:
    mem = GoCMemory(budget_active=1024, budget_unfold=1024, unfold_k=2, max_dep_depth=3)
    old_id = mem.add_node(thread="main", kind="msg", text="A old requirement")
    pivot_id = mem.add_node(thread="main", kind="msg", text="B pivot requirement")
    assert mem.add_edge("depends", pivot_id, old_id)
    assert mem.add_avoids_edge(pivot_id, old_id)

    mem.storage = [old_id, pivot_id]
    mem._storage_index_add([old_id, pivot_id], force_flush=True)  # noqa: SLF001 - intentional white-box test

    activated = mem.unfold_with_seed_ids("pivot", [pivot_id], k=1)
    assert pivot_id in activated
    assert old_id not in activated
