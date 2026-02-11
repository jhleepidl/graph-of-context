from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def test_phase18_generator_records_core_necessity_and_hidden_core_metadata() -> None:
    threads, meta = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=123,
        threads=16,
        indirection_rate=1.0,
        trap_distractor_count=4,
        trap_similarity_boost=1.0,
        core_size_min=3,
        core_size_max=4,
        alias_chain_len=2,
        indirect_pivot_style="blended",
        core_necessity_enable=True,
        core_necessity_require_all=True,
        trap_decision_flip_enable=True,
        trap_flip_salience=0.25,
        trap_flip_attach_kind="avoided",
        trap_graph_excludable_rate=1.0,
        trap_graph_excludable_kinds="stale,inapplicable,avoided",
        trap_invalidation_text_strength=0.6,
        hidden_core_enable=True,
        hidden_core_kind="low_overlap_clause",
        hidden_core_link_mode="depends_on",
    )

    assert meta.get("traceops_core_necessity_enable") is True
    assert meta.get("traceops_core_necessity_require_all") is True
    assert meta.get("traceops_trap_decision_flip_enable") is True
    assert float(meta.get("traceops_trap_flip_salience", -1.0)) == 0.25
    assert meta.get("traceops_trap_flip_attach_kind") == "avoided"
    assert float(meta.get("traceops_trap_graph_excludable_rate", -1.0)) == 1.0
    assert meta.get("traceops_trap_graph_excludable_kinds") == "avoided,inapplicable,stale"
    assert float(meta.get("traceops_trap_invalidation_text_strength", -1.0)) == 0.6
    assert meta.get("traceops_hidden_core_enable") is True
    assert meta.get("traceops_hidden_core_kind") == "low_overlap_clause"
    assert meta.get("traceops_hidden_core_link_mode") == "depends_on"

    pivot_steps = [
        step
        for thread in threads
        for step in thread.steps
        if str(step.kind) == "pivot_check"
    ]
    assert pivot_steps

    has_all_required = False
    has_trap_flip = False
    has_hidden_core = False
    for thread in threads:
        clauses = thread.clauses
        for step in thread.steps:
            if str(step.kind) != "pivot_check":
                continue
            md = dict(step.metadata or {})
            assert "core_necessity_enable" in md
            assert "core_necessity_flip_count" in md
            assert "core_necessity_all_required" in md
            assert "core_necessity_failed" in md
            assert "trap_decision_label" in md
            assert "trap_decision_flip" in md
            assert "trap_flip_target_id" in md
            assert "trap_flip_target_kind" in md
            assert "trap_graph_excludable_count" in md
            assert "trap_graph_excludable_ids" in md
            assert "trap_invalidation_attached_to_update" in md
            assert "hidden_core_ids" in md
            assert "hidden_core_parent_ids" in md

            has_all_required = has_all_required or bool(md.get("core_necessity_all_required"))
            has_trap_flip = has_trap_flip or bool(md.get("trap_decision_flip"))

            hidden_ids = list(md.get("hidden_core_ids") or [])
            hidden_parents = set(str(x) for x in (md.get("hidden_core_parent_ids") or []))
            if hidden_ids:
                has_hidden_core = True
            for hid in hidden_ids:
                clause = clauses.get(hid)
                assert clause is not None
                assert clause.metadata.get("hidden_core") is True
                if hidden_parents:
                    assert set(clause.depends_on or []).intersection(hidden_parents)

    assert has_all_required
    assert has_trap_flip
    assert has_hidden_core
