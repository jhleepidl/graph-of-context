from __future__ import annotations

from policyops.traceops_v0.generator import generate_traceops_threads


def test_phase18_graph_excludable_traps_are_attached_to_update_avoid_targets() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=9,
        threads=6,
        indirection_rate=1.0,
        trap_distractor_count=4,
        trap_similarity_boost=0.9,
        core_size_min=2,
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
    )

    saw_excludable = False
    saw_attached_hint = False
    saw_avoided_target_kind = False
    for thread in threads:
        for step in thread.steps:
            if str(step.kind) != "pivot_check":
                continue
            md = dict(step.metadata or {})
            excludable_ids = [str(cid) for cid in (md.get("trap_graph_excludable_ids") or []) if str(cid)]
            if not excludable_ids:
                continue
            saw_excludable = True
            if str(md.get("trap_flip_target_kind", "")) == "avoided":
                saw_avoided_target_kind = True

            prior_updates = [
                s
                for s in thread.steps
                if int(s.step_idx) < int(step.step_idx) and str(s.kind) == "update"
            ]
            assert prior_updates, "expected at least one update step before pivot"
            attached = any(
                set(excludable_ids).issubset(set(str(cid) for cid in (u.avoid_target_ids or [])))
                for u in prior_updates
            )
            assert attached, "graph-excludable traps must appear in a prior update avoid_target_ids"

            if bool(md.get("trap_invalidation_attached_to_update", False)):
                for u in prior_updates:
                    for cid in list(u.introduced_clause_ids or []):
                        clause = thread.clauses.get(str(cid))
                        if clause is None or str(clause.node_type or "") != "UPDATE":
                            continue
                        text = str(clause.text or "")
                        if "supersedes earlier notes" in text or "not controlling under current state" in text:
                            saw_attached_hint = True

    assert saw_excludable
    assert saw_avoided_target_kind
    assert saw_attached_hint

