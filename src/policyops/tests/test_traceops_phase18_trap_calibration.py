from __future__ import annotations

from policyops.traceops_v0.generator import _jaccard_text, generate_traceops_threads


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
        trap_graph_excludable_kinds="stale,inapplicable,avoided,decision_checkpoint",
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


def test_phase18_forced_topk_includes_most_similar_eligible_trap() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=17,
        threads=8,
        indirection_rate=1.0,
        trap_distractor_count=5,
        trap_similarity_boost=0.9,
        trap_decision_flip_enable=True,
        trap_flip_attach_kind="none",
        trap_graph_excludable_rate=0.0,
        trap_graph_excludable_kinds="stale,inapplicable,avoided,decision_checkpoint",
        trap_graph_force_topk=1,
        trap_graph_force_include_flip_target=False,
        trap_graph_force_include_decision_checkpoint=False,
        hidden_core_enable=True,
    )

    saw_case = False
    for thread in threads:
        for step in thread.steps:
            if str(step.kind) != "pivot_check":
                continue
            md = dict(step.metadata or {})
            trap_ids = [str(cid) for cid in (md.get("trap_distractor_ids") or []) if str(cid)]
            if not trap_ids:
                continue
            eligible = [
                cid
                for cid in trap_ids
                if str(thread.clauses.get(cid).metadata.get("trap_kind", "")).strip().lower()
                in {"stale", "inapplicable", "avoided", "decision_checkpoint"}
            ]
            if not eligible:
                continue
            ranked = sorted(
                (
                    (_jaccard_text(str(step.message or ""), str(thread.clauses[cid].text or "")), str(cid))
                    for cid in eligible
                ),
                key=lambda item: (-item[0], item[1]),
            )
            best_id = str(ranked[0][1])
            forced_ids = [str(cid) for cid in (md.get("trap_graph_excludable_forced_ids") or []) if str(cid)]
            assert forced_ids, "forced ids should not be empty when topk force is enabled"
            assert best_id in forced_ids
            saw_case = True
    assert saw_case


def test_phase18_forced_includes_flip_target_when_eligible() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=19,
        threads=8,
        indirection_rate=1.0,
        trap_distractor_count=5,
        trap_similarity_boost=0.9,
        trap_decision_flip_enable=True,
        trap_flip_attach_kind="avoided",
        trap_graph_excludable_rate=0.0,
        trap_graph_excludable_kinds="stale,inapplicable,avoided,decision_checkpoint",
        trap_graph_force_topk=0,
        trap_graph_force_include_flip_target=True,
        trap_graph_force_include_decision_checkpoint=False,
        trap_require_avoided=True,
        hidden_core_enable=True,
    )

    saw_case = False
    for thread in threads:
        for step in thread.steps:
            if str(step.kind) != "pivot_check":
                continue
            md = dict(step.metadata or {})
            flip_target_id = str(md.get("trap_flip_target_id", "") or "").strip()
            if not flip_target_id:
                continue
            excludable_ids = [str(cid) for cid in (md.get("trap_graph_excludable_ids") or []) if str(cid)]
            forced_ids = [str(cid) for cid in (md.get("trap_graph_excludable_forced_ids") or []) if str(cid)]
            if flip_target_id in excludable_ids:
                assert flip_target_id in forced_ids
                saw_case = True
    assert saw_case


def test_phase18_forced_includes_decision_checkpoint_when_present() -> None:
    threads, _ = generate_traceops_threads(
        level=3,
        scenarios=["indirect"],
        seed=23,
        threads=10,
        indirection_rate=1.0,
        trap_distractor_count=5,
        trap_similarity_boost=0.9,
        trap_decision_flip_enable=True,
        trap_flip_attach_kind="avoided",
        trap_graph_excludable_rate=0.0,
        trap_graph_excludable_kinds="stale,inapplicable,avoided,decision_checkpoint",
        trap_graph_force_topk=0,
        trap_graph_force_include_flip_target=False,
        trap_graph_force_include_decision_checkpoint=True,
        hidden_core_enable=True,
    )

    saw_checkpoint_candidate = False
    saw_forced_checkpoint = False
    for thread in threads:
        for step in thread.steps:
            if str(step.kind) != "pivot_check":
                continue
            md = dict(step.metadata or {})
            checkpoint_ids = [
                str(cid) for cid in (md.get("decision_checkpoint_trap_ids") or []) if str(cid)
            ]
            if not checkpoint_ids:
                continue
            saw_checkpoint_candidate = True
            forced_ids = [str(cid) for cid in (md.get("trap_graph_excludable_forced_ids") or []) if str(cid)]
            excludable_ids = [str(cid) for cid in (md.get("trap_graph_excludable_ids") or []) if str(cid)]
            if any(cid in forced_ids for cid in checkpoint_ids):
                saw_forced_checkpoint = True
                assert any(cid in excludable_ids for cid in checkpoint_ids)
    assert saw_checkpoint_candidate
    assert saw_forced_checkpoint
