from __future__ import annotations

from policyops.traceops_v0.failure_taxonomy import compare_pair


def test_failure_taxonomy_pairwise_directions_and_equiv_noise_category() -> None:
    common_key = {
        "thread_id": "TR1",
        "step_idx": 2,
    }
    events_a = [
        {
            "task_id": "T1",
            **common_key,
            "e3_answer_correct": True,
            "gold": {"decision": "require_condition", "conditions": [], "evidence_core_ids": ["CEX1"]},
            "pred": {"decision": "require_condition", "conditions": [], "evidence": ["CEX1"]},
            "context_clause_ids": ["CEX2"],
            "context_clauses": [
                {
                    "id": "CEX2",
                    "type": "EXCEPTION",
                    "text": "Footnote exception: residency mismatch can be tolerated with manual review.",
                }
            ],
            "state": {"region": "us", "residency": "eu"},
            "evidence_core_covered_strict": True,
            "evidence_core_covered_equiv": True,
            "evidence_core_id_mismatch_but_equiv_present": False,
            "goc_avoid_target_clause_ids": [],
        },
        {
            "task_id": "T2",
            **common_key,
            "e3_answer_correct": False,
            "gold": {"decision": "allow", "conditions": [], "evidence_core_ids": ["C3"]},
            "pred": {"decision": "deny", "conditions": [], "evidence": []},
            "context_clause_ids": [],
            "context_clauses": [],
            "state": {"region": "us"},
            "evidence_core_covered_strict": False,
            "evidence_core_covered_equiv": False,
            "evidence_core_id_mismatch_but_equiv_present": False,
            "goc_avoid_target_clause_ids": [],
        },
    ]
    events_b = [
        {
            "task_id": "T1",
            **common_key,
            "e3_answer_correct": False,
            "gold": {"decision": "require_condition", "conditions": [], "evidence_core_ids": ["CEX1"]},
            "pred": {"decision": "require_condition", "conditions": [], "evidence": ["CEX2"]},
            "context_clause_ids": ["CEX2"],
            "context_clauses": [
                {
                    "id": "CEX2",
                    "type": "EXCEPTION",
                    "text": "Footnote exception: residency mismatch can be tolerated with manual review.",
                }
            ],
            "state": {"region": "us", "residency": "eu"},
            "evidence_core_covered_strict": False,
            "evidence_core_covered_equiv": True,
            "evidence_core_id_mismatch_but_equiv_present": True,
            "goc_avoid_target_clause_ids": [],
        },
        {
            "task_id": "T2",
            **common_key,
            "e3_answer_correct": True,
            "gold": {"decision": "allow", "conditions": [], "evidence_core_ids": ["C3"]},
            "pred": {"decision": "allow", "conditions": [], "evidence": ["C3"]},
            "context_clause_ids": ["C3"],
            "context_clauses": [],
            "state": {"region": "us"},
            "evidence_core_covered_strict": True,
            "evidence_core_covered_equiv": True,
            "evidence_core_id_mismatch_but_equiv_present": False,
            "goc_avoid_target_clause_ids": [],
        },
    ]

    rows, examples = compare_pair(
        events_a,
        events_b,
        scenario="mixed",
        variant_a="baseline__full",
        variant_b="goc_phase16_depwalk__goc",
    )

    directions = {row["direction"] for row in rows}
    assert "A_correct_B_wrong" in directions
    assert "B_correct_A_wrong" in directions
    categories = {row["category"] for row in rows}
    assert "gold_core_id_mismatch_but_equiv_present" in categories
    assert "A_correct_B_wrong" in examples
    assert "B_correct_A_wrong" in examples
