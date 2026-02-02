# Spec Packet: sp_0002_mc_dag_medium_preset (config-only)

## Goal
Create a “medium difficulty” Multi-Commit DAG preset that:
1) reduces exploration/loopiness (fewer steps, fewer deadlocks),
2) keeps merge difficulty (hard composition + masked refs + closed-book final),
3) preserves fairness across methods.

This recovers v71-like *reasonable* accuracy while still revealing GoC tail-token advantages.

## Non-goals
- No code changes (preset-only).
- No controller/bandit policy yet.
- No declared-deps ablation work yet.

## Fairness constraints (must hold)
- Same stages for all methods: SUBTASK1..N, MERGE(s), FINAL.
- Same tool restrictions: closed-book final (and optionally closed-book merge), across all methods.
- Same number of LLM calls (stages), across all methods.

## Proposed knobs (starting point)
### Reduce exploration difficulty
- return_gating_min_steps: 8 -> 2 or 3 (SUBTASK only)
- return_gating_min_open_pages: 2 -> 1 (SUBTASK only)
- noise: 6 -> 1 or 2
- chain length: 6 -> 2 or 3
- doc_repeat: 3 -> 1
- branch_trap: 3 -> 1
- lost-in-middle: keep, but reduce extremity

### Keep merge difficulty
- masked refs in merge/final (no direct title reuse)
- final closed-book remains ON
- composition rule in final (commit fact combination / merge winners)

### Ensure runs finish
- max_steps: 80 -> 120 (medium preset only). Later, for “FullHistory collapse” experiments, reduce max_steps or re-add stress.

## Success criteria
- N=3, 30 tasks:
  - completion_rate >= 0.70
  - accuracy improves out of near-zero regime
  - token_p90(GoC) << token_p90(FullHistory)
- Failure mode distribution dominated by merge mistakes (not search loops or max_steps exits).

## Commands
- Run: `python run_sweep.py --preset mc_dag_medium_n3 --out_dir runs/mc_dag_medium_n3 --fresh`
- Compare: GoC, FullHistory, SimilarityOnly (optionally LinearSummary).

## Risks & rollback
- Risk: medium becomes too easy and hides GoC advantages. Mitigation: keep merge hard; later sweep stress knobs upward.
