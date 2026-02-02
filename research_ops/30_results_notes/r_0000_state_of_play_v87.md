# Results Note: r_0000_state_of_play_v87 (sanity)

## Setup
- exp_id: mc_dag_sanity_v87_gating_toolcalls_dupopen_allow_titles2
- date: 2026-02-02
- scope: sanity (tiny sample), multi-commit DAG N=3
- note: Metrics are not filled here because sanity sample is too small and some runs may not finish.

## What sanity traces suggested
- Multi-commit stages appear and advance via `return` (auto-inject disabled in commit-flow).
- Merge gating deadlock was addressed by disabling gating for MERGE (closed-book merge cannot satisfy open_page constraints).
- Remaining pain: tasks are extremely hard/long under heavy stress knobs (noise/chain/doc_repeat/branch_trap + strict gating).
  - Completion can be low when `max_steps` is small, driving apparent “accuracy collapse”.
  - Repeated `return_blocked` loops indicate gating/stressor mismatch rather than purely reasoning failure.

## Likely causes of low accuracy right now
1) Many runs do not reach FINAL/finish due to max_steps and stress → accuracy collapses to near-zero.
2) Search/exploration difficulty is too high so commits become wrong.
3) GoC folding may not trigger enough, causing token tail to balloon.

## Next actions (recommended)
- Implement a “medium difficulty” preset (sp_0002) to separate:
  - easy retrieval (few steps)
  - hard merge (composition, masked refs, closed-book final)
- Re-run N=3 with 30 tasks across GoC/FullHistory/SimilarityOnly.
- Track completion explicitly; only then interpret accuracy differences.

## Paper-facing interpretation
Sanity validated the *mechanics* (multi-commit staging + traceability). Next is to tune difficulty so:
- baselines finish reliably at low N (meaningful comparisons),
- yet FullHistory shows tail-token blowup and degrades as N increases.
