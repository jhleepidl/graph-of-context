# Spec Packet: sp_0001_multicommit_dag_stabilization_v84_v87

## Goal
Stabilize Multi-Commit DAG execution so stages advance only via `return`, and no gating deadlocks occur (especially in closed-book MERGE/FINAL). Improve traceability for debugging.

## Non-goals
- Not attempting to improve accuracy yet.
- Not introducing controller/bandit policies yet.

## Fairness constraints (must hold)
- Same stage structure and tool restrictions across methods.
- Any gating rules apply identically across methods.

## Changes implemented (history summary)
- Ensure SUBTASK 1 is visible at the start (no “questionless wandering”).
- Disable auto-inject in commit-flow tasks (advance only via `return`).
- Disable gating in closed-book MERGE (prevent deadlock).
- Align gating counters to open_page tool-call counts (not just uncached fetch).
- Avoid dup_open deadlock when no alternative candidate exists (allow rather than hard-block).
- Enforce commit schema strictly: supporting_titles length must be exactly 2.

## Success criteria
### Trace / logs
- `user_turn_injected` events show `reason="return"` only (no `auto`) for multi-commit tasks.
- No repeated `return_blocked` loops caused by merge-stage gating.
- Commit schema failures produce explicit `return_blocked(reason="multicommit_schema")`.

## Commands
- Sanity preset: `hotpotqa_multicommit_dag_sanity` (task_limit=2).
- Greps:
  - `"user_turn_injected"`
  - `"return_blocked"` with stage_kind
  - `"[FINAL"` marker in prompts

## Risks & rollback
- Risk: disabling auto-inject could reduce compatibility for other multi-turn tasks. Mitigation: guard by task_meta (two_stage/multi_commit).
