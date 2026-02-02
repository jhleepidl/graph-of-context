# Results Note: r_0001_mc_dag_medium_n3_sanity_20260202

**Date:** 2026-02-02

## Setup
- Preset: `configs/mc_dag_medium_n3.json`
- Sanity overrides (local): `task_limit=1`, `max_steps=60`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config /tmp/mc_dag_medium_n3_sanity.json --out_dir runs/mc_dag_medium_n3_sanity_20260202_goc1 --fresh`
- Run id: `a22bb5667e0d`
- Artifact path: `runs/mc_dag_medium_n3_sanity_20260202_goc1/a22bb5667e0d/`

## Outcome (GoC, 1 task)
- Completion: 0/1 (max_steps reached, no finish)
- Accuracy: 0/1
- Tokens (total): ~334k
- Return gating: `return_blocked` 41 (subtask gating 2)
- Stage markers: SUBTASK present; MERGE/FINAL not reached

## Contract checker
- Issue: trace prompt content stored under `messages`, not `prompt/user_prompt`.
- Fix applied to `scripts/check_multicommit_contract.py` to scan `messages[*].content`.
- Contract check now passes for this run (no auto-inject, subtask markers present).

## Decision needed
- Sanity run failed to finish at `max_steps=60`. Should we:
  1) Run the 30-task experiment with full `max_steps=120` unchanged,
  2) Adjust medium knobs further (e.g., reduce noise/chain or loosen return gating), or
  3) Do a second 1-task sanity run with full `max_steps=120` before scaling?
