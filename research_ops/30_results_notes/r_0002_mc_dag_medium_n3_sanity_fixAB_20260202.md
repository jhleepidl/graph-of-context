# Results Note: r_0002_mc_dag_medium_n3_sanity_fixAB_20260202

**Date:** 2026-02-02

## Setup
- Preset: `configs/mc_dag_medium_n3.json`
- Sanity overrides (local): `task_limit=1`, `max_steps=120`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config /tmp/mc_dag_medium_n3_sanity_fixA_fixB.json --out_dir runs/mc_dag_medium_n3_sanity_fixAB_20260202_goc1 --fresh`
- Run id: `b353133aa2a5`
- Artifact path: `runs/mc_dag_medium_n3_sanity_fixAB_20260202_goc1/`

## Outcome (GoC, 1 task)
- Completion: 1/1 (MERGE/FINAL reached)
- Accuracy: 0/1
- Tokens (total): ~196k
- Return blocked: avg 14 (schema/gating corrections injected)

## Notes
- Fix A (correction injection on return_gating/multicommit_schema) reduced schema-blocked loops vs prior run.
- Fix B (SUBTASK1 injected as follow-up) keeps current user prompt aligned with active subtask.
