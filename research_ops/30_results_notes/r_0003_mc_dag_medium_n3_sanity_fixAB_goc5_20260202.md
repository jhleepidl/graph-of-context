# Results Note: r_0003_mc_dag_medium_n3_sanity_fixAB_goc5_20260202

**Date:** 2026-02-02

## Setup
- Preset: `configs/mc_dag_medium_n3.json`
- Overrides: `task_limit=5`, `max_steps=120`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config /tmp/mc_dag_medium_n3_sanity_fixAB_goc5.json --out_dir runs/mc_dag_medium_n3_sanity_fixAB_20260202_goc5 --resume`
- Run id: `d648d9c3e886`
- Artifact path: `runs/mc_dag_medium_n3_sanity_fixAB_20260202_goc5/`

## Outcome (GoC, 5 tasks)
- Completion: 4/5 (0.8)
- Accuracy: 1/5 (0.2)
- Tokens: p50 ~65.9k, p90 ~764.3k
- Return blocked: avg 29.4, max 107
- Failures: 1 max_steps_exit

## Notes
- MERGE/FINAL reached in completed tasks.
- Schema-blocked loops reduced vs pre-fix, but still heavy on one task (max_return_blocked=107).
