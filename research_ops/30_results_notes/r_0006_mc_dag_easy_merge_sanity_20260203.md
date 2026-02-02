# Results Note: r_0006_mc_dag_easy_merge_sanity_20260203

**Date:** 2026-02-03

## Setup
- Preset: `configs/mc_dag_easy_merge_n3.json`
- Overrides: `task_limit=1`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config _scratch/mc_dag_easy_merge_sanity_1task.json --out_dir runs/mc_dag_easy_merge_sanity_20260203 --fresh`
- Artifact path: `runs/mc_dag_easy_merge_sanity_20260203/`

## Outcome (GoC, 1 task)
- Completion: 1/1
- Accuracy: 0/1
- Tokens: p50/p90 ~59.1k
- Return blocked: avg/max 1

## Notes
- Sanity run to validate easy-merge preset wiring.
