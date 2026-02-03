# Results Note: r_0007_mc_dag_medium_n5_candidate_inject_sanity_20260203

**Date:** 2026-02-03

## Setup
- Base config: `configs/sweep_mc_dag_medium_n5_autofix_on.json`
- Overrides: `task_limit=1`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config _scratch/sweep_mc_dag_medium_n5_autofix_on_sanity_1task.json --out_dir runs/mc_dag_medium_n5_autofix_on_candidate_inject_sanity_20260203 --fresh`
- Artifact path: `runs/mc_dag_medium_n5_autofix_on_candidate_inject_sanity_20260203/`

## Outcome (GoC, 1 task)
- Completion: 1/1
- Contract: OK (`--strict`)

## Evidence
- Trace includes `candidate_commits_injected` for MERGE and FINAL stages.
- Injected block size within cap (<=900 chars) with truncation flags logged.
