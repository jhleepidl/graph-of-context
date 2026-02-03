# Results Note: r_0009_mc_dag_medium_n5_candidate_key_sanity_20260203

**Date:** 2026-02-03

## Setup
- Base config: `configs/sweep_mc_dag_medium_n5_autofix_on.json`
- Overrides: `task_limit=1`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config configs/_scratch_n5_candidate_key_sanity.json --out_dir runs/mc_dag_medium_n5_candidate_key_sanity_20260203 --fresh`
- Artifact path: `runs/mc_dag_medium_n5_candidate_key_sanity_20260203/`

## Outcome (GoC, 1 task)
- Completion: 1/1
- Contract: OK (`--strict`)

## Evidence
- Trace includes [CANDIDATE_COMMITS] with `key=...` and MERGE prompt instructs key-only comparison.
- `candidate_commits_injected` events include `keys` and `key_truncated` fields.
