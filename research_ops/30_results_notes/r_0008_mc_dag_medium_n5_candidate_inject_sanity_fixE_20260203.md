# Results Note: r_0008_mc_dag_medium_n5_candidate_inject_sanity_fixE_20260203

**Date:** 2026-02-03

## Setup
- Base config: `configs/sweep_mc_dag_medium_n5_autofix_on.json`
- Overrides: `task_limit=1`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config configs/_scratch_n5_candidate_inject_sanity.json --out_dir runs/mc_dag_medium_n5_candidate_inject_sanity_fixE_20260203 --fresh`
- Artifact path: `runs/mc_dag_medium_n5_candidate_inject_sanity_fixE_20260203/`

## Outcome (GoC, 1 task)
- Completion: 1/1
- Accuracy: 1.0
- selected_commit: 3 (not biased to 1)
- Contract: OK (`--strict`)

## Evidence
- Trace includes `candidate_commits_injected` at MERGE and FINAL with candidate indices.
- [CANDIDATE_COMMITS] block present in ACTIVE_CONTEXT.
- No fallback event observed (candidates resolved without fallback).
