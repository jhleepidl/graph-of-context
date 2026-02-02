# Results Note: r_0004_mc_dag_medium_n3_sanity_fixC_goc5_20260202

**Date:** 2026-02-02

## Setup
- Preset: `configs/mc_dag_medium_n3.json`
- Overrides: `task_limit=5`, `max_steps=120`, `methods=["GoC"]`, `parallel_tasks=1`
- Command: `python run_sweep.py --config /tmp/mc_dag_medium_n3_sanity_fixC_goc5.json --out_dir runs/mc_dag_medium_n3_sanity_fixC_20260202_goc5 --resume`
- Run id: `d648d9c3e886`
- Artifact path: `runs/mc_dag_medium_n3_sanity_fixC_20260202_goc5/`

## Outcome (GoC, 5 tasks)
- Completion: 5/5 (1.0)
- Accuracy: 1/5 (0.2)
- Tokens: p50 ~58.3k, p90 ~120.6k
- Return blocked: avg 3.6, max 12
- Failures: none

## Notes
- Fix C eliminated correction spam; token tail collapsed well below 200k.
- MERGE/FINAL consistently reached.
