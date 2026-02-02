# Results Note: r_0005_mc_dag_medium_n3_20260203_a

**Date:** 2026-02-03

## Setup
- Preset: `configs/mc_dag_medium_n3.json`
- Command: `python run_sweep.py --preset mc_dag_medium_n3 --out_dir runs/mc_dag_medium_n3_20260203_a --fresh`
- Artifact path: `runs/mc_dag_medium_n3_20260203_a/`
- Contract check: `runs/mc_dag_medium_n3_20260203_a/contract_check.json` (ok)

## Summary (30 tasks)
| method | completion | accuracy | token_p50 | token_p90 |
|---|---:|---:|---:|---:|
| GoC | 0.967 | 0.300 | 61,762.5 | 266,574.1 |
| FullHistory | 0.933 | 0.233 | 59,956.5 | 616,104.7 |
| SimilarityOnly | 0.933 | 0.233 | 59,299.5 | 661,884.0 |

## Observations
- Completion target met (≥70%).
- GoC token p90 << FullHistory/SimilarityOnly (tail control visible).
- Failures dominated by max_steps_exit (GoC: 1, FH: 2, SO: 2).
- Contract checker: no auto-inject; markers present (SUBTASK/MERGE/FINAL).

## Debug Packet
- summary_min.json, evidence.txt, traces_samples.zip, contract_check.json are present under the run folder.
