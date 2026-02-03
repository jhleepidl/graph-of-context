# Spec Packet: sp_0009_medium_n_sweep_autofix_on

## Goal
Run medium N-sweep with schema_autofix_commit_mismatch enabled to reduce schema-loop tail noise.

## Scope
- New configs (autofix ON):
  - `configs/sweep_mc_dag_medium_n2_autofix_on.json`
  - `configs/sweep_mc_dag_medium_n3_autofix_on.json`
  - `configs/sweep_mc_dag_medium_n5_autofix_on.json`
- No code changes.

## Evidence
- `python -m py_compile run_sweep.py`
- 1-task sanity using a temporary override config (not committed)
- Contract check `--strict` on sanity traces
