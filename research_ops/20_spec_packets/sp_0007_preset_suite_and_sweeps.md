# Spec Packet: sp_0007_preset_suite_and_sweeps

## Goal
Create a preset suite (easy/medium/stress) plus sweep configs for N and budget grids.

## Presets
- Easy-Merge: `configs/mc_dag_easy_merge_n3.json`
- Medium: `configs/mc_dag_medium_n3.json` (baseline)
- Stress-Tail: `configs/mc_dag_stress_tail_n3.json`

## Sweeps
- N sweep configs: `configs/sweep_mc_dag_medium_n2.json`, `configs/sweep_mc_dag_medium_n3.json`, `configs/sweep_mc_dag_medium_n5.json`
- Budget grid config: `configs/sweep_mc_dag_medium_budget_grid_n3.json`

## Output organization (run commands)
- N sweep:
  - `python run_sweep.py --config configs/sweep_mc_dag_medium_n2.json --out_dir runs/mc_dag_n_sweep_<date>/n2 --fresh`
  - `python run_sweep.py --config configs/sweep_mc_dag_medium_n3.json --out_dir runs/mc_dag_n_sweep_<date>/n3 --fresh`
  - `python run_sweep.py --config configs/sweep_mc_dag_medium_n5.json --out_dir runs/mc_dag_n_sweep_<date>/n5 --fresh`
- Budget grid:
  - `python run_sweep.py --config configs/sweep_mc_dag_medium_budget_grid_n3.json --out_dir runs/mc_dag_budget_sweep_<date> --fresh`

## Notes
- All configs keep masked refs + closed-book final + commit anchor enforcement.
