# Spec Packet: sp_0008_easy_merge_autofix_sweep

## Goal
Add an Easy-Merge N=3 sweep config with schema autofix enabled for commit mismatch.

## Scope
- New config: `configs/sweep_easy_merge_n3_autofix_on.json`
- No code changes.

## Constraints
- Autofix only toggled via bench_kwargs (default OFF elsewhere).
- Same methods/stages across GoC, FullHistory, SimilarityOnly.

## Evidence
- `python -m py_compile run_sweep.py`
- 1-task sanity using a temporary override config (not committed)
- Contract check `--strict` on sanity traces
