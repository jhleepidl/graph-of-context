# Phase17 TraceOps

Phase17 focuses on scenario completeness, not budget tuning:

- indirect pivots (ordinal/alias/blended references)
- lexical trap distractors (high overlap but stale/inapplicable/avoided)
- compositional gold core (2-4 clauses across multiple node types)

Primary evaluation still uses the same shared dataset across all methods.

## Recommended Metrics

- `pivot_e3_only_accuracy`
- `strict_pivot_accuracy`
- `mean_indirection_overlap_gold`
- `mean_trap_gap`
- `trap_present_rate`
- `mean_core_size`
- pairwise failure taxonomy in `quick_access/failure_taxonomy.csv`

## Smoke Run

```bash
PYTHONPATH=src python scripts/run_phase17_traceops_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --traceops_threads 10 \
  --traceops_llm_max_pivots 10
```

## Main Run

```bash
PYTHONPATH=src python scripts/run_phase17_traceops_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --traceops_threads 80 \
  --traceops_llm_max_pivots 200
```
