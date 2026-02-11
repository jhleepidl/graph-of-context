# Phase15 TraceOps

Phase15 introduces `traceops_v0`, a long-trace benchmark designed to require:
- Fold (stable history compression)
- Negative-edge forgetting (avoid invalidated branches)
- Unfold revival (recover long-delayed dependencies)

## Levels

| Level | Steps | Pivot checks | Notes |
|---|---:|---:|---|
| L0 | 3 | 1 | explore -> commit -> pivot_check |
| L1 | 5 | 1 | delayed relevance with one pivot |
| L2 | 6 | 2 | two pivot cycles |
| L3 | 9-12 | 2-3 | mixed updates, delay 6-8 |
| L4 | 15-24 | 3-5 | stress test with multi-pivot |

## Headline Metrics

- Primary: `pivot_e3_only_accuracy`
- Secondary: `strict_pivot_accuracy`
- Diagnostics:
  - `pivot_decision_accuracy`
  - `mean_avoid_targets_per_pivot`
  - `avoided_injected_rate`
  - `revive_success_rate`
  - `tokens_pivot_mean`, `tokens_total_mean`

## Main Run

```bash
python scripts/run_phase15_traceops_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --traceops_level 1 \
  --traceops_scenarios mixed
```

## Smoke Example

```bash
python scripts/run_phase15_traceops_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --traceops_level 1 \
  --traceops_scenarios mixed \
  --max_threads 1 \
  --max_steps 6 \
  --smoke
```
