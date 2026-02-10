# Phase 14 Pivot/Avoids Run Notes

This repo now supports two new controls for PolicyOps threaded pivot runs:

- `--pivot_message_style {transcript,banner}`
- `--goc_enable_avoids` / `--no_goc_enable_avoids`

Defaults:

- `pivot_message_style=transcript`
- `goc_enable_avoids=true`

## Full bundle run (Phase 13 entrypoint)

```bash
python scripts/run_phase13_e2e_universe_frontier_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --pivot_message_style transcript \
  --goc_enable_avoids
```

To reproduce legacy pivot prompt behavior:

```bash
python scripts/run_phase13_e2e_universe_frontier_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --pivot_message_style banner \
  --goc_enable_avoids
```

To disable avoids filtering:

```bash
python scripts/run_phase13_e2e_universe_frontier_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --pivot_message_style transcript \
  --no_goc_enable_avoids
```

## Direct compare run (manual)

```bash
python -m policyops.run compare \
  --preset threaded_v1_3_fu_decoy_calib_jitter_n10 \
  --methods goc \
  --llm openai \
  --model gpt-4.1-mini \
  --judge symbolic_packed \
  --pivot_message_style transcript \
  --goc_enable_avoids \
  --out_dir runs/phase14_compare \
  --dotenv .env
```
