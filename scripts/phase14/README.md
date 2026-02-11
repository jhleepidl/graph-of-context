# Phase 14 Entity-Switch Run Notes

Phase14 keeps the Phase13 evaluation stack and adds relationship-driven E3 GoC selection:

- `--pivot_message_style {transcript,banner}`
- `--pivot_gold_mode {respect_ticket_updated,original,both}`
- `--goc_enable_avoids` / `--no_goc_enable_avoids`
- `--goc_avoids_mode {applicability,legacy_commit,off}`
- `--goc_applicability_seed_enable`
- `--goc_applicability_seed_topk`
- `--goc_dependency_closure_enable`
- `--goc_dependency_closure_topk`
- `--goc_dependency_closure_hops`
- `--goc_dependency_closure_universe {candidates,world,memory_opened}`

Defaults:

- `pivot_message_style=transcript`
- `pivot_gold_mode=respect_ticket_updated`
- `goc_enable_avoids=true`
- `goc_avoids_mode=applicability`
- `goc_applicability_seed_enable=true`
- `goc_applicability_seed_topk=8`
- `goc_dependency_closure_enable=true`
- `goc_dependency_closure_topk=12`
- `goc_dependency_closure_hops=1`
- `goc_dependency_closure_universe=candidates`

Metric note:

- Primary late-pivot headline: `e3_pivot_e3_only_accuracy`
- Secondary strict pipeline metric: `strict_final_pivot_accuracy`
- Diagnostic: `critical_coverage_pivot_rate`

## Phase14 bundle run (entity_switch focus)

```bash
python scripts/run_phase14_e2e_entity_switch_bundle.py \
  --dotenv .env \
  --model gpt-4.1-mini \
  --pivot_message_style transcript \
  --pivot_gold_mode respect_ticket_updated \
  --goc_enable_avoids \
  --goc_avoids_mode applicability \
  --goc_applicability_seed_enable \
  --goc_applicability_seed_topk 8 \
  --goc_dependency_closure_enable \
  --goc_dependency_closure_topk 12 \
  --goc_dependency_closure_hops 1 \
  --goc_dependency_closure_universe candidates
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
  --pivot_gold_mode respect_ticket_updated \
  --goc_enable_avoids \
  --goc_avoids_mode applicability \
  --goc_applicability_seed_enable \
  --goc_applicability_seed_topk 8 \
  --goc_dependency_closure_enable \
  --goc_dependency_closure_topk 12 \
  --goc_dependency_closure_hops 1 \
  --goc_dependency_closure_universe candidates \
  --out_dir runs/phase14_compare \
  --dotenv .env
```
