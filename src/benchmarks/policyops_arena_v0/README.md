# PolicyOps Arena v0

Synthetic policy/notice/FAQ documents plus synthetic ticket tasks for tool-augmented policy reasoning.

## Layout

- `data/worlds/`: generated documents + clauses (`.jsonl`)
- `data/tasks/`: generated tasks (`.jsonl`)
- `src/policyops/`: generator, tools, baselines, eval, CLI
- `tests/`: smoke tests

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

If you want `python -m policyops.run ...` to work without installation, run from this folder with:

```bash
PYTHONPATH=src:../.. python -m policyops.run --help
```

## Generate

```bash
PYTHONPATH=src:../.. python -m policyops.run generate --seed 0 --n_docs 30 --n_tasks 200
```

## Evaluate

```bash
PYTHONPATH=src:../.. python -m policyops.run eval --method topk --model gpt-4o-mini
PYTHONPATH=src:../.. python -m policyops.run eval --method full --model gpt-4o-mini
PYTHONPATH=src:../.. python -m policyops.run eval --method goc --model gpt-4o-mini
```

## Standard Commands

Data generation:

```bash
PYTHONPATH=src:../.. python -m policyops.run generate --seed 0 --n_docs 60 --n_tasks 200
```

Controller train (LLM off):

```bash
PYTHONPATH=src:../.. python -m policyops.run compare \
  --llm dummy --model dummy \
  --methods goc \
  --use_controller --controller_mode train \
  --task_split holdout --train_ratio 0.7 --split_seed 0
```

Eval (LLM on):

```bash
PYTHONPATH=src:../.. python -m policyops.run compare \
  --llm openai --model gpt-4.1-mini \
  --methods topk full goc oracle \
  --use_controller --controller_mode eval \
  --task_split holdout --train_ratio 0.7 --split_seed 0 \
  --force_open_top_n 1 \
  --save_raw --save_prompts
```

Failure slice analysis:

```bash
PYTHONPATH=src:../.. python -m policyops.run analyze --report runs/compare/<latest>.json
```

## Notes

- v0 uses BM25 only (no embeddings/vector DB).
- `pydantic` is optional; dataclasses are used if unavailable.
- Baselines build prompts but require an external LLM client to produce real predictions.
- bridged_v1_1 can mix canonical-in-ticket tasks via `--bridged_mix_canonical_in_ticket_rate` to measure when bridges/hop2 are unnecessary.
- Calibrated presets:
  - `--preset bridged_v1_1_calib_n8_exclcore` (n_docs=8, exclusive_core_evidence=ON)
  - `--preset bridged_v1_1_calib_n10_exclcore` (n_docs=10, exclusive_core_evidence=ON)
  - `--preset bridged_v1_1_calibrated` is an alias for the n10 preset.
- Why keep both n8 and n10?
  - n8 anchors the “harder” point where core evidence is rarer but still retrievable.
  - n10 anchors the “stable” point where hop2/bridge yields consistent, analyzable retrieval.
  - Together they bracket the target difficulty band for selection-gap diagnostics.
  - This avoids tuning to a single sweet spot and makes controller cost-control claims robust.
  - The pair makes regression detection clearer when changing retrieval or selection logic.
- `--open_policy soft_core_rerank` is analysis-only (monotonic rerank with top-2 fixed).
- `--open_policy hop2_priority` is analysis-only (bridge once, then hop2-only opens).
- `--open_policy bridge_one_only` is analysis-only (bridge cap + meta-avoidance for cost-control diagnostics).
- Selection metrics: prefer `rank_success_rate`, `winning_in_union_rate`, and `policy_gain_over_rank` (selection_gap/feasible/realized are deprecated for reporting).

## Research Validity Runs (Bridged v1.1)
These commands are for **measurement validation only** (not performance tuning).
They are **not executed here**; copy/paste to run locally.

### Representative Compare (fresh)
```bash
PYTHONPATH=src:../.. python -m policyops.run generate \
  --preset bridged_v1_1_calibrated \
  --seed 0 --n_tasks 300 \
  --scenario_mode bridged_v1_1 \
  --bridged_mix_canonical_in_ticket_rate 0.25 \
  --exclusive_core_evidence

PYTHONPATH=src:../.. python -m policyops.run compare \
  --preset bridged_v1_1_calibrated \
  --scenario_mode bridged_v1_1 \
  --llm dummy --model dummy \
  --judge symbolic \
  --methods goc goc_base \
  --agent_query_policy two_hop_bridge \
  --save_goc_graph --save_goc_dot

# A×B reports (goc, goc_base)
PYTHONPATH=src:../.. python -m policyops.run analyze \
  --report runs/compare/<latest>.json --mode bridged_ab --method goc
PYTHONPATH=src:../.. python -m policyops.run analyze \
  --report runs/compare/<latest>.json --mode bridged_ab --method goc_base

# Triage export (ACC_NO_CORE_EVIDENCE bucket included)
PYTHONPATH=src:../.. python -m policyops.triage \
  --compare_report runs/compare/<latest>.json --method goc --max_per_bucket 20

# Selection triage CSV + patterns.md (SEL_GAP / ACC_NO_CORE_EVIDENCE / A3B2_CORE0)
PYTHONPATH=src:../.. python -m policyops.run analyze \
  --mode selection_triage --report runs/compare/<latest>.json --method goc --max_per_bucket 20
```

### Split Sweep (open_split_mode=split_hop1_hop2)
```bash
for seed in 0 1 2; do
  BASE_DIR="runs/split_sweep/seed=${seed}"
  PYTHONPATH=src:../.. python -m policyops.run generate \
    --preset bridged_v1_1_calibrated \
    --seed ${seed} --n_tasks 300 \
    --scenario_mode bridged_v1_1 \
    --bridged_mix_canonical_in_ticket_rate 0.25 \
    --exclusive_core_evidence \
    --out_dir "$BASE_DIR"

  for hop1 in 0 1 2 3 4 5; do
    PYTHONPATH=src:../.. python -m policyops.run compare \
      --preset bridged_v1_1_calibrated \
      --scenario_mode bridged_v1_1 \
      --llm dummy --model dummy \
      --judge symbolic \
      --methods goc goc_base \
      --agent_query_policy two_hop_bridge \
      --open_split_mode split_hop1_hop2 \
      --open_split_hop1 ${hop1} \
      --out_dir "$BASE_DIR"
  done
done

# Append A3×B2 table to results_split_sweep.md
PYTHONPATH=src:../.. python -m policyops.run analyze \
  --mode split_sweep_ab --sweep_dir runs/split_sweep/<run_id> --method goc
```

### Analysis Bundle (Paper-ready summaries)
```bash
cd src/benchmarks/policyops_arena_v0
PYTHONPATH=src:../.. python -m policyops.run analyze \
  --mode analysis_bundle \
  --run_dir runs/calib_step_ab_20260206_180204
```

## Threaded v1.2 (3-episode, late-binding)
These commands are for **threaded evaluation** (traceability/late-binding/cost control).
They are **not executed here**; copy/paste to run locally.

```bash
# 0) tests
python -m pytest -q src/benchmarks/policyops_arena_v0/tests

# 1) generate (small smoke)
cd src/benchmarks/policyops_arena_v0
PYTHONPATH=src:../.. python -m policyops.run generate \
  --seed 0 --n_threads 30 \
  --scenario_mode threaded_v1_2 \
  --preset threaded_v1_2_calib_n10_exclcore

# 2) compare (symbolic judge) — methods + baselines
PYTHONPATH=src:../.. python -m policyops.run compare \
  --scenario_mode threaded_v1_2 \
  --preset threaded_v1_2_calib_n10_exclcore \
  --n_threads 30 \
  --judge symbolic \
  --llm dummy --model dummy \
  --methods goc full_history similarity_only agent_fold \
  --save_goc_graph --save_goc_dot
```
