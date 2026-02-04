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
PYTHONPATH=src python -m policyops.run --help
```

## Generate

```bash
PYTHONPATH=src python -m policyops.run generate --seed 0 --n_docs 30 --n_tasks 200
```

## Evaluate

```bash
PYTHONPATH=src python -m policyops.run eval --method topk --model gpt-4o-mini
PYTHONPATH=src python -m policyops.run eval --method full --model gpt-4o-mini
PYTHONPATH=src python -m policyops.run eval --method goc --model gpt-4o-mini
```

## Standard Commands

Data generation:

```bash
PYTHONPATH=src python -m policyops.run generate --seed 0 --n_docs 60 --n_tasks 200
```

Controller train (LLM off):

```bash
PYTHONPATH=src python -m policyops.run compare \
  --llm dummy --model dummy \
  --methods goc \
  --use_controller --controller_mode train \
  --task_split holdout --train_ratio 0.7 --split_seed 0
```

Eval (LLM on):

```bash
PYTHONPATH=src python -m policyops.run compare \
  --llm openai --model gpt-4.1-mini \
  --methods topk full goc oracle \
  --use_controller --controller_mode eval \
  --task_split holdout --train_ratio 0.7 --split_seed 0 \
  --force_open_top_n 1 \
  --save_raw --save_prompts
```

Failure slice analysis:

```bash
PYTHONPATH=src python -m policyops.run analyze --report runs/compare/<latest>.json
```

## Notes

- v0 uses BM25 only (no embeddings/vector DB).
- `pydantic` is optional; dataclasses are used if unavailable.
- Baselines build prompts but require an external LLM client to produce real predictions.
