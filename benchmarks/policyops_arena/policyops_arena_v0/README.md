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

## Notes

- v0 uses BM25 only (no embeddings/vector DB).
- `pydantic` is optional; dataclasses are used if unavailable.
- Baselines build prompts but require an external LLM client to produce real predictions.
