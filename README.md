# graph-of-context
Graph of Context (GoC) is a lightweight research harness for testing agent
memory strategies on long-horizon “needle-in-haystack” tasks. It includes a
synthetic dataset generator, several baselines, and a GoC implementation with
hierarchical folding and vector-based retrieval.

## Contents
- `experiment_goc.py`: dataset generator, agent implementations, and experiment runner.
- `analyze_runs.py`: summarize run logs across experiments.
- `graph_needle_test*.jsonl`: example datasets.

## Setup
Requirements:
- Python 3.9+
- `openai`, `sentence-transformers`, `tenacity`

Install:
```
pip install openai sentence-transformers tenacity
```

Set your API key (use `.env`):
```
OPENAI_API_KEY=sk-...
```

## Quickstart
Generate a dataset:
```
python experiment_goc.py generate \
  --output graph_needle_test.jsonl \
  --num-cases 10 \
  --haystack-len 30 \
  --seed 1337 \
  --needle-location main
```

Run an experiment:
```
python experiment_goc.py run \
  --dataset graph_needle_test.jsonl \
  --results experiment_goc_runs.jsonl \
  --model gpt-4o-mini
```

Analyze recent runs:
```
python analyze_runs.py --runs experiment_goc_runs.jsonl --last 3
```

## Dataset format (JSONL)
Each line is a test case:
- `id`, `seed`, `haystack_len`
- `needle_location`: `main` or `branch`
- `needle_index`: location of `get_initial_clue` when in branch
- `initial_user_prompt`, `final_user_prompt`
- `haystack`: list of dummy tool calls
- `expected_key`: the correct `KEY_####`

### Needle placement
- `main`: `get_initial_clue()` happens before the branch (baseline-safe).
- `branch`: `get_initial_clue()` is injected into the haystack (stresses Context-Folding).

## Agents
Implemented in `experiment_goc.py`:
- **Baseline_ReAct**: appends full history.
- **Baseline_AgentFold**: summarizes every N steps into one sentence.
- **Baseline_ContextFolding**: branches haystack steps and keeps only a branch return.
- **Ours_GoC**: hierarchical folding + embedding-based retrieval.

### GoC recursive graph + task-driven clustering
GoC maintains a recursive graph of context (L0 → L1 → L2 …) with task-driven
merging and chronological rendering:
- `MAX_ACTIVE_NODES = 5` (configurable via `--goc-bundle-size`)
- Each step increments a global counter; nodes track `min_step`, `max_step`,
  and `level` for timeline ordering.
- New steps merge into the latest active node when dependency is high:
  entity reuse (numbers/IDs/quotes), rare-noun overlap, and embedding cosine.
- When active nodes exceed the limit, a bottom-up recursive fold groups adjacent
  low-dependency nodes into higher-level SuperNodes (heterogeneous children).

Each node stores:
- append-only `summary` + vector embedding (sentence-transformers)
- full raw messages for visual “unfold” during rendering

When answering, GoC retrieves top-k relevant active nodes, visually unfolds
them, then renders all nodes in chronological order before calling the LLM.

## Key CLI flags
Run:
- `--model`: OpenAI model (default `gpt-4o-mini`)
- `--goc-bundle-size`: max active nodes (default 5)
- `--goc-top-k`: number of nodes to unfold (default 2)
- `--goc-embed-model`: sentence-transformers model (default `all-MiniLM-L6-v2`)
- `--goc-eot-token`: optional delimiter token for cycle boundaries

Generate:
- `--needle-location`: `main` or `branch`
- `--haystack-len`: number of dummy tool calls

## Metrics
Runs log to `experiment_goc_runs.jsonl` with:
- `accuracy`: final answer contains the correct key
- `avg_prompt_tokens`: mean input tokens per agent (from API usage)

## Notes
- This is a synthetic harness; no real tools are called.
- If OpenAI API is unstable, requests are retried with exponential backoff.
