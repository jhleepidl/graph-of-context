# goc_fold_agent

A minimal, runnable toy framework to compare context management strategies in a **Context-Folding style** agent setup.

## Included methods
- **FullHistory**: keep everything (oldest pruned only when hard budget exceeded).
- **ContextFolding-Discard**: branch/return, but branch internals are *discarded* after `return` (lossy).
- **LinearSummary**: periodic global summarization, discarding originals (lossy).
- **AgentFold-Range**: heuristic baseline that folds **contiguous ranges** into summaries (AgentFold-like, lossy).
- **GoC (Graph-of-Context)**: folds out of active context but **preserves originals in storage**, and can **unfold a minimal dependency closure** when needed.

This uses a Context-Folding style tool schema: `search`, `open_page`, `branch`, `return`, `finish`.

## Quick start

```bash
python run_experiment.py --n_tasks 50 --seed 7
```

Outputs:
- `results.jsonl` (per-task results)
- `report.md` (summary table)

## Key knobs (to amplify differences)
```bash
python run_experiment.py   --n_tasks 80   --budget_active 1200   --budget_unfold 600   --summary_keep_fields 0   --unfold_k 8   --noise_docs 80   --distractors_per_entity 2   --seed 7
```

- `budget_active`: active context token budget (approx)
- `budget_unfold`: max token budget added by an unfold (GoC only)
- `summary_keep_fields`: how lossy branch return summaries are (0/1 makes it lossy)
- `unfold_k`: cap on how many stored nodes to bring back on an unfold
- `agentfold_fold_chunk`: contiguous chunk size folded per operation for AgentFold-Range

## Optional: real LLM integration
The default experiments do **not** require an LLM (they use a deterministic, context-limited solver).

If you want a tool-using LLM agent:
- `src/llm_agent.py` implements a JSON tool-call protocol: the model returns `{"tool":"search","args":{...}}` etc.
- `src/llm_openai.py` provides an optional OpenAI wrapper (requires `pip install openai` and `OPENAI_API_KEY`).

These modules are intentionally decoupled from the default runner so the toy benchmark stays runnable out-of-the-box.

## Notes
- Accuracy differences are driven by **lossy branch summaries** + distractor documents: when a method cannot recover omitted attributes, it re-searches and may be misled.
- GoC differs by preserving folded branch traces and retrieving a **minimal dependency closure** (simple graph with `depends` + `doc_ref` edges).

## Run with a real LLM (optional)

1) Install OpenAI SDK:
```bash
pip install openai
```

2) Create `.env` from `.env.example` and set your key:
```bash
cp .env.example .env
# edit .env
```

3) Ensure dataset exists (one-time):
```bash
python run_experiment.py --n_tasks 50 --seed 7
```

4) Run LLM benchmark (single method or ALL):
```bash
python run_llm_benchmark.py --method GoC --n_tasks 10 --model gpt-4o-mini
python run_llm_benchmark.py --method ALL --n_tasks 5 --model gpt-4o-mini
```

Outputs:
- `llm_results.jsonl`
- `llm_report.md`

Notes:
- The LLM runner uses a strict **JSON-only tool-call protocol**. If your SDK supports it, `response_format={"type":"json_object"}` is used automatically.



### LLM runner extra logging & JSON recovery
- The LLM runner logs tool usage stats per task (tool_calls/search/open_page/repeated_search_count).
- If JSON parsing fails, it automatically reprompts up to `max_json_retries` times.

## Extending to new benchmarks

This repo now supports a pluggable benchmark interface:

- Implement a new class in `src/benchmarks/<your_benchmark>.py` that follows `Benchmark` in `src/benchmarks/base.py`.
- Register it in `src/benchmarks/registry.py`.
- Run it via the unified runner:

```bash
python run_benchmark.py --benchmark synthetic_browsecomp --prepare --runner deterministic --methods ALL
python run_benchmark.py --benchmark synthetic_browsecomp --runner llm --methods GoC --n_tasks 50
```

What you need to provide per benchmark:
- `prepare(data_dir, ...)` (optional): generate/download data
- `build_tools(data_dir)`: construct tools (e.g., local corpus tools, web tools, code tools, etc.)
- `load_tasks(data_dir)`: load tasks into `Task` objects
- `evaluate(pred_answer, pred_expl, task)`: correctness + optional evidence metrics

This makes it easy to plug in future long-horizon agent benchmarks without rewriting the GoC/AgentFold baselines.

## FAISS vector retriever (optional)

This repo supports a pluggable retriever backend for corpus search:

- `bm25` (default, no extra deps)
- `faiss` (vector index)

To use FAISS:
```bash
pip install faiss-cpu
python run_benchmark.py --benchmark synthetic_browsecomp --prepare --runner deterministic --methods GoC --retriever faiss
```

Notes:
- The current FAISS retriever uses a lightweight **hashing embedder** (deterministic, no model downloads) as a structural baseline.
- You can later replace it with a real embedding model by swapping the embedder in `src/retrievers/faiss_retriever.py`.

## Debugging: per-step stdout + LLM I/O traces

For LLM runs, you can print progress at every step and also write JSONL traces of prompts/outputs:

```bash
python run_benchmark.py --runner llm --methods GoC --task_limit 3 --verbose_steps --log_dir logs/traces
```

This writes one trace file per (method, task) into `logs/traces/` as JSONL:
- prompt snapshots (bounded ACTIVE_CONTEXT tail)
- model outputs and JSON recovery attempts
- tool invocations and brief results (docids / content preview)

## Sweeps: run many configs and aggregate into one master file

Create a sweep JSON (see `sweep_example.json`) and run:

```bash
python run_sweep.py --config sweep_example.json --out_dir sweeps
```

Outputs:
- `sweeps/<run_id>/` per-run artifacts (results + report + optional traces)
- a single master file: `sweeps/sweep_summary_<...>.jsonl` containing params + summary metrics per method

## Summarize sweeps into CSV (+ simple heatmaps)

After running `run_sweep.py`, you get a master JSONL file like:
`sweeps/sweep_summary_<...>.jsonl`.

Convert it into CSV (and optionally parse traces):

```bash
python summarize_sweep.py --master sweeps/sweep_summary_XXXX.jsonl --out_dir sweep_summary
python summarize_sweep.py --master sweeps/sweep_summary_XXXX.jsonl --out_dir sweep_summary --include_traces
```

Pivot a metric (e.g., accuracy) over two params and generate a pivot CSV + heatmap image (PGM):

```bash
python summarize_sweep.py \
  --master sweeps/sweep_summary_XXXX.jsonl \
  --out_dir sweep_summary \
  --pivot_x budget_active --pivot_y unfold_k --pivot_value accuracy --pivot_method GoC
```

Outputs:
- `sweep_summary/sweep_summary_by_method.csv`
- (optional) `sweep_summary/trace_summary.csv`
- (optional) `sweep_summary/pivot_<...>.csv`
- (optional) `sweep_summary/heatmap_<...>.pgm` (portable graymap)

