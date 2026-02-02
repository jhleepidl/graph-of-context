# Benchmarks (v59)

This repo now includes **task-local, real-data** benchmarks designed to make
GoC's strengths visible (dependency closure, evidence-chain traceability,
late-binding robustness), without relying on WebArena.

All new benchmarks are runnable via `run_benchmark.py` and support a flexible
`--bench_cfg` JSON to expose *difficulty levers*.

## Common levers

- **Late-binding / multi-turn**
  - Benchmarks accept `variant` values like `late_title`, `late_sentence`, `late_support_titles`, ...
  - Use `filler_turns` (+ `filler_kind`) to insert distracting turns between question and follow-up.
- **Context difficulty**
  - `gold_position`: place the gold document at `front|middle|back|random|original`.
  - `branch_trap_k`: include distractors that share high token overlap with the gold doc.
  - `doc_max_chars`: truncate docs for easier settings.
  - `doc_repeat`: repeat each doc text to create long contexts (harder).

## Methods available in this repo

These are passed via `--methods` (or `--methods ALL`). The most relevant ones for
memory research are:

- `FullHistory` (sliding window by token budget)
- `LinearSummary` (lossy periodic summary)
- `SimpleRAG` (retrieve-only long-term memory; no graph)
- `SimilarityOnly` (GoC storage + folding, but retrieval-only unfold: **no dependency closure**)
- `GoC` (Graph-of-Context)

## 1) Lost-in-the-Middle (`lost_in_middle`)

**Goal:** open-book QA with long, task-local contexts and a late-binding follow-up that asks
for evidence title or a supporting sentence.

### Expected dataset
A JSONL or JSONL.GZ file with (format-tolerant):
- `question` (or `query`)
- `answer` (or `answers`)
- `contexts` / `ctxs` (list of dicts with `title` and `text`/`content`)

If contexts don't have an explicit `is_gold`, we try to infer gold by string-matching the answer.

### Example
```bash
python run_benchmark.py \
  --benchmark lost_in_middle \
  --bench_cfg '{
    "raw_path": "data/lost_in_middle/litm_test.jsonl.gz",
    "variant": "late_title",
    "total_ctx": 16,
    "gold_position": "middle",
    "branch_trap_k": 2,
    "filler_turns": 4,
    "filler_kind": "summarize",
    "doc_repeat": 2
  }' \
  --methods FullHistory GoC \
  --budget_active 20000 --budget_unfold 4000 \
  --max_steps 40
```

## 2) HotpotQA (`hotpotqa`)

**Goal:** multi-hop QA with explicit supporting-fact titles. Late-binding follow-up requires
`supporting_titles` list.

### Expected dataset
HotpotQA JSON list with fields: `_id`/`id`, `question`, `answer`, `context`, `supporting_facts`.

Supported formats for `context` and `supporting_facts`:
- **Official format**: `context = [[title, [sent1, sent2, ...]], ...]`, `supporting_facts = [[title, idx], ...]`
- **HuggingFace datasets format**: `context = {"title": [...], "sentences": [[...], ...]}`, `supporting_facts = {"title": [...], "sent_id": [...]}`

### Example
```bash
python run_benchmark.py \
  --benchmark hotpotqa \
  --bench_cfg '{
    "path": "data/hotpotqa/hotpot_dev_distractor_v1.json",
    "variant": "late_support_titles",
    "supporting_position": "middle",
    "filler_turns": 6,
    "doc_repeat": 3
  }' \
  --methods FullHistory GoC \
  --budget_active 20000 --budget_unfold 4000 \
  --max_steps 45
```

## 3) FEVER prepared (`fever_prepared`)

**Goal:** fact verification with evidence titles, using a *prepared* JSONL where doc text is included.

### Expected dataset
Prepared JSONL/JSONL.GZ (one example per line) with:
- `id`
- `claim`
- `label` (supports/refutes/not enough info)
- `docs`: list of `{docid,title,content,(url)}`
- optional `evidence_titles`: list[str]

### Example
```bash
python run_benchmark.py \
  --benchmark fever_prepared \
  --bench_cfg '{
    "prepared_path": "data/fever/fever_prepared.jsonl.gz",
    "variant": "late_label_titles",
    "filler_turns": 4,
    "doc_repeat": 2
  }' \
  --methods FullHistory GoC \
  --budget_active 20000 --budget_unfold 4000 \
  --max_steps 40
```

## Multi-turn injection knobs (runner)

You can tune *when* the follow-up turn is injected:

- `--multi_turn_min_step` (default 8)
- `--multi_turn_min_open_pages` (default 3)
- `--no_multi_turn_auto_inject` to disable injection.

These act like a coarse difficulty dial: higher thresholds force the agent to gather more evidence
before the late-binding turn arrives.

## 4) FEVER raw -> prepared (with local wiki-pages)

If you have the official FEVER jsonl (claims) and the FEVER "wiki-pages" dump
already downloaded (as in `data/fever/wiki/wiki-pages/wiki-*.jsonl`), you can
create a prepared dataset compatible with this repo.

### Step A: build a local docstore (SQLite + optional FTS5)

**Full (recommended for real runs)**
```bash
python scripts/build_fever_wiki_sqlite.py \
  --wiki_dir data/fever/wiki/wiki-pages \
  --out_db  data/fever/wiki/wiki.sqlite \
  --max_chars 6000
```

**Mini (fast debug; stores evidence pages + random reservoir)**
```bash
python scripts/build_fever_wiki_sqlite.py \
  --wiki_dir data/fever/wiki/wiki-pages \
  --out_db  data/fever/wiki/wiki_mini.sqlite \
  --keep_titles_from_fever data/fever/fever-data/dev.jsonl \
  --limit_examples 300 \
  --random_reservoir 50000
```

### Step B: prepare FEVER into task-local docs

```bash
python scripts/prepare_fever_from_wiki.py \
  --fever   data/fever/fever-data/dev.jsonl \
  --wiki_db data/fever/wiki/wiki.sqlite \
  --out     data/fever/fever_prepared/dev_prepared.jsonl.gz \
  --docs_per_task 20 \
  --gold_titles_n 2 \
  --seed 7
```

Then run the benchmark or sweep using `fever_prepared` with `prepared_path` set
to that output file.
