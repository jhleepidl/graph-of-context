# GoC hybrid annotations: token-saving prompt gating + compact schema

This patch updates `src/llm_agent.py` to reduce prompt overhead for hybrid GoC annotations.

## What changed
- Removed always-on `STEP_INDEX: ...` injection.
- Keeps the system prompt free of long annotation instructions.
- Adds a **single compact hint line** only on gated steps (configurable).
- Supports compact schema in model outputs:
  - `goc.d`: relative step offsets (negative ints), e.g. `[-1, -3]`
  - `goc.c`: committed-title indices (e.g., `[1]` or `[2]`)
- Still supports legacy `goc.depends_on_steps`.

## How to apply
From repo root:
```bash
unzip -o goc_v71_goc_prompt_gating_compact_patch.zip -d .
```

## Config knobs
In sweep config / method config:
- `goc_annotation_mode`: `hybrid_depends` or `tracefirst`
- `goc_annotation_gate`: comma-separated triggers (default `doc_switch,pre_finish`)
  - `doc_switch`, `pre_finish`, `stage1`, `stage2`, `every_k`, `always`
- `goc_annotation_gate_pre_finish_steps`: default 2
- `goc_annotation_gate_every_k_steps`: default 0
- `goc_annotation_schema`: `compact` (default) or `legacy` (only affects hint text)

---

# v85: Multi-Commit / Two-Stage follow-up injection correctness

This patch fixes a fairness/semantics bug where follow-up user turns could be **auto-injected** (via `multi_turn_auto_inject`) before a commit/merge `return` completed, causing stage skipping in Multi-Commit DAG scenarios.

## What changed
- In `src/llm_agent.py`, auto-injection is now **disabled** for commit-based benchmarks:
  - `task.meta.two_stage == True` OR `task.meta.multi_commit == True` OR `multi_commit_n >= 2`.
- Added `user_turn_injected` trace event with fields: `reason` (`return`/`auto`), `next_head`, `pending_remaining`.
- Fixed `tool:return` trace field `ignored` to mean **"return did not inject a follow-up"** (instead of always-true in MAIN).


---

# v87: Return-gating uses open_page TOOL calls + no-candidate dup_open deadlock fix

This patch improves Multi-Commit DAG sanity stability and avoids a subtle deadlock between:
- return-gating (requires a minimum number of `open_page` actions), and
- caching / duplicate-open policies.

## What changed
- **Return gating now counts `open_page` TOOL calls in MAIN**, not only uncached doc fetches.
  - New counter: `open_page_tool_calls_main` (increments on every executed `open_page` in MAIN, including cached views/docs).
  - `return_blocked` trace now includes both:
    - `open_page_tool_calls_total` (tool calls)
    - `open_page_fetches_total` (uncached doc fetches, legacy `open_page_calls`)
- **Duplicate open hard-block**: if a duplicate `open_page(docid=...)` is requested and no alternative docid candidate exists,
  we now **ALLOW** the call (policy event `dup_open_allow`) instead of hard-blocking it, to avoid return-gating deadlocks.
- **Multi-commit SUBTASK schema validation** now enforces **exactly N supporting titles** (default 2), and that each title is a non-empty string.

