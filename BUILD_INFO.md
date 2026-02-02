# goc_v79This folder is a full snapshot of the working code after applying the following changes on top of the previous goc_v71 snapshot:

- Added CLI flags for GoC annotation / prompt-gating controls in `run_benchmark.py`:
  - `--goc_annotation_mode`, `--goc_annotation_gate`, `--goc_annotation_gate_pre_finish_steps`, `--goc_annotation_gate_every_k_steps`, `--goc_annotation_schema`
  - `--goc_declared_dep_max_back`, `--goc_declared_dep_max_per_step`
- Updated `src/runners/llm.py:run_llm()` to accept and forward `goc_declared_dep_max_back/per_step` (plus the annotation flags) into `ToolLoopConfig` via `dataclasses.replace`, fixing:
  - `TypeError: run_llm() got an unexpected keyword argument 'goc_declared_dep_max_back'`



## Changes in v73

- Fixed memory unfold accounting in `src/llm_agent.py` so `mem_unfold_added_tokens_est` and `mem_unfold_activated_nodes` are derived from `used_tokens_est` / `activated` when `added_tokens` / `activated_count` are absent (previously showed 0 in reports).
- Reduced token overhead and improved compliance for GoC dependency annotations:
  - Hint line text changed from `Optional ...` to `If relevant, add ...`.
  - Added one-shot gating for `stage1` / `stage2` triggers (the hint is injected at most once per task per stage).


## goc_v75
- Added CLI alias `--sweep` (maps to `--bench_cfg_path`) for convenience.
- Added CLI flag `--goc_annotation_force {0,1}` and plumbed through to `run_llm`/ToolLoopConfig.
- Fixed `run_llm` signature/overrides to accept `goc_annotation_force`.


## goc_v77

- Fixed GoC annotation prompt-gating so multiple gates are OR'ed (e.g., `stage2,doc_switch,pre_finish`).
  Previously, including `stage2` would suppress other triggers on non-stage2 steps.
- Improved two-stage stage-tag detection for HotpotQA-style prompts that use strings like "FOLLOW-UP" / "Now CALL finish".
- Shortened the injected hint line to reduce token overhead while strongly encouraging `goc:{"d":[-1]}`.
- Added a trace event (`goc_annotation_missing`) + safe auto-injection of `goc:{}` when `goc_annotation_force` is enabled
  but the model still omits the `goc` key on a hinted step.
