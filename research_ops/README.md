# Research Ops Pack (GoC)

This folder is the shared workspace for **GPT (design/analysis/writing)**, **Codex (implementation/execution/branches)**, and **You (owner/PM)**.

Goal: keep decisions, experiment provenance, and results *traceable* and *reproducible*—without slowing iteration.

## Folder layout
- `00_status.md` — live status board (update often; keep it short).
- `10_experiment_registry.csv` — one line per experiment run-group (provenance + key metrics).
- `20_spec_packets/` — design specs mapping 1:1 to PRs/branches.
- `30_results_notes/` — short analysis memos that later become paper sections.
- `_auto/` — machine-written manifests (Codex updates only).

## Naming conventions
- Spec packets: `sp_####_short_title.md`
- Results notes: `r_####_short_title.md`
- Experiment IDs (registry): `mc_dag_<difficulty>_n<N>_<YYYYMMDD>_<optional_tag>`

## Minimum process rules
1) Every code/config change ships with a spec packet (even if small).
2) Every experiment gets a registry row (git SHA, preset path, artifact path).
3) Results notes include: completion, accuracy, token p50/p90, and 2–3 failure modes.
4) Main branch stays “paper-clean”; experiments live on feature branches.
5) Every run-group produces a Debug Packet under `runs/<exp_id>/` and appends `research_ops/_auto/run_manifest.json`.

## Roles (RACI)
- GPT: designs + prompts + analyses + paper writing (Responsible)
- Codex: implements + runs + branch/PR hygiene (Responsible)
- You: prioritizes + approves + manages artifacts + paper files (Accountable)

## Quick start
- Update `00_status.md` with the next 3 actions.
- Create a spec packet in `20_spec_packets/` for the next PR.
- After running: append a row to `10_experiment_registry.csv`, then write a results note in `30_results_notes/`.
