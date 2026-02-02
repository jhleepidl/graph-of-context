# Spec Packet: sp_0000_research_ops_pack

## Goal
Create a shared workspace for GPT (design/analysis/writing), Codex (implementation/execution/branches), and You (owner/PM) so iteration is faster and paper-ready.

## Non-goals
- No enforcement tooling yet (no CI gates).
- No automated metrics extraction yet (optional later).

## Fairness constraints (must hold)
- All comparative methods must run the same benchmark stages and constraints.
- Any additional LLM calls required by a method must be counted and/or applied to all methods equivalently.

## Proposed changes
Add `research_ops/` with:
- `00_status.md`
- `10_experiment_registry.csv`
- `20_spec_packets/`
- `30_results_notes/`
- `_auto/`

## Success criteria
- Everyone updates/reads the same status board.
- Each experiment has: (git SHA, preset path, artifact path) recorded.
- Results notes are short and paper-ready.

## Commands
- None (docs only).

## Risks & rollback
- Minimal risk. If structure doesn't fit, rename folders later.
