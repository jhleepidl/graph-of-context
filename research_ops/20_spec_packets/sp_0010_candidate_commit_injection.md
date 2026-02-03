# Spec Packet: sp_0010_candidate_commit_injection

## Goal
Inject deterministic candidate commit summaries into MERGE/FINAL prompts to avoid commit1 bias at N=5.

## Scope
- Inject [CANDIDATE_COMMITS] into ACTIVE_CONTEXT for MERGE/FINAL (all methods).
- Emit trace event `candidate_commits_injected` with truncation stats.
- Store merge winners for resolving merge-node candidates.
- Harden winner parsing + candidate resolution:
  - Parse merge winner from winner_commit/selected_commit/winner/chosen_commit (int or "C3"/"commit3").
  - Resolve merge nodes recursively to leaf commits (loop-safe).
  - FINAL fallback: root merge winner, then leaf commits if candidates < 2.
  - Emit `candidate_commits_injection_fallback` and `candidate_commits_injection_skipped` when applicable.
- Add benchmark knobs (defaults ON for multi-commit):
  - inject_candidate_commits
  - candidate_commit_max_chars
  - candidate_commit_a1_max_chars
  - candidate_commit_title_max_chars

## Constraints
- Use stored commit outputs only (no regeneration).
- Bounded size (<= max chars) and truncation flags.
- Fair across methods (benchmark-level behavior).

## Evidence
- `python -m py_compile src/llm_agent.py src/benchmarks/hotpotqa.py run_sweep.py`
- 1-task sanity (N=5) with trace showing candidate_commits_injected
- Contract check strict on sanity traces
