# Status (living doc)

**Date:** 2026-02-02 (Asia/Seoul)  
**Project:** Graph-of-Context (GoC) — Benchmarking structured context vs linear history

## 1) What we have built so far (high-level)
### Research goal (paper framing)
Not “tune for accuracy” but show—under fair benchmark stress—that treating context as a single linear string (history) has limits, and that **GoC** provides:
- Traceability
- Late-binding
- Branch robustness
- Cost control (tail-token control)

Core mechanism: unfold only needed evidence closure (dependency + doc_ref), fold irrelevant episodes into proxies to prevent token tail blow-up.

### GoC implementation core
- Graph memory: nodes (user/assistant/tool/observation/proxy/failure), edges (seq/doc_ref/depends/depends_llm).
- Active context: only active nodes are injected into LLM prompt; storage remains lossless.
- Fold: compress old/complete episodes into proxy nodes.
- Unfold: restore only needed nodes for final/checkpoints (dependency closure + doc_ref expansion).

### “Fair stress” augmentation (HotpotQA track)
- Two-stage: commit → final + closed-book final
- Return gating: minimum steps/open_pages before advancing
- Stressors: branch trap, doc repeat, lost-in-middle, noise injection
- Masked refs in final to block trivial title leakage
- Fairness principle: same constraints for all methods; extra LLM calls count or apply to all

### Stability milestones (v71-ish → v83+)
- Commit anchor enforcement: Stage1 stores exactly 2 supporting_titles; Stage2 must reuse them (no drift).
- Fold policy: dfs_doc worked well in HotpotQA stress to cut p90 tail tokens while holding accuracy.
- Declared dependency (LLM self-annotation, goc.d/c) recorded as depends_llm edges (ablation; separate from GoC core claim).

### v83 direction: Multi-Commit DAG (N-commit)
- N subtask commits inside one HotpotQA instance to force **converging DAG dependencies**
- Merge plans (binary_tree/chain/none), optional closed-book merge/final
- Goal: make “graph” advantages literal (re-using/merging branches) rather than shallow two-stage.

## 2) Recent debugging progress (v84–v87 patch line)
We stabilized multi-commit mechanics (based on sanity traces):
- SUBTASK not visible early → fixed by showing SUBTASK1 at start.
- Auto-inject caused stage skipping → disabled in commit-flow (advance only via `return`).
- MERGE-stage return-gating deadlock (closed-book) → gating disabled for MERGE.
- Gating counters mismatch (tool calls vs uncached fetch) → aligned gating to open_page tool-call counts.
- dup_open policy deadlock when no alternative candidate → allow instead of hard block.
- Commit schema strictness → supporting_titles must be exactly 2.

**Current status:** Multi-commit pipeline reaches MERGE/FINAL more reliably, but tasks are too hard/long. Completion/accuracy not yet at “v71-like reasonable” levels. We need a medium preset that reduces exploration steps while keeping merge hard.

## 3) Current objective (this week)
Create a **medium difficulty Multi-Commit DAG preset**:
- Reduce exploration/loopiness (fewer steps, fewer deadlocks)
- Keep merge difficulty (hard composition + masked refs + closed-book final)
- Preserve fairness across methods

KPI targets (N=3, 30 tasks):
- Completion rate ≥ 70%
- Accuracy out of near-zero regime
- Token p90: GoC << FullHistory (tail control visible)

## 4) Next 3 actions
1) Spec + implement preset: `mc_dag_medium_n3` (config-only) and run 30-task comparison: GoC / FullHistory / SimilarityOnly.
2) Results memo: failure modes (search loops vs merge mistakes vs drift) with trace examples.
3) Stress ramp plan: knob sweep schedule to find FullHistory collapse point while GoC degrades more gracefully.

## 5) Decisions needed
- “Medium” knobs: return-gating (min_steps/min_open), noise levels, doc_repeat, branch_trap, chain length.
- Whether to provide candidate title lists (retriever top-k) to reduce exploration without leaking answers (must apply to all methods).

## 6) Where artifacts live
- Run artifacts: `runs/` or `sweeps/` (store run folders + traces)
- Registry: `research_ops/10_experiment_registry.csv`
- Results notes: `research_ops/30_results_notes/`
