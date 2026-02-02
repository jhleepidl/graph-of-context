## Spec Packet
- Link: research_ops/20_spec_packets/sp_####_....

## What / Why
- What changed:
- Why:

## Files changed
- [ ] src/...
- [ ] sweep_configs/...
- [ ] scripts/...

## Fairness / Benchmark contract (must hold)
- [ ] Same stages/constraints across methods
- [ ] No hidden extra LLM calls for a single method
- [ ] Closed-book rules identical across methods (if enabled)

## Commands run (paste exact commands + exit code)
- [ ] python -m py_compile ...
- [ ] sanity run ...
- [ ] contract check script ...

## Evidence (paste outputs)
### 1) Multi-commit stage progression
- `grep -R "user_turn_injected" ... | head -n 20`
- `grep -R "\"reason\": \"auto\"" ...` (should be empty for commit-flow)

### 2) Blocked loops diagnostics
- `grep -R "\"type\": \"return_blocked\"" ... | head -n 20`

### 3) Final reached
- `grep -R "\[FINAL" ... | head -n 20`

## Artifacts
- git_sha:
- preset_path:
- artifact_path (runs/...):
- minimal traces attached? (success 1, failure 1)

## Metrics summary (small)
| method | completion | accuracy | token_p50 | token_p90 |
|---|---:|---:|---:|---:|
| GoC |  |  |  |  |
| FullHistory |  |  |  |  |
| SimilarityOnly |  |  |  |  |

## Risks / Rollback
- Risk:
- Rollback plan: