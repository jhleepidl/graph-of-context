# Spec Packet: <title>

## Goal
- What we want to change and why.

## Non-goals
- What we will *not* change.

## Fairness constraints (must hold)
- Same stages/constraints for all methods.
- No hidden extra LLM calls for one method.
- If extra steps are required, apply them to all methods and count cost.

## Proposed changes
### Config / knobs
- ...

### Code changes (if any)
- ...

## Success criteria
### Trace / logs
- Grep patterns:
  - ...
### Metrics
- Completion rate >= ...
- Accuracy >= ...
- Token p90 reduction visible for GoC vs FullHistory

## Commands
- Sanity:
  - ...
- Main experiment:
  - ...

## Risks & rollback
- What might break, how to revert quickly.
