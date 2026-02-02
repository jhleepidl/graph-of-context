# Spec Packet: sp_0006_eval_hygiene_autofix_and_outliers

## Goal
Improve evaluation hygiene by reducing schema-loop noise and producing standardized outlier diagnostics.

## Scope
- Optional commit-mismatch autofix in multi-commit return validator (default OFF).
- Explicit trace event for autofix.
- Summary_min includes schema error type counts + autofix count.
- Add outlier report generator for max_steps_exit cases.

## Non-goals
- Changing difficulty knobs or preset semantics.
- Auto-fixing parse errors or missing fields.

## Requirements
- Autofix triggers only when: payload parses, required fields present, titles len==2, commit mismatch only.
- Must apply equally to all methods (config controlled).
- Trace event: `type=schema_autofix`, fields: expected_commit, provided_commit, stage_kind, task_id, method.
- Summary_min includes:
  - schema_autofix_commit_mismatch_count
  - schema_error_type_counts
- Outlier report writes `outlier_report.json` + `outlier_report.md` under run dir.

## Evidence
- py_compile new scripts
- 1-task sanity with autofix OFF
- 1-task sanity with autofix ON
- contract checker strict on sanity artifacts
