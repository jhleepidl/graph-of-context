#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

ACTIONS = ['none', 'unfold', 'fork', 'unfold_then_fork']


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for ln in path.read_text(encoding='utf-8').splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _hash01(text: str) -> float:
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return int(h, 16) / float(16 ** 8)


def _augment_features(base: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    feats = dict(base)
    max_steps = int(row.get('max_steps') or 0)
    step = int(row.get('step') or 0)
    feats['task_slice_support_closure'] = 1.0 if str(row.get('task_slice') or '') == 'support_closure' else 0.0
    feats['task_slice_provenance_required'] = 1.0 if str(row.get('task_slice') or '') == 'provenance_required' else 0.0
    feats['task_slice_anchor_control'] = 1.0 if str(row.get('task_slice') or '') == 'anchor_control' else 0.0
    feats['proof_required_count'] = float(int(row.get('proof_required_count') or 0))
    feats['step_ratio'] = float(step / max(1, max_steps)) if max_steps > 0 else 0.0
    feats['is_hard_support_task'] = 1.0 if bool(row.get('decision_requires_support_closure', False)) else 0.0
    return feats


def score_actions(row: Dict[str, Any]) -> Dict[str, float]:
    f = _augment_features(dict(row.get('features') or {}), row)
    sg = float(f.get('support_gap_score', 0.0) or 0.0)
    amb = float(f.get('ambiguity_score', 0.0) or 0.0)
    br = float(f.get('branch_score', 0.0) or 0.0)
    pressure = float(f.get('evidence_pressure_score', 0.0) or 0.0)
    budget = float(f.get('budget_utilization', 0.0) or 0.0)
    fork_ready = 1.0 if bool(f.get('fork_ready', False)) else 0.0
    conflict = 1.0 if bool(f.get('has_conflict', False)) else 0.0
    candidates = min(1.0, float(int(f.get('candidate_count', 0) or 0)) / 3.0)
    missing = min(1.0, float(int(f.get('missing_terms_count', 0) or 0)) / 6.0)
    step_ratio = float(f.get('step_ratio', 0.0) or 0.0)
    support_slice = 1.0 if bool(f.get('task_slice_support_closure', 0.0)) else 0.0
    prov_slice = 1.0 if bool(f.get('task_slice_provenance_required', 0.0)) else 0.0
    pivot_like = 1.0 if bool(f.get('is_pivot_like', False)) else 0.0
    final_like = 1.0 if bool(f.get('is_final_like', False)) else 0.0
    no_finish = 1.0 if str(row.get('final_outcome') or '') == 'no_finish' else 0.0
    correct = 1.0 if bool(row.get('final_correct', False)) else 0.0
    taken = str(row.get('action_taken') or 'none').strip().lower()

    none = (
        0.58 * (1.0 - sg)
        + 0.22 * (1.0 - amb)
        + 0.08 * (1.0 - pressure)
        - 0.20 * final_like
        - 0.10 * no_finish
        - 0.08 * support_slice
    )
    unfold = (
        0.50 * sg
        + 0.18 * missing
        + 0.12 * support_slice
        + 0.10 * pivot_like
        + 0.08 * (1.0 - amb)
        + 0.06 * pressure
        - 0.10 * budget
    )
    fork = (
        0.34 * amb
        + 0.22 * br
        + 0.14 * conflict
        + 0.10 * fork_ready
        + 0.08 * candidates
        + 0.06 * prov_slice
        - 0.08 * sg
    )
    utf = (
        0.28 * sg
        + 0.28 * amb
        + 0.16 * br
        + 0.12 * pressure
        + 0.08 * fork_ready
        + 0.06 * final_like
        + 0.04 * support_slice
    )

    # Late no-finish states should escalate more aggressively.
    if no_finish > 0.0 and step_ratio >= 0.55:
        if sg >= 0.18 and amb < 0.42:
            unfold += 0.18
        if amb >= 0.42 and fork_ready > 0.0:
            fork += 0.10
            utf += 0.16
        if sg >= 0.18 and amb >= 0.30:
            utf += 0.10
        none -= 0.18

    if fork_ready <= 0.0:
        fork -= 0.12
        utf -= 0.08

    if taken in ACTIONS and correct > 0.0:
        if taken == 'none':
            none += 0.04
        elif taken == 'unfold':
            unfold += 0.04
        elif taken == 'fork':
            fork += 0.04
        elif taken == 'unfold_then_fork':
            utf += 0.04

    scores = {
        'none': _clamp(none),
        'unfold': _clamp(unfold),
        'fork': _clamp(fork),
        'unfold_then_fork': _clamp(utf),
    }

    # Hard guards to avoid degenerate labels.
    if scores['none'] == max(scores.values()) and (sg >= 0.20 or amb >= 0.45 or no_finish > 0.0):
        if sg >= amb:
            scores['unfold'] = max(scores['unfold'], scores['none'] + 0.02)
        else:
            if fork_ready > 0.0:
                scores['fork'] = max(scores['fork'], scores['none'] + 0.02)
            else:
                scores['unfold_then_fork'] = max(scores['unfold_then_fork'], scores['none'] + 0.02)

    return scores


def build_row(row: Dict[str, Any], dev_ratio: float) -> Dict[str, Any]:
    features = _augment_features(dict(row.get('features') or {}), row)
    scores = score_actions(row)
    best_action = max(ACTIONS, key=lambda a: (scores[a], a))
    split = 'dev' if _hash01(f"{row.get('task_id')}::{row.get('method')}") < float(dev_ratio) else 'test'
    actions = {
        a: {
            'utility': float(scores[a]),
            'score': float(scores[a]),
            'stats': {
                'task_slice': str(row.get('task_slice') or ''),
                'final_outcome': str(row.get('final_outcome') or ''),
                'action_taken': str(row.get('action_taken') or ''),
            },
        }
        for a in ACTIONS
    }
    return {
        'thread_id': str(row.get('task_id') or ''),
        'step_id': f"{row.get('task_id')}::{row.get('method')}::step{int(row.get('step') or 0)}",
        'step_idx': int(row.get('step') or 0),
        'split': split,
        'method': str(row.get('method') or ''),
        'task_id': str(row.get('task_id') or ''),
        'task_slice': str(row.get('task_slice') or ''),
        'seed': int(row.get('seed') or 0),
        'best_action': str(best_action),
        'best_utility': float(scores[best_action]),
        'features': features,
        'actions': actions,
        'meta': {
            'benchmark_profile': str(row.get('benchmark_profile') or ''),
            'proof_required_count': int(row.get('proof_required_count') or 0),
            'decision_requires_support_closure': bool(row.get('decision_requires_support_closure', False)),
            'final_outcome': str(row.get('final_outcome') or ''),
            'final_correct': bool(row.get('final_correct', False)),
            'action_taken': str(row.get('action_taken') or ''),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a practical controller dataset from collected controller state traces.')
    ap.add_argument('--trace_glob', type=str, default='artifacts/practical_controller/**/controller_states.jsonl')
    ap.add_argument('--out_jsonl', type=Path, required=True)
    ap.add_argument('--out_flat_jsonl', type=Path, default=None)
    ap.add_argument('--dev_ratio', type=float, default=0.8)
    args = ap.parse_args()

    paths = sorted(Path('.').glob(args.trace_glob))
    if not paths:
        raise SystemExit(f'No trace jsonl files matched: {args.trace_glob}')

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_flat_jsonl is not None:
        args.out_flat_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in paths:
        rows.extend(_iter_jsonl(p))
    if not rows:
        raise SystemExit('No controller state rows found.')

    built = [build_row(r, float(args.dev_ratio)) for r in rows]
    with args.out_jsonl.open('w', encoding='utf-8') as f:
        for row in built:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if args.out_flat_jsonl is not None:
        with args.out_flat_jsonl.open('w', encoding='utf-8') as f:
            for row in built:
                for action_name, action_obj in row['actions'].items():
                    flat = {
                        'thread_id': row['thread_id'],
                        'step_id': row['step_id'],
                        'step_idx': row['step_idx'],
                        'split': row['split'],
                        'task_id': row['task_id'],
                        'task_slice': row['task_slice'],
                        'seed': row['seed'],
                        'best_action': row['best_action'],
                        'best_utility': row['best_utility'],
                        'action': action_name,
                        'utility': action_obj['utility'],
                        'is_best_action': bool(action_name == row['best_action']),
                        'features': row['features'],
                        'score': action_obj['score'],
                        'stats': action_obj['stats'],
                        'meta': row['meta'],
                    }
                    f.write(json.dumps(flat, ensure_ascii=False) + '\n')
    print(json.dumps({
        'out_jsonl': str(args.out_jsonl),
        'n_rows': len(built),
        'dev_rows': sum(1 for r in built if r['split'] == 'dev'),
        'test_rows': sum(1 for r in built if r['split'] == 'test'),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
