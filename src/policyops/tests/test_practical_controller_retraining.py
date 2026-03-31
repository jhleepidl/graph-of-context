from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_practical_controller_dataset import build_row, score_actions


def _base_state(**kw):
    row = {
        'seed': 7,
        'task_id': 'TASK_X',
        'method': 'GoC-Mixed-Heuristic',
        'step': 18,
        'action_taken': 'none',
        'features': {
            'support_gap_score': 0.10,
            'ambiguity_score': 0.10,
            'branch_score': 0.10,
            'evidence_pressure_score': 0.10,
            'budget_utilization': 0.40,
            'fork_ready': False,
            'has_conflict': False,
            'candidate_count': 1,
            'missing_terms_count': 0,
            'is_pivot_like': True,
            'is_final_like': False,
        },
        'task_slice': 'support_closure',
        'proof_required_count': 3,
        'decision_requires_support_closure': True,
        'final_outcome': 'wrong',
        'final_correct': False,
        'max_steps': 44,
    }
    row.update(kw)
    return row


def test_score_actions_prefers_unfold_on_support_gap_no_finish():
    row = _base_state(
        action_taken='none',
        final_outcome='no_finish',
        step=30,
        features={
            'support_gap_score': 0.62,
            'ambiguity_score': 0.18,
            'branch_score': 0.20,
            'evidence_pressure_score': 0.55,
            'budget_utilization': 0.55,
            'fork_ready': False,
            'has_conflict': False,
            'candidate_count': 1,
            'missing_terms_count': 3,
            'is_pivot_like': True,
            'is_final_like': True,
        },
    )
    scores = score_actions(row)
    assert scores['unfold'] == max(scores.values())


def test_score_actions_prefers_fork_or_utf_on_high_ambiguity():
    row = _base_state(
        final_outcome='no_finish',
        step=32,
        task_slice='provenance_required',
        features={
            'support_gap_score': 0.28,
            'ambiguity_score': 0.74,
            'branch_score': 0.63,
            'evidence_pressure_score': 0.60,
            'budget_utilization': 0.45,
            'fork_ready': True,
            'has_conflict': True,
            'candidate_count': 3,
            'missing_terms_count': 2,
            'is_pivot_like': True,
            'is_final_like': True,
        },
    )
    scores = score_actions(row)
    best = max(scores, key=scores.get)
    assert best in {'fork', 'unfold_then_fork'}


def test_build_row_emits_phase18_style_payload():
    row = _base_state()
    built = build_row(row, 0.8)
    assert built['best_action'] in {'none', 'unfold', 'fork', 'unfold_then_fork'}
    assert 'actions' in built and 'features' in built
    assert 'task_slice_support_closure' in built['features']


def test_train_practical_controller_smoke(tmp_path: Path):
    rows = []
    for i in range(30):
        r = _base_state(task_id=f'TASK_{i}', step=10 + (i % 10))
        if i % 3 == 0:
            r['features']['support_gap_score'] = 0.6
            r['features']['ambiguity_score'] = 0.2
            r['final_outcome'] = 'no_finish'
        elif i % 3 == 1:
            r['features']['support_gap_score'] = 0.2
            r['features']['ambiguity_score'] = 0.7
            r['features']['fork_ready'] = True
            r['features']['has_conflict'] = True
            r['task_slice'] = 'provenance_required'
            r['final_outcome'] = 'no_finish'
        built = build_row(r, 0.75)
        rows.append(built)
    dataset = tmp_path / 'practical_controller_dataset.jsonl'
    with dataset.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    out_model = tmp_path / 'controller_tree_practical.pkl'
    out_report = tmp_path / 'controller_tree_practical_report.json'
    cmd = [
        sys.executable,
        'scripts/train_practical_controller.py',
        '--dataset', str(dataset),
        '--out_model', str(out_model),
        '--out_report', str(out_report),
        '--model_type', 'tree',
        '--min_train_rows', '10',
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert out_model.exists()
    assert out_report.exists()
