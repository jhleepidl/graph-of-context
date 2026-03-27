from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


def _write_min_traceops_dataset(base_dir: Path) -> Path:
    ds_root = base_dir / 'bundle_root' / 'phase18' / 'data' / 'mixed'
    data_dir = ds_root / 'data' / 'traceops'
    data_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        'scenario': 'mixed',
        'level': 3,
        'seed': 7,
        'traceops_delay_to_relevance': 6,
    }
    thread = {
        'thread_id': 'TR0001',
        'level': 3,
        'scenario': 'mixed',
        'initial_state': {'city': 'A'},
        'steps': [
            {
                'step_id': 'S0',
                'thread_id': 'TR0001',
                'step_idx': 0,
                'kind': 'explore',
                'message': 'read old rule',
                'state': {'city': 'A'},
                'introduced_clause_ids': ['C0'],
                'avoid_target_ids': [],
                'pivot_required_ids': [],
                'gold': None,
                'metadata': {},
            },
            {
                'step_id': 'S1',
                'thread_id': 'TR0001',
                'step_idx': 1,
                'kind': 'update',
                'message': 'confirm airport and rent',
                'state': {'city': 'A', 'city_a_status': 'eligible'},
                'introduced_clause_ids': ['C1'],
                'avoid_target_ids': [],
                'pivot_required_ids': [],
                'gold': None,
                'metadata': {},
            },
            {
                'step_id': 'SP',
                'thread_id': 'TR0001',
                'step_idx': 2,
                'kind': 'pivot_check',
                'message': 'Should City A be allowed under the airport and rent rule?',
                'state': {'city': 'A', 'city_a_status': 'eligible'},
                'introduced_clause_ids': [],
                'avoid_target_ids': [],
                'pivot_required_ids': ['C0', 'C1'],
                'gold': {
                    'decision': 'allow',
                    'conditions': [],
                    'evidence_ids': ['C0', 'C1'],
                    'evidence_core_ids': ['C0', 'C1'],
                    'evidence_meta_ids': [],
                },
                'metadata': {'trap_decision_checkpoint_ids': ['C0']},
            },
        ],
        'clauses': {
            'C0': {
                'clause_id': 'C0',
                'thread_id': 'TR0001',
                'step_idx': 0,
                'node_type': 'DECISION',
                'text': 'Decision checkpoint: allow if airport access and low rent.',
            },
            'C1': {
                'clause_id': 'C1',
                'thread_id': 'TR0001',
                'step_idx': 1,
                'node_type': 'UPDATE',
                'text': 'Update: City A airport access confirmed and rent changed to low.',
                'state_key': 'city_a_status',
                'state_value': 'eligible',
                'depends_on': ['C0'],
            },
        },
        'meta': {},
    }
    (data_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (data_dir / 'threads.jsonl').write_text(json.dumps(thread) + '\n', encoding='utf-8')
    return ds_root.parent.parent.parent


def test_build_phase18_action_utility_dataset_accepts_bundle_zip(tmp_path: Path) -> None:
    bundle_root = _write_min_traceops_dataset(tmp_path)
    zip_path = tmp_path / 'phase18_bundle.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_root.rglob('*'):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(bundle_root.parent))

    out_jsonl = tmp_path / 'action_utility.jsonl'
    out_flat = tmp_path / 'action_utility_flat.jsonl'
    proc = subprocess.run(
        [
            sys.executable,
            'scripts/build_phase18_action_utility_dataset.py',
            '--input',
            str(zip_path),
            '--out_jsonl',
            str(out_jsonl),
            '--out_flat_jsonl',
            str(out_flat),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
        env={'PYTHONPATH': '.:src'},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert 'Wrote 1 pivot rows' in proc.stdout
    lines = [line for line in out_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(lines) == 1
