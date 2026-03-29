from __future__ import annotations

import json
from pathlib import Path

from src.benchmarks.synthetic_browsecomp import SyntheticBrowseComp


def _load(tmp_path: Path):
    corpus = json.loads((tmp_path / 'corpus.json').read_text(encoding='utf-8'))
    tasks = json.loads((tmp_path / 'tasks.json').read_text(encoding='utf-8'))
    return corpus, tasks


def test_phase21_support_closure_profile_generates_proof_slices(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    meta = bench.prepare(
        data_dir=str(tmp_path),
        n_entities=24,
        n_tasks=18,
        distractors_per_entity=1,
        noise_docs=12,
        seed=29,
        long_horizon=True,
        benchmark_profile='phase21_support_closure',
    )
    assert meta['benchmark_profile'] == 'phase21_support_closure'
    _corpus, tasks = _load(tmp_path)
    slices = {str(task.get('task_slice')) for task in tasks}
    assert {'anchor_control', 'support_closure', 'provenance_required'}.issubset(slices)

    proof_tasks = [task for task in tasks if task.get('decision_requires_support_closure')]
    assert proof_tasks
    for task in proof_tasks:
        proof_docids = list(task.get('proof_required_docids') or [])
        assert proof_docids
        assert float(task.get('proof_complete_threshold') or 0.0) == 1.0
        assert int(task.get('proof_required_count') or 0) == len(proof_docids)
        q = str(task.get('question') or '').lower()
        assert 'support chain' in q or 'proof-complete' in q or 'current-support chain' in q


def test_phase21_evaluate_reports_proof_complete_metrics(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    bench.prepare(
        data_dir=str(tmp_path),
        n_entities=20,
        n_tasks=12,
        distractors_per_entity=1,
        noise_docs=8,
        seed=31,
        long_horizon=True,
        benchmark_profile='phase21_support_closure',
    )
    tasks = bench.load_tasks(str(tmp_path))
    task = next(t for t in tasks if t.meta.get('decision_requires_support_closure'))
    proof_docids = list(task.meta.get('proof_required_docids') or [])

    ev_full = bench.evaluate(task.answer, 'Evidence docids: ' + ', '.join(proof_docids), task)
    assert ev_full['correct'] is True
    assert ev_full['proof_complete'] is True
    assert ev_full['proof_complete_correct'] is True
    assert float(ev_full['proof_docid_cov']) == 1.0

    ev_partial = bench.evaluate(task.answer, 'Evidence docids: ' + proof_docids[0], task)
    assert ev_partial['correct'] is True
    assert ev_partial['proof_complete'] is False
    assert ev_partial['proof_complete_correct'] is False
    assert float(ev_partial['proof_docid_cov']) < 1.0
