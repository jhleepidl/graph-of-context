from __future__ import annotations
import json
from pathlib import Path

from src.benchmarks.synthetic_browsecomp import SyntheticBrowseComp


def _load(tmp_path: Path):
    corpus = json.loads((tmp_path / 'corpus.json').read_text(encoding='utf-8'))
    tasks = json.loads((tmp_path / 'tasks.json').read_text(encoding='utf-8'))
    return corpus, tasks


def test_structured_profile_generates_three_slices_without_policy_tags(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    meta = bench.prepare(
        data_dir=str(tmp_path),
        n_entities=24,
        n_tasks=18,
        distractors_per_entity=1,
        noise_docs=12,
        seed=7,
        long_horizon=True,
        benchmark_profile='structured',
    )
    assert meta['benchmark_profile'] == 'structured'

    corpus, tasks = _load(tmp_path)
    assert tasks
    slices = {str(task.get('task_slice')) for task in tasks}
    assert {'retrieval_sufficient', 'dependency_necessary', 'branch_resolution'}.issubset(slices)

    titles = [str(doc.get('title') or '') for doc in corpus]
    assert any(title.startswith('Operating City Approval - ') for title in titles)
    assert any(title.startswith('Archived Status Board - ') for title in titles)
    assert any(title.startswith('Legacy Operating Exception - ') for title in titles)

    for task in tasks:
        assert task.get('benchmark_profile') == 'structured'
        assert task.get('finish_answer_format') == 'project_city_pair'
        q = str(task.get('question') or '')
        assert 'policy tag' not in q.lower()
        turns = task.get('turns') or []
        assert all('policy tag' not in str(t).lower() for t in turns)


def test_structured_branch_tasks_require_natural_support_chain(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    bench.prepare(
        data_dir=str(tmp_path),
        n_entities=20,
        n_tasks=12,
        distractors_per_entity=1,
        noise_docs=8,
        seed=13,
        long_horizon=True,
        benchmark_profile='structured_lite',
    )
    corpus, tasks = _load(tmp_path)
    docs = {str(doc.get('docid')): doc for doc in corpus}

    branch_tasks = [task for task in tasks if task.get('task_slice') == 'branch_resolution']
    assert branch_tasks
    for task in branch_tasks:
        assert task.get('task_type') == 'structured_branch_resolution'
        gold_docids = list(task.get('gold_docids') or [])
        assert any(str(d).startswith('D_ALIAS_') for d in gold_docids)
        assert any(str(d).startswith('D_APPROVAL_') for d in gold_docids)
        assert any(str(d).startswith('D_ARCHIVED_') for d in gold_docids)
        q = str(task.get('question') or '').lower()
        assert 'exception revocation' in q
        assert 'archived status boards' in q
        if str(task.get('exception_state')) == 'revoked':
            assert any(str(d).startswith('D_REVOCATION_') for d in gold_docids)
            rev_docs = [docs[str(d)] for d in gold_docids if str(d).startswith('D_REVOCATION_')]
            assert rev_docs and all('revokes_exception_ticket' in str(doc.get('content')) for doc in rev_docs)


def test_structured_lite_dependency_tasks_are_calibrated_for_runtime(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    bench.prepare(
        data_dir=str(tmp_path),
        n_entities=20,
        n_tasks=16,
        distractors_per_entity=1,
        noise_docs=8,
        seed=17,
        long_horizon=True,
        benchmark_profile='structured_lite',
    )
    _corpus, tasks = _load(tmp_path)

    dep_tasks = [task for task in tasks if task.get('task_slice') == 'dependency_necessary']
    assert dep_tasks
    for task in dep_tasks:
        assert task.get('task_type') == 'structured_dependency_current_city'
        assert int(task.get('n_candidates') or 0) <= 2
        assert str(task.get('target_exception_state')) in {'none', 'revoked'}
        assert str(task.get('support_variant')) == 'approval_or_revoked'
        assert int(task.get('required_support_count') or 0) <= 3
        q = str(task.get('question') or '').lower()
        assert 'candidate handles (2 total)' in q
        assert 'legacy operating exception' not in q
        assert 'archived status boards' not in q


def test_structured_support_pilot_has_support_recovery_slice(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    meta = bench.prepare(
        data_dir=str(tmp_path),
        n_entities=20,
        n_tasks=18,
        distractors_per_entity=1,
        noise_docs=8,
        seed=23,
        long_horizon=True,
        benchmark_profile='structured_support_pilot',
    )
    assert meta['benchmark_profile'] == 'structured_support_pilot'
    _corpus, tasks = _load(tmp_path)
    slices = {str(task.get('task_slice')) for task in tasks}
    assert {'retrieval_sufficient', 'support_recovery', 'branch_resolution'}.issubset(slices)
    sr_tasks = [task for task in tasks if task.get('task_slice') == 'support_recovery']
    assert sr_tasks
    for task in sr_tasks:
        assert task.get('task_type') == 'structured_support_recovery'
        assert int(task.get('required_support_count') or 0) >= 2
        q = str(task.get('question') or '').lower()
        assert 'legacy operating exception' in q
        assert 'exception revocation' in q
        assert 'answer exactly as' in q
