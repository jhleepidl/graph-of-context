from __future__ import annotations
import json
from pathlib import Path

from src.benchmarks.synthetic_browsecomp import SyntheticBrowseComp


def test_synthetic_browsecomp_hard_profile_generates_alias_and_policy_docs(tmp_path: Path) -> None:
    bench = SyntheticBrowseComp()
    meta = bench.prepare(
        data_dir=str(tmp_path),
        n_entities=18,
        n_tasks=10,
        distractors_per_entity=2,
        noise_docs=12,
        seed=7,
        long_horizon=True,
        late_binding=True,
        late_binding_ratio=0.6,
        branch_merge=True,
        branch_merge_ratio=0.35,
        benchmark_profile='hard',
        hard_mode=True,
    )
    assert meta['benchmark_profile'] == 'hard'

    corpus = json.loads((tmp_path / 'corpus.json').read_text(encoding='utf-8'))
    tasks = json.loads((tmp_path / 'tasks.json').read_text(encoding='utf-8'))

    assert tasks
    assert any(str(doc.get('docid', '')).startswith('D_ALIAS_') for doc in corpus)
    assert any(str(doc.get('docid', '')).startswith('D_POLICY_') for doc in corpus)

    for task in tasks:
        assert task.get('benchmark_profile') == 'hard'
        assert task.get('task_slice') == 'memory_necessary'
        assert task.get('needs_alias_resolution') is True
        assert task.get('needs_rule_doc') is True
        assert task.get('has_stale_rule_distractor') is True
        gold_docids = list(task.get('gold_docids') or [])
        assert any(str(d).startswith('D_ALIAS_') for d in gold_docids)
        assert any(str(d).startswith('D_POLICY_') for d in gold_docids)
