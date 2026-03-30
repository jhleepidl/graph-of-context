from __future__ import annotations

from src.llm_agent import ToolLoopConfig, ToolLoopLLMAgent


class _DummyResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _DummyLLM:
    def __init__(self, text: str = '{"status": "need_more_evidence", "support_summary": "Need approval", "recommended_queries": ["Operating City Approval - Project_0001"], "confidence": 0.31}') -> None:
        self._text = text

    def generate(self, messages, tools=None):
        return _DummyResp(self._text)


class _DummyTools:
    pass


class _DummyMem:
    def __init__(self) -> None:
        self.active = []
        self.nodes = {}
        self.edges_out = {'avoids': {}}
        self.budget_active = 1200
        self.unfold_k = 6
        self._global_step = 0
        self._event_buf = []
        self.current_thread = 'main'

    def get_active_text(self) -> str:
        return ''

    def record_msg(self, text: str):
        return text

    def add_node(self, thread: str, kind: str, text: str, docids=None):
        return 'n1'

    def add_edge(self, etype: str, u: str, v: str) -> bool:
        return True

    def compute_unfold_candidates(self, prompt: str, k: int, topk: int):
        return []

    def _select_unfold_seeds(self, cands, k: int):
        return [], [], 0

    def build_fork_view(self, **kwargs):  # pragma: no cover - existence only
        return None

    def record_fork_result(self, *args, **kwargs):  # pragma: no cover - existence only
        return 'fork_node_1'


def _make_agent(llm=None, **cfg_overrides) -> ToolLoopLLMAgent:
    cfg = ToolLoopConfig(
        proof_closure_guard=True,
        proof_closure_search_planner=True,
        proof_closure_auto_open=True,
        proof_closure_autofinish=True,
        enable_scoped_fork=True,
        proof_closure_fork_verify=True,
        proof_closure_fork_min_step=10,
        proof_closure_fork_late_window=8,
        max_steps=40,
        **cfg_overrides,
    )
    return ToolLoopLLMAgent(llm or _DummyLLM(), _DummyTools(), _DummyMem(), cfg=cfg)


def test_support_closure_fork_verify_triggers_proactively_on_hard_slice(monkeypatch) -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {'alpha-001-01': 'Project_0001'}
    agent._structured_project_years = {'Project_0001': 2011}
    agent._structured_project_docids = {'Project_0001': 'D_TRUTH_0001'}
    task_meta = {
        'task_slice': 'support_closure',
        'decision_requires_support_closure': True,
        'target_project': 'Project_0001',
        'proof_expected_types': ['field_note', 'official_profile', 'approval'],
        'proof_required_docids': ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_APPROVAL_0001'],
    }
    agent.evidence_docids = ['D_ALIAS_0001', 'D_TRUTH_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    agent.counters['search_calls'] = 2
    agent.counters['open_page_calls'] = 6
    agent.counters['proof_closure_search_rewrites'] = 1
    agent.counters['premature_finish_blocked'] = 1
    called = {}

    def _fake_run_scoped_fork(**kwargs):
        called.update(kwargs)
        return 'fork_node_1'

    monkeypatch.setattr(agent, '_run_scoped_fork', _fake_run_scoped_fork)
    ok = agent._maybe_run_support_closure_fork_verify(
        step=30,
        current_user_prompt='Determine the current operating city.',
        task_meta=task_meta,
        task_id='t1',
        method='SimilarityOnly-Prove-Fork-Verify',
        run_tag='r1',
    )
    assert ok is True
    assert called['reason'] == 'proof_closure_fork_verify'
    assert 'Operating City Approval' in called['query']


def test_support_closure_fork_verify_respects_max_calls(monkeypatch) -> None:
    agent = _make_agent(proof_closure_fork_max_calls=1)
    agent._structured_handle_to_project = {'alpha-001-01': 'Project_0001'}
    agent._structured_project_years = {'Project_0001': 2011}
    agent._structured_project_docids = {'Project_0001': 'D_TRUTH_0001'}
    task_meta = {
        'task_slice': 'support_closure',
        'decision_requires_support_closure': True,
        'target_project': 'Project_0001',
        'proof_expected_types': ['field_note', 'official_profile', 'approval', 'exception'],
        'proof_required_docids': ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_APPROVAL_0001', 'D_EXCEPTION_0001'],
    }
    agent.evidence_docids = ['D_ALIAS_0001', 'D_TRUTH_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    agent.counters['search_calls'] = 2
    agent.counters['open_page_calls'] = 3
    agent.counters['fork_calls'] = 1
    monkeypatch.setattr(agent, '_run_scoped_fork', lambda **kwargs: 'fork_node_1')
    ok = agent._maybe_run_support_closure_fork_verify(
        step=18,
        current_user_prompt='Determine the current operating city.',
        task_meta=task_meta,
        task_id='t1',
        method='SimilarityOnly-Prove-Fork-Verify',
        run_tag='r1',
    )
    assert ok is False


def test_run_scoped_fork_keeps_need_more_evidence_for_proof_verify() -> None:
    agent = _make_agent()
    class _Fork:
        fork_id = 'FK1'
        query = 'q'
        seed_ids = []
        node_ids = ['n1']
        scoped_text = 'scoped support text'
        token_count = 42

    recorded = {}
    agent.mem.build_fork_view = lambda **kwargs: _Fork()  # type: ignore[attr-defined]
    def _record(fork, text, kind='summary'):
        recorded['text'] = text
        recorded['kind'] = kind
        return 'fork_node_1'
    agent.mem.record_fork_result = _record  # type: ignore[attr-defined]

    node = agent._run_scoped_fork(
        query='Need approval and exception chain.',
        reason='proof_closure_fork_verify',
        step=14,
        task_id='t1',
        method='SimilarityOnly-Prove-Fork-Verify',
        run_tag='r1',
    )
    assert node == 'fork_node_1'
    assert agent.counters['fork_calls'] == 1
    assert 'recommended_queries' in recorded['text']


def test_run_scoped_fork_non_proof_verify_still_defers_need_more_evidence() -> None:
    agent = _make_agent()
    class _Fork:
        fork_id = 'FK1'
        query = 'q'
        seed_ids = []
        node_ids = ['n1']
        scoped_text = 'scoped support text'
        token_count = 42

    agent.mem.build_fork_view = lambda **kwargs: _Fork()  # type: ignore[attr-defined]
    agent.mem.record_fork_result = lambda *args, **kwargs: 'fork_node_1'  # type: ignore[attr-defined]

    node = agent._run_scoped_fork(
        query='Need approval and exception chain.',
        reason='always_step',
        step=14,
        task_id='t1',
        method='GoC-SimSeed-Fork-Dep',
        run_tag='r1',
    )
    assert node is None
    assert agent.counters['fork_calls'] == 0
    assert agent.counters['fork_deferred'] == 1


def test_support_closure_fork_verify_skips_disallowed_slice(monkeypatch) -> None:
    agent = _make_agent(proof_closure_fork_allowed_slices=("support_closure",))
    agent._structured_handle_to_project = {'alpha-001-01': 'Project_0001'}
    agent._structured_project_years = {'Project_0001': 2011}
    agent._structured_project_docids = {'Project_0001': 'D_TRUTH_0001'}
    task_meta = {
        'task_slice': 'provenance_required',
        'decision_requires_support_closure': True,
        'target_project': 'Project_0001',
        'proof_expected_types': ['field_note', 'official_profile', 'approval', 'exception'],
        'proof_required_docids': ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_APPROVAL_0001', 'D_EXCEPTION_0001'],
    }
    agent.evidence_docids = ['D_ALIAS_0001', 'D_TRUTH_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    agent.counters['search_calls'] = 3
    agent.counters['open_page_calls'] = 6
    monkeypatch.setattr(agent, '_run_scoped_fork', lambda **kwargs: 'fork_node_1')
    ok = agent._maybe_run_support_closure_fork_verify(
        step=28,
        current_user_prompt='Determine the current operating city.',
        task_meta=task_meta,
        task_id='t1',
        method='SimilarityOnly-Prove-Fork-Selective',
        run_tag='r1',
    )
    assert ok is False
    assert agent.counters['proof_closure_fork_verify_skipped_slice'] == 1


def test_support_closure_fork_verify_skips_low_severity_gap(monkeypatch) -> None:
    agent = _make_agent(proof_closure_fork_allowed_slices=("support_closure",), proof_closure_fork_min_missing_docids=2)
    agent._structured_handle_to_project = {'alpha-001-01': 'Project_0001'}
    agent._structured_project_years = {'Project_0001': 2011}
    agent._structured_project_docids = {'Project_0001': 'D_TRUTH_0001'}
    task_meta = {
        'task_slice': 'support_closure',
        'decision_requires_support_closure': True,
        'target_project': 'Project_0001',
        'proof_expected_types': ['field_note', 'official_profile'],
        'proof_required_docids': ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_EXTRA_0001'],
    }
    # Only one proof docid remains missing, with no unresolved support type gaps; this should not trigger the selective verifier.
    agent.evidence_docids = ['D_ALIAS_0001', 'D_TRUTH_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    agent.counters['search_calls'] = 4
    agent.counters['open_page_calls'] = 9
    agent.counters['premature_finish_blocked'] = 1
    monkeypatch.setattr(agent, '_run_scoped_fork', lambda **kwargs: 'fork_node_1')
    ok = agent._maybe_run_support_closure_fork_verify(
        step=33,
        current_user_prompt='Determine the current operating city.',
        task_meta=task_meta,
        task_id='t1',
        method='SimilarityOnly-Prove-Fork-Selective',
        run_tag='r1',
    )
    assert ok is False
    assert agent.counters['proof_closure_fork_verify_skipped_low_severity'] == 1
