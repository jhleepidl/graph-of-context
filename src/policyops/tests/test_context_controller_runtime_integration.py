from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from src.context_controller import ControllerDecision
from src.llm_agent import ToolLoopConfig, ToolLoopLLMAgent


class _DummyLLM:
    pass


class _DummyTools:
    pass


class _DummyMem:
    def __init__(self) -> None:
        self.active = ['n1', 'n2', 'n3']
        self.nodes = {
            'n1': SimpleNamespace(text='candidate alpha evidence', token_len=80, step_idx=1),
            'n2': SimpleNamespace(text='however candidate beta also appears', token_len=90, step_idx=2),
            'n3': SimpleNamespace(text='final verification evidence', token_len=70, step_idx=3),
        }
        self.edges_out = {'avoids': {}}
        self.budget_active = 1200
        self.unfold_k = 6
        self._global_step = 6
        self._event_buf = []
        self.fork_keep_recent_active = 4

    def get_active_text(self) -> str:
        return 'candidate alpha evidence however candidate beta final verification support'

    def compute_unfold_candidates(self, prompt: str, k: int, topk: int):
        return []

    def _select_unfold_seeds(self, cands, k: int):
        return [], [], 0

    def build_fork_view(self, **kwargs):
        return SimpleNamespace(
            fork_id='FK1',
            query=kwargs.get('query', ''),
            seed_ids=['n1'],
            node_ids=['n1', 'n2'],
            scoped_text='scoped evidence',
            token_count=120,
        )

    def record_fork_result(self, fork, merge_payload, kind='summary'):
        return 'fork_node_1'

    def unfold(self, query, k=None):
        return ['n4']


class _DummyController:
    def decide(self, **kwargs):
        return ControllerDecision(
            action='unfold_then_fork',
            reason='test_utf',
            unfold_query='need more support',
            fork_query='specialist fork query',
            metadata={},
        )


def _make_agent(**cfg_overrides):
    cfg = ToolLoopConfig(
        enable_context_controller=True,
        context_controller_policy='uncertainty_aware',
        context_controller_fork_gate_mode='integrated',
        fork_controller_max_calls=2,
        fork_controller_cooldown_steps=5,
        fork_controller_min_open_pages=2,
        fork_controller_min_active_tokens=100,
        fork_controller_min_branch_score=0.18,
        fork_controller_min_ambiguity=0.35,
        fork_controller_min_pressure=0.45,
        **cfg_overrides,
    )
    agent = ToolLoopLLMAgent(_DummyLLM(), _DummyTools(), _DummyMem(), cfg=cfg)
    agent.counters = Counter()
    return agent


def test_integrated_gate_allows_open_only_branching() -> None:
    agent = _make_agent(fork_controller_allow_open_only=True)
    feats = {
        'active_tokens_est': 420,
        'open_page_calls': 3,
        'search_calls': 0,
        'ambiguity_score': 0.55,
        'support_gap_score': 0.30,
        'pivot_risk': 0.60,
        'branch_score': 0.34,
        'evidence_pressure_score': 0.58,
        'candidate_count': 2,
        'has_conflict': False,
    }
    ready, reason, meta = agent._controller_fork_readiness(step=6, features=feats)
    assert ready is True
    assert reason == 'ok'
    assert meta['gate_mode'] == 'integrated'



def test_integrated_gate_respects_cooldown() -> None:
    agent = _make_agent()
    agent.counters['fork_calls'] = 1
    agent._last_fork_step = 5
    feats = {
        'active_tokens_est': 420,
        'open_page_calls': 3,
        'search_calls': 1,
        'ambiguity_score': 0.55,
        'support_gap_score': 0.30,
        'pivot_risk': 0.60,
        'branch_score': 0.34,
        'evidence_pressure_score': 0.58,
        'candidate_count': 2,
        'has_conflict': False,
    }
    ready, reason, _meta = agent._controller_fork_readiness(step=7, features=feats)
    assert ready is False
    assert reason == 'cooldown'



def test_unfold_then_fork_rechecks_after_unfold(monkeypatch) -> None:
    agent = _make_agent(context_controller_recheck_after_unfold=True)
    agent.context_controller = _DummyController()

    feature_seq = [
        {
            'active_tokens_est': 300,
            'open_page_calls': 3,
            'search_calls': 0,
            'support_gap_score': 0.5,
            'ambiguity_score': 0.5,
            'pivot_risk': 0.7,
            'branch_score': 0.10,
            'evidence_pressure_score': 0.55,
            'candidate_count': 1,
            'has_conflict': False,
        },
        {
            'active_tokens_est': 450,
            'open_page_calls': 3,
            'search_calls': 0,
            'support_gap_score': 0.5,
            'ambiguity_score': 0.6,
            'pivot_risk': 0.8,
            'branch_score': 0.30,
            'evidence_pressure_score': 0.66,
            'candidate_count': 2,
            'has_conflict': True,
        },
    ]
    readiness_seq = [
        (False, 'branch_lt_min', {'gate_mode': 'integrated'}),
        (True, 'ok', {'gate_mode': 'integrated'}),
    ]

    monkeypatch.setattr(agent, '_build_context_controller_features', lambda **kwargs: dict(feature_seq.pop(0)))
    monkeypatch.setattr(agent, '_controller_fork_readiness', lambda **kwargs: readiness_seq.pop(0))
    unfold_calls = []
    fork_calls = []

    def _fake_unfold(*args, **kwargs):
        unfold_calls.append((args, kwargs))
        return ['n4']

    def _fake_fork(*args, **kwargs):
        fork_calls.append((args, kwargs))
        return 'fork_node_x'

    monkeypatch.setattr(agent, '_run_unfold', _fake_unfold)
    monkeypatch.setattr(agent, '_run_scoped_fork', _fake_fork)

    dec, executed = agent._maybe_run_context_controller(
        step=6,
        current_user_prompt='verify best supported candidate',
        task_id='t1',
        method='GoC',
        run_tag='r1',
    )
    assert dec is not None
    assert dec.action == 'unfold_then_fork'
    assert executed is True
    assert len(unfold_calls) == 1
    assert len(fork_calls) == 1
    assert agent.counters['context_controller_calls'] == 1
    assert agent.counters['context_controller_unfold_then_fork'] == 1
    assert agent.counters['context_controller_executed'] == 1


def test_structured_lookup_rewrites_resolved_handle_query() -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {'juniper-052-61': 'Project_0052'}
    evt = agent._maybe_override_structured_lookup_search(
        query='Juniper-052-61 field note',
        qnorm=agent._normalize_query('Juniper-052-61 field note'),
        step=5,
        topk=10,
        task_id='t1',
        method='GoC',
        run_tag='r1',
    )
    assert evt is not None
    reason, new_call = evt
    assert reason == 'structured_handle_resolved_rewrite'
    assert new_call['tool'] == 'search'
    assert 'Project_0052 official profile' == new_call['args']['query']


def test_copy_goc_annotation_preserves_dict() -> None:
    agent = _make_agent()
    source = {"tool": "search", "args": {"query": "q"}, "goc": {"action": "unfold"}}
    target = {"tool": "open_page", "args": {"docid": "d1"}}
    out = agent._copy_goc_annotation(source, target)
    assert out.get("goc") == {"action": "unfold"}


def test_structured_dependency_status_tracks_resolution() -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {
        'alpha-001-01': 'Project_0001',
        'beta-002-02': 'Project_0002',
    }
    agent._structured_project_years = {'Project_0001': 2011, 'Project_0002': 2014}
    agent._structured_project_approval_city = {'Project_0001': 'City_07'}
    status = agent._structured_dependency_status({'task_slice': 'dependency_necessary', 'candidate_handles': ['Alpha-001-01', 'Beta-002-02']})
    assert status['selected_project'] == 'Project_0001'
    assert status['selected_approval_city'] == 'City_07'
    assert status['unresolved_handles'] == []
    assert status['missing_profiles'] == []


def test_structured_dependency_rewrites_toward_missing_approval() -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {
        'alpha-001-01': 'Project_0001',
        'beta-002-02': 'Project_0002',
    }
    agent._structured_project_years = {'Project_0001': 2011, 'Project_0002': 2014}
    evt = agent._maybe_override_structured_dependency_search(
        query='global search about current city',
        topk=10,
        task_meta={'task_slice': 'dependency_necessary', 'candidate_handles': ['Alpha-001-01', 'Beta-002-02']},
    )
    assert evt is not None
    reason, new_call = evt
    assert reason == 'structured_dependency_missing_approval'
    assert new_call['tool'] == 'search'
    assert new_call['args']['query'] == 'Project_0001 operating city approval'
