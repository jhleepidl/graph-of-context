from __future__ import annotations

from types import SimpleNamespace

from src.llm_agent import ToolLoopConfig, ToolLoopLLMAgent


class _DummyLLM:
    pass


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


def _make_agent() -> ToolLoopLLMAgent:
    cfg = ToolLoopConfig(
        proof_closure_guard=True,
        proof_closure_search_planner=True,
        proof_closure_auto_open=True,
        proof_closure_autofinish=True,
    )
    agent = ToolLoopLLMAgent(_DummyLLM(), _DummyTools(), _DummyMem(), cfg=cfg)
    return agent


def test_support_closure_status_prefers_exception_until_revoked() -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {
        'alpha-001-01': 'Project_0001',
        'beta-002-02': 'Project_0002',
    }
    agent._structured_project_years = {'Project_0001': 2011, 'Project_0002': 2014}
    agent._structured_project_approval_city = {'Project_0001': 'City_09'}
    agent._structured_project_approval_docids = {'Project_0001': 'D_APPROVAL_0001'}
    agent._structured_project_exception_city = {'Project_0001': 'City_03'}
    agent._structured_project_exception_docids = {'Project_0001': 'D_EXCEPTION_0001'}
    task_meta = {
        'task_slice': 'support_closure',
        'decision_requires_support_closure': True,
        'candidate_handles': ['Alpha-001-01', 'Beta-002-02'],
        'proof_required_docids': ['D_ALIAS_0001', 'D_ALIAS_0002', 'D_TRUTH_0001', 'D_TRUTH_0002', 'D_APPROVAL_0001', 'D_EXCEPTION_0001'],
        'proof_expected_types': ['field_note', 'official_profile', 'approval', 'exception'],
    }
    agent.evidence_docids = ['D_ALIAS_0001', 'D_ALIAS_0002', 'D_TRUTH_0001', 'D_TRUTH_0002', 'D_APPROVAL_0001', 'D_EXCEPTION_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    status = agent._structured_support_closure_status(task_meta)
    assert status['selected_project'] == 'Project_0001'
    assert status['selected_city'] == 'City_03'
    assert status['missing_support_types'] == []
    assert status['missing_proof_docids'] == []
    ans, expl = agent._try_autofinish('support closure question', task_meta=task_meta)
    assert ans == 'Project_0001 | City_03'
    assert 'D_EXCEPTION_0001' in expl


def test_needed_structured_result_docid_prefers_missing_support_page() -> None:
    agent = _make_agent()
    agent._structured_handle_to_project = {'alpha-001-01': 'Project_0001'}
    agent._structured_project_years = {'Project_0001': 2011}
    agent._structured_project_docids = {'Project_0001': 'D_TRUTH_0001'}
    agent._structured_project_approval_city = {'Project_0001': 'City_09'}
    agent._structured_project_approval_docids = {'Project_0001': 'D_APPROVAL_0001'}
    task_meta = {
        'task_slice': 'provenance_required',
        'decision_requires_support_closure': True,
        'target_project': 'Project_0001',
        'proof_required_docids': ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_APPROVAL_0001', 'D_EXCEPTION_0001'],
        'proof_expected_types': ['field_note', 'official_profile', 'approval', 'exception'],
    }
    agent.evidence_docids = ['D_ALIAS_0001', 'D_TRUTH_0001', 'D_APPROVAL_0001']
    agent.opened_cache = {d: 'opened' for d in agent.evidence_docids}
    results = [
        {'docid': 'D_ARCHIVED_0001', 'title': 'Archived Status Board - Project_0001'},
        {'docid': 'D_EXCEPTION_0001', 'title': 'Legacy Operating Exception - Project_0001'},
    ]
    chosen = agent._find_needed_structured_result_docid(query='broad search', results=results, task_meta=task_meta)
    assert chosen == 'D_EXCEPTION_0001'
