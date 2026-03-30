from __future__ import annotations

from src.llm_agent import ToolLoopConfig, ToolLoopLLMAgent


class _DummyLLM:
    pass


class _DummyTools:
    def __init__(self) -> None:
        self.search_queries = []
        self.opened_docids = []

    def search(self, query: str, topk: int = 10):
        self.search_queries.append((query, topk))
        return [
            {
                'docid': 'D_APPROVAL_0001',
                'title': 'Operating City Approval - Project_0001',
                'score': 1.0,
                'snippet': 'approval city: City_09',
            }
        ]

    def open_page(self, docid: str, section: str | None = None):
        self.opened_docids.append((docid, section))
        return {
            'docid': docid,
            'title': 'Operating City Approval - Project_0001',
            'content': 'Operating City Approval - Project_0001\napproved_city: City_09',
        }


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
        self._counter = 0

    def get_active_text(self) -> str:
        return ''

    def record_msg(self, text: str):
        return text

    def add_node(self, thread: str, kind: str, text: str, docids=None):
        self._counter += 1
        nid = f'n{self._counter}'
        self.active.append(nid)
        return nid

    def add_edge(self, etype: str, u: str, v: str) -> bool:
        return True

    def compute_unfold_candidates(self, prompt: str, k: int, topk: int):
        return []

    def _select_unfold_seeds(self, cands, k: int):
        return [], [], 0

    def record_tool(self, tool_name, args, observation, docids=None, storage_text=None):
        self._counter += 1
        nid = f't{self._counter}'
        self.active.append(nid)
        return nid

    def drain_events(self):
        return []


def _make_agent(**cfg_overrides) -> ToolLoopLLMAgent:
    cfg = ToolLoopConfig(
        proof_closure_guard=True,
        proof_closure_search_planner=True,
        proof_closure_auto_open=True,
        proof_closure_autofinish=True,
        proof_closure_repair=True,
        **cfg_overrides,
    )
    tools = _DummyTools()
    mem = _DummyMem()
    agent = ToolLoopLLMAgent(_DummyLLM(), tools, mem, cfg=cfg)
    return agent


def test_support_closure_repair_searches_and_opens_missing_support() -> None:
    agent = _make_agent(proof_closure_repair_min_step=8, max_steps=20)
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
    agent.counters['open_page_calls'] = 2
    agent._last_finish_block_reason = 'proof_closure_missing_support'

    ok = agent._maybe_run_support_closure_repair(
        step=14,
        current_user_prompt='Find the current operating city.',
        task_meta=task_meta,
        task_id='t1',
        method='SimilarityOnly-Prove-Repair',
        run_tag='r1',
    )
    assert ok is True
    assert agent.tools.search_queries[0][0] == 'Project_0001 operating city approval'
    assert agent.tools.opened_docids[0][0] == 'D_APPROVAL_0001'
    assert agent.counters['proof_closure_repair_calls'] == 1
    assert agent.counters['proof_closure_repair_opens'] == 1


def test_support_closure_repair_respects_slice_gate() -> None:
    agent = _make_agent(proof_closure_repair_allowed_slices=('support_closure',))
    task_meta = {
        'task_slice': 'anchor_control',
        'decision_requires_support_closure': False,
    }
    ok = agent._maybe_run_support_closure_repair(
        step=20,
        current_user_prompt='x',
        task_meta=task_meta,
        task_id='t2',
        method='SimilarityOnly-Prove-Repair',
        run_tag='r2',
    )
    assert ok is False
