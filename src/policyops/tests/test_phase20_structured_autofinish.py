from __future__ import annotations

from collections import Counter

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
        self.fork_keep_recent_active = 4

    def get_active_text(self) -> str:
        return ""

    def compute_unfold_candidates(self, prompt: str, k: int, topk: int):
        return []

    def _select_unfold_seeds(self, cands, k: int):
        return [], [], 0

    def unfold(self, query, k=None):
        return []


def _make_agent() -> ToolLoopLLMAgent:
    agent = ToolLoopLLMAgent(_DummyLLM(), _DummyTools(), _DummyMem(), cfg=ToolLoopConfig())
    agent.counters = Counter()
    agent.opened_cache = {
        'D_TRUTH_0042': 'Project_0042 OFFICIAL PROFILE\nstart_year: 2003\nheadquarters: City_33',
        'D_APPROVAL_0042': 'OPERATING CITY APPROVAL\nproject: Project_0042\napproved_operating_city: City_50',
        'D_EXCEPTION_0042': 'LEGACY OPERATING EXCEPTION NOTICE\nproject: Project_0042\noverride_city: City_33',
        'D_REVOCATION_0037': 'EXCEPTION REVOCATION NOTICE\nproject: Project_0037',
    }
    agent.evidence_docids = ['D_TRUTH_0042', 'D_APPROVAL_0042', 'D_EXCEPTION_0042']
    return agent


def test_autofinish_current_city_active_exception() -> None:
    agent = _make_agent()
    agent._structured_candidate_handles = ['Nimbus-064-10', 'Beacon-042-21', 'Pioneer-018-55']
    agent._structured_handle_to_project = {
        'nimbus-064-10': 'Project_0064',
        'beacon-042-21': 'Project_0042',
        'pioneer-018-55': 'Project_0018',
    }
    agent._structured_project_years = {
        'Project_0064': 2020,
        'Project_0042': 2003,
        'Project_0018': 2018,
    }
    agent._structured_project_hq = {'Project_0042': 'City_33'}
    agent._structured_project_approval_city = {'Project_0042': 'City_50'}
    agent._structured_project_exception_city = {'Project_0042': 'City_33'}
    agent._structured_project_revoked = set()

    question = (
        "You must use evidence from opened pages. Candidate handles (3 total): Nimbus-064-10, Beacon-042-21, Pioneer-018-55. "
        "First resolve each handle via its FIELD NOTE page to the canonical project. Then open the OFFICIAL PROFILE pages and identify "
        "the project with the earliest start_year. For that selected project, determine the CURRENT operating city using the OFFICIAL PROFILE "
        "plus any Operating City Approval, Legacy Operating Exception, and Exception Revocation notices for the same project. "
        "Answer exactly as '<ProjectName> | <City>'."
    )
    task_meta = {
        'task_slice': 'dependency_necessary',
        'supports_current_city_chain': True,
        'candidate_handles': ['Nimbus-064-10', 'Beacon-042-21', 'Pioneer-018-55'],
        'target_exception_state': 'active',
    }
    ans, expl = agent._try_autofinish(question, task_meta=task_meta)
    assert ans == 'Project_0042 | City_33'
    assert 'D_EXCEPTION_0042' in expl


def test_autofinish_current_city_revoked_exception() -> None:
    agent = _make_agent()
    agent._structured_candidate_handles = ['Beacon-037-74', 'Mosaic-061-26', 'Quartz-060-12']
    agent._structured_handle_to_project = {
        'beacon-037-74': 'Project_0037',
        'mosaic-061-26': 'Project_0061',
        'quartz-060-12': 'Project_0060',
    }
    agent._structured_project_years = {
        'Project_0037': 2012,
        'Project_0061': 2016,
        'Project_0060': 2018,
    }
    agent._structured_project_hq = {'Project_0037': 'City_3'}
    agent._structured_project_approval_city = {'Project_0037': 'City_1'}
    agent._structured_project_exception_city = {'Project_0037': 'City_3'}
    agent._structured_project_revoked = {'Project_0037'}
    agent.opened_cache.update({
        'D_TRUTH_0037': 'Project_0037 OFFICIAL PROFILE\nstart_year: 2012\nheadquarters: City_3',
        'D_APPROVAL_0037': 'OPERATING CITY APPROVAL\nproject: Project_0037\napproved_operating_city: City_1',
        'D_EXCEPTION_0037': 'LEGACY OPERATING EXCEPTION NOTICE\nproject: Project_0037\noverride_city: City_3',
        'D_REVOCATION_0037': 'EXCEPTION REVOCATION NOTICE\nproject: Project_0037',
    })
    agent.evidence_docids = ['D_TRUTH_0037', 'D_APPROVAL_0037', 'D_EXCEPTION_0037', 'D_REVOCATION_0037']

    question = (
        "You must use evidence from opened pages. Candidate handles (3 total): Beacon-037-74, Mosaic-061-26, Quartz-060-12. "
        "First resolve each handle via its FIELD NOTE page to the canonical project. Then open the OFFICIAL PROFILE pages and identify "
        "the project with the earliest start_year. For that selected project, determine the CURRENT operating city using the OFFICIAL PROFILE "
        "plus any Operating City Approval, Legacy Operating Exception, and Exception Revocation notices for the same project. "
        "Answer exactly as '<ProjectName> | <City>'."
    )
    task_meta = {
        'task_slice': 'dependency_necessary',
        'supports_current_city_chain': True,
        'candidate_handles': ['Beacon-037-74', 'Mosaic-061-26', 'Quartz-060-12'],
        'target_exception_state': 'revoked',
    }
    ans, expl = agent._try_autofinish(question, task_meta=task_meta)
    assert ans == 'Project_0037 | City_1'
    assert 'D_REVOCATION_0037' in expl
