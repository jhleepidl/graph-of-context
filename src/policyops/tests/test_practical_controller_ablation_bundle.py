from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.runners import llm as llm_runner


@dataclass
class _DummyTask:
    id: str = 't1'
    question: str = 'q'
    answer: str = 'a'
    turns: list | None = None
    meta: dict | None = None


class _DummyBenchmark:
    name = 'dummy'

    def load_tasks(self, data_dir: str, limit=None, **kwargs):
        return [_DummyTask(turns=[], meta={})]

    def build_tools(self, data_dir: str, **kwargs):
        return object()

    def evaluate(self, pred: str, explanation: str, task: _DummyTask):
        return {'correct': True, 'docid_cov': 1.0}


class _DummyAgent:
    def __init__(self, llm, tools, mem, cfg, controller_llm=None, bandit_controller=None):
        self.cfg = cfg
        self.mem = mem

    def run(self, *args, **kwargs):
        return {'answer': 'a', 'explanation': '', 'usage': {'total_tokens': 1}, 'tool_stats': {}, 'steps': 1, 'elapsed_sec': 0.0}


def test_run_llm_accepts_new_practical_ablation_methods(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(llm_runner, 'make_openai_client', lambda **kwargs: object())
    monkeypatch.setattr(llm_runner, 'ToolLoopLLMAgent', _DummyAgent)
    bench = _DummyBenchmark()
    out_results = tmp_path / 'results.jsonl'
    out_report = tmp_path / 'report.json'

    llm_runner.run_llm(
        benchmark=bench,
        data_dir='.',
        methods=[
            'FullHistory-Prove',
            'SimilarityOnly-Prove',
            'ProxySummary-Prove',
            'GoC-Closure-Only',
            'GoC-ForkOnly',
            'GoC-Mixed-Heuristic',
        ],
        out_results_path=str(out_results),
        out_report_path=str(out_report),
        task_limit=1,
        parallel_tasks=1,
    )

    assert out_report.exists()
