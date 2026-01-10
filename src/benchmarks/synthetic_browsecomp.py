from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import json
from pathlib import Path

from .base import Task, Benchmark
from ..synth_data import make_corpus_and_tasks
from ..env import CorpusEnv
from ..tools import ToolBox
from ..metrics import exact_match, docid_coverage

class SyntheticBrowseComp(Benchmark):
    name = "synthetic_browsecomp"

    def prepare(
        self,
        data_dir: str,
        n_entities: int = 80,
        n_tasks: int = 50,
        distractors_per_entity: int = 2,
        noise_docs: int = 120,
        seed: int = 7,
        **kwargs
    ) -> Dict[str, Any]:
        data = Path(data_dir)
        data.mkdir(parents=True, exist_ok=True)
        corpus_path = data / "corpus.json"
        tasks_path = data / "tasks.json"

        make_corpus_and_tasks(
            out_corpus_path=str(corpus_path),
            out_tasks_path=str(tasks_path),
            n_entities=n_entities,
            n_tasks=n_tasks,
            distractors_per_entity=distractors_per_entity,
            noise_docs=noise_docs,
            seed=seed,
            long_horizon=bool(kwargs.get("long_horizon", False)),
            long_desc_words=int(kwargs.get("long_desc_words", 320)),
            related_degree=int(kwargs.get("related_degree", 3)),
            n_projects_per_task=int(kwargs.get("n_projects_per_task", 10)),
            hop_steps=int(kwargs.get("hop_steps", 4)),
            long_task_ratio=float(kwargs.get("long_task_ratio", 0.7)),
        )
        return {
            "corpus_path": str(corpus_path),
            "tasks_path": str(tasks_path),
            "n_entities": n_entities,
            "n_tasks": n_tasks,
            "distractors_per_entity": distractors_per_entity,
            "noise_docs": noise_docs,
            "seed": seed,
        }

    def load_tasks(self, data_dir: str, limit: Optional[int] = None, **kwargs) -> List[Task]:
        tasks_path = Path(data_dir) / "tasks.json"
        raw = json.load(open(tasks_path, "r", encoding="utf-8"))
        if limit is not None:
            raw = raw[:limit]
        out: List[Task] = []
        for r in raw:
            out.append(Task(
                id=r["id"],
                question=r["question"],
                answer=r["answer"],
                entities=r.get("entities"),
                required=r.get("required"),
                gold_docids=r.get("gold_docids"),
                meta={k:v for k,v in r.items() if k not in {"id","question","answer","entities","required","gold_docids"}}
            ))
        return out

    def build_tools(self, data_dir: str, **kwargs):
        corpus_path = Path(data_dir) / "corpus.json"
        retriever_kind = kwargs.get("retriever_kind", "bm25")
        faiss_dim = int(kwargs.get("faiss_dim", 384))
        env = CorpusEnv.from_json(str(corpus_path), retriever_kind=retriever_kind, faiss_dim=faiss_dim)
        return ToolBox(env=env)

    def evaluate(self, pred_answer: str, pred_expl: str, task: Task) -> Dict[str, Any]:
        correct = exact_match(pred_answer, task.answer)
        cov = docid_coverage(pred_expl, task.gold_docids or [])
        return {"correct": correct, "docid_cov": cov}
