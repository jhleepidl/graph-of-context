from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .env import CorpusEnv


@dataclass
class TaskScopedToolBox:
    """ToolBox that can be re-scoped per task.

    Many real benchmarks (HotpotQA, Lost-in-the-Middle, etc.) provide a *task-local*
    context set. If we build one global retriever for the whole benchmark, the
    agent can leak across tasks by retrieving a passage from a different example.

    This ToolBox supports `set_task(task)` and rebuilds a small retriever over the
    task's docs (typically <= 50 docs), keeping evaluation clean and reproducible.

    Expected task.meta format:
      task.meta["docs"] = [{"docid","title","content",(url)}...]
    """

    retriever_kind: str = "bm25"
    faiss_dim: int = 384
    default_topk: int = 10

    _env: Optional[CorpusEnv] = None
    _task_id: Optional[str] = None

    def set_task(self, task: Any):
        meta = getattr(task, "meta", None) or {}
        docs = meta.get("docs")
        if not docs:
            raise ValueError("TaskScopedToolBox requires task.meta['docs']")
        self._env = CorpusEnv.from_docs(docs, retriever_kind=self.retriever_kind, faiss_dim=self.faiss_dim)
        self._task_id = getattr(task, "id", None)

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("TaskScopedToolBox.search called before set_task(task)")
        return self._env.search(query=query, topk=topk or self.default_topk)

    def open_page(self, docid: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("TaskScopedToolBox.open_page called before set_task(task)")
        return self._env.open_page(docid=docid, url=url)
