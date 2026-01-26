from __future__ import annotations

from typing import Any, Dict, List, Optional
import gzip
import json
from pathlib import Path

from .base import Task, Benchmark
from .session_utils import (
    parse_structured_answer,
    extract_list_field,
    set_f1,
    normalize_text,
)
from ..metrics import docid_coverage
from ..task_tools import TaskScopedToolBox


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows: List[Dict[str, Any]] = []
    if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_label(s: str) -> str:
    s = normalize_text(s)
    # Common FEVER labels
    if "support" in s:
        return "supports"
    if "refute" in s:
        return "refutes"
    if "not enough" in s or "nei" in s or "unknown" in s or "insufficient" in s:
        return "not_enough_info"
    return s


class FeverPrepared(Benchmark):
    """FEVER-style fact verification, but using a prepared context file.

    Rationale:
      - The original FEVER dataset does not ship evidence sentence text.
      - Reproducing the full FEVER pipeline requires a Wikipedia dump or DB.

    This benchmark expects a *prepared* JSONL/JSONL.GZ where each row contains
    task-local docs with actual text.

    Required fields per row:
      - id: str/int
      - claim: str
      - label: one of {SUPPORTS, REFUTES, NOT ENOUGH INFO}
      - docs: list[{docid,title,content,(url)}]

    Optional:
      - evidence_titles: list[str]   # gold titles

    Variants:
      - "late_label_titles" (default): multi-turn; final answer includes label and evidence_titles.
      - "single": single-turn label only.
    """

    name = "fever_prepared"

    def prepare(self, data_dir: str, **kwargs) -> Dict[str, Any]:
        path = kwargs.get("path") or kwargs.get("prepared_path")
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(str(p))
            return {"path": str(p)}
        return {"note": "Provide --bench_cfg with prepared_path to FEVER prepared jsonl(.gz)."}

    def load_tasks(self, data_dir: str, limit: Optional[int] = None, **kwargs) -> List[Task]:
        path = kwargs.get("path") or kwargs.get("prepared_path")
        if not path:
            path = str(Path(data_dir) / "fever" / "fever_prepared.jsonl.gz")

        rows = _load_jsonl(path)
        if limit is not None:
            rows = rows[:limit]

        variant = str(kwargs.get("variant", "late_label_titles")).lower()
        filler_turns = int(kwargs.get("filler_turns", 0))
        filler_kind = str(kwargs.get("filler_kind", "confidence"))
        require_joint_output = bool(kwargs.get("require_joint_output", True))
        answer_format = str(kwargs.get("answer_format", "json")).lower()
        titles_f1_threshold = float(kwargs.get("titles_f1_threshold", 1.0))
        doc_max_chars = int(kwargs.get("doc_max_chars", 0))
        doc_repeat = int(kwargs.get("doc_repeat", 1))

        def _make_fillers(n: int) -> List[str]:
            turns = []
            for _ in range(int(n)):
                if filler_kind == "confidence":
                    turns.append("(filler) Give a confidence (0-100) number only. Do NOT call finish.")
                else:
                    turns.append("(filler) Waiting for follow-up. Do NOT call finish.")
            return turns

        tasks: List[Task] = []
        for i, ex in enumerate(rows):
            claim = str(ex.get("claim") or ex.get("question") or "").strip()
            if not claim:
                continue
            label = _norm_label(str(ex.get("label") or ex.get("verdict") or ""))
            docs = ex.get("docs") or ex.get("contexts") or []
            if not isinstance(docs, list) or not docs:
                continue

            # Optional difficulty scaling: truncate or repeat doc text.
            if doc_max_chars > 0 or doc_repeat > 1:
                new_docs = []
                for d in docs:
                    if not isinstance(d, dict):
                        continue
                    dd = dict(d)
                    text = str(dd.get("content") or dd.get("text") or "")
                    if doc_max_chars > 0:
                        text = text[:doc_max_chars]
                    if doc_repeat > 1 and text:
                        text = "\n\n".join([text] * doc_repeat)
                    # normalize field name for tools
                    dd["content"] = text
                    new_docs.append(dd)
                docs = new_docs

            gold_titles = ex.get("evidence_titles") or ex.get("gold_titles") or []
            if isinstance(gold_titles, str):
                gold_titles = [x.strip() for x in gold_titles.split(",") if x.strip()]
            if not isinstance(gold_titles, list):
                gold_titles = []

            gold_struct: Dict[str, Any] = {"label": label, "evidence_titles": gold_titles}
            turns: Optional[List[str]] = None
            if variant in {"single", "label_only"}:
                turns = None
            else:
                intro = (
                    "You have access to a task-local document set via tools search/open_page.\n"
                    f"CLAIM: {claim}\n"
                    "Determine whether the claim is supported, refuted, or not enough info.\n"
                    "Do NOT call finish until a follow-up arrives."
                )
                if answer_format == "json":
                    follow = (
                        "FOLLOW-UP (late-binding): Now CALL finish. "
                        "In finish.args.answer output JSON with keys: label, evidence_titles. "
                        "label in {supports, refutes, not_enough_info}. "
                        "evidence_titles is a list of titles used as evidence (can be empty for not_enough_info). "
                        "JSON only."
                    )
                else:
                    follow = (
                        "FOLLOW-UP (late-binding): Now CALL finish. "
                        "Output two lines: label: ... and evidence_titles: ..."
                    )
                turns = [intro] + _make_fillers(filler_turns) + [follow]

            gold_answer_str = json.dumps(gold_struct, ensure_ascii=False)
            if (not require_joint_output):
                gold_answer_str = label

            # mark gold docids by title match (if provided)
            gold_docids = []
            if gold_titles:
                gold_set = {str(t) for t in gold_titles}
                for d in docs:
                    if isinstance(d, dict) and str(d.get("title")) in gold_set and d.get("docid"):
                        gold_docids.append(str(d["docid"]))

            tasks.append(Task(
                id=str(ex.get("id") or ex.get("_id") or f"fever_{i}"),
                question=turns[0] if turns else claim,
                turns=turns,
                answer=gold_answer_str,
                gold_docids=gold_docids or None,
                meta={
                    "docs": docs,
                    "variant": variant,
                    "gold": gold_struct,
                    "titles_f1_threshold": titles_f1_threshold,
                    "require_joint_output": require_joint_output,
                }
            ))
        return tasks

    def build_tools(self, data_dir: str, **kwargs):
        retriever_kind = kwargs.get("retriever_kind", "bm25")
        faiss_dim = int(kwargs.get("faiss_dim", 384))
        return TaskScopedToolBox(retriever_kind=retriever_kind, faiss_dim=faiss_dim)

    def evaluate(self, pred_answer: str, pred_expl: str, task: Task) -> Dict[str, Any]:
        meta = task.meta or {}
        variant = str(meta.get("variant", "late_label_titles")).lower()
        gold = meta.get("gold") or {}
        require_joint = bool(meta.get("require_joint_output", True))
        titles_f1_threshold = float(meta.get("titles_f1_threshold", 1.0))
        cov = docid_coverage(pred_expl, task.gold_docids or [])

        if variant in {"single", "label_only"}:
            ok = _norm_label(pred_answer) == _norm_label(str(gold.get("label", "")))
            return {"correct": bool(ok), "correct_strict": bool(ok), "docid_cov": cov}

        obj, strict_json = parse_structured_answer(pred_answer)
        if not obj:
            return {"correct": False, "correct_strict": False, "docid_cov": cov}

        pred_label = _norm_label(str(obj.get("label") or obj.get("verdict") or ""))
        gold_label = _norm_label(str(gold.get("label") or ""))
        ok1 = (pred_label == gold_label) and bool(gold_label)

        pred_titles = extract_list_field(obj, "evidence_titles") or extract_list_field(obj, "titles")
        gold_titles = list(gold.get("evidence_titles") or [])
        f1_titles = set_f1(pred_titles, gold_titles) if gold_titles else (1.0 if not pred_titles else 0.0)
        ok2 = f1_titles >= titles_f1_threshold

        correct = bool(ok1 if (not require_joint) else (ok1 and ok2))
        correct_strict = bool(strict_json and correct)

        return {
            "correct": correct,
            "correct_strict": correct_strict,
            "docid_cov": cov,
            "titles_f1": f1_titles,
        }
