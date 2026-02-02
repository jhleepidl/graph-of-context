from __future__ import annotations

from typing import Any, Dict, List, Optional
import unicodedata
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


def _norm_title(s: str) -> str:
    """Normalize evidence titles for robust matching.

    Wikipedia titles sometimes appear with different Unicode normalization
    forms (composed vs decomposed). We canonicalize with NFKC and normalize
    whitespace.
    """
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    # Keep underscores as-is; just trim and collapse whitespace.
    s = " ".join(s.strip().split())
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

        # Optional: restrict to a subset of gold labels (e.g., ['supports','refutes'])
        label_filter = kwargs.get("label_filter", None)
        label_filter_set = None
        if label_filter:
            try:
                if isinstance(label_filter, str):
                    label_filter_set = {normalize_text(x) for x in label_filter.split(",") if x.strip()}
                elif isinstance(label_filter, (list, tuple, set)):
                    label_filter_set = {normalize_text(str(x)) for x in label_filter}
            except Exception:
                label_filter_set = None
            if label_filter_set:
                label_filter_set = {_norm_label(x) for x in label_filter_set}

        # --- Difficulty / augmentation knobs (aligned with HotpotQA setting style) ---
        # Basic multi-turn pressure
        filler_turns = int(kwargs.get("filler_turns", 0))
        filler_kind = str(kwargs.get("filler_kind", "confidence"))
        require_joint_output = bool(kwargs.get("require_joint_output", True))
        answer_format = str(kwargs.get("answer_format", "json")).lower()
        titles_f1_threshold = float(kwargs.get("titles_f1_threshold", 1.0))

        # Doc-level pressure
        doc_max_chars = int(kwargs.get("doc_max_chars", 0))
        doc_repeat = int(kwargs.get("doc_repeat", 1))
        branch_trap_k = int(kwargs.get("branch_trap_k", 0))

        # Two-stage commit (commit evidence_titles via return -> final label)
        two_stage = bool(kwargs.get("two_stage", False))
        closed_book_final = bool(kwargs.get("closed_book_final", False))
        delay_after_stage1 = int(kwargs.get("delay_after_stage1", 0))

        # Anaphoric / trajectory chain (adds horizon without extra tool calls)
        anaphoric_mode = str(kwargs.get("anaphoric_mode", "level")).lower()  # level|trajectory
        anaphoric_level = int(kwargs.get("anaphoric_level", 0))
        trajectory_chain_turns = int(kwargs.get("trajectory_chain_turns", 0))
        trajectory_chain_kind = str(kwargs.get("trajectory_chain_kind", "masked_refs")).lower()
        trajectory_chain_closed_book = bool(kwargs.get("trajectory_chain_closed_book", False))

        # Controller-side knobs (passed through task.meta; used by ToolLoopLLMAgent)
        noise_nodes_after_stage1 = int(kwargs.get("noise_nodes_after_stage1", 0))
        noise_node_chars = int(kwargs.get("noise_node_chars", 320))
        noise_seed = int(kwargs.get("noise_seed", 7))
        return_gating_min_open_pages = int(kwargs.get("return_gating_min_open_pages", 0))
        return_gating_min_steps = int(kwargs.get("return_gating_min_steps", 0))

        # GoC folding knobs (optional; forwarded to agent via task.meta)
        goc_fold_policy = kwargs.get("goc_fold_policy", None)
        goc_dfs_switch_keep_last = kwargs.get("goc_dfs_switch_keep_last", None)

        def _make_fillers(n: int) -> List[str]:
            turns = []
            for _ in range(int(n)):
                if filler_kind == "confidence":
                    turns.append("(filler) Give a confidence (0-100) number only. Do NOT call finish.")
                else:
                    turns.append("(filler) Waiting for follow-up. Do NOT call finish.")
            return turns

        tasks: List[Task] = []

        def _tokenize_title(t: str) -> set:
            import re
            toks = re.findall(r"[A-Za-z0-9]+", (t or "").lower())
            return set(x for x in toks if len(x) >= 3)

        def _title_overlap(a: str, b: str) -> float:
            sa, sb = _tokenize_title(a), _tokenize_title(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / max(1, min(len(sa), len(sb)))

        def _apply_branch_trap(docs: List[Dict[str, Any]], gold_titles: List[str], k: int) -> List[Dict[str, Any]]:
            """Reorder docs to surface 'near-miss' distractors before gold evidence.

            This is a benchmark-level augmentation (applies equally to all methods).
            """
            if k <= 0 or not gold_titles:
                return docs
            gold_set = {str(t) for t in gold_titles}
            gold_docs = [d for d in docs if str(d.get("title")) in gold_set]
            other = [d for d in docs if str(d.get("title")) not in gold_set]
            if not gold_docs or not other:
                return docs
            # Score non-gold titles by max overlap to any gold title
            scored = []
            for d in other:
                t = str(d.get("title") or "")
                mx = max((_title_overlap(t, gt) for gt in gold_titles), default=0.0)
                scored.append((mx, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            traps = [d for _, d in scored[:k]]
            rest = [d for _, d in scored[k:]]
            # Put traps first, then gold, then rest
            return traps + gold_docs + rest

        for i, ex in enumerate(rows):
            claim = str(ex.get("claim") or ex.get("question") or "").strip()
            if not claim:
                continue
            label = _norm_label(str(ex.get("label") or ex.get("verdict") or ""))
            if label_filter_set and (label not in label_filter_set):
                continue
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
                # Apply branch-trap ordering for harder wrong-branch failures.
                try:
                    docs = _apply_branch_trap(docs, gold_titles, branch_trap_k)
                except Exception:
                    pass

                intro = (
                    "You have access to a task-local document set via tools search/open_page.\n"
                    f"Q1: CLAIM: {claim}\n"
                    "Determine whether the claim is supported, refuted, or not enough info.\n"
                    "IMPORTANT (FEVER-style): output supports/refutes ONLY if the docs contain an explicit statement that proves/refutes the claim. "
                    "If it requires inference beyond what is written, or you cannot find a direct statement, output not_enough_info. "
                    "Do NOT use outside/world knowledge.\n"
                    "Do NOT call finish until instructed by a follow-up.\n"
                    "If you need the next follow-up, CALL `return`."
                )

                def _make_trajectory_checkpoint(j: int) -> str:
                    base = (
                        f"[FOLLOW-UP 1.{j} / TRAJECTORY-CHECKPOINT] "
                        "Do NOT call finish. This checkpoint does NOT repeat Q1. "
                    )
                    if trajectory_chain_kind == "echo_titles":
                        base += (
                            "CALL `return` with args.message as JSON only: "
                            "{\"primary_title\":\"<the FIRST committed title>\",\"secondary_title\":\"<the SECOND committed title (or empty)>\"}. "
                            "The titles MUST EXACTLY match what you returned in FOLLOW-UP 1 (same spelling and order)."
                        )
                    else:
                        # Default: masked reference (weakens similarity-only reliance on fresh title tokens)
                        base += (
                            "CALL `return` with args.message as JSON only: "
                            "{\"page_a\":\"FIRST_FROM_FOLLOWUP1\",\"page_b\":\"SECOND_FROM_FOLLOWUP1\",\"order_rule\":\"A_then_B\"}. "
                            "Do NOT include the actual titles; use these placeholders exactly."
                        )
                    if trajectory_chain_closed_book:
                        base += " [CLOSED-BOOK]"
                    return base

                if two_stage:
                    follow1 = (
                        "[FOLLOW-UP 1 / COMMIT] Provide the evidence document titles you will rely on (ONLY if they contain an explicit statement about the claim). "
                        "DO NOT call finish. Instead CALL `return` with args.message as JSON only: "
                        "{\"evidence_titles\":[\"Title A\",\"Title B\"]}. "
                        "For not_enough_info, return an empty list: {\"evidence_titles\":[]}. "
                        "Use the exact TITLE strings you saw in open_page (copy/paste; case + punctuation must match)."
                    )

                    # Stage 2 final: optionally anaphoric (do not re-state the claim) and optionally closed-book.
                    if answer_format == "json":
                        if anaphoric_mode == "trajectory" or anaphoric_level >= 2:
                            follow2 = (
                                "[FOLLOW-UP 2 / FINAL] Now CALL finish. "
                                "This follow-up does NOT repeat Q1/CLAIM. Use only the evidence you committed in FOLLOW-UP 1. "
                                "In finish.args.answer output JSON only with keys: label, evidence_titles. "
                                "label in {supports, refutes, not_enough_info}. "
                                "Reminder: supports/refutes ONLY if the committed evidence contains an explicit statement; otherwise use not_enough_info. "
                                "evidence_titles MUST EXACTLY match what you returned in FOLLOW-UP 1 (same spelling and order)."
                            )
                        else:
                            follow2 = (
                                "[FOLLOW-UP 2 / FINAL] Now CALL finish. "
                                f"CLAIM (repeated): {claim}\n"
                                "In finish.args.answer output JSON only with keys: label, evidence_titles. "
                                "label in {supports, refutes, not_enough_info}. "
                                "Reminder: supports/refutes ONLY if the committed evidence contains an explicit statement; otherwise use not_enough_info. "
                                "evidence_titles MUST EXACTLY match what you returned in FOLLOW-UP 1."
                            )
                    else:
                        follow2 = (
                            "[FOLLOW-UP 2 / FINAL] Now CALL finish. Output two lines: label: ... and evidence_titles: ..."
                        )

                    if closed_book_final:
                        follow2 += " [CLOSED-BOOK]"

                    stage2_fillers = _make_fillers(delay_after_stage1) if delay_after_stage1 > 0 else []
                    checkpoints = [_make_trajectory_checkpoint(j + 1) for j in range(max(0, trajectory_chain_turns))]
                    turns = [intro] + _make_fillers(filler_turns) + [follow1] + checkpoints + stage2_fillers + [follow2]
                else:
                    # One-stage late-binding (baseline)
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
                    if closed_book_final:
                        follow += " [CLOSED-BOOK]"
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

                    # Agent/controller knobs
                    "closed_book_final": closed_book_final,
                    "noise_nodes_after_stage1": noise_nodes_after_stage1,
                    "noise_node_chars": noise_node_chars,
                    "noise_seed": noise_seed,
                    "return_gating_min_open_pages": return_gating_min_open_pages,
                    "return_gating_min_steps": return_gating_min_steps,

                    # Finish schema hints (used by ToolLoopLLMAgent for robust JSON salvage)
                    "finish_answer_format": "fever_json" if (answer_format == "json" and variant not in {"single", "label_only"}) else "",

                    # Optional GoC fold knobs (bench-level control)
                    **({"goc_fold_policy": goc_fold_policy} if goc_fold_policy is not None else {}),
                    **({"goc_dfs_switch_keep_last": goc_dfs_switch_keep_last} if goc_dfs_switch_keep_last is not None else {}),
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

        pred_titles = (
            extract_list_field(obj, "evidence_titles")
            or extract_list_field(obj, "supporting_titles")
            or extract_list_field(obj, "titles")
        )
        # Normalize titles to avoid false mismatches due to Unicode normalization
        # (e.g., "Café" as composed vs decomposed).
        pred_titles_n = [_norm_title(x) for x in (pred_titles or [])]
        gold_titles = list(gold.get("evidence_titles") or [])
        gold_titles_n = [_norm_title(x) for x in gold_titles]
        f1_titles = set_f1(pred_titles_n, gold_titles_n) if gold_titles_n else (1.0 if not pred_titles_n else 0.0)
        ok2 = f1_titles >= titles_f1_threshold

        correct = bool(ok1 if (not require_joint) else (ok1 and ok2))
        correct_strict = bool(strict_json and correct)

        return {
            "correct": correct,
            "correct_strict": correct_strict,
            "docid_cov": cov,
            "titles_f1": f1_titles,
        }
