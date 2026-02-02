from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import gzip
import json
import random
import re
from pathlib import Path

from .base import Task, Benchmark
from .session_utils import (
    robust_qa_match,
    parse_structured_answer,
    title_match,
    token_f1,
)
from ..metrics import docid_coverage
from ..task_tools import TaskScopedToolBox


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".gz":
        rows = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    # plain jsonl
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_answer(ex: Dict[str, Any]) -> str:
    if "answer" in ex and ex["answer"] is not None:
        return str(ex["answer"]).strip()
    ans = ex.get("answers")
    if isinstance(ans, list) and ans:
        return str(ans[0]).strip()
    if isinstance(ans, str):
        return ans.strip()
    # fallback
    return str(ex.get("gold", "")).strip()


def _get_question(ex: Dict[str, Any]) -> str:
    for k in ("question", "query", "q"):
        if k in ex and ex[k]:
            return str(ex[k]).strip()
    return ""


def _get_contexts(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    ctxs = ex.get("contexts") or ex.get("ctxs") or ex.get("documents") or ex.get("passages")
    if not isinstance(ctxs, list):
        return []
    out = []
    for c in ctxs:
        if isinstance(c, dict):
            title = str(c.get("title", "") or c.get("heading", "") or "").strip()
            text = c.get("text") or c.get("content") or c.get("passage") or c.get("paragraph") or ""
            out.append({
                "title": title,
                "text": str(text),
                "is_gold": bool(c.get("is_gold") or c.get("gold") or c.get("has_answer")),
                "score": float(c.get("score") or 0.0),
            })
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append({"title": str(c[0]), "text": str(c[1]), "is_gold": False, "score": 0.0})
    return out


def _find_gold_index(ctxs: List[Dict[str, Any]], answer: str) -> int:
    for i, c in enumerate(ctxs):
        if c.get("is_gold"):
            return i
    ans = (answer or "").strip()
    if ans:
        a = ans.lower()
        for i, c in enumerate(ctxs):
            if a in (c.get("text") or "").lower():
                return i
    return 0


def _select_contexts(
    ctxs: List[Dict[str, Any]],
    *,
    gold_idx: int,
    total_ctx: int,
    branch_trap_k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Select a subset of contexts, keeping gold and optionally adding branch traps."""
    rng = random.Random(seed)
    if not ctxs:
        return [], 0
    gold_idx = max(0, min(gold_idx, len(ctxs) - 1))
    gold = ctxs[gold_idx]

    others = [c for i, c in enumerate(ctxs) if i != gold_idx]

    def _overlap_score(c: Dict[str, Any]) -> int:
        # heuristic: overlap between gold title tokens and candidate title/text tokens
        g_toks = set((gold.get("title") or "").lower().split())
        c_toks = set(((c.get("title") or "") + " " + (c.get("text") or ""))
                     .lower().split())
        return len(g_toks & c_toks)

    traps = []
    if branch_trap_k > 0 and others:
        traps = sorted(others, key=_overlap_score, reverse=True)[: int(branch_trap_k)]
        # Remove traps from others so we don't double select.
        trap_ids = set(id(x) for x in traps)
        others = [c for c in others if id(c) not in trap_ids]

    remaining_needed = max(0, int(total_ctx) - 1 - len(traps))
    rng.shuffle(others)
    picked = traps + others[:remaining_needed]
    rng.shuffle(picked)

    selected = [gold] + picked
    # Recompute gold_idx within selected list
    new_gold_idx = 0
    return selected, new_gold_idx


def _position_gold(ctxs: List[Dict[str, Any]], gold_idx: int, position: str, seed: int) -> Tuple[List[Dict[str, Any]], int]:
    if not ctxs:
        return ctxs, gold_idx
    position = (position or "original").lower()
    if position == "original":
        return ctxs, gold_idx
    rng = random.Random(seed)
    gold_idx = max(0, min(gold_idx, len(ctxs)-1))
    gold = ctxs[gold_idx]
    rest = [c for i, c in enumerate(ctxs) if i != gold_idx]
    if position == "front":
        new = [gold] + rest
        return new, 0
    if position == "back":
        new = rest + [gold]
        return new, len(new)-1
    if position == "middle":
        mid = len(rest)//2
        new = rest[:mid] + [gold] + rest[mid:]
        return new, mid
    if position == "random":
        rng.shuffle(rest)
        pos = rng.randint(0, len(rest))
        new = rest[:pos] + [gold] + rest[pos:]
        return new, pos
    return ctxs, gold_idx


def _make_filler_turns(n: int, kind: str) -> List[str]:
    kind = (kind or "confidence").lower()
    turns = []
    for i in range(int(n)):
        if kind == "confidence":
            turns.append("(filler) Before the follow-up arrives, briefly state your current confidence (0-100) as a number only. Do NOT call finish.")
        elif kind == "paraphrase":
            turns.append("(filler) Paraphrase the original question in one sentence. Do NOT call finish.")
        elif kind == "summarize":
            turns.append("(filler) Summarize the key evidence you have so far in <= 2 bullet points. Do NOT call finish.")
        else:
            turns.append("(filler) Waiting for a follow-up question. Do NOT call finish.")
    return turns


def _extract_evidence_sentence(text: str, answer: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # split into coarse sentences
    sents = [s.strip() for s in re.split(r"(?<=[\.!\?])\s+", t) if s.strip()]
    if not sents:
        sents = [t]
    a = (answer or "").strip().lower()
    if a:
        for s in sents:
            if a in s.lower():
                return s[:240]
    return sents[0][:240]


class LostInMiddle(Benchmark):
    """Lost-in-the-Middle style benchmark (task-local contexts).

    This implementation is intentionally *data-format tolerant* and focuses on
    supporting multi-turn late-binding variants that highlight memory
    traceability and dependency recovery.

    Expected inputs (one of the following):
      - A JSONL/JSONL.GZ file with fields like question/answer/contexts/ctxs.
      - Or a prepared JSONL with the same information.
    """

    name = "lost_in_middle"

    def prepare(self, data_dir: str, **kwargs) -> Dict[str, Any]:
        # No downloader (offline-safe). We just validate the file exists.
        raw_path = kwargs.get("raw_path") or kwargs.get("path")
        if raw_path:
            p = Path(raw_path)
            if not p.exists():
                raise FileNotFoundError(str(p))
            return {"raw_path": str(p)}
        return {"note": "Provide --bench_cfg with raw_path to the dataset (jsonl or jsonl.gz)."}

    def load_tasks(self, data_dir: str, limit: Optional[int] = None, **kwargs) -> List[Task]:
        raw_path = kwargs.get("raw_path") or kwargs.get("path")
        if not raw_path:
            # default expected location
            raw_path = str(Path(data_dir) / "lost_in_middle" / "data.jsonl.gz")

        rows = _load_jsonl(raw_path)
        if limit is not None:
            rows = rows[:limit]

        seed = int(kwargs.get("seed", 7))
        total_ctx = int(kwargs.get("total_ctx", kwargs.get("n_ctx", 16)))
        branch_trap_k = int(kwargs.get("branch_trap_k", 0))
        gold_position = str(kwargs.get("gold_position", "middle"))
        variant = str(kwargs.get("variant", "late_title"))
        filler_turns = int(kwargs.get("filler_turns", 0))
        filler_kind = str(kwargs.get("filler_kind", "confidence"))
        require_joint_output = bool(kwargs.get("require_joint_output", True))
        answer_format = str(kwargs.get("answer_format", "json")).lower()
        f1_threshold = float(kwargs.get("f1_threshold", 0.8))
        evidence_f1_threshold = float(kwargs.get("evidence_f1_threshold", 0.6))
        doc_max_chars = int(kwargs.get("doc_max_chars", 0))
        doc_repeat = int(kwargs.get("doc_repeat", 1))

        # --- HotpotQA-aligned multi-turn augmentations ---
        two_stage = bool(kwargs.get("two_stage", False))
        closed_book_final = bool(kwargs.get("closed_book_final", False))
        delay_after_stage1 = int(kwargs.get("delay_after_stage1", 0))

        anaphoric_mode = str(kwargs.get("anaphoric_mode", "trajectory")).lower()  # level|trajectory
        anaphoric_level = int(kwargs.get("anaphoric_level", 0))
        trajectory_chain_turns = int(kwargs.get("trajectory_chain_turns", 0))
        trajectory_chain_kind = str(kwargs.get("trajectory_chain_kind", "masked_refs")).lower()
        trajectory_chain_closed_book = bool(kwargs.get("trajectory_chain_closed_book", False))

        # Agent/controller knobs (forwarded via task.meta)
        noise_nodes_after_stage1 = int(kwargs.get("noise_nodes_after_stage1", 0))
        noise_node_chars = int(kwargs.get("noise_node_chars", 320))
        noise_seed = int(kwargs.get("noise_seed", 7))
        return_gating_min_open_pages = int(kwargs.get("return_gating_min_open_pages", 0))
        return_gating_min_steps = int(kwargs.get("return_gating_min_steps", 0))

        # Optional GoC fold knobs (bench-level control)
        goc_fold_policy = kwargs.get("goc_fold_policy", None)
        goc_dfs_switch_keep_last = kwargs.get("goc_dfs_switch_keep_last", None)

        out: List[Task] = []

        for i, ex in enumerate(rows):
            q = _get_question(ex)
            a = _get_answer(ex)
            ctxs = _get_contexts(ex)
            if not q or not a or not ctxs:
                continue

            gold_idx = _find_gold_index(ctxs, a)

            selected_ctxs, gold_idx2 = _select_contexts(
                ctxs,
                gold_idx=gold_idx,
                total_ctx=total_ctx,
                branch_trap_k=branch_trap_k,
                seed=(seed + i),
            )
            selected_ctxs, gold_idx3 = _position_gold(selected_ctxs, gold_idx2, gold_position, seed=(seed + i))

            # Build docs
            docs = []
            gold_title = ""
            gold_text = ""
            gold_docid = ""
            for j, c in enumerate(selected_ctxs):
                title = (c.get("title") or "Untitled").strip() or "Untitled"
                text = (c.get("text") or "").strip()
                if doc_max_chars > 0:
                    text = text[:doc_max_chars]
                if doc_repeat > 1 and text:
                    text = ("\n\n".join([text] * doc_repeat))
                docid = f"D_LIM_{i:06d}_{j:02d}"
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
                docs.append({"docid": docid, "title": title, "content": text, "url": url})
                if j == gold_idx3:
                    gold_title = title
                    gold_text = text
                    gold_docid = docid

            # Build multi-turn prompts
            turns: Optional[List[str]] = None
            gold_struct: Dict[str, Any] = {"a1": a}
            if variant.lower() in {"late_title", "late_binding_title", "2turn_title"}:
                gold_struct["a2"] = gold_title
            elif variant.lower() in {"late_sentence", "late_binding_sentence", "2turn_sentence"}:
                gold_struct["a2"] = _extract_evidence_sentence(gold_text, a)
            else:
                # single-turn
                gold_struct = {"a1": a}

            if variant.lower() != "single":
                intro = (
                    f"You have access to a task-local document set via tools search/open_page.\n"
                    f"Q1: {q}\n"
                    "Use tools to find evidence. Do NOT call finish until instructed by a follow-up.\n"
                    "If you need the next follow-up, CALL `return`."
                )
                follow = ""

                def _make_trajectory_checkpoint(j: int) -> str:
                    base = (
                        f"[FOLLOW-UP 1.{j} / TRAJECTORY-CHECKPOINT] "
                        "Do NOT call finish. This checkpoint does NOT repeat Q1. "
                    )
                    if trajectory_chain_kind == "echo_titles":
                        base += (
                            "CALL `return` with args.message as JSON only: "
                            "{\"primary_title\":\"<the committed title>\"}. "
                            "The title MUST EXACTLY match what you returned in FOLLOW-UP 1."
                        )
                    else:
                        base += (
                            "CALL `return` with args.message as JSON only: "
                            "{\"page_a\":\"ONLY_FROM_FOLLOWUP1\",\"order_rule\":\"A\"}. "
                            "Do NOT include the actual title; use this placeholder exactly."
                        )
                    if trajectory_chain_closed_book:
                        base += " [CLOSED-BOOK]"
                    return base

                if two_stage and ("a2" in gold_struct) and ("title" in variant.lower()):
                    # Stage 1: commit the evidence title (used later under closed-book + anaphora).
                    follow1 = (
                        "[FOLLOW-UP 1 / COMMIT] Identify the ONE Wikipedia title of the document you will cite as evidence. "
                        "DO NOT call finish. Instead CALL `return` with args.message as JSON only: "
                        "{\"supporting_titles\":[\"<Title>\"]}. "
                        "Use the exact TITLE string you saw in open_page."
                    )

                    # Stage 2: final (optionally do not re-state Q1)
                    if answer_format == "json":
                        if anaphoric_mode == "trajectory" or anaphoric_level >= 2:
                            follow2 = (
                                "[FOLLOW-UP 2 / FINAL] Now CALL finish. "
                                "This follow-up does NOT repeat Q1. Use only the committed title from FOLLOW-UP 1. "
                                "In finish.args.answer output JSON only with keys a1 and a2. "
                                "a1 = answer to Q1. a2 = the evidence title. "
                                "a2 MUST EXACTLY match what you returned in FOLLOW-UP 1 (same spelling)."
                            )
                        else:
                            follow2 = (
                                "[FOLLOW-UP 2 / FINAL] Now CALL finish. "
                                f"Q1 (repeated): {q}\n"
                                "In finish.args.answer output JSON only with keys a1 and a2. "
                                "a1 = answer to Q1. a2 = the evidence title. "
                                "a2 MUST EXACTLY match what you returned in FOLLOW-UP 1."
                            )
                    else:
                        follow2 = (
                            "[FOLLOW-UP 2 / FINAL] Now CALL finish. "
                            "Output two lines: 'a1: ...' and 'a2: ...'. a2 must match your committed title."
                        )

                    if closed_book_final:
                        follow2 += " [CLOSED-BOOK]"

                    stage2_fillers = _make_filler_turns(delay_after_stage1, filler_kind) if delay_after_stage1 > 0 else []
                    checkpoints = [_make_trajectory_checkpoint(j + 1) for j in range(max(0, trajectory_chain_turns))]
                    turns = [intro] + _make_filler_turns(filler_turns, filler_kind) + [follow1] + checkpoints + stage2_fillers + [follow2]
                else:
                    # One-stage late-binding (baseline)
                    if "a2" in gold_struct:
                        if variant.lower().startswith("late_title") or "title" in variant.lower():
                            a2_desc = "the Wikipedia title of ONE document that contains the answer"
                        else:
                            a2_desc = "one evidence sentence (verbatim-ish) from a document that supports the answer"
                        if answer_format == "json":
                            follow = (
                                "FOLLOW-UP (late-binding): Now CALL finish. "
                                "In finish.args.answer, output a JSON object with keys a1 and a2. "
                                f"a1 = answer to Q1. a2 = {a2_desc}. "
                                "JSON only (no extra text)."
                            )
                        else:
                            follow = (
                                "FOLLOW-UP (late-binding): Now CALL finish. "
                                "In finish.args.answer, output two lines: 'a1: ...' and 'a2: ...'."
                            )
                    else:
                        follow = "FOLLOW-UP: Now CALL finish and provide the answer to Q1."

                    if closed_book_final:
                        follow += " [CLOSED-BOOK]"
                    turns = [intro] + _make_filler_turns(filler_turns, filler_kind) + [follow]

            # Gold answer string for logging
            gold_answer_str = json.dumps(gold_struct, ensure_ascii=False)
            if (not require_joint_output) and ("a2" in gold_struct):
                # If the caller only wants to score the late-binding a2, expose it as primary gold.
                gold_answer_str = str(gold_struct["a2"])

            out.append(Task(
                id=str(ex.get("id") or ex.get("_id") or f"lim_{i}"),
                question=turns[0] if turns else q,
                turns=turns,
                answer=gold_answer_str,
                gold_docids=[gold_docid] if gold_docid else None,
                meta={
                    "docs": docs,
                    "variant": variant,
                    "gold": gold_struct,
                    "f1_threshold": f1_threshold,
                    "evidence_f1_threshold": evidence_f1_threshold,
                    "require_joint_output": require_joint_output,

                    # Agent/controller knobs
                    "closed_book_final": closed_book_final,
                    "noise_nodes_after_stage1": noise_nodes_after_stage1,
                    "noise_node_chars": noise_node_chars,
                    "noise_seed": noise_seed,
                    "return_gating_min_open_pages": return_gating_min_open_pages,
                    "return_gating_min_steps": return_gating_min_steps,

                    # Finish schema hints (ToolLoopLLMAgent can salvage/normalize)
                    "finish_answer_format": (
                        "litm_json_title" if (answer_format == "json" and "title" in variant.lower() and variant.lower() != "single") else
                        "litm_json_sentence" if (answer_format == "json" and "sentence" in variant.lower() and variant.lower() != "single") else
                        ""
                    ),

                    **({"goc_fold_policy": goc_fold_policy} if goc_fold_policy is not None else {}),
                    **({"goc_dfs_switch_keep_last": goc_dfs_switch_keep_last} if goc_dfs_switch_keep_last is not None else {}),
                },
            ))

        return out

    def build_tools(self, data_dir: str, **kwargs):
        retriever_kind = kwargs.get("retriever_kind", "bm25")
        faiss_dim = int(kwargs.get("faiss_dim", 384))
        return TaskScopedToolBox(retriever_kind=retriever_kind, faiss_dim=faiss_dim)

    def evaluate(self, pred_answer: str, pred_expl: str, task: Task) -> Dict[str, Any]:
        meta = task.meta or {}
        variant = str(meta.get("variant", "late_title"))
        gold = (meta.get("gold") or {})
        require_joint = bool(meta.get("require_joint_output", True))
        f1_threshold = float(meta.get("f1_threshold", 0.8))
        ev_f1_threshold = float(meta.get("evidence_f1_threshold", 0.6))

        cov = docid_coverage(pred_expl, task.gold_docids or [])

        if variant.lower() == "single":
            ok, info = robust_qa_match(pred_answer, gold.get("a1", task.answer), f1_threshold=f1_threshold)
            return {
                "correct": bool(ok),
                "correct_strict": bool(ok),
                "pred_norm": info.get("pred_norm"),
                "gold_norm": info.get("gold_norm"),
                "docid_cov": cov,
            }

        obj, strict_json = parse_structured_answer(pred_answer)
        if not obj:
            return {
                "correct": False,
                "correct_strict": False,
                "pred_norm": None,
                "gold_norm": None,
                "docid_cov": cov,
            }

        # Map possible keys
        a1_pred = obj.get("a1") or obj.get("answer") or obj.get("ans") or ""
        a2_pred = obj.get("a2") or obj.get("title") or obj.get("evidence") or obj.get("evidence_title") or ""
        a1_gold = gold.get("a1", "")
        a2_gold = gold.get("a2", "")

        ok1, info1 = robust_qa_match(str(a1_pred), str(a1_gold), f1_threshold=f1_threshold)
        ok2 = False
        if "title" in variant.lower():
            ok2 = title_match(str(a2_pred), str(a2_gold))
        else:
            ok2 = token_f1(str(a2_pred), str(a2_gold)) >= ev_f1_threshold

        correct = bool(ok2 if (not require_joint) else (ok1 and ok2))
        correct_strict = bool(strict_json and correct)

        return {
            "correct": correct,
            "correct_strict": correct_strict,
            "pred_norm": info1.get("pred_norm"),
            "gold_norm": info1.get("gold_norm"),
            "docid_cov": cov,
        }
