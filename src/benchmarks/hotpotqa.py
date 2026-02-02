from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import random
import re
from pathlib import Path

from .base import Task, Benchmark
from .session_utils import (
    robust_qa_match,
    parse_structured_answer,
    extract_list_field,
    set_f1,
    normalize_text,
)
from ..metrics import docid_coverage
from ..task_tools import TaskScopedToolBox


def _load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # HotpotQA releases are JSON lists.
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported HotpotQA JSON format")


def _normalize_context(ex: Dict[str, Any]) -> List[List[Any]]:
    """Normalize HotpotQA context to the canonical list-of-pairs format."""
    ctx = ex.get("context")
    if isinstance(ctx, list):
        return ctx

    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles")
        sents = ctx.get("sentences") or ctx.get("sentence") or ctx.get("sent")
        if isinstance(titles, list) and isinstance(sents, list):
            out: List[List[Any]] = []
            for t, ss in zip(titles, sents):
                out.append([t, ss])
            return out

    return []


def _normalize_supporting_facts(ex: Dict[str, Any]) -> List[List[Any]]:
    """Normalize supporting_facts to list-of-pairs [title, sent_idx]."""
    sf = ex.get("supporting_facts")
    if isinstance(sf, list):
        return sf
    if isinstance(sf, dict):
        titles = sf.get("title") or sf.get("titles")
        idxs = sf.get("sent_id") or sf.get("sent_ids") or sf.get("sent") or sf.get("sent_idx")
        if isinstance(titles, list) and isinstance(idxs, list):
            out: List[List[Any]] = []
            for t, i in zip(titles, idxs):
                out.append([t, i])
            return out
    return []


def _make_filler_turns(n: int, kind: str) -> List[str]:
    kind = (kind or "confidence").lower()
    turns = []
    for _ in range(int(n)):
        if kind == "confidence":
            turns.append("(filler) Give a confidence (0-100) number only. Do NOT call finish.")
        elif kind == "summarize":
            turns.append("(filler) Summarize the evidence you have so far in <= 2 bullet points. Do NOT call finish.")
        elif kind == "paraphrase":
            turns.append("(filler) Paraphrase the original question in one sentence. Do NOT call finish.")
        else:
            turns.append("(filler) Waiting for follow-up. Do NOT call finish.")
    return turns


def _tokenize_title(s: str) -> List[str]:
    # Keep it simple & stable: alnum tokens, lowercase.
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


def _title_overlap(a: str, b: str) -> int:
    ta = set(_tokenize_title(a))
    tb = set(_tokenize_title(b))
    if not ta or not tb:
        return 0
    return len(ta & tb)


def _apply_branch_trap(docs: List[Dict[str, Any]], gold_titles: List[str], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Make similarity-based retrieval more error-prone by placing near-gold distractors adjacent.

    This is *benchmark augmentation* (applies to all methods equally): we reorder the doc list.
    The goal is to increase wrong-branch risk for similarity-only baselines, making
    dependency-aware recovery more valuable.
    """
    k = int(k or 0)
    if k <= 0 or not gold_titles:
        return docs

    gold_set = set(gold_titles)
    gold_docs = [d for d in docs if d.get("title") in gold_set]
    rest = [d for d in docs if d.get("title") not in gold_set]
    if not gold_docs or not rest:
        return docs

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, d in enumerate(rest):
        title = str(d.get("title") or "")
        score = max(_title_overlap(title, gt) for gt in gold_titles)
        scored.append((score, i, d))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    trap = [d for s, _, d in scored[:k] if s > 0]
    if not trap:
        return docs

    trap_set = {id(d) for d in trap}
    rest2 = [d for d in rest if id(d) not in trap_set]
    rng.shuffle(rest2)

    # Interleave: (some rest) + trap + gold + (rest)
    # Put trap right before gold so a similarity seed is likely to pick it.
    out: List[Dict[str, Any]] = []
    # Put half of rest before, half after.
    mid = len(rest2) // 2
    out.extend(rest2[:mid])
    out.extend(trap)
    out.extend(gold_docs)
    out.extend(rest2[mid:])
    return out


# --- Multi-Commit DAG synthesis helpers ---
# These create a *benchmark-level* long-horizon episode by chaining multiple
# independent HotpotQA subtasks in one trajectory. This is designed to create
# natural dependency edges (commit -> merge -> final) without adding extra tools.

_MULTI_COMMIT_RULES = {
    "reduce_lex_min",
    "reduce_shortest",
    "reduce_longest",
    "reduce_len_then_lex",
}


def _mc_rule_desc(rule: str) -> str:
    r = (rule or "").lower().strip()
    if r == "reduce_lex_min":
        return "pick the winner whose normalized answer is lexicographically smallest"
    if r == "reduce_shortest":
        return "pick the winner whose normalized answer is shortest"
    if r == "reduce_longest":
        return "pick the winner whose normalized answer is longest"
    if r == "reduce_len_then_lex":
        return "pick the winner with shortest normalized answer; break ties by lexicographic order"
    return "pick the winner by reduce_lex_min"


def _mc_key(ans: str) -> str:
    # Reuse the benchmark's normalization (punct/article stripping).
    return normalize_text(str(ans or ""))


def _mc_better(rule: str, a: str, b: str) -> bool:
    """Return True if a beats b under rule."""
    r = (rule or "reduce_lex_min").lower().strip()
    ka = _mc_key(a)
    kb = _mc_key(b)
    if r == "reduce_lex_min":
        return ka < kb
    if r == "reduce_shortest":
        return (len(ka), ka) < (len(kb), kb)
    if r == "reduce_longest":
        return (len(ka), ka) > (len(kb), kb)
    if r == "reduce_len_then_lex":
        return (len(ka), ka) < (len(kb), kb)
    return ka < kb


def _mc_reduce_winner(rule: str, answers_by_commit: Dict[int, str]) -> int:
    """Deterministically reduce answers into a single winning commit index."""
    items = [(int(k), str(v)) for k, v in (answers_by_commit or {}).items()]
    items.sort(key=lambda x: x[0])
    if not items:
        return 1
    win_k, win_a = items[0]
    for k, a in items[1:]:
        if _mc_better(rule, a, win_a):
            win_k, win_a = k, a
    return int(win_k)


def _mc_build_merge_plan(n_commits: int, plan: str) -> List[Tuple[str, str, str]]:
    """Return a list of merge ops as (merge_id, left_id, right_id).

    Node ids are strings:
      - "C1".."Cn" are leaf commits
      - "M1".. are merge nodes

    The acting model is asked to output a *leaf* winner_commit (1..n) for each merge,
    even when left/right are merge nodes.
    """
    n = int(n_commits or 0)
    if n <= 1:
        return []
    p = (plan or "binary_tree").lower().strip()

    leaves = [f"C{i}" for i in range(1, n + 1)]
    merges: List[Tuple[str, str, str]] = []
    m_idx = 1

    if p in {"none", "no", "off"}:
        return []

    if p in {"chain", "linear"}:
        cur = leaves[0]
        for nxt in leaves[1:]:
            mid = f"M{m_idx}"
            merges.append((mid, cur, nxt))
            cur = mid
            m_idx += 1
        return merges

    # Default: balanced-ish binary tree tournament
    level = leaves
    while len(level) > 1:
        next_level: List[str] = []
        i = 0
        while i < len(level):
            if i == len(level) - 1:
                next_level.append(level[i])
                i += 1
                continue
            left = level[i]
            right = level[i + 1]
            mid = f"M{m_idx}"
            merges.append((mid, left, right))
            next_level.append(mid)
            m_idx += 1
            i += 2
        level = next_level
    return merges


class HotpotQA(Benchmark):
    """HotpotQA (distractor) with task-local context.

    This benchmark includes optional augmentations to emphasize long-horizon memory:
      - two_stage: stage-1 requires committing supporting_titles via `return`, stage-2 finish later.
      - delay_after_stage1: adds filler turns after stage-1 commit (adds extra LLM calls).
      - closed_book_final: disallow tools in the final stage (finish must rely on memory).
      - branch_trap_k: reorder to place near-gold distractors adjacent, increasing wrong-branch risk.
      - anaphoric_level: make final follow-up less self-contained (forces reliance on earlier trace).
      - noise_nodes_after_stage1: inject internal "noise" nodes into memory after stage-1 commit
        to increase context pressure *without adding extra LLM calls*.

    All augmentations apply to *all methods equally* (fair evaluation).
    """

    name = "hotpotqa"

    def prepare(self, data_dir: str, **kwargs) -> Dict[str, Any]:
        path = kwargs.get("path") or kwargs.get("raw_path")
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(str(p))
            return {"path": str(p)}
        return {"note": "Provide --bench_cfg with path to hotpotqa dev/train JSON."}

    def load_tasks(self, data_dir: str, limit: Optional[int] = None, **kwargs) -> List[Task]:
        path = kwargs.get("path") or kwargs.get("raw_path")
        if not path:
            path = str(Path(data_dir) / "hotpotqa" / "hotpot_dev_distractor_v1.json")

        rows = _load_json(path)

        # ---- Task selection controls (for reproducible per-task comparisons) ----
        # Provide either:
        #   - task_ids: list[str] or comma-separated string
        #   - task_ids_path: file with one id per line
        # or use shuffling for deterministic sampling:
        #   - shuffle: bool
        #   - task_sample_seed: int
        task_ids: Optional[List[str]] = None
        if "task_ids_path" in kwargs and kwargs.get("task_ids_path"):
            try:
                p = Path(str(kwargs.get("task_ids_path")))
                if p.exists():
                    task_ids = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                task_ids = None
        if task_ids is None and kwargs.get("task_ids"):
            ti = kwargs.get("task_ids")
            if isinstance(ti, str):
                task_ids = [x.strip() for x in ti.split(",") if x.strip()]
            elif isinstance(ti, list):
                task_ids = [str(x).strip() for x in ti if str(x).strip()]

        if task_ids:
            # Keep the requested ordering.
            by_id = {str(ex.get("_id") or ex.get("id")): ex for ex in rows}
            rows = [by_id[x] for x in task_ids if x in by_id]
        else:
            if bool(kwargs.get("shuffle", False)):
                ss = int(kwargs.get("task_sample_seed", kwargs.get("seed", 7)))
                rng = random.Random(ss)
                rows = list(rows)
                rng.shuffle(rows)

        # Multi-commit synthesis: if multi_commit_n>1, `limit` applies to the number of
        # *composite* tasks, so we need mc_n*limit raw examples.
        multi_commit_n_pre = int(kwargs.get("multi_commit_n", 1) or 1)
        if limit is not None:
            lim = int(limit)
            rows = rows[: (lim * multi_commit_n_pre if multi_commit_n_pre > 1 else lim)]

        seed = int(kwargs.get("seed", 7))
        variant = str(kwargs.get("variant", "late_support_titles")).lower()

        # Base long-horizon knobs
        filler_turns = int(kwargs.get("filler_turns", 0))
        filler_kind = str(kwargs.get("filler_kind", "confidence"))
        supporting_position = str(kwargs.get("supporting_position", "middle")).lower()
        doc_max_chars = int(kwargs.get("doc_max_chars", 0))
        doc_repeat = int(kwargs.get("doc_repeat", 1))

        # Evaluation knobs
        require_joint_output = bool(kwargs.get("require_joint_output", True))
        answer_format = str(kwargs.get("answer_format", "json")).lower()
        f1_threshold = float(kwargs.get("f1_threshold", 0.8))
        titles_f1_threshold = float(kwargs.get("titles_f1_threshold", 1.0))

        # Augmentation knobs (fair, benchmark-level)
        two_stage = bool(kwargs.get("two_stage", False))
        delay_after_stage1 = int(kwargs.get("delay_after_stage1", 0))
        closed_book_final = bool(kwargs.get("closed_book_final", False))
        # Anaphoric follow-up: make the FINAL prompt less self-contained.
        # 0 = baseline (explicit, self-contained)
        # 1 = remove extra reminders about Q1/commit (still specifies output schema)
        # 2 = ultra-minimal final prompt (schema-only; maximally anaphoric)
        anaphoric_level = int(kwargs.get("anaphoric_level", 0))
        # Alternative anaphora mode that is *trajectory-dependent* rather than
        # "hide the entity". This keeps the problem definition stable but forces
        # reliance on the session trace (commit + ordering).
        #
        # Supported values:
        #   - "level" (default): use anaphoric_level behavior above
        #   - "trajectory": FINAL prompt refers only to the previously COMMITTED titles
        #
        # This is a benchmark-level lever and applies to all methods equally.
        anaphoric_mode = str(kwargs.get("anaphoric_mode", "level") or "level").lower()
        # Optional extra post-commit turns that *do not* restate Q1 and require the
        # agent to preserve commitment/order under context pressure.
        # Each checkpoint turn asks for a JSON payload via `return`.
        trajectory_chain_turns = int(kwargs.get("trajectory_chain_turns", 0))
        trajectory_chain_closed_book = bool(kwargs.get("trajectory_chain_closed_book", True))
        trajectory_chain_kind = str(kwargs.get("trajectory_chain_kind", "echo_titles") or "echo_titles").lower()
        # Noise injection (no extra LLM calls): implemented in the agent based on task_meta.
        noise_nodes_after_stage1 = int(kwargs.get("noise_nodes_after_stage1", 0))
        noise_node_chars = int(kwargs.get("noise_node_chars", 320))
        noise_seed = int(kwargs.get("noise_seed", seed))
        # Return gating (discourage too-early return which collapses horizon)
        return_gating_min_steps = int(kwargs.get("return_gating_min_steps", 0))
        return_gating_min_open_pages = int(kwargs.get("return_gating_min_open_pages", 0))
        # Similarity trap
        branch_trap_k = int(kwargs.get("branch_trap_k", 0))

        # Multi-commit synthesis (benchmark-level). When enabled, we stitch N independent
        # HotpotQA subtasks into one episode with COMMIT -> MERGE -> FINAL turns.
        multi_commit_n = int(kwargs.get("multi_commit_n", multi_commit_n_pre) or 1)

        # Accept a few legacy/sweep key names:
        #  - multi_commit_merge_rule: older name for compose_rule (winner selection)
        #  - multi_commit_shuffle: older name for doc shuffle
        _mc_rule = (
            kwargs.get("multi_commit_compose_rule")
            or kwargs.get("multi_commit_merge_rule")
            or "reduce_lex_min"
        )
        multi_commit_rule = str(_mc_rule or "reduce_lex_min").lower().strip()
        if multi_commit_rule in {"alpha_min", "alphabetical_min", "alpha"}:
            multi_commit_rule = "reduce_lex_min"

        multi_commit_merge_plan = str(kwargs.get("multi_commit_merge_plan", "binary_tree") or "binary_tree").lower().strip()
        multi_commit_include_merges = bool(kwargs.get("multi_commit_include_merges", True))
        multi_commit_merge_closed_book = bool(kwargs.get("multi_commit_merge_closed_book", True))
        multi_commit_doc_shuffle = bool(kwargs.get("multi_commit_doc_shuffle", kwargs.get("multi_commit_shuffle", True)))
        # Optional: inject noise after EVERY commit (not just stage-1). Implemented in the agent.
        noise_nodes_after_commit = int(kwargs.get("noise_nodes_after_commit", 0))

        # Lost-in-the-middle positioning
        gold_at = kwargs.get("gold_at")
        gold_at = int(gold_at) if gold_at is not None else None

        # Synonyms for prior presets
        if supporting_position in {"early"}:
            supporting_position = "front"
        if supporting_position in {"late"}:
            supporting_position = "back"

        rng = random.Random(seed)
        out: List[Task] = []

        # ---- Multi-Commit DAG (benchmark-level synthesis) ----
        # If enabled, stitch mc_n independent examples into a single long-horizon task.
        # Each subtask requires a COMMIT (return), optional MERGE commits, and one FINAL finish.
        if int(multi_commit_n) > 1:
            mc_n = int(multi_commit_n)
            if multi_commit_rule not in _MULTI_COMMIT_RULES:
                multi_commit_rule = "reduce_lex_min"

            merge_ops = _mc_build_merge_plan(mc_n, multi_commit_merge_plan) if bool(multi_commit_include_merges) else []

            n_groups = len(rows) // mc_n
            for g in range(n_groups):
                chunk = rows[g * mc_n : (g + 1) * mc_n]

                docs_all: List[Dict[str, Any]] = []
                answers_by_commit: Dict[int, str] = {}
                gold_titles_by_commit: Dict[int, List[str]] = {}
                gold_docids_by_commit: Dict[int, List[str]] = {}

                for si, ex in enumerate(chunk, start=1):
                    q = str(ex.get("question", "")).strip()
                    a = str(ex.get("answer", "")).strip()
                    if not q or not a:
                        continue

                    context = _normalize_context(ex)
                    if not context:
                        continue

                    # Build docs for this subtask (one doc per title)
                    docs_i: List[Dict[str, Any]] = []
                    for j, pair in enumerate(context):
                        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                            continue
                        title = str(pair[0]).strip() or "Untitled"
                        sents = pair[1]
                        if isinstance(sents, list):
                            text = " ".join([str(s) for s in sents])
                        else:
                            text = str(sents)
                        if doc_max_chars > 0:
                            text = text[:doc_max_chars]
                        if doc_repeat > 1 and text:
                            text = ("\n\n".join([text] * doc_repeat))
                        docid = f"D_HP_MC_{g:06d}_{si:02d}_{j:02d}"
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        docs_i.append({"docid": docid, "title": title, "content": text, "url": url})

                    sf = _normalize_supporting_facts(ex)
                    gold_titles: List[str] = []
                    if isinstance(sf, list):
                        for it in sf:
                            if isinstance(it, (list, tuple)) and it:
                                gold_titles.append(str(it[0]))
                    seen = set()
                    gold_titles = [t for t in gold_titles if not (t in seen or seen.add(t))]

                    # Position gold evidence per-subtask (lost-in-middle stress)
                    if supporting_position != "original" and gold_titles:
                        gold_set = set(gold_titles)
                        gold_docs = [d for d in docs_i if d.get("title") in gold_set]
                        rest_docs = [d for d in docs_i if d.get("title") not in gold_set]
                        rng.shuffle(rest_docs)
                        if supporting_position == "front":
                            docs_i = gold_docs + rest_docs
                        elif supporting_position == "back":
                            docs_i = rest_docs + gold_docs
                        elif supporting_position == "middle":
                            mid = len(rest_docs) // 2
                            docs_i = rest_docs[:mid] + gold_docs + rest_docs[mid:]
                        elif supporting_position == "lost_in_middle":
                            rest = rest_docs
                            pos = gold_at if (gold_at is not None) else (len(rest) // 2)
                            pos = max(0, min(len(rest), int(pos)))
                            docs_i = rest[:pos] + gold_docs + rest[pos:]
                        else:
                            rng.shuffle(gold_docs)
                            pos = rng.randint(0, len(rest_docs))
                            docs_i = rest_docs[:pos] + gold_docs + rest_docs[pos:]

                    # Similarity trap within subtask
                    docs_i = _apply_branch_trap(docs_i, gold_titles, branch_trap_k, rng)

                    answers_by_commit[si] = a
                    gold_titles_by_commit[si] = gold_titles
                    gold_docids_by_commit[si] = [d["docid"] for d in docs_i if d.get("title") in set(gold_titles)]
                    docs_all.extend(docs_i)

                if len(answers_by_commit) != mc_n:
                    continue

                # Optionally shuffle all docs to mix subtasks
                if bool(multi_commit_doc_shuffle):
                    rng.shuffle(docs_all)

                winner = _mc_reduce_winner(multi_commit_rule, answers_by_commit)
                winner = max(1, min(mc_n, int(winner)))

                gold_struct = {
                    "a1": str(answers_by_commit.get(winner, "") or "").strip(),
                    "supporting_titles": list(gold_titles_by_commit.get(winner, []) or []),
                    "selected_commit": int(winner),
                }

                intro = (
                    "You have access to a task-local document set via tools search/open_page.\n"
                    f"This is a MULTI-COMMIT DAG episode with N={mc_n} subtasks.\n"
                    "For EACH subtask i, you must use tools search/open_page to gather evidence, then COMMIT by calling the `return` tool.\n"
                    "When you call `return`, it MUST be a single JSON tool call with args.message set to a JSON object.\n"
                    "Commit schema (return.args.message): {\"commit\":<i>,\"a1\":\"...\",\"supporting_titles\":[\"Title A\",\"Title B\"]}\n"
                    "Do NOT call finish until the FINAL.\n"
                    f"The overall winner is selected deterministically by: {_mc_rule_desc(multi_commit_rule)}.\n"
                    "You will later MERGE commits (memory-only) and then FINISH with the winner.\n"
                )

                commit_turns: List[str] = []
                for i in range(1, mc_n + 1):
                    # Keep COMMIT prompts self-contained (they contain the question), but do not repeat titles later.
                    q_i = str(chunk[i-1].get("question", ""))
                    commit_turns.append(
                        f"[SUBTASK {i} / COMMIT] Q{i}: {q_i}\n"
                        "Use tools. DO NOT call finish. When ready, CALL the `return` tool with args.message containing the commit JSON.\n"
                        f"Example tool call: {{\"tool\":\"return\",\"args\":{{\"message\":{{\"commit\":{i},\"a1\":\"...\",\"supporting_titles\":[\"Title A\",\"Title B\"]}}}}}}"
                    )

                # IMPORTANT: keep SUBTASK turns as follow-ups so the current user message
                # always contains only the active subtask (matches later injection behavior).
                turns: List[str] = [intro]
                if commit_turns:
                    turns.extend(commit_turns)

                for mi, (mid, left, right) in enumerate(merge_ops, start=1):
                    msg = (
                        f"[MERGE {mi} / COMMIT] You must MERGE {left} vs {right} under the SAME deterministic rule.\n"
                        "Do NOT restate earlier answers/titles. Use ONLY your previously committed information.\n"
                        f"CALL the `return` tool with args.message: {{\"merge\":{mi},\"left\":\"{left}\",\"right\":\"{right}\",\"winner_commit\":<1..{mc_n}>}}\n"
                        f"Example tool call: {{\"tool\":\"return\",\"args\":{{\"message\":{{\"merge\":{mi},\"left\":\"{left}\",\"right\":\"{right}\",\"winner_commit\":1}}}}}}"
                    )
                    if bool(multi_commit_merge_closed_book):
                        msg += "\n[CLOSED-BOOK] Do NOT use tools now (no search/open_page)."
                    turns.append(msg)

                final = (
                    "[FINAL / FINISH] CALL finish NOW.\n"
                    "Use ONLY your committed information.\n"
                    "Call the `finish` tool with args.answer as a JSON object with keys: a1, supporting_titles, selected_commit.\n"
                    "a1 MUST be the winner's answer. supporting_titles MUST be the winner's supporting_titles.\n"
                    "selected_commit MUST be the winner leaf commit number.\n"
                    "Example tool call: {\"tool\":\"finish\",\"args\":{\"answer\":{\"a1\":\"...\",\"supporting_titles\":[\"Title A\",\"Title B\"],\"selected_commit\":1},\"explanation\":\"Evidence docids: D_HP_...\",\"confidence\":\"\"}}\n"
                )

                if bool(closed_book_final):
                    final += "[CLOSED-BOOK] Do NOT use tools now (no search/open_page).\n"

                turns.append(final)
                out.append(Task(
                    id=f"hp_mc_{g:06d}",
                    question=turns[0],
                    turns=turns,
                    answer=json.dumps(gold_struct, ensure_ascii=False),
                    gold_docids=gold_docids_by_commit.get(winner, None),
                    meta={
                        "benchmark": "hotpotqa",
                        "docs": docs_all,
                        "variant": variant,
                        "gold": gold_struct,
                        "f1_threshold": f1_threshold,
                        "titles_f1_threshold": titles_f1_threshold,
                        "require_joint_output": True,
                        "finish_answer_format": "hotpotqa_json",
                        "finish_supporting_titles_n": 2,
                        "two_stage": False,
                        "multi_commit": True,
                        "multi_commit_n": int(mc_n),
                        "multi_commit_merge_plan": str(multi_commit_merge_plan),
                        "return_gating_per_turn": True,
                        "closed_book_final": bool(closed_book_final),
                        "anaphoric_level": int(anaphoric_level),
                        "anaphoric_mode": str(anaphoric_mode),
                        "trajectory_chain_turns": int(trajectory_chain_turns),
                        "trajectory_chain_kind": str(trajectory_chain_kind),
                        "trajectory_chain_closed_book": bool(trajectory_chain_closed_book),
                        "noise_nodes_after_stage1": int(noise_nodes_after_stage1),
                        "noise_nodes_after_commit": int(noise_nodes_after_commit),
                        "noise_node_chars": int(noise_node_chars),
                        "noise_seed": int(noise_seed),
                        "return_gating_min_steps": int(return_gating_min_steps),
                        "return_gating_min_open_pages": int(return_gating_min_open_pages),
                        "branch_trap_k": int(branch_trap_k),
                        "supporting_position": supporting_position,
                        "multi_commit_n": int(mc_n),
                        "multi_commit_compose_rule": str(multi_commit_rule),
                        "multi_commit_merge_plan": str(multi_commit_merge_plan),
                        "multi_commit_include_merges": bool(multi_commit_include_merges),
                        "multi_commit_doc_shuffle": bool(multi_commit_doc_shuffle),
                        "multi_commit_merge_closed_book": bool(multi_commit_merge_closed_book),
                        "schema_autofix_commit_mismatch": bool(kwargs.get("schema_autofix_commit_mismatch", False)),
                        # GoC folding policy overrides
                        "goc_fold_policy": kwargs.get("goc_fold_policy", kwargs.get("fold_policy", None)),
                        "goc_phase_end_fold": kwargs.get("goc_phase_end_fold", None),
                        "goc_dfs_hi_mult": kwargs.get("goc_dfs_hi_mult", None),
                        "goc_dfs_lo_mult": kwargs.get("goc_dfs_lo_mult", None),
                        "goc_dfs_roll_keep_last": kwargs.get("goc_dfs_roll_keep_last", None),
                        "goc_dfs_roll_min_chunk": kwargs.get("goc_dfs_roll_min_chunk", None),
                        "goc_dfs_switch_keep_last": kwargs.get("goc_dfs_switch_keep_last", None),
                        "goc_dfs_phase_keep_last": kwargs.get("goc_dfs_phase_keep_last", None),
                        "goc_pef_hi_mult": kwargs.get("goc_pef_hi_mult", None),
                        "goc_pef_lo_mult": kwargs.get("goc_pef_lo_mult", None),
                        "goc_pef_roll_keep_last": kwargs.get("goc_pef_roll_keep_last", None),
                        "goc_pef_roll_min_chunk": kwargs.get("goc_pef_roll_min_chunk", None),
                    },
                ))

            # Respect original limit (composite tasks).
            if limit is not None:
                out = out[: int(limit)]
            return out

        for idx, ex in enumerate(rows):
            q = str(ex.get("question", "")).strip()
            a = str(ex.get("answer", "")).strip()
            if not q or not a:
                continue

            context = _normalize_context(ex)
            if not context:
                continue

            # Build docs (one doc per title)
            docs: List[Dict[str, Any]] = []
            for j, pair in enumerate(context):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                title = str(pair[0]).strip() or "Untitled"
                sents = pair[1]
                if isinstance(sents, list):
                    text = " ".join([str(s) for s in sents])
                else:
                    text = str(sents)
                if doc_max_chars > 0:
                    text = text[:doc_max_chars]
                if doc_repeat > 1 and text:
                    text = ("\n\n".join([text] * doc_repeat))
                docid = f"D_HP_{idx:06d}_{j:02d}"
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                docs.append({"docid": docid, "title": title, "content": text, "url": url})

            # Supporting titles from gold
            sf = _normalize_supporting_facts(ex)
            gold_titles: List[str] = []
            if isinstance(sf, list):
                for it in sf:
                    if isinstance(it, (list, tuple)) and it:
                        gold_titles.append(str(it[0]))
            # Unique order-preserving
            seen = set()
            gold_titles = [t for t in gold_titles if not (t in seen or seen.add(t))]

            # Reorder docs to move supporting titles around (stress memory).
            if supporting_position != "original" and gold_titles:
                gold_set = set(gold_titles)
                gold_docs = [d for d in docs if d.get("title") in gold_set]
                rest_docs = [d for d in docs if d.get("title") not in gold_set]
                rng.shuffle(rest_docs)

                if supporting_position == "front":
                    docs = gold_docs + rest_docs
                elif supporting_position == "back":
                    docs = rest_docs + gold_docs
                elif supporting_position == "middle":
                    mid = len(rest_docs) // 2
                    docs = rest_docs[:mid] + gold_docs + rest_docs[mid:]
                elif supporting_position == "lost_in_middle":
                    # Place gold around an explicit index to emulate LITM setups.
                    rest = rest_docs
                    pos = gold_at if (gold_at is not None) else (len(rest) // 2)
                    pos = max(0, min(len(rest), int(pos)))
                    docs = rest[:pos] + gold_docs + rest[pos:]
                else:
                    # random insertion
                    rng.shuffle(gold_docs)
                    pos = rng.randint(0, len(rest_docs))
                    docs = rest_docs[:pos] + gold_docs + rest_docs[pos:]

            # Apply a similarity trap by placing near-gold distractors adjacent.
            docs = _apply_branch_trap(docs, gold_titles, branch_trap_k, rng)

            # Gold docids
            gold_docids = [d["docid"] for d in docs if d.get("title") in set(gold_titles)]

            # Prompts
            gold_struct: Dict[str, Any] = {"a1": a, "supporting_titles": gold_titles}

            turns: Optional[List[str]]
            if variant in {"single", "q_only"}:
                turns = None
            else:
                intro = (
                    "You have access to a task-local document set via tools search/open_page.\n"
                    f"Q1: {q}\n"
                    "Use tools to find evidence. Do NOT call finish until instructed by a follow-up.\n"
                    "If you need the next follow-up, CALL `return`."
                )

                if two_stage:
                    # Stage 1: commit supporting titles via return.
                    follow1 = (
                        "[FOLLOW-UP 1 / COMMIT] Provide the supporting Wikipedia titles you will rely on. "
                        "DO NOT call finish. Instead CALL `return` with args.message as a JSON object: "
                        "{\"supporting_titles\":[\"Title A\",\"Title B\"]}. "
                        "Use the exact TITLE strings you saw in open_page. JSON only."
                    )

                    def _make_trajectory_checkpoint(i: int) -> str:
                        """Optional post-commit checkpoint.

                        These turns are *benchmark augmentations* that increase horizon/pressure
                        without changing the gold answer. They are intentionally anaphoric and
                        refer to the session trajectory (commit + order) rather than re-stating Q1.
                        """
                        base = (
                            f"[FOLLOW-UP 1.{i} / TRAJECTORY-CHECKPOINT] "
                            "Do NOT call finish. "
                            "This checkpoint does NOT repeat Q1. "
                        )

                        if trajectory_chain_kind == "echo_titles":
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"primary_title\":\"<the FIRST committed title>\",\"secondary_title\":\"<the SECOND committed title>\"}. "
                                "The titles MUST EXACTLY match what you returned in FOLLOW-UP 1 (same spelling and order)."
                            )
                        elif trajectory_chain_kind in {"masked_refs", "masked_ref", "masked"}:
                            # Masked checkpoint: add horizon/pressure without re-emitting titles.
                            # This is designed to weaken similarity-only methods that benefit from
                            # fresh lexical overlap near the end of a long trajectory.
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"page_a\":\"FIRST_FROM_FOLLOWUP1\",\"page_b\":\"SECOND_FROM_FOLLOWUP1\",\"order_rule\":\"A_then_B\"}. "
                                "Do NOT include the actual titles; use these placeholders exactly."
                            )
                        elif trajectory_chain_kind in {"opened_first_ab", "open_order_ab", "open_order"}:
                            # Order-dependent but still masked: forces reliance on trajectory
                            # (which page was opened first) without restating titles.
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"opened_first\":\"A_or_B\",\"A\":\"FIRST_FROM_FOLLOWUP1\",\"B\":\"SECOND_FROM_FOLLOWUP1\"}. "
                                "Set opened_first to 'A' if you opened the first (A) committed page earlier than the second (B), otherwise 'B'. "
                                "Do NOT include the actual titles."
                            )
                        elif trajectory_chain_kind == "bridge_entity":
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"bridge_entity\":\"<the entity that links the TWO committed pages>\"}. "
                                "Use an entity name that appears in BOTH committed pages." 
                            )
                        elif trajectory_chain_kind == "evidence_hints":
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"primary_hint\":\"<short clue from FIRST committed page>\",\"secondary_hint\":\"<short clue from SECOND committed page>\"}. "
                                "Each hint must be <= 6 words and should be useful later." 
                            )
                        else:
                            # Default to the safest checkpoint.
                            base += (
                                "CALL `return` with args.message as JSON only: "
                                "{\"primary_title\":\"<the FIRST committed title>\",\"secondary_title\":\"<the SECOND committed title>\"}."
                            )

                        if trajectory_chain_closed_book:
                            base += "\n[CLOSED-BOOK] Do NOT use tools now (no search/open_page). Use memory only."
                        return base

                    # Stage 2: final answer (late-binding). Optional anaphoric variants.
                    if answer_format == "json":
                        if anaphoric_mode == "trajectory":
                            # Trajectory-dependent FINAL:
                            # - Does NOT hide the task definition by removing Q1 tokens at random.
                            # - Instead, it forces reliance on the *trace*: the committed titles and their order.
                            # This weakens similarity-only baselines in a fair way by making the final prompt
                            # low-overlap with the evidence text while keeping the objective unchanged.
                            follow2 = (
                                "[FOLLOW-UP 2 / FINAL / TRAJECTORY] CALL finish NOW. "
                                "The original question (Q1) was asked earlier in this session and is NOT repeated here. "
                                "Answer that SAME Q1. "
                                "You MUST use ONLY the two Wikipedia pages you provided in FOLLOW-UP 1. "
                                "In finish.args.answer output JSON only with keys: a1, supporting_titles. "
                                "supporting_titles MUST EXACTLY match the two page titles you provided in FOLLOW-UP 1, in the SAME order (no extra titles). "
                                "Example: {\"a1\":\"...\",\"supporting_titles\":[\"Title A\",\"Title B\"]}."
                            )
                        else:
                            if anaphoric_level <= 0:
                                follow2 = (
                                    f"[FOLLOW-UP 2 / FINAL] Now CALL finish. Q1: {q} "
                                    "In finish.args.answer output a JSON object with keys: a1, supporting_titles. "
                                    "a1 MUST be concise (no extra words). If yes/no, exactly 'yes' or 'no' (lowercase). "
                                    "supporting_titles MUST be a list of Wikipedia page titles (usually 2). "
                                    "Prefer the committed titles from FOLLOW-UP 1. JSON only."
                                )
                            elif anaphoric_level == 1:
                                follow2 = (
                                    f"[FOLLOW-UP 2 / FINAL] Now CALL finish. Q1: {q} "
                                    "In finish.args.answer output a JSON object with keys: a1, supporting_titles. "
                                    "JSON only (no prose, no markdown)."
                                )
                            else:
                                # Ultra-anaphoric but still well-formed.
                                # We intentionally DO NOT restate Q1 here: the agent must recover the earlier
                                # problem statement via memory/trace (lost-in-the-middle stress).
                                # We keep a concrete JSON example to avoid schema failures.
                                follow2 = (
                                    "[FOLLOW-UP 2 / FINAL] CALL finish NOW. "
                                    "The original question (Q1) was asked earlier in this session and is NOT repeated here. "
                                    "finish.args.answer MUST be JSON only (no prose/markdown). "
                                    "Required keys: a1, supporting_titles. "
                                    "supporting_titles MUST EXACTLY match the two titles you previously returned (same order). "
                                    "Example: {\"a1\":\"...\",\"supporting_titles\":[\"Title A\",\"Title B\"]}."
                                )
                    else:
                        follow2 = (
                            "[FOLLOW-UP 2 / FINAL] Now CALL finish. Output:\n"
                            "a1: <answer>\n"
                            "supporting_titles: <title1>, <title2>"
                        )
                    if closed_book_final:
                        follow2 += "\n[CLOSED-BOOK] Do NOT use tools now (no search/open_page). Use memory only."

                    # Optional trajectory checkpoints (increase horizon / dependency on commit+order).
                    checkpoints: List[str] = []
                    if (anaphoric_mode == "trajectory") and int(trajectory_chain_turns) > 0:
                        for i in range(1, int(trajectory_chain_turns) + 1):
                            checkpoints.append(_make_trajectory_checkpoint(i))

                    turns = (
                        [intro]
                        + _make_filler_turns(filler_turns, filler_kind)
                        + [follow1]
                        + checkpoints
                        + _make_filler_turns(delay_after_stage1, filler_kind)
                        + [follow2]
                    )
                else:
                    # Single-stage late-binding: finish only on final follow-up.
                    if answer_format == "json":
                        follow = (
                            "FOLLOW-UP (late-binding): Now CALL finish. "
                            "In finish.args.answer output a JSON object with keys: a1, supporting_titles. "
                            "a1 is the final answer to Q1 and MUST be concise (no extra words/sentences). "
                            "If Q1 is a yes/no question, a1 MUST be exactly 'yes' or 'no' (lowercase). "
                            "supporting_titles MUST be a list of Wikipedia page titles that support your reasoning "
                            "(usually 2) and SHOULD match the exact TITLE lines you saw in open_page. "
                            "JSON only (no markdown, no prose)."
                        )
                    else:
                        follow = (
                            "FOLLOW-UP (late-binding): Now CALL finish. "
                            "In finish.args.answer output:\n"
                            "a1: <answer>\n"
                            "supporting_titles: <title1>, <title2>"
                        )
                    turns = [intro] + _make_filler_turns(filler_turns, filler_kind) + [follow]

            gold_answer_str = json.dumps(gold_struct, ensure_ascii=False)
            if (not require_joint_output):
                gold_answer_str = ", ".join(gold_titles)

            out.append(Task(
                id=str(ex.get("_id") or ex.get("id") or f"hp_{idx}"),
                question=turns[0] if turns else q,
                turns=turns,
                answer=gold_answer_str,
                gold_docids=gold_docids or None,
                meta={
                    "benchmark": "hotpotqa",
                    "docs": docs,
                    "variant": variant,
                    "gold": gold_struct,
                    "f1_threshold": f1_threshold,
                    "titles_f1_threshold": titles_f1_threshold,
                    "require_joint_output": require_joint_output,
                    # Finish-output contract (used by the agent to validate/salvage finish.args.answer)
                    "finish_answer_format": "hotpotqa_json" if (answer_format == "json" and require_joint_output) else "text",
                    "finish_supporting_titles_n": 2,
                    # Augmentation config (passed to runner/agent as task_meta)
                    "two_stage": bool(two_stage),
                    "delay_after_stage1": int(delay_after_stage1),
                    "closed_book_final": bool(closed_book_final),
                    "anaphoric_level": int(anaphoric_level),
                    "anaphoric_mode": anaphoric_mode,
                    "trajectory_chain_turns": int(trajectory_chain_turns),
                    "trajectory_chain_kind": trajectory_chain_kind,
                    "trajectory_chain_closed_book": bool(trajectory_chain_closed_book),
                    "noise_nodes_after_stage1": int(noise_nodes_after_stage1),
                    "noise_node_chars": int(noise_node_chars),
                    "noise_seed": int(noise_seed),
                    # GoC folding policy overrides (optional; used by GoC only)
                    "goc_fold_policy": kwargs.get("goc_fold_policy", kwargs.get("fold_policy", None)),
                    "goc_phase_end_fold": kwargs.get("goc_phase_end_fold", None),
                    "goc_dfs_hi_mult": kwargs.get("goc_dfs_hi_mult", None),
                    "goc_dfs_lo_mult": kwargs.get("goc_dfs_lo_mult", None),
                    "goc_dfs_roll_keep_last": kwargs.get("goc_dfs_roll_keep_last", None),
                    "goc_dfs_roll_min_chunk": kwargs.get("goc_dfs_roll_min_chunk", None),
                    "goc_dfs_switch_keep_last": kwargs.get("goc_dfs_switch_keep_last", None),
                    "goc_dfs_phase_keep_last": kwargs.get("goc_dfs_phase_keep_last", None),
                    "goc_pef_hi_mult": kwargs.get("goc_pef_hi_mult", None),
                    "goc_pef_lo_mult": kwargs.get("goc_pef_lo_mult", None),
                    "goc_pef_roll_keep_last": kwargs.get("goc_pef_roll_keep_last", None),
                    "goc_pef_roll_min_chunk": kwargs.get("goc_pef_roll_min_chunk", None),
                    "return_gating_min_steps": int(return_gating_min_steps),
                    "return_gating_min_open_pages": int(return_gating_min_open_pages),
                    "branch_trap_k": int(branch_trap_k),
                    "supporting_position": supporting_position,
                },
            ))

        return out

    def build_tools(self, data_dir: str, **kwargs):
        retriever_kind = kwargs.get("retriever_kind", "bm25")
        faiss_dim = int(kwargs.get("faiss_dim", 384))
        return TaskScopedToolBox(retriever_kind=retriever_kind, faiss_dim=faiss_dim)

    def evaluate(self, pred_answer: str, pred_expl: str, task: Task) -> Dict[str, Any]:
        meta = task.meta or {}
        variant = str(meta.get("variant", "late_support_titles")).lower()
        gold = meta.get("gold") or {}
        require_joint = bool(meta.get("require_joint_output", True))
        f1_threshold = float(meta.get("f1_threshold", 0.8))
        titles_f1_threshold = float(meta.get("titles_f1_threshold", 1.0))

        cov = docid_coverage(pred_expl, task.gold_docids or [])

        if variant in {"single", "q_only"}:
            ok, info = robust_qa_match(pred_answer, gold.get("a1", ""), f1_threshold=f1_threshold)
            return {
                "correct": bool(ok),
                "correct_strict": bool(ok),
                "pred_norm": info.get("pred_norm"),
                "gold_norm": info.get("gold_norm"),
                "docid_cov": cov,
            }

        obj, strict_json = parse_structured_answer(pred_answer)
        if not obj:
            return {"correct": False, "correct_strict": False, "pred_norm": None, "gold_norm": None, "docid_cov": cov}

        a1_pred = obj.get("a1") or obj.get("answer") or obj.get("ans") or ""
        a1_gold = gold.get("a1", "")
        ok1, info1 = robust_qa_match(str(a1_pred), str(a1_gold), f1_threshold=f1_threshold)

        pred_titles = extract_list_field(obj, "supporting_titles")
        if not pred_titles:
            pred_titles = extract_list_field(obj, "titles") or extract_list_field(obj, "evidence_titles")
        gold_titles = list(gold.get("supporting_titles") or [])

        f1_titles = set_f1(pred_titles, gold_titles)
        ok2 = f1_titles >= titles_f1_threshold

        correct = bool(ok2 if (not require_joint) else (ok1 and ok2))
        correct_strict = bool(strict_json and correct)

        return {
            "correct": correct,
            "correct_strict": correct_strict,
            "pred_norm": info1.get("pred_norm"),
            "gold_norm": info1.get("gold_norm"),
            "docid_cov": cov,
            "support_titles_f1": f1_titles,
        }
