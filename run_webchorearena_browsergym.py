#!/usr/bin/env python3
"""
Run WebChoreArena tasks through BrowserGym using our memory managers.

Prereqs (high level):
  - Install BrowserGym + WebArena task package
  - Install Playwright + Chromium
  - Bring up the WebArena backend websites (docker-compose) and set base URL env vars

Example (small set):
  # NOTE: For WebChoreArena/WCA configs, prefer a *fixed* wrapper env id like
  # "browsergym/webarena" and inject per-task configs via --task_config_mode.
  python run_webchorearena_browsergym.py \
    --tasks_json path/to/WebChoreArena/BrowserGym/config_files/test_shopping.raw.json \
    --small_set_ids path/to/WebChoreArena/BrowserGym/config_files/small_set_ids.txt \
    --env_id browsergym/webarena \
    --method GoC --budget_active 2000 --budget_unfold 800 \
    --task_config_mode auto --new_env_per_task \
    --out results_wca_goc.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.browsergym_runner import BrowserGymRunConfig, run_browsergym_tasks
from src.llm_openai import make_openai_client
from src.memory import (
    AgentFoldRangeMemory,
    ContextFoldingDiscardMemory,
    FullHistoryMemory,
    GoCMemory,
    LinearSummaryMemory,
    SimilarityOnlyMemory,
)


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)

    # JSONL fallback
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_small_set_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    ids: Set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(line)
    return ids or None


def _filter_tasks(tasks: List[Dict[str, Any]], keep_ids: Optional[Set[str]]) -> List[Dict[str, Any]]:
    if not keep_ids:
        return tasks
    out = []
    for t in tasks:
        tid = t.get("task_id")
        if tid is None:
            continue
        if str(tid) in keep_ids:
            out.append(t)
    return out


def _make_mem(
    method: str,
    budget_active: int,
    budget_unfold: int,
    unfold_k: int,
    linear_every: int,
    fold_chunk: int,
    goc_fold_policy: str,
    goc_pef_backstop_mult: float,
    goc_pef_hi_mult: float,
    goc_pef_lo_mult: float,
    goc_pef_roll_keep_last: int,
    enforce_budget: bool,
):
    if method == "FullHistory":
        return FullHistoryMemory(budget_active=budget_active, budget_unfold=budget_unfold, enforce_budget=enforce_budget)
    if method == "ContextFolding-Discard":
        return ContextFoldingDiscardMemory(budget_active=budget_active, budget_unfold=budget_unfold, enforce_budget=enforce_budget)
    if method == "LinearSummary":
        return LinearSummaryMemory(budget_active=budget_active, budget_unfold=budget_unfold, summary_every=linear_every, enforce_budget=enforce_budget)
    if method == "AgentFold-Range":
        return AgentFoldRangeMemory(budget_active=budget_active, budget_unfold=budget_unfold, fold_chunk=fold_chunk, enforce_budget=enforce_budget)
    if method == "GoC":
        return GoCMemory(
            budget_active=budget_active,
            budget_unfold=budget_unfold,
            unfold_k=unfold_k,
            fold_policy=goc_fold_policy,
            pef_backstop_mult=float(goc_pef_backstop_mult),
            pef_hi_mult=float(goc_pef_hi_mult),
            pef_lo_mult=float(goc_pef_lo_mult),
            pef_roll_keep_last=int(goc_pef_roll_keep_last),
            enforce_budget=enforce_budget,
        )
    if method == "SimilarityOnly":
        return SimilarityOnlyMemory(budget_active=budget_active, budget_unfold=budget_unfold, unfold_k=unfold_k, enforce_budget=enforce_budget)
    raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tasks_json", required=True, help="Path to WebChoreArena task JSON (list[dict] or JSONL)")
    ap.add_argument("--small_set_ids", default=None, help="Optional path to small_set_ids.txt (filter tasks)")

    ap.add_argument("--method", default="GoC",
                    choices=["FullHistory", "ContextFolding-Discard", "LinearSummary", "AgentFold-Range", "SimilarityOnly", "GoC"])
    ap.add_argument("--budget_active", type=int, default=2000)
    ap.add_argument("--budget_unfold", type=int, default=800)
    ap.add_argument("--enforce_budget", action="store_true",
                    help="Enforce budget_active by folding/pruning. If omitted, budgets are NOT enforced (accuracy-first).")
    ap.add_argument("--unfold_k", type=int, default=6)
    ap.add_argument("--linear_every", type=int, default=8)
    ap.add_argument("--fold_chunk", type=int, default=10)

    ap.add_argument("--goc_fold_policy", default="pef_url", choices=["budget", "pef_url"],
                    help="GoC folding policy. 'pef_url' folds mainly on URL changes (BrowserGym/WebArena).")
    ap.add_argument("--goc_pef_backstop_mult", type=float, default=2.5,
                    help="Safety backstop multiplier for pef_url. If active_tokens > budget_active * mult, fall back to budget folding.")
    ap.add_argument("--goc_pef_hi_mult", type=float, default=1.25,
                    help="PEF hysteresis high-watermark multiplier. Fold when active_tokens > budget_active * hi_mult.")
    ap.add_argument("--goc_pef_lo_mult", type=float, default=0.85,
                    help="PEF hysteresis low-watermark multiplier. After folding, stop when active_tokens <= budget_active * lo_mult.")
    ap.add_argument("--goc_pef_roll_keep_last", type=int, default=10,
                    help="When rolling-folding within a URL episode, keep at least this many newest ACTIVE nodes.")

    # Loop/revisit guard (helps reduce noop/goto loops on the same page).
    ap.add_argument("--loop_guard", action="store_true", help="Enable loop/revisit guard.")
    ap.add_argument("--no_loop_guard", action="store_true", help="Disable loop/revisit guard.")
    ap.add_argument("--loop_guard_force_action", action="store_true", help="If the model proposes a previously-failed action, override it with a suggested alternative.")
    ap.add_argument("--loop_guard_no_force", action="store_true", help="Do not override model actions; only add guard hints.")
    ap.add_argument("--loop_guard_repeat_threshold", type=int, default=2)
    ap.add_argument("--loop_guard_noop_threshold", type=int, default=2)
    ap.add_argument("--loop_guard_ttl", type=int, default=10)

    ap.add_argument("--env_id", default="browsergym/webarena",
                    help="Gym env id (for WCA configs, prefer a fixed wrapper like browsergym/webarena).")
    ap.add_argument("--env_id_template", default=None,
                    help="Optional template like browsergym/webarena.{task_id} (for built-in WebArena task ids).")
    ap.add_argument("--task_config_mode", default="auto",
                    choices=["auto", "task_kwargs", "config_file", "none"],
                    help="How to inject per-task configs from WebChoreArena JSON into BrowserGym.")
    ap.add_argument("--new_env_per_task", action="store_true",
                    help="Create a fresh browser environment per task (safer, slower).")
    ap.add_argument("--reset_option_key", default="config_file",
                    help="When using config_file mode, pass options[{key}]=<path> to env.reset().")
    ap.add_argument("--auto_wait_noop", action="store_true",
                    help="If the task record has required_wait=true, insert a noop() step after each action.")

    ap.add_argument("--auto_wait_on_no_change", action="store_true",
                    help="If an action appears to make no progress (same URL/state signature), insert a one-step noop() to let the UI settle.")
    ap.add_argument("--no_auto_wait_on_no_change", action="store_true",
                    help="Disable auto settle-noop on apparent no-change states.")

    # WebArena: auto-submit when review distribution is fully observed (saves many steps).
    ap.add_argument("--auto_submit_review_distribution", action="store_true",
                    help="Automatically call send_msg_to_user(...) once a full 1..5-star review COUNT distribution is detected.")
    ap.add_argument("--no_auto_submit_review_distribution", action="store_true",
                    help="Disable auto-submit even if distribution is detected.")

    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--obs_truncate_chars", type=int, default=4000)
    ap.add_argument("--store_full_obs", action="store_true")


    ap.add_argument("--trace_dir", default="trace", help="Directory to write per-task JSONL traces (debugging).")
    ap.add_argument("--trace_tag", default=None, help="Optional tag appended to trace file names (default: method name).")
    ap.add_argument("--trace_include_prompt", action="store_true", help="Include prompt tail in traces (can be large).")
    ap.add_argument("--trace_prompt_chars", type=int, default=2000)
    ap.add_argument("--trace_include_obs", action="store_true", help="Include observation head in traces (can be large).")
    ap.add_argument("--trace_obs_chars", type=int, default=2000)

    ap.add_argument("--headless", action="store_true", help="Run headless browser (recommended for sweeps).")

    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature (Chat Completions only). Ignored for GPT-5* models by default.")

    # LLM backend selection
    ap.add_argument(
        "--api_mode",
        default="auto",
        choices=["auto", "chat", "responses"],
        help="Which OpenAI endpoint to use. auto=Responses for GPT-5*, Chat Completions otherwise.",
    )
    ap.add_argument(
        "--reasoning_effort",
        default=None,
        help="(Responses API / GPT-5*) reasoning.effort, e.g. minimal|low|medium|high. If omitted, model default applies.",
    )
    ap.add_argument(
        "--verbosity",
        default=None,
        choices=["low", "medium", "high"],
        help="(Responses API / GPT-5*) text.verbosity: low|medium|high",
    )
    ap.add_argument(
        "--max_output_tokens",
        type=int,
        default=None,
        help="(Responses API) max_output_tokens cap (includes reasoning tokens).",
    )

    ap.add_argument("--out", default="results_wca.jsonl")
    args = ap.parse_args()

    tasks = _load_tasks(args.tasks_json)
    keep_ids = _load_small_set_ids(args.small_set_ids)
    tasks = _filter_tasks(tasks, keep_ids)

    llm = make_openai_client(
        model=args.model,
        temperature=args.temperature,
        api_mode=args.api_mode,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_output_tokens=args.max_output_tokens,
        force_json=False,
    )
    mem = _make_mem(
        args.method,
        args.budget_active,
        args.budget_unfold,
        args.unfold_k,
        args.linear_every,
        args.fold_chunk,
        args.goc_fold_policy,
        args.goc_pef_backstop_mult,
        args.goc_pef_hi_mult,
        args.goc_pef_lo_mult,
        args.goc_pef_roll_keep_last,
        enforce_budget=bool(args.enforce_budget),
    )

    trace_tag = args.trace_tag if args.trace_tag is not None else args.method

    run_cfg = BrowserGymRunConfig(
        env_id=args.env_id,
        env_id_template=args.env_id_template,
        max_steps=args.max_steps,
        obs_truncate_chars=args.obs_truncate_chars,
        store_full_obs=args.store_full_obs,
        new_env_per_task=args.new_env_per_task,
        task_config_mode=args.task_config_mode,
        reset_option_key=args.reset_option_key,
        auto_wait_noop=args.auto_wait_noop,
        auto_wait_on_no_change=(False if args.no_auto_wait_on_no_change else True if args.auto_wait_on_no_change else True),
        auto_submit_review_distribution=(False if args.no_auto_submit_review_distribution else True if args.auto_submit_review_distribution else True),
        loop_guard=(False if args.no_loop_guard else True if args.loop_guard else True),
        loop_guard_force_action=(False if args.loop_guard_no_force else True if args.loop_guard_force_action else True),
        loop_guard_repeat_threshold=args.loop_guard_repeat_threshold,
        loop_guard_noop_threshold=args.loop_guard_noop_threshold,
        loop_guard_ttl=args.loop_guard_ttl,
        trace_dir=args.trace_dir,
        trace_tag=trace_tag,
        trace_include_prompt=args.trace_include_prompt,
        trace_prompt_chars=args.trace_prompt_chars,
        trace_include_obs=args.trace_include_obs,
        trace_obs_chars=args.trace_obs_chars,
    )

    rows = run_browsergym_tasks(
        tasks=tasks,
        llm_client=llm,
        mem=mem,
        cfg=run_cfg,
        env_kwargs={"headless": bool(args.headless)},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote:", out_path.resolve())


if __name__ == "__main__":
    main()
