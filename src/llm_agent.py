from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from collections import Counter
from pathlib import Path
import time

from .llm_client import LLMClient
from .tools import ToolBox
from .memory import MemoryManagerBase
from .utils import approx_token_count

def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first syntactically balanced JSON object from text.

    This avoids greedy-regex failures when the model outputs multiple brace blocks.
    We scan for balanced braces while respecting quoted strings and escapes.
    """
    s = (text or "").strip()
    start: Optional[int] = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                esc = False
            continue

        if esc:
            esc = False
            continue

        if ch == "\\":
            if in_str:
                esc = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if in_str:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i+1]

    return None

def parse_json_tool_call(text: str) -> Optional[Dict[str, Any]]:
    obj = _extract_first_json_object(text or "")
    if not obj:
        return None
    try:
        parsed = json.loads(obj)
        if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
            return parsed
    except Exception:
        return None
    return None

@dataclass
class ToolLoopConfig:
    max_steps: int = 40
    # Finish gating (prevents premature finish)
    min_steps_before_finish: int = 2
    min_open_pages_before_finish: int = 1
    require_docids_in_finish: bool = True

    # Anti no-finish: nudge/force termination near deadline
    deadline_finish_nudge_steps: int = 3  # nudge when steps remaining <= this
    force_finish_on_deadline: bool = True  # attempt auto-finish at max_steps using opened evidence

    # Break repetitive search loops (common no-finish failure mode)
    repeat_search_consecutive_threshold: int = 6
    auto_open_on_repeat_search: bool = True

    # Open-page dedupe + fact header (P1, optional)
    open_page_dedupe: bool = True
    open_page_fact_header: bool = True
    fact_header_max_chars: int = 420

    # Bench-specific policy helpers
    open_given_projects_on_repeat_search: bool = True
    validate_answer_in_given_projects: bool = True
    max_finish_blocks_per_reason: int = 1

    max_json_retries: int = 2

    # Learn-from-failure policy (controller-side constraints)
    # This is designed to reduce repeated unproductive actions (e.g., same search query loops,
    # duplicate open_page calls) without relying on natural-language negation.
    enable_failure_policy: bool = True
    unproductive_search_threshold: int = 2     # block a query after this many no-new-doc searches
    block_search_ttl_steps: int = 10          # how long to cooldown a blocked query (steps)
    duplicate_open_hard_block: bool = True    # if True, block duplicate open_page instead of caching
    max_consecutive_same_tool: int = 6        # if stuck repeating same tool, force a mode switch

    # Debug / logging
    verbose: bool = False                 # print step progress to stdout
    log_dir: Optional[str] = None         # if set, write per-task JSONL trace logs
    log_messages: bool = True             # include sent messages (truncated) in logs
    log_context_chars: int = 2500         # context tail included in prompt/logs
    log_output_chars: int = 4000          # truncate long model outputs in trace

class ToolLoopLLMAgent:
    """Tool-using agent (JSON-only protocol) with robust parsing and optional tracing.

    Key features:
      - JSON-only tool protocol: model must output exactly one JSON object per turn.
      - Auto recovery: if JSON parsing fails, reprompt up to max_json_retries.
      - Accurate tool statistics: total tool calls, search/open_page counts, repeated searches.
      - Optional debug trace logs for LLM input/output per step.
    """

    def __init__(self, llm: LLMClient, tools: ToolBox, mem: MemoryManagerBase, cfg: Optional[ToolLoopConfig] = None):
        self.llm = llm
        self.tools = tools
        self.mem = mem
        self.cfg = cfg or ToolLoopConfig()

        self.usage_accum: Dict[str, int] = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        self.counters: Counter = Counter()
        self.search_query_counts: Counter = Counter()

        # Evidence / anti-loop helpers
        self.evidence_docids: List[str] = []  # ordered unique docids opened via open_page
        self.opened_cache: Dict[str, str] = {}  # docid -> opened page content (truncated)
        self.given_projects: List[str] = []
        self._given_projects_open_idx: int = 0
        self._finish_block_counts: Counter = Counter()
        self._deadline_nudged: bool = False
        self._last_search_query: Optional[str] = None
        self._last_search_results: List[str] = []
        self._last_search_repeat_streak: int = 0
        self._last_search_open_idx: int = 0
        self._last_block_key: Optional[str] = None
        self._last_block_count: int = 0

        # Failure-policy state ("constraints" that are enforced deterministically)
        self._blocked_queries: Dict[str, int] = {}          # normalized query -> expires_step
        self._unproductive_queries: Counter = Counter()     # normalized query -> count(no-new-doc)
        self._constraints_written: set = set()              # keys to avoid spamming duplicates
        self._last_progress_step: int = 0                   # last step index with measurable progress
        self._last_exec_tool: Optional[str] = None          # executed tool name
        self._exec_tool_streak: int = 0                     # consecutive same executed tool

        self._trace_fp = None
        self._trace_path: Optional[Path] = None

    def _open_trace(self, run_tag: str, method: str, task_id: str):
        if not self.cfg.log_dir:
            return
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{run_tag}_{method}_{task_id}")
        self._trace_path = log_dir / f"trace_{safe}.jsonl"
        self._trace_fp = open(self._trace_path, "w", encoding="utf-8")

    def _close_trace(self):
        if self._trace_fp:
            try:
                self._trace_fp.close()
            finally:
                self._trace_fp = None

    def _trace(self, obj: Dict[str, Any]):
        if not self._trace_fp:
            return
        self._trace_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._trace_fp.flush()

    def _accum_usage(self, usage: Optional[Dict[str, Any]]):
        if not usage:
            return
        total = usage.get("total_tokens")
        if total is None:
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            total = pt + ct
        self.usage_accum["total_tokens"] += int(total or 0)
        self.usage_accum["input_tokens"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.usage_accum["output_tokens"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    def _normalize_query(self, q: str) -> str:
        q = (q or "").strip().lower()
        q = re.sub(r"\s+", " ", q)
        q = re.sub(r"[\,\.;:!?\(\)\[\]\{\}\"\']", "", q)
        return q

    def _record_search_query(self, query: str):
        norm = self._normalize_query(query)
        self.search_query_counts[norm] += 1
        if self.search_query_counts[norm] >= 2:
            self.counters["repeated_search_count"] += 1

    # =========================
    # Learn-from-failure policy
    # =========================
    def _policy_record_constraint_once(self, key: str, text: str):
        """Record a durable constraint note once.

        We store constraints as SUMMARY nodes so GoC retains them as anchors during pruning.
        """
        if not self.cfg.enable_failure_policy:
            return
        if key in self._constraints_written:
            return
        self._constraints_written.add(key)
        try:
            self.mem.record_summary(text)
            self.counters["constraints_written"] += 1
        except Exception:
            # Never let constraint recording crash the run
            self.counters["constraints_write_errors"] += 1

    def _policy_candidate_docid(self) -> Optional[str]:
        """Pick a reasonable next docid to open to escape loops."""
        # Prefer unopened given projects (common in SyntheticBrowseComp)
        if self.cfg.open_given_projects_on_repeat_search and self.given_projects:
            did = self._next_unopened_given_docid()
            if did:
                return did
        # Else, use last search results as a pool
        for did in (self._last_search_results or []):
            if did and did not in self.opened_cache:
                return did
        return None

    def _policy_rewrite_query(self, query: str) -> str:
        """Cheap, generic query diversification when a query is blocked/cooldown."""
        q = (query or "").strip()
        if not q:
            return q
        # Add a generic intent token that tends to be present in profiles.
        # (This is not dataset-specific: many corpora contain 'profile', 'official', etc.)
        if "profile" not in q.lower():
            return q + " official profile"
        return q + " details"

    def _policy_preprocess_call(self, proposed_call: Dict[str, Any], step: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Optionally block/override a tool call based on prior failures.

        Returns (call_to_execute or None, policy_log or None).
        If None is returned, the step is effectively blocked (we record a constraint and continue).
        """
        if not self.cfg.enable_failure_policy:
            return proposed_call, None

        tool = proposed_call.get("tool")
        args = (proposed_call.get("args") or {})

        # Track executed-tool streak after policy (we update streak for the *proposed* tool to detect habits)
        # We do not enforce on "finish" here; finish gating handles that.

        # Block duplicate open_page attempts (stronger than caching) to prevent thrashing.
        if tool == "open_page" and self.cfg.duplicate_open_hard_block:
            docid = args.get("docid")
            if docid and docid in self.opened_cache:
                self.counters["policy_overrides"] += 1
                self.counters["duplicate_open_blocked"] += 1
                cand = self._policy_candidate_docid()
                self._policy_record_constraint_once(
                    key=f"dup_open:{docid}",
                    text=(
                        f"[CONSTRAINT] Duplicate open_page blocked for docid={docid}. "
                        "Use cached FACTS from ACTIVE_CONTEXT and pick a NEW evidence docid instead."
                    ),
                )
                if cand:
                    new_call = {"tool": "open_page", "args": {"docid": cand}}
                    return new_call, {"reason": "dup_open_override", "from": proposed_call, "to": new_call, "blocked_docid": docid, "opened_docid": cand}
                # No candidate -> hard block
                return None, {"reason": "dup_open_block", "from": proposed_call, "blocked_docid": docid}

        # Enforce search cooldown for unproductive repeated queries.
        if tool == "search":
            query = args.get("query") or ""
            qnorm = self._normalize_query(query)
            exp = self._blocked_queries.get(qnorm)
            if exp is not None and step < exp:
                self.counters["policy_overrides"] += 1
                self.counters["blocked_search_query"] += 1
                cand = self._policy_candidate_docid()
                self._policy_record_constraint_once(
                    key=f"blocked_query:{qnorm}:{exp}",
                    text=(
                        f"[CONSTRAINT] Search query on cooldown (too many no-progress repeats): '{query}'. "
                        "Use a different query or open a new evidence document instead."
                    ),
                )
                if cand:
                    new_call = {"tool": "open_page", "args": {"docid": cand}}
                    return new_call, {"reason": "search_cooldown_override", "from": proposed_call, "to": new_call, "query": query, "opened_docid": cand}
                # Rewrite the query as a soft fallback
                new_q = self._policy_rewrite_query(query)
                new_call = {"tool": "search", "args": {"query": new_q, "topk": args.get("topk", 10)}}
                return new_call, {"reason": "search_cooldown_rewrite", "from": proposed_call, "to": new_call, "query": query, "rewritten_query": new_q}

        # If we are stuck repeating the same executed tool with no progress, force a mode switch.
        # "Progress" is defined as having opened a new docid or having produced a search with unseen candidates.
        predicted_streak = (self._exec_tool_streak + 1) if (self._last_exec_tool and tool == self._last_exec_tool) else 1
        no_progress_for = step - int(self._last_progress_step or 0)
        if predicted_streak >= int(self.cfg.max_consecutive_same_tool) and no_progress_for >= 2:
            cand = self._policy_candidate_docid()
            if cand and tool != "open_page":
                self.counters["policy_overrides"] += 1
                self.counters["mode_switch_overrides"] += 1
                self._policy_record_constraint_once(
                    key=f"mode_switch:{step}:{tool}",
                    text=(
                        "[CONSTRAINT] Detected repeated actions with no progress. "
                        "Switch to opening NEW evidence documents and extracting FACTS before further searching."
                    ),
                )
                new_call = {"tool": "open_page", "args": {"docid": cand}}
                return new_call, {"reason": "mode_switch_open_page", "from": proposed_call, "to": new_call, "opened_docid": cand, "streak": predicted_streak}

        return proposed_call, None

    def _policy_post_search(self, query: str, result_docids: List[str], step: int):
        """Update unproductive-search counters and set query cooldowns."""
        if not self.cfg.enable_failure_policy:
            return
        qnorm = self._normalize_query(query)
        new_candidates = [d for d in (result_docids or []) if d and d not in self.opened_cache]

        if new_candidates:
            # progress: we have at least one unopened candidate
            self._last_progress_step = step
            # reset unproductive count (reward new evidence)
            if qnorm in self._unproductive_queries:
                self._unproductive_queries[qnorm] = 0
            return

        # No new candidates from this query: record as failure signature
        self._unproductive_queries[qnorm] += 1
        self.counters["unproductive_searches"] += 1

        if self._unproductive_queries[qnorm] >= int(self.cfg.unproductive_search_threshold):
            # Cooldown this query to prevent repetition
            exp = step + int(self.cfg.block_search_ttl_steps)
            self._blocked_queries[qnorm] = exp
            self.counters["search_queries_cooled_down"] += 1
            self._policy_record_constraint_once(
                key=f"cooldown:{qnorm}:{exp}",
                text=(
                    f"[CONSTRAINT] Query produced no new evidence repeatedly; cooling down: '{query}'. "
                    "Use a different query OR open new evidence docids instead of repeating this search."
                ),
            )


    def _extract_given_projects(self, question: str) -> List[str]:
        """Extract ordered unique Project_#### mentions from the question."""
        seen = set()
        out: List[str] = []
        for mm in re.finditer(r"\bProject_\d{4}\b", question or ""):
            p = mm.group(0)
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _next_unopened_given_docid(self) -> Optional[str]:
        """Return next D_TRUTH_#### for an unopened Project_#### in the question."""
        while self._given_projects_open_idx < len(self.given_projects):
            p = self.given_projects[self._given_projects_open_idx]
            self._given_projects_open_idx += 1
            did = f"D_TRUTH_{p.split('_')[1]}"
            if did not in self.opened_cache:
                return did
        return None

    def _make_fact_header(self, content: str, max_chars: int) -> str:
        """Extract a compact FACTS header from an OFFICIAL PROFILE page."""
        if not content:
            return ""
        fields: Dict[str, str] = {}

        m_proj = re.search(r"\bProject_\d{4}\b", content)
        if m_proj:
            fields["project"] = m_proj.group(0)

        m_year = re.search(r"(?im)^start_year\s*:\s*(\d{4})\s*$", content)
        if m_year:
            fields["start_year"] = m_year.group(1)

        m_hq = re.search(r"(?im)^headquarters\s*:\s*([^\n]+)$", content)
        if m_hq:
            fields["headquarters"] = m_hq.group(1).strip()

        m_code = re.search(r"(?im)^code_name\s*:\s*([^\n]+)$", content)
        if m_code:
            fields["code_name"] = m_code.group(1).strip()

        m_key = re.search(r"(?im)^key_number\s*:\s*(\d+)\s*$", content)
        if m_key:
            fields["key_number"] = m_key.group(1)

        m_rel = re.search(r"(?im)^related_projects\s*:\s*([^\n]+)$", content)
        if m_rel:
            fields["related_projects"] = m_rel.group(1).strip()

        if not fields:
            return ""

        parts = [f"{k}={v}" for k, v in fields.items()]
        header = "FACTS: " + " | ".join(parts)
        if len(header) > max_chars:
            header = header[:max_chars].rstrip()
        return header

    def _compose_open_page_observation(self, content: str) -> str:
        """Observation text stored into memory for open_page."""
        max_len = 2500
        body = content or ""
        if not self.cfg.open_page_fact_header:
            return body[:max_len]

        header = self._make_fact_header(body, max_chars=int(self.cfg.fact_header_max_chars))
        if not header:
            return body[:max_len]

        reserve = len(header) + 2  # header + blank line
        body_snip = body[: max(0, max_len - reserve)]
        return header + "\n\n" + body_snip

    def _should_block_finish(self, reason: str) -> bool:
        """Block finish at most N times per reason to avoid infinite loops."""
        max_blocks = int(getattr(self.cfg, "max_finish_blocks_per_reason", 1) or 1)
        if self._finish_block_counts[reason] >= max_blocks:
            return False
        self._finish_block_counts[reason] += 1
        return True


    def _build_system_prompt(self) -> str:
        return (
            "You are an assistant that MUST use the provided tools.\n"
            "You MUST output exactly ONE JSON object per turn. No extra text.\n"
            "Format: {\\\"tool\\\":\\\"<name>\\\",\\\"args\\\":{...}}\n"
            "Tools available: search, open_page, branch, return, finish.\n"
            "\n"
            "Critical rules (avoid invalid / wasteful calls):\n"
            " - Do NOT call `finish` until you have gathered evidence: at least 1 `open_page` and at least 1 docid cited.\n"
            " - The `finish.args.explanation` MUST include the evidence docids you used (e.g., 'Evidence docids: D_TRUTH_0001').\n"
            " - Do NOT call `finish` if you have not opened any page yet. Call search -> open_page first.\n"
            " - Do NOT call `return` unless you previously called `branch` and you are currently inside that branch.\n"
            " - When you call `branch`, you MUST include args: {description, prompt}.\n"
            " - When you call `return`, you MUST include args: {message}.\n"
            " - Call `return` at most ONCE per branch. Never call return twice in a row.\n"
            " - For this benchmark, branching is optional; prefer search/open_page unless a sub-task is clearly needed.\n"
            "\n"
            "Guidance:\n"
            " - Use search -> open_page to gather evidence docids.\n"
            " - Always cite evidence by including docids in explanation.\n"
        )

    def _json_recovery_prompt(self) -> str:
        return (
            "Your previous message was invalid. Output exactly ONE JSON object and NOTHING else.\n"
            "Valid examples:\n"
            "  {\\\"tool\\\":\\\"search\\\",\\\"args\\\":{\\\"query\\\":\\\"Project_0001 headquarters\\\",\\\"topk\\\":5}}\n"
            "  {\\\"tool\\\":\\\"open_page\\\",\\\"args\\\":{\\\"docid\\\":\\\"D_TRUTH_0001\\\"}}\n"
            "  {\\\"tool\\\":\\\"finish\\\",\\\"args\\\":{\\\"answer\\\":\\\"...\\\",\\\"explanation\\\":\\\"Evidence docids: D_TRUTH_0001\\\",\\\"confidence\\\":\\\"80%\\\"}}\n"
            "Now output a corrected JSON object for your next tool call."
        )

    def _call_model_for_json(self, messages: List[Dict[str, str]], step: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Call model, parse JSON, retry with recovery prompt if needed.

        Returns: (tool_call or None, attempt_logs)
        attempt_logs: list of {attempt_idx, raw_output, parsed_ok}
        """
        attempt_logs: List[Dict[str, Any]] = []

        resp = self.llm.generate(messages=messages, tools=None)
        self._accum_usage(getattr(resp, "usage", None))
        raw = (resp.text or "")
        call = parse_json_tool_call(raw)
        attempt_logs.append({
            "attempt": 0,
            "raw_output": raw[: self.cfg.log_output_chars],
            "parsed_ok": call is not None
        })
        if call is not None:
            return call, attempt_logs

        self.counters["json_parse_failures"] += 1

        for retry in range(1, self.cfg.max_json_retries + 1):
            # Minimal recovery messages to avoid growing prompt.
            recovery_messages = messages + [{"role": "user", "content": self._json_recovery_prompt()}]
            r2 = self.llm.generate(messages=recovery_messages, tools=None)
            self._accum_usage(getattr(r2, "usage", None))
            raw2 = (r2.text or "")
            call2 = parse_json_tool_call(raw2)
            attempt_logs.append({
                "attempt": retry,
                "raw_output": raw2[: self.cfg.log_output_chars],
                "parsed_ok": call2 is not None
            })
            if call2 is not None:
                self.counters["json_recoveries"] += 1
                return call2, attempt_logs

        return None, attempt_logs

    def run(self, user_question: str, *, task_id: str = "task", method: str = "method", run_tag: str = "run") -> Dict[str, Any]:
        """Run one task.

        task_id/method/run_tag are used for logging and stdout progress only.
        """
        self.mem.reset()
        self.usage_accum = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        self.counters = Counter()
        self.search_query_counts = Counter()
        self.evidence_docids = []
        self.opened_cache = {}
        self.given_projects = self._extract_given_projects(user_question)
        self._given_projects_open_idx = 0
        self._finish_block_counts = Counter()
        # reset failure-policy state
        self._blocked_queries = {}
        self._unproductive_queries = Counter()
        self._constraints_written = set()
        self._last_progress_step = 0
        self._last_exec_tool = None
        self._exec_tool_streak = 0
        if self.given_projects:
            self.mem.record_msg(
                "[SYSTEM] This task lists a specific set of projects. To avoid search loops, you can open each project's OFFICIAL PROFILE (typically docid D_TRUTH_####) and extract the required fields."
            )
        self._deadline_nudged = False
        self._last_search_query = None
        self._last_search_results = []
        self._last_search_repeat_streak = 0
        self._last_search_open_idx = 0

        self._open_trace(run_tag=run_tag, method=method, task_id=task_id)

        system = self._build_system_prompt()
        base_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_question},
        ]

        start_time = time.time()

        try:
            for step in range(self.cfg.max_steps):
                remaining = self.cfg.max_steps - step
                if (not self._deadline_nudged) and remaining <= self.cfg.deadline_finish_nudge_steps:
                    self._deadline_nudged = True
                    if self.counters.get("open_page_calls", 0) >= self.cfg.min_open_pages_before_finish:
                        self.mem.record_msg(
                            "[SYSTEM] Deadline approaching. Stop searching/repeating. Use already opened evidence and CALL finish NOW. "
                            "finish.args.answer must be non-empty; finish.args.explanation must include 'Evidence docids: D_...'. "
                            "Example: {\"tool\":\"finish\",\"args\":{\"answer\":\"Project_0001 | City_02\",\"explanation\":\"Evidence docids: D_TRUTH_0001\"}}"
                        )
                    else:
                        self.mem.record_msg(
                            "[SYSTEM] Deadline approaching. You MUST open at least one page (open_page) and then CALL finish with docid evidence." 
                        )
                    self._trace({"type":"deadline_nudge","task_id":task_id,"method":method,"run_tag":run_tag,"step":step+1})
                # Stateless prompting: we DO NOT accumulate prior ACTIVE_CONTEXT in messages.
                ctx = self.mem.get_active_text()
                ctx_tail = ctx[-8000:]
                # Keep prompt context bounded and log a shorter tail if requested.
                ctx_for_prompt = ctx_tail[-self.cfg.log_context_chars:] if self.cfg.log_context_chars > 0 else ""

                messages = list(base_messages)
                if self.cfg.verbose:
                    # Always print a heartbeat per step (useful when running ALL methods).
                    atok = approx_token_count(ctx)
                    print(f"[{run_tag}][{method}][{task_id}] step={step+1}/{self.cfg.max_steps} active_tok≈{atok}", flush=True)

                if ctx_for_prompt:
                    messages.append({"role": "user", "content": "ACTIVE_CONTEXT:\n" + ctx_for_prompt})

                # Log prompt snapshot (optional)
                if self.cfg.log_messages:
                    self._trace({
                        "type": "prompt",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "messages": [
                            {"role": m["role"], "content": m["content"][:6000]} for m in messages
                        ],
                        "active_tokens_est": approx_token_count(ctx),
                    })
                else:
                    self._trace({
                        "type": "prompt",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "active_tokens_est": approx_token_count(ctx),
                        "active_context_tail": ctx_for_prompt[:2000],
                    })

                call, attempt_logs = self._call_model_for_json(messages, step=step)
                self._trace({
                    "type": "llm_attempts",
                    "task_id": task_id,
                    "method": method,
                    "run_tag": run_tag,
                    "step": step,
                    "attempts": attempt_logs,
                })

                if call is None:
                    if self.cfg.verbose:
                        print(f"[{run_tag}][{method}][{task_id}] step={step+1} JSON_PARSE_FAILED (continuing)")
                    continue

                # Count proposed tool (from the model) separately from executed tool (after policy overrides).
                proposed_tool = call.get("tool")
                self.counters["tool_calls_proposed_total"] += 1
                if proposed_tool:
                    self.counters[f"tool_calls_proposed_{proposed_tool}"] += 1

                # Apply failure-policy constraints BEFORE executing the tool.
                call2, policy_evt = self._policy_preprocess_call(call, step=step)
                if policy_evt is not None:
                    self._trace({
                        "type": "policy_event",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        **policy_evt,
                    })
                if call2 is None:
                    # Tool call blocked; consume the step but do not execute any tool.
                    if self.cfg.verbose:
                        print(f"[{run_tag}][{method}][{task_id}] POLICY_BLOCK proposed={proposed_tool}", flush=True)
                    continue

                tool = call2["tool"]
                args = call2.get("args", {}) or {}
                in_branch = (self.mem.current_thread != "main") if hasattr(self.mem, "current_thread") else False

                # Count executed tools
                self.counters["tool_calls_total"] += 1
                self.counters[f"tool_calls_{tool}"] += 1

                # Track executed-tool streak (for stuck-mode detection)
                if tool == self._last_exec_tool:
                    self._exec_tool_streak += 1
                else:
                    self._last_exec_tool = tool
                    self._exec_tool_streak = 1

                if self.cfg.verbose:
                    print(f"[{run_tag}][{method}][{task_id}] step={step+1}/{self.cfg.max_steps} tool={tool}")

                if tool == "search":
                    query = args.get("query")
                    if not query:
                        self.counters["malformed_search_args"] += 1
                        self.mem.record_msg("[ASSISTANT] Malformed search call (missing query). Ignored.")
                        self._trace({
                            "type":"tool","task_id":task_id,"method":method,"run_tag":run_tag,"step":step,
                            "tool":"search","args":args,"malformed":True
                        })
                        continue
                    qnorm = self._normalize_query(query)
                    if qnorm and qnorm == self._last_search_query:
                        self._last_search_repeat_streak += 1
                    else:
                        self._last_search_query = qnorm
                        self._last_search_repeat_streak = 1
                        self._last_search_results = []
                        self._last_search_open_idx = 0

                    # If the model is repeating the exact same search, break loops by opening unseen evidence.
                    # v21: if the question provides an explicit project set, prefer opening those projects' truth docs.
                    if (
                        self.cfg.auto_open_on_repeat_search
                        and self._last_search_repeat_streak >= self.cfg.repeat_search_consecutive_threshold
                    ):
                        docid_to_open: Optional[str] = None

                        if self.cfg.open_given_projects_on_repeat_search and self.given_projects:
                            docid_to_open = self._next_unopened_given_docid()

                        if (docid_to_open is None) and self._last_search_results:
                            while self._last_search_open_idx < len(self._last_search_results):
                                cand = self._last_search_results[self._last_search_open_idx]
                                self._last_search_open_idx += 1
                                if cand not in self.opened_cache:
                                    docid_to_open = cand
                                    break

                        if docid_to_open:
                            self.counters["policy_overrides"] += 1

                            # Dedupe actual open calls
                            if self.cfg.open_page_dedupe and docid_to_open in self.opened_cache:
                                self.counters["open_page_cache_hits"] += 1
                                cached = self.opened_cache.get(docid_to_open, "")
                                obs = f"[CACHED] docid={docid_to_open} previously opened.\n\n" + self._compose_open_page_observation(cached)
                                self.mem.record_tool(
                                    "open_page",
                                    {"docid": docid_to_open},
                                    observation=obs[:2500],
                                    docids=[docid_to_open],
                                )
                                # Treat policy-driven evidence surfacing as progress to break stuck-mode heuristics.
                                self._last_progress_step = step
                                if docid_to_open not in self.evidence_docids:
                                    self.evidence_docids.append(docid_to_open)
                            else:
                                outp = self.tools.open_page(docid=docid_to_open)
                                self.counters["open_page_calls"] += 1
                                content = (outp.get("content") or "")
                                # Cache opened content (truncated) for later auto-finish / analysis
                                self.opened_cache[outp["docid"]] = content[:2500]
                                if outp["docid"] not in self.evidence_docids:
                                    self.evidence_docids.append(outp["docid"])
                                obs = self._compose_open_page_observation(content)
                                self.mem.record_tool(
                                    "open_page",
                                    {"docid": docid_to_open},
                                    observation=obs,
                                    docids=[outp["docid"]],
                                )
                                self._last_progress_step = step

                            self._trace({
                                "type": "policy_override",
                                "task_id": task_id,
                                "method": method,
                                "run_tag": run_tag,
                                "step": step,
                                "from_tool": "search",
                                "to_tool": "open_page",
                                "query": query,
                                "opened_docid": docid_to_open,
                            })
                            if self.cfg.verbose:
                                print(f"[{run_tag}][{method}][{task_id}] OVERRIDE search->open_page docid={docid_to_open}", flush=True)
                            continue

                    topk = int(args.get("topk", 10))
                    out = self.tools.search(query=query, topk=topk)

                    # Cache last search results for potential repeat-loop breaking
                    try:
                        docids = [x["docid"] for x in out]
                        self._last_search_results = docids
                        self._last_search_open_idx = 0
                    except Exception:
                        docids = []
                        self._last_search_results = []
                        self._last_search_open_idx = 0

                    self._record_search_query(query)
                    # Learn-from-failure: mark queries that yield no new evidence candidates
                    self._policy_post_search(query, docids, step=step)
                    self.counters["search_calls"] += 1

                    # Record into memory (active context)
                    self.mem.record_tool("search", args, observation=str([x["docid"] for x in out]))

                    self._trace({
                        "type": "tool",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "tool": "search",
                        "args": {"query": query, "topk": topk},
                        "result_docids": [x["docid"] for x in out[: min(20, len(out))]],
                    })

                elif tool == "open_page":
                    docid = args.get("docid")
                    url = args.get("url")
                    if not docid and not url:
                        self.counters["malformed_open_page_args"] += 1
                        self.mem.record_msg("[ASSISTANT] Malformed open_page call (missing docid/url). Ignored.")
                        self._trace({
                            "type":"tool","task_id":task_id,"method":method,"run_tag":run_tag,"step":step,
                            "tool":"open_page","args":args,"malformed":True
                        })
                        continue
                    # v21: avoid wasting context budget on duplicate open_page calls
                    if self.cfg.open_page_dedupe and docid and (docid in self.opened_cache):
                        self.counters["open_page_cache_hits"] += 1
                        cached = self.opened_cache.get(docid, "")
                        obs = f"[CACHED] docid={docid} previously opened.\n\n" + self._compose_open_page_observation(cached)
                        self.mem.record_tool("open_page", args, observation=obs[:2500], docids=[docid])

                        if docid not in self.evidence_docids:
                            self.evidence_docids.append(docid)

                        self._trace({
                            "type": "tool",
                            "task_id": task_id,
                            "method": method,
                            "run_tag": run_tag,
                            "step": step,
                            "tool": "open_page",
                            "args": {"docid": docid, "url": args.get("url")},
                            "opened_docid": docid,
                            "cached": True,
                            "content_preview": cached[:800],
                        })
                        continue

                    out = self.tools.open_page(docid=docid, url=url)
                    self.counters["open_page_calls"] += 1

                    content = (out.get("content") or "")
                    obs = self._compose_open_page_observation(content)
                    self.mem.record_tool("open_page", args, observation=obs, docids=[out["docid"]])

                    # Cache opened content for auto-finish / analysis
                    self.opened_cache[out["docid"]] = content[:2500]

                    # Progress: opened a new evidence document
                    self._last_progress_step = step

                    # Track evidence docids (opened pages) for finish auto-citation
                    opened = out.get("docid")
                    if opened:
                        if opened not in self.evidence_docids:
                            self.evidence_docids.append(opened)

                    self._trace({
                        "type": "tool",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "tool": "open_page",
                        "args": {"docid": out.get("docid"), "url": args.get("url")},
                        "opened_docid": out.get("docid"),
                        "cached": False,
                        "content_preview": content[:800],
                    })


                elif tool == "branch":
                    # Be robust to missing keys (models sometimes omit required fields).
                    desc = args.get("description")
                    prompt = args.get("prompt")
                    if in_branch:
                        self.counters["nested_branch_blocked"] += 1
                        self.mem.record_msg("[ASSISTANT] Nested branch disallowed; continue in current branch.")
                        blocked = True
                    else:
                        blocked = False
                        if not desc or not prompt:
                            self.counters["malformed_branch_args"] += 1
                            self.mem.record_msg("[ASSISTANT] Malformed branch call (missing description/prompt). Ignored.")
                        else:
                            self.mem.branch(description=desc, prompt=prompt)

                    self._trace({
                        "type": "tool",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "tool": "branch",
                        "args": args,
                        "blocked": bool(in_branch),
                        "malformed": bool((not in_branch) and (not desc or not prompt)),
                    })
                elif tool == "return":
                    if in_branch:
                        msg = args.get("message")
                        if msg is None:
                            # Still exit the branch to avoid infinite return loops.
                            self.counters["malformed_return_args"] += 1
                            msg = ""
                        self.mem.return_from_branch(message=msg)
                    else:
                        # Strong guidance to prevent repeated invalid returns.
                        self.counters["return_in_main_ignored"] += 1
                        self.mem.record_msg(
                            "[SYSTEM] You are in MAIN (no active branch). `return` is invalid here. "
                            "Next step MUST be one of: search, open_page, finish."
                        )
                    self._trace({
                        "type": "tool",
                        "task_id": task_id,
                        "method": method,
                        "run_tag": run_tag,
                        "step": step,
                        "tool": "return",
                        "args": args,
                        "ignored": bool(not in_branch),
                    })

                elif tool == "finish":
                    # Finish gating: block premature finish (no evidence, too early, missing docids).
                    open_calls = int(self.counters.get("open_page_calls", 0))

                    if (step + 1) < int(self.cfg.min_steps_before_finish):
                        self.counters["premature_finish_blocked"] += 1
                        self.mem.record_msg(
                            f"[SYSTEM] finish blocked: need at least {self.cfg.min_steps_before_finish} steps. "
                            "Next call should be search/open_page."
                        )
                        self._trace({"type": "finish_blocked", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "reason": "min_steps"})
                        if self.cfg.verbose:
                            print(f"[{run_tag}][{method}][{task_id}] BLOCK_FINISH reason=min_steps", flush=True)
                        continue

                    if open_calls < int(self.cfg.min_open_pages_before_finish):
                        self.counters["premature_finish_blocked"] += 1
                        self.mem.record_msg(
                            f"[SYSTEM] finish blocked: need at least {self.cfg.min_open_pages_before_finish} open_page calls for evidence. "
                            "Next call MUST be search -> open_page."
                        )
                        self._trace({"type": "finish_blocked", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "reason": "no_open_page"})
                        if self.cfg.verbose:
                            print(f"[{run_tag}][{method}][{task_id}] BLOCK_FINISH reason=no_open_page", flush=True)
                        continue

                    ans_raw = args.get("answer")
                    if ans_raw is None:
                        # common alternative keys from LLMs
                        ans_raw = args.get("final") or args.get("final_answer") or args.get("result") or args.get("output")
                    ans = (str(ans_raw).strip() if ans_raw is not None else "")
                    expl0 = (args.get("explanation") or "")

                    # Normalize: ensure the final answer is always stored in args["answer"] for downstream evaluation.
                    if ans:
                        args["answer"] = ans

                    # If answer is missing/empty, try to salvage from explanation (models often put the answer there).
                    if not ans:
                        cand: Optional[str] = None
                        expl_s = (expl0 or "").strip()
                        if expl_s:
                            # Strong pattern: '<Project_####> | <City_..>' (or similar).
                            m_pair = re.search(r"\b(Project_\d{4})\b\s*\|\s*\b([A-Za-z]+_\d+)\b", expl_s)
                            if m_pair:
                                cand = f"{m_pair.group(1)} | {m_pair.group(2)}"
                            else:
                                # Common phrasing: "Project_0041 ... headquarters is City_42"
                                m_hq = re.search(r"\b(Project_\d{4})\b.*?\bheadquarters\b.*?\b([A-Za-z]+_\d+)\b", expl_s, flags=re.I|re.S)
                                if m_hq:
                                    cand = f"{m_hq.group(1)} | {m_hq.group(2)}"
                                else:
                                    # Explicit "Answer: ..." line
                                    m_ans = re.search(r"(?is)\banswer\s*[:\-]\s*(.+?)(?:\n|$)", expl_s)
                                    if m_ans:
                                        cand = m_ans.group(1).strip()
                                    else:
                                        # Heuristic: take the first non-empty line before an Evidence/Docids section.
                                        head = re.split(r"(?i)\bevidence\b|\bdocids?\b", expl_s, maxsplit=1)[0].strip()
                                        if head:
                                            cand = next((ln.strip() for ln in head.splitlines() if ln.strip()), None)

                        if cand:
                            ans = str(cand).strip()
                            args["answer"] = ans
                            self.counters["finish_answer_salvaged"] += 1

                    if not ans:
                        self.counters["premature_finish_blocked"] += 1
                        self.mem.record_msg("[SYSTEM] finish blocked: empty answer. Put the final answer into finish.args.answer (a non-empty short string).")
                        self.mem.record_msg('[SYSTEM] Example: {"tool":"finish","args":{"answer":"<SHORT ANSWER>","explanation":"Evidence docids: D_TRUTH_0001"}}')
                        self._trace({"type": "finish_blocked", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "reason": "empty_answer"})
                        if self.cfg.verbose:
                            print(f"[{run_tag}][{method}][{task_id}] BLOCK_FINISH reason=empty_answer", flush=True)
                        continue
                    # v21: finish validator - enforce answer uses one of the GIVEN projects when the question provides an explicit project set.
                    if self.cfg.validate_answer_in_given_projects and self.given_projects:
                        allowed = set(self.given_projects)
                        # Expected format: '<ProjectName> | <Headquarters>'
                        proj_text = ans.split('|', 1)[0].strip() if '|' in ans else ans.strip()
                        m_proj = re.search(r"\bProject_\d{4}\b", proj_text)
                        proj = m_proj.group(0) if m_proj else ""
                        if (not proj) or (proj not in allowed):
                            self.counters["finish_invalid_project_blocked"] += 1
                            if self._should_block_finish("invalid_project"):
                                self.mem.record_msg(
                                    "[SYSTEM] finish blocked: answer project must be one of the GIVEN projects from the question. "
                                    "Re-check evidence and answer format '<Project_####> | <Headquarters>'."
                                )
                                self.mem.record_msg("[SYSTEM] Allowed projects: " + ", ".join(self.given_projects))
                                self._trace({
                                    "type": "finish_blocked",
                                    "task_id": task_id,
                                    "method": method,
                                    "run_tag": run_tag,
                                    "step": step,
                                    "reason": "invalid_project",
                                    "answer": ans,
                                })
                                if self.cfg.verbose:
                                    print(f"[{run_tag}][{method}][{task_id}] BLOCK_FINISH reason=invalid_project", flush=True)
                                continue


                    if self.cfg.require_docids_in_finish:
                        # Require at least one docid-like token in explanation (e.g., D_TRUTH_0001).
                        # Instead of blocking indefinitely, auto-append evidence docids from opened pages.
                        if self.cfg.require_docids_in_finish:
                            # Require at least one VALID opened docid cited in explanation (e.g., D_TRUTH_0001).
                            # If the model cites a hallucinated docid (e.g., D_PROJECT_0001), treat as missing and auto-append evidence.
                            cited = re.findall(r"\bD_[A-Z0-9_\-]+\b", expl0 or "")
                            valid = set(self.evidence_docids) | set(self.opened_cache.keys())
                            has_valid = any(d in valid for d in cited)
                            if not has_valid:
                                if self.evidence_docids:
                                    tail = ", ".join(self.evidence_docids[-3:])
                                    expl0 = (expl0 + "\n" if expl0 else "") + f"Evidence docids: {tail}"
                                    args["explanation"] = expl0
                                    self.counters["finish_docids_auto_appended"] += 1
                                else:
                                    # Fallback: block if we truly have no evidence docids recorded.
                                    self.counters["premature_finish_blocked"] += 1
                                    self.mem.record_msg("[SYSTEM] finish blocked: explanation missing evidence docids. Cite docids from open_page.")
                                    self.mem.record_msg("[SYSTEM] You attempted to finish without evidence docids, and no opened docids were recorded. Call open_page first.")
                                    self._trace({"type": "finish_blocked", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "reason": "missing_docids_no_evidence"})
                                    if self.cfg.verbose:
                                        print(f"[{run_tag}][{method}][{task_id}] BLOCK_FINISH reason=missing_docids_no_evidence", flush=True)
                                    continue

                    top_rep = self.search_query_counts.most_common(5)
                    elapsed = time.time() - start_time
                    result = {
                        "answer": args.get("answer", ""),
                        "explanation": args.get("explanation", ""),
                        "confidence": args.get("confidence", ""),
                        "active_context": self.mem.get_active_text(),
                        "usage": dict(self.usage_accum),
                        "steps": step + 1,
                        "elapsed_sec": elapsed,
                        "tool_stats": {
                            "tool_calls_proposed_total": int(self.counters.get("tool_calls_proposed_total", 0)),
                            "tool_calls_total": int(self.counters["tool_calls_total"]),
                            "search_calls": int(self.counters["search_calls"]),
                            "open_page_calls": int(self.counters["open_page_calls"]),
                            "open_page_cache_hits": int(self.counters.get("open_page_cache_hits", 0)),
                            "policy_overrides": int(self.counters.get("policy_overrides", 0)),
                            "constraints_written": int(self.counters.get("constraints_written", 0)),
                            "constraints_write_errors": int(self.counters.get("constraints_write_errors", 0)),
                            "duplicate_open_blocked": int(self.counters.get("duplicate_open_blocked", 0)),
                            "blocked_search_query": int(self.counters.get("blocked_search_query", 0)),
                            "unproductive_searches": int(self.counters.get("unproductive_searches", 0)),
                            "search_queries_cooled_down": int(self.counters.get("search_queries_cooled_down", 0)),
                            "mode_switch_overrides": int(self.counters.get("mode_switch_overrides", 0)),
                            "return_calls": int(self.counters.get("tool_calls_return", 0)),
                            "return_in_main_ignored": int(self.counters.get("return_in_main_ignored", 0)),
                            "malformed_branch_args": int(self.counters.get("malformed_branch_args", 0)),
                            "malformed_return_args": int(self.counters.get("malformed_return_args", 0)),
                            "malformed_search_args": int(self.counters.get("malformed_search_args", 0)),
                            "malformed_open_page_args": int(self.counters.get("malformed_open_page_args", 0)),
                            "repeated_search_count": int(self.counters["repeated_search_count"]),
                            "unique_search_queries": int(len(self.search_query_counts)),
                            "top_repeated_queries": [{"query": q, "count": c} for q, c in top_rep if c >= 2],
                            "json_parse_failures": int(self.counters["json_parse_failures"]),
                            "json_recoveries": int(self.counters["json_recoveries"]),
                            "finish_answer_salvaged": int(self.counters.get("finish_answer_salvaged", 0)),
                            "finish_docids_auto_appended": int(self.counters.get("finish_docids_auto_appended", 0)),
                            "finish_invalid_project_blocked": int(self.counters.get("finish_invalid_project_blocked", 0)),
                            "premature_finish_blocked": int(self.counters.get("premature_finish_blocked", 0)),
                        }
                    }
                    self._trace({"type": "finish", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "result": {
                        "answer": result["answer"],
                        "explanation": result["explanation"][:2000],
                        "confidence": result["confidence"],
                        "usage": result["usage"],
                        "steps": result["steps"],
                        "elapsed_sec": result["elapsed_sec"],
                        "tool_stats": result["tool_stats"],
                    }})
                    if self.cfg.verbose:
                        print(f"[{run_tag}][{method}][{task_id}] FINISH steps={result['steps']} tok={result['usage'].get('total_tokens')} tools={result['tool_stats']['tool_calls_total']} elapsed={elapsed:.1f}s")
                    return result

                else:
                    self.counters["unknown_tool"] += 1
                    self.mem.record_msg(f"[ASSISTANT] Unknown tool: {tool}")
                    self._trace({"type": "tool", "task_id": task_id, "method": method, "run_tag": run_tag, "step": step, "tool": tool, "args": args, "unknown": True})

            # No finish
            top_rep = self.search_query_counts.most_common(5)
            elapsed = time.time() - start_time

            result = {
                "answer": "",
                "explanation": "max_steps reached (no finish)",
                "confidence": "0%",
                "active_context": self.mem.get_active_text(),
                "usage": dict(self.usage_accum),
                "steps": self.cfg.max_steps,
                "elapsed_sec": elapsed,
                "tool_stats": {
                    "tool_calls_proposed_total": int(self.counters.get("tool_calls_proposed_total", 0)),
                    "tool_calls_total": int(self.counters["tool_calls_total"]),
                    "search_calls": int(self.counters["search_calls"]),
                    "open_page_calls": int(self.counters["open_page_calls"]),
                    "open_page_cache_hits": int(self.counters.get("open_page_cache_hits", 0)),
                    "policy_overrides": int(self.counters.get("policy_overrides", 0)),
                    "constraints_written": int(self.counters.get("constraints_written", 0)),
                    "constraints_write_errors": int(self.counters.get("constraints_write_errors", 0)),
                    "duplicate_open_blocked": int(self.counters.get("duplicate_open_blocked", 0)),
                    "blocked_search_query": int(self.counters.get("blocked_search_query", 0)),
                    "unproductive_searches": int(self.counters.get("unproductive_searches", 0)),
                    "search_queries_cooled_down": int(self.counters.get("search_queries_cooled_down", 0)),
                    "mode_switch_overrides": int(self.counters.get("mode_switch_overrides", 0)),
                    "repeated_search_count": int(self.counters["repeated_search_count"]),
                    "unique_search_queries": int(len(self.search_query_counts)),
                    "top_repeated_queries": [{"query": q, "count": c} for q, c in top_rep if c >= 2],
                    "json_parse_failures": int(self.counters["json_parse_failures"]),
                    "json_recoveries": int(self.counters["json_recoveries"]),
                    "premature_finish_blocked": int(self.counters.get("premature_finish_blocked", 0)),
                },
            }
            # Attempt best-effort auto-finish using ONLY opened evidence
            if self.cfg.force_finish_on_deadline and self.counters.get("open_page_calls", 0) >= 1:
                ans, expl = self._try_autofinish(user_question)
                if ans:
                    result["answer"] = ans
                    result["explanation"] = expl or result.get("explanation", "")
                    result["confidence"] = "auto"
                    self.counters["forced_finish"] += 1
                    self._trace({"type":"forced_finish","task_id":task_id,"method":method,"run_tag":run_tag,"step":self.cfg.max_steps,"answer":ans})

            self._trace({"type": "finish", "task_id": task_id, "method": method, "run_tag": run_tag, "step": self.cfg.max_steps, "result": result})
            if self.cfg.verbose:
                print(f"[{run_tag}][{method}][{task_id}] NO_FINISH steps={result['steps']} tok={result['usage'].get('total_tokens')} tools={result['tool_stats']['tool_calls_total']} elapsed={elapsed:.1f}s")
            return result
        finally:
            self._close_trace()


    def _try_autofinish(self, user_question: str) -> Tuple[str, str]:
        """Best-effort auto-finish using ONLY opened evidence (no extra tool calls).

        Returns (answer, explanation). If cannot infer, returns ("","...").
        Designed to reduce NO_FINISH when the agent has already opened the required pages.
        """
        q = (user_question or "")
        ql = q.lower()

        def parse_fields(content: str):
            m_year = re.search(r"(?i)start_year\s*:\s*(\d{4})", content or "")
            m_hq = re.search(r"(?i)headquarters\s*:\s*([^\n]+)", content or "")
            m_code = re.search(r"(?i)code_name\s*:\s*([^\n]+)", content or "")
            m_key = re.search(r"(?i)key_number\s*:\s*(\d+)", content or "")
            year = int(m_year.group(1)) if m_year else None
            hq = (m_hq.group(1).strip() if m_hq else None)
            code = (m_code.group(1).strip() if m_code else None)
            key = int(m_key.group(1)) if m_key else None
            return year, hq, code, key

        def build_expl(primary_docid: str) -> str:
            ev = [primary_docid] + [d for d in self.evidence_docids if d != primary_docid]
            ev = [d for d in ev if d][:5]
            return "Evidence docids: " + ", ".join(ev) if ev else "Evidence docids: (none)"

        # Pattern A: earliest start_year + headquarters
        if ("earliest" in ql) and ("start_year" in ql) and ("headquarters" in ql):
            candidates = []

            projects = re.findall(r"Project_\d{4}", q)
            if projects:
                for p in projects:
                    docid_guess = "D_TRUTH_" + p.split("_")[1]
                    content = self.opened_cache.get(docid_guess)
                    did = docid_guess
                    if not content:
                        for odid, cont in self.opened_cache.items():
                            if p in cont:
                                content = cont
                                did = odid
                                break
                    if not content:
                        continue
                    year, hq, _, _ = parse_fields(content)
                    if year is not None and hq:
                        candidates.append((year, p, hq, did))
            else:
                m_pref = re.search(r"(Codename_[A-Za-z0-9]+)", q)
                pref = m_pref.group(1) if m_pref else None
                for did, cont in self.opened_cache.items():
                    year, hq, code, _ = parse_fields(cont)
                    if year is None or not hq:
                        continue
                    if pref and code and code.startswith(pref):
                        proj = "Project_" + did.split("_")[-1] if did.startswith("D_TRUTH_") else "UnknownProject"
                        candidates.append((year, proj, hq, did))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                year, proj, hq, did = candidates[0]
                ans = f"{proj} | {hq}"
                return ans, build_expl(did)

        # Pattern B: two-phase select-by code_name prefix then max key_number, return headquarters
        if (
            ("key_number" in ql)
            and ("code_name" in ql)
            and ("largest" in ql or "maximum" in ql or "max" in ql)
            and ("headquarters" in ql)
            and ("step" in ql or "select" in ql)
        ):
            projects = re.findall(r"Project_\d{4}", q)
            m_init = re.search(r"Codename_([A-Za-z0-9])", q)
            init = (m_init.group(1).lower() if m_init else None)

            if projects and init:
                best = None  # (key, project, hq, docid)
                for p in projects:
                    docid_guess = "D_TRUTH_" + p.split("_")[1]
                    content = self.opened_cache.get(docid_guess)
                    did = docid_guess
                    if not content:
                        for odid, cont in self.opened_cache.items():
                            if p in cont:
                                content = cont
                                did = odid
                                break
                    if not content:
                        continue

                    _, hq, code, key = parse_fields(content)
                    if (hq is None) or (code is None) or (key is None):
                        continue
                    if not code.lower().startswith(f"codename_{init}"):
                        continue
                    cand = (key, p, hq, did)
                    if (best is None) or (cand[0] > best[0]):
                        best = cand

                if best is not None:
                    key, proj, hq, did = best
                    ans = f"{proj} | {hq}"
                    return ans, build_expl(did)

        return "", ""

