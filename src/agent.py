from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re

from .tools import ToolBox
from .memory import MemoryManagerBase
from .utils import approx_token_count

ATTR_PATTERNS_COLON = {
    "start_year": re.compile(r"start_year:\s*(\d{4})", re.I),
    "headquarters": re.compile(r"headquarters:\s*([A-Za-z0-9_\-]+)", re.I),
    "lead": re.compile(r"lead:\s*([A-Za-z0-9_\-]+)", re.I),
    "code_name": re.compile(r"code_name:\s*([A-Za-z0-9_\-]+)", re.I),
    "key_number": re.compile(r"key_number:\s*(\d+)", re.I),
}
ATTR_PATTERNS_EQ = {
    "start_year": re.compile(r"start_year\s*=\s*(\d{4})", re.I),
    "headquarters": re.compile(r"headquarters\s*=\s*([A-Za-z0-9_\-]+)", re.I),
    "lead": re.compile(r"lead\s*=\s*([A-Za-z0-9_\-]+)", re.I),
    "code_name": re.compile(r"code_name\s*=\s*([A-Za-z0-9_\-]+)", re.I),
    "key_number": re.compile(r"key_number\s*=\s*(\d+)", re.I),
}

ENTITY_RE = re.compile(r"Project_\d{4}")
DOCID_RE = re.compile(r"D_[A-Z]+_[0-9_]+")

def extract_entities(text: str) -> List[str]:
    return ENTITY_RE.findall(text)

def _parse_attrs(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a, pat in ATTR_PATTERNS_COLON.items():
        m = pat.search(text)
        if m:
            out[a] = m.group(1)
    for a, pat in ATTR_PATTERNS_EQ.items():
        if a in out:
            continue
        m = pat.search(text)
        if m:
            out[a] = m.group(1)
    return out

def parse_context_facts(context: str) -> Dict[str, Dict[str, str]]:
    """Parse facts from context without cross-entity bleeding.

    Split into blocks by markers, assign block to first entity mentioned, parse attrs in-block.
    Last-wins so that later recovery can override earlier partial info.
    """
    blocks: List[str] = []
    split_re = re.compile(r"(?=\[TOOL:open_page\]|\[RETURN\]|\[RECOVER\]|\[OBS\]|\[USER\]|\[BRANCH_PROXY:)")
    parts = split_re.split(context)
    for p in parts:
        p = p.strip()
        if p:
            blocks.append(p)

    facts: Dict[str, Dict[str, str]] = {}

    for b in blocks:
        ents = extract_entities(b)
        if not ents:
            continue
        ent = ents[0]
        attrs = _parse_attrs(b)
        if not attrs:
            continue
        facts.setdefault(ent, {})
        for k, v in attrs.items():
            facts[ent][k] = v  # last-wins
    return facts

def parse_docids(context: str) -> List[str]:
    return sorted(set(DOCID_RE.findall(context or "")))

@dataclass
class AgentConfig:
    topk: int = 5
    summary_keep_fields: int = 1  # number of fields kept in return summary besides start_year
    unfold_k: int = 6

class ContextLimitedAgent:
    """Deterministic agent that uses *active context* as its only memory.

    It uses Context-Folding style `branch` to gather facts. `return` summaries are lossy
    (simulate fold info loss). GoC can unfold from preserved storage.
    """

    def __init__(self, tools: ToolBox, mem: MemoryManagerBase, cfg: AgentConfig):
        self.tools = tools
        self.mem = mem
        self.cfg = cfg
        self.llm_calls = 0
        self.llm_in_tokens = 0
        self.llm_out_tokens = 0
        self.tool_calls = 0
        self.peak_active_tokens = 0

    def _tick_llm(self, out_text: str):
        self.llm_calls += 1
        inp = approx_token_count(self.mem.get_active_text())
        outp = approx_token_count(out_text)
        self.llm_in_tokens += inp
        self.llm_out_tokens += outp
        self.peak_active_tokens = max(self.peak_active_tokens, self.mem.active_tokens())

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mem.reset()
        self.llm_calls = self.llm_in_tokens = self.llm_out_tokens = 0
        self.tool_calls = 0
        self.peak_active_tokens = 0

        q = task["question"]
        entities = task.get("entities") or extract_entities(q)
        required = task.get("required", ["start_year", "headquarters"])

        self.mem.record_msg(f"[USER] {q}")
        self._tick_llm("Plan: gather per-entity facts via branches; then answer.")

        # Branch per entity: search + open_page + extract (deterministic)
        for ent in entities:
            self.mem.branch(description=f"facts_{ent}", prompt=f"Find {ent} facts: {', '.join(required)}.")
            self._tick_llm(f"Branching for {ent}.")

            query = f"{ent} start_year headquarters lead code_name key_number OFFICIAL PROFILE authoritative"
            self.tool_calls += 1
            hits = self.tools.search(query=query, topk=self.cfg.topk)
            self.mem.record_tool("search", {"query": query, "topk": self.cfg.topk}, observation=str([h['docid'] for h in hits]))

            top = hits[0]
            self.tool_calls += 1
            page = self.tools.open_page(docid=top["docid"])
            self.mem.record_tool("open_page", {"docid": top["docid"]}, observation=page["content"][:2200], docids=[top["docid"]])

            facts = _parse_attrs(page["content"])
            self.mem.record_msg(f"[OBS] {ent} from {top['docid']}: " + ", ".join([f"{k}={v}" for k,v in facts.items()]))
            self._tick_llm(f"Extracted facts for {ent}.")

            # lossy return: keep start_year and N other fields, HQ last by design
            kept = {}
            if "start_year" in facts:
                kept["start_year"] = facts["start_year"]
            other_keys = [k for k in ["lead", "code_name", "key_number", "headquarters"] if k in facts and k != "start_year"]
            for k in other_keys[: max(0, self.cfg.summary_keep_fields)]:
                kept[k] = facts[k]

            ret_msg = f"{ent} facts (doc={top['docid']}): " + ", ".join([f"{k}={v}" for k,v in kept.items()])
            self.mem.return_from_branch(message=ret_msg)
            self._tick_llm(f"Returned summary for {ent}.")

        # Parse facts from active context (main summaries/proxies)
        context = self.mem.get_active_text()
        ent_facts = parse_context_facts(context)
        for e in entities:
            ent_facts.setdefault(e, {})

        # Decide earliest by start_year (usually available via return summary)
        years = []
        for e in entities:
            y = ent_facts[e].get("start_year")
            years.append((e, int(y) if y is not None else 9999))
        best_e, _ = min(years, key=lambda x: x[1])

        # Targeted unfold: only retrieve missing facts for the chosen entity (min-closure)
        if "headquarters" not in ent_facts[best_e] and hasattr(self.mem, "unfold"):
            uq = f"{best_e} headquarters"
            unfolded = self.mem.unfold(uq, k=self.cfg.unfold_k)
            if unfolded:
                self._tick_llm(f"Unfolded nodes (targeted): {unfolded}")
                context = self.mem.get_active_text()
                ent_facts = parse_context_facts(context)
                for e in entities:
                    ent_facts.setdefault(e, {})

        # If still missing HQ, do final re-search (may be misled by distractors for lossy methods)
        if "headquarters" not in ent_facts[best_e]:
            query = f"{best_e} headquarters"
            self.tool_calls += 1
            hits = self.tools.search(query=query, topk=self.cfg.topk)
            self.mem.record_tool("search", {"query": query, "topk": self.cfg.topk}, observation=str([h['docid'] for h in hits]))
            top = hits[0]
            self.tool_calls += 1
            page = self.tools.open_page(docid=top["docid"])
            self.mem.record_tool("open_page", {"docid": top["docid"]}, observation=page["content"][:1800], docids=[top["docid"]])
            facts = _parse_attrs(page["content"])
            if "headquarters" in facts:
                self.mem.record_msg(f"[RECOVER] {best_e} headquarters from {top['docid']}: headquarters={facts['headquarters']}")
                self._tick_llm("Recovered headquarters.")
                context = self.mem.get_active_text()
                ent_facts = parse_context_facts(context)

        hq = ent_facts.get(best_e, {}).get("headquarters", "UNKNOWN")
        answer = f"{best_e} | {hq}"

        docids = parse_docids(self.mem.get_active_text())
        explanation = "Evidence docids: " + ", ".join(docids[:6])

        self._tick_llm("Final answer computed.")
        return {
            "answer": answer,
            "explanation": explanation,
            "metrics": {
                "llm_calls": self.llm_calls,
                "llm_in_tokens": self.llm_in_tokens,
                "llm_out_tokens": self.llm_out_tokens,
                "tool_calls": self.tool_calls,
                "peak_active_tokens": self.peak_active_tokens,
                "final_active_tokens": self.mem.active_tokens()
            }
        }
