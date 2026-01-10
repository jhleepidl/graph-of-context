from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
import re

from .utils import approx_token_count, tokenize
from .retrievers.base import TextItem, TextRetriever
from .retrievers.factory import build_retriever

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

DOCID_RE = re.compile(r"D_[A-Z]+_[0-9_]+")

@dataclass
class MemoryNode:
    id: str
    thread: str
    kind: str  # 'msg', 'tool', 'summary'
    text: str
    docids: List[str] = field(default_factory=list)
    token_len: int = 0
    step_idx: int = 0  # global, increasing
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    ttl: Optional[int] = None

    def __post_init__(self):
        if self.token_len <= 0:
            self.token_len = approx_token_count(self.text)

@dataclass
class MemoryManagerBase:
    budget_active: int = 2000
    budget_unfold: int = 800
    ttl_unfold: int = 4

    active: List[str] = field(default_factory=list)         # node ids in active context
    nodes: Dict[str, MemoryNode] = field(default_factory=dict)

    current_thread: str = "main"
    branch_stack: List[Tuple[str, List[str]]] = field(default_factory=list)  # (branch_id, branch_node_ids)

    _global_step: int = 0

    def reset(self):
        self.active = []
        self.nodes = {}
        self.current_thread = "main"
        self.branch_stack = []
        self._global_step = 0

    # ------------ recording helpers ------------
    def _next_step(self) -> int:
        self._global_step += 1
        return self._global_step

    def add_node(self, thread: str, kind: str, text: str, docids: Optional[List[str]] = None) -> str:
        nid = _new_id("N")
        step_idx = self._next_step()
        n = MemoryNode(id=nid, thread=thread, kind=kind, text=text, docids=docids or [], step_idx=step_idx)
        self.nodes[nid] = n
        return nid

    def active_tokens(self) -> int:
        return sum(self.nodes[nid].token_len for nid in self.active if nid in self.nodes)

    def get_active_text(self) -> str:
        return "\n".join(self.nodes[nid].text for nid in self.active if nid in self.nodes)

    def _decay_ttl(self):
        # called on each record_*; remove expired nodes from active (but keep in nodes)
        new_active = []
        for nid in self.active:
            n = self.nodes.get(nid)
            if not n:
                continue
            if n.ttl is None:
                new_active.append(nid)
                continue
            n.ttl -= 1
            if n.ttl > 0:
                new_active.append(nid)
        self.active = new_active

    # ------------ tool-like semantics ------------
    def branch(self, description: str, prompt: str) -> str:
        bid = _new_id("BR")
        self.branch_stack.append((bid, []))
        self.current_thread = bid
        nid = self.add_node(thread=bid, kind="msg", text=f"[BRANCH:{description}] {prompt}")
        self._on_branch_step(nid)
        return bid

    def return_from_branch(self, message: str) -> str:
        if not self.branch_stack:
            raise RuntimeError("return called without active branch")
        bid, branch_nodes = self.branch_stack.pop()

        self.current_thread = "main"
        ret_id = self.add_node(thread="main", kind="summary", text=f"[RETURN] {message}")
        self.active.append(ret_id)

        self._on_branch_return(bid, branch_nodes, ret_id)
        return ret_id

    # ------------ hooks (override in subclasses) ------------
    def _on_branch_step(self, nid: str):
        self.active.append(nid)

    def _on_branch_return(self, bid: str, branch_nodes: List[str], ret_id: str):
        pass

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids: Optional[List[str]] = None):
        self._decay_ttl()
        nid = self.add_node(
            thread=self.current_thread,
            kind="tool",
            text=f"[TOOL:{tool_name}] args={args}\nobs={observation}",
            docids=docids or []
        )
        self._on_branch_step(nid)
        self.maybe_fold()

    def record_msg(self, text: str):
        self._decay_ttl()
        nid = self.add_node(thread=self.current_thread, kind="msg", text=text)
        self._on_branch_step(nid)
        self.maybe_fold()

    def record_summary(self, text: str, docids: Optional[List[str]] = None, ttl: Optional[int] = None):
        """Record a summary-like node into ACTIVE context.

        This is useful for durable, compact annotations (e.g., constraints, policy notes)
        that should be preferentially retained by memory managers (GoC keeps summaries
        as anchors during pruning).
        """
        self._decay_ttl()
        nid = self.add_node(thread=self.current_thread, kind="summary", text=text, docids=docids or [])
        if ttl is not None and nid in self.nodes:
            self.nodes[nid].ttl = ttl
        self._on_branch_step(nid)
        self.maybe_fold()

    # ------------ folding/unfolding ------------
    def maybe_fold(self):
        # default: prune oldest until under budget (no summary)
        while self.active_tokens() > self.budget_active and len(self.active) > 1:
            self.active.pop(0)

    def unfold(self, query: str, k: int = None):
        return []


# ======================
# Baselines
# ======================

@dataclass
class FullHistoryMemory(MemoryManagerBase):
    def maybe_fold(self):
        while self.active_tokens() > self.budget_active and len(self.active) > 1:
            self.active.pop(0)

@dataclass
class ContextFoldingDiscardMemory(MemoryManagerBase):
    def branch(self, description: str, prompt: str) -> str:
        bid = super().branch(description, prompt)
        self.branch_stack[-1] = (bid, [n for n in self.active if self.nodes[n].thread == bid])
        return bid

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids=None):
        super().record_tool(tool_name, args, observation, docids=docids)
        if self.branch_stack:
            bid, lst = self.branch_stack[-1]
            lst.append(self.active[-1])
            self.branch_stack[-1] = (bid, lst)

    def record_msg(self, text: str):
        super().record_msg(text)
        if self.branch_stack:
            bid, lst = self.branch_stack[-1]
            lst.append(self.active[-1])
            self.branch_stack[-1] = (bid, lst)

    def _on_branch_return(self, bid: str, branch_nodes: List[str], ret_id: str):
        # remove all branch nodes from active and delete them (hard discard)
        self.active = [nid for nid in self.active if self.nodes[nid].thread != bid]
        for nid in list(self.nodes.keys()):
            if self.nodes[nid].thread == bid:
                del self.nodes[nid]
        self.maybe_fold()

@dataclass
class LinearSummaryMemory(MemoryManagerBase):
    summary_every: int = 8
    _step_count: int = 0

    def reset(self):
        super().reset()
        self._step_count = 0

    def maybe_fold(self):
        self._step_count += 1
        if self._step_count % self.summary_every == 0 and len(self.active) > 2:
            lines = []
            # lossy: summarize last few nodes
            for nid in self.active[-10:]:
                t = self.nodes[nid].text.splitlines()[0][:160]
                lines.append(t)
            stext = "[LINEAR_SUMMARY] " + " | ".join(lines[:8])
            sid = self.add_node(thread="main", kind="summary", text=stext)
            # discard originals (simulate loss)
            self.active = [sid]

        while self.active_tokens() > self.budget_active and len(self.active) > 1:
            self.active.pop(0)

@dataclass
class AgentFoldRangeMemory(MemoryManagerBase):
    """AgentFold-like baseline: fold a contiguous range of old steps into a summary node.

    This is a heuristic stand-in for AgentFold's range-based folding directives.
    It creates a folding directive record and discards the originals (lossy).
    """
    fold_chunk: int = 10

    def maybe_fold(self):
        while self.active_tokens() > self.budget_active and len(self.active) > 2:
            # fold the oldest contiguous chunk
            chunk = self.active[: min(self.fold_chunk, len(self.active)-1)]
            if len(chunk) <= 1:
                break

            start_idx = self.nodes[chunk[0]].step_idx
            end_idx = self.nodes[chunk[-1]].step_idx
            # lossy summary: first line per node
            parts = []
            for nid in chunk:
                parts.append(self.nodes[nid].text.splitlines()[0][:140])
            summary = "[AGENTFOLD_SUMMARY] " + " | ".join(parts[:10])

            directive = {"range": [start_idx, end_idx], "summary": summary}
            dir_text = f"[FOLDING] {directive}"
            dir_id = self.add_node(thread="main", kind="summary", text=dir_text)
            sum_id = self.add_node(thread="main", kind="summary", text=summary)
            # replace chunk with sum_id (keep directive for traceability)
            self.active = [dir_id, sum_id] + self.active[len(chunk):]

            # discard folded originals for lossy behavior
            for nid in chunk:
                if nid in self.nodes:
                    del self.nodes[nid]


# ======================
# GoC (Graph-of-Context)
# ======================

@dataclass
class GoCMemory(MemoryManagerBase):
    unfold_k: int = 6
    max_dep_depth: int = 3
    doc_ref_expand: int = 6

    # preserved but not active
    storage: List[str] = field(default_factory=list)
    _storage_retriever: Optional[TextRetriever] = None

    # storage retrieval backend
    storage_retriever_kind: str = "bm25"
    storage_faiss_dim: int = 384

    # lightweight graph
    docid_to_nodes: Dict[str, List[str]] = field(default_factory=dict)  # docid -> node ids
    edges_out: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: {"depends": {}, "doc_ref": {}, "seq": {}})
    edges_in: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: {"depends": {}, "doc_ref": {}, "seq": {}})

    _last_in_thread: Dict[str, str] = field(default_factory=dict)  # thread -> last node id

    def reset(self):
        super().reset()
        self.storage = []
        self._storage_retriever = None
        self.docid_to_nodes = {}
        self.edges_out = {"depends": {}, "doc_ref": {}, "seq": {}}
        self.edges_in = {"depends": {}, "doc_ref": {}, "seq": {}}
        self._last_in_thread = {}

    # ---- graph helpers ----
    def _add_edge(self, etype: str, u: str, v: str):
        self.edges_out.setdefault(etype, {}).setdefault(u, set()).add(v)
        self.edges_in.setdefault(etype, {}).setdefault(v, set()).add(u)

    def _neighbors(self, etype: str, u: str) -> Set[str]:
        return set(self.edges_out.get(etype, {}).get(u, set()))

    def _parents(self, etype: str, v: str) -> Set[str]:
        return set(self.edges_in.get(etype, {}).get(v, set()))

    def _index_docids(self, nid: str):
        n = self.nodes.get(nid)
        if not n:
            return
        for d in n.docids:
            lst = self.docid_to_nodes.setdefault(d, [])
            # connect doc_ref edges to prior nodes with same docid
            for prior in lst[-20:]:
                if prior != nid and prior in self.nodes:
                    self._add_edge("doc_ref", nid, prior)
                    self._add_edge("doc_ref", prior, nid)
            lst.append(nid)

    def _link_sequential(self, nid: str):
        prev = self._last_in_thread.get(self.current_thread)
        if prev and prev in self.nodes:
            self._add_edge("seq", prev, nid)
            # also a weak "depends": later steps depend on earlier steps in same thread (helps closure)
            self._add_edge("depends", nid, prev)
        self._last_in_thread[self.current_thread] = nid

    def add_node(self, thread: str, kind: str, text: str, docids: Optional[List[str]] = None) -> str:
        nid = super().add_node(thread=thread, kind=kind, text=text, docids=docids)
        # graph updates
        self._link_sequential(nid)
        self._index_docids(nid)
        return nid

    # ---- storage index ----
    def _rebuild_storage_index(self):
        items: List[TextItem] = []
        for nid in self.storage:
            n = self.nodes.get(nid)
            if not n:
                continue
            items.append(TextItem(id=nid, text=n.text, meta={"url": f"mem://{nid}", "title": n.kind}))
        self._storage_retriever = build_retriever(self.storage_retriever_kind, items, faiss_dim=self.storage_faiss_dim) if items else None

    # ---- overrides to track branch nodes ----
    def branch(self, description: str, prompt: str) -> str:
        bid = super().branch(description, prompt)
        self.branch_stack[-1] = (bid, [n for n in self.active if self.nodes[n].thread == bid])
        return bid

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids=None):
        super().record_tool(tool_name, args, observation, docids=docids)
        if self.branch_stack:
            bid, lst = self.branch_stack[-1]
            lst.append(self.active[-1])
            self.branch_stack[-1] = (bid, lst)

    def record_msg(self, text: str):
        # attach docids found in text as doc_ref anchors
        docids = DOCID_RE.findall(text or "")
        self._decay_ttl()
        nid = self.add_node(thread=self.current_thread, kind="msg", text=text, docids=docids)
        self._on_branch_step(nid)
        if self.branch_stack:
            bid, lst = self.branch_stack[-1]
            lst.append(nid)
            self.branch_stack[-1] = (bid, lst)
        self.maybe_fold()

    def _on_branch_return(self, bid: str, branch_nodes: List[str], ret_id: str):
        # Fold out branch nodes from active, preserve in storage
        to_remove = [nid for nid in self.active if self.nodes.get(nid) and self.nodes[nid].thread == bid]
        self.active = [nid for nid in self.active if nid not in to_remove]

        for nid in to_remove:
            if nid not in self.storage:
                self.storage.append(nid)

        # Summary proxy node for branch + dependency edges to children
        proxy_text = self.nodes[ret_id].text
        proxy_id = self.add_node(thread="main", kind="summary", text=f"[BRANCH_PROXY:{bid}] {proxy_text}")
        self.nodes[proxy_id].children = list(to_remove)

        # depends edges: proxy depends on its children (so unfolding proxy can bring children)
        for child in to_remove:
            if child in self.nodes:
                self._add_edge("depends", proxy_id, child)

        self.active.append(proxy_id)

        self._rebuild_storage_index()
        self.maybe_fold()

    
    # ---- proxy summarization helpers (hierarchical folding) ----
    _PROXY_DEPTH_RE = re.compile(r"\[GOCPROXY depth=(\d+)\]")
    _PROJECT_RE = re.compile(r"(Project_[0-9]{4})")
    _KV_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$")

    def _node_proxy_depth(self, nid: str) -> int:
        n = self.nodes.get(nid)
        if not n:
            return 0
        m = self._PROXY_DEPTH_RE.search(n.text or "")
        return int(m.group(1)) if m else 0

    def _extract_facts_from_text(self, txt: str) -> Dict[str, str]:
        """
        Best-effort extraction of key fields from an OPEN_PAGE-like profile or proxy summary.
        Returns a dict with keys like: project, start_year, headquarters, code_name, key_number, related_projects.
        """
        facts: Dict[str, str] = {}
        if not txt:
            return facts

        # Project id
        pm = self._PROJECT_RE.search(txt)
        if pm:
            facts["project"] = pm.group(1)

        # Parse line-based key: value
        lines = txt.splitlines()
        for ln in lines[:200]:
            m = self._KV_RE.match(ln)
            if not m:
                continue
            k = m.group(1).strip().lower()
            v = m.group(2).strip()
            if k in ("start_year", "headquarters", "code_name", "key_number", "lead", "related_projects"):
                # keep first occurrence
                if k not in facts and v:
                    facts[k] = v

        # Also accept compact formats in proxy bullets: "start_year=1997", "HQ=City_43", etc.
        if "start_year" not in facts:
            m = re.search(r"\bstart_year\s*=\s*([0-9]{3,4})\b", txt)
            if m:
                facts["start_year"] = m.group(1)
        if "headquarters" not in facts:
            m = re.search(r"\bHQ\s*=\s*([A-Za-z0-9_]+)\b", txt)
            if m:
                facts["headquarters"] = m.group(1)
        if "code_name" not in facts:
            m = re.search(r"\bcode(?:_name)?\s*=\s*([A-Za-z0-9_]+)\b", txt)
            if m:
                facts["code_name"] = m.group(1)
        if "key_number" not in facts:
            m = re.search(r"\bkey(?:_number)?\s*=\s*([0-9]+)\b", txt)
            if m:
                facts["key_number"] = m.group(1)

        return facts

    def _summarize_chunk_facts(self, chunk: List[str], max_lines: int = 12) -> str:
        """
        Build a compact, parseable "facts-first" proxy summary for a set of nodes.
        We intentionally keep this short so it can remain in ACTIVE_CONTEXT under budget.
        """
        # Aggregate per-project facts when possible
        by_project: Dict[str, Dict[str, str]] = {}
        docids: List[str] = []
        for nid in chunk:
            n = self.nodes.get(nid)
            if not n:
                continue
            for d in (n.docids or []):
                if d not in docids:
                    docids.append(d)
            facts = self._extract_facts_from_text(n.text or "")
            proj = facts.get("project")
            if proj:
                cur = by_project.setdefault(proj, {})
                for k, v in facts.items():
                    if k not in cur and v:
                        cur[k] = v

        lines: List[str] = []
        if by_project:
            # Deterministic ordering: by earliest step the project appears in this chunk
            proj_order: List[str] = []
            for nid in chunk:
                n = self.nodes.get(nid)
                if not n:
                    continue
                m = self._PROJECT_RE.search(n.text or "")
                if m and m.group(1) in by_project and m.group(1) not in proj_order:
                    proj_order.append(m.group(1))
            # Add any missing
            for p in sorted(by_project.keys()):
                if p not in proj_order:
                    proj_order.append(p)

            for p in proj_order[:max_lines]:
                f = by_project[p]
                parts = [p]
                if "start_year" in f:
                    parts.append(f"start_year={f['start_year']}")
                if "headquarters" in f:
                    parts.append(f"HQ={f['headquarters']}")
                if "code_name" in f:
                    parts.append(f"code={f['code_name']}")
                if "key_number" in f:
                    parts.append(f"key={f['key_number']}")
                if "related_projects" in f:
                    rp = f["related_projects"]
                    # keep short
                    rp = rp[:80] + ("..." if len(rp) > 80 else "")
                    parts.append(f"rel={rp}")
                lines.append("- " + " | ".join(parts))

        # Fallback if we couldn't extract structured facts
        if not lines:
            for nid in chunk[:max_lines]:
                n = self.nodes.get(nid)
                if not n:
                    continue
                first = (n.text or "").strip().splitlines()[:1]
                if first:
                    s = first[0]
                    s = s[:120] + ("..." if len(s) > 120 else "")
                    lines.append("- " + s)

        if docids:
            dids = ", ".join(docids[:8])
            if len(docids) > 8:
                dids += ", ..."
            lines.append(f"Docids: {dids}")

        return "\n".join(lines[:max_lines])

    def _create_proxy_node(self, chunk: List[str]) -> str:
        """
        Create a proxy summary node that stands in for `chunk` and can be recursively folded.
        IMPORTANT: We assign step_idx to the earliest child step so chronological ordering works.
        """
        if not chunk:
            raise ValueError("empty chunk for proxy")
        child_steps = [self.nodes[nid].step_idx for nid in chunk if nid in self.nodes]
        step_min = min(child_steps) if child_steps else 0
        step_max = max(child_steps) if child_steps else step_min
        depth = 1 + max((self._node_proxy_depth(nid) for nid in chunk), default=0)

        # Union docids
        docids: List[str] = []
        for nid in chunk:
            n = self.nodes.get(nid)
            if not n:
                continue
            for d in (n.docids or []):
                if d not in docids:
                    docids.append(d)

        body = self._summarize_chunk_facts(chunk)
        header = f"[GOCPROXY depth={depth} steps={step_min}-{step_max} n={len(chunk)}]"
        proxy_text = header + "\n" + body

        # Create node WITHOUT sequential linking (use base add_node), then override step_idx.
        proxy_id = MemoryManagerBase.add_node(self, thread="main", kind="summary", text=proxy_text, docids=docids)
        if proxy_id in self.nodes:
            self.nodes[proxy_id].children = list(chunk)
            self.nodes[proxy_id].step_idx = step_min  # keep proxy near the original time window
            # refresh token_len after edits
            self.nodes[proxy_id].token_len = approx_token_count(self.nodes[proxy_id].text)

        # Graph hooks: docid index + depends edges so unfolding proxy can bring children
        self._index_docids(proxy_id)
        for child in chunk:
            if child in self.nodes:
                self.nodes[child].parent = proxy_id
                self._add_edge("depends", proxy_id, child)
        return proxy_id

    def _replace_prefix_with_proxy(self, chunk: List[str], proxy_id: str):
        """Replace the prefix `chunk` in ACTIVE with the proxy node, keep proxy ACTIVE."""
        if not chunk:
            return
        # Expect chunk to be a prefix of active (we construct it that way)
        k = len(chunk)
        if self.active[:k] != chunk:
            # Fallback: remove any occurrences of chunk members
            remaining = [nid for nid in self.active if nid not in set(chunk)]
            self.active = [proxy_id] + remaining
        else:
            self.active = [proxy_id] + self.active[k:]

    def _prune_oldest_non_summary_to_storage(self):
        """As a last resort, remove the oldest NON-summary node (keep summaries as anchors)."""
        if len(self.active) <= 1:
            return
        # protect the most recent node
        prefix = self.active[:-1]
        idx = None
        for i, nid in enumerate(prefix):
            n = self.nodes.get(nid)
            if n and n.kind != "summary":
                idx = i
                break
        if idx is None:
            # if everything is summary, prune the oldest summary
            idx = 0
        nid = self.active.pop(idx)
        if nid in self.nodes and nid not in self.storage:
            self.storage.append(nid)

    # ---- folding: keep removed nodes in storage for recovery ----
    def maybe_fold(self):
        """
        Hierarchical folding policy:
        - When ACTIVE exceeds budget, fold the oldest prefix into a compact proxy summary node that stays ACTIVE.
        - Move original nodes into STORAGE (lossless; recoverable via unfold).
        - If proxy nodes accumulate, they can be folded again (proxy-of-proxy) to keep ACTIVE small.
        - As a last resort, prune oldest non-summary nodes first (keep summaries as anchors).
        """
        # Fast exit
        if self.active_tokens() <= self.budget_active or len(self.active) <= 1:
            return

        # Fold until under budget (or we cannot make progress)
        safety = 0
        while self.active_tokens() > self.budget_active and len(self.active) > 1 and safety < 50:
            safety += 1
            overflow = self.active_tokens() - self.budget_active
            # We keep the most recent node to preserve immediate context.
            prefix = self.active[:-1]
            if not prefix:
                break

            # Target: remove at least overflow tokens, plus a small buffer.
            target_remove = max(overflow + 50, int(self.budget_active * 0.25))
            max_chunk = 8

            chunk: List[str] = []
            removed = 0
            for nid in prefix:
                if nid not in self.nodes:
                    continue
                chunk.append(nid)
                removed += self.nodes[nid].token_len
                if removed >= target_remove or len(chunk) >= max_chunk:
                    break

            # Allow single-node folding if the oldest node is huge (common with long descriptions).
            if len(chunk) == 1:
                n0 = self.nodes.get(chunk[0])
                if not n0 or (n0.token_len < int(self.budget_active * 0.40) and removed < target_remove):
                    chunk = []  # fall back to prune

            if chunk:
                before = self.active_tokens()
                proxy_id = self._create_proxy_node(chunk)
                # Remove folded nodes from active and keep proxy active
                self._replace_prefix_with_proxy(chunk, proxy_id)
                # Move children to storage (lossless)
                for nid in chunk:
                    if nid in self.nodes and nid not in self.storage:
                        self.storage.append(nid)
                after = self.active_tokens()
                # If folding didn't reduce anything (should be rare), prune one node to avoid infinite loops.
                if after >= before:
                    self._prune_oldest_non_summary_to_storage()
                continue

            # Fallback pruning: keep summaries as anchors
            self._prune_oldest_non_summary_to_storage()

        self._rebuild_storage_index()

    # ---- minimal-closure unfold ----
    def _dep_closure(self, seeds: List[str], depth: int) -> Set[str]:
        visited: Set[str] = set()
        frontier: Set[str] = set(seeds)
        for _ in range(depth):
            new_frontier: Set[str] = set()
            for u in frontier:
                if u in visited:
                    continue
                visited.add(u)
                # parents along depends (bring prerequisites)
                for p in self._neighbors("depends", u):
                    if p not in visited:
                        new_frontier.add(p)
            frontier = new_frontier
            if not frontier:
                break
        return visited

    def _doc_ref_expand(self, nodes: Set[str], limit_per_node: int) -> Set[str]:
        expanded = set(nodes)
        for u in list(nodes):
            neigh = list(self._neighbors("doc_ref", u))
            for v in neigh[:limit_per_node]:
                expanded.add(v)
        return expanded

    def unfold(self, query: str, k: int = None):
        if k is None:
            k = self.unfold_k
        if not self._storage_retriever or not self.storage:
            return []

        hits = self._storage_retriever.search(query, topk=max(k*3, 10))

        # Build candidate closures
        candidates = []
        for nid, score in hits:
            if nid not in self.nodes:
                continue
            if nid in self.active:
                continue
            # closure: node + depends closure + doc_ref neighbors
            closure = self._dep_closure([nid], depth=self.max_dep_depth)
            closure = self._doc_ref_expand(closure, limit_per_node=self.doc_ref_expand)
            # ensure closure is within known nodes
            closure = {x for x in closure if x in self.nodes}
            cost = sum(self.nodes[x].token_len for x in closure if x not in self.active)
            candidates.append((score, cost, closure))

        # Greedy selection under unfold budget
        candidates.sort(key=lambda x: x[0] / max(1, x[1]), reverse=True)
        chosen: Set[str] = set()
        used = 0
        for score, cost, closure in candidates:
            add_cost = sum(self.nodes[x].token_len for x in closure if x not in self.active and x not in chosen)
            if used + add_cost > self.budget_unfold:
                continue
            chosen |= closure
            used += add_cost
            if len(chosen) >= k:
                break

        activated = []
        for nid in sorted(chosen, key=lambda x: self.nodes[x].step_idx):
            if nid not in self.active:
                self.nodes[nid].ttl = self.ttl_unfold
                self.active.append(nid)
                activated.append(nid)

        # Re-order active nodes chronologically by step index so recovered nodes
        # do not end up at the very end of the context.
        self.active.sort(key=lambda x: self.nodes[x].step_idx)
        return activated
