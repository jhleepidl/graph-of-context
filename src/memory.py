from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
import re

# Network flow / min-cut for token-weighted folding.
import networkx as nx

from .utils import approx_token_count, tokenize
from .retrievers.base import TextItem, TextRetriever
from .retrievers.factory import build_retriever


class RRFMultiRetriever:
    """Combine multiple retrievers with Reciprocal Rank Fusion (RRF).

    We use RRF because raw scores across independently-built BM25/FAISS indices
    are not reliably comparable. RRF is rank-based and cheap.
    """

    def __init__(self, retrievers: List[TextRetriever], *, k0: int = 60):
        self.retrievers = [r for r in (retrievers or []) if r is not None]
        self.k0 = int(k0)

    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        topk = max(1, int(topk))
        if not self.retrievers:
            return []
        # Pull a bit more from each segment then fuse.
        per = max(10, topk * 3)
        scores: Dict[str, float] = {}
        for r in self.retrievers:
            try:
                hits = r.search(query, topk=per)
            except Exception:
                hits = []
            for rank, (nid, _raw) in enumerate(hits):
                scores[nid] = scores.get(nid, 0.0) + 1.0 / float(self.k0 + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    def size(self) -> int:
        s = 0
        for r in self.retrievers:
            try:
                s += int(r.size())
            except Exception:
                pass
        return s

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

DOCID_RE = re.compile(r"D_[A-Z]+_[0-9_]+")

@dataclass
class MemoryNode:
    id: str
    thread: str
    kind: str  # 'msg', 'tool', 'summary'
    text: str
    # Optional lossless storage (e.g., full open_page content). Not counted against active budget.
    storage_text: Optional[str] = None
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
    enforce_budget: bool = True  # if False, never fold/prune based on budget_active

    active: List[str] = field(default_factory=list)         # node ids in active context
    nodes: Dict[str, MemoryNode] = field(default_factory=dict)

    current_thread: str = "main"
    branch_stack: List[Tuple[str, List[str]]] = field(default_factory=list)  # (branch_id, branch_node_ids)

    _global_step: int = 0
    # Optional event buffer for tracing (e.g., fold/unfold decisions). Subclasses
    # can push structured events here; the agent can drain and log them.
    _event_buf: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def reset(self):
        self.active = []
        self.nodes = {}
        self.current_thread = "main"
        self.branch_stack = []
        self._global_step = 0
        self._event_buf = []

    def _emit_event(self, ev: Dict[str, Any]):
        """Add an event to the buffer for later tracing."""
        try:
            self._event_buf.append(ev)
        except Exception:
            # Never let tracing crash memory
            pass

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return and clear buffered events."""
        evs = self._event_buf
        self._event_buf = []
        return evs

    # ------------ recording helpers ------------
    def _next_step(self) -> int:
        self._global_step += 1
        return self._global_step

    def add_node(self, thread: str, kind: str, text: str, docids: Optional[List[str]] = None, storage_text: Optional[str] = None) -> str:
        nid = _new_id("N")
        step_idx = self._next_step()
        n = MemoryNode(id=nid, thread=thread, kind=kind, text=text, storage_text=storage_text, docids=docids or [], step_idx=step_idx)
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

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids: Optional[List[str]] = None, storage_text: Optional[str] = None):
        self._decay_ttl()
        nid = self.add_node(
            thread=self.current_thread,
            kind="tool",
            text=f"[TOOL:{tool_name}] args={args}\nobs={observation}",
            docids=docids or [],
            storage_text=storage_text,
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
        if (not self.enforce_budget) or (not self.budget_active) or (int(self.budget_active) <= 0):
            return
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
        if (not self.enforce_budget) or (not self.budget_active) or (int(self.budget_active) <= 0):
            return
        while self.active_tokens() > self.budget_active and len(self.active) > 1:
            self.active.pop(0)

@dataclass
class ContextFoldingDiscardMemory(MemoryManagerBase):
    def branch(self, description: str, prompt: str) -> str:
        bid = super().branch(description, prompt)
        self.branch_stack[-1] = (bid, [n for n in self.active if self.nodes[n].thread == bid])
        return bid

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids=None, storage_text: Optional[str] = None):
        super().record_tool(tool_name, args, observation, docids=docids, storage_text=storage_text)
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
        if (not self.enforce_budget) or (not self.budget_active) or (int(self.budget_active) <= 0):
            return
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
        if (not self.enforce_budget) or (not self.budget_active) or (int(self.budget_active) <= 0):
            return
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


@dataclass
class SimpleRAGMemory(MemoryManagerBase):
    """Simple RAG-style long-term memory baseline.

    This baseline is intentionally *not* graph-aware:
      - It stores past nodes losslessly (like a log).
      - On each step it retrieves top-k past items by similarity to the latest user/tool text.
      - It injects compact recall snippets into ACTIVE context with a short TTL.

    The goal is to represent a common "simple RAG memory" approach for long-horizon agents.
    """

    window_last_n: int = 10
    rag_k: int = 6
    retriever_kind: str = "bm25"
    faiss_dim: int = 384
    snippet_max_chars: int = 700
    recall_ttl: int = 5

    storage_ids: List[str] = field(default_factory=list)
    _retriever: Optional[TextRetriever] = None
    _dirty: bool = True

    def reset(self):
        super().reset()
        self.storage_ids = []
        self._retriever = None
        self._dirty = True

    def _extract_snippet(self, full: str, query: str, max_chars: int) -> str:
        """Cheap snippet selector (keyword window)."""
        if not full:
            return ""
        q = (query or "").strip()
        if not q:
            return full[:max_chars].strip()
        terms = re.findall(r"[A-Za-z_]{4,}", q.lower())[:6]
        hay = full
        hay_low = hay.lower()
        idx = -1
        for t in terms:
            i = hay_low.find(t)
            if i >= 0:
                idx = i
                break
        if idx < 0:
            idx = max(0, len(hay) - max_chars)
        win = max_chars // 2
        start = max(0, idx - win)
        end = min(len(hay), idx + win)
        snip = hay[start:end]
        # align to line boundaries
        s_nl = snip.rfind("\n", 0, min(200, len(snip)))
        if s_nl > 0:
            snip = snip[s_nl + 1 :]
        e_nl = snip.find("\n", max(0, len(snip) - 200))
        if e_nl > 0:
            snip = snip[:e_nl]
        if start > 0:
            snip = "..." + snip
        if end < len(hay):
            snip = snip + "..."
        return snip[:max_chars].strip()

    def _rebuild_retriever(self):
        items: List[TextItem] = []
        for nid in self.storage_ids:
            n = self.nodes.get(nid)
            if not n:
                continue
            text = (n.storage_text or n.text or "").strip()
            if not text:
                continue
            items.append(TextItem(id=nid, text=text, meta={"kind": n.kind, "docids": list(n.docids)}))
        self._retriever = build_retriever(self.retriever_kind, items, faiss_dim=int(self.faiss_dim)) if items else None
        self._dirty = False

    def _apply_window(self):
        """Keep last N non-recall nodes, plus recall snippets with TTL."""
        # Identify recall snippet nodes by prefix.
        recall_nodes = [
            nid for nid in self.active
            if (nid in self.nodes) and (self.nodes[nid].kind == "summary") and self.nodes[nid].text.startswith("[RAG_RECALL")
        ]
        base_nodes = [nid for nid in self.active if nid in self.nodes and nid not in recall_nodes]
        if self.window_last_n > 0 and len(base_nodes) > self.window_last_n:
            base_nodes = base_nodes[-int(self.window_last_n):]
        # Merge and keep chronological ordering.
        merged = base_nodes + recall_nodes
        merged = [nid for nid in merged if nid in self.nodes]
        merged.sort(key=lambda x: self.nodes[x].step_idx)
        self.active = merged

    def maybe_fold(self):
        # For this baseline, folding == enforce token budget by dropping oldest.
        if (not self.enforce_budget) or (not self.budget_active) or (int(self.budget_active) <= 0):
            return
        while self.active_tokens() > self.budget_active and len(self.active) > 1:
            self.active.pop(0)

    def _inject_recalls(self, query: str, k: Optional[int] = None) -> List[str]:
        if not query:
            return []
        if self._dirty:
            self._rebuild_retriever()
        if not self._retriever:
            return []

        k = int(k or self.rag_k)
        hits = self._retriever.search(query, topk=max(10, k * 4))
        activated: List[str] = []
        for nid, score in hits:
            if nid not in self.nodes:
                continue
            if nid in self.active:
                continue
            n = self.nodes[nid]
            src = n.storage_text or n.text
            snip = self._extract_snippet(src, query=query, max_chars=int(self.snippet_max_chars))
            if not snip:
                continue
            text = f"[RAG_RECALL src={nid} score={score:.3f}]\n{snip}"
            rid = self.add_node(thread="main", kind="summary", text=text, docids=list(n.docids))
            self.nodes[rid].ttl = int(self.recall_ttl)
            self.active.append(rid)
            activated.append(rid)
            if len(activated) >= k:
                break
        self.active.sort(key=lambda x: self.nodes[x].step_idx)
        if activated:
            self._emit_event({"type": "rag_unfold", "query": query, "k": k, "activated": activated[:20]})
        return activated

    # Expose unfold() so the agent's adaptive_unfold can use it.
    def unfold(self, query: str, k: int = None):
        return self._inject_recalls(query, k=k)

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids=None, storage_text: Optional[str] = None):
        super().record_tool(tool_name, args, observation, docids=docids, storage_text=storage_text)
        nid = self.active[-1] if self.active else None
        if nid:
            self.storage_ids.append(nid)
            self._dirty = True
        # Use tool observation as a retrieval cue (cheap but effective for evidence tasks).
        q = (observation or "")[:600]
        self._inject_recalls(q)
        self._apply_window()
        self.maybe_fold()

    def record_msg(self, text: str):
        super().record_msg(text)
        nid = self.active[-1] if self.active else None
        if nid:
            self.storage_ids.append(nid)
            self._dirty = True
        self._inject_recalls((text or "")[:600])
        self._apply_window()
        self.maybe_fold()


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
    _storage_segments: List[Tuple[List[str], TextRetriever]] = field(default_factory=list)
    _storage_buffer: List[str] = field(default_factory=list)
    _storage_buffer_retriever: Optional[TextRetriever] = None
    _storage_indexed: Set[str] = field(default_factory=set)

    # Incremental / segmented storage indexing (LSM-style)
    storage_segment_size: int = 120          # buffer size before creating a new segment index
    storage_max_segments: int = 12           # max segment indices before compacting into a cold archive
    storage_hot_segments: int = 3            # keep these most-recent segments "hot" (richer indexing)

    # Folding policy (connectivity / cut-minimization)
    fold_window_max: int = 40               # consider this many oldest ACTIVE nodes as fold candidates
    fold_max_chunk: int = 10                # (soft) cap for folded nodes per fold op (min-cut may exceed; we post-process)

    # Token-weighted min-cut folding (default): choose a subset that (a) removes enough tokens
    # while (b) minimizing cross edges (cut) to the remaining ACTIVE context.
    fold_method: str = "mincut"             # "mincut" or "greedy" (fallback)
    fold_edge_w_depends: float = 1.0
    fold_edge_w_docref: float = 1.0
    fold_edge_w_seq: float = 0.2
    fold_mincut_lambda_init: float = 1e-3
    fold_mincut_lambda_max: float = 1e6
    fold_mincut_iters: int = 22

    # Folding policy selector.
    # - "budget": token-weighted min-cut folding driven by budget_active (default).
    # - "pef_url": Page-Episode Folding (PEF) for BrowserGym/WebArena: fold primarily on URL changes.
    fold_policy: str = "budget"

    # PEF(URL) state (internal)
    _pef_last_url: Optional[str] = field(default=None, init=False, repr=False)
    _pef_episode_start_pos: int = field(default=0, init=False, repr=False)

    # Safety backstop for PEF: if ACTIVE blows up beyond this multiplier, fall back to budget folding.
    pef_backstop_mult: float = 2.5

    # PEF rolling compaction (within the same URL):
    # Rather than folding on every budget hit, we use hysteresis.
    # - Fold only when ACTIVE tokens exceed budget_active * pef_hi_mult.
    # - Fold down until <= budget_active * pef_lo_mult.
    # The fold itself collapses the *older* part of the current URL episode into an EPISODE_PROXY,
    # keeping the most recent pef_roll_keep_last nodes in full detail.
    pef_hi_mult: float = 1.25
    pef_lo_mult: float = 0.85
    pef_roll_keep_last: int = 10
    pef_roll_min_chunk: int = 6

    # Greedy folding weights (kept as fallback / debugging)
    fold_w_internal: float = 1.0            # reward edges within the chunk
    fold_w_cut: float = 1.4                 # penalize edges from chunk to remaining ACTIVE

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
        self._storage_segments = []
        self._storage_buffer = []
        self._storage_buffer_retriever = None
        self._storage_indexed = set()
        self.docid_to_nodes = {}
        self.edges_out = {"depends": {}, "doc_ref": {}, "seq": {}}
        self.edges_in = {"depends": {}, "doc_ref": {}, "seq": {}}
        self._last_in_thread = {}
        # PEF(URL) state
        self._pef_last_url = None
        self._pef_episode_start_pos = 0

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

    def add_node(self, thread: str, kind: str, text: str, docids: Optional[List[str]] = None, storage_text: Optional[str] = None) -> str:
        nid = super().add_node(thread=thread, kind=kind, text=text, docids=docids, storage_text=storage_text)
        # graph updates
        self._link_sequential(nid)
        self._index_docids(nid)
        return nid

    # ---- storage index ----
    def _index_text(self, n: MemoryNode, *, include_full: bool) -> str:
        """Text representation used for storage indexing.

        - include_full=True: index clipped head+tail of full content when available.
        - include_full=False: index only compact text ("cold" archive).
        """
        if not n:
            return ""
        if (not include_full) or (not n.storage_text):
            return n.text
        full = n.storage_text
        if len(full) > 8000:
            full = full[:4000] + "\n...\n" + full[-4000:]
        return n.text + "\n\n[FULL_CONTENT]\n" + full

    def _build_storage_segment(self, nids: List[str], *, include_full: bool) -> Optional[TextRetriever]:
        items: List[TextItem] = []
        for nid in nids:
            n = self.nodes.get(nid)
            if not n:
                continue
            items.append(TextItem(id=nid, text=self._index_text(n, include_full=include_full), meta={"url": f"mem://{nid}", "title": n.kind}))
        return build_retriever(self.storage_retriever_kind, items, faiss_dim=self.storage_faiss_dim) if items else None

    def _refresh_storage_retriever(self):
        segs = [r for (_ids, r) in (self._storage_segments or []) if r is not None]
        if self._storage_buffer_retriever is not None:
            segs = segs + [self._storage_buffer_retriever]
        self._storage_retriever = RRFMultiRetriever(segs) if segs else None

    def _flush_storage_buffer(self, *, force: bool = False):
        if not self._storage_buffer:
            return
        if (not force) and len(self._storage_buffer) < int(self.storage_segment_size):
            return
        nids = list(self._storage_buffer)
        self._storage_buffer = []
        self._storage_buffer_retriever = None
        retr = self._build_storage_segment(nids, include_full=True)
        if retr is not None:
            self._storage_segments.append((nids, retr))

    def _compact_storage_segments(self):
        """Compact old segments into a single cold archive segment.

        This prevents unbounded index rebuilds while keeping recent segments richer.
        """
        max_segs = int(self.storage_max_segments)
        hot = max(1, int(self.storage_hot_segments))
        if len(self._storage_segments) <= max_segs:
            return
        # Keep last `hot` segments as-is; compact the rest.
        keep_tail = self._storage_segments[-hot:]
        to_merge = self._storage_segments[:-hot]
        merge_ids: List[str] = []
        for ids, _r in to_merge:
            merge_ids.extend(ids)
        # Build a compressed cold index (no full content).
        cold = self._build_storage_segment(merge_ids, include_full=False) if merge_ids else None
        new_segments: List[Tuple[List[str], TextRetriever]] = []
        if cold is not None:
            new_segments.append((merge_ids, cold))
        new_segments.extend(keep_tail)
        self._storage_segments = new_segments

    def _ensure_storage_index(self, *, force_flush: bool = False):
        # Ensure the newest stored nodes are searchable immediately:
        #  - if the buffer isn't large enough to flush to a permanent segment,
        #    keep a small "buffer retriever" over the buffer.
        #  - periodically flush to segments and compact older segments.
        if force_flush:
            self._flush_storage_buffer(force=True)
        else:
            self._flush_storage_buffer(force=False)
            if self._storage_buffer:
                self._storage_buffer_retriever = self._build_storage_segment(list(self._storage_buffer), include_full=True)
            else:
                self._storage_buffer_retriever = None

        self._compact_storage_segments()
        self._refresh_storage_retriever()

    def _storage_index_add(self, nids: List[str], *, force_flush: bool = False):
        """Incrementally add storage nodes into the segmented index."""
        if not nids:
            return
        for nid in nids:
            if nid in self._storage_indexed:
                continue
            if nid not in self.nodes:
                continue
            self._storage_indexed.add(nid)
            self._storage_buffer.append(nid)
        self._ensure_storage_index(force_flush=force_flush)

    def _rebuild_storage_index(self):
        """Full rebuild (fallback / debug)."""
        self._storage_segments = []
        self._storage_buffer = []
        self._storage_indexed = set()
        self._storage_index_add(list(self.storage), force_flush=True)

    # ---- overrides to track branch nodes ----
    def branch(self, description: str, prompt: str) -> str:
        bid = super().branch(description, prompt)
        self.branch_stack[-1] = (bid, [n for n in self.active if self.nodes[n].thread == bid])
        return bid

    def record_tool(self, tool_name: str, args: Dict[str, Any], observation: str, docids=None, storage_text: Optional[str] = None):
        """Record a tool event.

        IMPORTANT: GoCMemory overrides the base implementation so we can support
        Page-Episode Folding (PEF) for BrowserGym/WebArena without triggering
        budget-based folding on every tool call.
        """
        self._decay_ttl()
        nid = self.add_node(
            thread=self.current_thread,
            kind="tool",
            text=f"[TOOL:{tool_name}] args={args}\nobs={observation}",
            docids=docids or [],
            storage_text=storage_text,
        )
        self._on_branch_step(nid)

        if self.branch_stack:
            bid, lst = self.branch_stack[-1]
            lst.append(nid)
            self.branch_stack[-1] = (bid, lst)

        # PEF(URL): fold primarily on URL changes, triggered by observations.
        if str(self.fold_policy).lower() == "pef_url" and str(tool_name).lower() == "obs":
            try:
                self._pef_maybe_fold_on_obs(nid)
            except Exception:
                pass

        # Always apply the policy dispatcher (PEF backstop or budget folding).
        self.maybe_fold()

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

        # Incrementally index newly stored nodes.
        self._storage_index_add(list(to_remove), force_flush=False)
        self.maybe_fold()


    # ---- PEF (Page-Episode Folding) for BrowserGym/WebArena ----
    _URL_IN_TEXT_RE = re.compile(r"\burl=([^\s\)]+)")

    def _pef_extract_url(self, nid: str) -> Optional[str]:
        n = self.nodes.get(nid)
        if not n:
            return None
        # Prefer explicit docids (browsergym_runner uses docids=[url] for obs nodes).
        for d in (n.docids or []):
            if isinstance(d, str) and d.startswith("http"):
                return d
        # Fallback: parse from text header.
        m = self._URL_IN_TEXT_RE.search(n.text or "")
        return m.group(1) if m else None

    def _pef_create_episode_proxy_node(self, chunk: List[str], url: str) -> str:
        """Create a compact proxy summary for a single page episode.

        This is intentionally lightweight and deterministic (no LLM summarization),
        tuned for BrowserGym/WebArena traces.
        """
        if not chunk:
            raise ValueError("empty chunk")

        child_steps = [self.nodes[nid].step_idx for nid in chunk if nid in self.nodes]
        step_min = min(child_steps) if child_steps else 0
        step_max = max(child_steps) if child_steps else step_min

        # Extract and compress actions.
        actions: List[str] = []
        act_re1 = re.compile(r"'action'\s*:\s*'([^']+)'")
        act_re2 = re.compile(r"\"action\"\s*:\s*\"([^\"]+)\"")
        last_obs_head: Optional[str] = None
        for nid in chunk:
            n = self.nodes.get(nid)
            if not n:
                continue
            txt = n.text or ""
            if txt.startswith("[TOOL:act]"):
                m = act_re1.search(txt) or act_re2.search(txt)
                if m:
                    actions.append(m.group(1).strip())
            elif txt.startswith("[TOOL:obs]"):
                # Keep a tiny hint of the latest observation for this episode.
                head = txt.splitlines()
                if head:
                    last_obs_head = head[0][:220]

        # Run-length encode identical consecutive actions (common for noop/scroll).
        comp: List[str] = []
        if actions:
            prev = actions[0]
            cnt = 1
            for a in actions[1:]:
                if a == prev:
                    cnt += 1
                    continue
                comp.append(f"{prev} x{cnt}" if cnt > 1 else prev)
                prev, cnt = a, 1
            comp.append(f"{prev} x{cnt}" if cnt > 1 else prev)

        action_line = "; ".join(comp[:12])
        if len(comp) > 12:
            action_line += "; ..."

        header = f"[EPISODE_PROXY url={url} steps={step_min}-{step_max} n={len(chunk)}]"
        body_lines: List[str] = []
        if action_line:
            body_lines.append("actions: " + action_line)
        if last_obs_head:
            body_lines.append("last_obs: " + last_obs_head)

        proxy_text = header + ("\n" + "\n".join(body_lines) if body_lines else "")

        # Create node WITHOUT sequential linking (use base add_node), then override step_idx.
        proxy_id = MemoryManagerBase.add_node(self, thread="main", kind="summary", text=proxy_text, docids=[url])
        if proxy_id in self.nodes:
            self.nodes[proxy_id].children = list(chunk)
            self.nodes[proxy_id].step_idx = step_min
            self.nodes[proxy_id].token_len = approx_token_count(self.nodes[proxy_id].text)

        # Graph hooks: docid index + depends edges so unfolding proxy can bring children.
        self._index_docids(proxy_id)
        for child in chunk:
            if child in self.nodes:
                self.nodes[child].parent = proxy_id
                self._add_edge("depends", proxy_id, child)
        return proxy_id

    def _pef_maybe_fold_on_obs(self, obs_nid: str):
        """Fold the previous page episode when the URL changes."""
        url = self._pef_extract_url(obs_nid)
        if not url:
            return

        # First observation establishes the current episode boundary.
        if self._pef_last_url is None:
            self._pef_last_url = url
            try:
                self._pef_episode_start_pos = self.active.index(obs_nid)
            except ValueError:
                self._pef_episode_start_pos = max(0, len(self.active) - 1)
            return

        if url == self._pef_last_url:
            return

        prev_url = self._pef_last_url
        try:
            obs_idx = self.active.index(obs_nid)
        except ValueError:
            obs_idx = len(self.active) - 1

        start = int(self._pef_episode_start_pos)
        start = max(0, min(start, obs_idx))

        # Fold all nodes belonging to the previous episode (between episode start and this obs).
        chunk = [nid for nid in self.active[start:obs_idx] if (nid in self.nodes and not self._is_anchor_node(nid))]
        if chunk:
            before = self.active_tokens()
            proxy_id = self._pef_create_episode_proxy_node(chunk, prev_url)
            self._replace_prefix_with_proxy(chunk, proxy_id)
            newly_stored: List[str] = []
            for nid in chunk:
                if nid in self.nodes and nid not in self.storage:
                    self.storage.append(nid)
                    newly_stored.append(nid)
            if newly_stored:
                self._storage_index_add(newly_stored, force_flush=False)
            after = self.active_tokens()

            # Trace
            self._emit_event({
                "type": "pef_fold",
                "mem": "GoC",
                "global_step": int(self._global_step),
                "prev_url": prev_url,
                "new_url": url,
                "episode_nodes": len(chunk),
                "episode_tokens_before": int(before),
                "episode_tokens_after": int(after),
                "proxy_id": proxy_id,
            })

        # Start the new episode at the current observation.
        self._pef_last_url = url
        try:
            self._pef_episode_start_pos = self.active.index(obs_nid)
        except ValueError:
            self._pef_episode_start_pos = max(0, len(self.active) - 1)


    def _pef_roll_fold_current_episode(self) -> bool:
        """Compact the *older* portion of the current URL episode into an EPISODE_PROXY.

        Returns True if a fold occurred.

        Motivation
        ----------
        On WebArena tasks, observations are large and can push ACTIVE over budget rapidly,
        causing frequent budget-driven folds. In PEF mode we prefer to keep the *current*
        page episode mostly intact, but we can safely compact early steps on the same URL
        while preserving the latest interaction context.
        """
        # Identify the current episode URL.
        cur_url = self._pef_last_url
        if not cur_url:
            # Best-effort fallback: infer from last ACTIVE obs node.
            for nid in reversed(self.active):
                u = self._pef_extract_url(nid)
                if u:
                    cur_url = u
                    break
        if not cur_url:
            return False

        start = int(self._pef_episode_start_pos)
        start = max(0, min(start, len(self.active)))

        # Keep the most recent nodes in full detail.
        keep_last = max(0, int(self.pef_roll_keep_last))
        end_exclusive = max(start, len(self.active) - keep_last)
        if end_exclusive <= start:
            return False

        # Exclude anchors from folding.
        window = [nid for nid in self.active[start:end_exclusive] if nid in self.nodes]
        chunk = [nid for nid in window if not self._is_anchor_node(nid)]
        if len(chunk) < int(self.pef_roll_min_chunk):
            return False

        before = self.active_tokens()
        proxy_id = self._pef_create_episode_proxy_node(chunk, cur_url)
        self._replace_prefix_with_proxy(chunk, proxy_id)

        newly_stored: List[str] = []
        for nid in chunk:
            if nid in self.nodes and nid not in self.storage:
                self.storage.append(nid)
                newly_stored.append(nid)
        if newly_stored:
            self._storage_index_add(newly_stored, force_flush=False)
        after = self.active_tokens()

        # Trace
        self._emit_event({
            "type": "pef_roll_fold",
            "mem": "GoC",
            "global_step": int(self._global_step),
            "url": cur_url,
            "episode_start_pos": int(self._pef_episode_start_pos),
            "rolled_nodes": int(len(chunk)),
            "episode_tokens_before": int(before),
            "episode_tokens_after": int(after),
            "proxy_id": proxy_id,
        })

        # Episode start becomes the proxy position (since we compacted the prefix).
        try:
            self._pef_episode_start_pos = self.active.index(proxy_id)
        except ValueError:
            self._pef_episode_start_pos = max(0, len(self.active) - 1)
        return True

    
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
        """Replace `chunk` nodes in ACTIVE with a proxy node.

        Unlike range-based folding, GoC folding may pick a *non-contiguous* chunk.
        We insert the proxy at the earliest position of any removed node.
        """
        if not chunk:
            return

        chunk_set = set(chunk)
        positions = [i for i, nid in enumerate(self.active) if nid in chunk_set]
        if not positions:
            return
        ins = min(positions)
        # Remove chunk nodes
        self.active = [nid for nid in self.active if nid not in chunk_set]
        # Insert proxy once
        if proxy_id not in self.active:
            self.active.insert(ins, proxy_id)

    def _is_anchor_node(self, nid: str) -> bool:
        """Anchor nodes should remain in ACTIVE_CONTEXT as long as possible.

        We treat failure traces and deterministic constraints as anchors.
        """
        n = self.nodes.get(nid)
        if not n or n.kind != "summary":
            return False
        t = (n.text or "").lstrip()
        # Be permissive: different runners may emit "[FAIL]" or "[FAILURE]".
        # We anchor any summary beginning with "[FAIL" as well as hard constraints.
        return t.startswith("[FAIL") or t.startswith("[CONSTRAINT]") or t.startswith("[GUARD]")

    def _active_neighbors(self, nid: str, active_set: Set[str], etypes: Tuple[str, ...] = ("depends", "doc_ref")) -> Set[str]:
        """Undirected neighbors within ACTIVE for fold heuristics."""
        out: Set[str] = set()
        for et in etypes:
            out |= set(self.edges_out.get(et, {}).get(nid, set()))
            out |= set(self.edges_in.get(et, {}).get(nid, set()))
        return out & active_set

    # ---- token-weighted min-cut folding ----
    def _cut_edge_weight(self, u: str, v: str) -> float:
        """Undirected edge weight between two nodes, aggregating edge types.

        We purposefully treat edges as *undirected* for folding: fold should group
        tightly connected nodes regardless of direction.
        """
        w = 0.0
        # depends
        if (v in self.edges_out.get("depends", {}).get(u, set())) or (u in self.edges_out.get("depends", {}).get(v, set())):
            w += float(self.fold_edge_w_depends)
        # doc_ref
        if (v in self.edges_out.get("doc_ref", {}).get(u, set())) or (u in self.edges_out.get("doc_ref", {}).get(v, set())):
            w += float(self.fold_edge_w_docref)
        # seq
        if (v in self.edges_out.get("seq", {}).get(u, set())) or (u in self.edges_out.get("seq", {}).get(v, set())):
            w += float(self.fold_edge_w_seq)
        return w

    def _active_cut_weight_to_outside(self, nid: str, candidates_set: Set[str], active_set: Set[str]) -> float:
        """Total undirected edge weight from `nid` to ACTIVE nodes outside candidates."""
        w = 0.0
        for et, ew in (
            ("depends", float(self.fold_edge_w_depends)),
            ("doc_ref", float(self.fold_edge_w_docref)),
            ("seq", float(self.fold_edge_w_seq)),
        ):
            outs = set(self.edges_out.get(et, {}).get(nid, set()))
            ins = set(self.edges_in.get(et, {}).get(nid, set()))
            nbrs = (outs | ins) & active_set
            outside = nbrs - candidates_set
            if outside:
                w += ew * float(len(outside))
        return w

    def _select_fold_chunk_mincut(
        self,
        candidates: List[str],
        *,
        active_set: Set[str],
        target_remove: int,
        seed: str,
    ) -> List[str]:
        """Select a fold chunk using token-weighted s-t min-cut.

        We solve a family of min-cut problems parameterized by `lambda`:

            minimize  cut(S)  +  sum_{v in T} lambda * token(v)

        where S is the SOURCE side (folded), T is the SINK side (kept).
        This encourages moving token-heavy nodes into S while paying the cut edges.

        We then search lambda so that token(S) >= target_remove.
        """
        candidates_set = set(candidates)
        if seed not in candidates_set:
            return []

        # Special nodes
        SRC = "__SRC__"
        SNK = "__SNK__"
        KEEP = "__KEEP__"  # represents all ACTIVE nodes outside `candidates`

        INF = 1e12

        # Precompute pair weights among candidates (sparse) and outside-cut weights.
        pair_w: Dict[Tuple[str, str], float] = {}
        # Use edges_out keys to avoid O(n^2).
        for u in candidates:
            for et in ("depends", "doc_ref", "seq"):
                for v in self.edges_out.get(et, {}).get(u, set()):
                    if v not in candidates_set or v == u:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    pair_w[(a, b)] = pair_w.get((a, b), 0.0) + float(
                        self.fold_edge_w_depends if et == "depends" else self.fold_edge_w_docref if et == "doc_ref" else self.fold_edge_w_seq
                    )
            # Include inbound edges too (treat as undirected).
            for et in ("depends", "doc_ref", "seq"):
                for v in self.edges_in.get(et, {}).get(u, set()):
                    if v not in candidates_set or v == u:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    pair_w[(a, b)] = pair_w.get((a, b), 0.0) + float(
                        self.fold_edge_w_depends if et == "depends" else self.fold_edge_w_docref if et == "doc_ref" else self.fold_edge_w_seq
                    )

        outside_w: Dict[str, float] = {}
        for u in candidates:
            outside_w[u] = self._active_cut_weight_to_outside(u, candidates_set, active_set)

        # Build and solve min-cut for a given lambda.
        def solve(lam: float) -> Tuple[List[str], float, int]:
            lam = float(max(0.0, lam))
            G = nx.DiGraph()
            G.add_node(SRC)
            G.add_node(SNK)
            G.add_node(KEEP)
            # Force KEEP to be on sink side.
            G.add_edge(KEEP, SNK, capacity=INF)

            # Candidate-candidate symmetric edges.
            for (a, b), w in pair_w.items():
                if w <= 0:
                    continue
                G.add_edge(a, b, capacity=w)
                G.add_edge(b, a, capacity=w)

            # Candidate-KEEP edges representing cut to outside ACTIVE.
            for u, w in outside_w.items():
                if w <= 0:
                    continue
                G.add_edge(u, KEEP, capacity=w)
                G.add_edge(KEEP, u, capacity=w)

            # Unary token penalty for leaving node on sink side.
            for u in candidates:
                if u == seed:
                    continue
                tok = float(self.nodes[u].token_len) if u in self.nodes else 0.0
                # If u ends up in sink side, edge SRC->u is cut and paid.
                cap = lam * tok
                if cap > 0:
                    G.add_edge(SRC, u, capacity=cap)
                else:
                    # Ensure node exists in graph.
                    G.add_node(u)

            # Force seed to SOURCE side.
            G.add_edge(SRC, seed, capacity=INF)

            try:
                cut_value, (S, T) = nx.minimum_cut(G, SRC, SNK, capacity="capacity")
            except Exception:
                return ([], float("inf"), 0)

            chunk = [u for u in S if (u in candidates_set)]
            tok_sum = sum(int(self.nodes[u].token_len) for u in chunk if u in self.nodes)
            return (chunk, float(cut_value), int(tok_sum))

        # Quick feasibility: if even max lambda can't hit target, take all candidates.
        lam_lo = 0.0
        lam_hi = float(self.fold_mincut_lambda_init)
        best_over: Optional[Tuple[List[str], float, int, float]] = None  # (chunk, cut, tok, lam)

        # Escalate lambda until we satisfy target or hit max.
        for _ in range(40):
            chunk, cutv, tok = solve(lam_hi)
            if tok >= target_remove and chunk:
                best_over = (chunk, cutv, tok, lam_hi)
                break
            lam_hi *= 2.0
            if lam_hi > float(self.fold_mincut_lambda_max):
                break

        if best_over is None:
            # Not enough removable tokens in candidates; fold all candidates (still avoids anchors).
            return list(candidates)

        # Binary search for the smallest lambda achieving target_remove.
        lam_best = best_over[3]
        chunk_best, cut_best, tok_best = best_over[0], best_over[1], best_over[2]
        for _ in range(int(self.fold_mincut_iters)):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            chunk, cutv, tok = solve(lam_mid)
            if not chunk:
                lam_lo = lam_mid
                continue
            if tok >= target_remove:
                lam_hi = lam_mid
                lam_best, chunk_best, cut_best, tok_best = lam_mid, chunk, cutv, tok
            else:
                lam_lo = lam_mid

        # Local refinement: try a few lambdas around the boundary and keep the lowest cut.
        lam_grid = sorted({lam_best, lam_best * 0.7, lam_best * 1.4, lam_hi, lam_lo})
        best = (chunk_best, cut_best, tok_best)
        for lam in lam_grid:
            chunk, cutv, tok = solve(lam)
            if chunk and tok >= target_remove:
                if cutv < best[1] or (abs(cutv - best[1]) < 1e-9 and tok < best[2]):
                    best = (chunk, cutv, tok)

        return list(best[0])

    def _prune_oldest_non_summary_to_storage(self):
        """As a last resort, prune the oldest node while preserving anchors."""
        if len(self.active) <= 1:
            return
        # protect the most recent node
        prefix = self.active[:-1]

        # 1) Prefer pruning non-summary, non-anchor nodes.
        idx = None
        for i, nid in enumerate(prefix):
            n = self.nodes.get(nid)
            if not n or self._is_anchor_node(nid):
                continue
            if n.kind != "summary":
                idx = i
                break

        # 2) Else prune the oldest non-anchor summary.
        if idx is None:
            for i, nid in enumerate(prefix):
                if nid in self.nodes and (not self._is_anchor_node(nid)):
                    idx = i
                    break

        # 3) If everything is an anchor, prune the oldest anchor as a last-last resort.
        if idx is None:
            idx = 0
        nid = self.active.pop(idx)
        if nid in self.nodes and nid not in self.storage:
            self.storage.append(nid)
            self._storage_index_add([nid], force_flush=False)

    # ---- folding: keep removed nodes in storage for recovery ----
    def _maybe_fold_budget(self):
        """Budget-driven folding (token-weighted min-cut by default).

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
        newly_stored: List[str] = []
        while self.active_tokens() > self.budget_active and len(self.active) > 1 and safety < 50:
            safety += 1
            overflow = self.active_tokens() - self.budget_active
            # Keep the most recent node to preserve immediate context.
            prefix = self.active[:-1]
            if not prefix:
                break

            # Target: remove at least overflow tokens, plus a small buffer.
            target_remove = max(overflow + 50, int(self.budget_active * 0.25))
            max_chunk = int(self.fold_max_chunk or 10)

            # Candidate window: oldest nodes only (avoid folding the latest context).
            window = [nid for nid in prefix if nid in self.nodes][: max(1, int(self.fold_window_max))]
            candidates = [nid for nid in window if not self._is_anchor_node(nid)]

            if not candidates:
                # Nothing foldable in the window; prune (will preserve anchors if possible).
                self._prune_oldest_non_summary_to_storage()
                continue

            active_set = set(self.active)
            seed = candidates[0]

            # Select chunk: token-weighted min-cut (default) or greedy fallback.
            chunk: List[str] = []
            if str(self.fold_method).lower() == "mincut":
                chunk = self._select_fold_chunk_mincut(
                    candidates,
                    active_set=active_set,
                    target_remove=int(target_remove),
                    seed=seed,
                )
                # Keep deterministic ordering based on ACTIVE chronology.
                chunk_set = set(chunk)
                chunk = [nid for nid in candidates if nid in chunk_set]

                # Soft cap: if min-cut returns a very large set, shrink it while trying
                # to retain connectivity and satisfy the token target.
                hard_max = max(20, int(max_chunk) * 4)
                if len(chunk) > hard_max:
                    # Rank by internal connectivity (within chunk) then token mass.
                    ch_set = set(chunk)
                    def _internal_deg(u: str) -> int:
                        return len(self._active_neighbors(u, ch_set, ("depends", "doc_ref", "seq")))

                    keep: List[str] = [seed] if seed in ch_set else [chunk[0]]
                    kept_set = set(keep)
                    removed = sum(int(self.nodes[u].token_len) for u in keep if u in self.nodes)
                    rest = [u for u in chunk if u not in kept_set]
                    rest.sort(key=lambda u: (_internal_deg(u), int(self.nodes[u].token_len) if u in self.nodes else 0), reverse=True)
                    for u in rest:
                        if len(keep) >= hard_max and removed >= target_remove:
                            break
                        keep.append(u)
                        kept_set.add(u)
                        removed += int(self.nodes[u].token_len) if u in self.nodes else 0
                        if len(keep) >= hard_max and removed >= target_remove:
                            break
                    chunk = keep

            else:
                pos = {nid: i for i, nid in enumerate(candidates)}
                neigh = {nid: self._active_neighbors(nid, active_set) for nid in candidates}

                # Seed with the oldest candidate.
                chunk = [seed]
                chunk_set = set(chunk)
                removed = int(self.nodes[seed].token_len)

                # Greedy expansion: prefer high internal connectivity + low cut to remaining ACTIVE.
                while removed < target_remove and len(chunk) < max_chunk:
                    best_v: Optional[str] = None
                    best_score = -1e18
                    best_tok = -1

                    for v in candidates:
                        if v in chunk_set:
                            continue
                        nbr = neigh.get(v, set())
                        internal = len(nbr & chunk_set)
                        cut = len(nbr - chunk_set)
                        # Slight bias towards older nodes (smaller pos) to keep chronology stable.
                        age_bonus = 0.05 * (1.0 - (pos.get(v, 0) / max(1.0, float(len(candidates) - 1))))
                        score = float(self.fold_w_internal) * float(internal) - float(self.fold_w_cut) * float(cut) + age_bonus
                        tok = int(self.nodes[v].token_len)
                        if (score > best_score) or (abs(score - best_score) < 1e-9 and tok > best_tok):
                            best_score = score
                            best_tok = tok
                            best_v = v

                    if best_v is None:
                        break
                    chunk.append(best_v)
                    chunk_set.add(best_v)
                    removed += int(self.nodes[best_v].token_len)

            removed = sum(int(self.nodes[u].token_len) for u in chunk if u in self.nodes)

            # Compute removed token mass (used for gating / logging).
            removed = sum(int(self.nodes[nid].token_len) for nid in chunk if nid in self.nodes)

            # Allow single-node folding only if it meaningfully helps.
            if len(chunk) == 1:
                n0 = self.nodes.get(chunk[0])
                if not n0 or (n0.token_len < int(self.budget_active * 0.40) and removed < target_remove):
                    chunk = []

            if chunk:
                before = self.active_tokens()
                # ----- trace fold decision (optional) -----
                try:
                    chunk_set = set(chunk)
                    outside_set = set(self.active) - chunk_set
                    boundary_pairs: Set[frozenset] = set()
                    internal_pairs: Set[frozenset] = set()
                    for et in ("depends", "doc_ref", "seq"):
                        out_map = self.edges_out.get(et, {})
                        in_map = self.edges_in.get(et, {})
                        for u in chunk_set:
                            neigh = set(out_map.get(u, set())) | set(in_map.get(u, set()))
                            for v in neigh:
                                if v not in self.nodes:
                                    continue
                                if v not in set(self.active):
                                    continue
                                if v == u:
                                    continue
                                pair = frozenset((u, v))
                                if len(pair) != 2:
                                    continue
                                if v in chunk_set:
                                    internal_pairs.add(pair)
                                else:
                                    boundary_pairs.add(pair)

                    self._emit_event({
                        "type": "fold",
                        "mem": "GoC",
                        "global_step": int(self._global_step),
                        "budget_active": int(self.budget_active),
                        "overflow_tokens": int(overflow),
                        "target_remove_tokens": int(target_remove),
                        "chunk_size": len(chunk),
                        "chunk_tokens": int(removed),
                        "cut_edges": len(boundary_pairs),
                        "internal_edges": len(internal_pairs),
                        "chunk_node_ids": chunk[:50],
                    })
                except Exception:
                    pass
                proxy_id = self._create_proxy_node(chunk)
                # Remove folded nodes from active and keep proxy active
                self._replace_prefix_with_proxy(chunk, proxy_id)
                # Move children to storage (lossless)
                for nid in chunk:
                    if nid in self.nodes and nid not in self.storage:
                        self.storage.append(nid)
                        newly_stored.append(nid)
                after = self.active_tokens()
                # If folding didn't reduce anything (should be rare), prune one node to avoid infinite loops.
                if after >= before:
                    self._prune_oldest_non_summary_to_storage()
                continue

            # Fallback pruning: keep summaries as anchors
            self._prune_oldest_non_summary_to_storage()

        # Incrementally index any newly stored nodes (buffer retriever keeps them searchable immediately).
        if newly_stored:
            self._storage_index_add(newly_stored, force_flush=False)
        else:
            self._ensure_storage_index(force_flush=False)

    def maybe_fold(self):
        """Dispatch folding according to the configured policy."""
        pol = str(self.fold_policy).lower().strip()
        if pol == "pef_url":
            # PEF folds are primarily triggered on obs() when URL changes.
            # Within a single URL episode, we prefer rolling compaction with hysteresis
            # to avoid folding on every budget edge.
            try:
                hi = float(self.budget_active) * float(self.pef_hi_mult)
                lo = float(self.budget_active) * float(self.pef_lo_mult)
                # Roll-fold only when we cross the high watermark, then fold down to low.
                if self.active_tokens() > hi:
                    # Fold at most a few times per call to avoid pathological loops.
                    for _ in range(6):
                        if self.active_tokens() <= lo:
                            break
                        did = self._pef_roll_fold_current_episode()
                        if not did:
                            break
                # Absolute safety backstop: if ACTIVE is truly runaway, fall back to budget folding.
                if self.active_tokens() > float(self.budget_active) * float(self.pef_backstop_mult):
                    self._maybe_fold_budget()
            except Exception:
                # If anything goes wrong, fall back to budget folding.
                self._maybe_fold_budget()
            return

        # Default: budget-driven folding.
        self._maybe_fold_budget()

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

    def _extract_storage_snippet(self, full: str, query: str, max_chars: int = 900) -> str:
        """Return a compact excerpt from full text that is likely relevant to the query.

        This enables 'lossless storage + selective recovery': store the full open_page content
        out-of-budget, then unfold only small relevant snippets back into ACTIVE_CONTEXT.
        """
        if not full:
            return ""
        q = (query or "").strip()
        # Prefer a few stable keywords for late-binding tasks.
        prefer = ["relocation_note", "relocation_year", "relocated_to", "relocation"]
        terms: List[str] = [t for t in prefer if t in q.lower()]
        if not terms:
            toks = re.findall(r"[A-Za-z_]{4,}", q)
            # de-dup, keep order
            seen: Set[str] = set()
            for t in toks:
                tl = t.lower()
                if tl in seen:
                    continue
                seen.add(tl)
                terms.append(tl)
                if len(terms) >= 6:
                    break

        hay = full
        hay_low = hay.lower()
        idx = -1
        for t in terms:
            i = hay_low.find(t)
            if i >= 0:
                idx = i
                break
        # Fallback: look for the field name directly
        if idx < 0:
            idx = hay_low.find("relocation_note")
        if idx < 0:
            idx = max(0, len(hay) - max_chars)

        # Window around match
        win = max_chars // 2
        start = max(0, idx - win)
        end = min(len(hay), idx + win)
        snip = hay[start:end]

        # Try to align to line boundaries for readability
        s_nl = snip.rfind("\n", 0, min(200, len(snip)))
        if s_nl > 0:
            snip = snip[s_nl + 1 :]
        e_nl = snip.find("\n", max(0, len(snip) - 200))
        if e_nl > 0:
            snip = snip[:e_nl]

        if start > 0:
            snip = "..." + snip
        if end < len(hay):
            snip = snip + "..."
        return snip[:max_chars].strip()

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
        # Activate nodes and, for lossless nodes, also materialize a small relevant snippet.
        remaining = int(self.budget_unfold)
        for nid in sorted(chosen, key=lambda x: self.nodes[x].step_idx):
            if nid not in self.nodes:
                continue
            n = self.nodes[nid]
            if nid not in self.active:
                cost = n.token_len
                if remaining - cost < 0:
                    continue
                remaining -= cost
                n.ttl = self.ttl_unfold
                self.active.append(nid)
                activated.append(nid)

            # If we have full stored content, recover an excerpt likely to contain the late-bound fact.
            if n.storage_text:
                snip = self._extract_storage_snippet(n.storage_text, query=query, max_chars=900)
                if snip:
                    sn_text = f"[UNFOLD_SNIPPET from {nid}]\n{snip}"
                    sn_cost = approx_token_count(sn_text)
                    if remaining - sn_cost >= 0:
                        sn_id = self.add_node(thread="main", kind="summary", text=sn_text, docids=list(n.docids))
                        self.nodes[sn_id].ttl = self.ttl_unfold
                        self.active.append(sn_id)
                        activated.append(sn_id)
                        remaining -= sn_cost

        # Re-order active nodes chronologically by step index so recovered nodes
        # do not end up at the very end of the context.
        self.active.sort(key=lambda x: self.nodes[x].step_idx)
        return activated


@dataclass
class SimilarityOnlyMemory(GoCMemory):
    """RAG-style baseline: unfold purely by retrieval similarity (no dependency closure).

    This isolates the contribution of GoC's *dependency closure* logic.
    We still keep lossless storage and the same folding mechanism.
    """

    def unfold(self, query: str, k: int = None):
        if k is None:
            k = self.unfold_k
        if not self._storage_retriever or not self.storage:
            return []

        hits = self._storage_retriever.search(query, topk=max(k * 5, 10))

        chosen: List[str] = []
        remaining = int(self.budget_unfold)
        for nid, _score in hits:
            if nid not in self.nodes:
                continue
            if nid in self.active:
                continue
            cost = int(self.nodes[nid].token_len)
            if remaining - cost < 0:
                continue
            chosen.append(nid)
            remaining -= cost
            if len(chosen) >= int(k):
                break

        activated: List[str] = []
        # Activate in chronological order.
        for nid in sorted(chosen, key=lambda x: self.nodes[x].step_idx):
            n = self.nodes[nid]
            if nid not in self.active:
                n.ttl = self.ttl_unfold
                self.active.append(nid)
                activated.append(nid)

            # Optional snippet materialization (kept consistent with GoC).
            if n.storage_text:
                snip = self._extract_storage_snippet(n.storage_text, query=query, max_chars=900)
                if snip:
                    sn_text = f"[UNFOLD_SNIPPET from {nid}]\n{snip}"
                    sn_cost = approx_token_count(sn_text)
                    if remaining - sn_cost >= 0:
                        sn_id = self.add_node(thread="main", kind="summary", text=sn_text, docids=list(n.docids))
                        self.nodes[sn_id].ttl = self.ttl_unfold
                        self.active.append(sn_id)
                        activated.append(sn_id)
                        remaining -= sn_cost

        self.active.sort(key=lambda x: self.nodes[x].step_idx)
        return activated
