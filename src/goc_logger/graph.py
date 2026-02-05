from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GoCGraph:
    goc_graph_version: str = "v0"
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step: int = 0
    _proxy_counter: int = 0

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        existing = self.nodes.get(node_id, {})
        merged = {"id": node_id, "type": node_type}
        merged.update(existing)
        merged.update({k: v for k, v in attrs.items() if v is not None and existing.get(k) in (None, "")})
        self.nodes[node_id] = merged

    def add_edge(self, edge_id: str, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        self.edges[edge_id] = {
            "id": edge_id,
            "src": src,
            "dst": dst,
            "type": edge_type,
            **attrs,
        }

    def next_proxy_id(self, task_id: str) -> str:
        self._proxy_counter += 1
        return f"proxy:{task_id}:{self._proxy_counter}"

    def fold(
        self,
        fold_id: str,
        source_node_ids: Iterable[str],
        proxy_node_id: str,
        proxy_style: str,
        quote_spans: Optional[List[str]] = None,
        reason: Optional[str] = None,
        controller_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.add_node(proxy_node_id, "proxy", style=proxy_style, active=True)
        for node_id in source_node_ids:
            self.add_edge(f"refers_to:{proxy_node_id}:{node_id}", proxy_node_id, node_id, "refers_to")
            if node_id in self.nodes:
                self.nodes[node_id]["active"] = False
        return {
            "fold_id": fold_id,
            "source_node_ids": list(source_node_ids),
            "proxy_node_id": proxy_node_id,
            "proxy_style": proxy_style,
            "quote_spans": quote_spans or [],
            "reason": reason,
            "controller_info": controller_info or {},
        }

    def unfold(
        self,
        target_node_ids: Iterable[str],
        reason: Optional[str] = None,
        controller_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        for node_id in target_node_ids:
            if node_id in self.nodes:
                self.nodes[node_id]["active"] = True
        return {
            "target_node_ids": list(target_node_ids),
            "reason": reason,
            "controller_info": controller_info or {},
        }

    def materialize_prompt(
        self,
        prompt_id: str,
        included_node_ids: Iterable[str],
        included_edge_ids: Optional[Iterable[str]] = None,
        budget_info: Optional[Dict[str, Any]] = None,
        fold_level: Optional[int] = None,
        controller_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "prompt_id": prompt_id,
            "included_node_ids": list(included_node_ids),
            "included_edge_ids": list(included_edge_ids) if included_edge_ids else [],
            "budget_info": budget_info or {},
            "fold_level": fold_level,
            "controller_info": controller_info or {},
        }

    def record_prediction(
        self,
        decision: str,
        conditions: List[str],
        evidence_before_pad: List[str],
        evidence_after_pad: List[str],
        parse_error: Optional[str] = None,
        raw_path: Optional[str] = None,
        prompt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "decision": decision,
            "conditions": conditions,
            "evidence_before_pad": evidence_before_pad,
            "evidence_after_pad": evidence_after_pad,
            "parse_error": parse_error,
            "raw_path": raw_path,
            "prompt_path": prompt_path,
        }

    def record_gold(self, gold_decision: str, gold_evidence_ids: List[str]) -> Dict[str, Any]:
        return {
            "gold_decision": gold_decision,
            "gold_evidence_ids": gold_evidence_ids,
        }

    def to_snapshot_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.goc_graph_version,
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values()),
        }
