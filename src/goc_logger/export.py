from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .graph import GoCGraph


def _load_events(path: Path, task_id: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("task_id") == task_id:
                events.append(data)
    events.sort(key=lambda item: item.get("step", 0))
    return events


def _apply_event(graph: GoCGraph, event: Dict[str, Any]) -> None:
    etype = event.get("event_type")
    payload = event.get("payload", {})
    if etype == "INIT":
        graph.add_node(payload.get("episode_id", ""), "episode")
        graph.add_node(payload.get("ticket_id", ""), "ticket")
    elif etype == "SEARCH":
        qid = payload.get("query_id")
        query = payload.get("query")
        if qid:
            graph.add_node(qid, "query", text=query)
        for item in payload.get("results", []):
            doc_id = item.get("doc_id") or f"doc:{item.get('clause_id')}"
            graph.add_node(
                doc_id,
                "doc_ref",
                clause_id=item.get("clause_id"),
                kind=item.get("kind"),
                slot=item.get("slot"),
                published_at=item.get("published_at"),
                authority=item.get("authority"),
                first_seen_rank=item.get("rank"),
                first_seen_score=item.get("score"),
            )
            edge_id = f"retrieved:{qid}:{doc_id}"
            graph.add_edge(
                edge_id,
                qid,
                doc_id,
                "retrieved",
                rank=item.get("rank"),
                score=item.get("score"),
                source=item.get("source"),
            )
    elif etype == "OPEN":
        clause_id = payload.get("clause_id")
        if clause_id:
            graph.add_edge(
                f"opened:{event.get('task_id')}:{clause_id}",
                f"episode:{event.get('task_id')}",
                f"doc:{clause_id}",
                "opened",
                reason=payload.get("reason"),
            )
    elif etype == "PROMPT":
        prompt_id = payload.get("prompt_id")
        if prompt_id:
            graph.add_node(prompt_id, "prompt")
            for nid in payload.get("included_node_ids", []):
                graph.add_edge(
                    f"selected:{prompt_id}:{nid}",
                    f"episode:{event.get('task_id')}",
                    nid,
                    "selected_for_prompt",
                    prompt_id=prompt_id,
                )
    elif etype == "PREDICTION":
        answer_id = f"answer:{event.get('task_id')}"
        graph.add_node(answer_id, "answer", decision=payload.get("decision"))
        for cid in payload.get("evidence_after_pad", []) or payload.get("evidence", []):
            graph.add_edge(
                f"cites:{event.get('task_id')}:{cid}",
                answer_id,
                f"doc:{cid}",
                "cites_evidence",
            )
    elif etype == "SNAPSHOT":
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        for node in nodes:
            graph.nodes[node.get("id")] = node
        for edge in edges:
            graph.edges[edge.get("id")] = edge


def _resolve_doc_meta(node: Dict[str, Any]) -> Dict[str, Any]:
    if not node:
        return {"kind": "unknown", "slot": "unknown", "published_at": "unknown"}
    kind = node.get("kind")
    slot = node.get("slot")
    published_at = node.get("published_at")
    attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
    if not kind:
        kind = attrs.get("kind")
    if not slot:
        slot = attrs.get("slot")
    if not published_at:
        published_at = attrs.get("published_at")
    return {
        "kind": kind or "unknown",
        "slot": slot or "unknown",
        "published_at": published_at or "unknown",
    }


def export_dot(in_path: Path, task_id: str, out_path: Path, step: int | None = None) -> None:
    events = _load_events(in_path, task_id)
    graph = GoCGraph()
    if step is None:
        snapshot = next((e for e in reversed(events) if e.get("event_type") == "SNAPSHOT"), None)
        if snapshot:
            _apply_event(graph, snapshot)
        else:
            for event in events:
                _apply_event(graph, event)
    else:
        for event in events:
            if event.get("step", 0) > step:
                break
            _apply_event(graph, event)

    lines: List[str] = ["digraph G {"]
    for node in graph.nodes.values():
        ntype = node.get("type")
        if ntype == "doc_ref":
            meta = _resolve_doc_meta(node)
            clause_id = node.get("clause_id") or node.get("id")
            label = f"{clause_id}\\n{meta['kind']}/{meta['slot']}\\n{meta['published_at']}"
            rank = node.get("first_seen_rank") or node.get("rank")
            score = node.get("first_seen_score") or node.get("score")
            if rank is not None or score is not None:
                label += f"\\nrank:{rank} score:{score}"
        elif ntype == "answer":
            label = f"answer\\n{node.get('decision')}"
        elif ntype == "query":
            label = f"query\\n{node.get('id')}"
            parts = str(node.get("id", "")).split(":")
            if len(parts) >= 3:
                label += f"\\n{parts[-1]}"
        else:
            label = f"{ntype}\\n{node.get('id')}"
        lines.append(f"  \"{node.get('id')}\" [label=\"{label}\"];")
    for edge in graph.edges.values():
        label = edge.get("type", "")
        if edge.get("reason"):
            label += f" ({edge.get('reason')})"
        lines.append(f"  \"{edge.get('src')}\" -> \"{edge.get('dst')}\" [label=\"{label}\"];")
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GoC Graph exporter")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    export_dot(Path(args.in_path), args.task_id, Path(args.out_path), step=args.step)


if __name__ == "__main__":
    main()
