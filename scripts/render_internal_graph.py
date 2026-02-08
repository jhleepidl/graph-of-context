#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EDGE_COLORS: Dict[str, str] = {
    "depends": "#1f77b4",
    "depends_llm": "#17becf",
    "doc_ref": "#2ca02c",
    "seq": "#7f7f7f",
}


def _extract_snapshot(obj: Any) -> Dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("nodes"), dict) and isinstance(obj.get("edges"), dict):
        return obj
    payload = obj.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), dict) and isinstance(payload.get("edges"), dict):
        return payload
    snap = obj.get("snapshot")
    if isinstance(snap, dict) and isinstance(snap.get("nodes"), dict) and isinstance(snap.get("edges"), dict):
        return snap
    return None


def load_final_snapshot(path: Path) -> Dict[str, Any]:
    last: Dict[str, Any] | None = None
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            snap = _extract_snapshot(obj)
            if snap is not None:
                last = snap
    if last is None:
        raise RuntimeError(f"No snapshot object found in: {path}")
    return last


def _iter_nodes(snapshot: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    nodes = snapshot.get("nodes") or {}
    if isinstance(nodes, dict):
        for nid, attrs in nodes.items():
            if isinstance(attrs, dict):
                yield str(nid), attrs


def _iter_edges(snapshot: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    edges = snapshot.get("edges") or {}
    if not isinstance(edges, dict):
        return
    for etype, adj in edges.items():
        if not isinstance(adj, dict):
            continue
        for u, vs in adj.items():
            if not isinstance(vs, list):
                continue
            for v in vs:
                yield str(etype), str(u), str(v)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_dot(snapshot: Dict[str, Any], out_path: Path) -> None:
    active = set(str(x) for x in (snapshot.get("active") or []))
    lines: List[str] = [
        "digraph GoCInternal {",
        '  rankdir="LR";',
        '  graph [fontname="Helvetica"];',
        '  node [shape="ellipse", style="filled", fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    for nid, attrs in _iter_nodes(snapshot):
        kind = str(attrs.get("kind") or "")
        step_idx = attrs.get("step_idx")
        preview = str(attrs.get("text_preview") or "")
        label = f"{nid}\\n{kind}"
        if step_idx is not None:
            label += f"\\nstep={step_idx}"
        if preview:
            label += f"\\n{preview[:80]}"
        is_active = nid in active
        is_proxy = "proxy_depth" in attrs
        fill = "#ffef9f" if is_active else "#dce7f7"
        if is_proxy:
            fill = "#f7d9b6" if not is_active else "#f4b183"
        shape = "box" if is_proxy else "ellipse"
        lines.append(
            f'  "{_esc(nid)}" [label="{_esc(label)}", shape="{shape}", fillcolor="{fill}"];'
        )

    for etype, u, v in _iter_edges(snapshot):
        color = EDGE_COLORS.get(etype, "#333333")
        lines.append(
            f'  "{_esc(u)}" -> "{_esc(v)}" [label="{_esc(etype)}", color="{color}"];'
        )

    lines.append("}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_pyvis(snapshot: Dict[str, Any], out_path: Path) -> bool:
    try:
        from pyvis.network import Network
    except Exception:
        return False

    active = set(str(x) for x in (snapshot.get("active") or []))
    net = Network(height="880px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    net.barnes_hut()

    for nid, attrs in _iter_nodes(snapshot):
        is_active = nid in active
        is_proxy = "proxy_depth" in attrs
        color = "#ffcc66" if is_active else "#9ecbff"
        if is_proxy:
            color = "#f4a261" if not is_active else "#e76f51"
        shape = "box" if is_proxy else "dot"
        title_obj = {
            "id": nid,
            "kind": attrs.get("kind"),
            "thread": attrs.get("thread"),
            "step_idx": attrs.get("step_idx"),
            "token_len": attrs.get("token_len"),
            "docids": attrs.get("docids"),
            "ttl": attrs.get("ttl"),
            "proxy_depth": attrs.get("proxy_depth"),
            "children": attrs.get("children"),
            "parent": attrs.get("parent"),
            "text_preview": attrs.get("text_preview"),
        }
        net.add_node(
            nid,
            label=nid,
            title=json.dumps(title_obj, ensure_ascii=False, indent=2),
            shape=shape,
            color=color,
        )

    for etype, u, v in _iter_edges(snapshot):
        net.add_edge(u, v, color=EDGE_COLORS.get(etype, "#333333"), label=etype, title=etype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GoC internal snapshots to DOT (and optional pyvis HTML).")
    ap.add_argument("snapshots_jsonl", type=Path, help="Path to internal snapshots JSONL.")
    ap.add_argument("--dot_out", type=Path, default=None, help="Output DOT path (default: <input>.dot).")
    ap.add_argument("--html_out", type=Path, default=None, help="Output HTML path (default: <input>.html).")
    args = ap.parse_args()

    src = args.snapshots_jsonl
    if not src.exists():
        raise SystemExit(f"Input does not exist: {src}")

    dot_out = args.dot_out or src.with_suffix(".dot")
    html_out = args.html_out or src.with_suffix(".html")

    snap = load_final_snapshot(src)
    render_dot(snap, dot_out)
    print(f"Wrote DOT: {dot_out}")

    if render_pyvis(snap, html_out):
        print(f"Wrote HTML: {html_out}")
    else:
        print("pyvis not available; skipped HTML output.")


if __name__ == "__main__":
    main()
