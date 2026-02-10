from __future__ import annotations

from typing import Any, Dict, List, Tuple


def graph_frontier_candidates(
    world: Any,
    seed_clause_ids: List[str],
    *,
    max_hops: int = 2,
    max_nodes: int = 50,
) -> Tuple[List[str], Dict[str, int]]:
    """Expand clause frontier from seed ids over dependency-style edges.

    This is intentionally benchmark-agnostic as long as `world` exposes:
    - world.clauses: Dict[str, clause_obj]
    - clause_obj.targets: Dict[str, List[str]] (optional)
    - clause_obj.terms_used: List[str] (optional)
    - world.meta['term_definitions']: Dict[str, str] (optional)
    """
    max_hops = max(0, int(max_hops or 0))
    max_nodes = max(0, int(max_nodes or 0))
    if max_hops <= 0 or max_nodes <= 0:
        return [], {}

    clauses = getattr(world, "clauses", None)
    if not isinstance(clauses, dict) or not clauses:
        return [], {}

    term_defs: Dict[str, Any] = {}
    try:
        meta = getattr(world, "meta", {}) or {}
        if isinstance(meta, dict):
            td = meta.get("term_definitions", {}) or {}
            if isinstance(td, dict):
                term_defs = td
    except Exception:
        term_defs = {}

    def _neighbors(cid: str) -> List[str]:
        clause = clauses.get(cid)
        if clause is None:
            return []
        nids: List[str] = []

        targets = getattr(clause, "targets", None)
        if isinstance(targets, dict):
            for _, vals in targets.items():
                if not vals:
                    continue
                for v in vals:
                    vid = str(v).strip()
                    if vid:
                        nids.append(vid)

        for term in getattr(clause, "terms_used", []) or []:
            tid = str(term).strip()
            if not tid:
                continue
            def_cid = term_defs.get(tid)
            if def_cid:
                nids.append(str(def_cid).strip())
        return nids

    seeds = [str(cid).strip() for cid in seed_clause_ids if cid]
    seen_seed: set[str] = set()
    seeds_u: List[str] = []
    for cid in seeds:
        if cid and cid in clauses and cid not in seen_seed:
            seeds_u.append(cid)
            seen_seed.add(cid)
    if not seeds_u:
        return [], {}

    visited: set[str] = set(seeds_u)
    dist: Dict[str, int] = {cid: 0 for cid in seeds_u}
    frontier: List[str] = list(seeds_u)
    out: List[str] = []

    for hop in range(1, max_hops + 1):
        if not frontier or len(out) >= max_nodes:
            break
        next_frontier: List[str] = []
        for cid in frontier:
            for nid in _neighbors(cid):
                if not nid or nid in visited or nid not in clauses:
                    continue
                visited.add(nid)
                dist[nid] = hop
                next_frontier.append(nid)
                out.append(nid)
                if len(out) >= max_nodes:
                    break
            if len(out) >= max_nodes:
                break
        frontier = next_frontier

    return out, dist


def merge_ranked_candidates(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two ranked result lists keyed by `clause_id`, using max(score)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for item in primary or []:
        cid = item.get("clause_id")
        if not cid:
            continue
        merged[cid] = dict(item)

    for item in secondary or []:
        cid = item.get("clause_id")
        if not cid:
            continue
        if cid not in merged:
            merged[cid] = dict(item)
            continue
        try:
            merged[cid]["score"] = max(
                float(merged[cid].get("score", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
            )
        except Exception:
            pass
        for k, v in item.items():
            if k in {"frontier_distance", "source"} and k not in merged[cid]:
                merged[cid][k] = v

    return sorted(
        merged.values(),
        key=lambda it: float(it.get("score", 0.0) or 0.0),
        reverse=True,
    )

