from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_toks = p.split()
    g_toks = g.split()
    common = {}
    for t in p_toks:
        common[t] = 0
    for t in g_toks:
        if t in common:
            common[t] += 1
    # Count overlap with multiplicities
    overlap = 0
    g_counts = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    p_counts = {}
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in set(p_toks) & set(g_toks):
        overlap += min(p_counts.get(t, 0), g_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_toks)
    rec = overlap / len(g_toks)
    return 2 * prec * rec / max(1e-9, (prec + rec))


def robust_qa_match(pred: str, gold: str, *, f1_threshold: float = 0.8) -> Tuple[bool, Dict[str, Any]]:
    """Return (correct, info) for open-QA style answers."""
    pn = normalize_text(pred)
    gn = normalize_text(gold)
    em = (pn == gn) and bool(gn)
    f1 = token_f1(pred, gold)
    # A common lenient criterion for short answers: gold is substring of pred.
    substr = (gn in pn) if (gn and pn) else (pn == gn)
    correct = bool(em or substr or (f1 >= float(f1_threshold)))
    return correct, {"pred_norm": pn, "gold_norm": gn, "em": em, "f1": f1, "substr": substr}


def parse_structured_answer(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Parse a structured answer from finish.answer.

    Accepts:
      - JSON object string
      - 'a1: ...\na2: ...' / 'A1=...; A2=...' variants
    Returns (parsed, strict_json_ok).
    """
    t = (text or "").strip()
    if not t:
        return None, False
    # Try JSON first
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass

    # Defensive: some runtimes accidentally serialize dicts with single-quotes (python repr).
    # Try to recover via ast.literal_eval.
    try:
        import ast
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return obj, False
    except Exception:
        pass

    # Line/kv fallback
    out: Dict[str, Any] = {}
    # Split by newline or ';'
    parts = re.split(r"[\n;]+", t)
    for p in parts:
        if not p.strip():
            continue
        m = re.match(r"\s*([a-zA-Z][a-zA-Z0-9_\-]*)\s*[:=]\s*(.+)\s*$", p)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        out[k] = v

    return (out if out else None), False


def norm_title(s: str) -> str:
    # Wikipedia titles should match pretty strictly, but be forgiving about underscores.
    return normalize_text((s or "").replace("_", " "))


def title_match(pred_title: str, gold_title: str) -> bool:
    return norm_title(pred_title) == norm_title(gold_title)


def set_f1(pred_set: List[str], gold_set: List[str]) -> float:
    p = {norm_title(x) for x in (pred_set or []) if x}
    g = {norm_title(x) for x in (gold_set or []) if x}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / max(1e-9, (prec + rec))


def extract_list_field(obj: Dict[str, Any], key: str) -> List[str]:
    v = obj.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    # allow comma-separated
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(v)]
