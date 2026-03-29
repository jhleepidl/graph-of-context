from __future__ import annotations
from typing import Dict, Any, List, Optional
import re

# -------------------------
# Answer matching utilities
# -------------------------
# Many tasks in this repo ask for answers exactly as:
#   "<ProjectName> | <Headquarters>"
# where the canonical forms are like Project_0047 and City_16.
#
# In practice, LLM outputs often contain extra text, different zero-padding,
# or slightly different separators. For benchmarking, we want a robust
# *semantic* match while keeping a strict signal for formatting compliance.

PAIR_RE = re.compile(
    r"(Project[ _]?0*(\d+))\s*\|\s*(City[ _]?0*(\d+))",
    flags=re.IGNORECASE,
)
PROJECT_RE = re.compile(r"\bProject[ _]?0*(\d+)\b", flags=re.IGNORECASE)
CITY_RE = re.compile(r"\bCity[ _]?0*(\d+)\b", flags=re.IGNORECASE)
PROSE_PAIR_PATTERNS = [
    re.compile(
        r"\b(Project[ _]?0*(\d+))\b.{0,140}?\b(?:current\s+operating\s+city|operating\s+city|approved_operating_city|headquarters)\b[^\n]{0,30}?\b(City[ _]?0*(\d+))\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(Project[ _]?0*(\d+))\b.{0,120}?\b(?:answer|therefore|thus|selected|winner|final)\b[^\n]{0,60}?\b(City[ _]?0*(\d+))\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


def _canonical_pair(project_num: int, city_num: int) -> str:
    return f"Project_{int(project_num):04d} | City_{int(city_num)}"


def normalize_project_city_pair(text: str) -> Optional[str]:
    """Extract and normalize 'Project_xxxx | City_y' from a string.

    Returns canonical string like 'Project_0047 | City_16', or None.
    Robust to light prose such as
    'Project_0047 ... current operating city is City_16'.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = PAIR_RE.search(t)
    if m:
        return _canonical_pair(int(m.group(2)), int(m.group(4)))

    for pat in PROSE_PAIR_PATTERNS:
        m2 = pat.search(t)
        if m2:
            return _canonical_pair(int(m2.group(2)), int(m2.group(4)))

    projects = PROJECT_RE.findall(t)
    cities = CITY_RE.findall(t)
    if len(projects) == 1 and len(cities) == 1:
        return _canonical_pair(int(projects[0]), int(cities[0]))
    return None


def parsed_pair_match(pred: str, gold: str) -> Optional[bool]:
    """Return True/False if both sides can be parsed as a Project|City pair.

    Returns None if parsing fails on either side.
    """
    p = normalize_project_city_pair(pred)
    g = normalize_project_city_pair(gold)
    if p is None or g is None:
        return None
    return p == g


def robust_match(pred: str, gold: str) -> bool:
    """Benchmark-facing match function.

    - If both answers can be parsed as the canonical Project|City pair, compare
      the normalized forms.
    - Otherwise fall back to (trimmed) exact match.

    This reduces false negatives from formatting noise while keeping behavior
    stable for other task types.
    """
    pm = parsed_pair_match(pred, gold)
    if pm is not None:
        return bool(pm)
    return exact_match(pred, gold)

def exact_match(pred: str, gold: str) -> bool:
    """Slightly-robust exact match.

    - Trims whitespace.
    - If both look like a pipe-delimited pair (common in this benchmark), compare
      normalized segments around the first '|'.
    """
    p = (pred or "").strip()
    g = (gold or "").strip()
    if not p or not g:
        return p == g

    if "|" in p and "|" in g:
        p1, p2 = [x.strip() for x in p.split("|", 1)]
        g1, g2 = [x.strip() for x in g.split("|", 1)]
        return (p1 == g1) and (p2 == g2)

    return p == g

DOCID_RE = re.compile(r"D_[A-Z]+_[0-9_]+")

def extract_docids(text: str) -> List[str]:
    return sorted(set(DOCID_RE.findall(text or "")))

def docid_coverage(expl: str, gold_docids: List[str]) -> float:
    pred = set(extract_docids(expl))
    gold = set(gold_docids)
    if not gold:
        return 0.0
    return len(pred & gold) / len(gold)

def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    acc = sum(1 for r in rows if r["correct"]) / len(rows)
    avg_total_tok = sum(r["metrics"]["llm_in_tokens"] + r["metrics"]["llm_out_tokens"] for r in rows) / len(rows)
    avg_peak = sum(r["metrics"]["peak_active_tokens"] for r in rows) / len(rows)
    avg_tools = sum(r["metrics"]["tool_calls"] for r in rows) / len(rows)
    avg_cov = sum(r.get("docid_cov", 0.0) for r in rows) / len(rows)
    return {
        "n": len(rows),
        "accuracy": acc,
        "avg_total_tokens": avg_total_tok,
        "avg_peak_active_tokens": avg_peak,
        "avg_tool_calls": avg_tools,
        "avg_docid_coverage": avg_cov
    }
