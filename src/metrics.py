from __future__ import annotations
from typing import Dict, Any, List
import re

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()

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
