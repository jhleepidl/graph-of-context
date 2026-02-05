from __future__ import annotations

import re
from typing import Dict, List


_REGION_MAP = {
    "EU": ["eu", "european union", "europe", "eea"],
    "US": ["us", "usa", "united states", "u.s."],
    "UK": ["uk", "united kingdom", "u.k."],
    "KR": ["kr", "korea", "south korea"],
    "GLOBAL": ["global", "worldwide"],
}

_PURPOSE_MAP = {
    "debugging": ["debug", "debugging", "troubleshoot", "troubleshooting"],
    "security": ["security", "fraud", "abuse"],
    "analytics": ["analytics", "analysis", "insights", "metrics"],
    "research": ["research", "study"],
    "support": ["support", "customer support", "helpdesk"],
}

_DATA_TYPE_MAP = {
    "logs": ["logs", "log data"],
    "telemetry": ["telemetry"],
    "health": ["health", "phi", "health data", "medical"],
    "identifiers": ["identifier", "identifiers", "user id", "account id"],
    "pii": ["pii", "personal data", "personal information"],
    "financial": ["financial", "payment", "billing"],
}

_PRODUCTS = ["alpha", "beta"]
_TIERS = ["free", "pro", "enterprise"]
_ACTIONS = ["share", "export", "store", "retain", "delete", "transfer"]


def _match_terms(text: str, terms: List[str]) -> bool:
    for term in terms:
        if re.search(rf"\\b{re.escape(term)}\\b", text):
            return True
    return False


def _extract_from_map(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    hits: List[str] = []
    for key, terms in mapping.items():
        if _match_terms(text, terms):
            hits.append(key)
    return hits


def extract_facets(ticket_text: str) -> Dict[str, List[str]]:
    text = ticket_text.lower()
    facets: Dict[str, List[str]] = {
        "region": _extract_from_map(text, _REGION_MAP),
        "purpose": _extract_from_map(text, _PURPOSE_MAP),
        "data_type": _extract_from_map(text, _DATA_TYPE_MAP),
        "product": [p for p in _PRODUCTS if re.search(rf"\\b{re.escape(p)}\\b", text)],
        "tier": [t for t in _TIERS if re.search(rf"\\b{re.escape(t)}\\b", text)],
        "action": [a for a in _ACTIONS if re.search(rf"\\b{re.escape(a)}\\b", text)],
    }
    return facets
