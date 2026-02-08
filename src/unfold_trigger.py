from __future__ import annotations

from typing import List, Tuple
import re


class UnfoldTrigger:
    """Lightweight heuristic for deciding when to call memory.unfold()."""

    _ALNUM_RE = re.compile(r"[A-Za-z0-9_]+")
    _DOCID_RE = re.compile(r"\bD_[A-Za-z0-9_]+\b")
    _FIELD_KEY_RE = re.compile(r"\b(?:TITLE|URL):[^\n,;|]+", re.IGNORECASE)

    def __init__(
        self,
        *,
        missing_terms_threshold: int = 3,
        min_token_len: int = 4,
        max_keywords: int = 48,
        required_key_pattern: str = r"\brelocation_[A-Za-z0-9_]+\b",
        always_trigger_on_required_keys: bool = True,
    ) -> None:
        self.missing_terms_threshold = max(1, int(missing_terms_threshold))
        self.min_token_len = max(1, int(min_token_len))
        self.max_keywords = max(1, int(max_keywords))
        self.required_key_re = re.compile(required_key_pattern, re.IGNORECASE)
        self.always_trigger_on_required_keys = bool(always_trigger_on_required_keys)

    def extract_keywords(self, text: str) -> List[str]:
        """Extract simple lexical keywords from query text."""
        src = str(text or "")
        terms: List[str] = []
        seen = set()

        def _add(term: str) -> None:
            t = re.sub(r"\s+", " ", str(term or "").strip())
            if not t:
                return
            key = t.lower()
            if key in seen:
                return
            seen.add(key)
            terms.append(t)

        # Structured terms first (docids and TITLE:/URL: keys).
        for m in self._DOCID_RE.finditer(src):
            _add(m.group(0))
        for m in self._FIELD_KEY_RE.finditer(src):
            _add(m.group(0))

        # Generic alnum tokens.
        for tok in self._ALNUM_RE.findall(src):
            if len(tok) < self.min_token_len:
                continue
            _add(tok)

        return terms[: self.max_keywords]

    def should_unfold(
        self,
        next_query: str,
        active_text: str,
    ) -> Tuple[bool, str, List[str]]:
        """Return (should_unfold, reason, missing_terms)."""
        query = str(next_query or "")
        hay = str(active_text or "")
        hay_low = hay.lower()
        missing_terms: List[str] = []

        for term in self.extract_keywords(query):
            if term.lower() not in hay_low:
                missing_terms.append(term)

        required_keys = [m.group(0) for m in self.required_key_re.finditer(query)]
        if self.always_trigger_on_required_keys and required_keys:
            # Ensure required keys appear in missing_terms summary when absent.
            for key in required_keys:
                if key.lower() not in hay_low and key not in missing_terms:
                    missing_terms.append(key)
            return True, "required_key", missing_terms

        if len(missing_terms) >= int(self.missing_terms_threshold):
            return True, "missing_terms", missing_terms

        return False, "covered", missing_terms
