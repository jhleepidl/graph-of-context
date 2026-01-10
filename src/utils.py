import re
from typing import List


def approx_token_count(text: str) -> int:
    """Approximate token count with a very rough heuristic."""
    return max(1, int(len(text) / 4))


_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")
_CODENAME_RE = re.compile(r"^codename_([a-z0-9])([a-z0-9]+)?$")


def tokenize(text: str) -> List[str]:
    """Tokenize into lowercase terms.

    v21 IR tweak:
      - SyntheticBrowseComp two_phase tasks search for a *prefix* like `codename_h`.
      - Documents store full code names like `Codename_HMMUO`, which tokenizes to `codename_hmmuo`.
      - Add a lightweight derived token `codename_<first_char>` for any `codename_*` token so BM25
        can match prefix queries without needing n-grams.

    This is intentionally conservative (adds at most 1 extra token per codename occurrence).
    """
    base = [t.lower() for t in _WORD_RE.findall(text or "")]

    extra: List[str] = []
    for t in base:
        m = _CODENAME_RE.match(t)
        if m:
            init = m.group(1)
            extra.append(f"codename_{init}")

    return base + extra


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
