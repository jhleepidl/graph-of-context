"""BrowserGym / WebChoreArena integration runner.

Why this exists
--------------
Our GoC work targets *long-horizon* LLM agents where naive full-history prompts
become too expensive and/or brittle (lost-in-the-middle, drift).

WebChoreArena is a good real-world stress test because it explicitly includes
"Long-Term Memory" and "Massive Memory" task types, and it is built on top of
the WebArena simulated websites. The WebChoreArena repo also warns that running
the full suite can cost *hundreds of dollars*, and provides a small-set list for
cheaper iteration.

This runner is **optional** and kept dependency-light: it only imports
`gymnasium` / `browsergym` when you execute it.

High-level design
-----------------
- We treat each environment step as a tool-like transition:
  - record the agent action as a "tool" (tool_name="act")
  - record the next observation as a "tool" (tool_name="obs")
- Observations can be large. We store a truncated version in active context and
  (optionally) store the full text as `storage_text` so GoC can unfold if needed.

Limitations
-----------
BrowserGym benchmarks differ in action/observation format. For WebArena-like
benchmarks, actions are typically strings (e.g., "click(...)", "type(...)").
This runner follows that pattern and is meant as a practical starting point.
If your BrowserGym install uses a different action format, adjust
`_step_env(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import json
import os
import re
import time
import tempfile
import hashlib
import collections
import sys
import shlex
import urllib.parse


DEFAULT_BROWSERGYM_SYSTEM_PROMPT = """You are an expert web-browsing agent operating inside a simulated website environment (BrowserGym/WebArena/WebChoreArena).

Your job is to complete the user's task by choosing the next browser action.

Core rules (follow strictly):
- You MUST output exactly ONE line: ACTION: <action>
- Actions are strings that look like function calls (e.g., click('a46'), fill('213','text'), scroll(0, 600), goto('http://...')).
- Prefer id-based actions using element ids (bids) shown in the observation. Do NOT invent element ids.
- Do NOT invent or hallucinate URLs. Use goto(...) only for the provided Start URL / allowed site URLs, or when the observation explicitly contains the URL.
- noop() is ONLY for waiting after an action that triggers a page update (navigation/click/fill/press). Never do noop() twice in a row.
- If you get stuck (observation does not change after a few steps), change strategy: go to Start URL, click a relevant link/button, use search, or open the reviews section.

Task success recipe (especially for Shopping review-distribution tasks):
1) Go to the product page (Start URL Lite) if provided.
2) Find and open the reviews/ratings section (often “Reviews”, “Customer Reviews”, star breakdown).
3) Extract the 1–5 star distribution.
4) When you have the final distribution, respond using send_msg_to_user(...) exactly in the required format.

Never output analysis or explanations. Only output the next ACTION line."""


DEFAULT_BROWSERGYM_ACTION_GUIDELINES = """Use BrowserGym action primitives (actions are **strings** that look like function calls).

Critical navigation/safety rules:
- Do NOT hallucinate or invent URLs. Prefer clicking element ids shown in the observation.
- Avoid clicking footer/legal pages that derail tasks: Privacy, Cookie, Terms, Policy, GDPR, Consent.
  If you land on such a page, use go_back().
- For "read/lookup" tasks (e.g., "provide the distribution of reviews"), do NOT click "Add/Write/Submit Review".

Noop rule:
- noop() is ONLY for waiting 1 step after an action. Never do noop() twice in a row.

Navigation rule (important):
- If the instruction includes a "Start URL:", you should **immediately** navigate to it with goto('<Start URL>') unless you are already on that exact URL.

Element id rule:
- Element ids may appear as bids like 'a46', or as plain numeric ids in fields like "browsergym_id": "213".
  Use the id exactly as shown in the observation, e.g. click('a46') or click('213').

Prefer **id-based** actions that reference element ids shown in the observation.

Bid primitives (preferred):
- fill('<bid>', '<text>')
- click('<bid>'[, '<button>'])            # button optional: 'left'/'middle'/'right'
- dblclick('<bid>'[, '<button>'])
- hover('<bid>')
- press('<bid>', '<key_comb>')            # e.g. 'Enter', 'Control+A', 'Shift+Tab'
- focus('<bid>')
- clear('<bid>')
- select_option('<bid>', ['<opt1>', '<opt2>'])    # one or multiple options
- drag_and_drop('<from_bid>', '<to_bid>')

Navigation / tabs:
- goto('<url>'), go_back(), go_forward()
- new_tab(), tab_close(), tab_focus(<index>)

Scrolling / waiting:
- scroll(<dx>, <dy>)                      # dy>0 scroll down; dy<0 scroll up (e.g. scroll(0, 600))
- noop()                                  # wait / do nothing

Stuck rule:
- If you repeat the same scroll/noop and the observation does not change, stop repeating and try a different strategy (goto(Start URL), click a link/button, use search, etc.).

Messaging (ONLY if the task explicitly asks you to respond in chat; most WebArena/WebChoreArena tasks do NOT):
- send_msg_to_user('<text>')

Other available actions exist (coordinate mouse_* and keyboard_*), but avoid them unless bid-based actions cannot work.

Safety / formatting:
- Do NOT use the unsafe `python` action.
- Use quotes for string arguments (single quotes are fine).
- Output **exactly one line**: ACTION: <one action>.

There is no explicit STOP action in BrowserGym. Keep acting until the environment terminates automatically when the evaluator detects task completion.
If you need to give the page time to update, use noop()."""


# Pages/links that frequently lead to dead-ends for WebArena/WebChoreArena tasks.
_BAD_PAGE_RE = re.compile(r"privacy|cookie|consent|terms|policy|gdpr", re.IGNORECASE)


def _is_review_distribution_task(instruction: str) -> bool:
    if not isinstance(instruction, str):
        return False
    s = instruction.lower()
    return (
        ("distribution" in s and "review" in s)
        or ("review" in s and "stars" in s)
        or ("rating" in s and "distribution" in s)
    )


def _stable_state_sig(active_obs: str, *, max_chars: int = 8000) -> Optional[str]:
    """Compute a stable-ish state signature to reduce false 'changes'.

    We normalize digits and whitespace so small numeric drifts (timestamps, counters)
    don't break loop detection.
    """
    if not isinstance(active_obs, str):
        return None
    txt = active_obs[:max_chars].lower()
    txt = re.sub(r"https?://\S+", "<url>", txt)
    txt = re.sub(r"\d+", "#", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    try:
        return hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()
    except Exception:
        return None



def _canonicalize_url_for_sig(url: Optional[str]) -> Optional[str]:
    """Normalize URL for state signatures (remove fragment/query, keep origin+path)."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parts = urllib.parse.urlsplit(url.strip())
        # Drop query+fragment for stability; keep scheme/netloc/path.
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return url.strip()


def _state_sig_from_url_and_bids(url: Optional[str], bids: List[str]) -> Optional[str]:
    """Compute a compact state signature based on canonical URL + visible actionable ids."""
    u = _canonicalize_url_for_sig(url) or ""
    bb = [b for b in bids if isinstance(b, str) and b.strip()]
    # Keep order but cap length for stability.
    bb = bb[:80]
    payload = u + "||" + "|".join(bb)
    if not payload.strip("|"):
        return None
    try:
        return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()
    except Exception:
        return None


def _compute_state_sig(url: Optional[str], meta: Dict[str, Any], active_obs: Optional[str] = None) -> Optional[str]:
    """Compute a state signature robust to noisy observation text.

    Priority:
      1) canonical URL + actionable element ids (bids)
      2) fallback to _stable_state_sig(active_obs)
    """
    bids: List[str] = []
    try:
        elems = meta.get("actionable_elements")
        if isinstance(elems, list):
            for ln in elems:
                if not isinstance(ln, str):
                    continue
                m = _ACTIONABLE_LINE_RE.match(ln)
                if m:
                    bids.append(m.group(1))
    except Exception:
        bids = []

    sig = _state_sig_from_url_and_bids(url, bids)
    if sig:
        # Mix in a normalized content signature so SPA content changes are detected even if URL+bids stay the same.
        if isinstance(active_obs, str):
            cs = _stable_state_sig(active_obs, max_chars=2000)
            if cs:
                try:
                    return hashlib.sha1((sig + "|" + cs).encode("utf-8", "ignore")).hexdigest()
                except Exception:
                    return sig
        return sig
    if isinstance(active_obs, str):
        return _stable_state_sig(active_obs)
    return None


def _make_state_docid(url: Optional[str], state_sig: Optional[str]) -> Optional[str]:
    """Create a docid that represents a 'page state' to connect failures/revisits in GoC."""
    u = _canonicalize_url_for_sig(url)
    if not u or not state_sig:
        return None
    return f"state:{u}#{state_sig[:12]}"
def _derive_static_avoid_actions(active_obs: str, instruction: str) -> Dict[str, str]:
    """Return {action: reason} that should be avoided in the current observation."""
    out: Dict[str, str] = {}
    cands = _extract_actionable_candidates(active_obs)
    if not cands:
        return out

    is_read_reviews = _is_review_distribution_task(instruction)

    for bid, role, name in cands:
        nm = (name or "").lower()
        rl = (role or "").lower()
        act = f"click('{bid}')"

        # Avoid dead-end legal/policy links.
        if _BAD_PAGE_RE.search(nm) or _BAD_PAGE_RE.search(rl):
            out[act] = "policy/cookie/legal link"
            continue

        # Avoid review-writing CTAs for read/lookup tasks.
        if is_read_reviews and any(k in nm for k in ["add your review", "write a review", "submit review", "add review", "write review"]):
            out[act] = "review-writing CTA (read-task)"
            continue

    return out


def _focus_lines_for_task(obs_text: str, instruction: str, *, max_lines: int = 30, max_chars: int = 1400) -> str:
    """Extract goal-related lines (cheap perception) to reduce noop/scroll loops."""
    if not isinstance(obs_text, str) or not isinstance(instruction, str):
        return ""
    if not _is_review_distribution_task(instruction):
        return ""

    # Look for review/rating breakdown signals.
    patterns = [
        re.compile(r"\breview\b", re.I),
        re.compile(r"\brating\b", re.I),
        re.compile(r"\bstar\b", re.I),
        re.compile(r"\bcustomer reviews\b", re.I),
        re.compile(r"\b5\s*stars?\b|\b4\s*stars?\b|\b3\s*stars?\b|\b2\s*stars?\b|\b1\s*stars?\b", re.I),
        re.compile(r"\b5\s*:\s*\d+|\b4\s*:\s*\d+|\b3\s*:\s*\d+|\b2\s*:\s*\d+|\b1\s*:\s*\d+", re.I),
    ]

    lines = []
    for ln in obs_text.splitlines():
        if len(ln) > 300:
            ln = ln[:300] + "…"
        hit = any(p.search(ln) for p in patterns)
        if hit:
            lines.append(ln)
        if len(lines) >= max_lines:
            break
    if not lines:
        return ""
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
    return out




def _dom_focus_lines_for_task(dom_text: str, instruction: str, *, max_lines: int = 80, max_chars: int = 2600) -> str:
    """Filter dom_object.strings into a compact, goal-relevant snippet.

    WebArena Shopping pages sometimes keep important text (e.g., star histogram labels/counts)
    in dom_object.strings rather than in the accessibility tree. However, dom_object.strings
    also contains noisy asset/JS paths. This helper aggressively filters noise and prioritizes
    lines that look like *actual* rating/count evidence.
    """
    if not isinstance(dom_text, str) or not isinstance(instruction, str):
        return ""
    if not dom_text.strip():
        return ""

    def _looks_like_asset_or_path(low: str) -> bool:
        if not low:
            return False
        if 'http://' in low or 'https://' in low:
            return True
        if '/static/' in low or 'static/version' in low:
            return True
        # Common asset extensions
        if re.search(r"\.(js|css|map|png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf)(\?|$)", low):
            return True
        # Long path-like strings
        if low.count('/') >= 2 and low.count(' ') == 0:
            return True
        if len(low) > 180 and low.count(' ') == 0:
            return True
        return False

    kws: list[str] = []
    if _is_review_distribution_task(instruction):
        kws += [
            'review', 'reviews', 'rating', 'ratings', 'star', 'stars',
            'customer reviews', 'global ratings', 'histogram', 'breakdown',
            'five star', 'four star', 'three star', 'two star', 'one star',
        ]
    else:
        kws += _keywords_for_instruction(instruction)

    # Dedupe keywords
    seen = set()
    kws2: list[str] = []
    for k in kws:
        kk = (k or '').strip().lower()
        if kk and kk not in seen:
            seen.add(kk)
            kws2.append(kk)

    candidates: list[tuple[int, str]] = []
    seen_line = set()

    for raw in dom_text.splitlines():
        ln = (raw or '').strip()
        if not ln:
            continue
        low = ln.lower()

        if _looks_like_asset_or_path(low):
            continue

        if not any(k in low for k in kws2):
            continue

        # Evidence features
        has_star = bool(re.search(r"\b[1-5]\s*stars?\b", low))
        nums = re.findall(r"\b\d{1,7}\b", ln)
        has_num = bool(nums)
        has_pct = bool(re.search(r"\b\d{1,3}\s*%", ln))
        has_paren_num = bool(re.search(r"\(\s*\d{1,7}\s*\)", ln))
        has_aria = ('aria-' in low) or ('aria_' in low) or ('aria=' in low)
        has_review_word = ('review' in low) or ('reviews' in low)
        has_rating_word = ('rating' in low) or ('ratings' in low)

        other_num = any(n not in {'1','2','3','4','5'} for n in nums)

        # Drop bare labels like "5 stars" with no other numeric evidence.
        if has_star and not other_num and not (has_pct or has_paren_num or has_aria):
            continue

        # Require at least some evidence: pct/paren/aria, or star+other_num, or review/rating word with a number.
        if not (has_pct or has_paren_num or has_aria or (has_star and other_num) or ((has_review_word or has_rating_word) and has_num)):
            continue

        # Trim and dedupe
        if len(ln) > 260:
            ln = ln[:260] + '…'
            low = ln.lower()

        key = re.sub(r"\s+", " ", low).strip()
        if key in seen_line:
            continue
        seen_line.add(key)

        # Score: prefer lines with explicit counts/percent and review words.
        score = 0
        if has_star:
            score += 2
        if has_review_word:
            score += 2
        if has_rating_word:
            score += 1
        if has_paren_num:
            score += 3
        if has_pct:
            score += 2
        if has_aria:
            score += 2
        if has_star and other_num:
            score += 2

        candidates.append((score, ln))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)

    lines_out: list[str] = []
    out_chars = 0
    for _, ln in candidates:
        lines_out.append(ln)
        out_chars += len(ln) + 1
        if len(lines_out) >= max_lines or out_chars >= max_chars:
            break

    if not lines_out:
        return ""
    out = "\n".join(lines_out)
    if len(out) > max_chars:
        out = out[: max_chars - 1] + '…'
    return out

def _snap_to_rating_scale(v: int) -> int:
    """Snap an integer in [0,100] to the nearest 20-step rating scale."""
    vv = max(0, min(100, int(v)))
    snapped = int(round(vv / 20.0) * 20)
    return max(0, min(100, snapped))


def _extract_review_distribution(text: str) -> Optional[Dict[int, int]]:
    """Best-effort extract of 1–5 star review *distribution* values (0..100 scale).

    IMPORTANT: WebArena task instructions include a fixed scale mapping (1->20, ...).
    Also, the Shopping UI often shows a fixed filter mapping (5 stars -> Rating 100, ...)
    that is *not* the distribution. We therefore require evidence of percentages or
    aria-value attributes (aria-valuenow/aria-valuetext) before returning a result.

    Returns a dict {star: value} only when all 1..5 stars are found with plausible
    0..100 scale values AND we saw distribution evidence (%, aria-valuenow, etc.).
    """
    if not isinstance(text, str) or not text.strip():
        return None

    out: Dict[int, int] = {}
    evidence = False

    # Scan line-by-line to avoid picking up task-instruction scale mappings.
    for ln in text.splitlines():
        low = ln.lower()
        # Only consider lines that look like actual review/rating UI.
        if not any(k in low for k in ['star', 'stars', 'review', 'rating', 'aria-valuenow', 'aria-valuetext', '%']):
            continue

        # Percentages like '5 stars 67%'
        for m in re.finditer(r"\b([1-5])\s*stars?\b.{0,80}?\b(\d{1,3})\s*%", ln, flags=re.IGNORECASE):
            star = int(m.group(1))
            val = int(m.group(2))
            if 0 <= val <= 100:
                snapped = _snap_to_rating_scale(val)
                if abs(val - snapped) <= 8:
                    out[star] = snapped
                    evidence = True

        # aria-valuenow / aria-valuetext signals (captured from properties suffix)
        for m in re.finditer(r"\b([1-5])\s*stars?\b.{0,140}?\baria-valuenow\s*=\s*(\d{1,3})\b", ln, flags=re.IGNORECASE):
            star = int(m.group(1))
            val = int(m.group(2))
            if 0 <= val <= 100:
                out[star] = _snap_to_rating_scale(val)
                evidence = True

        for m in re.finditer(r"\b([1-5])\s*stars?\b.{0,140}?\baria-valuetext\s*=\s*(\d{1,3})\s*%", ln, flags=re.IGNORECASE):
            star = int(m.group(1))
            val = int(m.group(2))
            if 0 <= val <= 100:
                snapped = _snap_to_rating_scale(val)
                if abs(val - snapped) <= 8:
                    out[star] = snapped
                    evidence = True

        # 'Rating 73' style, but only if the line also contains evidence markers.
        if any(k in low for k in ['%', 'aria-valuenow', 'aria-valuetext', 'reviews']):
            for m in re.finditer(r"\b([1-5])\s*stars?\b.{0,80}?\bRating\s*(\d{1,3})\b", ln, flags=re.IGNORECASE):
                star = int(m.group(1))
                val = int(m.group(2))
                if 0 <= val <= 100:
                    out[star] = _snap_to_rating_scale(val)
                    evidence = True

    if not evidence:
        return None

    # Require full distribution.
    if all(k in out for k in [1, 2, 3, 4, 5]):
        # Reject the fixed star->rating mapping (not a distribution): 5:100,4:80,3:60,2:40,1:20
        fixed = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
        cand = {k: int(out[k]) for k in [1, 2, 3, 4, 5]}
        if cand == fixed:
            return None
        if all(cand[k] in _RATING_SCALE_VALUES for k in [1, 2, 3, 4, 5]):
            return cand

    return None


def _extract_review_star_counts_with_evidence(text: str) -> Tuple[Optional[Dict[int, int]], list[str]]:
    """Extract 1–5 star *review counts* from observation text, with evidence lines.

    WebArena shopping UIs often expose rating *labels* like "5 stars" (no counts).
    A naive regex will misread these labels as counts (5->5, 4->4, ...). We only accept
    counts when there is strong UI evidence: parentheses counts, explicit "reviews/ratings"
    tokens, or aria-label/valuetext signals.

    Returns (dist, evidence_lines), where dist is {1..5: count}.
    """
    if not isinstance(text, str) or not text.strip():
        return None, []

    def _looks_like_asset_or_path(low: str) -> bool:
        if not low:
            return False
        if 'http://' in low or 'https://' in low:
            return True
        if '/static/' in low or 'static/version' in low:
            return True
        if re.search(r"\.(js|css|map|png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf)(\?|$)", low):
            return True
        if low.count('/') >= 2 and low.count(' ') == 0:
            return True
        return False

    out: Dict[int, int] = {}
    evidence_lines: list[str] = []

    star_pat = re.compile(r"\b([1-5])\s*stars?\b", re.IGNORECASE)

    # Strong patterns (counts)
    pat_paren = re.compile(r"\b([1-5])\s*stars?\b[^\n]{0,80}?\(\s*(\d{1,7})\s*\)", re.IGNORECASE)
    # For *counts*, require explicit "review(s)" tokens to avoid misreading UI labels like
    # "Rating 5 stars" (filter labels) as counts.
    pat_after_reviews = re.compile(r"\b([1-5])\s*stars?\b[^\n\d%]{0,40}(\d{1,7})\b[^\n]{0,18}\breviews?\b", re.IGNORECASE)
    pat_before_reviews = re.compile(r"\b(\d{1,7})\b[^\n]{0,18}\breviews?\b[^\n]{0,60}?\b([1-5])\s*stars?\b", re.IGNORECASE)
    pat_aria = re.compile(r"\b([1-5])\s*stars?\b[^\n]{0,200}?\baria-(?:label|valuetext)\s*[=:]\s*[^\n]{0,120}?(\d{1,7})\b", re.IGNORECASE)

    def _record(star: int, cnt: int, ln: str):
        if 0 <= cnt <= 10_000_000:
            prev = out.get(star)
            out[star] = cnt if prev is None else max(prev, cnt)
            if ln:
                evidence_lines.append(ln[:260])

    # Actionable element summary lines look like:
    #   "- 1589 | radio | Rating ... 5 stars"
    # The leading numeric token is an *element id*, not a review count. If we
    # naively parse numbers from such lines, we will produce bogus distributions
    # like {5:5, 4:4, ...} or treat element ids as counts. We therefore strip the
    # actionable prefix before attempting any extraction.
    _ACTIONABLE_PREFIX_RE = re.compile(r"^\s*-\s*\d+\s*\|\s*[^|]*\|\s*", re.IGNORECASE)

    for raw in text.splitlines():
        ln0 = (raw or '').strip()
        if not ln0:
            continue

        # Remove "- <id> | <role> |" prefix if present.
        had_actionable_prefix = bool(_ACTIONABLE_PREFIX_RE.match(ln0))
        ln = _ACTIONABLE_PREFIX_RE.sub('', ln0).strip()
        low = ln.lower()

        if _looks_like_asset_or_path(low):
            continue

        # counts tasks want counts, not % distributions
        if re.search(r"\b\d{1,3}\s*%", ln):
            continue

        if not star_pat.search(ln):
            continue

        for m in pat_paren.finditer(ln):
            _record(int(m.group(1)), int(m.group(2)), ln)

        for m in pat_after_reviews.finditer(ln):
            _record(int(m.group(1)), int(m.group(2)), ln)

        for m in pat_before_reviews.finditer(ln):
            _record(int(m.group(2)), int(m.group(1)), ln)

        for m in pat_aria.finditer(ln):
            _record(int(m.group(1)), int(m.group(2)), ln)

        # Conservative fallback: allow only when the line explicitly mentions reviews.
        # Never fall back on actionable-summary lines (those often contain element ids).
        if (not had_actionable_prefix) and ('review' in low or 'reviews' in low):
            nums = re.findall(r"\b\d{1,7}\b", ln)
            if nums:
                for sm in star_pat.finditer(ln):
                    s = int(sm.group(1))
                    cand = None
                    for n in nums:
                        # Avoid treating the star label itself as the count.
                        if n not in {'1','2','3','4','5'}:
                            cand = int(n)
                            break
                    if cand is not None:
                        _record(s, cand, ln)

    # Deduplicate evidence lines (keep order)
    dedup: list[str] = []
    seen = set()
    for e in evidence_lines:
        k = re.sub(r"\s+", ' ', e).strip().lower()
        if k and k not in seen:
            seen.add(k)
            dedup.append(e)
        if len(dedup) >= 10:
            break

    if all(k in out for k in [1, 2, 3, 4, 5]):
        cand = {k: int(out[k]) for k in [1, 2, 3, 4, 5]}
        # Reject obvious bogus mappings
        if cand == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}:
            return None, dedup
        if cand == {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}:
            return None, dedup
        return cand, dedup

    return None, dedup


def _extract_review_star_counts(text: str) -> Optional[Dict[int, int]]:
    dist, _ev = _extract_review_star_counts_with_evidence(text)
    return dist

def _safe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"



"""Bid / element-id detection.

BrowserGym/WebArena observations vary by version:
- Some expose bids like 'a46'
- Others expose plain numeric ids under keys like "browsergym_id": "213"
"""

# WebArena-style bids like 'a46'
_BID_RE = re.compile(r"\b[a-zA-Z]\d{1,5}\b")

# BrowserGym ids exposed as JSON fields (quoted or unquoted)
_BROWSERGYM_ID_RE = re.compile(r"browsergym_id\"\s*:\s*\"?\d+\"?")

# Actionable-element list lines like: - 213 | link | ...
_ACTIONABLE_ID_LINE_RE = re.compile(r"^\s*-\s*(\d{1,6}|[a-zA-Z]\d{1,5})\s*(?:\||$)", re.MULTILINE)



def _has_any_element_id(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(
        _BID_RE.search(text)
        or _BROWSERGYM_ID_RE.search(text)
        or _ACTIONABLE_ID_LINE_RE.search(text)
    )


# Parse our compact actionable-element list lines (emitted by _summarize_actionable_elements).
_ACTIONABLE_LINE_RE = re.compile(r"^\s*-\s*(\S+)\s*(?:\|\s*([^|]+?)\s*(?:\|\s*(.*?))?)?\s*$")


def _extract_actionable_candidates(active_obs: str) -> list[tuple[str, str, str]]:
    """Extract (bid, role, name) candidates from the 'Actionable element ids' block."""
    if not isinstance(active_obs, str):
        return []
    out: list[tuple[str, str, str]] = []
    for ln in active_obs.splitlines():
        if not ln.lstrip().startswith('-'):
            continue
        m = _ACTIONABLE_LINE_RE.match(ln)
        if not m:
            continue
        bid = (m.group(1) or '').strip()
        role = (m.group(2) or '').strip()
        name = (m.group(3) or '').strip()
        if bid:
            out.append((bid, role, name))
    return out


def _keywords_for_instruction(instruction: str) -> list[str]:
    """Heuristic keywords to select relevant clickable elements for a task."""
    if not isinstance(instruction, str):
        return []
    s = instruction.lower()
    kws: list[str] = []
    # Common WebArena intents
    if 'review' in s or 'rating' in s or 'stars' in s:
        kws += ['review', 'reviews', 'rating', 'ratings', 'star', 'stars', 'customer', 'comment']
    if 'checkout' in s or 'cart' in s:
        kws += ['cart', 'checkout', 'add to cart', 'basket']
    if 'search' in s:
        kws += ['search']
    if 'login' in s or 'sign in' in s:
        kws += ['login', 'sign in', 'sign-in', 'account', 'password', 'email', 'username']
    if 'gitlab' in s:
        kws += ['issues', 'merge request', 'project', 'search', 'login']
    if 'wikipedia' in s:
        kws += ['search', 'article', 'page']
    # Add some words from the instruction itself (topical terms)
    toks = re.findall(r"[a-z]{4,}", s)
    kws += toks[:12]
    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for k in kws:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:20]


def _suggest_action_from_candidates(
    *,
    active_obs: str,
    instruction: str,
    avoid_actions: set[str],
    extra_prefer_bids: Optional[list[str]] = None,
) -> Optional[str]:
    """Suggest a non-avoided next action based on keyword-matching candidates."""
    cands = _extract_actionable_candidates(active_obs)
    if not cands:
        return None
    kws = _keywords_for_instruction(instruction)

    # Prefer some explicit bids if provided (e.g., from previous success heuristics).
    if extra_prefer_bids:
        for bid in extra_prefer_bids:
            act = f"click('{bid}')"
            if act not in avoid_actions:
                return act

    def score(c: tuple[str, str, str]) -> tuple[int, int, int]:
        bid, role, name = c
        txt = (name or role or '').lower()
        hit = sum(1 for k in kws if k in txt)
        role_bonus = 1 if (role or '').strip().lower() in {'link', 'button', 'tab', 'menuitem'} else 0
        name_len = len(name or '')
        return (hit, role_bonus, name_len)

    best = None
    best_sc = (-1, -1, -1)
    for c in cands:
        bid, role, name = c
        act = f"click('{bid}')"
        if act in avoid_actions:
            continue
        sc = score(c)
        if sc > best_sc:
            best_sc = sc
            best = act

    if best is not None and best_sc[0] > 0:
        return best

    # If no keyword hits, fall back to the first non-avoided click on a likely interactive role.
    for bid, role, name in cands:
        if (role or '').strip().lower() in {'link', 'button', 'tab'}:
            act = f"click('{bid}')"
            if act not in avoid_actions:
                return act
    for bid, role, name in cands:
        act = f"click('{bid}')"
        if act not in avoid_actions:
            return act
    return None


def _extract_actionable_obs(text: str, max_chars: int) -> str:
    """Extract a prompt-friendly slice from a potentially huge observation.

    BrowserGym WebArena observations may include long headers or JSON wrappers.
    We try to anchor on the first bid id (e.g., a46) or common section markers.
    Fallbacks: marker slice -> bid slice -> tail.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    if max_chars is None or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    # Prefer anchors that tend to be close to actionable element identifiers.
    # WARNING: the substring "dom" often appears very early (e.g., "dom_object") and
    # can cause us to slice into huge, low-signal numeric arrays. Avoid anchoring on it.
    markers = [
        'browsergym_id', '"browsergym_id"',
        'Accessibility Tree', 'A11Y', 'A11y', 'a11y',
        'OBSERVATION', 'Observation', 'PAGE CONTENT', 'Page content',
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < len(text) - 200:
            return text[idx: idx + max_chars]

    # Anchor around the first discovered element id.
    m = _BID_RE.search(text)
    if m:
        start = max(0, m.start() - 400)
        return text[start: start + max_chars]

    m2 = _BROWSERGYM_ID_RE.search(text)
    if m2:
        start = max(0, m2.start() - 600)
        return text[start: start + max_chars]

    # Last resort: keep the tail (often contains the latest page content).
    return text[-max_chars:]


def _json_default(o: object) -> object:
    """Best-effort JSON serializer for BrowserGym observations.

    Observations may include numpy arrays/scalars, bytes, Paths, etc.
    We convert common cases to JSON-friendly values and fall back to str().
    """
    # numpy arrays / scalars often implement .tolist() / .item()
    try:
        tolist = getattr(o, "tolist", None)
        if callable(tolist):
            return tolist()
    except Exception:
        pass
    try:
        item = getattr(o, "item", None)
        if callable(item):
            return item()
    except Exception:
        pass

    if isinstance(o, (bytes, bytearray)):
        try:
            return o.decode("utf-8", "replace")
        except Exception:
            return str(o)

    try:
        from pathlib import Path as _Path
        if isinstance(o, _Path):
            return str(o)
    except Exception:
        pass

    return str(o)




def _summarize_actionable_elements(obs: Any, max_items: int = 25) -> List[str]:
    """Return a compact list of actionable element ids.

    Many BrowserGym/WebArena installs expose element ids as plain numeric strings
    under accessibility nodes like {"browsergym_id": "213"}. These ids can be
    used directly in bid-style actions (e.g., click('213'), fill('213','text')).

    We traverse the raw observation payload and extract ids with some human-facing
    signal (role/name/value/properties). This list is printed near the top of the
    prompt so the model can reliably choose id-based actions.

    NOTE: This function is intentionally schema-agnostic. Different BrowserGym
    versions may use different keys (browsergym_id/bid/id) and store text/value
    inside nested dicts or a "properties" list.
    """

    roles_keep = {
        'link', 'button', 'textbox', 'searchbox', 'combobox', 'menuitem', 'tab',
        'checkbox', 'radio', 'option', 'listbox', 'textarea', 'text', 'heading',
        'slider', 'progressbar', 'spinbutton', 'switch', 'group'
    }

    important_props = {
        'aria-label', 'aria-labelledby', 'aria-describedby',
        'aria-valuenow', 'aria-valuetext', 'aria-valuemin', 'aria-valuemax',
        'value', 'checked', 'selected', 'expanded', 'pressed', 'disabled',
    }

    def _val_to_str(v: Any) -> str:
        if v is None:
            return ''
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)):
            return str(v)
        if isinstance(v, dict):
            vv = v.get('value') if 'value' in v else None
            if isinstance(vv, (str, int, float, bool)):
                return str(vv)
        return ''

    def _is_valid_bid(b: Any) -> bool:
        if b is None:
            return False
        s = str(b).strip()
        if not s:
            return False
        return bool(re.fullmatch(r"\d{1,6}|[a-zA-Z]\d{1,5}", s))

    def _extract_bid(d: dict) -> str:
        for k in ('browsergym_id', 'browsergymId', 'browsergymID', 'bid', 'bg_id'):
            if k in d and _is_valid_bid(d.get(k)):
                return str(d.get(k)).strip()
        # Some schemas use a generic "id" field; only accept if it looks like a bid.
        if 'id' in d and _is_valid_bid(d.get('id')):
            return str(d.get('id')).strip()
        return ''

    def _extract_role(d: dict) -> str:
        rv = d.get('role')
        s = _val_to_str(rv).strip()
        return s

    def _extract_props_suffix(d: dict) -> str:
        props = d.get('properties')
        if not isinstance(props, list) or not props:
            return ''
        parts: list[str] = []
        for p in props:
            if not isinstance(p, dict):
                continue
            pn = (p.get('name') or p.get('property') or p.get('key') or '')
            pv = p.get('value') if 'value' in p else p.get('val')
            pn_s = str(pn).strip()
            if not pn_s:
                continue
            pn_l = pn_s.lower()
            if pn_l not in important_props:
                continue
            pv_s = _val_to_str(pv).strip()
            if not pv_s:
                continue
            # Keep short; these end up in prompts.
            if len(pv_s) > 60:
                pv_s = pv_s[:59] + '…'
            parts.append(f"{pn_l}={pv_s}")
            if len(parts) >= 4:
                break
        if not parts:
            return ''
        return ' [' + ', '.join(parts) + ']'

    def _extract_name(d: dict) -> str:
        for k in ('name', 'label', 'text', 'value'):
            s = _val_to_str(d.get(k)).strip()
            if s:
                base = s
                break
        else:
            base = ''

        # If base is empty, sometimes aria-label lives in properties.
        if not base:
            props = d.get('properties')
            if isinstance(props, list):
                for p in props:
                    if isinstance(p, dict) and str(p.get('name') or '').strip().lower() == 'aria-label':
                        base = _val_to_str(p.get('value')).strip()
                        if base:
                            break

        suffix = _extract_props_suffix(d)
        out = (base + suffix).strip()
        if len(out) > 180:
            out = out[:179] + '…'
        return out

    best: dict[str, tuple[str, str]] = {}

    def visit(x: Any):
        if isinstance(x, dict):
            bid = _extract_bid(x)
            if bid:
                role = _extract_role(x)
                name = _extract_name(x)
                keep = bool(name) or (role.lower() in roles_keep)
                if keep:
                    prev = best.get(bid)
                    score = (1 if name else 0, len(name))
                    if prev is None:
                        best[bid] = (role, name)
                    else:
                        prev_score = (1 if prev[1] else 0, len(prev[1]))
                        if score > prev_score:
                            best[bid] = (role, name)
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)

    try:
        visit(obs)
    except Exception:
        return []

    items = [(bid, role, name) for bid, (role, name) in best.items() if isinstance(bid, str)]
    # Prefer named elements first.
    items.sort(key=lambda t: (0 if t[2] else 1, -len(t[2] or ''), t[0]))

    out: list[str] = []
    for bid, role, name in items[:max_items]:
        role = (role or '').strip()
        name = (name or '').strip()
        if name and role:
            out.append(f"- {bid} | {role} | {name}")
        elif name:
            out.append(f"- {bid} | {name}")
        elif role:
            out.append(f"- {bid} | {role}")
        else:
            out.append(f"- {bid}")
    return out

# WebChoreArena task JSONs often use placeholders like "__SHOPPING__".
# We map them to the corresponding BrowserGym/WebArena environment variables.
_PLACEHOLDER_TO_ENV = {
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__REDDIT__": "WA_REDDIT",
    "__GITLAB__": "WA_GITLAB",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
    "__MAP__": "WA_MAP",
    "__HOMEPAGE__": "WA_HOMEPAGE",
}


def _substitute_placeholders(value: Any) -> Any:
    """Replace __FOO__ placeholders with WA_* env var values when available."""
    if not isinstance(value, str):
        return value
    s = value
    for placeholder, env_name in _PLACEHOLDER_TO_ENV.items():
        if placeholder in s:
            env_val = os.environ.get(env_name)
            if env_val:
                s = s.replace(placeholder, env_val)
    return s


def _substitute_placeholders_deep(obj: Any) -> Any:
    """Recursively substitute __FOO__ placeholders inside nested configs."""
    if isinstance(obj, str):
        return _substitute_placeholders(obj)
    if isinstance(obj, list):
        return [_substitute_placeholders_deep(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_placeholders_deep(v) for k, v in obj.items()}
    return obj


def _shape_hint(x: Any) -> Optional[str]:
    """Return a short shape hint for array-like objects."""
    try:
        shp = getattr(x, "shape", None)
        if shp is not None:
            return str(tuple(int(i) for i in shp))
    except Exception:
        pass
    try:
        if isinstance(x, list) and x and isinstance(x[0], list):
            # very rough for nested lists
            return f"({len(x)}, {len(x[0])}, ...)"
    except Exception:
        pass
    return None


def _prune_obs_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove/replace extremely large fields (e.g., screenshots) for prompting/logging.

    NOTE: WebArena observations often nest huge screenshot arrays under `dom_object`.
    We therefore prune *recursively* (depth-limited) to keep the meaningful text.
    """
    heavy_keys = {
        "screenshot",
        "screenshots",
        "image",
        "images",
        "pixels",
        "rgb",
        "raw_screenshot",
        "rgba",
        "frame",
        "frames",
    }

    def _looks_like_numeric_blob(v: Any) -> bool:
        try:
            if isinstance(v, list) and len(v) > 512:
                if all(isinstance(x, (int, float)) for x in v[:256]):
                    return True
            if isinstance(v, list) and len(v) > 64 and isinstance(v[0], list):
                sample = v[0]
                if isinstance(sample, list) and len(sample) > 64 and all(isinstance(x, (int, float)) for x in sample[:64]):
                    return True
        except Exception:
            return False
        return False

    def _prune_value(v: Any, depth: int) -> Any:
        if v is None:
            return None
        if depth <= 0:
            if isinstance(v, (dict, list, tuple)):
                return f"<omitted; depth_limit; shape={_shape_hint(v) or 'unknown'}>"
            return v
        if _looks_like_numeric_blob(v):
            return f"<omitted; numeric_blob; shape={_shape_hint(v) or 'unknown'}>"
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            # keep likely-useful keys first
            preferred = [
                "goal","url","title","page_title","text","observation","content",
                "axtree_txt","ax_tree_txt","accessibility_tree_txt","a11y_tree","a11y",
                "pruned_html","html",
                "chat_messages","open_pages_titles","open_pages_urls","active_page_index",
            ]
            seen = set()
            for k in preferred:
                if k in v:
                    vv = v.get(k)
                    if k in heavy_keys or _looks_like_numeric_blob(vv):
                        out[k] = f"<omitted; shape={_shape_hint(vv) or 'unknown'}>"
                    else:
                        out[k] = _prune_value(vv, depth - 1)
                    seen.add(k)
            for k, vv in v.items():
                if k in seen:
                    continue
                if k in heavy_keys or _looks_like_numeric_blob(vv):
                    out[k] = f"<omitted; shape={_shape_hint(vv) or 'unknown'}>"
                else:
                    out[k] = _prune_value(vv, depth - 1)
            return out
        if isinstance(v, list):
            if _looks_like_numeric_blob(v):
                return f"<omitted; numeric_blob; shape={_shape_hint(v) or 'unknown'}>"
            # truncate very long lists to keep logs/prompt stable
            n = len(v)
            if n > 200:
                head = [_prune_value(x, depth - 1) for x in v[:50]]
                head.append(f"<omitted; {n-50} more items>")
                return head
            return [_prune_value(x, depth - 1) for x in v]
        if isinstance(v, tuple):
            return tuple(_prune_value(x, depth - 1) for x in v)
        return v

    # Put likely-useful keys first to maximize signal before any truncation.
    preferred_order = [
        "goal",
        "chat_messages",
        "open_pages_titles",
        "open_pages_urls",
        "active_page_index",
        "url",
        "title",
        "page_title",
        "text",
        "observation",
        "a11y",
        "a11y_tree",
        "dom",
        "dom_object",
        "content",
        "axtree_txt",
        "pruned_html",
    ]

    pruned: Dict[str, Any] = {}
    seen = set()

    for k in preferred_order:
        if k in obs:
            vv = obs.get(k)
            if k in heavy_keys or _looks_like_numeric_blob(vv):
                pruned[k] = f"<omitted; shape={_shape_hint(vv) or 'unknown'}>"
            else:
                pruned[k] = _prune_value(vv, depth=3)
            seen.add(k)

    for k, vv in obs.items():
        if k in seen:
            continue
        if k in heavy_keys or _looks_like_numeric_blob(vv):
            pruned[k] = f"<omitted; shape={_shape_hint(vv) or 'unknown'}>"
        else:
            pruned[k] = _prune_value(vv, depth=3)

    return pruned




def _coerce_to_text(v: Any) -> Optional[str]:
    """Convert common observation payload variants into a non-empty text string."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v
    elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
        s = "\n".join(v)
    elif not isinstance(v, (dict, list, tuple)):
        try:
            s = str(v)
        except Exception:
            return None
    else:
        return None
    s = s.strip("\n")
    return s if s.strip() else None


def _render_axtree_object_to_text(ax: Any, *, max_lines: int = 2500, max_depth: int = 14) -> Optional[str]:
    """Render BrowserGym/WebArena `axtree_object` (structured accessibility tree) into text.

    Some BrowserGym builds provide only a structured tree (e.g., `axtree_object`)
    and omit `axtree_txt`. In that case our previous fallback (`json_pruned`) is
    very low-signal for LLMs and often hides the clickable ids.

    This renderer is intentionally *best-effort* and schema-agnostic:
    - It prefers nodes that expose `browsergym_id` (the bid used by click/fill).
    - It prints one line per bid with role/name/value when available.
    - It attempts to preserve hierarchy (indentation) when children are embedded.
    """
    if ax is None:
        return None
    if not isinstance(ax, (dict, list)):
        return None

    def _val_to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)):
            return str(v)
        if isinstance(v, dict):
            # BrowserGym often wraps as {"value": "..."}
            vv = v.get("value") if "value" in v else None
            if isinstance(vv, str):
                return vv
        return ""

    def _get_role(d: dict) -> str:
        rv = d.get("role")
        if isinstance(rv, str):
            return rv
        if isinstance(rv, dict):
            return _val_to_str(rv)
        return ""

    def _get_name(d: dict) -> str:
        for k in ("name", "label", "text", "value"):
            sv = d.get(k)
            s = _val_to_str(sv)
            if s:
                return s
        return ""

    def _props_suffix(d: dict) -> str:
        """Extract compact, high-signal a11y properties (e.g., aria-valuenow)."""
        props = d.get('properties')
        if not isinstance(props, list):
            return ''
        keep = {'aria-valuenow', 'aria-valuetext', 'aria-label', 'value', 'checked', 'selected', 'expanded', 'pressed', 'disabled'}
        parts = []
        for p in props:
            if not isinstance(p, dict):
                continue
            pn = p.get('name') or p.get('property')
            if not pn:
                continue
            pn_s = str(pn).strip().lower()
            if pn_s not in keep:
                continue
            pv = p.get('value')
            pv_s = _val_to_str(pv)
            if not pv_s and isinstance(pv, (int, float, bool)):
                pv_s = str(pv)
            if not pv_s:
                continue
            parts.append(f"{pn_s}={pv_s}")
            if len(parts) >= 4:
                break
        if not parts:
            return ''
        return ' [' + ', '.join(parts) + ']'

    # If the tree is stored as a node list + id references, try to build a map.
    node_map: dict[Any, dict] = {}
    root_id: Any = None
    if isinstance(ax, dict):
        for rk in ("rootId", "root_id", "root", "rootNodeId", "root_node_id"):
            if rk in ax:
                root_id = ax.get(rk)
                break
        nodes = ax.get("nodes")
        if isinstance(nodes, list) and nodes and all(isinstance(n, dict) for n in nodes[:10]):
            for n in nodes:
                nid = n.get("nodeId") if "nodeId" in n else n.get("id")
                if nid is not None:
                    node_map[nid] = n
            # Some schemas use integer 0 as root.
            if root_id is None and 0 in node_map:
                root_id = 0
            if root_id is None and nodes:
                # Fallback: first node.
                root_id = nodes[0].get("nodeId", nodes[0].get("id"))

    lines: list[str] = ["Accessibility Tree (rendered):"]
    seen_bids: set[str] = set()
    seen_text: set[str] = set()
    bid_line_count = 0

    def emit_bid_line(depth: int, bid: str, role: str, name: str):
        nonlocal bid_line_count
        if not bid or bid in seen_bids:
            return
        seen_bids.add(bid)
        bid_line_count += 1
        indent = "  " * max(0, depth)
        role = (role or "").strip()
        name = (name or "").strip()
        if role and name:
            lines.append(f"{indent}- {bid} | {role} | {name}")
        elif name:
            lines.append(f"{indent}- {bid} | {name}")
        elif role:
            lines.append(f"{indent}- {bid} | {role}")
        else:
            lines.append(f"{indent}- {bid}")

    def emit_text_line(depth: int, role: str, name: str):
        """Emit a non-actionable text line (helps when no browsergym_id is present)."""
        role = (role or "").strip()
        name = (name or "").strip()
        if not name:
            return
        key = f"{role}|{name}"
        if key in seen_text:
            return
        seen_text.add(key)
        indent = "  " * max(0, depth)
        if role:
            lines.append(f"{indent}{role}: {name}")
        else:
            lines.append(f"{indent}{name}")

    def walk_node(obj: Any, depth: int):
        if len(lines) >= max_lines or depth > max_depth:
            return
        if isinstance(obj, dict):
            bid = obj.get("browsergym_id")
            role = _get_role(obj)
            name = _get_name(obj)
            name = (name + _props_suffix(obj)) if name else _props_suffix(obj)
            if bid is not None:
                emit_bid_line(depth, str(bid), role, name)
            else:
                # Only emit a small amount of non-actionable text to avoid bloat.
                rlow = (role or "").strip().lower()
                if rlow in {"heading", "text", "statictext", "link", "button", "label"}:
                    emit_text_line(depth, role, name)

            # Children can be embedded dicts or id references.
            for ck in ("children", "childIds", "child_ids", "childrenIds", "children_ids"):
                ch = obj.get(ck)
                if isinstance(ch, list) and ch:
                    for c in ch:
                        if isinstance(c, dict):
                            walk_node(c, depth + 1)
                        elif node_map and c in node_map:
                            walk_node(node_map[c], depth + 1)
                    return

            # Heuristic: traverse a small set of structural keys first.
            for k in ("document", "tree", "node", "root", "body", "content"):
                v = obj.get(k)
                if isinstance(v, (dict, list)):
                    walk_node(v, depth + 1)

            # Generic traversal (depth-limited) but avoid huge blobs.
            for k, v in obj.items():
                if k in {"screenshot", "screenshots", "image", "images", "pixels", "rgb", "rgba", "raw_screenshot"}:
                    continue
                if isinstance(v, (dict, list)):
                    walk_node(v, depth + 1)

        elif isinstance(obj, list):
            for v in obj:
                if len(lines) >= max_lines:
                    break
                if isinstance(v, (dict, list)):
                    walk_node(v, depth)

    if node_map and root_id in node_map:
        walk_node(node_map[root_id], 0)
    else:
        walk_node(ax, 0)

    if len(lines) <= 1:
        return None
    return "\n".join(lines)


def _best_text_from_obs_dict(obs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Pick the best available *textual* representation from a BrowserGym observation dict.

    Many BrowserGym/WebArena builds tuck the human-readable view under keys like
    'axtree_txt' (accessibility tree text) or 'pruned_html'. Some versions nest
    these under 'dom_object' or a similar sub-dict.
    """
    preferred_keys = (
        # Accessibility-tree style text (best)
        "axtree_txt",
        "ax_tree_txt",
        "axtree",
        "accessibility_tree_txt",
        "a11y_tree_txt",
        "a11y_tree",
        "a11y",
        "a11y_txt",
        # HTML variants
        "pruned_html",
        "html",
        # Generic (sometimes good, sometimes noisy)
        "text",
        "observation",
        "content",
        "dom",
    )

    # Some BrowserGym builds provide only a structured accessibility tree.
    # Try rendering it before we fall back to json_pruned.
    for ax_key in ("axtree_object", "ax_tree_object", "a11y_tree_object", "a11y_object"):
        ax = obs.get(ax_key)
        if isinstance(ax, (dict, list)):
            rendered = _render_axtree_object_to_text(ax)
            if rendered:
                return rendered, f"{ax_key}_rendered"

    def _search_dict(d: Dict[str, Any], prefix: str) -> Tuple[Optional[str], Optional[str]]:
        for k in preferred_keys:
            if k in d:
                s = _coerce_to_text(d.get(k))
                if s:
                    return s, f"{prefix}{k}"

        md = d.get("metadata")
        if isinstance(md, dict):
            for k in preferred_keys:
                if k in md:
                    s = _coerce_to_text(md.get(k))
                    if s:
                        return s, f"{prefix}metadata.{k}"
        return None, None

    s, k = _search_dict(obs, "")
    if s is not None:
        return s, k

    # Common nested containers in BrowserGym/WebArena observations.
    for container_key in ("dom_object", "dom", "page", "state", "info", "observation"):
        sub = obs.get(container_key)
        if isinstance(sub, dict):
            # Some builds nest the structured accessibility tree under dom_object.
            for ax_key in ("axtree_object", "ax_tree_object", "a11y_tree_object", "a11y_object"):
                ax = sub.get(ax_key)
                if isinstance(ax, (dict, list)):
                    rendered = _render_axtree_object_to_text(ax)
                    if rendered:
                        return rendered, f"{container_key}.{ax_key}_rendered"
            s2, k2 = _search_dict(sub, f"{container_key}.")
            if s2 is not None:
                return s2, k2

    return None, None


def _normalize_obs(obs: Any) -> Tuple[str, Dict[str, Any]]:
    """Best-effort conversion of BrowserGym observations into a text blob.

    Returns (text, meta) where meta may include url/title/etc if present.
    """
    meta: Dict[str, Any] = {}

    # BrowserGym often returns a dict with keys like "axtree_txt"/"text", "url", "title".
    if isinstance(obs, dict):
        for k in ("url", "title", "page_title", "time", "step"):
            if k in obs and obs[k] is not None:
                meta[k] = obs[k]

        # Debugging helpers: record available keys (kept small for trace logs)
        try:
            meta["obs_top_keys"] = sorted(list(obs.keys()))[:50]
            domobj = obs.get("dom_object")
            if isinstance(domobj, dict):
                meta["obs_dom_keys"] = sorted(list(domobj.keys()))[:50]
                # Extract dom_object.strings (can contain star histogram labels/counts).
                try:
                    strings = domobj.get("strings")
                    if isinstance(strings, list):
                        buf: list[str] = []
                        total = 0
                        limit = 80000
                        for s in strings:
                            if not isinstance(s, str):
                                continue
                            ss = s.strip()
                            if not ss:
                                continue
                            buf.append(ss)
                            total += len(ss) + 1
                            if total >= limit:
                                break
                        if buf:
                            meta["dom_strings_text"] = "\n".join(buf)
                            meta["dom_strings_truncated"] = (len(buf) < len(strings))
                            meta["dom_strings_count"] = len(strings)
                except Exception:
                    pass
        except Exception:
            pass

        # Prefer BrowserGym's accessibility-tree text view when available.
        # Even when we find a good text view, we still try to extract a compact
        # actionable-id list for the prompt (helps the model latch onto bids).
        best, best_key = _best_text_from_obs_dict(obs)
        if best is not None:
            meta["obs_text_source"] = best_key
            text_view = best
        else:
            text_view = None

        # Extract a small actionable id list (helps when bids are numeric browsergym_id values).
        try:
            # Surface more candidate bids; shopping pages can have many named links.
            meta["actionable_elements"] = _summarize_actionable_elements(obs, max_items=60)
        except Exception:
            pass

        if text_view is not None:
            return text_view, meta

        # Fallback: stringify dict (but keep it readable) -- prune huge fields first.
        meta["obs_text_source"] = "json_pruned"
        pruned = _prune_obs_dict(obs)
        return json.dumps(pruned, ensure_ascii=False, indent=2, default=_json_default), meta

    if isinstance(obs, str):
        return obs, meta

    # Fallback
    try:
        return str(obs), meta
    except Exception:
        return "<unserializable observation>", meta


def _extract_url(meta: Dict[str, Any]) -> Optional[str]:
    url = meta.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def _ensure_chat_messages_nonempty(env: Any) -> None:
    """Work around a BrowserGym bug where env.post_step indexes chat.messages[-1] without checking emptiness.

    Some BrowserGym versions may raise IndexError in `_wait_for_user_message()` if
    `chat.messages` is empty. We seed it with a dummy system message when needed.
    """
    try:
        base = getattr(env, "unwrapped", env)
        chat = getattr(base, "chat", None) or getattr(env, "chat", None)
        if chat is None:
            return
        msgs = getattr(chat, "messages", None)
        if msgs is None:
            chat.messages = []
            msgs = chat.messages
        if isinstance(msgs, list) and len(msgs) == 0:
            msgs.append({"role": "system", "content": ""})
    except Exception:
        return



def _register_browsergym_benchmark(env_id: str) -> None:
    """Best-effort import to ensure the benchmark environments are registered."""
    try:
        if env_id.startswith("browsergym/webarena") or env_id.startswith("browsergym/visualwebarena"):
            import browsergym.webarena  # type: ignore  # noqa: F401
            # VisualWebArena registration may live in a separate module depending on version.
            try:
                import browsergym.visualwebarena  # type: ignore  # noqa: F401
            except Exception:
                pass
        elif env_id.startswith("browsergym/workarena"):
            import browsergym.workarena  # type: ignore  # noqa: F401
        elif env_id.startswith("browsergym/miniwob"):
            import browsergym.miniwob  # type: ignore  # noqa: F401
        elif env_id.startswith("browsergym/openended"):
            import browsergym.core  # type: ignore  # noqa: F401
    except Exception:
        # If the import fails, gym.make will raise a clearer error.
        return




def _patch_webarena_task_accept_task_configs() -> None:
    """Monkey-patch BrowserGym's GenericWebArenaTask to accept task_configs.

    Some BrowserGym versions do not yet support passing custom task configs via
    task_kwargs (i.e., GenericWebArenaTask.__init__(..., task_configs=...)).
    WebChoreArena relies on custom per-task configs, so we patch at runtime
    (without editing site-packages) for compatibility.
    """
    try:
        import browsergym.webarena.task as _t  # type: ignore
        cls = getattr(_t, 'GenericWebArenaTask', None)
        if cls is None:
            return
        if getattr(cls, '_goc_patched_task_configs', False):
            return
        orig_init = cls.__init__

        def __init__(
            self,
            *args,
            task_configs=None,
            config_file=None,
            config_path=None,
            task_config_file=None,
            **kwargs,
        ):
            """Compat shim for multiple BrowserGym/WebArena variants.

            We accept custom task configs passed via:
              - task_configs: list[dict] | dict | json-str
              - config_file/config_path/task_config_file: path to JSON (list or dict)

            The upstream GenericWebArenaTask typically does *not* accept these yet,
            so we strip them before calling the original __init__ and then override
            self.task_configs.
            """
            # Also accept these keys if passed purely via kwargs.
            if task_configs is None and "task_configs" in kwargs:
                task_configs = kwargs.pop("task_configs")
            if config_file is None and "config_file" in kwargs:
                config_file = kwargs.pop("config_file")
            if config_path is None and "config_path" in kwargs:
                config_path = kwargs.pop("config_path")
            if task_config_file is None and "task_config_file" in kwargs:
                task_config_file = kwargs.pop("task_config_file")

            orig_init(self, *args, **kwargs)

            # Prefer explicit task_configs. Otherwise load from a config file path.
            tc = task_configs
            if tc is None:
                path = config_file or config_path or task_config_file
                if isinstance(path, str) and path.strip():
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            tc = json.load(f)
                    except Exception:
                        tc = None

            if tc is None:
                return

            # Normalize to a list of configs.
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except Exception:
                    tc = [tc]
            if isinstance(tc, dict):
                tc = [tc]

            # At this point tc should be a list of configs.
            self.task_configs = tc

        cls.__init__ = __init__  # type: ignore
        cls._goc_patched_task_configs = True
    except Exception:
        return
def _resolve_env_id_for_task(cfg: "BrowserGymRunConfig", task: Dict[str, Any]) -> str:
    """Resolve the gym env id for a given task.

    BrowserGym WebArena environments are typically registered as:
        browsergym/webarena.<task_id>

    For custom tasks (e.g., WebChoreArena/WCA), prefer a fixed *wrapper* env_id
    (often "browsergym/webarena") and provide the per-task config via task_kwargs
    or env.reset(options=...).
    """
    task_id = task.get("task_id", None)
    if cfg.env_id_template:
        try:
            return cfg.env_id_template.format(task_id=task_id)
        except Exception:
            return cfg.env_id_template

    env_id = cfg.env_id

    # If env_id already pins a specific built-in task (e.g., browsergym/webarena.123), keep it.
    if re.search(r"\.[0-9]+$", env_id):
        return env_id

    # Heuristic: if this is a WebArena-like prefix and task_id is present, append it.
    if task_id is not None and isinstance(task_id, (int, str)):
        if env_id.startswith("browsergym/webarena") and not env_id.startswith("browsergym/webarena."):
            return f"{env_id}.{task_id}"
        if env_id.startswith("browsergym/visualwebarena") and not env_id.startswith("browsergym/visualwebarena."):
            return f"{env_id}.{task_id}"

    return env_id


def _build_webarena_task_config(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a WebArena-style task config dict from a WebChoreArena record.

    BrowserGym's WebArena benchmark typically selects tasks via a *config file*
    (a JSON dict per task) or via `task_kwargs={"task_configs": [...]}` (newer versions).
    WebChoreArena already provides per-task configs; we just normalize a few
    known key variants and (when possible) resolve paths.

    Returns None when the task record doesn't look like a WebArena-style config.
    """
    if not isinstance(task, dict):
        return None

    # Heuristic: if it has an instruction and at least one of these fields,
    # treat it as a config record.
    has_instruction = bool(task.get("intent") or task.get("instruction"))
    has_webarena_fields = any(k in task for k in ("sites", "start_url", "storage_state", "strage_state", "eval"))
    if not (has_instruction and has_webarena_fields):
        return None
    cfg = dict(task)

    # Substitute placeholders like '__SHOPPING__' across the whole config (including nested fields).
    cfg = _substitute_placeholders_deep(cfg)

    # Substitute placeholders like "__SHOPPING__" with actual WA_* base URLs.
    for key in ("start_url", "start_url_lite"):
        if key in cfg:
            cfg[key] = _substitute_placeholders(cfg.get(key))

    # Normalize instruction key
    if "intent" not in cfg and "instruction" in cfg:
        cfg["intent"] = cfg.get("instruction")

    # Normalize historical typo in some JSONs
    if "storage_state" not in cfg and "strage_state" in cfg:
        cfg["storage_state"] = cfg.get("strage_state")

    # Resolve storage_state path if it exists on disk.
    ss = cfg.get("storage_state")
    if isinstance(ss, str) and ss.strip():
        p = Path(ss)
        if not p.is_absolute():
            # Try relative to CWD first.
            if p.exists():
                cfg["storage_state"] = str(p.resolve())
            else:
                # Optional helper: allow WCA_ROOT to be set to the WebChoreArena repo root.
                root = os.environ.get("WCA_ROOT")
                if root:
                    p2 = Path(root) / ss
                    if p2.exists():
                        cfg["storage_state"] = str(p2.resolve())
        else:
            if p.exists():
                cfg["storage_state"] = str(p)

    # Add minimal defaults required by some BrowserGym WebArena task implementations.
    # Older versions access certain keys directly (e.g., config["geolocation"]).
    cfg = _ensure_webarena_required_defaults(cfg)

    return cfg


def _ensure_webarena_required_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Add minimal defaults required by some BrowserGym WebArena task implementations.

    In some BrowserGym versions, WebArena tasks call Playwright context setters like
    `set_geolocation(self.config["geolocation"])` without guarding for missing keys.
    WebChoreArena task records may omit these fields.
    """
    if not isinstance(cfg, dict):
        return cfg

    # Default to a reasonable US location (SF) unless provided.
    if "geolocation" not in cfg or cfg.get("geolocation") in (None, ""):
        cfg["geolocation"] = {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 10}

    # Common optional defaults (safe if ignored by a given BrowserGym version)
    cfg.setdefault("locale", "en-US")
    cfg.setdefault("timezone_id", "America/Los_Angeles")

    return cfg



@dataclass
class BrowserGymRunConfig:
    env_id: str
    env_id_template: Optional[str] = None
    max_steps: int = 50
    obs_truncate_chars: int = 4000
    store_full_obs: bool = True
    new_env_per_task: bool = False
    task_config_mode: str = "auto"  # auto|task_kwargs|config_file|none
    reset_option_key: str = "config_file"
    auto_wait_noop: bool = False  # if task requires extra wait, insert noop() after each action
    auto_wait_on_no_change: bool = True  # if action triggers async update but obs signature doesn't change, wait 1 step
    auto_goto_start_url: bool = True  # if Start URL is provided, do a one-shot goto before LLM steps

    # --- loop / revisit guard ---
    loop_guard: bool = True
    loop_guard_force_action: bool = True
    loop_guard_repeat_threshold: int = 2
    loop_guard_noop_threshold: int = 2
    loop_guard_ttl: int = 10

    # --- WebArena-specific helpers (sane defaults) ---
    # Auto-recover if we land on common dead-end pages (privacy/cookie/terms/policy).
    auto_recover_bad_pages: bool = True
    # Add static avoid-actions (e.g., don't click Privacy/Cookie links).
    add_static_avoid_actions: bool = True
    # For review-distribution tasks, do one bootstrap scroll after navigating to the product page.
    auto_review_bootstrap_scroll: bool = True
    review_bootstrap_scroll_dy: int = 1200

    # If we can confidently read the full 1–5 star distribution, auto-submit it via send_msg_to_user().
    auto_submit_review_distribution: bool = True

    # --- tracing/debug ---
    trace_dir: Optional[str] = None
    # Optional tag appended to trace file name (e.g., method name like GoC/FullHistory).
    trace_tag: Optional[str] = None
    trace_include_prompt: bool = False
    trace_prompt_chars: int = 2000
    trace_include_obs: bool = False
    trace_obs_chars: int = 2000
    # Prompt pieces
    system_prompt: str = DEFAULT_BROWSERGYM_SYSTEM_PROMPT
    # Action grammar shown to the LLM.
    # NOTE: BrowserGym actions are stringified primitives like "click('a46')".
    action_guidelines: str = DEFAULT_BROWSERGYM_ACTION_GUIDELINES


def run_browsergym_tasks(
    *,
    tasks: List[Dict[str, Any]],
    llm_client: Any,
    mem: Any,
    cfg: BrowserGymRunConfig,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a list of WebArena-like tasks through BrowserGym.

    Parameters
    ----------
    tasks:
        Each task should at least include:
          - task_id (optional, used for logging)
          - intent (instruction string)
        If you plan to create custom BrowserGym tasks from configs, include
        fields your environment wrapper expects.
    llm_client:
        Any object exposing .complete(messages: List[dict], force_json=False)
        compatible with src/llm_openai.py client.
    mem:
        One of our MemoryManager implementations (FullHistoryMemory, GoCMemory, ...)
    cfg:
        Runner configuration.
    env_kwargs:
        Passed to gym.make(env_id, **env_kwargs).
    """

    # Import lazily so default installs (synthetic benchmark) stay lightweight.
    import gymnasium as gym  # type: ignore

    # BrowserGym's WebArena integration has drifted across versions.
    # Patch once up-front so both task_kwargs and config_file injection modes work.
    _patch_webarena_task_accept_task_configs()

    def _make_env_with_config_file(env_id: str, config_path: str) -> Any:
        """Try to construct a BrowserGym WebArena env such that the task reads a custom config file.

        Different BrowserGym versions accept different kwargs (often via `task_kwargs`).
        We try a few common keys and fall back to a plain env.
        """
        _register_browsergym_benchmark(env_id)
        _patch_webarena_task_accept_task_configs()
        _patch_webarena_task_accept_task_configs()

        # Try common task_kwargs keys first.
        candidate_keys = ["config_file", "config_path", "task_config_file", cfg.reset_option_key]
        for key in candidate_keys:
            try:
                merged = dict(env_kwargs)
                tk = dict(merged.pop("task_kwargs", {}) or {})
                tk[key] = config_path
                merged["task_kwargs"] = tk
                return gym.make(env_id, **merged)
            except TypeError:
                continue
            except Exception:
                # Some versions may throw other errors; keep trying.
                continue

        # As a fallback, construct without custom kwargs.
        return gym.make(env_id, **env_kwargs)

    def _resolve_registered_env_id(env_id: str) -> str:
        """Resolve a *registered* env id.

        Some BrowserGym installs register WebArena envs only as
        `browsergym/webarena.<N>` (no plain `browsergym/webarena`).
        If the requested id is not registered and looks like a base id,
        we select the smallest numeric variant available.
        """
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            pass

        # If already versioned, keep it (let gym.make raise a clearer error).
        if re.search(r"\.[0-9]+$", env_id):
            return env_id

        prefix = env_id + "."
        try:
            keys = list(gym.registry.keys())
        except Exception:
            keys = []
        candidates = [k for k in keys if isinstance(k, str) and k.startswith(prefix)]
        if not candidates:
            return env_id

        def _key_fn(k: str) -> int:
            suf = k.rsplit(".", 1)[-1]
            return int(suf) if suf.isdigit() else 10**9

        candidates.sort(key=_key_fn)
        return candidates[0]

    env_kwargs = env_kwargs or {}

    results: List[Dict[str, Any]] = []

    # We allow env_id to vary per task (e.g., browsergym/webarena.<task_id>).
    env = None
    current_env_id: Optional[str] = None

    tmp_config_files: List[str] = []

    # --- Trace output ---
    # User preference: keep a stable folder (default: ./trace) and timestamp the *file name*.
    # This avoids confusion when comparing multiple runs.
    trace_enabled = bool(cfg.trace_dir) or bool(cfg.trace_include_prompt) or bool(cfg.trace_include_obs)
    trace_base_dir: Optional[Path] = None
    trace_run_ts = time.strftime("%Y%m%d_%H%M%S")
    # Capture the exact command used to run this script (useful for debugging/repro).
    try:
        run_command = " ".join(shlex.quote(x) for x in sys.argv)
    except Exception:
        run_command = None
    if trace_enabled:
        base_dir = Path(cfg.trace_dir) if cfg.trace_dir else Path("trace")
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            trace_base_dir = base_dir
        except Exception:
            trace_base_dir = None

    def _trace_write(fp, ev: Dict[str, Any]):
        """Write one trace event (JSONL). Never throws."""
        if fp is None:
            return
        try:
            fp.write(json.dumps(ev, ensure_ascii=False, default=_json_default) + "\n")
            fp.flush()
        except Exception:
            pass


    try:
        for i, task in enumerate(tasks):
            # Build a per-task WebArena-style config dict if present.
            task_config = _build_webarena_task_config(task)
            seed = task.get("seed")

            # Track the temp config file path (if any) so we can retry with a different injection method.
            tmp_config_path: Optional[str] = None

            # Decide how to inject the custom config (if any) *before* selecting env_id.
            mode = (cfg.task_config_mode or "auto").lower()
            use_task_kwargs = mode in ("auto", "task_kwargs") and task_config is not None
            use_config_file = mode in ("auto", "config_file") and task_config is not None

            # IMPORTANT: when using custom per-task configs (WebChoreArena/WCA),
            # we generally want a *fixed* WebArena wrapper env_id (e.g., "browsergym/webarena")
            # rather than a baked-in task env like "browsergym/webarena.310".
            if task_config is not None and mode != "none":
                if cfg.env_id_template:
                    try:
                        env_id_task = cfg.env_id_template.format(task_id=task.get("task_id"))
                    except Exception:
                        env_id_task = cfg.env_id_template
                else:
                    env_id_task = cfg.env_id
            else:
                env_id_task = _resolve_env_id_for_task(cfg, task)

            # Resolve to an actually registered env id (some BrowserGym versions
            # only register browsergym/webarena.<N>, not browsergym/webarena).
            env_id_task_requested = env_id_task
            _register_browsergym_benchmark(env_id_task)
            env_id_task = _resolve_registered_env_id(env_id_task)

            trace_fp = None
            trace_file: Optional[str] = None
            if trace_base_dir is not None:
                try:
                    raw_name = str(task.get("task_id", f"task_{i}"))
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
                    tag = (cfg.trace_tag or "").strip()
                    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag) if tag else ""
                    suffix = f"_{tag}" if tag else ""
                    trace_path = trace_base_dir / f"trace_{safe_name}_{trace_run_ts}{suffix}.jsonl"
                    trace_file = str(trace_path)
                    trace_fp = trace_path.open("w", encoding="utf-8")
                except Exception:
                    trace_fp = None

            _trace_write(trace_fp, {
                "type": "task_start",
                "task_id": task.get("task_id", f"task_{i}"),
                "env_id_task": env_id_task,
                "env_id_task_requested": env_id_task_requested,
                "trace_tag": cfg.trace_tag,
                "trace_dir": str(trace_base_dir) if trace_base_dir is not None else None,
                "trace_file": trace_file,
                "trace_file_pattern": "trace_{task_id}_{timestamp}_{trace_tag}.jsonl",
                "run_command": run_command,
                "llm_model": getattr(llm_client, "model", None),
                "task_config_mode": cfg.task_config_mode,
                "reset_option_key": cfg.reset_option_key,
                "seed": seed,
                "sites": (task_config or {}).get("sites"),
                "start_url": (task_config or {}).get("start_url"),
                "storage_state": (task_config or {}).get("storage_state"),
            })

            obs = None
            info = {}

            if use_task_kwargs:
                # Easiest way to guarantee the correct task is selected: create a fresh env per task.
                if env is not None:
                    env.close()
                env = None
                current_env_id = None

                try:
                    _register_browsergym_benchmark(env_id_task)
                    _patch_webarena_task_accept_task_configs()

                    merged = dict(env_kwargs)
                    tk = dict(merged.pop("task_kwargs", {}) or {})
                    tk.update({"task_configs": [task_config]})
                    merged["task_kwargs"] = tk
                    env = gym.make(env_id_task, **merged)
                    current_env_id = env_id_task
                    obs, info = env.reset(seed=seed)
                except TypeError:
                    # Fallback to config_file mode if supported (especially for older BrowserGym versions).
                    if mode == "task_kwargs":
                        raise
                    # IMPORTANT: env may have been created with unsupported task_kwargs; recreate it.
                    try:
                        if env is not None:
                            env.close()
                    except Exception:
                        pass
                    env = None
                    current_env_id = None
                    use_task_kwargs = False
                    use_config_file = True

            if obs is None and use_config_file:
                # Always create a fresh env per task for config-file injection.
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                env = None
                current_env_id = None

                # Write a temp config json.
                tmpf = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
                # Many BrowserGym WebArena task loaders expect a JSON *list* of task configs.
                # We write a singleton list for compatibility.
                json.dump([task_config], tmpf)
                tmpf.flush()
                tmpf.close()
                tmp_config_files.append(tmpf.name)
                tmp_config_path = tmpf.name

                _trace_write(trace_fp, {
                    "type": "task_config",
                    "mode": "config_file",
                    "tmp_config_file": tmpf.name,
                    "task_config": task_config,
                })

                # Prefer passing config file via gym.make(task_kwargs=...) for compatibility,
                # because env.reset(options=...) does not reach the task constructor in some versions.
                env = _make_env_with_config_file(env_id_task, tmpf.name)
                current_env_id = env_id_task
                try:
                    obs, info = env.reset(seed=seed)
                except TypeError:
                    # Fallback: some versions may accept options-based config injection.
                    obs, info = env.reset(seed=seed, options={cfg.reset_option_key: tmpf.name})

            if obs is None:
                # No custom task config provided; just reset normally.
                if env is None or env_id_task != current_env_id:
                    if env is not None:
                        env.close()
                    _register_browsergym_benchmark(env_id_task)
                    env = gym.make(env_id_task, **env_kwargs)
                    current_env_id = env_id_task
                obs, info = env.reset(seed=seed)

            # Seed chat messages to avoid BrowserGym IndexError in post_step (some versions).
            _ensure_chat_messages_nonempty(env)

            mem.reset()

            task_name = task.get("task_id", f"task_{i}")
            instruction = task.get("intent") or task.get("instruction") or ""
            if task_config and task_config.get('start_url_lite'):
                instruction += f"\nStart URL: {task_config.get('start_url_lite')}"

            obs_text, meta = _normalize_obs(obs)
            url = _extract_url(meta)

            # If a custom task_config was provided but reset landed on a different site,
            # it usually means this BrowserGym version ignored reset(options=...) and/or
            # does not support task_kwargs without patching. Try a one-shot retry using
            # patched task_kwargs injection.
            expected_start = (task_config or {}).get('start_url') if task_config else None
            if (task_config is not None and expected_start and url and isinstance(url, str)
                and not url.startswith(str(expected_start))
                and mode in ('auto', 'config_file')):
                _trace_write(trace_fp, {
                    'type': 'reset_mismatch',
                    'expected_start_url': expected_start,
                    'actual_url': url,
                })
                # Retry by reconstructing the env with an explicit config file passed via task_kwargs.
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass

                # Ensure we have a temp config file.
                try:
                    if not tmp_config_path:
                        tmpf2 = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
                        json.dump([task_config], tmpf2)
                        tmpf2.flush(); tmpf2.close()
                        tmp_config_files.append(tmpf2.name)
                        tmp_config_path = tmpf2.name
                except Exception:
                    tmp_config_path = None

                try:
                    if tmp_config_path:
                        env = _make_env_with_config_file(env_id_task, tmp_config_path)
                        current_env_id = env_id_task
                        obs, info = env.reset(seed=seed)
                        _ensure_chat_messages_nonempty(env)
                        obs_text, meta = _normalize_obs(obs)
                        url = _extract_url(meta)
                        _trace_write(trace_fp, {
                            'type': 'reset_retry_done',
                            'url': url,
                            'title': meta.get('title') if isinstance(meta, dict) else None,
                        })
                    else:
                        raise RuntimeError("No tmp config file for retry")
                except Exception as e:
                    _trace_write(trace_fp, {
                        'type': 'reset_retry_failed',
                        'error': str(e),
                    })

            cur_state_sig = _compute_state_sig(url, meta if isinstance(meta, dict) else {}, active_obs=None)
            cur_state_docid = _make_state_docid(url, cur_state_sig)

            _trace_write(trace_fp, {
                "type": "reset_done",
                "url": url,
                "title": meta.get("title") if isinstance(meta, dict) else None,
                "info_keys": list(info.keys()) if isinstance(info, dict) else None,
                "obs_text_source": meta.get("obs_text_source") if isinstance(meta, dict) else None,
                "obs_top_keys": meta.get("obs_top_keys") if isinstance(meta, dict) else None,
                "obs_dom_keys": meta.get("obs_dom_keys") if isinstance(meta, dict) else None,
                "state_sig": cur_state_sig,
                "state_docid": cur_state_docid,
            })

            # Prime memory
            mem.record_msg(f"[TASK] {instruction}")
            # Compute a stable state signature (URL + actionable ids) and attach as a docid,
# so GoC can connect revisits/failures even when the raw obs text is noisy.
            cur_url = _extract_url(meta) if isinstance(meta, dict) else None
            cur_state_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
            cur_state_docid = _make_state_docid(cur_url, cur_state_sig)
            active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)

            # Optional bootstrap: if the task provides a concrete Start URL (often a product page),
            # navigate there once before invoking the LLM. This prevents a common failure mode where
            # the model keeps scrolling on the homepage while the observation barely changes.
            if (
                cfg.auto_goto_start_url
                and task_config
                and isinstance(task_config.get("start_url_lite"), str)
            ):
                target_url = str(task_config.get("start_url_lite"))
                cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                if target_url and isinstance(cur_url, str) and not cur_url.startswith(target_url):
                    safe_target = target_url.replace("'", "%27")
                    boot_action = f"goto('{safe_target}')"
                    _trace_write(trace_fp, {
                        "type": "auto_goto_start_url",
                        "from_url": cur_url,
                        "to_url": target_url,
                        "action": boot_action,
                    })
                    try:
                        mem.record_tool("act", {"action": boot_action}, observation="")
                        obs, reward, terminated, truncated, info = _step_env(env, boot_action)
                        obs_text, meta = _normalize_obs(obs)
                        cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                        cur_state_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                        cur_state_docid = _make_state_docid(cur_url, cur_state_sig)
                        active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                        _trace_write(trace_fp, {
                            "type": "auto_goto_done",
                            "url": _extract_url(meta) if isinstance(meta, dict) else None,
                            "title": meta.get("title") if isinstance(meta, dict) else None,
                        })
                    except Exception as e:
                        _trace_write(trace_fp, {"type": "auto_goto_failed", "error": str(e)})

            # Optional: for review-distribution tasks on Shopping, do a one-shot scroll after landing.
            # This helps reveal the reviews block without the model spending many noop/scroll steps.
            if (
                cfg.auto_review_bootstrap_scroll
                and _is_review_distribution_task(instruction)
                and task_config
                and isinstance(task_config.get("start_url_lite"), str)
            ):
                try:
                    cur_url2 = _extract_url(meta) if isinstance(meta, dict) else None
                    target2 = str(task_config.get("start_url_lite"))
                    if isinstance(cur_url2, str) and target2 and cur_url2.startswith(target2):
                        boot_action = f"scroll(0, {int(cfg.review_bootstrap_scroll_dy)})"
                        _trace_write(trace_fp, {"type": "auto_bootstrap_scroll", "action": boot_action})
                        mem.record_tool("act", {"action": boot_action}, observation="")
                        obs, reward, terminated, truncated, info = _step_env(env, boot_action)
                        obs_text, meta = _normalize_obs(obs)
                        cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                        cur_state_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                        cur_state_docid = _make_state_docid(cur_url, cur_state_sig)
                        active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                except Exception as e:
                    _trace_write(trace_fp, {"type": "auto_bootstrap_scroll_failed", "error": str(e)})

            total_tokens = 0
            total_tool_calls = 0
            done = False
            last_action: Optional[str] = None
            reward: float = 0.0
            start_t = time.time()

            # Loop / revisit guard state.
            prev_obs_sig: Optional[str] = None
            prev_url: Optional[str] = _extract_url(meta) if isinstance(meta, dict) else None
            same_obs_count: int = 0
            failed_actions_by_state: Dict[Tuple[Optional[str], Optional[str]], Set[str]] = collections.defaultdict(set)
            failed_counts: Dict[Tuple[Optional[str], Optional[str], str], int] = collections.defaultdict(int)
            emitted_fail_notes: Set[Tuple[Optional[str], Optional[str], str]] = set()

            # Initialize state signature from the post-reset (and optional auto_goto) observation.
            prev_obs_sig = _compute_state_sig(prev_url, meta if isinstance(meta, dict) else {}, active_obs=active_obs if isinstance(active_obs, str) else None)
            prev_state_docid = _make_state_docid(prev_url, prev_obs_sig)

            for step in range(cfg.max_steps):
                # Current state key (before acting)
                state_key = (prev_url, prev_obs_sig)
                avoid_actions = set(failed_actions_by_state.get(state_key, set()) or set())
                static_avoid = {}
                if cfg.add_static_avoid_actions and isinstance(active_obs, str):
                    try:
                        static_avoid = _derive_static_avoid_actions(active_obs, instruction)
                        avoid_actions |= set(static_avoid.keys())
                    except Exception:
                        static_avoid = {}
                suggested_action = None
                guard_text = None
                if cfg.loop_guard and isinstance(active_obs, str) and avoid_actions:
                    suggested_action = _suggest_action_from_candidates(
                        active_obs=active_obs,
                        instruction=instruction,
                        avoid_actions=set(avoid_actions),
                    )
                    avoid_list = list(sorted(avoid_actions))[:8]
                    static_notes = []
                    if static_avoid:
                        # Show a couple of reasons so the model understands why to avoid.
                        for a, r in list(static_avoid.items())[:3]:
                            static_notes.append(f"{a} ({r})")
                    static_block = ("\nStatic avoids: " + ", ".join(static_notes)) if static_notes else ""
                    guard_text = (
                        f"You are in the same page state (url={prev_url}).\\n"
                        f"Avoid repeating actions that already failed to change the page: {avoid_list}.\\n"
                        + static_block + "\\n"
                        + (f"Suggested next action: ACTION: {suggested_action}.\\n" if suggested_action else "")
                    )
                # Auto-submit: if we can reliably extract the full review distribution, submit it directly.
                action = None
                if cfg.auto_submit_review_distribution and _is_review_distribution_task(instruction) and isinstance(active_obs, str):
                    dist, ev_lines = _extract_review_star_counts_with_evidence(active_obs)
                    # Only submit when we have strong evidence lines (prevents misreading labels like '5 stars' as counts).
                    if dist and ev_lines:
                        ans = f"5: {dist[5]}, 4: {dist[4]}, 3: {dist[3]}, 2: {dist[2]}, 1: {dist[1]}"
                        action = f"send_msg_to_user('{ans}')"
                        _trace_write(trace_fp, {
                            'type': 'auto_submit_review_distribution',
                            'step': step,
                            'url': prev_url,
                            'state_sig': prev_obs_sig,
                            'answer': ans,
                            'action': action,
                            'evidence_lines': ev_lines[:8],
                        })

                if action is None:
                    # Build prompt from *active context*.
                    prompt = _build_prompt(mem, instruction, cfg, guard_text=guard_text)
                    messages = [
                        {'role': 'system', 'content': cfg.system_prompt},
                        {'role': 'user', 'content': prompt},
                    ]

                    llm_out = llm_client.complete(messages=messages, force_json=False)
                    _trace_write(trace_fp, {
                        'type': 'llm',
                        'step': step,
                        'prompt_sha1': hashlib.sha1(prompt.encode('utf-8', 'ignore')).hexdigest(),
                        'prompt_head': (prompt[:cfg.trace_prompt_chars] if cfg.trace_include_prompt else None),
                        'prompt_tail': (prompt[-cfg.trace_prompt_chars:] if cfg.trace_include_prompt else None),
                        'llm_text': llm_out.get('text', ''),
                        'usage': llm_out.get('usage', {}),
                    })
                    total_tokens += int(llm_out.get('usage', {}).get('total_tokens', 0) or 0)

                    action = _parse_action(llm_out.get('text', ''))
                    if action is None:
                        # Treat parse failure as a failure node to avoid repeated loops.
                        mem.record_summary(
                            '[FAIL] Could not parse an action from the model output. Remember to output: ACTION: <action>.',
                            ttl=int(cfg.loop_guard_ttl),
                            docids=[d for d in [prev_url, prev_state_docid] if d],
                        )
                        action = 'noop()'

                # If the model proposes an action we already know is unproductive for this state,
                # optionally override it with a keyword-based suggestion.
                if cfg.loop_guard and action in avoid_actions and isinstance(active_obs, str):
                    if suggested_action is None:
                        suggested_action = _suggest_action_from_candidates(
                            active_obs=active_obs,
                            instruction=instruction,
                            avoid_actions=set(avoid_actions),
                        )
                    _trace_write(trace_fp, {
                        "type": "loop_guard",
                        "step": step,
                        "url": prev_url,
                        "state_sig": prev_obs_sig,
                        "avoid_action": action,
                        "suggested_action": suggested_action,
                        "forced": bool(cfg.loop_guard_force_action and suggested_action),
                    })
                    if cfg.loop_guard_force_action and suggested_action:
                        action = suggested_action

                # Simple loop-break hint for repeated identical actions.
                if last_action is not None and action == last_action:
                    mem.record_summary(
                        "[FAIL] Repeating the same action without progress. Consider a different strategy (click a relevant element id, scroll, or navigate).",
                        ttl=int(cfg.loop_guard_ttl),
                        docids=[d for d in [prev_url, prev_state_docid] if d],
                    )
                last_action = action

                # Record and step
                mem.record_tool("act", {"action": action}, observation="")
                total_tool_calls += 1

                # Ensure chat state isn't empty to avoid IndexError in some BrowserGym versions.
                _ensure_chat_messages_nonempty(env)

                obs, reward, terminated, truncated, info = _step_env(env, action)
                done = bool(terminated or truncated or (isinstance(info, dict) and info.get("done") is True))

                obs_text, meta = _normalize_obs(obs)
                cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                cur_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                cur_state_docid = _make_state_docid(cur_url, cur_sig)
                active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                # Recompute signature with active_obs so SPA content changes are detected.
                cur_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=active_obs if isinstance(active_obs, str) else None)

                
                # If the action triggered an async page update but our signature didn't change,
                # insert a one-step noop() to let the UI settle (common in WebArena Shopping).
                try:
                    no_change_pre = (cur_sig is not None and cur_sig == prev_obs_sig and cur_url == prev_url)
                    if (no_change_pre and cfg.auto_wait_on_no_change and action.strip() != "noop()" and not done):
                        _trace_write(trace_fp, {
                            "type": "auto_wait_on_no_change",
                            "step": step,
                            "from_action": action,
                            "url": cur_url,
                        })
                        mem.record_tool("act", {"action": "noop()"}, observation="")
                        total_tool_calls += 1
                        obs2, reward2, terminated2, truncated2, info2 = _step_env(env, "noop()")
                        done = bool(terminated2 or truncated2 or (isinstance(info2, dict) and info2.get("done") is True))
                        reward = reward2
                        terminated = terminated2
                        truncated = truncated2
                        info = info2
                        obs_text, meta = _normalize_obs(obs2)
                        cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                        cur_sig2 = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                        cur_state_docid = _make_state_docid(cur_url, cur_sig2)
                        active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                        cur_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=active_obs if isinstance(active_obs, str) else None)
                        total_tool_calls += 1  # count obs record
                except Exception as e:
                    _trace_write(trace_fp, {"type": "auto_wait_on_no_change_failed", "step": step, "error": str(e)})

# Stuck detection: if the visible observation does not change across steps,
                # discourage repeating scroll/noop loops.

                # Auto-recover if we landed on a dead-end policy/cookie page.
                if (
                    cfg.auto_recover_bad_pages
                    and isinstance(cur_url, str)
                    and _BAD_PAGE_RE.search(cur_url)
                ):
                    _trace_write(trace_fp, {
                        "type": "auto_recover_bad_page",
                        "step": step,
                        "bad_url": cur_url,
                        "from_action": action,
                    })
                    # Mark the triggering action as failed for the *previous* state.
                    try:
                        failed_actions_by_state[state_key].add(action)
                    except Exception:
                        pass
                    mem.record_summary(
                        f"[FAIL] Landed on a policy/cookie page ({cur_url}). Go back and avoid that link.",
                        ttl=int(cfg.loop_guard_ttl),
                        docids=[d for d in [cur_url, cur_state_docid] if d],
                    )
                    try:
                        rec_action = "go_back()"
                        mem.record_tool("act", {"action": rec_action}, observation="")
                        obs, reward, terminated, truncated, info = _step_env(env, rec_action)
                        done = bool(terminated or truncated or (isinstance(info, dict) and info.get("done") is True))
                        obs_text, meta = _normalize_obs(obs)
                        cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                        cur_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                        cur_state_docid = _make_state_docid(cur_url, cur_sig)
                        active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                        _trace_write(trace_fp, {
                            "type": "auto_recover_done",
                            "step": step,
                            "url": cur_url,
                        })
                    except Exception as e:
                        _trace_write(trace_fp, {
                            "type": "auto_recover_failed",
                            "step": step,
                            "error": str(e),
                        })

                # Determine whether the action changed the visible state.
                no_change = (cur_sig is not None and cur_sig == prev_obs_sig and cur_url == prev_url)
                if no_change:
                    same_obs_count += 1
                else:
                    same_obs_count = 0

                # Record failed actions for this exact state signature.
                if cfg.loop_guard and no_change:
                    k = (prev_url, prev_obs_sig, action)
                    failed_counts[k] = int(failed_counts.get(k, 0)) + 1
                    thr = int(cfg.loop_guard_noop_threshold) if action.strip() == "noop()" else int(cfg.loop_guard_repeat_threshold)
                    # goto() to the same url is almost always wasted if the state didn't change.
                    if action.strip().startswith("goto("):
                        thr = 1
                    if failed_counts[k] >= thr:
                        failed_actions_by_state[state_key].add(action)
                        if k not in emitted_fail_notes:
                            suggestion = _suggest_action_from_candidates(
                                active_obs=active_obs,
                                instruction=instruction,
                                avoid_actions=set(failed_actions_by_state[state_key]),
                            )
                            note = (
                                f"[FAIL] Action '{action}' did not change the page state (url={prev_url}). "
                                f"Avoid repeating it in this state." +
                                (f" Suggested next: {suggestion}." if suggestion else "")
                            )
                            mem.record_summary(
                                note,
                                ttl=int(cfg.loop_guard_ttl),
                                docids=[d for d in [prev_url, prev_state_docid] if d],
                            )
                            emitted_fail_notes.add(k)

                if cfg.loop_guard and same_obs_count >= int(cfg.loop_guard_noop_threshold):
                    mem.record_summary(
                        "[FAIL] Observation did not change after multiple steps. "
                        "Stop repeating noop/scroll. Click a relevant element id (e.g., reviews) or navigate explicitly.",
                        ttl=int(cfg.loop_guard_ttl),
                        docids=[d for d in [cur_url, cur_state_docid] if d],
                    )

                prev_obs_sig = cur_sig
                prev_url = cur_url
                prev_state_docid = cur_state_docid
                _trace_write(trace_fp, {
                    "type": "env_step",
                    "step": step,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": bool(done),
                    "url": _extract_url(meta) if isinstance(meta, dict) else None,
                    "title": meta.get('title') if isinstance(meta, dict) else None,
                    "obs_text_source": meta.get("obs_text_source") if isinstance(meta, dict) else None,
                    "obs_top_keys": meta.get("obs_top_keys") if isinstance(meta, dict) else None,
                    "obs_dom_keys": meta.get("obs_dom_keys") if isinstance(meta, dict) else None,
                    "obs_sha1": (hashlib.sha1(obs_text.encode('utf-8', 'ignore')).hexdigest() if cfg.trace_include_obs else None),
                    "active_obs_sha1": (cur_sig if cfg.trace_include_obs else None),
                    "obs_head": obs_text[: cfg.trace_obs_chars] if cfg.trace_include_obs else None,
                    "active_obs_head": (active_obs[: cfg.trace_obs_chars] if cfg.trace_include_obs else None),
                    "obs_has_bid": (_has_any_element_id(active_obs) if isinstance(active_obs, str) else False),
                    "same_obs_count": same_obs_count,
                })
                try:
                    evs = mem.drain_events() if hasattr(mem, 'drain_events') else []
                    if evs:
                        _trace_write(trace_fp, {'type': 'mem_events', 'step': step, 'events': evs})
                except Exception:
                    pass
                total_tool_calls += 1

                # Some WebArena/WebChoreArena tasks require an explicit wait for state to settle.
                # NOTE: Only record an extra obs() if we actually take the noop() step; otherwise avoid duplicating obs.
                if (not done) and cfg.auto_wait_noop and task.get("required_wait"):
                    _trace_write(trace_fp, {"type": "auto_wait_required_wait", "step": step, "url": cur_url})
                    mem.record_tool("act", {"action": "noop()"}, observation="")
                    total_tool_calls += 1
                    obs, reward, terminated, truncated, info = _step_env(env, "noop()")
                    done = bool(terminated or truncated or (isinstance(info, dict) and info.get("done") is True))
                    obs_text, meta = _normalize_obs(obs)
                    cur_url = _extract_url(meta) if isinstance(meta, dict) else None
                    cur_sig2 = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=None)
                    cur_state_docid = _make_state_docid(cur_url, cur_sig2)
                    active_obs = _record_obs(mem, obs_text, meta, cfg, instruction=instruction, state_docid=cur_state_docid)
                    # Recompute signature with active_obs so SPA content changes are detected.
                    cur_sig = _compute_state_sig(cur_url, meta if isinstance(meta, dict) else {}, active_obs=active_obs if isinstance(active_obs, str) else None)
                    total_tool_calls += 1  # count obs record
                    prev_obs_sig = cur_sig
                    prev_url = cur_url
                    prev_state_docid = cur_state_docid

                if done:
                    break

            elapsed = time.time() - start_t
            _trace_write(trace_fp, {
                "type": "task_end",
                "task_id": task_name,
                "done": bool(done),
                "reward": float(reward),
                "elapsed_s": float(elapsed),
                "total_tokens": int(total_tokens),
                "tool_calls": int(total_tool_calls),
                "final_url": _extract_url(meta) if isinstance(meta, dict) else None,
            })
            if trace_fp is not None:
                try:
                    trace_fp.close()
                except Exception:
                    pass
            results.append(
                {
                    "task_id": task_name,
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(
                        (done and float(reward) > 0.5)
                        or (isinstance(info, dict) and (info.get("success") is True or info.get("task_success") is True))
                    ),
                    "steps": step + 1,
                    "total_tokens": total_tokens,
                    "tool_calls": total_tool_calls,
                    "peak_active_tokens": getattr(mem, "active_tokens", lambda: None)(),
                    "final_url": _extract_url(meta) if isinstance(meta, dict) else None,
                    "elapsed_s": elapsed,
                }
            )

    finally:
        # Cleanup temp config files (if we used config_file reset).
        for p in tmp_config_files:
            try:
                os.unlink(p)
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return results


def _record_obs(mem: Any, obs_text: str, meta: Dict[str, Any], cfg: BrowserGymRunConfig, *, instruction: Optional[str] = None, state_docid: Optional[str] = None):
    url = _extract_url(meta)
    header: list[str] = []
    if url:
        header.append(f"url={url}")
    if meta.get("title"):
        header.append(f"title={meta['title']}")
    header_txt = ("(" + ", ".join(header) + ")\n") if header else ""

    elems = meta.get("actionable_elements")
    if isinstance(elems, list) and elems:
        header_txt += "Actionable element ids (sample):\n" + "\n".join(elems[:40]) + "\n\n"

    focus_block = ""
    if instruction:
        try:
            focus = _focus_lines_for_task(obs_text, instruction)
            if focus:
                focus_block = "Focus (goal-related lines):\n" + focus + "\n\n"
        except Exception:
            focus_block = ""

    dom_focus_block = ""
    if instruction:
        try:
            dom_txt = meta.get("dom_strings_text")
            if isinstance(dom_txt, str) and dom_txt.strip():
                dom_focus = _dom_focus_lines_for_task(dom_txt, instruction)
                if dom_focus:
                    dom_focus_block = "DOM Focus (filtered):\n" + dom_focus + "\n\n"
        except Exception:
            dom_focus_block = ""

    active_text = header_txt + focus_block + dom_focus_block + _extract_actionable_obs(obs_text, cfg.obs_truncate_chars)
    storage_text = obs_text if cfg.store_full_obs else None
    docids = [url] if url else []
    if state_docid:
        docids.append(state_docid)
    mem.record_tool("obs", {"meta": header}, observation=active_text, docids=docids, storage_text=storage_text)
    return active_text

def _build_prompt(mem: Any, instruction: str, cfg: BrowserGymRunConfig, guard_text: Optional[str] = None) -> str:
    active = mem.get_active_text() if hasattr(mem, "get_active_text") else ""
    guard_block = f"\n\nLoopGuard:\n{guard_text}" if guard_text else ""
    completion_hint = ""
    if _is_review_distribution_task(instruction):
        completion_hint = (
            "\n\nCompletion:\n"
            "When you have the final review-rating distribution, respond with: "
            "ACTION: send_msg_to_user('5: <n>, 4: <n>, 3: <n>, 2: <n>, 1: <n>')."
        )
    return (
        f"Task: {instruction}\n\n"
        f"Memory (most relevant context):\n{active}\n\n"
        f"{guard_block}{completion_hint}\n\n"
        f"Action guidelines:\n{cfg.action_guidelines}\n\n"
        "Now decide the next action."
    )


_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_action(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action = m.group(1).strip()
    # Allow a bare "stop" as a completion action.
    if action.lower() == "stop":
        return "stop"
    return action

_GOTO_URL_RE = re.compile(r"^goto\(\s*['\"]([^'\"]+)['\"]\s*\)\s*$")

def _origin(url: str) -> Optional[str]:
    try:
        parts = urllib.parse.urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return None
        return f"{parts.scheme}://{parts.netloc}"
    except Exception:
        return None

def _collect_allowed_origins(task_config: Optional[Dict[str, Any]]) -> List[str]:
    """Collect allowed URL origins for goto() to prevent hallucinated/external navigation."""
    origins: List[str] = []

    def add(u: Any):
        if isinstance(u, str) and u.strip():
            o = _origin(u.strip())
            if o and o not in origins:
                origins.append(o)

    if isinstance(task_config, dict):
        add(task_config.get("start_url"))
        add(task_config.get("start_url_lite"))

    # Known WebArena site env vars (may include paths; origin extracts scheme+netloc).
    for k in ["WA_SHOPPING","WA_SHOPPING_ADMIN","WA_REDDIT","WA_GITLAB","WA_WIKIPEDIA","WA_MAP","WA_HOMEPAGE"]:
        add(os.environ.get(k))

    return origins

def _is_allowed_goto(url: str, allowed_origins: List[str]) -> bool:
    o = _origin(url)
    if not o:
        return False
    return o in set(allowed_origins)

def _sanitize_action(
    action: str,
    *,
    task_config: Optional[Dict[str, Any]],
    prev_url: Optional[str],
    last_action: Optional[str],
    same_obs_count: int,
    suggested_action: Optional[str],
    active_obs: Optional[str],
) -> str:
    """Sanitize/repair actions to reduce common WebArena failure modes.

    - Prevent repeated noop loops.
    - Prevent hallucinated external goto URLs; redirect to Start URL instead.
    """
    a = (action or "").strip()

    # Prevent repeated noop loops.
    if a == "noop()" and last_action == "noop()":
        if suggested_action:
            return suggested_action
        if isinstance(active_obs, str) and _has_any_element_id(active_obs):
            cand = _suggest_action_from_candidates(
                active_obs=active_obs,
                instruction="",
                avoid_actions={"noop()"},
            )
            if cand:
                return cand
        return "scroll(0, 900)"

    # Prevent hallucinated external goto URLs.
    m = _GOTO_URL_RE.match(a)
    if m:
        target = m.group(1)
        allowed = _collect_allowed_origins(task_config)
        if allowed and (not _is_allowed_goto(target, allowed)):
            fallback = None
            if isinstance(task_config, dict):
                fallback = task_config.get("start_url_lite") or task_config.get("start_url")
            if isinstance(fallback, str) and fallback.strip():
                safe = fallback.strip().replace("'", "%27")
                return f"goto('{safe}')"
            if suggested_action:
                return suggested_action
            return "noop()"

    return a


def _step_env(env: Any, action: Any):
    """Call env.step with some robustness across action formats.

    Notes
    -----
    - BrowserGym expects actions as strings like "click('a46')".
    - Some agent prompts use convenience tokens like "stop" or "wait()"; we map them to noop().
    """
    if isinstance(action, str):
        a = action.strip()
        al = a.lower()
        if al == "stop" or al.startswith("stop"):
            action = "noop()"
        elif al in {"wait", "wait()"}:
            action = "noop()"

    try:
        return env.step(action)
    except TypeError:
        # Some envs expect list[str] or dict.
        return env.step([action])
