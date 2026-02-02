#!/usr/bin/env python
"""Prepare FEVER into this repo's 'fever_prepared' format.

Input
- FEVER jsonl (e.g., data/fever/fever-data/dev.jsonl)
- Wikipedia docstore built by scripts/build_fever_wiki_sqlite.py

Output (jsonl or jsonl.gz)
Each row contains a small task-local doc set:
  {
    "id": ...,
    "claim": ...,
    "label": "supports"|"refutes"|"not_enough_info",
    "docs": [ {"docid","title","content"...}, ... ],
    "evidence_titles": ["TitleA","TitleB",...]
  }

This file can be consumed by the benchmark: src/benchmarks/fever_prepared.py

Recommended workflow
1) Build docstore (full or mini)
   python scripts/build_fever_wiki_sqlite.py --wiki_dir data/fever/wiki/wiki-pages --out_db data/fever/wiki/wiki.sqlite
2) Prepare dev (small doc sets)
   python scripts/prepare_fever_from_wiki.py --fever data/fever/fever-data/dev.jsonl --wiki_db data/fever/wiki/wiki.sqlite \
       --out data/fever/fever_prepared/dev_prepared.jsonl.gz --docs_per_task 20 --gold_titles_n 2 --seed 7
3) Run sweep
   python run_sweep.py --preset fever_wiki_dev_goc_signal --out_dir sweeps/fever_dev_goc
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _open_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _dump_jsonl(path: str, rows: Sequence[Dict[str, Any]]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".gz":
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _norm_label(label: str) -> str:
    s = (label or "").strip().lower()
    if s.startswith("support"):
        return "supports"
    if s.startswith("refute"):
        return "refutes"
    if "not enough" in s or s in {"nei", "unknown", "insufficient"}:
        return "not_enough_info"
    return s


def _extract_evidence_titles(ex: Dict[str, Any], *, max_n: int = 2) -> List[str]:
    """Extract unique evidence page titles from FEVER's nested evidence field."""
    out: List[str] = []
    ev = ex.get("evidence")
    if not isinstance(ev, list):
        return out

    def _add(page: Any):
        if page is None:
            return
        t = str(page).strip()
        if not t:
            return
        # FEVER pages typically use underscores
        t = t.replace(" ", "_")
        if t not in out:
            out.append(t)

    # evidence: list[list[ [annot_id, ev_id, page, line], ... ]]
    for group in ev:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page = item[2]
            line = item[3]
            try:
                # keep only sentence-level evidence; line=-1 indicates "no sentence"
                if isinstance(line, int) and line < 0:
                    continue
            except Exception:
                pass
            _add(page)
            if len(out) >= max_n:
                return out
    return out


def _db_has_fts(conn: sqlite3.Connection) -> bool:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pages_fts'")
        return cur.fetchone() is not None
    except Exception:
        return False


def _fetch_page(conn: sqlite3.Connection, title: str) -> Optional[str]:
    cur = conn.execute("SELECT content FROM pages WHERE title=?", (title,))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _fts_search(conn: sqlite3.Connection, query: str, topk: int) -> List[Tuple[str, str]]:
    """Return (title, content) pairs."""
    # Defensive: some titles/claims can be too short or contain FTS syntax. Normalize to safe tokens.
    query = (query or "").strip()
    if not query:
        return []
    # Disable column-filter syntax like "13:" by stripping ':' (FTS interprets it as <column>:<term>).
    query = query.replace(":", " ")
    # FTS5 bm25 exists in most builds; if not, drop ORDER BY.
    try:
        cur = conn.execute(
            "SELECT title, content FROM pages_fts WHERE pages_fts MATCH ? ORDER BY bm25(pages_fts) LIMIT ?",
            (query, int(topk)),
        )
        return [(str(t), str(c)) for (t, c) in cur.fetchall()]
    except Exception:
        cur = conn.execute(
            "SELECT title, content FROM pages_fts WHERE pages_fts MATCH ? LIMIT ?",
            (query, int(topk)),
        )
        return [(str(t), str(c)) for (t, c) in cur.fetchall()]


def _random_pages(conn: sqlite3.Connection, k: int, seed: int) -> List[Tuple[str, str]]:
    # SQLite ORDER BY RANDOM() can be slow on huge tables; this is a fallback path.
    rng = random.Random(seed)
    # Try to sample by rowid range
    cur = conn.execute("SELECT max(rowid) FROM pages")
    mx = cur.fetchone()[0] or 0
    out: List[Tuple[str, str]] = []
    if mx <= 0:
        return out
    for _ in range(int(k) * 5):
        rid = rng.randint(1, int(mx))
        cur2 = conn.execute("SELECT title, content FROM pages WHERE rowid=?", (rid,))
        row = cur2.fetchone()
        if row:
            out.append((str(row[0]), str(row[1])))
            if len(out) >= int(k):
                break
    return out


def _compact_query(claim: str, mode: str = 'or') -> str:
    """Build an FTS5 query that is *not* overly restrictive.

    NOTE: In FTS5, whitespace means AND. Using the whole claim as tokens often yields
    zero hits (no document contains all tokens). We instead OR a small set of
    high-signal tokens (prefer capitalized/proper-noun tokens).
    """
    # Tokenize into alphanumerics only. Keep short digit tokens as they can be salient years / ordinals.
    raw_toks = re.findall(r"[A-Za-z0-9]+", claim)
    raw_toks = [t for t in raw_toks if (len(t) >= 3) or t.isdigit()]
    # Avoid FTS operators being interpreted as syntax.
    raw_toks = [t for t in raw_toks if t.lower() not in {"and", "or", "not", "near"}]
    # FEVER uses bracket artifacts like -LRB-/-RRB- which can trip FTS column syntax.
    raw_toks = [t for t in raw_toks if t.upper() not in {"LRB", "RRB", "LSB", "RSB", "LCB", "RCB"}]
    if not raw_toks:
        # Fallback for very short strings like "13:" that would otherwise trip FTS column syntax.
        raw2 = re.findall(r"[A-Za-z0-9]+", claim)
        raw2 = [t for t in raw2 if t.lower() not in {"and", "or", "not", "near"}]
        raw2 = [t for t in raw2 if t.upper() not in {"LRB", "RRB", "LSB", "RSB", "LCB", "RCB"}]
        if not raw2:
            return ""
        # Keep up to 3 unique tokens.
        uniq2 = []
        for t in raw2:
            if t not in uniq2:
                uniq2.append(t)
            if len(uniq2) >= 3:
                break
        uniq2_q = [f'"{t}"' for t in uniq2]
        return " OR ".join(uniq2_q)

    # prioritize likely entity tokens
    def _prio(t: str) -> int:
        if t[:1].isupper():
            return 3
        if t.isdigit():
            return 2
        return 1

    # unique preserve best priority
    uniq = {}
    for t in raw_toks:
        k = t
        uniq[k] = max(uniq.get(k, 0), _prio(t))

    toks = sorted(uniq.keys(), key=lambda t: (_prio(t), len(t)), reverse=True)
    toks = toks[:12]
    toks_q = [f'"{t}"' for t in toks]

    # Query modes:
    # - 'or' (default): OR across tokens (broad, avoids zero-hit)
    # - 'and4'/'and6': AND across top tokens (stricter but still workable)
    if mode == 'or':
        return ' OR '.join(toks_q)
    if mode == 'and4':
        return ' '.join(toks_q[:4])
    if mode == 'and6':
        return ' '.join(toks_q[:6])
    # fallback
    return ' OR '.join(toks_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fever", required=True, help="Path to FEVER jsonl (train/dev).")
    ap.add_argument("--wiki_db", required=True, help="SQLite docstore path built by build_fever_wiki_sqlite.py")
    ap.add_argument("--out", required=True, help="Output prepared jsonl(.gz)")

    ap.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--docs_per_task", type=int, default=20)
    ap.add_argument("--gold_titles_n", type=int, default=2, help="How many evidence titles to keep as gold")
    ap.add_argument("--retrieval_topk", type=int, default=40, help="Candidate docs to retrieve before trimming")
    ap.add_argument("--fts_mode", type=str, default="or", choices=["or", "and4", "and6"], help="FTS query compacting mode: 'or' (broad) or 'and4'/'and6' (stricter).")
    ap.add_argument("--max_doc_chars", type=int, default=2500)

    ap.add_argument("--gold_position", type=str, default="middle", choices=["front", "middle", "back", "random"], help="Where to place gold docs")
    ap.add_argument("--add_traps_from_gold", type=int, default=8, help="Extra candidate docs retrieved using gold-title tokens")

    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    conn = sqlite3.connect(str(args.wiki_db))
    conn.row_factory = sqlite3.Row
    has_fts = _db_has_fts(conn)
    if not has_fts:
        print("[prepare_fever] WARNING: pages_fts not found; retrieval will be weak/random.")

    out_rows: List[Dict[str, Any]] = []

    def _insert_gold(docs_map: Dict[str, str], gold_titles: List[str]):
        for t in gold_titles:
            if t in docs_map:
                continue
            content = _fetch_page(conn, t)
            if content:
                docs_map[t] = content

    def _position_gold(doc_titles: List[str], gold_titles: List[str]) -> List[str]:
        if not gold_titles:
            return doc_titles
        gold_set = set(gold_titles)
        gold = [t for t in doc_titles if t in gold_set]
        rest = [t for t in doc_titles if t not in gold_set]
        if not gold:
            return doc_titles
        if args.gold_position == "front":
            return gold + rest
        if args.gold_position == "back":
            return rest + gold
        if args.gold_position == "random":
            rest2 = rest[:]
            rng.shuffle(rest2)
            pos = rng.randint(0, len(rest2))
            return rest2[:pos] + gold + rest2[pos:]
        # middle
        mid = len(rest) // 2
        return rest[:mid] + gold + rest[mid:]

    for i, ex in enumerate(_open_jsonl(args.fever)):
        if args.limit is not None and len(out_rows) >= int(args.limit):
            break

        claim = str(ex.get("claim") or "").strip()
        if not claim:
            continue
        label = _norm_label(str(ex.get("label") or ""))

        gold_titles = _extract_evidence_titles(ex, max_n=int(args.gold_titles_n))

        # Candidate retrieval
        docs_map: Dict[str, str] = {}

        if has_fts:
            q = _compact_query(claim, str(args.fts_mode))
            for t, c in _fts_search(conn, q, int(args.retrieval_topk)):
                docs_map.setdefault(t, c)

            # Add branch-trap candidates by querying using gold titles (near-miss pages)
            if gold_titles and int(args.add_traps_from_gold) > 0:
                for gt in gold_titles:
                    gtq = _compact_query(gt.replace("_", " "), str(args.fts_mode))
                    for t, c in _fts_search(conn, gtq, int(args.add_traps_from_gold)):
                        docs_map.setdefault(t, c)

            # If the query was still too strict and returned nothing, fall back to random pages.
            if not docs_map:
                for t, c in _random_pages(conn, int(args.retrieval_topk), seed=int(args.seed) + i):
                    docs_map.setdefault(t, c)
        else:
            # Fallback: random pages + exact gold pages
            for t, c in _random_pages(conn, int(args.retrieval_topk), seed=int(args.seed) + i):
                docs_map.setdefault(t, c)

        _insert_gold(docs_map, gold_titles)

        # Trim + order
        titles = list(docs_map.keys())
        if not titles:
            continue

        # Prefer keeping gold + higher overlap on title tokens
        def _score_title(t: str) -> int:
            toks = set(re.findall(r"[A-Za-z0-9]+", (t or "").lower()))
            qtok = set(re.findall(r"[A-Za-z0-9]+", claim.lower()))
            return len(toks & qtok)

        titles.sort(key=_score_title, reverse=True)

        # Ensure gold are included in final trimmed set
        keep: List[str] = []
        for t in titles:
            if t in keep:
                continue
            keep.append(t)
            if len(keep) >= int(args.docs_per_task):
                break

        # If gold got trimmed out, force insert and trim from end
        for gt in gold_titles:
            if gt not in keep and gt in docs_map:
                keep.insert(0, gt)
        keep = keep[: int(args.docs_per_task)]

        keep = _position_gold(keep, gold_titles)

        docs: List[Dict[str, Any]] = []
        for t in keep:
            content = str(docs_map.get(t) or "")
            if args.max_doc_chars and len(content) > int(args.max_doc_chars):
                content = content[: int(args.max_doc_chars)]
            docs.append({
                "docid": t,
                "title": t,
                "content": content,
            })

        out_rows.append({
            "id": ex.get("id", i),
            "claim": claim,
            "label": label,
            "docs": docs,
            "evidence_titles": gold_titles,
        })

    conn.close()
    _dump_jsonl(args.out, out_rows)
    print(f"[prepare_fever] DONE out={args.out} rows={len(out_rows)}")
    if len(out_rows) == 0:
        print(
            "[prepare_fever] WARNING: 0 rows produced. Most common causes: (1) the FEVER input jsonl is empty, "
            "or (2) the file uses a different schema than expected. Expected keys per line include at least: claim, label. "
            "If your data/fever/fever-data/dev.jsonl is empty, re-download the official FEVER train/dev files (see fever.ai / shared task release)."
        )


if __name__ == "__main__":
    main()
