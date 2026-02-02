#!/usr/bin/env python
"""Build a SQLite(+FTS5) docstore from FEVER wiki-pages.

This repo uses a TaskScopedToolBox (per-task docs) to keep evaluation clean.
FEVER raw data doesn't include per-task docs, so we first build a local
Wikipedia docstore, then (in a second step) prepare per-task doc sets.

Two practical modes:
1) FULL (recommended for real runs)
   - Stores *all* pages in the wiki dump.
2) MINI (fast debug)
   - Stores evidence page titles from a FEVER jsonl (first N examples)
     + a random reservoir of extra pages for distractors.

Example (FULL):
  python scripts/build_fever_wiki_sqlite.py \
    --wiki_dir data/fever/wiki/wiki-pages \
    --out_db  data/fever/wiki/wiki.sqlite \
    --max_chars 6000

Example (MINI debug):
  python scripts/build_fever_wiki_sqlite.py \
    --wiki_dir data/fever/wiki/wiki-pages \
    --out_db  data/fever/wiki/wiki_mini.sqlite \
    --keep_titles_from_fever data/fever/fever-data/dev.jsonl \
    --limit_examples 300 \
    --random_reservoir 50000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _iter_wiki_files(wiki_dir: str) -> List[str]:
    pats = ["wiki-*.jsonl", "*.jsonl"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(str(Path(wiki_dir) / p)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No wiki jsonl files found under: {wiki_dir}")
    return files


def _extract_title_and_text(obj: Dict) -> Tuple[str, str]:
    # FEVER wiki-pages usually: {"id": "Title", "lines": "0\tSent...\n1\tSent..."}
    title = str(obj.get("id") or obj.get("title") or obj.get("page") or "").strip()
    raw = obj.get("lines")
    if raw is None:
        raw = obj.get("text")
    if raw is None:
        raw = obj.get("content")

    text = ""
    if isinstance(raw, list):
        # Sometimes: [[sent_id, sent_text], ...]
        parts = []
        for r in raw:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                parts.append(str(r[1]))
            else:
                parts.append(str(r))
        text = "\n".join(parts)
    elif isinstance(raw, str):
        # If tab-separated sentence ids exist, strip them.
        parts = []
        for line in raw.splitlines():
            line = line.strip("\n")
            if not line:
                continue
            segs = line.split("\t")
            if len(segs) >= 2:
                parts.append(segs[1])
            else:
                parts.append(line)
        text = "\n".join(parts)
    else:
        text = str(raw or "")

    return title, text


def _collect_titles_from_fever(fever_path: str, limit_examples: int) -> Set[str]:
    titles: Set[str] = set()
    n = 0
    with open(fever_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ev = ex.get("evidence")
            if isinstance(ev, list):
                for group in ev:
                    if not isinstance(group, list):
                        continue
                    for item in group:
                        if not isinstance(item, list) or len(item) < 3:
                            continue
                        page = item[2]
                        if page is None:
                            continue
                        titles.add(str(page))
            n += 1
            if limit_examples and n >= limit_examples:
                break
    return titles


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_test USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_test")
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_dir", type=str, required=True)
    ap.add_argument("--out_db", type=str, required=True)
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--drop", action="store_true", help="Delete existing DB first")

    # MINI mode
    ap.add_argument("--keep_titles_from_fever", type=str, default=None,
                    help="If set, only keep evidence titles from this FEVER jsonl (first N examples)")
    ap.add_argument("--limit_examples", type=int, default=0)
    ap.add_argument("--random_reservoir", type=int, default=0,
                    help="In MINI mode, also keep this many random pages as distractors")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    wiki_dir = args.wiki_dir
    out_db = Path(args.out_db)
    _ensure_parent(out_db)

    if args.drop and out_db.exists():
        out_db.unlink()

    keep_titles: Optional[Set[str]] = None
    mini_mode = bool(args.keep_titles_from_fever)
    if mini_mode:
        keep_titles = _collect_titles_from_fever(args.keep_titles_from_fever, int(args.limit_examples or 0))
        print(f"[build_wiki_sqlite] MINI mode: keep_titles={len(keep_titles)}")

    files = _iter_wiki_files(wiki_dir)

    conn = sqlite3.connect(str(out_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    conn.execute("CREATE TABLE IF NOT EXISTS pages (title TEXT PRIMARY KEY, content TEXT)")

    fts_ok = _has_fts5(conn)
    if fts_ok:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(title, content)")
    else:
        print("[build_wiki_sqlite] WARNING: sqlite3 FTS5 not available. Retrieval quality will be poor.")

    rng = random.Random(int(args.seed))

    # Reservoir sampling for MINI mode distractors
    reservoir_cap = int(args.random_reservoir or 0) if mini_mode else 0
    reservoir: List[Tuple[str, str]] = []
    seen_nonkeep = 0

    def _maybe_add_to_reservoir(title: str, content: str):
        nonlocal seen_nonkeep
        if reservoir_cap <= 0:
            return
        seen_nonkeep += 1
        if len(reservoir) < reservoir_cap:
            reservoir.append((title, content))
            return
        j = rng.randint(0, seen_nonkeep - 1)
        if j < reservoir_cap:
            reservoir[j] = (title, content)

    n_pages = 0
    n_kept = 0

    cur = conn.cursor()
    cur2 = conn.cursor()

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                title, text = _extract_title_and_text(obj)
                if not title:
                    continue
                if not text:
                    continue
                if args.max_chars and len(text) > int(args.max_chars):
                    text = text[: int(args.max_chars)]

                n_pages += 1

                if mini_mode and keep_titles is not None:
                    if title in keep_titles:
                        n_kept += 1
                        cur.execute("INSERT OR REPLACE INTO pages(title, content) VALUES (?,?)", (title, text))
                        if fts_ok:
                            cur2.execute("INSERT INTO pages_fts(title, content) VALUES (?,?)", (title, text))
                    else:
                        _maybe_add_to_reservoir(title, text)
                else:
                    n_kept += 1
                    cur.execute("INSERT OR REPLACE INTO pages(title, content) VALUES (?,?)", (title, text))
                    if fts_ok:
                        cur2.execute("INSERT INTO pages_fts(title, content) VALUES (?,?)", (title, text))

                if n_pages % 20000 == 0:
                    conn.commit()
                    print(f"[build_wiki_sqlite] scanned={n_pages:,} kept={n_kept:,}")

    # Insert reservoir samples if in MINI mode
    if mini_mode and reservoir:
        print(f"[build_wiki_sqlite] inserting reservoir distractors: {len(reservoir)}")
        for title, text in reservoir:
            cur.execute("INSERT OR REPLACE INTO pages(title, content) VALUES (?,?)", (title, text))
            if fts_ok:
                cur2.execute("INSERT INTO pages_fts(title, content) VALUES (?,?)", (title, text))
        n_kept += len(reservoir)

    conn.commit()
    conn.close()
    print(f"[build_wiki_sqlite] DONE db={out_db} scanned={n_pages:,} kept={n_kept:,} fts5={fts_ok}")


if __name__ == "__main__":
    main()
