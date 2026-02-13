#!/usr/bin/env python3
"""
run_phase18_sweep.py

Batch runner for Phase18 TraceOps experiments with:
1) Smoke stage: run low/high (or all categorical) per sweep group, extract diagnostics, and stop if a knob looks no-op.
2) Full stage: if smoke passes, run the full experiment list and store outputs with stable names.

Key design points:
- No bash multiline/backslash: uses argv lists and subprocess without shell=True.
- Does NOT modify any TraceOps cache behavior (.cache/traceops_llm untouched).
- Does NOT alter bundle generation code; it only parses the printed "Zip path:" and copies the resulting zip
  into a stable filename.

Directory layout:
  <out_dir>/smoke/<exp_id>__<name>.{log,zip,json}
  <out_dir>/full/<exp_id>__<name>.{log,zip,json}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ZIP_RE = re.compile(r"Zip path:\s*(?P<path>.+\.zip)\s*$", re.MULTILINE)

# ---- small helpers ----

def _sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_zip_from_output(text: str) -> Optional[str]:
    m = ZIP_RE.search(text)
    return m.group("path").strip() if m else None

def _replace_flag_value(argv: List[str], flag: str, new_value: str) -> List[str]:
    """Replace '--flag <value>' if present, else append it."""
    out = list(argv)
    if flag in out:
        i = out.index(flag)
        # if has a value token
        if i + 1 < len(out) and not out[i + 1].startswith("--"):
            out[i + 1] = str(new_value)
        else:
            out.insert(i + 1, str(new_value))
    else:
        out += [flag, str(new_value)]
    return out

def _get_flag_value(argv: List[str], flag: str) -> Optional[str]:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
        return "true"
    return None

def _group_id(exp_id: str) -> str:
    # Group by prefix before first underscore: K1a_0_2 -> K1a, M1 -> M1
    return exp_id.split("_", 1)[0]

# ---- artifact parsing (inside zip) ----

@dataclass
class BundleDiagnostics:
    exp_id: str
    name: str
    zip_path: Path  # copied stable zip path
    manifest: Dict[str, Any]
    threads: int
    pivots_total: int
    indirect_pivots_total: int
    indirect_frac: float
    pivot_msgs_total: int
    pivot_handle_hits: int
    pivot_ordinal_hits: int
    clause_handle_hits: int
    gold_needs_more_info_rate: Optional[float]

def _find_member(z: zipfile.ZipFile, suffix: str) -> Optional[str]:
    for n in z.namelist():
        if n.endswith(suffix):
            return n
    return None

def _read_json(z: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    with z.open(member) as f:
        return json.load(f)

def _read_csv_first_row(z: zipfile.ZipFile, member: str) -> Dict[str, str]:
    with z.open(member) as f:
        txt = f.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(txt.splitlines())
    for row in reader:
        return row
    return {}

_ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth",
    "previous", "earlier", "prior", "before", "above",
]

def _analyze_bundle(exp_id: str, name: str, zip_path: Path) -> BundleDiagnostics:
    with zipfile.ZipFile(zip_path, "r") as z:
        manifest_member = _find_member(z, "phase18/run_manifest.json")
        if not manifest_member:
            raise RuntimeError(f"run_manifest.json not found in {zip_path}")
        manifest = _read_json(z, manifest_member)

        summary_member = _find_member(z, "phase18/analysis/phase18_traceops_summary.csv")
        gold_nmi = None
        if summary_member:
            row0 = _read_csv_first_row(z, summary_member)
            # this is run-level, repeated per method; first row ok
            try:
                gold_nmi = float(row0.get("gold_needs_more_info_rate", "") or "")
            except Exception:
                gold_nmi = None

        threads_member = None
        # data path depends on scenario(s); threads.jsonl stored under phase18/data/<scenario>/...
        # Pick the first threads.jsonl we find.
        for n in z.namelist():
            if n.endswith("/data/traceops/threads.jsonl"):
                threads_member = n
                break
        if not threads_member:
            raise RuntimeError(f"threads.jsonl not found in {zip_path}")

        threads = 0
        pivots_total = 0
        indirect_pivots_total = 0
        pivot_msgs_total = 0
        pivot_handle_hits = 0
        pivot_ordinal_hits = 0
        clause_handle_hits = 0

        with z.open(threads_member) as f:
            for raw in f:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                obj = json.loads(line)
                threads += 1
                meta = obj.get("meta", {}) or {}
                pivots_total += int(meta.get("pivot_count", 0) or 0)
                indirect_pivots_total += int(meta.get("indirect_pivot_count", 0) or 0)

                # pivot messages (from steps)
                for step in (obj.get("steps") or []):
                    if step.get("kind") == "pivot_check":
                        msg = str(step.get("message") or "")
                        pivot_msgs_total += 1
                        lm = msg.lower()
                        if "handle" in lm:
                            pivot_handle_hits += 1
                        if any(w in lm for w in _ORDINAL_WORDS):
                            pivot_ordinal_hits += 1

                # clause texts: count 'handle' occurrences
                clauses = obj.get("clauses") or {}
                if isinstance(clauses, dict):
                    for c in clauses.values():
                        txt = str(c.get("text") or "")
                        clause_handle_hits += txt.lower().count("handle")

        indirect_frac = (indirect_pivots_total / pivots_total) if pivots_total > 0 else 0.0

        return BundleDiagnostics(
            exp_id=exp_id,
            name=name,
            zip_path=zip_path,
            manifest=manifest,
            threads=threads,
            pivots_total=pivots_total,
            indirect_pivots_total=indirect_pivots_total,
            indirect_frac=indirect_frac,
            pivot_msgs_total=pivot_msgs_total,
            pivot_handle_hits=pivot_handle_hits,
            pivot_ordinal_hits=pivot_ordinal_hits,
            clause_handle_hits=clause_handle_hits,
            gold_needs_more_info_rate=gold_nmi,
        )

# ---- diagnosis logic ----

@dataclass
class SmokeGroupPlan:
    group: str
    sweep_flag: Optional[str]
    values: List[str]
    exp_ids: List[str]         # in config order
    smoke_ids: List[str]       # to run in smoke stage (low/high or all categorical)

def _infer_sweep_flag(exps: List[dict]) -> Tuple[Optional[str], List[str]]:
    """
    Infer which flag is being swept within a group by finding flags whose value differs across experiments.
    Returns (flag, values_in_order). If ambiguous, returns (None, []).

    Note: handles only '--flag value' style (not boolean-only flags).
    """
    # Build mapping exp -> flag->value
    maps: List[Dict[str, str]] = []
    for e in exps:
        argv = e.get("args") or []
        m: Dict[str, str] = {}
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok.startswith("--"):
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    m[tok] = argv[i + 1]
                    i += 2
                else:
                    # boolean flag; ignore for sweep detection
                    i += 1
            else:
                i += 1
        maps.append(m)

    # candidate flags where values differ across exps
    all_flags = set().union(*maps) if maps else set()
    candidates: List[Tuple[str, List[str]]] = []
    for flg in sorted(all_flags):
        vals = [m.get(flg) for m in maps]
        if any(v is None for v in vals):
            continue
        if len(set(vals)) > 1:
            candidates.append((flg, vals))

    if not candidates:
        return None, []

    # Prefer known sweep knobs first
    preferred = [
        "--traceops_indirection_rate",
        "--traceops_alias_chain_len",
        "--traceops_indirect_pivot_style",
        "--traceops_trap_similarity_boost",
        "--traceops_trap_distractor_count",
        "--traceops_delay_to_relevance",
    ]
    for p in preferred:
        for flg, vals in candidates:
            if flg == p:
                return flg, vals

    # Otherwise pick the one with the most unique values
    candidates.sort(key=lambda x: len(set(x[1])), reverse=True)
    return candidates[0][0], candidates[0][1]

def _pick_smoke_ids(plan: SmokeGroupPlan, exps_by_id: Dict[str, dict]) -> List[str]:
    """Numeric sweeps -> min/max; categorical -> all (if small) else first/last."""
    if not plan.sweep_flag:
        return [plan.exp_ids[0]]  # fallback

    # Get values from argv for the sweep flag
    vals = []
    for eid in plan.exp_ids:
        v = _get_flag_value(exps_by_id[eid]["args"], plan.sweep_flag)
        vals.append(v or "")

    # numeric?
    num_vals = []
    is_numeric = True
    for v in vals:
        try:
            num_vals.append(float(v))
        except Exception:
            is_numeric = False
            break

    if is_numeric:
        # choose min/max based on float values; keep corresponding ids
        min_i = min(range(len(num_vals)), key=lambda i: num_vals[i])
        max_i = max(range(len(num_vals)), key=lambda i: num_vals[i])
        if min_i == max_i:
            return [plan.exp_ids[min_i]]
        return [plan.exp_ids[min_i], plan.exp_ids[max_i]]

    # categorical
    if len(plan.exp_ids) <= 3:
        return list(plan.exp_ids)
    # big categorical: first/last
    return [plan.exp_ids[0], plan.exp_ids[-1]]

def _diagnose_group(plan: SmokeGroupPlan, diags: Dict[str, BundleDiagnostics]) -> Tuple[bool, str]:
    """
    Return (ok, message). Designed to catch common no-op sweeps.
    """
    sf = plan.sweep_flag or ""
    # build list in smoke order
    items = [diags[eid] for eid in plan.smoke_ids if eid in diags]
    if len(items) != len(plan.smoke_ids):
        return False, f"{plan.group}: missing diagnostics for some smoke runs (zip parse failed)."

    def fmt(d: BundleDiagnostics) -> str:
        m = d.manifest
        return (f"{d.exp_id}: scenarios={m.get('traceops_scenarios')} "
                f"{sf}={m.get(sf.lstrip('-'), m.get(sf.replace('--','traceops_'), None))} "
                f"pivots={d.pivots_total} indirect_frac={d.indirect_frac:.2f} "
                f"pivot_handle_rate={(d.pivot_handle_hits/max(d.pivot_msgs_total,1)):.2f} "
                f"pivot_ordinal_rate={(d.pivot_ordinal_hits/max(d.pivot_msgs_total,1)):.2f} "
                f"clause_handle_hits={d.clause_handle_hits}")

    # Specialized checks
    if sf == "--traceops_indirection_rate":
        # Expect indirect_frac to increase with indirection_rate in mixed scenario
        # Find low/high by manifest rate
        rates = []
        for d in items:
            try:
                rates.append((float(d.manifest.get("traceops_indirection_rate", 0.0)), d))
            except Exception:
                rates.append((0.0, d))
        rates.sort(key=lambda x: x[0])
        if len(rates) < 2:
            return False, f"{plan.group}: need >=2 smoke points for indirection_rate."
        low_r, low_d = rates[0]
        high_r, high_d = rates[-1]
        diff = high_d.indirect_frac - low_d.indirect_frac
        ok = diff >= 0.15  # conservative threshold
        msg = "\n  ".join([f"{plan.group} smoke diagnostics (indirection sweep):",
                           fmt(low_d),
                           fmt(high_d),
                           f"Expected indirect_frac to rise with rate; observed diff={diff:.2f} (threshold 0.15)."])
        return ok, msg

    if sf == "--traceops_alias_chain_len":
        # Heuristic: more alias chain -> more 'handle' mentions in clauses/pivots
        lens = []
        for d in items:
            try:
                lens.append((int(float(d.manifest.get("traceops_alias_chain_len", 0))), d))
            except Exception:
                lens.append((0, d))
        lens.sort(key=lambda x: x[0])
        if len(lens) < 2:
            return False, f"{plan.group}: need >=2 smoke points for alias_chain_len."
        low_l, low_d = lens[0]
        high_l, high_d = lens[-1]
        # Use clause handle hits (more stable) + pivot handle rate
        low_score = low_d.clause_handle_hits + low_d.pivot_handle_hits
        high_score = high_d.clause_handle_hits + high_d.pivot_handle_hits
        ok = (high_score >= low_score + 3)  # small threshold for smoke
        msg = "\n  ".join([f"{plan.group} smoke diagnostics (alias_chain_len sweep):",
                           fmt(low_d),
                           fmt(high_d),
                           f"Expected more handle signals at higher chain_len; observed {low_score}->{high_score} (threshold +3)."])
        return ok, msg

    if sf == "--traceops_indirect_pivot_style":
        # Heuristic signatures on pivot messages:
        # - alias_handle: higher handle rate, lower ordinal rate
        # - ordinal_ref: higher ordinal rate, lower handle rate
        # - blended: both moderate
        # We don't hard-fail if all close, but we flag if totally indistinguishable.
        rates = []
        for d in items:
            style = str(d.manifest.get("traceops_indirect_pivot_style", "") or "")
            handle_r = d.pivot_handle_hits / max(d.pivot_msgs_total, 1)
            ord_r = d.pivot_ordinal_hits / max(d.pivot_msgs_total, 1)
            rates.append((style, handle_r, ord_r, d))
        # indistinguishable check
        handle_span = max(r[1] for r in rates) - min(r[1] for r in rates)
        ord_span = max(r[2] for r in rates) - min(r[2] for r in rates)
        ok = (handle_span >= 0.15) or (ord_span >= 0.15)
        msg_lines = [f"{plan.group} smoke diagnostics (pivot_style sweep):"]
        for style, hr, orr, d in rates:
            msg_lines.append(fmt(d))
        msg_lines.append(f"Span(handle_rate)={handle_span:.2f}, Span(ordinal_rate)={ord_span:.2f} (need >=0.15 on either to be confident).")
        return ok, "\n  ".join(msg_lines)

    # Generic: if sweep flag exists, at least confirm manifest reflects it.
    ok = True
    msg_lines = [f"{plan.group} smoke diagnostics (generic):"]
    for d in items:
        msg_lines.append(fmt(d))
    msg_lines.append("No specialized diagnostic for this sweep_flag; only confirmed smoke bundles generated and parsed.")
    return ok, "\n  ".join(msg_lines)

# ---- running experiments ----

def _run_one(
    repo_root: Path,
    exp_id: str,
    name: str,
    argv_extra: List[str],
    exp_args: List[str],
    out_dir: Path,
    pythonpath: str,
    dry_run: bool,
    skip_existing: bool,
) -> Tuple[int, Optional[Path], Path, Path]:
    """
    Returns: (returncode, copied_zip_path_or_None, log_path, meta_path)
    """
    script = repo_root / "scripts" / "run_phase18_traceops_bundle.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find {script}. Are you in the repo root?")

    safe_name = _sanitize(name)
    log_path = out_dir / f"{exp_id}__{safe_name}.log"
    meta_path = out_dir / f"{exp_id}__{safe_name}.json"
    zip_dst = out_dir / f"{exp_id}__{safe_name}.zip"

    if skip_existing and zip_dst.exists():
        print(f"[SKIP] {exp_id} (zip exists): {zip_dst}")
        return 0, zip_dst, log_path, meta_path

    cmd = [sys.executable, str(script)] + list(exp_args) + argv_extra

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    if dry_run:
        print(f"[DRY] {exp_id}: {' '.join(cmd)}")
        return 0, None, log_path, meta_path

    out_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# exp_id={exp_id}\n# name={name}\n# cmd={' '.join(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # parse log for zip path
    out_text = log_path.read_text(encoding="utf-8", errors="ignore")
    src_zip = find_zip_from_output(out_text)

    copied = None
    if src_zip and Path(src_zip).exists():
        try:
            shutil.copy2(src_zip, zip_dst)
            copied = zip_dst
        except Exception as e:
            print(f"[WARN] Could not copy zip for {exp_id}: {e}")

    meta = {
        "exp_id": exp_id,
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "source_zip": src_zip,
        "copied_zip": str(copied) if copied else "",
        "argv_extra": argv_extra,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return proc.returncode, copied, log_path, meta_path

# ---- main ----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Repo root (contains scripts/ and src/).")
    ap.add_argument("--config", type=str, default="phase18_sweep_config.json", help="JSON config file.")
    ap.add_argument("--out_dir", type=str, default="sweep_outputs", help="Output directory.")
    ap.add_argument("--only", type=str, default="", help="Comma-separated experiment IDs to run (optional).")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    ap.add_argument("--pythonpath", type=str, default="src", help="PYTHONPATH value to set for subprocess.")
    ap.add_argument("--mode", choices=["smoke", "full", "smoke_then_full"], default="smoke_then_full")
    ap.add_argument("--smoke_max_pivots", type=int, default=20, help="Override --traceops_llm_max_pivots for smoke stage.")
    ap.add_argument("--smoke_threads", type=int, default=20, help="Override --traceops_threads for smoke stage (0 to disable).")
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs whose stable zip already exists.")
    ap.add_argument("--force_full", action="store_true", help="Run full stage even if smoke diagnosis fails.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = load_config(cfg_path)

    experiments = cfg.get("experiments") or []
    if not isinstance(experiments, list) or not experiments:
        print("[ERROR] Config has no experiments.")
        return 2

    only_set = set([s.strip() for s in args.only.split(",") if s.strip()])
    if only_set:
        experiments = [e for e in experiments if e.get("id") in only_set]
        if not experiments:
            print("[ERROR] --only filtered all experiments; nothing to run.")
            return 2

    exps_by_id = {e["id"]: e for e in experiments}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    smoke_dir = out_dir / "smoke"
    full_dir = out_dir / "full"

    # Build group plans
    groups: Dict[str, List[dict]] = {}
    for e in experiments:
        gid = _group_id(e["id"])
        groups.setdefault(gid, []).append(e)

    plans: List[SmokeGroupPlan] = []
    for gid, exps in groups.items():
        # config order preserved
        exp_ids = [e["id"] for e in exps]
        sweep_flag, values_in_order = _infer_sweep_flag(exps)
        plan = SmokeGroupPlan(group=gid, sweep_flag=sweep_flag, values=values_in_order, exp_ids=exp_ids, smoke_ids=[])
        plan.smoke_ids = _pick_smoke_ids(plan, exps_by_id)
        plans.append(plan)

    # SMOKE STAGE
    smoke_ok = True
    smoke_diags: Dict[str, BundleDiagnostics] = {}

    def smoke_overrides(exp_args: List[str]) -> List[str]:
        argv = list(exp_args)
        argv = _replace_flag_value(argv, "--traceops_llm_max_pivots", str(int(args.smoke_max_pivots)))
        if int(args.smoke_threads) > 0:
            argv = _replace_flag_value(argv, "--traceops_threads", str(int(args.smoke_threads)))
        return argv

    if args.mode in ("smoke", "smoke_then_full"):
        print(f"== Smoke stage == (max_pivots={args.smoke_max_pivots}, threads_override={args.smoke_threads})")
        # Always smoke M1 if present (pipeline sanity)
        if "M1" in exps_by_id:
            e = exps_by_id["M1"]
            rc, copied, logp, metap = _run_one(
                repo_root=repo_root,
                exp_id=e["id"],
                name=e.get("name", e["id"]) + "__SMOKE",
                argv_extra=[],
                exp_args=smoke_overrides(e.get("args") or []),
                out_dir=smoke_dir,
                pythonpath=args.pythonpath,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            if (not args.dry_run) and (rc == 0) and copied:
                try:
                    smoke_diags[e["id"]] = _analyze_bundle(e["id"], e.get("name", e["id"]), copied)
                except Exception as ex:
                    print(f"[SMOKE ERROR] Failed to analyze {e['id']}: {ex}")
                    smoke_ok = False
            elif (not args.dry_run) and rc != 0:
                smoke_ok = False

        # Smoke sweep groups (skip M1-only group)
        for plan in sorted(plans, key=lambda p: p.group):
            if plan.group == "M1":
                continue
            # If group has only 1 experiment, skip smoke for it
            if len(plan.exp_ids) < 2:
                continue

            print(f"\n-- Smoke group {plan.group} -- sweep_flag={plan.sweep_flag} smoke_ids={plan.smoke_ids}")
            for eid in plan.smoke_ids:
                e = exps_by_id[eid]
                rc, copied, logp, metap = _run_one(
                    repo_root=repo_root,
                    exp_id=e["id"],
                    name=e.get("name", e["id"]) + "__SMOKE",
                    argv_extra=[],
                    exp_args=smoke_overrides(e.get("args") or []),
                    out_dir=smoke_dir,
                    pythonpath=args.pythonpath,
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                )
                if args.dry_run:
                    continue
                if rc != 0 or not copied:
                    print(f"[SMOKE FAIL] {eid} failed (rc={rc}) or zip missing. log={logp}")
                    smoke_ok = False
                    continue
                try:
                    smoke_diags[eid] = _analyze_bundle(eid, e.get("name", eid), copied)
                except Exception as ex:
                    print(f"[SMOKE FAIL] {eid} analysis failed: {ex}")
                    smoke_ok = False

            if args.dry_run:
                continue

            # group diagnosis
            if all(eid in smoke_diags for eid in plan.smoke_ids):
                ok, msg = _diagnose_group(plan, smoke_diags)
                print("  " + msg.replace("\n", "\n  "))
                if not ok:
                    smoke_ok = False
                    print(f"[SMOKE DIAG FAIL] {plan.group}: suspected no-op or weak separation.")

        if args.dry_run:
            print("\n[DRY] Smoke stage only printed commands.")
        else:
            print(f"\nSmoke stage result: {'PASS' if smoke_ok else 'FAIL'}")
            if (not smoke_ok) and (not args.force_full) and args.mode == "smoke_then_full":
                print("Stopping before full stage (use --force_full to proceed anyway).")
                return 3

    # FULL STAGE
    if args.mode in ("full", "smoke_then_full"):
        print("\n== Full stage ==")
        for e in experiments:
            rc, copied, logp, metap = _run_one(
                repo_root=repo_root,
                exp_id=e["id"],
                name=e.get("name", e["id"]),
                argv_extra=[],
                exp_args=(e.get("args") or []),
                out_dir=full_dir,
                pythonpath=args.pythonpath,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            if args.dry_run:
                continue
            if rc != 0:
                print(f"[ERROR] Experiment failed (rc={rc}). See log: {logp}")
            elif copied:
                print(f"[OK] {e['id']} copied zip -> {copied}")
            else:
                print(f"[WARN] {e['id']} finished but zip not copied. See log/meta: {logp}, {metap}")

        print(f"\nDone. Outputs in: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
