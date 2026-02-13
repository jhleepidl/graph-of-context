#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


ZIP_RE = re.compile(r"Zip path:\s*(?P<path>.+\.zip)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    name: str
    delay: int
    cap_update: int
    excl_rate: float
    include_smart_controls: bool


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return safe or "run"


def _build_experiments(include_excludable_sweep: bool) -> List[Experiment]:
    out = [
        Experiment(
            exp_id="M200_d6",
            name="main200_delay6",
            delay=6,
            cap_update=4,
            excl_rate=0.7,
            include_smart_controls=False,
        ),
        Experiment(
            exp_id="M200_d0",
            name="main200_delay0_smartctrl",
            delay=0,
            cap_update=4,
            excl_rate=0.7,
            include_smart_controls=True,
        ),
        Experiment(
            exp_id="D0_capU2",
            name="delay0_cap_update_2",
            delay=0,
            cap_update=2,
            excl_rate=0.7,
            include_smart_controls=True,
        ),
        Experiment(
            exp_id="D0_capU8",
            name="delay0_cap_update_8",
            delay=0,
            cap_update=8,
            excl_rate=0.7,
            include_smart_controls=True,
        ),
    ]
    if include_excludable_sweep:
        out.append(
            Experiment(
                exp_id="D0_excl0",
                name="delay0_excludable_0",
                delay=0,
                cap_update=4,
                excl_rate=0.0,
                include_smart_controls=True,
            )
        )
    return out


def _build_bundle_args(args: argparse.Namespace, exp: Experiment) -> List[str]:
    cmd = [
        "--dotenv",
        str(args.dotenv),
        "--model",
        str(args.model),
        "--traceops_level",
        "3",
        "--traceops_scenarios",
        "indirect",
        "--traceops_threads",
        str(int(args.threads)),
        "--traceops_seed",
        str(int(args.seed)),
        "--traceops_eval_mode",
        "llm",
        "--traceops_llm_eval_scope",
        "pivots",
        "--traceops_llm_max_pivots",
        str(int(args.max_pivots)),
        "--traceops_delay_to_relevance",
        str(int(exp.delay)),
        "--traceops_core_necessity_enable",
        "--traceops_core_necessity_require_all",
        "--traceops_trap_decision_flip_enable",
        "--traceops_hidden_core_enable",
        "--traceops_hidden_core_kind",
        "low_overlap_clause",
        "--traceops_hidden_core_link_mode",
        "depends_on",
        "--traceops_trap_flip_salience",
        "0.25",
        "--traceops_trap_flip_attach_kind",
        "avoided",
        "--traceops_trap_graph_excludable_rate",
        str(float(exp.excl_rate)),
        "--traceops_trap_graph_excludable_kinds",
        "stale,inapplicable,avoided,decision_checkpoint",
        "--traceops_trap_invalidation_text_strength",
        "0.6",
        "--traceops_defer_budget_rate",
        "0.15",
        "--traceops_trap_graph_force_topk",
        "1",
        "--traceops_trap_graph_force_include_flip_target",
        "--traceops_trap_graph_force_include_decision_checkpoint",
        "--goc_smart_cap_option",
        "0",
        "--goc_smart_cap_assumption",
        "2",
        "--goc_smart_cap_update",
        str(int(exp.cap_update)),
        "--goc_smart_cap_exception",
        "2",
        "--goc_smart_cap_evidence",
        "2",
    ]
    if exp.include_smart_controls:
        cmd.append("--include_smart_controls")
    return cmd


def _find_zip_path(text: str) -> Optional[str]:
    match = ZIP_RE.search(text)
    return match.group("path").strip() if match else None


def _find_member(zf: zipfile.ZipFile, suffix: str) -> Optional[str]:
    for member in zf.namelist():
        if member.endswith(suffix):
            return member
    return None


def _extract_sanity(copied_zip: Path) -> Dict[str, Any]:
    delay_from_meta: Any = None
    deltas: List[int] = []
    meta_member_name = ""
    threads_member_name = ""
    with zipfile.ZipFile(copied_zip, "r") as zf:
        meta_member = _find_member(zf, "phase18/data/indirect/data/traceops/meta.json")
        if meta_member:
            meta_member_name = meta_member
            with zf.open(meta_member) as fh:
                meta_obj = json.load(fh)
            if isinstance(meta_obj, dict):
                delay_from_meta = meta_obj.get("traceops_delay_to_relevance")

        threads_member = _find_member(zf, "phase18/data/indirect/data/traceops/threads.jsonl")
        if not threads_member:
            threads_member = _find_member(zf, "/data/traceops/threads.jsonl")
        if threads_member:
            threads_member_name = threads_member
            with zf.open(threads_member) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    for step in obj.get("steps") or []:
                        if str(step.get("kind") or "") != "pivot_check":
                            continue
                        metadata = step.get("metadata") or {}
                        if not isinstance(metadata, dict):
                            continue
                        cutoff = metadata.get("gold_state_cutoff_step")
                        if cutoff is None:
                            continue
                        try:
                            step_idx = int(step.get("step_idx"))
                            deltas.append(step_idx - int(cutoff))
                        except Exception:
                            continue
    inferred = {
        "count": int(len(deltas)),
        "min": int(min(deltas)) if deltas else None,
        "max": int(max(deltas)) if deltas else None,
        "unique": [int(v) for v in sorted(set(deltas))],
    }
    return {
        "traceops_delay_to_relevance_from_meta": delay_from_meta,
        "inferred_delay_summary": inferred,
        "members": {
            "meta": meta_member_name,
            "threads": threads_member_name,
        },
    }


def _run_experiment(
    *,
    repo_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    exp: Experiment,
) -> int:
    safe_name = _sanitize_name(exp.name)
    stem = f"{exp.exp_id}__{safe_name}"
    log_path = out_dir / f"{stem}.log"
    zip_path = out_dir / f"{stem}.zip"
    meta_path = out_dir / f"{stem}.meta.json"

    script_path = repo_root / "scripts" / "run_phase18_traceops_bundle.py"
    cmd = [sys.executable, str(script_path)] + _build_bundle_args(args, exp)

    if args.skip_existing and zip_path.exists():
        print(f"[SKIP] {exp.exp_id}: {zip_path}")
        if not meta_path.exists():
            meta_obj = {
                "exp_id": exp.exp_id,
                "name": exp.name,
                "argv": cmd,
                "source_zip_path": "",
                "copied_zip_path": str(zip_path),
                "returncode": 0,
                "skipped": True,
                "sanity_checks": {},
            }
            meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
        return 0

    if args.dry_run:
        print(f"[DRY] {exp.exp_id}: {' '.join(cmd)}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# exp_id={exp.exp_id}\n")
        log_f.write(f"# name={exp.name}\n")
        log_f.write(f"# cmd={' '.join(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    src_zip_raw = _find_zip_path(log_text)
    src_zip_path = ""
    copied_zip_path = ""
    sanity_checks: Dict[str, Any] = {}

    if src_zip_raw:
        src_candidate = Path(src_zip_raw)
        if not src_candidate.is_absolute():
            src_candidate = (repo_root / src_candidate).resolve()
        src_zip_path = str(src_candidate)
        if src_candidate.exists():
            shutil.copy2(src_candidate, zip_path)
            copied_zip_path = str(zip_path)
            try:
                sanity_checks = _extract_sanity(zip_path)
            except Exception as exc:
                sanity_checks = {"error": f"failed to extract sanity checks: {exc}"}

    meta_obj = {
        "exp_id": exp.exp_id,
        "name": exp.name,
        "argv": cmd,
        "source_zip_path": src_zip_path,
        "copied_zip_path": copied_zip_path,
        "returncode": int(proc.returncode),
        "skipped": False,
        "sanity_checks": sanity_checks,
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
    if proc.returncode == 0 and copied_zip_path:
        print(f"[OK] {exp.exp_id}: {copied_zip_path}")
    elif proc.returncode == 0:
        print(f"[WARN] {exp.exp_id}: completed but zip was not found in log output ({log_path})")
    else:
        print(f"[FAIL] {exp.exp_id}: rc={proc.returncode} ({log_path})")
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--dotenv", type=str, default=".env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=200)
    ap.add_argument("--max_pivots", type=int, default=200)
    ap.add_argument("--include_excludable_sweep", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = _build_experiments(bool(args.include_excludable_sweep))
    print(f"Planned runs: {len(experiments)}")
    failures = 0
    for exp in experiments:
        rc = _run_experiment(repo_root=repo_root, out_dir=out_dir, args=args, exp=exp)
        if rc != 0:
            failures += 1

    if args.dry_run:
        print("Dry-run complete.")
        return 0
    if failures:
        print(f"Completed with failures: {failures}")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
