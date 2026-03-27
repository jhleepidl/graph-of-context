#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

# Ensure repo root on path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.policyops.traceops_v0 import load_traceops_dataset
from src.policyops.traceops_v0.controller_dataset import iter_pivot_evals, pivot_eval_to_dict


def _discover_dataset_dirs(path: Path) -> List[Path]:
    if path.is_file() and path.name == "threads.jsonl":
        parent = path.parent
        if parent.name == "traceops" and parent.parent.name == "data":
            return [parent.parent.parent]
        return [parent]
    if (path / "data" / "traceops" / "threads.jsonl").exists() and (path / "data" / "traceops" / "meta.json").exists():
        return [path]
    if (path / "threads.jsonl").exists() and (path / "meta.json").exists():
        if path.name == "traceops" and path.parent.name == "data":
            return [path.parent.parent]
        return [path]
    out: List[Path] = []
    for cand in sorted(path.rglob("threads.jsonl")):
        parent = cand.parent
        if not (parent / "meta.json").exists():
            continue
        if parent.name == "traceops" and parent.parent.name == "data":
            out.append(parent.parent.parent)
        else:
            out.append(parent)
    uniq: List[Path] = []
    seen = set()
    for item in out:
        if item in seen:
            continue
        uniq.append(item)
        seen.add(item)
    return uniq


def _default_args_from_meta(meta: Dict[str, Any], overrides: Dict[str, Any]) -> SimpleNamespace:
    base: Dict[str, Any] = {
        "traceops_max_steps": 0,
        "traceops_similarity_topk": 8,
        "goc_enable_avoids": True,
        "goc_applicability_seed_enable": True,
        "goc_applicability_seed_topk": 8,
        "goc_dependency_closure_enable": False,
        "goc_dependency_closure_topk": 12,
        "goc_dependency_closure_hops": 1,
        "goc_dependency_closure_universe": "candidates",
        "goc_depwalk_enable": True,
        "goc_depwalk_hops": 2,
        "goc_depwalk_topk_per_hop": 6,
        "goc_smart_context_enable": True,
        "fork_k": 6,
        "fork_max_tokens": 160,
        "fork_include_recent_active": True,
        "fork_recent_active_n": 4,
        "fork_dependency_hops": 2,
    }
    # Prefer dataset meta when present.
    for key in list(base.keys()):
        if key in meta:
            base[key] = meta.get(key)
    # Also recover common phase18 names.
    if "traceops_delay_to_relevance" in meta:
        base["traceops_delay_to_relevance"] = meta.get("traceops_delay_to_relevance")
    base.update(overrides)
    return SimpleNamespace(**base)


def _materialize_input_path(raw: str, scratch_dirs: List[Path]) -> Path:
    path = Path(raw)
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="phase18_bundle_"))
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_dir)
        scratch_dirs.append(tmp_dir)
        roots = sorted([cand for cand in tmp_dir.iterdir() if cand.is_dir()])
        if len(roots) == 1:
            return roots[0]
        return tmp_dir
    return path


def _iter_dataset_roots(inputs: Sequence[str], scratch_dirs: List[Path]) -> Iterable[Path]:
    seen = set()
    for raw in inputs:
        materialized = _materialize_input_path(raw, scratch_dirs)
        for ds in _discover_dataset_dirs(materialized):
            if ds in seen:
                continue
            seen.add(ds)
            yield ds


def main() -> None:
    ap = argparse.ArgumentParser(description="Build phase18 pivot-level action-utility dataset for controller learning.")
    ap.add_argument("--input", dest="inputs", action="append", required=True, help="Dataset dir or parent directory. May be repeated.")
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--out_flat_jsonl", type=Path, default=None)
    ap.add_argument("--none_mode", choices=["agent_fold", "similarity_only"], default="agent_fold")
    ap.add_argument("--token_weight", type=float, default=0.10)
    ap.add_argument("--coverage_weight", type=float, default=0.15)
    ap.add_argument("--leakage_weight", type=float, default=0.0)
    ap.add_argument("--dev_ratio", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=7)
    ap.add_argument("--max_threads", type=int, default=0)
    ap.add_argument("--max_pivots", type=int, default=0)
    ap.add_argument("--traceops_similarity_topk", type=int, default=8)
    ap.add_argument("--goc_applicability_seed_topk", type=int, default=8)
    ap.add_argument("--goc_depwalk_hops", type=int, default=2)
    ap.add_argument("--goc_depwalk_topk_per_hop", type=int, default=6)
    ap.add_argument("--fork_k", type=int, default=6)
    ap.add_argument("--fork_max_tokens", type=int, default=160)
    ap.add_argument("--fork_recent_active_n", type=int, default=4)
    ap.add_argument("--fork_dependency_hops", type=int, default=2)
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_flat_jsonl is not None:
        args.out_flat_jsonl.parent.mkdir(parents=True, exist_ok=True)

    overrides = {
        "traceops_similarity_topk": int(args.traceops_similarity_topk),
        "goc_applicability_seed_topk": int(args.goc_applicability_seed_topk),
        "goc_depwalk_hops": int(args.goc_depwalk_hops),
        "goc_depwalk_topk_per_hop": int(args.goc_depwalk_topk_per_hop),
        "fork_k": int(args.fork_k),
        "fork_max_tokens": int(args.fork_max_tokens),
        "fork_recent_active_n": int(args.fork_recent_active_n),
        "fork_dependency_hops": int(args.fork_dependency_hops),
    }

    scratch_dirs: List[Path] = []
    dataset_roots = list(_iter_dataset_roots(args.inputs, scratch_dirs))
    if not dataset_roots:
        for scratch in scratch_dirs:
            shutil.rmtree(scratch, ignore_errors=True)
        joined = ", ".join(args.inputs)
        raise SystemExit(
            "No TraceOps dataset directories were found under the provided --input path(s): "
            f"{joined}. Expected a TraceOps directory containing data/traceops/threads.jsonl, "
            "or a phase bundle zip that contains those files."
        )

    written = 0
    flat_written = 0
    with args.out_jsonl.open("w", encoding="utf-8") as f_out:
        f_flat = args.out_flat_jsonl.open("w", encoding="utf-8") if args.out_flat_jsonl is not None else None
        try:
            for ds_root in dataset_roots:
                threads, meta = load_traceops_dataset(ds_root)
                if args.max_threads > 0:
                    threads = threads[: int(args.max_threads)]
                eval_args = _default_args_from_meta(meta, overrides)
                for pivot in iter_pivot_evals(
                    threads,
                    args=eval_args,
                    none_mode=str(args.none_mode),
                    token_weight=float(args.token_weight),
                    coverage_weight=float(args.coverage_weight),
                    leakage_weight=float(args.leakage_weight),
                    dev_ratio=float(args.dev_ratio),
                    split_seed=int(args.split_seed),
                ):
                    row = pivot_eval_to_dict(pivot)
                    row["dataset_root"] = str(ds_root)
                    row["meta"] = {
                        "scenario": meta.get("scenario") or meta.get("scenarios"),
                        "level": meta.get("level"),
                        "seed": meta.get("seed"),
                        "traceops_delay_to_relevance": meta.get("traceops_delay_to_relevance"),
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if f_flat is not None:
                        for action_name, action_obj in row["actions"].items():
                            flat = {
                                "thread_id": row["thread_id"],
                                "step_id": row["step_id"],
                                "step_idx": row["step_idx"],
                                "split": row["split"],
                                "dataset_root": row["dataset_root"],
                                "best_action": row["best_action"],
                                "best_utility": row["best_utility"],
                                "action": action_name,
                                "utility": action_obj["utility"],
                                "is_best_action": bool(action_name == row["best_action"]),
                                "features": row["features"],
                                "score": action_obj["score"],
                                "stats": action_obj["stats"],
                                "meta": row["meta"],
                            }
                            f_flat.write(json.dumps(flat, ensure_ascii=False) + "\n")
                            flat_written += 1
                    if args.max_pivots > 0 and written >= int(args.max_pivots):
                        break
                if args.max_pivots > 0 and written >= int(args.max_pivots):
                    break
        finally:
            if f_flat is not None:
                f_flat.close()
            for scratch in scratch_dirs:
                shutil.rmtree(scratch, ignore_errors=True)

    if written == 0:
        raise SystemExit(
            "TraceOps datasets were found, but no pivot rows were generated. "
            "Check whether the dataset contains pivot_check steps with gold annotations."
        )

    print(f"Wrote {written} pivot rows to {args.out_jsonl}")
    if args.out_flat_jsonl is not None:
        print(f"Wrote {flat_written} flat rows to {args.out_flat_jsonl}")


if __name__ == "__main__":
    main()
