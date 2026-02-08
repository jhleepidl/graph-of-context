#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METHOD_ORDER = [
    "similarity_only",
    "goc_v1",
    "goc_no_activity",
    "goc_no_mmr",
    "goc_no_anchor",
]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "none":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    return int(round(f))


def _latest_sweep_csv(method_dir: Path) -> Optional[Path]:
    cands = list((method_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def collect_rows(phase4_root: Path) -> List[Dict[str, Any]]:
    runs_root = phase4_root / "runs"
    rows: List[Dict[str, Any]] = []
    for variant_dir in sorted([p for p in runs_root.glob("*") if p.is_dir()]):
        variant = variant_dir.name
        for method_dir in sorted([p for p in variant_dir.glob("*") if p.is_dir()]):
            method_label = method_dir.name
            sweep_csv = _latest_sweep_csv(method_dir)
            if sweep_csv is None:
                continue
            with sweep_csv.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for rec in reader:
                    budget = _to_int(rec.get("budget"))
                    row = {
                        "variant": variant,
                        "budget": budget,
                        "method": method_label,
                        "judge_accuracy_packed": _to_float(rec.get("judge_accuracy_packed")),
                        "wrong_branch_recall_rate_mean": _to_float(rec.get("wrong_branch_recall_rate_mean")),
                        "closure_recall_core_mean": _to_float(rec.get("closure_recall_core_mean")),
                        "packed_all_critical_rate": _to_float(rec.get("e3_packed_all_critical_rate")),
                        "source_sweep_csv": str(sweep_csv),
                    }
                    rows.append(row)
    return rows


def _mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "budget",
        "method",
        "judge_accuracy_packed",
        "wrong_branch_recall_rate_mean",
        "closure_recall_core_mean",
        "packed_all_critical_rate",
        "source_sweep_csv",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                str(r.get("variant")),
                int(r.get("budget") or 0),
                METHOD_ORDER.index(str(r.get("method")))
                if str(r.get("method")) in METHOD_ORDER
                else 999,
                str(r.get("method")),
            ),
        ):
            writer.writerow({k: row.get(k) for k in fieldnames})


def _infer_v3_conclusion(rows: List[Dict[str, Any]]) -> str:
    v3_rows = [r for r in rows if str(r.get("variant")) == "V3_n20_docs"]
    if not v3_rows:
        return "V3_n20_docs rows are missing, so a policy-vs-graph conclusion cannot be inferred."

    by_method_acc: Dict[str, List[Optional[float]]] = defaultdict(list)
    by_method_closure: Dict[str, List[Optional[float]]] = defaultdict(list)
    for row in v3_rows:
        method = str(row.get("method"))
        by_method_acc[method].append(_to_float(row.get("judge_accuracy_packed")))
        by_method_closure[method].append(_to_float(row.get("closure_recall_core_mean")))

    mean_acc = {m: _mean(vals) for m, vals in by_method_acc.items()}
    mean_closure = {m: _mean(vals) for m, vals in by_method_closure.items()}
    sim_acc = mean_acc.get("similarity_only")
    goc_v1_acc = mean_acc.get("goc_v1")
    goc_variants = ["goc_v1", "goc_no_activity", "goc_no_mmr", "goc_no_anchor"]
    goc_best_method = None
    goc_best_acc = None
    for m in goc_variants:
        v = mean_acc.get(m)
        if v is None:
            continue
        if goc_best_acc is None or v > goc_best_acc:
            goc_best_acc = v
            goc_best_method = m
    if goc_best_acc is None:
        return "No GoC ablation rows found for V3_n20_docs."

    policy_delta = None
    if goc_v1_acc is not None:
        policy_delta = goc_best_acc - goc_v1_acc

    closure_vals = [mean_closure.get(m) for m in goc_variants if mean_closure.get(m) is not None]
    closure_mean = _mean(closure_vals)

    if policy_delta is not None and policy_delta >= 0.05:
        return (
            f"V3 appears policy-driven: best ablation ({goc_best_method}) improves GoC by "
            f"{policy_delta:.3f} accuracy over goc_v1."
        )
    if sim_acc is not None and goc_best_acc < sim_acc - 0.08 and (closure_mean is not None and closure_mean < 0.6):
        return (
            "V3 appears graph/closure-driven: all GoC ablations remain below similarity_only "
            "and closure_recall_core stays low."
        )
    return (
        "V3 appears mixed: policy ablations change outcomes only modestly, suggesting both "
        "selection policy and closure quality contribute."
    )


def _write_md(rows: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Phase4 Ablation Analysis")
    lines.append("")
    lines.append("## Metrics Table")
    lines.append("")
    lines.append(
        "| Variant | Budget | Method | judge_accuracy_packed | wrong_branch_recall_rate_mean | closure_recall_core_mean | packed_all_critical_rate |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for row in sorted(
        rows,
        key=lambda r: (
            str(r.get("variant")),
            int(r.get("budget") or 0),
            METHOD_ORDER.index(str(r.get("method")))
            if str(r.get("method")) in METHOD_ORDER
            else 999,
            str(r.get("method")),
        ),
    ):
        lines.append(
            "| {variant} | {budget} | {method} | {acc:.4f} | {wrr:.4f} | {cr:.4f} | {pac:.4f} |".format(
                variant=row.get("variant"),
                budget=row.get("budget"),
                method=row.get("method"),
                acc=_to_float(row.get("judge_accuracy_packed")) or 0.0,
                wrr=_to_float(row.get("wrong_branch_recall_rate_mean")) or 0.0,
                cr=_to_float(row.get("closure_recall_core_mean")) or 0.0,
                pac=_to_float(row.get("packed_all_critical_rate")) or 0.0,
            )
        )
    lines.append("")
    lines.append("## V3 Diagnosis")
    lines.append(f"- {_infer_v3_conclusion(rows)}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase4 ablation compare outputs.")
    parser.add_argument("--phase4_root", required=True, type=Path, help="Path to <bundle>/phase4")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output dir (default: <phase4_root>/analysis)")
    args = parser.parse_args()

    phase4_root = args.phase4_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (phase4_root / "analysis")
    rows = collect_rows(phase4_root)
    out_csv = out_dir / "phase4_ablation_summary.csv"
    out_md = out_dir / "phase4_ablation_summary.md"
    _write_csv(rows, out_csv)
    _write_md(rows, out_md)
    print(f"PHASE4_ABLATION_CSV={out_csv}")
    print(f"PHASE4_ABLATION_MD={out_md}")
    print(f"ROWS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

