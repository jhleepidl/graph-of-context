#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _latest_sweep_csv(split_dir: Path) -> Optional[Path]:
    cands = list((split_dir / "runs" / "context_budget_sweep").glob("*/results_context_budget_sweep.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_usage_avg(report_json_path: Path, method: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        obj = json.loads(report_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    method_report = (obj.get("method_reports") or {}).get(method) or {}
    usage = method_report.get("usage") or {}
    tool_calls = _to_float(usage.get("tool_calls_avg"))
    open_calls = _to_float(usage.get("open_calls_avg"))
    if tool_calls is None:
        total = _to_float(method_report.get("tool_calls"))
        count = _to_float((method_report.get("counts") or {}).get("n"))
        if total is not None and count and count > 0:
            tool_calls = total / count
    if open_calls is None:
        total = _to_float(method_report.get("open_calls"))
        count = _to_float((method_report.get("counts") or {}).get("n"))
        if total is not None and count and count > 0:
            open_calls = total / count
    return tool_calls, open_calls


def _resolve_report_json(path_str: str, sweep_csv_dir: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p
    rel = (sweep_csv_dir / path_str).resolve()
    if rel.is_file():
        return rel
    return None


def _best_method_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        budget = _to_int(row.get("budget"))
        if budget is None:
            continue
        grouped[(str(row.get("variant")), budget)].append(row)

    best_rows: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ranked = sorted(
            group,
            key=lambda r: (
                _to_float(r.get("judge_accuracy_packed")) or -1e9,
                _to_float(r.get("acc_per_1k_tokens")) or -1e9,
                str(r.get("method") or ""),
            ),
            reverse=True,
        )
        best_rows.append(ranked[0])
    return best_rows


def _plot_if_available(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    out_paths: List[Path] = []
    by_variant_method: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    variants = sorted({str(r.get("variant")) for r in rows})
    methods = sorted({str(r.get("method")) for r in rows})
    for row in rows:
        by_variant_method[(str(row.get("variant")), str(row.get("method")))].append(row)

    for variant in variants:
        # accuracy vs budget
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            seq = sorted(
                by_variant_method.get((variant, method), []),
                key=lambda r: _to_int(r.get("budget")) or 0,
            )
            xs = [_to_int(r.get("budget")) for r in seq]
            ys = [_to_float(r.get("judge_accuracy_packed")) for r in seq]
            xs2 = [x for x, y in zip(xs, ys) if x is not None and y is not None]
            ys2 = [y for y in ys if y is not None]
            if xs2 and ys2:
                ax.plot(xs2, ys2, marker="o", label=method)
        ax.set_title(f"{variant}: Accuracy vs Budget")
        ax.set_xlabel("Budget")
        ax.set_ylabel("judge_accuracy_packed")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p = out_dir / f"{variant}_accuracy_vs_budget.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out_paths.append(p)

        # acc_per_1k_tokens vs budget
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            seq = sorted(
                by_variant_method.get((variant, method), []),
                key=lambda r: _to_int(r.get("budget")) or 0,
            )
            xs = [_to_int(r.get("budget")) for r in seq]
            ys = [_to_float(r.get("acc_per_1k_tokens")) for r in seq]
            xs2 = [x for x, y in zip(xs, ys) if x is not None and y is not None]
            ys2 = [y for y in ys if y is not None]
            if xs2 and ys2:
                ax.plot(xs2, ys2, marker="o", label=method)
        ax.set_title(f"{variant}: acc_per_1k_tokens vs Budget")
        ax.set_xlabel("Budget")
        ax.set_ylabel("acc_per_1k_tokens")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p = out_dir / f"{variant}_acc_per_1k_tokens_vs_budget.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out_paths.append(p)
    return out_paths


def analyze_phase3(phase3_root: Path, out_dir: Path, split: str = "main") -> Dict[str, Any]:
    runs_root = phase3_root / "runs"
    variants = [p for p in runs_root.glob("*") if p.is_dir()]
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for variant_dir in sorted(variants):
        variant = variant_dir.name
        split_dir = variant_dir / split
        if not split_dir.exists():
            missing.append(f"{variant}:{split}:missing_split")
            continue
        sweep_csv = _latest_sweep_csv(split_dir)
        if sweep_csv is None:
            missing.append(f"{variant}:{split}:missing_sweep_csv")
            continue
        with sweep_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for rec in reader:
                budget = _to_int(rec.get("budget"))
                method = str(rec.get("method") or "")
                report_json_path = _resolve_report_json(str(rec.get("report_json") or ""), sweep_csv.parent)
                tool_calls, open_calls = (
                    _load_usage_avg(report_json_path, method) if report_json_path else (None, None)
                )
                row = {
                    "variant": variant,
                    "split": split,
                    "budget": budget,
                    "method": method,
                    "judge_accuracy_packed": _to_float(rec.get("judge_accuracy_packed")),
                    "packed_all_critical_rate": _to_float(rec.get("e3_packed_all_critical_rate")),
                    "e3_context_token_est_mean": _to_float(rec.get("e3_context_token_est_mean")),
                    "acc_per_1k_tokens": _to_float(rec.get("acc_per_1k_tokens")),
                    "wrong_branch_recall_rate_mean": _to_float(rec.get("wrong_branch_recall_rate_mean")),
                    "closure_recall_core_mean": _to_float(rec.get("closure_recall_core_mean")),
                    "tool_calls": tool_calls,
                    "open_calls": open_calls,
                    "report_json": str(report_json_path) if report_json_path else str(rec.get("report_json") or ""),
                }
                rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase3_summary.csv"
    fieldnames = [
        "variant",
        "budget",
        "method",
        "judge_accuracy_packed",
        "packed_all_critical_rate",
        "e3_context_token_est_mean",
        "acc_per_1k_tokens",
        "wrong_branch_recall_rate_mean",
        "closure_recall_core_mean",
        "tool_calls",
        "open_calls",
        "report_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    best_rows = _best_method_rows(rows)
    goc_wins = [r for r in best_rows if str(r.get("method")) == "goc"]
    win_by_variant: Dict[str, List[int]] = defaultdict(list)
    for row in goc_wins:
        budget = _to_int(row.get("budget"))
        if budget is not None:
            win_by_variant[str(row.get("variant"))].append(budget)

    md_lines: List[str] = []
    md_lines.append("# Phase3 Summary")
    md_lines.append("")
    md_lines.append("## Best Method Per Budget Per Variant")
    md_lines.append("")
    md_lines.append("| Variant | Budget | Best Method | Accuracy | acc_per_1k_tokens |")
    md_lines.append("|---|---:|---|---:|---:|")
    for row in best_rows:
        md_lines.append(
            "| {variant} | {budget} | {method} | {acc:.4f} | {eff:.4f} |".format(
                variant=row.get("variant"),
                budget=row.get("budget"),
                method=row.get("method"),
                acc=_to_float(row.get("judge_accuracy_packed")) or 0.0,
                eff=_to_float(row.get("acc_per_1k_tokens")) or 0.0,
            )
        )
    md_lines.append("")
    md_lines.append("## Where GoC Wins")
    if not goc_wins:
        md_lines.append("- GoC was not top-ranked on judge_accuracy_packed across analyzed budgets.")
    else:
        for variant in sorted(win_by_variant.keys()):
            budgets = ", ".join(str(b) for b in sorted(win_by_variant[variant]))
            md_lines.append(f"- {variant}: GoC best at budgets [{budgets}].")
        md_lines.append(
            f"- Overall GoC wins: {len(goc_wins)} / {len(best_rows)} variant-budget cells."
        )
    if missing:
        md_lines.append("")
        md_lines.append("## Missing Artifacts")
        for item in missing:
            md_lines.append(f"- {item}")

    plot_paths = _plot_if_available(rows, out_dir)
    if plot_paths:
        md_lines.append("")
        md_lines.append("## Plots")
        for path in plot_paths:
            md_lines.append(f"- {path.name}")

    md_path = out_dir / "phase3_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "phase3_root": str(phase3_root),
        "split": split,
        "rows": len(rows),
        "variants": sorted({str(r.get("variant")) for r in rows}),
        "best_rows": len(best_rows),
        "goc_wins": len(goc_wins),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "plot_paths": [str(p) for p in plot_paths],
        "missing": missing,
    }
    summary_path = out_dir / "phase3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase3 PolicyOps results across variants.")
    parser.add_argument("--phase3_root", required=True, type=Path, help="Path to <bundle>/phase3")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output dir (default: <phase3_root>/analysis)")
    parser.add_argument("--split", type=str, default="main", choices=["main", "dev"])
    args = parser.parse_args()

    phase3_root = args.phase3_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (phase3_root / "analysis")
    summary = analyze_phase3(phase3_root, out_dir, split=args.split)
    print(f"PHASE3_SUMMARY_CSV={summary['csv_path']}")
    print(f"PHASE3_SUMMARY_MD={summary['md_path']}")
    print(f"PHASE3_SUMMARY_JSON={out_dir / 'phase3_summary.json'}")
    print(f"ROWS={summary['rows']}")
    print(f"GOC_WINS={summary['goc_wins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

