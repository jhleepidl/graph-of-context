#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def _mean_bool(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for rec in records:
        b = _to_bool(rec.get(key))
        if b is None:
            continue
        vals.append(1.0 if b else 0.0)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _mean_numeric(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for rec in records:
        v = _to_float(rec.get(key))
        if v is None:
            continue
        vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def _read_manifest(phase7_root: Path) -> List[Dict[str, Any]]:
    manifest_path = phase7_root / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def _compute_row(
    *,
    pivot_type: str,
    method: str,
    goc_k: Optional[int],
    goc_h: Optional[int],
    method_report: Dict[str, Any],
) -> Dict[str, Any]:
    records = list(method_report.get("records") or [])
    final_records = [r for r in records if _to_int(r.get("episode_id")) == 3]
    pivot_final_records = [r for r in final_records if bool(r.get("is_pivot_task"))]
    metrics = method_report.get("metrics", {}) or {}
    unfolded_node_count_mean = _to_float(metrics.get("goc_unfolded_node_count_mean"))
    if unfolded_node_count_mean is None:
        unfolded_node_count_mean = _mean_numeric(records, "goc_unfolded_node_count")
    return {
        "pivot_type": pivot_type,
        "method": method,
        "goc_K": goc_k if method == "goc" else None,
        "goc_H": goc_h if method == "goc" else None,
        "final_tasks": len(final_records),
        "pivot_final_tasks": len(pivot_final_records),
        "final_accuracy": _mean_bool(final_records, "judge_correct"),
        "final_pivot_correct_updated_rate": _mean_bool(
            pivot_final_records, "judge_correct"
        ),
        "e3_context_token_est_mean": _to_float(metrics.get("e3_context_token_est_mean")),
        "unfolded_node_count_mean": unfolded_node_count_mean,
    }


def _pareto_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        r
        for r in rows
        if _to_float(r.get("e3_context_token_est_mean")) is not None
        and _to_float(r.get("final_pivot_correct_updated_rate")) is not None
    ]
    candidates.sort(
        key=lambda r: (
            _to_float(r.get("e3_context_token_est_mean")) or 1e18,
            -(_to_float(r.get("final_pivot_correct_updated_rate")) or -1.0),
            _to_int(r.get("goc_K")) or 0,
            _to_int(r.get("goc_H")) or 0,
        )
    )
    best_acc = -1.0
    out: List[Dict[str, Any]] = []
    for row in candidates:
        acc = _to_float(row.get("final_pivot_correct_updated_rate"))
        if acc is None:
            continue
        if acc >= best_acc - 1e-12:
            out.append(row)
            if acc > best_acc:
                best_acc = acc
    return out


def analyze(phase7_root: Path, out_dir: Path) -> Tuple[Path, Path, Path, Path, List[Dict[str, Any]]]:
    manifest_runs = _read_manifest(phase7_root)
    summary_rows: List[Dict[str, Any]] = []

    for run in manifest_runs:
        if not bool(run.get("success")):
            continue
        report_json_val = run.get("report_json")
        if not report_json_val:
            continue
        report_path = Path(str(report_json_val))
        if not report_path.exists():
            continue
        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        method_reports = report_obj.get("method_reports", {}) or {}
        pivot_type = str(run.get("pivot_type") or "unknown")
        run_type = str(run.get("run_type") or "")
        goc_k = _to_int(run.get("goc_K"))
        goc_h = _to_int(run.get("goc_H"))
        methods: List[str]
        if run_type == "baseline":
            methods = ["full", "similarity_only"]
        else:
            methods = ["goc"]
        for method in methods:
            method_report = method_reports.get(method)
            if not isinstance(method_report, dict):
                continue
            summary_rows.append(
                _compute_row(
                    pivot_type=pivot_type,
                    method=method,
                    goc_k=goc_k,
                    goc_h=goc_h,
                    method_report=method_report,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "phase7_summary.csv"
    summary_md = out_dir / "phase7_summary.md"
    pareto_json = out_dir / "phase7_pareto.json"
    fieldnames = [
        "pivot_type",
        "method",
        "goc_K",
        "goc_H",
        "final_tasks",
        "pivot_final_tasks",
        "final_accuracy",
        "final_pivot_correct_updated_rate",
        "e3_context_token_est_mean",
        "unfolded_node_count_mean",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    by_pivot: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        by_pivot.setdefault(str(row.get("pivot_type") or "unknown"), []).append(row)

    md_lines: List[str] = []
    md_lines.append("# Phase 7 Unfold Controls Summary")
    md_lines.append("")
    md_lines.append("- Objectives: maximize final pivot accuracy, minimize context tokens.")
    md_lines.append("")

    pareto_payload: Dict[str, Any] = {}
    for pivot_type in sorted(by_pivot.keys()):
        rows = by_pivot[pivot_type]
        baseline_rows = [r for r in rows if r.get("method") in {"full", "similarity_only"}]
        goc_rows = [r for r in rows if r.get("method") == "goc"]
        goc_rows_sorted = sorted(
            goc_rows,
            key=lambda r: (
                _to_int(r.get("goc_K")) or 0,
                _to_int(r.get("goc_H")) or 0,
            ),
        )
        pareto_rows = _pareto_rows(goc_rows)
        pareto_payload[pivot_type] = {
            "pareto_rows": pareto_rows,
            "best_pareto": (
                sorted(
                    pareto_rows,
                    key=lambda r: (
                        -(_to_float(r.get("final_pivot_correct_updated_rate")) or -1.0),
                        _to_float(r.get("e3_context_token_est_mean")) or 1e18,
                    ),
                )[0]
                if pareto_rows
                else None
            ),
        }

        md_lines.append(f"## {pivot_type}")
        md_lines.append("")
        md_lines.append("### Baselines @ external budget=4000")
        md_lines.append("")
        if baseline_rows:
            md_lines.append("|method|final_accuracy|final_pivot_correct_updated_rate|tokens|")
            md_lines.append("|---|---:|---:|---:|")
            for row in sorted(baseline_rows, key=lambda r: str(r.get("method"))):
                md_lines.append(
                    "|{m}|{a}|{p}|{t}|".format(
                        m=row.get("method"),
                        a=_fmt_float(_to_float(row.get("final_accuracy"))),
                        p=_fmt_float(_to_float(row.get("final_pivot_correct_updated_rate"))),
                        t=_fmt_float(_to_float(row.get("e3_context_token_est_mean"))),
                    )
                )
        else:
            md_lines.append("- No baseline rows found.")
        md_lines.append("")

        md_lines.append("### GoC Grid (K,H)")
        md_lines.append("")
        if goc_rows_sorted:
            md_lines.append("|K|H|final_accuracy|final_pivot_correct_updated_rate|tokens|unfolded_node_count_mean|")
            md_lines.append("|---:|---:|---:|---:|---:|---:|")
            for row in goc_rows_sorted:
                md_lines.append(
                    "|{k}|{h}|{a}|{p}|{t}|{u}|".format(
                        k=_to_int(row.get("goc_K")),
                        h=_to_int(row.get("goc_H")),
                        a=_fmt_float(_to_float(row.get("final_accuracy"))),
                        p=_fmt_float(_to_float(row.get("final_pivot_correct_updated_rate"))),
                        t=_fmt_float(_to_float(row.get("e3_context_token_est_mean"))),
                        u=_fmt_float(_to_float(row.get("unfolded_node_count_mean"))),
                    )
                )
        else:
            md_lines.append("- No GoC rows found.")
        md_lines.append("")

        md_lines.append("### Pareto-Optimal GoC Configs")
        md_lines.append("")
        if pareto_rows:
            md_lines.append("|K|H|final_pivot_correct_updated_rate|tokens|")
            md_lines.append("|---:|---:|---:|---:|")
            for row in pareto_rows:
                md_lines.append(
                    "|{k}|{h}|{p}|{t}|".format(
                        k=_to_int(row.get("goc_K")),
                        h=_to_int(row.get("goc_H")),
                        p=_fmt_float(_to_float(row.get("final_pivot_correct_updated_rate"))),
                        t=_fmt_float(_to_float(row.get("e3_context_token_est_mean"))),
                    )
                )
        else:
            md_lines.append("- No Pareto rows found.")
        md_lines.append("")

    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    pareto_json.write_text(json.dumps(pareto_payload, indent=2), encoding="utf-8")

    return summary_csv, summary_md, pareto_json, phase7_root / "run_manifest.json", summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase 7 unfold-control experiments.")
    parser.add_argument("--phase7_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    phase7_root = args.phase7_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (phase7_root / "analysis")
    summary_csv, summary_md, pareto_json, manifest_path, rows = analyze(phase7_root, out_dir)
    print(f"MANIFEST={manifest_path}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SUMMARY_MD={summary_md}")
    print(f"PARETO_JSON={pareto_json}")
    print(f"ROWS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
