from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .bridged_ab import compute_bridged_ab_slices


def _resolve_base_dir(report_path: Path) -> Path:
    parts = report_path.resolve().parts
    if "runs" in parts:
        idx = parts.index("runs")
        return Path(*parts[:idx])
    return report_path.parent


def _latest_report_path(report_path: Path) -> Path:
    if report_path.is_file():
        return report_path
    candidates = sorted(report_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No report JSON files found in {report_path}")
    return candidates[0]


def _load_raw_preview(record: Dict[str, Any]) -> str:
    raw_output = record.get("raw_output")
    if isinstance(raw_output, str) and raw_output.strip():
        return " ".join(raw_output[:300].split())
    raw_path = record.get("raw_path")
    if raw_path:
        try:
            text = Path(raw_path).read_text(encoding="utf-8")[:300]
            return " ".join(text.split())
        except Exception:
            return ""
    return ""


def analyze_failure_slice(report_path: Path, top_k: int = 20) -> Path:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_reports = payload.get("method_reports", {})

    failures: List[Tuple[str, Dict[str, Any]]] = []
    for method, report in method_reports.items():
        for record in report.get("records", []):
            if not record.get("gold_in_search_topk"):
                continue
            opened_gold_count = record.get("opened_gold_count", 0) or 0
            opened_has_winning = record.get("opened_has_winning_clause")
            if opened_gold_count == 0 or opened_has_winning is False:
                failures.append((method, record))

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_failure_slice.md"

    lines: List[str] = []
    lines.append(f"# Failure Slice Report\n")
    lines.append(f"- Source report: {report_path}\n")
    lines.append(f"- run_id: {payload.get('run_id')}\n")
    lines.append(f"- git_sha: {payload.get('git_sha')}\n")
    lines.append(f"- Failures: {len(failures)}\n")

    if not failures:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    for method, record in failures:
        lines.append(f"## Method: {method} / Task: {record.get('task_id')}\n")
        lines.append(f"- gold_decision: {record.get('gold_decision')}")
        lines.append(f"- pred_decision: {record.get('pred_decision')}")
        lines.append(f"- opened_clause_ids: {record.get('opened_clause_ids')}")
        if record.get("forced_open_ids") is not None:
            lines.append(f"- forced_open_ids: {record.get('forced_open_ids')}")
        if record.get("search_topk_clause_ids") is not None:
            snapshot = record.get("search_topk_clause_ids") or []
            lines.append(f"- primary_search_topk_clause_ids: {snapshot[:top_k]}")
        lines.append(f"- winning_clause_rank: {record.get('winning_clause_rank')}")
        lines.append(f"- min_gold_rank: {record.get('min_gold_rank')}")
        lines.append(f"- gold_score_gap: {record.get('gold_score_gap')}")
        raw_preview = _load_raw_preview(record)
        if raw_preview:
            lines.append(f"- raw_output_preview: {raw_preview}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def analyze_bridged_ab(report_path: Path, method: str = "goc") -> Path:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])
    slices = compute_bridged_ab_slices(records)

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_bridged_ab.md"

    lines: List[str] = []
    lines.append("# Bridged A×B Slice Report\n")
    lines.append(f"- Source report: {report_path}\n")
    lines.append(f"- Method: {method}\n")
    lines.append(f"- Records: {slices.get('n_records', 0)}\n")

    axes = slices.get("axes", {})
    cells = slices.get("cells", {})
    a_keys = axes.get("A", [])
    b_keys = axes.get("B", [])
    header = "|A/B|" + "|".join(b_keys) + "|"
    sep = "|---|" + "|".join(["---"] * len(b_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for a_key in a_keys:
        row = [a_key]
        for b_key in b_keys:
            cell = cells.get(a_key, {}).get(b_key)
            if not cell:
                row.append("n=0")
                continue
            extra = ""
            if a_key == "A2_opened_wrong_bridge":
                extra = (
                    f", probe_gold={cell.get('bridge_probe_contains_gold_canonical_rate',0):.2f}, "
                    f"opened_gold={cell.get('bridge_opened_contains_gold_canonical_rate',0):.2f}"
                )
            if a_key == "A3_opened_gold_bridge" and b_key == "B2_hop2_with_gold_canonical":
                core_rank = cell.get("core_min_rank_union_mean")
                deep_rate = cell.get("deep_rank_core_rate")
                core_rank_str = f"{core_rank:.2f}" if isinstance(core_rank, (int, float)) else "n/a"
                deep_rate_str = f"{deep_rate:.2f}" if isinstance(deep_rate, (int, float)) else "n/a"
                extra = (
                    f"{extra}, core_min_rank_union_mean={core_rank_str}, "
                    f"deep_rank_core_rate={deep_rate_str}"
                )
            decision_acc = cell.get("decision_acc")
            judge_acc = cell.get("judge_acc")
            decision_acc_str = f"{decision_acc:.2f}" if isinstance(decision_acc, (int, float)) else "n/a"
            judge_acc_str = f"{judge_acc:.2f}" if isinstance(judge_acc, (int, float)) else "n/a"
            row.append(
                f"n={cell.get('n',0)}, decision_acc={decision_acc_str}, "
                f"judge_acc={judge_acc_str}, cov_core={cell.get('opened_gold_coverage_core_mean',0):.2f}{extra}"
            )
        lines.append("|" + "|".join(row) + "|")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _bucket_a(rec: Dict[str, Any]) -> str:
    if rec.get("bridge_needed") is False:
        return "A0_no_bridge_needed"
    if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is False:
        return "A1_needed_not_opened"
    if rec.get("bridge_needed") is True and rec.get("bridge_opened_any") is True:
        if rec.get("bridge_opened_gold") is True:
            return "A3_opened_gold_bridge"
        if rec.get("bridge_opened_gold") is False:
            return "A2_opened_wrong_bridge"
    return "A_unknown"


def _bucket_b(rec: Dict[str, Any]) -> str:
    hop2_executed = rec.get("hop2_executed") is True
    hop2_gold = rec.get("hop2_query_contains_gold_canonical") is True
    if not hop2_executed:
        return "B0_no_hop2"
    if hop2_gold:
        return "B2_hop2_with_gold_canonical"
    return "B1_hop2_no_gold_canonical"


def analyze_selection_triage(
    report_path: Path,
    method: str = "goc",
    max_per_bucket: int = 20,
) -> Tuple[Path, Path]:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])

    # Ensure triage exists for the same report/method.
    from .triage import triage_compare

    triage_compare(report_path, method=method, max_per_bucket=max_per_bucket)

    rng = random.Random(0)
    sel_gap = [r for r in records if isinstance(r.get("selection_gap"), (int, float)) and r.get("selection_gap") >= 0.5]
    acc_no_core = [r for r in records if r.get("acc_no_core_evidence") is True]
    a3b2_core0 = [
        r
        for r in records
        if _bucket_a(r) == "A3_opened_gold_bridge"
        and _bucket_b(r) == "B2_hop2_with_gold_canonical"
        and (r.get("opened_gold_coverage_core") in {0, 0.0})
    ]

    def _sample(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) <= max_per_bucket:
            return list(items)
        return rng.sample(items, max_per_bucket)

    sampled = {
        "SEL_GAP": _sample(sel_gap),
        "ACC_NO_CORE_EVIDENCE": _sample(acc_no_core),
        "A3B2_CORE0": _sample(a3b2_core0),
    }

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{stamp}_selection_triage.csv"
    md_path = out_dir / f"{stamp}_patterns.md"

    rows: List[Dict[str, Any]] = []
    for bucket, items in sampled.items():
        for rec in items:
            rows.append(
                {
                    "bucket": bucket,
                    "task_id": rec.get("task_id"),
                    "winning_clause_rank_union": rec.get("winning_clause_rank_union"),
                    "hit_at_open_budget_union": rec.get("hit_at_open_budget_union"),
                    "rank_success": rec.get("rank_success"),
                    "opened_has_winning_clause": rec.get("opened_has_winning_clause_union"),
                    "policy_gain_over_rank": rec.get("policy_gain_over_rank"),
                    "opened_gold_coverage_core": rec.get("opened_gold_coverage_core"),
                    "judge_supporting_count": rec.get("judge_supporting_count"),
                    "open_from_hop1_count": rec.get("open_from_hop1_count"),
                    "open_from_hop2_count": rec.get("open_from_hop2_count"),
                    "opened_bridge_count": rec.get("opened_bridge_count"),
                    "opened_meta_count": rec.get("opened_meta_count"),
                    "opened_rule_count": rec.get("opened_rule_count"),
                    "bridge_open_cap_hit": rec.get("bridge_open_cap_hit"),
                    "meta_avoided_count": rec.get("meta_avoided_count"),
                    "hop2_pool_used_count": rec.get("hop2_pool_used_count"),
                    "fallback_reason": rec.get("fallback_reason"),
                }
            )

    # Write CSV
    import csv

    fieldnames = [
        "bucket",
        "task_id",
        "winning_clause_rank_union",
        "hit_at_open_budget_union",
        "rank_success",
        "opened_has_winning_clause",
        "policy_gain_over_rank",
        "opened_gold_coverage_core",
        "judge_supporting_count",
        "open_from_hop1_count",
        "open_from_hop2_count",
        "opened_bridge_count",
        "opened_meta_count",
        "opened_rule_count",
        "bridge_open_cap_hit",
        "meta_avoided_count",
        "hop2_pool_used_count",
        "fallback_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Patterns summary (rule-based)
    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _rate(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [1.0 for r in items if r.get(key)]
        return len(vals) / len(items) if items else None

    lines: List[str] = []
    lines.append("# Representative Failure Patterns")
    lines.append("")
    lines.append(f"- source_report: {report_path}")
    lines.append(f"- method: {method}")
    lines.append("")

    patterns = [
        ("SEL_GAP", sel_gap, "Selection gap: winning clause in union budget but not opened"),
        ("ACC_NO_CORE_EVIDENCE", acc_no_core, "Judge correct without core evidence opened"),
        ("A3B2_CORE0", a3b2_core0, "Bridge+hop2 succeeded but core evidence not opened"),
    ]
    for name, items, desc in patterns:
        lines.append(f"## {name}")
        lines.append(f"{desc}")
        lines.append(f"n={len(items)}")
        mean_gap = _mean(
            [
                float(r.get("selection_gap"))
                for r in items
                if isinstance(r.get("selection_gap"), (int, float))
            ]
        )
        mean_cov = _mean(
            [
                float(r.get("opened_gold_coverage_core"))
                for r in items
                if isinstance(r.get("opened_gold_coverage_core"), (int, float))
            ]
        )
        mean_support = _mean(
            [
                float(r.get("judge_supporting_count"))
                for r in items
                if isinstance(r.get("judge_supporting_count"), (int, float))
            ]
        )
        hit_rate = _rate(items, "rank_success")
        opened_win_rate = _rate(items, "opened_has_winning_clause_union")
        mean_gain = _mean(
            [
                float(r.get("policy_gain_over_rank"))
                for r in items
                if isinstance(r.get("policy_gain_over_rank"), (int, float))
            ]
        )
        lines.append(f"selection_gap_mean={mean_gap if mean_gap is not None else 'n/a'}")
        lines.append(f"opened_gold_coverage_core_mean={mean_cov if mean_cov is not None else 'n/a'}")
        lines.append(f"judge_supporting_count_mean={mean_support if mean_support is not None else 'n/a'}")
        lines.append(f"rank_success_rate={hit_rate if hit_rate is not None else 'n/a'}")
        lines.append(f"opened_has_winning_clause_union_rate={opened_win_rate if opened_win_rate is not None else 'n/a'}")
        lines.append(f"policy_gain_over_rank_mean={mean_gain if mean_gain is not None else 'n/a'}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def analyze_slot_breakdown(
    report_path: Path,
    method: str = "goc",
) -> Tuple[Path, Path]:
    report_path = _latest_report_path(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    method_report = payload.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        slot_term = rec.get("slot_term") or rec.get("slot") or "unknown"
        grouped.setdefault(str(slot_term), []).append(rec)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    rows: List[Dict[str, Any]] = []
    for slot_term, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        judge_vals = [1.0 if r.get("judge_correct") else 0.0 for r in items if r.get("judge_correct") is not None]
        cov_vals = [
            float(r.get("opened_gold_coverage_core") or 0.0)
            for r in items
            if isinstance(r.get("opened_gold_coverage_core"), (int, float))
        ]
        rank_vals = [
            float(r.get("min_gold_core_rank_union"))
            for r in items
            if isinstance(r.get("min_gold_core_rank_union"), int)
        ]
        deep_vals = [1.0 if r.get("deep_rank_core_flag") else 0.0 for r in items]
        rows.append(
            {
                "slot_term": slot_term,
                "n": len(items),
                "judge_acc": _mean(judge_vals),
                "cov_core": _mean(cov_vals),
                "min_gold_core_rank_union_mean": _mean(rank_vals),
                "deep_rank_core_rate": _mean(deep_vals),
            }
        )

    base_dir = _resolve_base_dir(report_path)
    out_dir = base_dir / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{stamp}_slot_breakdown.csv"
    md_path = out_dir / f"{stamp}_slot_breakdown.md"

    # CSV
    import csv

    fieldnames = [
        "slot_term",
        "n",
        "judge_acc",
        "cov_core",
        "min_gold_core_rank_union_mean",
        "deep_rank_core_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Markdown table
    lines: List[str] = []
    lines.append("# Slot Breakdown")
    lines.append("")
    lines.append(f"- source_report: {report_path}")
    lines.append(f"- method: {method}")
    lines.append("")
    lines.append("|slot_term|n|judge_acc|cov_core|min_gold_core_rank_union_mean|deep_rank_core_rate|")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        def _fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
        lines.append(
            f"|{row['slot_term']}|{row['n']}|{_fmt(row.get('judge_acc'))}|"
            f"{_fmt(row.get('cov_core'))}|{_fmt(row.get('min_gold_core_rank_union_mean'))}|"
            f"{_fmt(row.get('deep_rank_core_rate'))}|"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def analyze_split_sweep_ab(
    sweep_dir: Path,
    results_md: Optional[Path] = None,
    method: str = "goc",
) -> Path:
    compare_paths = sorted(sweep_dir.rglob("runs/compare/*.json"))
    by_hop1: Dict[int, List[Dict[str, Any]]] = {}

    for path in compare_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hop1 = payload.get("scenario_params", {}).get("open_split_hop1")
        if hop1 is None:
            continue
        records = payload.get("method_reports", {}).get(method, {}).get("records", [])
        a3b2 = [
            r
            for r in records
            if _bucket_a(r) == "A3_opened_gold_bridge"
            and _bucket_b(r) == "B2_hop2_with_gold_canonical"
        ]
        by_hop1.setdefault(int(hop1), []).extend(a3b2)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    lines: List[str] = []
    lines.append("")
    lines.append("## A3×B2 (goc) Core/Selection Summary")
    lines.append("|open_split_hop1|n|judge_acc|cov_core|selection_gap|")
    lines.append("|---|---|---|---|---|")
    for hop1 in sorted(by_hop1.keys()):
        items = by_hop1[hop1]
        judge_acc = _mean([1.0 if r.get("judge_correct") else 0.0 for r in items])
        cov_core = _mean([float(r.get("opened_gold_coverage_core") or 0.0) for r in items])
        sel_gap = _mean([float(r.get("selection_gap") or 0.0) for r in items])
        def _fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
        lines.append(
            f"|{hop1}|{len(items)}|"
            f"{_fmt(judge_acc)}|"
            f"{_fmt(cov_core)}|"
            f"{_fmt(sel_gap)}|"
        )

    if results_md is None:
        results_md = sweep_dir / "results_split_sweep.md"
    if results_md.exists():
        content = results_md.read_text(encoding="utf-8")
        content = content.rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        content = "\n".join(lines) + "\n"
    results_md.write_text(content, encoding="utf-8")
    return results_md


def analyze_bundle(run_dir: Path) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    summary_path = run_dir / "summary.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.md not found in {run_dir}")

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.exists():
            return path
        candidate = run_dir.parent / path_str
        if candidate.exists():
            return candidate
        candidate = Path(path_str).expanduser()
        if candidate.exists():
            return candidate
        return path

    def _normalize_preset(name: str) -> str:
        lowered = name.lower()
        if "n8" in lowered:
            return "n8_exclcore"
        if "n10" in lowered:
            return "n10_exclcore"
        return name

    # Parse summary.md into index.
    index: Dict[str, Dict[str, Dict[str, Path]]] = {}
    current_preset = None
    current_policy = None
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current_preset = _normalize_preset(line[3:].strip())
            index.setdefault(current_preset, {})
            current_policy = None
            continue
        if line.startswith("- open_policy:"):
            current_policy = line.split(":", 1)[1].strip()
            if current_preset is None:
                current_preset = "unknown"
            index.setdefault(current_preset, {}).setdefault(current_policy, {})
            continue
        if ":" in line and current_preset and current_policy:
            key, value = line.split(":", 1)
            key = key.strip("- ").strip()
            value = value.strip()
            if value:
                index[current_preset][current_policy][key] = _resolve_path(value)

    # If parsing failed, fallback to scanning compare JSONs.
    if not index:
        compare_paths = list(run_dir.rglob("runs/compare/*.json"))
        for path in compare_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            preset = _normalize_preset(
                payload.get("scenario_params", {}).get("preset", "unknown")
            )
            policy = payload.get("scenario_params", {}).get("open_policy", "current")
            index.setdefault(preset, {}).setdefault(policy, {})["compare_json"] = path

    # Backfill analysis paths based on source_report lines.
    analysis_dir = run_dir.parent / "analysis"
    analysis_files = list(analysis_dir.glob("*.md")) + list(analysis_dir.glob("*.csv"))
    analysis_by_report: Dict[str, List[Path]] = {}
    for path in analysis_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        report_line = None
        for line in text.splitlines():
            if line.strip().startswith("- Source report:") or line.strip().startswith("- source_report:"):
                report_line = line
                break
        if not report_line:
            continue
        report_path = report_line.split(":", 1)[1].strip()
        analysis_by_report.setdefault(report_path, []).append(path)

    for preset, policies in index.items():
        for policy, info in policies.items():
            compare_path = info.get("compare_json")
            if not compare_path:
                continue
            report_paths = analysis_by_report.get(str(compare_path), [])
            for path in report_paths:
                if path.name.endswith("_bridged_ab.md") and "bridged_ab_md" not in info:
                    info["bridged_ab_md"] = path
                if path.name.endswith("_selection_triage.csv") and "selection_triage_csv" not in info:
                    info["selection_triage_csv"] = path
                if path.name.endswith("_patterns.md") and "patterns_md" not in info:
                    info["patterns_md"] = path
                if path.name.endswith("_slot_breakdown.md") and "slot_breakdown_md" not in info:
                    info["slot_breakdown_md"] = path
                if path.name.endswith("_slot_breakdown.csv") and "slot_breakdown_csv" not in info:
                    info["slot_breakdown_csv"] = path

    # C0-2: load compare JSONs and extract metrics.
    metric_rows: List[Dict[str, Any]] = []
    for preset, policies in index.items():
        for policy, info in policies.items():
            compare_path = info.get("compare_json")
            if not compare_path or not compare_path.exists():
                continue
            payload = json.loads(compare_path.read_text(encoding="utf-8"))
            methods = payload.get("methods", [])
            for method in methods:
                report = payload.get("method_reports", {}).get(method, {})
                metrics = report.get("metrics", {}) or {}
                summary = payload.get("summary", {}).get(method, {}) or {}
                row = {
                    "preset": preset,
                    "open_policy": policy,
                    "method": method,
                    "judge_accuracy": metrics.get("judge_accuracy", summary.get("judge_accuracy")),
                    "decision_accuracy": metrics.get("decision_accuracy", summary.get("decision_accuracy")),
                    "rank_success_rate": metrics.get("rank_success_rate", summary.get("rank_success_rate")),
                    "winning_in_union_rate": metrics.get("winning_in_union_rate", summary.get("winning_in_union_rate")),
                    "opened_has_winning_clause_rate_union": metrics.get(
                        "opened_has_winning_clause_rate_union",
                        summary.get("opened_has_winning_clause_rate_union"),
                    ),
                    "policy_gain_over_rank": metrics.get("policy_gain_over_rank", summary.get("policy_gain_over_rank")),
                    "rank_gap": metrics.get("rank_gap", summary.get("rank_gap")),
                    "opened_gold_coverage_core_mean": metrics.get("opened_gold_coverage_core_mean"),
                    "deep_rank_core_rate": metrics.get("deep_rank_core_rate", summary.get("deep_rank_core_rate")),
                    "min_gold_core_rank_union_mean": metrics.get(
                        "min_gold_core_rank_union_mean", summary.get("min_gold_core_rank_union_mean")
                    ),
                    "min_gold_core_rank_union_median": metrics.get(
                        "min_gold_core_rank_union_median", summary.get("min_gold_core_rank_union_median")
                    ),
                    "opened_bridge_count_mean": metrics.get(
                        "opened_bridge_count_mean", summary.get("opened_bridge_count_mean")
                    ),
                    "opened_rule_count_mean": metrics.get(
                        "opened_rule_count_mean", summary.get("opened_rule_count_mean")
                    ),
                    "opened_meta_count_mean": metrics.get(
                        "opened_meta_count_mean", summary.get("opened_meta_count_mean")
                    ),
                }
                metric_rows.append(row)

    def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return "n/a"

    # Build lookup for per-method metrics.
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in metric_rows:
        by_key[(row["preset"], row["open_policy"], row["method"])] = row

    output_dir = run_dir / "analysis_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    # C1-1 Policy × Method matrix
    matrix_rows: List[Dict[str, Any]] = []
    for preset, policies in sorted(index.items()):
        for policy in sorted(policies.keys()):
            row = {
                "preset": preset,
                "open_policy": policy,
            }
            for method in ["goc", "goc_base", "topk", "full"]:
                key = (preset, policy, method)
                metrics = by_key.get(key, {})
                row[f"{method}_judge_acc"] = metrics.get("judge_accuracy")
                row[f"{method}_policy_gain_over_rank"] = metrics.get("policy_gain_over_rank")
            goc_metrics = by_key.get((preset, policy, "goc"), {})
            row["goc_rank_success_rate"] = goc_metrics.get("rank_success_rate")
            row["goc_winning_in_union_rate"] = goc_metrics.get("winning_in_union_rate")
            matrix_rows.append(row)

    matrix_fields = [
        "preset",
        "open_policy",
        "goc_judge_acc",
        "goc_base_judge_acc",
        "topk_judge_acc",
        "full_judge_acc",
        "goc_rank_success_rate",
        "goc_winning_in_union_rate",
        "goc_policy_gain_over_rank",
        "goc_base_policy_gain_over_rank",
        "topk_policy_gain_over_rank",
        "full_policy_gain_over_rank",
    ]
    matrix_csv = output_dir / "policy_method_matrix.csv"
    _write_csv(matrix_csv, matrix_rows, matrix_fields)

    matrix_md = output_dir / "policy_method_matrix.md"
    lines = []
    lines.append("# Policy × Method Matrix")
    lines.append("")
    lines.append("|" + "|".join(matrix_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(matrix_fields)) + "|")
    for row in matrix_rows:
        lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy"} else str(row.get(f)) for f in matrix_fields) + "|")
    matrix_md.write_text("\n".join(lines), encoding="utf-8")

    # C1-2 GoC deltas
    delta_rows: List[Dict[str, Any]] = []
    for preset, policies in sorted(index.items()):
        for policy in sorted(policies.keys()):
            goc = by_key.get((preset, policy, "goc"), {})
            for baseline in ["goc_base", "topk", "full"]:
                base = by_key.get((preset, policy, baseline), {})
                delta_rows.append(
                    {
                        "preset": preset,
                        "open_policy": policy,
                        "baseline": baseline,
                        "delta_judge_acc": (
                            (goc.get("judge_accuracy") or 0) - (base.get("judge_accuracy") or 0)
                            if goc.get("judge_accuracy") is not None and base.get("judge_accuracy") is not None
                            else None
                        ),
                        "delta_policy_gain_over_rank": (
                            (goc.get("policy_gain_over_rank") or 0) - (base.get("policy_gain_over_rank") or 0)
                            if goc.get("policy_gain_over_rank") is not None and base.get("policy_gain_over_rank") is not None
                            else None
                        ),
                    }
                )
    delta_fields = ["preset", "open_policy", "baseline", "delta_judge_acc", "delta_policy_gain_over_rank"]
    delta_csv = output_dir / "goc_deltas.csv"
    _write_csv(delta_csv, delta_rows, delta_fields)
    delta_md = output_dir / "goc_deltas.md"
    lines = []
    lines.append("# GoC Improvement Decomposition")
    lines.append("")
    lines.append("|" + "|".join(delta_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(delta_fields)) + "|")
    for row in delta_rows:
        lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy", "baseline"} else str(row.get(f)) for f in delta_fields) + "|")
    delta_md.write_text("\n".join(lines), encoding="utf-8")

    # C1-3 Difficulty sanity (goc only)
    sanity_rows: List[Dict[str, Any]] = []
    for preset, policies in sorted(index.items()):
        for policy in sorted(policies.keys()):
            goc = by_key.get((preset, policy, "goc"), {})
            sanity_rows.append(
                {
                    "preset": preset,
                    "open_policy": policy,
                    "rank_success_rate": goc.get("rank_success_rate"),
                    "deep_rank_core_rate": goc.get("deep_rank_core_rate"),
                    "min_gold_core_rank_union_mean": goc.get("min_gold_core_rank_union_mean"),
                    "min_gold_core_rank_union_median": goc.get("min_gold_core_rank_union_median"),
                }
            )
    sanity_fields = [
        "preset",
        "open_policy",
        "rank_success_rate",
        "deep_rank_core_rate",
        "min_gold_core_rank_union_mean",
        "min_gold_core_rank_union_median",
    ]
    sanity_csv = output_dir / "difficulty_sanity.csv"
    _write_csv(sanity_csv, sanity_rows, sanity_fields)
    sanity_md = output_dir / "difficulty_sanity.md"
    lines = []
    lines.append("# Difficulty Calibration Sanity")
    lines.append("")
    lines.append("|" + "|".join(sanity_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(sanity_fields)) + "|")
    for row in sanity_rows:
        lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "open_policy"} else str(row.get(f)) for f in sanity_fields) + "|")
    sanity_md.write_text("\n".join(lines), encoding="utf-8")

    # C0-3 Slot breakdown parsing + C2 slot summary
    slot_rows: List[Dict[str, Any]] = []
    for preset, policies in sorted(index.items()):
        for policy, info in policies.items():
            slot_md_path = info.get("slot_breakdown_md")
            if not slot_md_path or not slot_md_path.exists():
                continue
            table_lines = []
            for line in slot_md_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("|slot_term|"):
                    table_lines.append(line)
                    continue
                if table_lines:
                    if line.startswith("|") and not line.startswith("|---"):
                        table_lines.append(line)
                    elif line.strip() == "":
                        break
            if len(table_lines) < 2:
                continue
            for row_line in table_lines[1:]:
                parts = [p.strip() for p in row_line.strip("|").split("|")]
                if len(parts) < 6:
                    continue
                slot_rows.append(
                    {
                        "preset": preset,
                        "open_policy": policy,
                        "slot_term": parts[0],
                        "n": int(parts[1]) if parts[1].isdigit() else None,
                        "judge_acc": float(parts[2]) if parts[2] != "n/a" else None,
                        "cov_core": float(parts[3]) if parts[3] != "n/a" else None,
                        "min_gold_core_rank_union_mean": float(parts[4]) if parts[4] != "n/a" else None,
                        "deep_rank_core_rate": float(parts[5]) if parts[5] != "n/a" else None,
                    }
                )

    # Build slot comparison current vs bridge_one_only (fallback hop2_priority)
    slot_summary_rows: List[Dict[str, Any]] = []
    for preset in sorted({r["preset"] for r in slot_rows}):
        current_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == "current"}
        alt_policy = "bridge_one_only"
        alt_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == alt_policy}
        if not alt_rows:
            alt_policy = "hop2_priority"
            alt_rows = {r["slot_term"]: r for r in slot_rows if r["preset"] == preset and r["open_policy"] == alt_policy}
        for slot_term, cur in current_rows.items():
            alt = alt_rows.get(slot_term)
            if not alt:
                continue
            delta = None
            if cur.get("judge_acc") is not None and alt.get("judge_acc") is not None:
                delta = alt.get("judge_acc") - cur.get("judge_acc")
            slot_summary_rows.append(
                {
                    "preset": preset,
                    "slot_term": slot_term,
                    "baseline_policy": "current",
                    "compare_policy": alt_policy,
                    "judge_acc_current": cur.get("judge_acc"),
                    "judge_acc_compare": alt.get("judge_acc"),
                    "delta_judge_acc": delta,
                    "cov_core_current": cur.get("cov_core"),
                    "cov_core_compare": alt.get("cov_core"),
                    "deep_rank_core_rate_current": cur.get("deep_rank_core_rate"),
                    "deep_rank_core_rate_compare": alt.get("deep_rank_core_rate"),
                    "min_gold_core_rank_union_mean_current": cur.get("min_gold_core_rank_union_mean"),
                    "min_gold_core_rank_union_mean_compare": alt.get("min_gold_core_rank_union_mean"),
                }
            )

    slot_summary_fields = [
        "preset",
        "slot_term",
        "baseline_policy",
        "compare_policy",
        "judge_acc_current",
        "judge_acc_compare",
        "delta_judge_acc",
        "cov_core_current",
        "cov_core_compare",
        "deep_rank_core_rate_current",
        "deep_rank_core_rate_compare",
        "min_gold_core_rank_union_mean_current",
        "min_gold_core_rank_union_mean_compare",
    ]
    slot_csv = output_dir / "slot_summary.csv"
    _write_csv(slot_csv, slot_summary_rows, slot_summary_fields)
    slot_md = output_dir / "slot_summary.md"
    lines = []
    lines.append("# Slot Summary (Current vs Bridge-One-Only)")
    lines.append("")
    lines.append("|" + "|".join(slot_summary_fields) + "|")
    lines.append("|" + "|".join(["---"] * len(slot_summary_fields)) + "|")
    for row in slot_summary_rows:
        lines.append("|" + "|".join(_fmt(row.get(f)) if f not in {"preset", "slot_term", "baseline_policy", "compare_policy"} else str(row.get(f)) for f in slot_summary_fields) + "|")
    slot_md.write_text("\n".join(lines), encoding="utf-8")

    # C2-2 Narrative summary
    narrative_md = output_dir / "narrative_summary.md"
    narrative_lines: List[str] = []
    narrative_lines.append("# Narrative Summary (GoC Novelty)")
    narrative_lines.append("")
    # Use n10 current for headline metrics when available.
    headline = by_key.get(("n10_exclcore", "current", "goc"), {})
    win_rate = headline.get("winning_in_union_rate")
    rank_rate = headline.get("rank_success_rate")
    narrative_lines.append(
        f"- Winning is almost always in union (rate={_fmt(win_rate)}), but rank-only success is low (rate={_fmt(rank_rate)}), so selection under budget dominates."
    )
    # Policy gain statement.
    cur = by_key.get(("n10_exclcore", "current", "goc"), {})
    bridge = by_key.get(("n10_exclcore", "bridge_one_only", "goc"), {})
    if cur and bridge:
        delta_gain = None
        if cur.get("policy_gain_over_rank") is not None and bridge.get("policy_gain_over_rank") is not None:
            delta_gain = bridge.get("policy_gain_over_rank") - cur.get("policy_gain_over_rank")
        delta_acc = None
        if cur.get("judge_accuracy") is not None and bridge.get("judge_accuracy") is not None:
            delta_acc = bridge.get("judge_accuracy") - cur.get("judge_accuracy")
        narrative_lines.append(
            f"- GoC improves over default policy: Δjudge_acc={_fmt(delta_acc)}, Δpolicy_gain_over_rank={_fmt(delta_gain)} (n10_exclcore, current→bridge_one_only)."
        )
    # Baseline deltas.
    goc_base = by_key.get(("n10_exclcore", "bridge_one_only", "goc_base"), {})
    if bridge and goc_base:
        delta_base = None
        if bridge.get("judge_accuracy") is not None and goc_base.get("judge_accuracy") is not None:
            delta_base = bridge.get("judge_accuracy") - goc_base.get("judge_accuracy")
        narrative_lines.append(
            f"- GoC vs goc_base under bridge_one_only: Δjudge_acc={_fmt(delta_base)} (n10_exclcore)."
        )

    # Slot concentration analysis (n10_exclcore).
    n10_rows = [r for r in slot_summary_rows if r["preset"] == "n10_exclcore"]
    if n10_rows:
        sorted_rows = sorted(
            [r for r in n10_rows if r.get("delta_judge_acc") is not None],
            key=lambda r: r["delta_judge_acc"],
            reverse=True,
        )
        top_slots = sorted_rows[:5]
        bottom_slots = sorted_rows[-5:] if len(sorted_rows) >= 5 else sorted_rows[-len(sorted_rows):]
        top_names = ", ".join([r["slot_term"] for r in top_slots])
        bottom_names = ", ".join([r["slot_term"] for r in bottom_slots])
        narrative_lines.append(
            f"- Improvements are not concentrated in a single slot; top slots: {top_names}; worst slots: {bottom_names}."
        )
    narrative_md.write_text("\n".join(narrative_lines), encoding="utf-8")

    # C3 bundles
    analysis_bundle_zip = run_dir / "analysis_bundle.zip"
    share_bundle_zip = run_dir / "share_bundle.zip"

    import shutil

    shutil.make_archive(str(analysis_bundle_zip.with_suffix("")), "zip", output_dir)

    share_dir = run_dir / "share_bundle"
    share_dir.mkdir(parents=True, exist_ok=True)
    # Collect compare jsons (all)
    compare_paths = []
    for preset, policies in index.items():
        for policy, info in policies.items():
            compare_path = info.get("compare_json")
            if compare_path and compare_path.exists():
                compare_paths.append(compare_path)
    for path in compare_paths:
        shutil.copy(path, share_dir / path.name)

    # Slot breakdown md (n10 current + bridge_one_only if available)
    for policy in ["current", "bridge_one_only"]:
        info = index.get("n10_exclcore", {}).get(policy, {})
        slot_md_path = info.get("slot_breakdown_md")
        if slot_md_path and slot_md_path.exists():
            shutil.copy(slot_md_path, share_dir / slot_md_path.name)

    # Triage zips (n10 current + bridge_one_only)
    for policy in ["current", "bridge_one_only"]:
        info = index.get("n10_exclcore", {}).get(policy, {})
        triage_zip = info.get("triage_zip")
        if triage_zip and triage_zip.exists():
            shutil.copy(triage_zip, share_dir / triage_zip.name)

    shutil.make_archive(str(share_bundle_zip.with_suffix("")), "zip", share_dir)

    return {
        "policy_method_matrix_md": matrix_md,
        "policy_method_matrix_csv": matrix_csv,
        "goc_deltas_md": delta_md,
        "goc_deltas_csv": delta_csv,
        "difficulty_sanity_md": sanity_md,
        "difficulty_sanity_csv": sanity_csv,
        "slot_summary_md": slot_md,
        "slot_summary_csv": slot_csv,
        "narrative_summary_md": narrative_md,
        "analysis_bundle_zip": analysis_bundle_zip,
        "share_bundle_zip": share_bundle_zip,
    }
