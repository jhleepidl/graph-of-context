from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
