from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

try:
    from goc_logger.export import export_dot
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    _root_src = Path(__file__).resolve().parents[4] / "src"
    if str(_root_src) not in sys.path:
        sys.path.append(str(_root_src))
    from goc_logger.export import export_dot


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run_graph_jsonl(report_path: Path, method: str) -> Path | None:
    base_dir = report_path.parent / report_path.stem
    candidates = [
        base_dir / method / "goc_graph.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _write_case_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_dot(case_dir: Path, task_id: str, note: str) -> Path:
    dot_path = case_dir / "graph.dot"
    dot_path.write_text(
        "digraph G {\n"
        f"  \"episode:{task_id}\" [label=\"episode\\n{note}\"];\n"
        "}\n",
        encoding="utf-8",
    )
    return dot_path


def _extract_task_events(run_jsonl: Path, task_id: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with run_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("task_id") == task_id:
                events.append(data)
    return events


def triage_compare(
    compare_report: Path,
    method: str = "goc",
    out_dir: Path | None = None,
    max_per_bucket: int = 20,
    buckets: List[str] | None = None,
) -> Path:
    report = _load_report(compare_report)
    out_dir = out_dir or compare_report.parent / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_report = report.get("method_reports", {}).get(method, {})
    records = method_report.get("records", [])
    run_graph = _find_run_graph_jsonl(compare_report, method)

    bucket_defs = {
        "C_retrieval_fail": (
            lambda r: r.get("gold_in_search_topk") is False,
            "gold_in_search_topk=False",
        ),
        "E_budget_edge_fail": (
            lambda r: r.get("gold_in_search_topk") is True
            and isinstance(r.get("winning_clause_rank"), int)
            and isinstance(r.get("open_budget"), int)
            and r.get("winning_clause_rank") > r.get("open_budget"),
            "gold_in_search_topk=True and winning_clause_rank>open_budget",
        ),
        "A_open_selection_fail": (
            lambda r: r.get("gold_in_search_topk") is True
            and isinstance(r.get("winning_clause_rank"), int)
            and isinstance(r.get("open_budget"), int)
            and r.get("winning_clause_rank") <= r.get("open_budget")
            and (r.get("opened_gold_coverage") == 0 or r.get("opened_gold_coverage") == 0.0),
            "gold_in_search_topk=True, winning_clause_rank<=open_budget, opened_gold_coverage==0",
        ),
        "B_reasoning_fail": (
            lambda r: r.get("opened_has_winning_clause") is True
            and r.get("decision_correct") is False,
            "opened_has_winning_clause=True and decision_correct=False",
        ),
        "D_decision_confusion": (
            lambda r: r.get("gold_decision") == "require_condition"
            and r.get("pred_decision") in {"allow", "deny"},
            "gold_decision=require_condition and pred_decision in {allow,deny}",
        ),
        "G_bridge_fail": (
            lambda r: r.get("scenario_mode") == "bridged_v1_1"
            and r.get("bridge_needed") is True
            and r.get("decision_correct") is False
            and r.get("bridge_opened") is False,
            "bridged_v1_1, bridge_needed=True, decision_correct=False, bridge_opened=False",
        ),
        "H_canonical_query_fail": (
            lambda r: r.get("scenario_mode") == "bridged_v1_1"
            and r.get("bridge_needed") is True
            and r.get("decision_correct") is False
            and r.get("hop2_executed") is True
            and r.get("hop2_query_contains_canonical") is False,
            "bridged_v1_1, bridge_needed=True, decision_correct=False, hop2_executed=True, canonical_used_in_query2=False",
        ),
        "SEL_GAP": (
            lambda r: isinstance(r.get("selection_gap"), (int, float))
            and r.get("selection_gap") >= 0.5,
            "selection_gap>=0.5",
        ),
        "ACC_NO_CORE_EVIDENCE": (
            lambda r: r.get("acc_no_core_evidence") is True,
            "judge_correct=True and opened_gold_coverage_core==0",
        ),
        "F_evidence_padding_artifact": (
            lambda r: (
                not r.get("evidence_before_pad")
                or len(r.get("evidence_before_pad") or []) == 0
                or len(
                    set(r.get("evidence_before_pad") or [])
                    & set(r.get("gold_evidence_core_ids") or r.get("gold_evidence_ids") or [])
                )
                == 0
            )
            and len(r.get("evidence_after_pad") or []) > 0,
            "evidence_before_pad empty or no gold hit, evidence_after_pad non-empty",
        ),
        "THREAD_E1_FAIL": (
            lambda r: r.get("episode_id") == 1 and r.get("commit_correct") is False,
            "threaded: commit1 missing core evidence",
        ),
        "THREAD_E2_FAIL": (
            lambda r: r.get("episode_id") == 2 and r.get("commit_correct") is False,
            "threaded: commit2 missing core evidence",
        ),
        "COMPOSE_FAIL": (
            lambda r: r.get("episode_id") == 3 and r.get("thread_judge_correct") is False,
            "threaded: final compose incorrect",
        ),
        "BRANCH_TRAP": (
            lambda r: r.get("episode_id") == 2 and r.get("branch_trap") is True,
            "threaded: distractor exception selected",
        ),
        "FOLD_DRIFT": (
            lambda r: r.get("episode_id") == 3 and r.get("fold_drift") is True,
            "threaded: fold/summary drifted from commit facts",
        ),
        "COST_BLOWUP": (
            lambda r: r.get("cost_blowup") is True,
            "threaded: prompt token budget exceeded threshold",
        ),
    }
    if buckets:
        bucket_defs = {k: v for k, v in bucket_defs.items() if k in buckets}

    for bucket, (predicate, reason_text) in bucket_defs.items():
        selected = [r for r in records if predicate(r)][:max_per_bucket]
        if not selected:
            continue
        bucket_dir = out_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for rec in selected:
            artifact_copy_errors: List[str] = []
            task_id = rec.get("task_id")
            if not task_id:
                continue
            case_dir = bucket_dir / task_id
            case_dir.mkdir(parents=True, exist_ok=True)
            open_budget = rec.get("open_budget")
            winning_clause_rank = rec.get("winning_clause_rank")
            min_gold_rank = rec.get("min_gold_rank")
            budget_gap = None
            if isinstance(winning_clause_rank, int) and isinstance(open_budget, int):
                budget_gap = winning_clause_rank - open_budget
            case = {
                "task_id": task_id,
                "thread_id": rec.get("thread_id"),
                "episode_id": rec.get("episode_id"),
                "episode_kind": rec.get("episode_kind"),
                "pred_decision": rec.get("pred_decision"),
                "gold_decision": rec.get("gold_decision"),
                "opened_clause_ids": rec.get("opened_clause_ids"),
                "forced_open_ids": rec.get("forced_open_ids"),
                "opened_probe_clause_ids": rec.get("opened_probe_clause_ids"),
                "opened_for_prompt_clause_ids": rec.get("opened_for_prompt_clause_ids"),
                "opened_total_clause_ids": rec.get("opened_total_clause_ids"),
                "opened_gold_coverage": rec.get("opened_gold_coverage"),
                "gold_in_search_topk": rec.get("gold_in_search_topk"),
                "winning_clause_rank": winning_clause_rank,
                "min_gold_rank": min_gold_rank,
                "open_budget": open_budget,
                "budget_used": rec.get("open_calls"),
                "budget_gap": budget_gap,
                "gold_score_gap": rec.get("gold_score_gap"),
                "bridge_clause_id": rec.get("bridge_clause_id"),
                "hop1_query": rec.get("hop1_query_text"),
                "hop2_query": rec.get("hop2_query"),
                "meta_evidence_present": rec.get("meta_evidence_present"),
                "core_evidence_size": rec.get("core_evidence_size"),
            }

            graph_jsonl = rec.get("goc_graph_jsonl_path")
            if graph_jsonl:
                graph_path = Path(graph_jsonl)
            elif run_graph:
                graph_path = run_graph
            else:
                graph_path = None

            task_graph_path = case_dir / "graph.jsonl"
            if graph_path and graph_path.exists():
                try:
                    if graph_path.name == f"{task_id}.jsonl":
                        shutil.copy(graph_path, task_graph_path)
                    else:
                        events = _extract_task_events(graph_path, task_id)
                        if events:
                            with task_graph_path.open("w", encoding="utf-8") as handle:
                                for event in events:
                                    handle.write(json.dumps(event, ensure_ascii=False))
                                    handle.write("\n")
                        else:
                            case["graph_missing_due_to_sampling"] = True
                except Exception as exc:
                    artifact_copy_errors.append(f"graph_jsonl_copy_failed:{exc}")
            else:
                artifact_copy_errors.append("graph_jsonl_missing")
                if rec.get("goc_graph_jsonl_path") is None and rec.get("goc_graph_dot_path") is None:
                    case["graph_missing_due_to_sampling"] = True

            graph_dot_path = rec.get("goc_graph_dot_path")
            if graph_dot_path and Path(graph_dot_path).exists():
                try:
                    shutil.copy(graph_dot_path, case_dir / "graph.dot")
                except Exception as exc:
                    artifact_copy_errors.append(f"graph_dot_copy_failed:{exc}")

            dot_path = case_dir / "graph.dot"
            if not dot_path.exists():
                if task_graph_path.exists():
                    export_dot(task_graph_path, task_id, dot_path)
                else:
                    note = "sampled_out" if case.get("graph_missing_due_to_sampling") else "regenerated-minimal"
                    _minimal_dot(case_dir, task_id, note)

            dot_text = dot_path.read_text(encoding="utf-8")
            if "None/None" in dot_text or "unknown/unknown" in dot_text:
                if task_graph_path.exists():
                    export_dot(task_graph_path, task_id, dot_path)
                    dot_text = dot_path.read_text(encoding="utf-8")
                elif case.get("graph_missing_due_to_sampling"):
                    _minimal_dot(case_dir, task_id, "sampled_out")
                    dot_text = dot_path.read_text(encoding="utf-8")
                if "None/None" in dot_text or "unknown/unknown" in dot_text:
                    case["artifact_quality_warning"] = "dot_missing_doc_meta"
                    print(f"[triage] warning: dot_missing_doc_meta for {task_id}")

            prompt_path = rec.get("prompt_path")
            if prompt_path and Path(prompt_path).exists():
                try:
                    shutil.copy(prompt_path, case_dir / "prompt.txt")
                except Exception as exc:
                    artifact_copy_errors.append(f"prompt_copy_failed:{exc}")
            raw_path = rec.get("raw_path")
            if raw_path and Path(raw_path).exists():
                try:
                    shutil.copy(raw_path, case_dir / "raw.txt")
                except Exception as exc:
                    artifact_copy_errors.append(f"raw_copy_failed:{exc}")
            if not (case_dir / "prompt.txt").exists() or not (case_dir / "raw.txt").exists():
                case["missing_prompt_raw"] = True

            if artifact_copy_errors:
                case["artifact_copy_errors"] = artifact_copy_errors
            case["bucket_reason"] = reason_text
            _write_case_json(case_dir / "case.json", case)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyOps triage exporter")
    parser.add_argument("--compare_report", required=True)
    parser.add_argument("--method", default="goc")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--max_per_bucket", type=int, default=20)
    parser.add_argument("--buckets", default="")
    args = parser.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()] if args.buckets else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    out_path = triage_compare(
        Path(args.compare_report),
        method=args.method,
        out_dir=out_dir,
        max_per_bucket=args.max_per_bucket,
        buckets=buckets,
    )
    print(f"Triage exported to {out_path}")


if __name__ == "__main__":
    main()
