#!/usr/bin/env python3
"""Summarize experiment runs from experiment_goc_runs.jsonl."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional


def load_runs(path: str) -> List[Dict[str, Any]]:
    runs = []
    if not os.path.exists(path):
        return runs
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs


def format_ts(ts: Optional[float]) -> str:
    if not ts:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def infer_needle_location(dataset_path: str) -> str:
    if not dataset_path or not os.path.exists(dataset_path):
        return "unknown"
    try:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                return record.get("needle_location", "main")
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return "unknown"


def get_metric(agent_metric: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = agent_metric.get("accuracy")
    avg_prompt_tokens = agent_metric.get("avg_prompt_tokens")
    if avg_prompt_tokens is None:
        avg_prompt_tokens = agent_metric.get("avg_tokens")
    return {"accuracy": accuracy, "avg_prompt_tokens": avg_prompt_tokens}


def summarize_runs(runs: List[Dict[str, Any]], limit: int, dataset_filter: str) -> None:
    if dataset_filter:
        runs = [r for r in runs if dataset_filter in str(r.get("dataset", ""))]
    if limit > 0:
        runs = runs[-limit:]

    if not runs:
        print("No runs found.")
        return

    for run in runs:
        dataset = run.get("dataset", "unknown")
        needle_location = infer_needle_location(dataset)
        print("Run:", run.get("run_id", "unknown"))
        print("  timestamp:", format_ts(run.get("timestamp")))
        print("  dataset:", dataset)
        print("  needle_location:", needle_location)
        print("  num_cases:", run.get("num_cases"))
        print("  model:", run.get("model"))
        print("  temperature:", run.get("temperature"))
        print("  fold_every:", run.get("fold_every"))
        print("  goc_bundle_size:", run.get("goc_bundle_size"))
        print("  metrics:")
        metrics = run.get("metrics", {})
        for agent_name, agent_metric in metrics.items():
            metric = get_metric(agent_metric)
            print(
                f"    - {agent_name}: accuracy={metric['accuracy']}, "
                f"avg_prompt_tokens={metric['avg_prompt_tokens']}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GoC run logs")
    parser.add_argument("--runs", default="experiment_goc_runs.jsonl")
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--dataset-filter", default="")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    summarize_runs(runs, args.last, args.dataset_filter)


if __name__ == "__main__":
    main()
