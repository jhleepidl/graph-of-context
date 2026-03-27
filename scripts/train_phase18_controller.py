#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ACTIONS = ["none", "unfold", "fork", "unfold_then_fork"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTIONS)}


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        rows.append(row)
    return rows


def _collect_feature_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    for row in rows:
        feats = row.get("features") or {}
        if isinstance(feats, dict):
            for key, val in feats.items():
                if isinstance(val, (int, float, bool)):
                    names.add(str(key))
    return sorted(names)


def _row_to_xy(row: Dict[str, Any], feature_names: Sequence[str]) -> Tuple[List[float], int]:
    feats = row.get("features") or {}
    x = []
    for name in feature_names:
        val = feats.get(name, 0.0)
        if isinstance(val, bool):
            x.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            x.append(float(val))
        else:
            x.append(0.0)
    y = ACTION_TO_ID[str(row.get("best_action") or "none")]
    return x, y


def _mean_utility(rows: Sequence[Dict[str, Any]], predicted_actions: Sequence[str]) -> float:
    vals: List[float] = []
    for row, action in zip(rows, predicted_actions):
        act = ((row.get("actions") or {}).get(str(action)) or {})
        try:
            vals.append(float(act.get("utility") or 0.0))
        except Exception:
            vals.append(0.0)
    return float(sum(vals) / max(1, len(vals)))


def _always_action(rows: Sequence[Dict[str, Any]], action: str) -> Dict[str, Any]:
    preds = [str(action)] * len(rows)
    return {
        "action": str(action),
        "mean_utility": _mean_utility(rows, preds),
        "action_accuracy": accuracy_score([str(r.get("best_action") or "none") for r in rows], preds),
    }


def _report_for_rows(rows: Sequence[Dict[str, Any]], predicted_actions: Sequence[str]) -> Dict[str, Any]:
    gold = [str(r.get("best_action") or "none") for r in rows]
    mean_utility = _mean_utility(rows, predicted_actions)
    oracle_utility = float(sum(float(r.get("best_utility") or 0.0) for r in rows) / max(1, len(rows)))
    regret = float(oracle_utility - mean_utility)
    return {
        "n": int(len(rows)),
        "mean_utility": float(mean_utility),
        "oracle_mean_utility": float(oracle_utility),
        "mean_regret": float(regret),
        "action_accuracy": float(accuracy_score(gold, predicted_actions)) if rows else 0.0,
        "action_counts": {a: int(sum(1 for x in predicted_actions if x == a)) for a in ACTIONS},
        "gold_counts": {a: int(sum(1 for x in gold if x == a)) for a in ACTIONS},
        "baselines": {_a: _always_action(rows, _a) for _a in ACTIONS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a simple phase18 learned controller from pivot-level action-utility data.")
    ap.add_argument("--dataset", type=Path, required=True, help="Pivot-level JSONL from build_phase18_action_utility_dataset.py")
    ap.add_argument("--out_model", type=Path, required=True)
    ap.add_argument("--out_report", type=Path, required=True)
    ap.add_argument("--model_type", choices=["logreg", "tree"], default="logreg")
    ap.add_argument("--tree_max_depth", type=int, default=4)
    ap.add_argument("--min_train_rows", type=int, default=20)
    args = ap.parse_args()

    rows = _load_rows(args.dataset)
    dev_rows = [r for r in rows if str(r.get("split") or "") == "dev"]
    test_rows = [r for r in rows if str(r.get("split") or "") == "test"]
    if len(dev_rows) < int(args.min_train_rows):
        raise RuntimeError(f"Need at least {int(args.min_train_rows)} dev rows; found {len(dev_rows)}")
    if not test_rows:
        raise RuntimeError("Dataset has no test rows; rerun builder with a non-degenerate split.")

    feature_names = _collect_feature_names(dev_rows)
    X_dev, y_dev = zip(*[_row_to_xy(r, feature_names) for r in dev_rows])
    X_test, y_test = zip(*[_row_to_xy(r, feature_names) for r in test_rows])
    X_dev_arr = np.asarray(X_dev, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)
    y_dev_arr = np.asarray(y_dev, dtype=int)
    y_test_arr = np.asarray(y_test, dtype=int)

    if str(args.model_type) == "tree":
        model = DecisionTreeClassifier(max_depth=int(args.tree_max_depth), random_state=7)
    else:
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=7,
        )
    model.fit(X_dev_arr, y_dev_arr)

    pred_dev_ids = model.predict(X_dev_arr)
    pred_test_ids = model.predict(X_test_arr)
    pred_dev = [ACTIONS[int(i)] for i in pred_dev_ids]
    pred_test = [ACTIONS[int(i)] for i in pred_test_ids]

    dev_report = _report_for_rows(dev_rows, pred_dev)
    test_report = _report_for_rows(test_rows, pred_test)
    report = {
        "model_type": str(args.model_type),
        "feature_names": list(feature_names),
        "dev": dev_report,
        "test": test_report,
        "classification_report_test": classification_report(
            [ACTIONS[int(i)] for i in y_test_arr],
            pred_test,
            labels=ACTIONS,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix_test": confusion_matrix(
            [ACTIONS[int(i)] for i in y_test_arr],
            pred_test,
            labels=ACTIONS,
        ).tolist(),
    }

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": list(feature_names),
        "actions": list(ACTIONS),
        "model_type": str(args.model_type),
    }
    with args.out_model.open("wb") as f:
        pickle.dump(payload, f)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model to {args.out_model}")
    print(f"Saved report to {args.out_report}")
    print(f"Test mean utility: {test_report['mean_utility']:.4f}")
    print(f"Test oracle utility: {test_report['oracle_mean_utility']:.4f}")
    print(f"Test mean regret: {test_report['mean_regret']:.4f}")


if __name__ == "__main__":
    main()
