#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _method_rows(bundle_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for p in bundle_root.rglob('phase19_results.jsonl'):
        rows.extend(_load_rows(p))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get('method')), []).append(r)
    return out


def _ccdf(values: List[float]) -> Tuple[List[float], List[float]]:
    xs = sorted(values)
    n = max(1, len(xs))
    ys = [1.0 - (i / n) for i, _ in enumerate(xs)]
    return xs, ys


def _avg(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r.get('method')), []).append(r)
    summary: List[Dict[str, Any]] = []
    for method, rs in sorted(by_method.items()):
        tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
        tok_sorted = sorted(tok)
        p95 = tok_sorted[min(len(tok_sorted)-1, int(0.95 * (len(tok_sorted)-1)))] if tok_sorted else 0.0
        summary.append({
            'method': method,
            'n': len(rs),
            'accuracy': _avg([1.0 if bool(r.get('correct')) else 0.0 for r in rs]),
            'accuracy_strict': _avg([1.0 if bool(r.get('correct_strict')) else 0.0 for r in rs if r.get('correct_strict') is not None]),
            'avg_total_tokens': _avg(tok),
            'p95_total_tokens': p95,
            'avg_docid_cov': _avg([float(r.get('docid_cov') or 0.0) for r in rs]),
            'avg_fork_calls': _avg([float((r.get('tool_stats', {}) or {}).get('fork_calls') or 0.0) for r in rs]),
            'avg_fork_tokens': _avg([float((r.get('tool_stats', {}) or {}).get('fork_tokens') or 0.0) for r in rs]),
        })
    return summary


def _slice_groups(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        method = str(r.get('method'))
        slice_name = str(r.get('task_slice') or (r.get('task_meta', {}) or {}).get('task_slice') or 'unspecified')
        groups.setdefault((method, slice_name), []).append(r)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle_root', type=str, required=True)
    args = ap.parse_args()
    root = Path(args.bundle_root)
    analysis = root / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    by_method = _method_rows(root)
    all_rows: List[Dict[str, Any]] = []
    for rs in by_method.values():
        all_rows.extend(rs)
    summary = _summary_rows(all_rows)

    with open(analysis / 'phase19_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ['method'])
        w.writeheader(); [w.writerow(r) for r in summary]

    plt.figure(figsize=(6.2, 4.3))
    for r in summary:
        plt.scatter(float(r['avg_total_tokens']), float(r['accuracy']), label=r['method'])
        plt.annotate(r['method'], (float(r['avg_total_tokens']), float(r['accuracy'])), fontsize=8)
    plt.xlabel('Average total tokens')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(analysis / 'phase19_e2e_tradeoff.png', dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.3))
    for method, rs in sorted(by_method.items()):
        tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
        if not tok:
            continue
        xs, ys = _ccdf(tok)
        plt.step(xs, ys, where='post', label=method)
    plt.yscale('log')
    plt.xlabel('Total tokens per task')
    plt.ylabel('CCDF P(Tokens ≥ x)')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(analysis / 'phase19_token_ccdf.png', dpi=180)
    plt.close()

    with open(analysis / 'phase19_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 End-to-End Fork Analysis\n\n')
        f.write('| method | n | accuracy | accuracy_strict | avg_total_tokens | p95_total_tokens | avg_docid_cov | avg_fork_calls | avg_fork_tokens |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in summary:
            f.write(f"| {r['method']} | {r['n']} | {r['accuracy']:.3f} | {r['accuracy_strict']:.3f} | {r['avg_total_tokens']:.1f} | {r['p95_total_tokens']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_fork_calls']:.2f} | {r['avg_fork_tokens']:.1f} |\n")

    slice_rows: List[Dict[str, Any]] = []
    for (method, slice_name), rs in sorted(_slice_groups(all_rows).items()):
        tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
        slice_rows.append({
            'method': method,
            'task_slice': slice_name,
            'n': len(rs),
            'accuracy': _avg([1.0 if bool(r.get('correct')) else 0.0 for r in rs]),
            'avg_total_tokens': _avg(tok),
            'avg_docid_cov': _avg([float(r.get('docid_cov') or 0.0) for r in rs]),
            'avg_fork_calls': _avg([float((r.get('tool_stats', {}) or {}).get('fork_calls') or 0.0) for r in rs]),
        })

    with open(analysis / 'phase19_slice_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()) if slice_rows else ['method', 'task_slice'])
        w.writeheader(); [w.writerow(r) for r in slice_rows]

    with open(analysis / 'phase19_slice_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 Slice Summary\n\n')
        if slice_rows:
            f.write('| method | task_slice | n | accuracy | avg_total_tokens | avg_docid_cov | avg_fork_calls |\n')
            f.write('|---|---|---:|---:|---:|---:|---:|\n')
            for r in slice_rows:
                f.write(f"| {r['method']} | {r['task_slice']} | {r['n']} | {r['accuracy']:.3f} | {r['avg_total_tokens']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_fork_calls']:.2f} |\n")


if __name__ == '__main__':
    main()
