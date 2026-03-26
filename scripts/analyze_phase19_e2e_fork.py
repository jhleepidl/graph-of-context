#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def _load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _method_rows(bundle_root: Path) -> Dict[str, List[Dict]]:
    rows: List[Dict] = []
    for p in bundle_root.rglob('phase19_results.jsonl'):
        rows.extend(_load_rows(p))
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(str(r.get('method')), []).append(r)
    return out


def _ccdf(values: List[float]):
    xs = sorted(values)
    n = max(1, len(xs))
    ys = [1.0 - (i / n) for i, _ in enumerate(xs)]
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle_root', type=str, required=True)
    args = ap.parse_args()
    root = Path(args.bundle_root)
    analysis = root / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    by_method = _method_rows(root)
    summary = []
    for method, rs in sorted(by_method.items()):
        tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
        tok_sorted = sorted(tok)
        p95 = tok_sorted[min(len(tok_sorted)-1, int(0.95 * (len(tok_sorted)-1)))] if tok_sorted else 0.0
        summary.append({
            'method': method,
            'n': len(rs),
            'accuracy': sum(1 for r in rs if bool(r.get('correct'))) / max(1, len(rs)),
            'accuracy_strict': sum(1 for r in rs if bool(r.get('correct_strict'))) / max(1, len(rs)),
            'avg_total_tokens': sum(tok) / max(1, len(tok)),
            'p95_total_tokens': p95,
            'avg_docid_cov': sum(float(r.get('docid_cov') or 0.0) for r in rs) / max(1, len(rs)),
            'avg_fork_calls': sum(float((r.get('tool_stats', {}) or {}).get('fork_calls') or 0.0) for r in rs) / max(1, len(rs)),
        })
    with open(analysis / 'phase19_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ['method'])
        w.writeheader(); [w.writerow(r) for r in summary]

    # tradeoff plot
    plt.figure(figsize=(6.2, 4.3))
    for r in summary:
        plt.scatter(float(r['avg_total_tokens']), float(r['accuracy']), label=r['method'])
        plt.annotate(r['method'], (float(r['avg_total_tokens']), float(r['accuracy'])), fontsize=8)
    plt.xlabel('Average total tokens')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(analysis / 'phase19_e2e_tradeoff.png', dpi=180)
    plt.close()

    # heavy-tail CCDF
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
        f.write('| method | n | accuracy | accuracy_strict | avg_total_tokens | p95_total_tokens | avg_docid_cov | avg_fork_calls |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in summary:
            f.write(f"| {r['method']} | {r['n']} | {r['accuracy']:.3f} | {r['accuracy_strict']:.3f} | {r['avg_total_tokens']:.1f} | {r['p95_total_tokens']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_fork_calls']:.2f} |\n")

if __name__ == '__main__':
    main()
