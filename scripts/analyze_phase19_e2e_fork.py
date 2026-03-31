#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def _load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _all_rows(bundle_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for p in bundle_root.rglob('phase19_results.jsonl'):
        rows.extend(_load_rows(p))
    return rows


def _group(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key) or 'unknown')].append(r)
    return dict(out)


def _ccdf(values: List[float]) -> Tuple[List[float], List[float]]:
    xs = sorted(values)
    n = max(1, len(xs))
    ys = [1.0 - (i / n) for i, _ in enumerate(xs)]
    return xs, ys


def _avg(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _method_summary(method: str, rs: List[Dict]) -> Dict:
    tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
    tok_sorted = sorted(tok)
    p95 = tok_sorted[min(len(tok_sorted)-1, int(0.95 * (len(tok_sorted)-1)))] if tok_sorted else 0.0
    seeds = {str(r.get('run_tag') or '').split('_')[0] for r in rs if r.get('run_tag')}
    return {
        'method': method,
        'n_seeds': len([s for s in seeds if s]),
        'n': len(rs),
        'accuracy': _avg([1.0 if bool(r.get('correct')) else 0.0 for r in rs]),
        'accuracy_strict': _avg([1.0 if bool(r.get('correct_strict')) else 0.0 for r in rs]),
        'avg_total_tokens': _avg(tok),
        'p95_total_tokens': p95,
        'avg_steps': _avg([float(r.get('steps') or 0.0) for r in rs]),
        'avg_docid_cov': _avg([float(r.get('docid_cov') or 0.0) for r in rs]),
        'avg_proof_docid_cov': _avg([float(r.get('proof_docid_cov') or 0.0) for r in rs]),
        'proof_complete_rate': _avg([1.0 if bool(r.get('proof_complete')) else 0.0 for r in rs if 'proof_complete' in r]),
        'proof_complete_correct_rate': _avg([1.0 if bool(r.get('proof_complete_correct')) else 0.0 for r in rs if 'proof_complete_correct' in r]),
        'avg_fork_calls': _avg([float((r.get('tool_stats', {}) or {}).get('fork_calls') or 0.0) for r in rs]),
        'avg_fork_tokens': _avg([float((r.get('tool_stats', {}) or {}).get('fork_tokens') or 0.0) for r in rs]),
        'avg_context_controller_calls': _avg([float((r.get('tool_stats', {}) or {}).get('context_controller_calls') or 0.0) for r in rs]),
        'avg_context_controller_none': _avg([float((r.get('tool_stats', {}) or {}).get('context_controller_none') or 0.0) for r in rs]),
        'avg_context_controller_unfold': _avg([float((r.get('tool_stats', {}) or {}).get('context_controller_unfold') or 0.0) for r in rs]),
        'avg_context_controller_fork': _avg([float((r.get('tool_stats', {}) or {}).get('context_controller_fork') or 0.0) for r in rs]),
        'avg_context_controller_unfold_then_fork': _avg([float((r.get('tool_stats', {}) or {}).get('context_controller_unfold_then_fork') or 0.0) for r in rs]),
        'avg_premature_finish_blocked': _avg([float((r.get('tool_stats', {}) or {}).get('premature_finish_blocked') or 0.0) for r in rs]),
        'no_finish_rate': _avg([1.0 if str(r.get('explanation') or '').startswith('max_steps reached') else 0.0 for r in rs]),
    }


def _recommendation(summary_rows: List[Dict], slice_rows: List[Dict]) -> List[str]:
    if not summary_rows:
        return ['No results found.']
    best = max(summary_rows, key=lambda r: (float(r['accuracy']), -float(r['avg_total_tokens'])))
    lines = []
    lines.append(f"Strongest practical phase19 method in this bundle: {best['method']} (acc={best['accuracy']:.3f}, avg_tokens={best['avg_total_tokens']:.1f}).")
    if str(best['method']) == 'SimilarityOnly':
        lines.append('Interpret phase19 as an exploratory pilot. Similarity retrieval is currently the strongest end-to-end baseline in this bundle.')
    if not any(float(r.get('avg_fork_calls') or 0.0) > 0.0 for r in summary_rows):
        lines.append('Fork did not materially activate in this bundle, so phase19 should not be used as the main fork claim.')
    depish = [r for r in slice_rows if str(r.get('task_slice')) in {'dependency_necessary', 'support_recovery'}]
    if depish:
        best_dep = max(depish, key=lambda r: (float(r['accuracy']), -float(r['avg_total_tokens'])))
        lines.append(f"Best support-recovery-like slice result: {best_dep['task_slice']} / {best_dep['method']} (acc={best_dep['accuracy']:.3f}).")
    proof_rows = [r for r in summary_rows if float(r.get('proof_complete_correct_rate') or 0.0) > 0.0]
    if proof_rows:
        best_proof = max(proof_rows, key=lambda r: (float(r['proof_complete_correct_rate']), float(r['accuracy']), -float(r['avg_total_tokens'])))
        lines.append(f"Best proof-complete method: {best_proof['method']} (proof_complete_correct={best_proof['proof_complete_correct_rate']:.3f}, acc={best_proof['accuracy']:.3f}).")
    lines.append('Recommended paper framing: phase18 as the main evidence; phase19 as a low-claim pilot with SimilarityOnly retained as the strongest practical baseline.')
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle_root', type=str, required=True)
    args = ap.parse_args()
    root = Path(args.bundle_root)
    analysis = root / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    rows = _all_rows(root)
    by_method = _group(rows, 'method')
    summary = [_method_summary(method, rs) for method, rs in sorted(by_method.items())]

    slice_summary: List[Dict] = []
    for slice_name, rs in sorted(_group(rows, 'task_slice').items()):
        for method, ms in sorted(_group(rs, 'method').items()):
            rec = _method_summary(method, ms)
            rec['task_slice'] = slice_name
            slice_summary.append(rec)

    with open(analysis / 'phase19_e2e_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['method','n_seeds','n','accuracy','accuracy_strict','avg_total_tokens','p95_total_tokens','avg_steps','avg_docid_cov','avg_proof_docid_cov','proof_complete_rate','proof_complete_correct_rate','avg_fork_calls','avg_fork_tokens','avg_context_controller_calls','avg_context_controller_none','avg_context_controller_unfold','avg_context_controller_fork','avg_context_controller_unfold_then_fork','avg_premature_finish_blocked','no_finish_rate'])
        w.writeheader(); [w.writerow(r) for r in summary]
    with open(analysis / 'phase19_slice_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['task_slice','method','n_seeds','n','accuracy','accuracy_strict','avg_total_tokens','p95_total_tokens','avg_steps','avg_docid_cov','avg_proof_docid_cov','proof_complete_rate','proof_complete_correct_rate','avg_fork_calls','avg_fork_tokens','avg_context_controller_calls','avg_context_controller_none','avg_context_controller_unfold','avg_context_controller_fork','avg_context_controller_unfold_then_fork','avg_premature_finish_blocked','no_finish_rate'])
        w.writeheader(); [w.writerow(r) for r in slice_summary]

    plt.figure(figsize=(6.2, 4.3))
    for r in summary:
        plt.scatter(float(r['avg_total_tokens']), float(r['accuracy']), label=r['method'])
        plt.annotate(r['method'], (float(r['avg_total_tokens']), float(r['accuracy'])), fontsize=8)
    plt.xlabel('Average total tokens')
    plt.ylabel('Accuracy')
    plt.tight_layout(); plt.savefig(analysis / 'phase19_e2e_tradeoff.png', dpi=180); plt.close()

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
    plt.tight_layout(); plt.savefig(analysis / 'phase19_token_ccdf.png', dpi=180); plt.close()

    with open(analysis / 'phase19_e2e_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 End-to-End Fork Summary\n\n')
        f.write('| method | n_seeds | acc | acc_strict | avg_tokens | p95_tokens | avg_steps | docid_cov | proof_docid_cov | proof_complete | proof_complete_correct | avg_fork_calls | avg_fork_tokens | avg_ctx_calls | avg_ctx_none | avg_ctx_unfold | avg_ctx_fork | avg_ctx_unfold_then_fork | avg_finish_blocked | no_finish_rate |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in summary:
            f.write(f"| {r['method']} | {r['n_seeds']} | {r['accuracy']:.3f} | {r['accuracy_strict']:.3f} | {r['avg_total_tokens']:.1f} | {r['p95_total_tokens']:.1f} | {r['avg_steps']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_proof_docid_cov']:.3f} | {r['proof_complete_rate']:.3f} | {r['proof_complete_correct_rate']:.3f} | {r['avg_fork_calls']:.2f} | {r['avg_fork_tokens']:.1f} | {r['avg_context_controller_calls']:.2f} | {r['avg_context_controller_none']:.2f} | {r['avg_context_controller_unfold']:.2f} | {r['avg_context_controller_fork']:.2f} | {r['avg_context_controller_unfold_then_fork']:.2f} | {r['avg_premature_finish_blocked']:.2f} | {r['no_finish_rate']:.3f} |\n")

    with open(analysis / 'phase19_slice_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 Slice Summary\n\n')
        f.write('| task_slice | method | n | acc | avg_tokens | avg_steps | docid_cov | proof_docid_cov | proof_complete_correct | avg_fork_calls | avg_ctx_calls | avg_ctx_unfold | avg_ctx_fork | avg_ctx_unfold_then_fork | avg_finish_blocked | no_finish_rate |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in slice_summary:
            f.write(f"| {r['task_slice']} | {r['method']} | {r['n']} | {r['accuracy']:.3f} | {r['avg_total_tokens']:.1f} | {r['avg_steps']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_proof_docid_cov']:.3f} | {r['proof_complete_correct_rate']:.3f} | {r['avg_fork_calls']:.2f} | {r['avg_context_controller_calls']:.2f} | {r['avg_context_controller_unfold']:.2f} | {r['avg_context_controller_fork']:.2f} | {r['avg_context_controller_unfold_then_fork']:.2f} | {r['avg_premature_finish_blocked']:.2f} | {r['no_finish_rate']:.3f} |\n")

    with open(analysis / 'phase19_recommendation.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 Recommendation\n\n')
        for line in _recommendation(summary, slice_summary):
            f.write(f'- {line}\n')

if __name__ == '__main__':
    main()
