#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.benchmarks.synthetic_browsecomp import SyntheticBrowseComp
from src.runners.llm import run_llm


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _avg(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _default_max_steps(profile: str) -> int:
    name = str(profile or 'standard').strip().lower()
    if name == 'hard_lite':
        return 42
    if name == 'hard':
        return 48
    if name == 'hard_extreme':
        return 56
    if name == 'structured_lite':
        return 40
    if name == 'structured':
        return 44
    if name == 'structured_extreme':
        return 52
    return 35


def _load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    return [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_rows(jsonl_path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r.get('method')), []).append(r)
    out: List[Dict[str, Any]] = []
    for method, rs in sorted(by_method.items()):
        tok = [float((r.get('usage', {}) or {}).get('total_tokens') or 0.0) for r in rs]
        tok_sorted = sorted(tok)
        p95 = tok_sorted[min(len(tok_sorted)-1, int(0.95 * (len(tok_sorted)-1)))] if tok_sorted else 0.0
        out.append({
            'method': method,
            'n': len(rs),
            'accuracy': _avg([1.0 if bool(r.get('correct')) else 0.0 for r in rs]),
            'accuracy_strict': _avg([1.0 if bool(r.get('correct_strict')) else 0.0 for r in rs if r.get('correct_strict') is not None]),
            'avg_total_tokens': _avg(tok),
            'p95_total_tokens': p95,
            'avg_steps': _avg([float(r.get('steps') or 0.0) for r in rs]),
            'avg_docid_cov': _avg([float(r.get('docid_cov') or 0.0) for r in rs]),
            'avg_fork_calls': _avg([float((r.get('tool_stats', {}) or {}).get('fork_calls') or 0.0) for r in rs]),
            'avg_fork_tokens': _avg([float((r.get('tool_stats', {}) or {}).get('fork_tokens') or 0.0) for r in rs]),
        })
    return out


def _write_seed_report(jsonl_path: Path, report_path: Path) -> None:
    rows = _load_rows(jsonl_path)
    summary = _summarize_rows(rows)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# LLM Report (synthetic_browsecomp)\n\n')
        if summary:
            hdr = '| method | n | accuracy | accuracy_strict | avg_total_tokens | p95_total_tokens | avg_steps | avg_docid_coverage | avg_fork_calls | avg_fork_tokens |\n'
            sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
            f.write(hdr); f.write(sep)
            for r in summary:
                f.write(f"| {r['method']} | {r['n']} | {r['accuracy']:.3f} | {r['accuracy_strict']:.3f} | {r['avg_total_tokens']:.1f} | {r['p95_total_tokens']:.1f} | {r['avg_steps']:.1f} | {r['avg_docid_cov']:.3f} | {r['avg_fork_calls']:.2f} | {r['avg_fork_tokens']:.1f} |\n")


def _write_bundle_summary(bundle_root: Path, seeds: List[int]) -> None:
    analysis_root = bundle_root / 'analysis'
    _ensure_dir(analysis_root)
    aggregate_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_jsonl = bundle_root / 'runs' / f'seed_{seed}' / 'phase19_results.jsonl'
        if not seed_jsonl.exists():
            continue
        for row in _summarize_rows(_load_rows(seed_jsonl)):
            row['seed'] = seed
            aggregate_rows.append(row)
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in aggregate_rows:
        by_method.setdefault(r['method'], []).append(r)
    summary_rows: List[Dict[str, Any]] = []
    for method, rs in sorted(by_method.items()):
        summary_rows.append({
            'method': method,
            'n_seeds': len(rs),
            'accuracy_mean': _avg([float(r['accuracy']) for r in rs]),
            'accuracy_strict_mean': _avg([float(r['accuracy_strict']) for r in rs]),
            'avg_total_tokens_mean': _avg([float(r['avg_total_tokens']) for r in rs]),
            'p95_total_tokens_mean': _avg([float(r['p95_total_tokens']) for r in rs]),
            'avg_steps_mean': _avg([float(r['avg_steps']) for r in rs]),
            'avg_docid_cov_mean': _avg([float(r['avg_docid_cov']) for r in rs]),
            'avg_fork_calls_mean': _avg([float(r['avg_fork_calls']) for r in rs]),
            'avg_fork_tokens_mean': _avg([float(r['avg_fork_tokens']) for r in rs]),
        })
    out_csv = analysis_root / 'phase19_e2e_summary.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ['method'])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    with open(analysis_root / 'phase19_e2e_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Phase 19 End-to-End Fork Summary\n\n')
        if summary_rows:
            hdr = '| method | n_seeds | acc | acc_strict | avg_tokens | p95_tokens | avg_steps | docid_cov | avg_fork_calls | avg_fork_tokens |\n'
            sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
            f.write(hdr); f.write(sep)
            for r in summary_rows:
                f.write(f"| {r['method']} | {r['n_seeds']} | {r['accuracy_mean']:.3f} | {r['accuracy_strict_mean']:.3f} | {r['avg_total_tokens_mean']:.1f} | {r['p95_total_tokens_mean']:.1f} | {r['avg_steps_mean']:.1f} | {r['avg_docid_cov_mean']:.3f} | {r['avg_fork_calls_mean']:.2f} | {r['avg_fork_tokens_mean']:.1f} |\n")


def _relabel_methods(rows: List[Dict[str, Any]], label_mode: str, suffix: str, custom_label: str | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        nr = dict(r)
        orig = str(nr.get('method'))
        if custom_label and label_mode == 'single':
            nr['method'] = custom_label
        elif suffix:
            nr['method'] = f'{orig}{suffix}'
        else:
            nr['method'] = orig
        out.append(nr)
    return out


def _dedupe_key(r: Dict[str, Any]) -> Tuple[str, Any]:
    return (str(r.get('method')), r.get('task_id'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Extend an existing Phase19 bundle without rerunning baseline methods.')
    ap.add_argument('--existing_bundle_root', type=str, required=True)
    ap.add_argument('--methods', type=str, required=True)
    ap.add_argument('--result_method_suffix', type=str, default='')
    ap.add_argument('--result_method_label', type=str, default=None, help='Single custom label when exactly one base method is requested.')
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=None)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--parallel_tasks', type=int, default=1)
    ap.add_argument('--budget_active', type=int, default=None)
    ap.add_argument('--budget_unfold', type=int, default=None)
    ap.add_argument('--unfold_k', type=int, default=None)
    ap.add_argument('--fork_max_tokens', type=int, default=None)
    ap.add_argument('--fork_k', type=int, default=None)
    ap.add_argument('--fork_trigger_mode', type=str, default=None)
    ap.add_argument('--fork_debug_force_step', type=int, default=10)
    ap.add_argument('--fork_debug_force_max_calls', type=int, default=1)
    ap.add_argument('--fork_gate_trace', action='store_true', default=None)
    ap.add_argument('--no_fork_gate_trace', action='store_false', dest='fork_gate_trace')
    ap.add_argument('--fork_gate_probe_run_on_ready', action='store_true', default=None)
    ap.add_argument('--no_fork_gate_probe_run_on_ready', action='store_false', dest='fork_gate_probe_run_on_ready')
    ap.add_argument('--fork_min_step', type=int, default=None)
    ap.add_argument('--fork_every_k_steps', type=int, default=None)
    ap.add_argument('--fork_min_open_pages', type=int, default=None)
    ap.add_argument('--fork_min_search_calls', type=int, default=None)
    ap.add_argument('--fork_min_active_tokens', type=int, default=None)
    ap.add_argument('--fork_merge_min_confidence', type=float, default=None)
    ap.add_argument('--fork_merge_policy', type=str, default=None)
    ap.add_argument('--fork_weak_merge_max_chars', type=int, default=None)
    ap.add_argument('--enable_context_controller', action='store_true', default=None)
    ap.add_argument('--disable_context_controller', action='store_false', dest='enable_context_controller')
    ap.add_argument('--context_controller_policy', type=str, default=None)
    ap.add_argument('--context_controller_trace', action='store_true', default=None)
    ap.add_argument('--no_context_controller_trace', action='store_false', dest='context_controller_trace')
    ap.add_argument('--context_controller_support_gap_threshold', type=float, default=None)
    ap.add_argument('--context_controller_budget_pressure_threshold', type=float, default=None)
    ap.add_argument('--context_controller_fork_ambiguity_threshold', type=float, default=None)
    ap.add_argument('--context_controller_model_path', type=str, default=None)
    ap.add_argument('--context_controller_min_confidence', type=float, default=None)
    ap.add_argument('--context_controller_fallback_action', type=str, default=None)
    ap.add_argument('--context_controller_disable_none_action', action='store_true', default=None)
    ap.add_argument('--no_context_controller_disable_none_action', action='store_false', dest='context_controller_disable_none_action')
    ap.add_argument('--context_controller_fork_gate_mode', type=str, default=None)
    ap.add_argument('--context_controller_recheck_after_unfold', action='store_true', default=None)
    ap.add_argument('--no_context_controller_recheck_after_unfold', action='store_false', dest='context_controller_recheck_after_unfold')
    ap.add_argument('--fork_controller_max_calls', type=int, default=None)
    ap.add_argument('--fork_controller_cooldown_steps', type=int, default=None)
    ap.add_argument('--fork_controller_min_open_pages', type=int, default=None)
    ap.add_argument('--fork_controller_min_active_tokens', type=int, default=None)
    ap.add_argument('--fork_controller_min_branch_score', type=float, default=None)
    ap.add_argument('--fork_controller_min_ambiguity', type=float, default=None)
    ap.add_argument('--fork_controller_min_pressure', type=float, default=None)
    ap.add_argument('--fork_controller_allow_open_only', action='store_true', default=None)
    ap.add_argument('--no_fork_controller_allow_open_only', action='store_false', dest='fork_controller_allow_open_only')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    bundle_root = Path(args.existing_bundle_root).resolve()
    manifest_path = bundle_root / 'run_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'run_manifest.json not found under {bundle_root}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    methods = [m.strip() for m in str(args.methods).split(',') if m.strip()]
    if not methods:
        raise SystemExit('No methods requested.')
    if args.result_method_label and len(methods) != 1:
        raise SystemExit('--result_method_label requires exactly one base method.')
    label_mode = 'single' if args.result_method_label else 'suffix'

    seeds = [int(s.strip()) for s in str(args.seeds).split(',') if s.strip()] if args.seeds else [int(s) for s in manifest.get('seeds', [])]
    if not seeds:
        raise SystemExit('No seeds available.')

    model = args.model or str(manifest.get('model') or 'gpt-4.1-mini')
    task_limit = int(args.task_limit if args.task_limit is not None else manifest.get('task_limit', 24))
    benchmark_profile = str(manifest.get('benchmark_profile') or 'standard')
    max_steps = int(args.max_steps if args.max_steps is not None else manifest.get('max_steps', _default_max_steps(benchmark_profile)))
    fork_cfg = dict(manifest.get('fork_runtime_kwargs') or {})
    ctrl_cfg = dict(manifest.get('context_controller_kwargs') or {})

    def pick(name: str, current: Any) -> Any:
        val = getattr(args, name)
        return current if val is None else val

    run_kwargs = {
        'fork_trigger_mode': pick('fork_trigger_mode', fork_cfg.get('fork_trigger_mode', 'evidence_gated')),
        'fork_gate_trace': bool(pick('fork_gate_trace', fork_cfg.get('fork_gate_trace', True))),
        'fork_gate_probe_run_on_ready': bool(pick('fork_gate_probe_run_on_ready', fork_cfg.get('fork_gate_probe_run_on_ready', False))),
        'fork_min_step': int(pick('fork_min_step', fork_cfg.get('fork_min_step', 4))),
        'fork_every_k_steps': int(pick('fork_every_k_steps', fork_cfg.get('fork_every_k_steps', 3))),
        'fork_min_open_pages': int(pick('fork_min_open_pages', fork_cfg.get('fork_min_open_pages', 2))),
        'fork_min_search_calls': int(pick('fork_min_search_calls', fork_cfg.get('fork_min_search_calls', 1))),
        'fork_min_active_tokens': int(pick('fork_min_active_tokens', fork_cfg.get('fork_min_active_tokens', 500))),
        'fork_merge_min_confidence': float(pick('fork_merge_min_confidence', fork_cfg.get('fork_merge_min_confidence', 0.67))),
        'fork_merge_policy': str(pick('fork_merge_policy', fork_cfg.get('fork_merge_policy', 'weak'))),
        'fork_weak_merge_max_chars': int(pick('fork_weak_merge_max_chars', fork_cfg.get('fork_weak_merge_max_chars', 240))),
        'fork_max_tokens': int(args.fork_max_tokens if args.fork_max_tokens is not None else 160),
        'fork_k': int(args.fork_k if args.fork_k is not None else 6),
        'budget_active': int(args.budget_active if args.budget_active is not None else 1200),
        'budget_unfold': int(args.budget_unfold if args.budget_unfold is not None else 650),
        'unfold_k': int(args.unfold_k if args.unfold_k is not None else 8),
        'enable_context_controller': bool(pick('enable_context_controller', ctrl_cfg.get('enable_context_controller', False))),
        'context_controller_policy': str(pick('context_controller_policy', ctrl_cfg.get('context_controller_policy', 'uncertainty_aware'))),
        'context_controller_trace': bool(pick('context_controller_trace', ctrl_cfg.get('context_controller_trace', True))),
        'context_controller_support_gap_threshold': float(pick('context_controller_support_gap_threshold', ctrl_cfg.get('context_controller_support_gap_threshold', 0.20))),
        'context_controller_budget_pressure_threshold': float(pick('context_controller_budget_pressure_threshold', ctrl_cfg.get('context_controller_budget_pressure_threshold', 0.80))),
        'context_controller_fork_ambiguity_threshold': float(pick('context_controller_fork_ambiguity_threshold', ctrl_cfg.get('context_controller_fork_ambiguity_threshold', 0.45))),
        'context_controller_model_path': pick('context_controller_model_path', ctrl_cfg.get('context_controller_model_path')),
        'context_controller_min_confidence': float(pick('context_controller_min_confidence', ctrl_cfg.get('context_controller_min_confidence', 0.0))),
        'context_controller_fallback_action': str(pick('context_controller_fallback_action', ctrl_cfg.get('context_controller_fallback_action', 'unfold'))),
        'context_controller_disable_none_action': bool(pick('context_controller_disable_none_action', ctrl_cfg.get('context_controller_disable_none_action', False))),
        'context_controller_fork_gate_mode': str(pick('context_controller_fork_gate_mode', ctrl_cfg.get('context_controller_fork_gate_mode', 'integrated'))),
        'context_controller_recheck_after_unfold': bool(pick('context_controller_recheck_after_unfold', ctrl_cfg.get('context_controller_recheck_after_unfold', True))),
        'fork_controller_max_calls': int(pick('fork_controller_max_calls', ctrl_cfg.get('fork_controller_max_calls', 2))),
        'fork_controller_cooldown_steps': int(pick('fork_controller_cooldown_steps', ctrl_cfg.get('fork_controller_cooldown_steps', 5))),
        'fork_controller_min_open_pages': int(pick('fork_controller_min_open_pages', ctrl_cfg.get('fork_controller_min_open_pages', 2))),
        'fork_controller_min_active_tokens': int(pick('fork_controller_min_active_tokens', ctrl_cfg.get('fork_controller_min_active_tokens', 350))),
        'fork_controller_min_branch_score': float(pick('fork_controller_min_branch_score', ctrl_cfg.get('fork_controller_min_branch_score', 0.18))),
        'fork_controller_min_ambiguity': float(pick('fork_controller_min_ambiguity', ctrl_cfg.get('fork_controller_min_ambiguity', 0.35))),
        'fork_controller_min_pressure': float(pick('fork_controller_min_pressure', ctrl_cfg.get('fork_controller_min_pressure', 0.45))),
        'fork_controller_allow_open_only': bool(pick('fork_controller_allow_open_only', ctrl_cfg.get('fork_controller_allow_open_only', True))),
    }

    if args.dry_run:
        print(json.dumps({
            'bundle_root': str(bundle_root),
            'methods': methods,
            'seeds': seeds,
            'result_method_suffix': args.result_method_suffix,
            'result_method_label': args.result_method_label,
            'model': model,
            'task_limit': task_limit,
            'run_kwargs': run_kwargs,
        }, ensure_ascii=False, indent=2))
        return

    bench = SyntheticBrowseComp()
    added_labels: List[str] = []
    for seed in seeds:
        seed_data = bundle_root / 'data' / f'seed_{seed}'
        seed_runs = bundle_root / 'runs' / f'seed_{seed}'
        if not seed_data.exists():
            raise SystemExit(f'Missing data dir for seed {seed}: {seed_data}')
        _ensure_dir(seed_runs)
        main_jsonl = seed_runs / 'phase19_results.jsonl'
        main_report = seed_runs / 'phase19_report.md'
        tmp_jsonl = seed_runs / '.phase19_extension_tmp.jsonl'
        tmp_report = seed_runs / '.phase19_extension_tmp.md'
        if tmp_jsonl.exists():
            tmp_jsonl.unlink()
        if tmp_report.exists():
            tmp_report.unlink()
        print(json.dumps({'seed': seed, 'data_dir': str(seed_data), 'methods': methods, 'tmp_out': str(tmp_jsonl)}, ensure_ascii=False))
        run_llm(
            benchmark=bench,
            data_dir=str(seed_data),
            methods=methods,
            out_results_path=str(tmp_jsonl),
            out_report_path=str(tmp_report),
            model=model,
            dotenv_path=str(args.dotenv),
            max_steps=35,
            max_json_retries=2,
            task_limit=task_limit,
            retriever_kind='bm25',
            parallel_tasks=int(args.parallel_tasks),
            prompt_context_chars=0,
            log_context_chars=2500,
            stage_aware_unfold_on_final=True,
            stage_aware_unfold_on_commit=True,
            enable_unfold_trigger=True,
            fork_include_recent_active=True,
            fork_recent_active_n=4,
            save_goc_internal_graph=False,
            resume=False,
            **run_kwargs,
        )
        existing_rows = _load_rows(main_jsonl)
        existing_keys = {_dedupe_key(r) for r in existing_rows}
        new_rows_raw = _load_rows(tmp_jsonl)
        new_rows = _relabel_methods(new_rows_raw, label_mode=label_mode, suffix=args.result_method_suffix, custom_label=args.result_method_label)
        merged = list(existing_rows)
        for r in new_rows:
            k = _dedupe_key(r)
            if k not in existing_keys:
                merged.append(r)
                existing_keys.add(k)
        _write_rows(main_jsonl, merged)
        _write_seed_report(main_jsonl, main_report)
        if tmp_jsonl.exists():
            tmp_jsonl.unlink()
        if tmp_report.exists():
            tmp_report.unlink()
        added_labels.extend(sorted({str(r.get('method')) for r in new_rows}))

    manifest.setdefault('extensions', []).append({
        'extended_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'base_methods': methods,
        'result_method_suffix': args.result_method_suffix,
        'result_method_label': args.result_method_label,
        'seeds': seeds,
        'model': model,
        'task_limit': task_limit,
        'dotenv': str(args.dotenv),
        'run_kwargs': run_kwargs,
    })
    existing_methods = [str(m) for m in manifest.get('methods', [])]
    manifest['methods'] = sorted(set(existing_methods + added_labels))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_bundle_summary(bundle_root, seeds=[int(s) for s in manifest.get('seeds', seeds)])
    print(json.dumps({'bundle_root': str(bundle_root), 'updated_methods': manifest['methods']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
