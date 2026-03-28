#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, sys, uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.benchmarks.synthetic_browsecomp import SyntheticBrowseComp
from src.runners.llm import run_llm


@dataclass
class RunEntry:
    seed: int
    benchmark: str
    methods: List[str]
    model: str
    task_limit: int
    data_dir: str
    out_results: str
    out_report: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _avg(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _summarize_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
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


def main() -> None:
    ap = argparse.ArgumentParser(description='Phase 19: end-to-end LLM fork bundle on Synthetic BrowseComp.')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--parallel_tasks', type=int, default=1)
    ap.add_argument('--budget_active', type=int, default=1200)
    ap.add_argument('--budget_unfold', type=int, default=650)
    ap.add_argument('--unfold_k', type=int, default=8)
    ap.add_argument('--fork_max_tokens', type=int, default=160)
    ap.add_argument('--fork_k', type=int, default=6)
    ap.add_argument('--fork_trigger_mode', type=str, default='evidence_gated')
    ap.add_argument('--fork_debug_force_step', type=int, default=10)
    ap.add_argument('--fork_debug_force_max_calls', type=int, default=1)
    ap.add_argument('--fork_gate_trace', action='store_true', default=True)
    ap.add_argument('--no_fork_gate_trace', action='store_false', dest='fork_gate_trace')
    ap.add_argument('--fork_gate_probe_run_on_ready', action='store_true', default=False)
    ap.add_argument('--fork_min_step', type=int, default=4)
    ap.add_argument('--fork_every_k_steps', type=int, default=3)
    ap.add_argument('--fork_min_open_pages', type=int, default=2)
    ap.add_argument('--fork_min_search_calls', type=int, default=1)
    ap.add_argument('--fork_min_active_tokens', type=int, default=500)
    ap.add_argument('--fork_merge_min_confidence', type=float, default=0.67)
    ap.add_argument('--fork_merge_policy', type=str, default='full')
    ap.add_argument('--fork_weak_merge_max_chars', type=int, default=240)
    ap.add_argument('--enable_context_controller', action='store_true', default=False)
    ap.add_argument('--context_controller_policy', type=str, default='uncertainty_aware')
    ap.add_argument('--context_controller_trace', action='store_true', default=True)
    ap.add_argument('--no_context_controller_trace', action='store_false', dest='context_controller_trace')
    ap.add_argument('--context_controller_support_gap_threshold', type=float, default=0.20)
    ap.add_argument('--context_controller_budget_pressure_threshold', type=float, default=0.80)
    ap.add_argument('--context_controller_fork_ambiguity_threshold', type=float, default=0.45)
    ap.add_argument('--context_controller_model_path', type=str, default=None)
    ap.add_argument('--context_controller_min_confidence', type=float, default=0.0)
    ap.add_argument('--context_controller_fallback_action', type=str, default='unfold')
    ap.add_argument('--context_controller_disable_none_action', action='store_true', default=False)
    ap.add_argument('--context_controller_fork_gate_mode', type=str, default='integrated')
    ap.add_argument('--context_controller_recheck_after_unfold', action='store_true', default=True)
    ap.add_argument('--no_context_controller_recheck_after_unfold', action='store_false', dest='context_controller_recheck_after_unfold')
    ap.add_argument('--fork_controller_max_calls', type=int, default=2)
    ap.add_argument('--fork_controller_cooldown_steps', type=int, default=5)
    ap.add_argument('--fork_controller_min_open_pages', type=int, default=2)
    ap.add_argument('--fork_controller_min_active_tokens', type=int, default=350)
    ap.add_argument('--fork_controller_min_branch_score', type=float, default=0.18)
    ap.add_argument('--fork_controller_min_ambiguity', type=float, default=0.35)
    ap.add_argument('--fork_controller_min_pressure', type=float, default=0.45)
    ap.add_argument('--fork_controller_allow_open_only', action='store_true', default=True)
    ap.add_argument('--no_fork_controller_allow_open_only', action='store_false', dest='fork_controller_allow_open_only')
    ap.add_argument('--methods', type=str, default='FullHistory,SimilarityOnly,GoC,GoC-Fork-Dep,GoC-Fork-Sim,GoC-Fork-Full')
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--n_entities', type=int, default=100)
    ap.add_argument('--n_tasks', type=int, default=48)
    ap.add_argument('--noise_docs', type=int, default=180)
    ap.add_argument('--distractors_per_entity', type=int, default=3)
    ap.add_argument('--benchmark_profile', type=str, default='standard', choices=['standard', 'hard'])
    ap.add_argument('--hard_compare_ratio', type=float, default=0.35)
    ap.add_argument('--hard_late_binding_ratio', type=float, default=0.35)
    ap.add_argument('--hard_branch_merge_ratio', type=float, default=0.30)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_id = uuid.uuid4().hex[:8]
    bundle_name = f'goc_phase19_e2e_fork_{ts}_{bundle_id}'
    bundle_root = ROOT / 'experiment_bundles' / bundle_name
    data_root = bundle_root / 'data'
    runs_root = bundle_root / 'runs'
    analysis_root = bundle_root / 'analysis'
    _ensure_dir(data_root); _ensure_dir(runs_root); _ensure_dir(analysis_root)

    task_limit = int(args.task_limit)
    n_entities = int(args.n_entities)
    n_tasks = int(args.n_tasks)
    noise_docs = int(args.noise_docs)
    distractors_per_entity = int(args.distractors_per_entity)
    if args.smoke:
        task_limit = min(task_limit, 8)
        n_tasks = min(n_tasks, 16)
        n_entities = min(n_entities, 40)
        noise_docs = min(noise_docs, 80)

    methods = [m.strip() for m in str(args.methods).split(',') if m.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(',') if s.strip()]
    bench = SyntheticBrowseComp()

    manifest: Dict[str, Any] = {
        'bundle_name': bundle_name,
        'created_at': ts,
        'benchmark': 'synthetic_browsecomp',
        'benchmark_profile': str(args.benchmark_profile),
        'model': args.model,
        'task_limit': task_limit,
        'methods': methods,
        'seeds': seeds,
        'fork_runtime_kwargs': {
            'fork_trigger_mode': str(args.fork_trigger_mode),
            'fork_gate_trace': bool(args.fork_gate_trace),
            'fork_gate_probe_run_on_ready': bool(args.fork_gate_probe_run_on_ready),
            'fork_min_step': int(args.fork_min_step),
            'fork_every_k_steps': int(args.fork_every_k_steps),
            'fork_min_open_pages': int(args.fork_min_open_pages),
            'fork_min_search_calls': int(args.fork_min_search_calls),
            'fork_min_active_tokens': int(args.fork_min_active_tokens),
            'fork_merge_min_confidence': float(args.fork_merge_min_confidence),
            'fork_merge_policy': str(args.fork_merge_policy),
            'fork_weak_merge_max_chars': int(args.fork_weak_merge_max_chars),
        },
        'context_controller_kwargs': {
            'enable_context_controller': bool(args.enable_context_controller),
            'context_controller_policy': str(args.context_controller_policy),
            'context_controller_trace': bool(args.context_controller_trace),
            'context_controller_support_gap_threshold': float(args.context_controller_support_gap_threshold),
            'context_controller_budget_pressure_threshold': float(args.context_controller_budget_pressure_threshold),
            'context_controller_fork_ambiguity_threshold': float(args.context_controller_fork_ambiguity_threshold),
            'context_controller_model_path': str(args.context_controller_model_path) if args.context_controller_model_path else None,
            'context_controller_min_confidence': float(args.context_controller_min_confidence),
            'context_controller_fallback_action': str(args.context_controller_fallback_action),
            'context_controller_disable_none_action': bool(args.context_controller_disable_none_action),
            'context_controller_fork_gate_mode': str(args.context_controller_fork_gate_mode),
            'context_controller_recheck_after_unfold': bool(args.context_controller_recheck_after_unfold),
            'fork_controller_max_calls': int(args.fork_controller_max_calls),
            'fork_controller_cooldown_steps': int(args.fork_controller_cooldown_steps),
            'fork_controller_min_open_pages': int(args.fork_controller_min_open_pages),
            'fork_controller_min_active_tokens': int(args.fork_controller_min_active_tokens),
            'fork_controller_min_branch_score': float(args.fork_controller_min_branch_score),
            'fork_controller_min_ambiguity': float(args.fork_controller_min_ambiguity),
            'fork_controller_min_pressure': float(args.fork_controller_min_pressure),
            'fork_controller_allow_open_only': bool(args.fork_controller_allow_open_only),
        },
        'prepare_kwargs': {
            'n_entities': n_entities,
            'n_tasks': n_tasks,
            'distractors_per_entity': distractors_per_entity,
            'noise_docs': noise_docs,
            'long_horizon': True,
            'long_desc_words': 320,
            'related_degree': 3,
            'n_projects_per_task': 10,
            'hop_steps': 4,
            'long_task_ratio': 0.70,
            'late_binding': True,
            'late_binding_ratio': 0.60,
            'late_binding_topn': 2,
            'branch_merge': True,
            'branch_merge_ratio': 0.35,
            'branch_merge_group_min': 2,
            'benchmark_profile': str(args.benchmark_profile),
            'hard_mode': bool(str(args.benchmark_profile) == 'hard'),
            'hard_compare_ratio': float(args.hard_compare_ratio),
            'hard_late_binding_ratio': float(args.hard_late_binding_ratio),
            'hard_branch_merge_ratio': float(args.hard_branch_merge_ratio),
        },
        'runs': [],
    }

    aggregate_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_data = data_root / f'seed_{seed}'
        seed_runs = runs_root / f'seed_{seed}'
        _ensure_dir(seed_data); _ensure_dir(seed_runs)
        bench.prepare(
            data_dir=str(seed_data),
            n_entities=n_entities,
            n_tasks=n_tasks,
            distractors_per_entity=distractors_per_entity,
            noise_docs=noise_docs,
            seed=seed,
            long_horizon=True,
            long_desc_words=320,
            related_degree=3,
            n_projects_per_task=10,
            hop_steps=4,
            long_task_ratio=0.70,
            late_binding=True,
            late_binding_ratio=0.60,
            late_binding_topn=2,
            branch_merge=True,
            branch_merge_ratio=0.35,
            branch_merge_group_min=2,
            benchmark_profile=str(args.benchmark_profile),
            hard_mode=bool(str(args.benchmark_profile) == 'hard'),
            hard_compare_ratio=float(args.hard_compare_ratio),
            hard_late_binding_ratio=float(args.hard_late_binding_ratio),
            hard_branch_merge_ratio=float(args.hard_branch_merge_ratio),
        )
        out_jsonl = seed_runs / 'phase19_results.jsonl'
        out_report = seed_runs / 'phase19_report.md'
        run_llm(
            benchmark=bench,
            data_dir=str(seed_data),
            methods=methods,
            out_results_path=str(out_jsonl),
            out_report_path=str(out_report),
            model=str(args.model),
            dotenv_path=str(args.dotenv),
            max_steps=35,
            max_json_retries=2,
            budget_active=int(args.budget_active),
            budget_unfold=int(args.budget_unfold),
            unfold_k=int(args.unfold_k),
            task_limit=task_limit,
            retriever_kind='bm25',
            parallel_tasks=int(args.parallel_tasks),
            prompt_context_chars=0,
            log_context_chars=2500,
            stage_aware_unfold_on_final=True,
            stage_aware_unfold_on_commit=True,
            enable_unfold_trigger=True,
            fork_trigger_mode=str(args.fork_trigger_mode),
            fork_gate_trace=bool(args.fork_gate_trace),
            fork_gate_probe_run_on_ready=bool(args.fork_gate_probe_run_on_ready),
            fork_min_step=int(args.fork_min_step),
            fork_every_k_steps=int(args.fork_every_k_steps),
            fork_min_open_pages=int(args.fork_min_open_pages),
            fork_min_search_calls=int(args.fork_min_search_calls),
            fork_min_active_tokens=int(args.fork_min_active_tokens),
            fork_merge_min_confidence=float(args.fork_merge_min_confidence),
            fork_merge_policy=str(args.fork_merge_policy),
            fork_weak_merge_max_chars=int(args.fork_weak_merge_max_chars),
            fork_max_tokens=int(args.fork_max_tokens),
            fork_k=int(args.fork_k),
            fork_include_recent_active=True,
            fork_recent_active_n=4,
            enable_context_controller=bool(args.enable_context_controller),
            context_controller_policy=str(args.context_controller_policy),
            context_controller_trace=bool(args.context_controller_trace),
            context_controller_support_gap_threshold=float(args.context_controller_support_gap_threshold),
            context_controller_budget_pressure_threshold=float(args.context_controller_budget_pressure_threshold),
            context_controller_fork_ambiguity_threshold=float(args.context_controller_fork_ambiguity_threshold),
            context_controller_model_path=str(args.context_controller_model_path) if args.context_controller_model_path else None,
            context_controller_min_confidence=float(args.context_controller_min_confidence),
            context_controller_fallback_action=str(args.context_controller_fallback_action),
            context_controller_disable_none_action=bool(args.context_controller_disable_none_action),
            context_controller_fork_gate_mode=str(args.context_controller_fork_gate_mode),
            context_controller_recheck_after_unfold=bool(args.context_controller_recheck_after_unfold),
            fork_controller_max_calls=int(args.fork_controller_max_calls),
            fork_controller_cooldown_steps=int(args.fork_controller_cooldown_steps),
            fork_controller_min_open_pages=int(args.fork_controller_min_open_pages),
            fork_controller_min_active_tokens=int(args.fork_controller_min_active_tokens),
            fork_controller_min_branch_score=float(args.fork_controller_min_branch_score),
            fork_controller_min_ambiguity=float(args.fork_controller_min_ambiguity),
            fork_controller_min_pressure=float(args.fork_controller_min_pressure),
            fork_controller_allow_open_only=bool(args.fork_controller_allow_open_only),
            save_goc_internal_graph=False,
            resume=False,
        )
        manifest['runs'].append(asdict(RunEntry(
            seed=seed,
            benchmark='synthetic_browsecomp',
            methods=methods,
            model=str(args.model),
            task_limit=task_limit,
            data_dir=str(seed_data.relative_to(bundle_root)),
            out_results=str(out_jsonl.relative_to(bundle_root)),
            out_report=str(out_report.relative_to(bundle_root)),
        )))
        seed_summary = _summarize_jsonl(out_jsonl)
        for row in seed_summary:
            row['seed'] = seed
            aggregate_rows.append(row)

    # aggregate across seeds
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

    import csv
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

    (bundle_root / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'bundle_root': str(bundle_root), 'summary_csv': str(out_csv)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
