#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PAPER_FAIR_MAP = {
    'FullHistory': 'FullHistory-PaperFair',
    'FullHistory-Prove': 'FullHistory-PaperFair',
    'SimilarityOnly': 'SimilarityOnly-PaperFair',
    'SimilarityOnly-Prove': 'SimilarityOnly-PaperFair',
    'ProxySummary': 'ProxySummary-PaperFair',
    'ProxySummary-Prove': 'ProxySummary-PaperFair',
    'GoC-Closure-Only': 'GoC-Closure-Only-PaperFair',
    'GoC-ForkOnly': 'GoC-ForkOnly-PaperFair',
    'GoC-Mixed-Heuristic': 'GoC-Mixed-Heuristic-PaperFair',
    'GoC-Mixed-Learned': 'GoC-Mixed-Learned-PaperFair',
}


def _map_paper_fair(methods):
    return _dedupe([PAPER_FAIR_MAP.get(m, m) for m in methods])

ROOT = Path(__file__).resolve().parents[1]


def _dedupe(seq):
    out = []
    seen = set()
    for x in seq:
        key = str(x).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run support-closure benchmark bundle.',
        allow_abbrev=False,
    )
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--methods', type=str, default='SimilarityOnly,SimilarityOnly-Prove,SimilarityOnly-Prove-Fork-Selective,GoC-SimSeed,GoC-SimSeed-Closure')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--task_slices', type=str, default='', help='Optional comma-separated task_slice filter, e.g. support_closure,provenance_required')
    ap.add_argument('--fork_trigger_mode', type=str, default='evidence_gated')
    ap.add_argument('--fork_merge_policy', type=str, default='weak')
    ap.add_argument('--run_fork_verify', action='store_true', default=False, help='Also include GoC-SimSeed-Fork-Verify.')
    ap.add_argument('--paper_fair', action='store_true', default=False, help='Map mixed-method names to their paper-fair variants that disable benchmark-aware proof hooks.')
    ap.add_argument('--paper_fair_only', action='store_true', default=False, help='Run only GoC-Mixed-Learned-PaperFair. Requires --context_controller_model_path.')
    ap.add_argument('--paper_fair_heuristic_only', action='store_true', default=False, help='Run only GoC-Mixed-Heuristic-PaperFair.')
    ap.add_argument('--context_controller_model_path', type=str, default=None, help='Optional learned controller model path for learned-controller methods.')
    ap.add_argument('--context_controller_policy', type=str, default=None, help='Optional learned controller policy label (e.g. phase18_tree or phase18_logreg).')
    ap.add_argument('--enable_context_controller', action='store_true', default=False, help='Explicitly enable downstream context-controller runtime wiring.')
    ap.add_argument('--log_dir', type=str, default=None, help='Optional per-task debug trace directory to pass to the phase19 bundle.')
    args, passthrough = ap.parse_known_args()

    methods = _dedupe([m.strip() for m in str(args.methods).split(',') if m.strip()])
    mode_flags = [bool(args.paper_fair_only), bool(args.paper_fair_heuristic_only)]
    if sum(1 for x in mode_flags if x) > 1:
        raise SystemExit('Use at most one of --paper_fair_only or --paper_fair_heuristic_only')
    if args.paper_fair_heuristic_only:
        methods = ['GoC-Mixed-Heuristic-PaperFair']
    elif args.paper_fair_only:
        methods = ['GoC-Mixed-Learned-PaperFair']
    elif args.paper_fair:
        methods = _map_paper_fair(methods)
    if args.run_fork_verify and 'GoC-SimSeed-Fork-Verify' not in methods:
        methods.append('GoC-SimSeed-Fork-Verify')
    if any(m in methods for m in ('GoC-Mixed-Learned', 'GoC-Mixed-Learned-PaperFair')) and not args.context_controller_model_path:
        raise SystemExit('Learned mixed methods require --context_controller_model_path')

    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_phase19_e2e_fork_bundle.py'),
        '--benchmark_profile', 'phase21_support_closure',
        '--model', args.model,
        '--dotenv', args.dotenv,
        '--task_limit', str(args.task_limit),
        '--seeds', args.seeds,
        '--methods', ','.join(methods),
        '--max_steps', str(args.max_steps),
        '--parallel_tasks', str(args.parallel_tasks),
        '--task_slices', args.task_slices,
        '--fork_trigger_mode', args.fork_trigger_mode,
        '--fork_merge_policy', args.fork_merge_policy,
    ]
    if args.context_controller_model_path:
        cmd.extend(['--context_controller_model_path', args.context_controller_model_path])
    if args.context_controller_policy:
        cmd.extend(['--context_controller_policy', args.context_controller_policy])
    if args.enable_context_controller:
        cmd.append('--enable_context_controller')
    if str(args.log_dir or '').strip():
        cmd.extend(['--log_dir', str(args.log_dir)])

    if passthrough:
        cmd.extend(passthrough)

    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
