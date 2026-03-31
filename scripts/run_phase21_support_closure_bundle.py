#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    ap.add_argument('--context_controller_model_path', type=str, default=None, help='Optional learned controller model path for learned-controller methods.')
    ap.add_argument('--context_controller_policy', type=str, default=None, help='Optional learned controller policy label (e.g. phase18_tree or phase18_logreg).')
    ap.add_argument('--enable_context_controller', action='store_true', default=False, help='Explicitly enable downstream context-controller runtime wiring.')
    args, passthrough = ap.parse_known_args()

    methods = _dedupe([m.strip() for m in str(args.methods).split(',') if m.strip()])
    if args.run_fork_verify and 'GoC-SimSeed-Fork-Verify' not in methods:
        methods.append('GoC-SimSeed-Fork-Verify')
    if 'GoC-Mixed-Learned' in methods and not args.context_controller_model_path:
        raise SystemExit('GoC-Mixed-Learned requires --context_controller_model_path')

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

    if passthrough:
        cmd.extend(passthrough)

    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
