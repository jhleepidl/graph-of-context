#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = "FullHistory-Prove,SimilarityOnly-Prove,ProxySummary-Prove,GoC-Closure-Only,GoC-ForkOnly,GoC-Mixed-Heuristic"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run practical controller/mixing ablations on the support-closure benchmark.")
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--task_slices', type=str, default='support_closure,provenance_required')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--methods', type=str, default=DEFAULT_METHODS, help=f'Comma-separated method list. Default: {DEFAULT_METHODS}')
    ap.add_argument('--context_controller_model_path', type=str, default=None, help='Path to a learned context-controller model. If provided and GoC-Mixed-Learned is absent from --methods, it is appended automatically.')
    args = ap.parse_args()

    methods = [m.strip() for m in str(args.methods).split(',') if m.strip()]
    if args.context_controller_model_path and 'GoC-Mixed-Learned' not in methods:
        methods.append('GoC-Mixed-Learned')
    if 'GoC-Mixed-Learned' in methods and not args.context_controller_model_path:
        raise SystemExit('GoC-Mixed-Learned requires --context_controller_model_path')

    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_phase21_support_closure_bundle.py'),
        '--model', args.model,
        '--dotenv', args.dotenv,
        '--task_limit', str(args.task_limit),
        '--seeds', args.seeds,
        '--parallel_tasks', str(args.parallel_tasks),
        '--task_slices', args.task_slices,
        '--max_steps', str(args.max_steps),
        '--methods', ','.join(methods),
    ]
    if args.context_controller_model_path:
        cmd.extend(['--context_controller_model_path', args.context_controller_model_path])
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
