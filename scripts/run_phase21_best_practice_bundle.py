#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = 'SimilarityOnly-Prove,SimilarityOnly-Prove-Repair,GoC-SimSeed-Closure'


def main() -> None:
    ap = argparse.ArgumentParser(description='Run the paper-facing Phase 21 best-practice comparison bundle.')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--task_slices', type=str, default='', help='Optional hard-slice filter, e.g. support_closure,provenance_required')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--methods', type=str, default=DEFAULT_METHODS,
                    help=f'Comma-separated method list. Default: {DEFAULT_METHODS}')
    args = ap.parse_args()

    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_phase21_support_closure_bundle.py'),
        '--model', args.model,
        '--dotenv', args.dotenv,
        '--task_limit', str(args.task_limit),
        '--seeds', args.seeds,
        '--parallel_tasks', str(args.parallel_tasks),
        '--max_steps', str(args.max_steps),
        '--task_slices', args.task_slices,
        '--methods', args.methods,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
