#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description='Run Phase 21 support-closure benchmark bundle.')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--methods', type=str, default='SimilarityOnly,SimilarityOnly-Prove,GoC-SimSeed,GoC-SimSeed-Closure')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--fork_trigger_mode', type=str, default='evidence_gated')
    ap.add_argument('--fork_merge_policy', type=str, default='weak')
    ap.add_argument('--run_fork_verify', action='store_true', default=False, help='Also include GoC-SimSeed-Fork-Verify.')
    args = ap.parse_args()

    methods = [m.strip() for m in str(args.methods).split(',') if m.strip()]
    if args.run_fork_verify and 'GoC-SimSeed-Fork-Verify' not in methods:
        methods.append('GoC-SimSeed-Fork-Verify')

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
        '--fork_trigger_mode', args.fork_trigger_mode,
        '--fork_merge_policy', args.fork_merge_policy,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
