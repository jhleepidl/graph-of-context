#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    ap = argparse.ArgumentParser(description='Low-cost phase19 pilot wrapper with recommended defaults.')
    ap.add_argument('--profile', type=str, default='phase20_support_recovery', choices=['standard','structured_support_pilot','phase20_support_recovery','structured_lite'])
    ap.add_argument('--task_limit', type=int, default=12)
    ap.add_argument('--seeds', type=str, default='7')
    ap.add_argument('--methods', type=str, default='FullHistory,SimilarityOnly,GoC,GoC-Fork-Dep')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('extra', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_phase19_e2e_fork_bundle.py'),
        '--benchmark_profile', str(args.profile),
        '--task_limit', str(args.task_limit),
        '--seeds', str(args.seeds),
        '--methods', str(args.methods),
        '--model', str(args.model),
        '--dotenv', str(args.dotenv),
        '--fork_trigger_mode', 'evidence_gated',
        '--fork_merge_policy', 'weak',
    ]
    if args.extra:
        cmd.extend(args.extra)
    raise SystemExit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
