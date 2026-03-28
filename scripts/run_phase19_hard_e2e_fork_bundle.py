#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'run_phase19_e2e_fork_bundle.py'


def main() -> None:
    ap = argparse.ArgumentParser(description='Convenience wrapper for Phase19 hard-profile bundle runs.')
    ap.add_argument('extra', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = [
        sys.executable,
        str(TARGET),
        '--benchmark_profile', 'hard',
    ]
    if args.extra:
        cmd.extend(args.extra)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
