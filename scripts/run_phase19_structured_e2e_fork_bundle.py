#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'run_phase19_e2e_fork_bundle.py'


def main() -> None:
    extra = sys.argv[1:]
    if extra and extra[0] == '--':
        extra = extra[1:]
    cmd = [sys.executable, str(TARGET), '--benchmark_profile', 'structured'] + extra
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
