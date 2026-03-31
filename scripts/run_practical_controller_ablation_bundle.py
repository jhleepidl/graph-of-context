#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


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


def _infer_controller_policy(model_path: str | None, requested: str | None) -> str | None:
    req = str(requested or '').strip()
    if req and req.lower() != 'auto':
        return req
    name = str(model_path or '').lower()
    if 'logreg' in name:
        return 'phase18_logreg'
    if 'tree' in name:
        return 'phase18_tree'
    return 'phase18_tree' if model_path else None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = "FullHistory-Prove,SimilarityOnly-Prove,ProxySummary-Prove,GoC-Closure-Only,GoC-ForkOnly,GoC-Mixed-Heuristic"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run practical controller/mixing ablations on the support-closure benchmark.",
        allow_abbrev=False,
    )
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7,13,23')
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--task_slices', type=str, default='support_closure,provenance_required')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--methods', type=str, default=DEFAULT_METHODS, help=f'Comma-separated method list. Default: {DEFAULT_METHODS}')
    ap.add_argument('--learned_only', action='store_true', default=False, help='Run only GoC-Mixed-Learned. Useful after baselines have already been measured, to avoid rerunning every method.')
    ap.add_argument('--paper_fair', action='store_true', default=False, help='Map mixed-method names to their paper-fair variants that disable benchmark-aware proof hooks.')
    ap.add_argument('--paper_fair_only', action='store_true', default=False, help='Run only GoC-Mixed-Learned-PaperFair. Requires --context_controller_model_path.')
    ap.add_argument('--context_controller_model_path', type=str, default=None, help='Path to a learned context-controller model (.pkl payload). If provided and GoC-Mixed-Learned is absent from --methods, it is appended automatically.')
    ap.add_argument('--context_controller_policy', type=str, default='auto', help='Learned controller policy label. Use auto to infer phase18_tree or phase18_logreg from the model filename.')
    ap.add_argument('--log_dir', type=str, default='traces', help='Per-task debug trace directory to bundle. Relative paths live under each seed run dir. Use empty string to disable.')
    args, passthrough = ap.parse_known_args()

    if args.learned_only and args.paper_fair_only:
        raise SystemExit('Use only one of --learned_only or --paper_fair_only')

    if args.paper_fair_only:
        if not args.context_controller_model_path:
            raise SystemExit('--paper_fair_only requires --context_controller_model_path')
        methods = ['GoC-Mixed-Learned-PaperFair']
    elif args.learned_only:
        if not args.context_controller_model_path:
            raise SystemExit('--learned_only requires --context_controller_model_path')
        methods = ['GoC-Mixed-Learned']
    else:
        methods = _dedupe([m.strip() for m in str(args.methods).split(',') if m.strip()])
        if args.paper_fair:
            mapped = []
            for m in methods:
                if m == 'GoC-Mixed-Learned':
                    mapped.append('GoC-Mixed-Learned-PaperFair')
                elif m == 'GoC-Mixed-Heuristic':
                    mapped.append('GoC-Mixed-Heuristic-PaperFair')
                else:
                    mapped.append(m)
            methods = _dedupe(mapped)
        if args.context_controller_model_path and not any(m in methods for m in ('GoC-Mixed-Learned', 'GoC-Mixed-Learned-PaperFair')):
            methods.append('GoC-Mixed-Learned-PaperFair' if args.paper_fair else 'GoC-Mixed-Learned')

    if any(m in methods for m in ('GoC-Mixed-Learned', 'GoC-Mixed-Learned-PaperFair')) and not args.context_controller_model_path:
        raise SystemExit('Learned mixed methods require --context_controller_model_path')

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
    inferred_policy = _infer_controller_policy(args.context_controller_model_path, args.context_controller_policy)
    if args.context_controller_model_path:
        cmd.extend(['--context_controller_model_path', args.context_controller_model_path])
        cmd.extend(['--enable_context_controller'])
    if inferred_policy:
        cmd.extend(['--context_controller_policy', inferred_policy])
    if str(args.log_dir or '').strip():
        cmd.extend(['--log_dir', str(args.log_dir)])
    if passthrough:
        cmd.extend(passthrough)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
