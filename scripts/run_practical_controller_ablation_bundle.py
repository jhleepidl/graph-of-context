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
    ap.add_argument('--paper_fair', action='store_true', default=False, help='Map supported methods to their paper-fair variants.')
    ap.add_argument('--paper_fair_baselines_only', action='store_true', default=False, help='Run the main practical baseline set in paper-fair mode.')
    ap.add_argument('--context_controller_model_path', type=str, default=None, help='Path to a learned context-controller model (.pkl payload). If provided and GoC-Mixed-Learned is absent from --methods, it is appended automatically.')
    ap.add_argument('--context_controller_policy', type=str, default='auto', help='Learned controller policy label. Use auto to infer phase18_tree or phase18_logreg from the model filename.')
    ap.add_argument('--log_dir', type=str, default='traces', help='Per-task debug trace directory to bundle. Relative paths live under each seed run dir. Use empty string to disable.')
    ap.add_argument('--openai_api_mode', type=str, default=None, help='Override OpenAI client API mode (e.g. auto or responses).')
    ap.add_argument('--openai_reasoning_effort', type=str, default=None, help='Optional reasoning effort for GPT-5.* models (e.g. none, low, medium, high).')
    ap.add_argument('--openai_verbosity', type=str, default=None, help='Optional verbosity for GPT-5.* models (low, medium, high).')
    ap.add_argument('--openai_max_output_tokens', type=int, default=None, help='Optional max_output_tokens override for the OpenAI Responses API.')
    args, passthrough = ap.parse_known_args()

    if args.learned_only and args.paper_fair_baselines_only:
        raise SystemExit('Use at most one of --learned_only or --paper_fair_baselines_only')

    if args.learned_only:
        if not args.context_controller_model_path:
            raise SystemExit('--learned_only requires --context_controller_model_path')
        methods = ['GoC-Mixed-Learned']
    elif args.paper_fair_baselines_only:
        methods = [
            'FullHistory-PaperFair',
            'SimilarityOnly-PaperFair',
            'ProxySummary-PaperFair',
            'GoC-Closure-Only-PaperFair',
            'GoC-ForkOnly-PaperFair',
            'GoC-Mixed-Heuristic-PaperFair',
            'GoC-Mixed-Learned-PaperFair',
        ]
    else:
        methods = _dedupe([m.strip() for m in str(args.methods).split(',') if m.strip()])
        if args.context_controller_model_path and 'GoC-Mixed-Learned' not in methods and 'GoC-Mixed-Learned-PaperFair' not in methods:
            methods.append('GoC-Mixed-Learned')

    if args.paper_fair:
        methods = _map_paper_fair(methods)

    if any(m in methods for m in ('GoC-Mixed-Learned', 'GoC-Mixed-Learned-PaperFair')) and not args.context_controller_model_path:
        raise SystemExit('GoC-Mixed-Learned methods require --context_controller_model_path')

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
    if args.openai_api_mode:
        cmd.extend(['--openai_api_mode', str(args.openai_api_mode)])
    if args.openai_reasoning_effort:
        cmd.extend(['--openai_reasoning_effort', str(args.openai_reasoning_effort)])
    if args.openai_verbosity:
        cmd.extend(['--openai_verbosity', str(args.openai_verbosity)])
    if args.openai_max_output_tokens is not None:
        cmd.extend(['--openai_max_output_tokens', str(int(args.openai_max_output_tokens))])
    if passthrough:
        cmd.extend(passthrough)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
