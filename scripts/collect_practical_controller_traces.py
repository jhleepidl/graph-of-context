#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for ln in path.read_text(encoding='utf-8').splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _materialize_bundle(raw: str) -> tuple[Path, Optional[Path]]:
    p = Path(raw)
    if p.is_dir():
        return p, None
    if not p.is_file() or p.suffix.lower() != '.zip':
        raise FileNotFoundError(f'Bundle path not found or unsupported: {p}')
    tmpdir = Path(tempfile.mkdtemp(prefix='practical_controller_bundle_'))
    with zipfile.ZipFile(p) as zf:
        zf.extractall(tmpdir)
    return tmpdir, tmpdir


def _discover_bundle_root(base: Path) -> Path:
    if (base / 'run_manifest.json').exists():
        return base
    cands = sorted(base.rglob('run_manifest.json'))
    if not cands:
        raise FileNotFoundError(f'Could not locate run_manifest.json under {base}')
    return cands[0].parent


def _parse_bundle_root_from_stdout(stdout: str) -> Optional[Path]:
    for ln in reversed(stdout.splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        try:
            payload = json.loads(ln)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get('bundle_root'):
            return Path(str(payload['bundle_root']))
    return None


def _run_bundle(args: argparse.Namespace, out_root: Path) -> Path:
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'run_phase19_e2e_fork_bundle.py'),
        '--benchmark_profile', str(args.benchmark_profile),
        '--model', str(args.model),
        '--dotenv', str(args.dotenv),
        '--task_limit', str(args.task_limit),
        '--seeds', str(args.seeds),
        '--methods', str(args.methods),
        '--parallel_tasks', str(args.parallel_tasks),
        '--task_slices', str(args.task_slices),
        '--max_steps', str(args.max_steps),
        '--log_dir', 'traces',
    ]
    if args.context_controller_trace:
        cmd.append('--context_controller_trace')
    else:
        cmd.append('--no_context_controller_trace')
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    bundle_root = _parse_bundle_root_from_stdout(proc.stdout)
    if bundle_root is None:
        raise RuntimeError('Failed to parse bundle_root from run_phase19_e2e_fork_bundle.py output.')
    (out_root / 'run_stdout.log').write_text(proc.stdout, encoding='utf-8')
    (out_root / 'run_stderr.log').write_text(proc.stderr, encoding='utf-8')
    return bundle_root


def _task_map(bundle_root: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for path in sorted((bundle_root / 'data').glob('seed_*')):
        if not path.is_dir():
            continue
        try:
            seed = int(path.name.split('_')[-1])
        except Exception:
            continue
        tasks_path = path / 'tasks.json'
        if not tasks_path.exists():
            continue
        try:
            tasks = _load_json(tasks_path)
        except Exception:
            continue
        if not isinstance(tasks, list):
            continue
        for t in tasks:
            if isinstance(t, dict) and t.get('id'):
                out[(seed, str(t['id']))] = t
    return out


def _results_map(bundle_root: Path) -> Dict[Tuple[int, str, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for path in sorted((bundle_root / 'runs').glob('seed_*')):
        if not path.is_dir():
            continue
        try:
            seed = int(path.name.split('_')[-1])
        except Exception:
            continue
        jsonl = path / 'phase19_results.jsonl'
        if not jsonl.exists():
            continue
        for row in _iter_jsonl(jsonl):
            k = (seed, str(row.get('method') or ''), str(row.get('task_id') or ''))
            out[k] = row
    return out


def _trace_files(bundle_root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for path in sorted((bundle_root / 'runs').glob('seed_*')):
        if not path.is_dir():
            continue
        try:
            seed = int(path.name.split('_')[-1])
        except Exception:
            continue
        trace_root = path / 'traces' / f'seed_{seed}'
        if not trace_root.exists():
            # fallback: any trace files under seed dir
            trace_root = path
        for tf in sorted(trace_root.rglob('trace_*.jsonl')):
            out.append((seed, tf))
    return out


def _final_outcome(row: Dict[str, Any]) -> str:
    if bool(row.get('correct')):
        return 'correct'
    explanation = str(row.get('explanation') or '').lower()
    if 'max_steps reached' in explanation or 'no finish' in explanation or 'no-finish' in explanation:
        return 'no_finish'
    pred = str(row.get('pred') or '').strip()
    if not pred:
        return 'no_finish'
    return 'wrong'


def _controller_events_from_trace(path: Path) -> List[Dict[str, Any]]:
    return [row for row in _iter_jsonl(path) if str(row.get('type') or '') == 'context_controller']


def extract_states(bundle_root: Path, out_jsonl: Path) -> Dict[str, Any]:
    manifest = _load_json(bundle_root / 'run_manifest.json')
    task_map = _task_map(bundle_root)
    result_map = _results_map(bundle_root)
    traces = _trace_files(bundle_root)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state_count = 0
    methods = Counter()
    slices = Counter()
    seeds = Counter()
    with out_jsonl.open('w', encoding='utf-8') as f:
        for seed, trace_path in traces:
            events = _controller_events_from_trace(trace_path)
            if not events:
                continue
            meta = events[0]
            task_id = str(meta.get('task_id') or '')
            method = str(meta.get('method') or '')
            result_row = result_map.get((seed, method, task_id)) or {}
            task = task_map.get((seed, task_id)) or {}
            final_outcome = _final_outcome(result_row) if result_row else 'unknown'
            for ev in events:
                row = {
                    'seed': int(seed),
                    'task_id': task_id,
                    'method': method,
                    'run_tag': str(ev.get('run_tag') or ''),
                    'step': int(ev.get('step') or 0),
                    'action_taken': str(ev.get('action') or 'none'),
                    'trace_reason': str(ev.get('reason') or ''),
                    'features': dict(ev.get('features') or {}),
                    'controller_metadata': dict(ev.get('metadata') or {}),
                    'task_slice': str(task.get('task_slice') or ''),
                    'task_type': str(task.get('task_type') or ''),
                    'benchmark_profile': str(task.get('benchmark_profile') or manifest.get('benchmark_profile') or ''),
                    'proof_required_count': int(task.get('proof_required_count') or 0),
                    'decision_requires_support_closure': bool(task.get('decision_requires_support_closure', False)),
                    'question': str(task.get('question') or ''),
                    'gold': str(task.get('answer') or ''),
                    'final_correct': bool(result_row.get('correct', False)),
                    'final_proof_complete': bool(result_row.get('proof_complete', False)),
                    'final_steps': int(result_row.get('steps') or 0),
                    'final_total_tokens': int(((result_row.get('usage') or {}).get('total_tokens') or 0)),
                    'final_fork_calls': int(((result_row.get('tool_stats') or {}).get('fork_calls') or 0)),
                    'final_unfold_calls': int(((result_row.get('tool_stats') or {}).get('adaptive_unfold_calls') or 0)),
                    'final_outcome': final_outcome,
                    'max_steps': int(manifest.get('max_steps') or 0),
                    'trace_path': str(trace_path.relative_to(bundle_root)),
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                state_count += 1
                methods[method] += 1
                slices[row['task_slice']] += 1
                seeds[int(seed)] += 1
    return {
        'bundle_root': str(bundle_root),
        'benchmark_profile': manifest.get('benchmark_profile'),
        'state_count': state_count,
        'method_counts': dict(methods),
        'slice_counts': dict(slices),
        'seed_counts': dict(seeds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Collect practical controller traces for support-closure controller retraining.')
    ap.add_argument('--bundle', type=str, default=None, help='Existing bundle root or zip to extract traces from. If omitted, a new bundle is run.')
    ap.add_argument('--out_dir', type=Path, default=None, help='Output directory. Default: artifacts/practical_controller/collect_<timestamp>')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--dotenv', type=str, default='.env')
    ap.add_argument('--benchmark_profile', type=str, default='phase21_support_closure')
    ap.add_argument('--methods', type=str, default='GoC-Mixed-Heuristic')
    ap.add_argument('--task_limit', type=int, default=24)
    ap.add_argument('--seeds', type=str, default='7')
    ap.add_argument('--parallel_tasks', type=int, default=12)
    ap.add_argument('--task_slices', type=str, default='support_closure,provenance_required')
    ap.add_argument('--max_steps', type=int, default=44)
    ap.add_argument('--context_controller_trace', action='store_true', default=True)
    ap.add_argument('--no_context_controller_trace', action='store_false', dest='context_controller_trace')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or (ROOT / 'artifacts' / 'practical_controller' / f'collect_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch: Optional[Path] = None
    try:
        if args.bundle:
            base, scratch = _materialize_bundle(str(args.bundle))
            bundle_root = _discover_bundle_root(base)
        else:
            bundle_root = _run_bundle(args, out_dir)
        summary = extract_states(bundle_root, out_dir / 'controller_states.jsonl')
        summary['collected_at'] = ts
        summary['source_bundle'] = str(bundle_root)
        (out_dir / 'collection_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'out_dir': str(out_dir), 'bundle_root': str(bundle_root), 'state_count': summary['state_count']}, ensure_ascii=False))
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
