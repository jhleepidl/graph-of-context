import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import csv

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def _flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # Keep simple scalar params; stringify others.
    out = {}
    for k,v in (params or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union keys
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _pivot_mean(rows: List[Dict[str, Any]], x: str, y: str, value: str, method: Optional[str] = None) -> Tuple[List[str], List[str], List[List[Optional[float]]]]:
    # Build pivot table: rows indexed by x, columns by y, values averaged.
    filtered = [r for r in rows if (method is None or r.get("method")==method)]
    xs = sorted({str(r.get(x)) for r in filtered})
    ys = sorted({str(r.get(y)) for r in filtered})
    # collect
    buckets: Dict[Tuple[str,str], List[float]] = {}
    for r in filtered:
        xi = str(r.get(x))
        yi = str(r.get(y))
        try:
            val = float(r.get(value))
        except Exception:
            continue
        buckets.setdefault((xi, yi), []).append(val)
    grid: List[List[Optional[float]]] = []
    for xi in xs:
        row = []
        for yi in ys:
            vals = buckets.get((xi, yi), [])
            row.append(sum(vals)/len(vals) if vals else None)
        grid.append(row)
    return xs, ys, grid

def _write_pgm_heatmap(path: Path, xs: List[str], ys: List[str], grid: List[List[Optional[float]]], title: str):
    """Write a very simple grayscale heatmap as PGM (portable graymap).

    This avoids dependencies (matplotlib). You can open it with many image viewers.
    Cells with None are white.
    """
    # Determine min/max among non-None
    vals = [v for row in grid for v in row if v is not None]
    if not vals:
        path.write_text("", encoding="utf-8")
        return
    vmin, vmax = min(vals), max(vals)
    # Avoid division by zero
    denom = (vmax - vmin) if vmax != vmin else 1.0

    h = len(xs)
    w = len(ys)
    # upscale each cell by factor for readability
    scale = 30
    img_w = w * scale
    img_h = h * scale

    # Build pixels
    pixels = []
    for i in range(h):
        for _ in range(scale):
            rowpix = []
            for j in range(w):
                v = grid[i][j]
                if v is None:
                    gray = 255
                else:
                    # invert: higher value darker
                    t = (v - vmin) / denom
                    gray = int(255 * (1.0 - t))
                rowpix.extend([gray]*scale)
            pixels.append(rowpix)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = f"P5\n# {title}\n{img_w} {img_h}\n255\n"
        f.write(header.encode("ascii", errors="ignore"))
        for r in pixels:
            f.write(bytes(r))

def main():
    ap = argparse.ArgumentParser(description="Summarize sweep master JSONL into CSV + optional heatmaps from traces.")
    ap.add_argument("--master", type=str, required=True, help="Path to sweep master JSONL (sweep_summary_*.jsonl).")
    ap.add_argument("--out_dir", type=str, default="sweep_summary", help="Output directory for CSV and plots.")
    ap.add_argument("--include_traces", action="store_true", help="If set, also parse trace files to get per-step stats.")
    ap.add_argument("--pivot_x", type=str, default=None, help="Param key for pivot rows (e.g., budget_active).")
    ap.add_argument("--pivot_y", type=str, default=None, help="Param key for pivot cols (e.g., unfold_k).")
    ap.add_argument("--pivot_value", type=str, default="accuracy", help="Metric for pivot table (accuracy/avg_total_tokens/avg_steps/...).")
    ap.add_argument("--pivot_method", type=str, default=None, help="If set, pivot for a single method only.")
    args = ap.parse_args()

    master_path = Path(args.master)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_rows = _read_jsonl(master_path)

    # Flatten per (run_id, method) rows
    flat: List[Dict[str, Any]] = []
    trace_flat: List[Dict[str, Any]] = []

    for rec in master_rows:
        run_id = rec.get("run_id")
        params = _flatten_params(rec.get("params") or {})
        bench = rec.get("benchmark")
        runner = rec.get("runner")
        artifacts = rec.get("artifacts") or {}
        run_tag = artifacts.get("run_tag")

        # summary_by_method is list of dicts (method -> metrics)
        for sm in rec.get("summary_by_method") or []:
            row = {
                "run_id": run_id,
                "benchmark": bench,
                "runner": runner,
                "run_tag": run_tag,
                "method": sm.get("method"),
                **params,
                **{k: v for k, v in sm.items() if k != "method"},
            }
            flat.append(row)

        if args.include_traces:
            # Try to find traces under sweeps/<run_id>/traces/*.jsonl if present
            # We infer sweep run directory as sibling of master file: out_dir/.. (user can pass absolute paths)
            # We search both relative to master and relative to CWD.
            cand_dirs = []
            # If master path is .../sweeps/sweep_summary_*.jsonl then run dirs are .../sweeps/<run_id>/
            cand_dirs.append(master_path.parent / run_id / "traces")
            cand_dirs.append(Path("sweeps") / run_id / "traces")
            trace_dir = None
            for d in cand_dirs:
                if d.exists():
                    trace_dir = d
                    break
            if trace_dir:
                for tf in trace_dir.glob("trace_*.jsonl"):
                    steps = 0
                    json_fail = 0
                    json_rec = 0
                    tool_calls = 0
                    searches = 0
                    opens = 0
                    finish_seen = False

                    for line in tf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if obj.get("type") == "llm_attempts":
                            steps += 1
                            attempts = obj.get("attempts") or []
                            if attempts and not attempts[0].get("parsed_ok"):
                                json_fail += 1
                                # if any later attempt parsed ok -> recovery
                                if any(a.get("parsed_ok") for a in attempts[1:]):
                                    json_rec += 1
                        elif obj.get("type") == "tool":
                            tool_calls += 1
                            if obj.get("tool") == "search":
                                searches += 1
                            elif obj.get("tool") == "open_page":
                                opens += 1
                        elif obj.get("type") == "finish":
                            finish_seen = True

                    # parse filename tokens: trace_<runTag>_<method>_<taskId>.jsonl
                    name = tf.stem
                    parts = name.split("_")
                    # best-effort: method is second last token(s), task last; but method names have '-' so OK.
                    # we stored safe string as runTag_method_taskId, so:
                    # trace_{runTag}_{method}_{taskId}
                    method = None
                    task_id = None
                    if len(parts) >= 4:
                        # remove leading "trace"
                        method = parts[-2]
                        task_id = parts[-1]

                    trace_flat.append({
                        "run_id": run_id,
                        "benchmark": bench,
                        "runner": runner,
                        "run_tag": run_tag,
                        "trace_file": str(tf),
                        "method_guess": method,
                        "task_id_guess": task_id,
                        "steps_logged": steps,
                        "json_fail_steps": json_fail,
                        "json_recovered_steps": json_rec,
                        "tool_calls_logged": tool_calls,
                        "search_calls_logged": searches,
                        "open_page_calls_logged": opens,
                        "finish_seen": finish_seen,
                        **params,
                    })

    # Write CSVs
    _write_csv(out_dir / "sweep_summary_by_method.csv", flat)
    if args.include_traces:
        _write_csv(out_dir / "trace_summary.csv", trace_flat)

    # Optional pivot heatmap + csv
    if args.pivot_x and args.pivot_y:
        xs, ys, grid = _pivot_mean(flat, args.pivot_x, args.pivot_y, args.pivot_value, method=args.pivot_method)

        # Write pivot CSV
        pivot_rows = []
        for i, xv in enumerate(xs):
            r = {args.pivot_x: xv}
            for j, yv in enumerate(ys):
                r[yv] = grid[i][j]
            pivot_rows.append(r)
        _write_csv(out_dir / f"pivot_{args.pivot_value}_{args.pivot_x}_by_{args.pivot_y}.csv", pivot_rows)

        # Write simple heatmap image
        title = f"{args.pivot_value} pivot ({args.pivot_method or 'ALL methods'})"
        _write_pgm_heatmap(out_dir / f"heatmap_{args.pivot_value}_{args.pivot_x}_by_{args.pivot_y}.pgm", xs, ys, grid, title=title)

    print("Wrote:", out_dir / "sweep_summary_by_method.csv")
    if args.include_traces:
        print("Wrote:", out_dir / "trace_summary.csv")

if __name__ == "__main__":
    main()
