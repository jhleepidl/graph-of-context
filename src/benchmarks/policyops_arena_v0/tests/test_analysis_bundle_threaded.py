import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from policyops.analysis import analyze_bundle


def _write_compare(
    run_dir: Path,
    name: str,
    *,
    budget: int,
    run_id: str,
    timestamp: str,
    full_history_trunc: float,
    full_history_critical: float,
    goc_critical: float,
    full_history_e3_judge: float = 0.4,
    goc_e3_judge: float = 0.6,
) -> None:
    compare_dir = run_dir / "runs" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "judge": "symbolic_packed",
        "scenario_params": {
            "scenario_mode": "threaded_v1_3_fu",
            "thread_context_budget_chars": budget,
            "thread_open_policy": "shared_topk",
        },
        "method_reports": {
            "full_history": {
                "metrics": {
                    "e3_context_truncated_rate": full_history_trunc,
                    "e3_packed_all_critical_rate": full_history_critical,
                    "e3_packed_any_critical_rate": 1.0,
                    "e3_packed_critical_count_mean": 1.0,
                    "e3_judge_accuracy_packed": full_history_e3_judge,
                    "e3_context_clause_count_mean": 3.0,
                    "e3_context_chars_used_mean": 900.0,
                    "e3_context_token_est_mean": 225.0,
                    "cost_per_correct_token_est": 300.0,
                    "acc_per_1k_tokens": 2.0,
                    "e3_decoy_clause_count_mean": 0.5,
                    "goc_unfolded_clause_count_mean": None,
                    "goc_unfolded_critical_clause_count_mean": None,
                    "goc_folded_episode_count_mean": None,
                    "judge_accuracy": full_history_e3_judge,
                },
                "records": [],
            },
            "goc": {
                "metrics": {
                    "e3_context_truncated_rate": full_history_trunc,
                    "e3_packed_all_critical_rate": goc_critical,
                    "e3_packed_any_critical_rate": 1.0,
                    "e3_packed_critical_count_mean": 2.0,
                    "e3_judge_accuracy_packed": goc_e3_judge,
                    "e3_context_clause_count_mean": 2.0,
                    "e3_context_chars_used_mean": 850.0,
                    "e3_context_token_est_mean": 210.0,
                    "cost_per_correct_token_est": 220.0,
                    "acc_per_1k_tokens": 3.0,
                    "e3_decoy_clause_count_mean": 0.1,
                    "goc_unfolded_clause_count_mean": 2.0,
                    "goc_unfolded_critical_clause_count_mean": 2.0,
                    "goc_folded_episode_count_mean": 2.0,
                    "judge_accuracy": goc_e3_judge,
                },
                "records": [],
            },
        },
    }
    path = compare_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_bundle_threaded_no_legacy(tmp_path: Path) -> None:
    run_dir = tmp_path / "threaded_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_compare(
        run_dir,
        "r1",
        budget=1300,
        run_id="20260206_010000",
        timestamp="2026-02-06T01:00:00Z",
        full_history_trunc=0.55,
        full_history_critical=0.12,
        goc_critical=0.20,
    )
    analyze_bundle(run_dir)
    output_dir = run_dir / "analysis_bundle"
    legacy = [
        "policy_method_matrix.csv",
        "policy_method_matrix.md",
        "goc_deltas.csv",
        "goc_deltas.md",
        "difficulty_sanity.csv",
        "difficulty_sanity.md",
        "narrative_summary.md",
    ]
    for name in legacy:
        assert not (output_dir / name).exists()


def test_results_context_budget_sweep_dedup(tmp_path: Path) -> None:
    run_dir = tmp_path / "threaded_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_compare(
        run_dir,
        "r1",
        budget=1300,
        run_id="20260206_010000",
        timestamp="2026-02-06T01:00:00Z",
        full_history_trunc=0.10,
        full_history_critical=0.10,
        goc_critical=0.20,
    )
    _write_compare(
        run_dir,
        "r2",
        budget=1300,
        run_id="20260206_020000",
        timestamp="2026-02-06T02:00:00Z",
        full_history_trunc=0.40,
        full_history_critical=0.12,
        goc_critical=0.25,
    )
    analyze_bundle(run_dir)
    sweep_csv = run_dir / "analysis_bundle" / "results_context_budget_sweep.csv"
    assert sweep_csv.exists()
    lines = sweep_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert "e3_judge_acc_packed" in header
    assert "e3_packed_all_critical_rate" in header
    assert "e3_packed_any_critical_rate" in header
    assert "e3_packed_critical_count_mean" in header
    assert "e3_decoy_clause_count_mean" in header
    assert "e3_context_chars_used_mean" in header
    assert "e3_context_clause_count_mean" in header
    assert "goc_unfolded_clause_count_mean" in header
    assert "goc_unfolded_critical_clause_count_mean" in header
    assert "goc_folded_episode_count_mean" in header
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    idx_budget = header.index("budget")
    idx_method = header.index("method")
    idx_trunc = header.index("e3_context_truncated_rate")
    pairs = {(row[idx_budget], row[idx_method]) for row in rows}
    assert len(pairs) == len(rows)
    target = [
        row
        for row in rows
        if row[idx_budget] == "1300" and row[idx_method] == "full_history"
    ]
    assert target
    assert float(target[0][idx_trunc]) == 0.40
    sweep_md = run_dir / "analysis_bundle" / "results_context_budget_sweep.md"
    header_line = sweep_md.read_text(encoding="utf-8").splitlines()[2]
    assert "..." not in header_line


def test_calibration_recommendation_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "threaded_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_compare(
        run_dir,
        "r1",
        budget=1300,
        run_id="20260206_010000",
        timestamp="2026-02-06T01:00:00Z",
        full_history_trunc=0.55,
        full_history_critical=0.55,
        goc_critical=0.65,
        full_history_e3_judge=0.4,
        goc_e3_judge=0.8,
    )
    _write_compare(
        run_dir,
        "r2",
        budget=1400,
        run_id="20260206_020000",
        timestamp="2026-02-06T02:00:00Z",
        full_history_trunc=0.35,
        full_history_critical=0.40,
        goc_critical=0.60,
        full_history_e3_judge=0.5,
        goc_e3_judge=0.7,
    )
    analyze_bundle(run_dir)
    calib_md = run_dir / "analysis_bundle" / "calibration_recommendation.md"
    assert calib_md.exists()
    text = calib_md.read_text(encoding="utf-8")
    assert "status: PASS" in text
    assert "recommended_budget: 1400" in text


def test_efficiency_thresholds_generated(tmp_path: Path) -> None:
    run_dir = tmp_path / "threaded_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_compare(
        run_dir,
        "r1",
        budget=1300,
        run_id="20260206_010000",
        timestamp="2026-02-06T01:00:00Z",
        full_history_trunc=0.55,
        full_history_critical=0.40,
        goc_critical=0.65,
    )
    _write_compare(
        run_dir,
        "r2",
        budget=1400,
        run_id="20260206_020000",
        timestamp="2026-02-06T02:00:00Z",
        full_history_trunc=0.35,
        full_history_critical=0.80,
        goc_critical=0.95,
    )
    analyze_bundle(run_dir)
    csv_path = run_dir / "analysis_bundle" / "efficiency_thresholds.csv"
    md_path = run_dir / "analysis_bundle" / "efficiency_thresholds.md"
    assert csv_path.exists()
    assert md_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    for col in [
        "threshold",
        "method",
        "min_budget",
        "e3_context_token_est_mean",
        "cost_per_correct_token_est",
        "acc_per_1k_tokens",
    ]:
        assert col in header
    idx_thr = header.index("threshold")
    idx_method = header.index("method")
    idx_budget = header.index("min_budget")
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    target = [
        row
        for row in rows
        if row[idx_thr] == "0.5" and row[idx_method] == "full_history"
    ]
    assert target
    assert target[0][idx_budget] == "1400"
