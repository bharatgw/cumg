import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "uniform_profile_baseline.py"
SPEC = importlib.util.spec_from_file_location("uniform_profile_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
uniform_profile_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uniform_profile_baseline)


def base_args():
    return argparse.Namespace(
        risk=["msd"],
        K=[2],
        n=[2],
        reps=2,
        rep_start=0,
        rep_stop=None,
        seed_base=123,
        gamma=0.0,
        alpha=1.0,
        epsilon=1e-2,
        low=0.0,
        high=1.0,
        csv=None,
        summary_csv=None,
        no_summary_csv=True,
        quiet=True,
    )


def test_seed_formula_matches_scalability_experiment_convention():
    assert uniform_profile_baseline.experiment_seed("msd", K=2, n=3, rep=0, seed_base=123) == 20_423
    assert uniform_profile_baseline.experiment_seed("cvar", K=2, n=3, rep=2, seed_base=123) == 1_020_425


def test_uniform_profile_baseline_records_certificate_metrics():
    args = base_args()
    row = uniform_profile_baseline.run_instance(args, risk="msd", K=2, n=2, rep=0)

    assert row["profile"] == "uniform"
    assert row["seed"] == 20_323
    assert isinstance(row["uniform_success"], bool)
    assert np.isfinite(row["uniform_eta"])
    assert np.isfinite(row["uniform_regret1"])
    assert np.isfinite(row["uniform_regret2"])
    assert np.isfinite(row["uniform_time_s"])


def test_uniform_profile_baseline_streams_csv_and_summarizes(tmp_path):
    args = base_args()
    csv_path = tmp_path / "uniform.csv"

    with uniform_profile_baseline.StreamingCsvWriter(csv_path) as writer:
        rows = uniform_profile_baseline.run_experiment(args, row_callback=writer.write_row)

    with csv_path.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    summary = uniform_profile_baseline.summarize(rows)

    assert len(rows) == 2
    assert len(csv_rows) == 2
    assert [row["seed"] for row in csv_rows] == ["20323", "20324"]
    assert summary[0]["risk"] == "msd"
    assert summary[0]["K"] == 2
    assert summary[0]["n"] == 2
    assert summary[0]["reps"] == 2
    assert 0.0 <= summary[0]["uniform_success_rate"] <= 1.0
