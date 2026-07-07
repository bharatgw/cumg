import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "compare_scalability_approaches.py"
SPEC = importlib.util.spec_from_file_location("compare_scalability_approaches", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_scalability_approaches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_scalability_approaches)


def base_args(methods):
    return argparse.Namespace(
        risk=["msd", "cvar"],
        K=[2],
        n=[2],
        reps=1,
        rep_start=0,
        rep_stop=None,
        seed_base=123,
        gamma=0.0,
        alpha=1.0,
        epsilon=1e-2,
        epsilon_scr=None,
        max_candidates=1,
        n_screen_starts=1,
        n_support_starts=1,
        screen_maxiter=20,
        support_maxiter=20,
        max_iter=0,
        batch_size=1,
        entropy_kappa=0.05,
        smoothing_tau=0.1,
        step_size=0.05,
        step_decay=0.0,
        logit_bound=20.0,
        gradient_clip_norm=None,
        record_every=0,
        certify_every=0,
        methods=methods,
        solver="pathampl",
        fallback_solver=None,
        low=0.0,
        high=1.0,
        csv=None,
        quiet=True,
    )


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_scalability_driver_records_non_solver_methods(risk):
    methods = ["screened_dual", "action_dual", "stochastic_full_batch", "stochastic_minibatch"]
    row = compare_scalability_approaches.run_instance(base_args(methods), risk=risk, K=2, n=2, seed=0)

    assert row["risk"] == risk
    for method in methods:
        assert f"{method}_success" in row
        assert f"{method}_eta" in row
        assert f"{method}_regret1" in row
        assert f"{method}_regret2" in row
        assert f"{method}_has_profile" in row
        assert f"{method}_error" in row
        assert np.isfinite(row[f"{method}_time_s"])
    assert np.isfinite(row["eta_diff_stochastic_minibatch_minus_full_batch"])
    assert np.isfinite(row["time_ratio_stochastic_minibatch_over_full_batch"])


def test_scalability_driver_records_mcp_errors(monkeypatch):
    args = base_args(["mcp"])

    def fail_mcp(*args, **kwargs):
        raise RuntimeError("no solver available")

    monkeypatch.setattr(compare_scalability_approaches, "solve_msd_mcp", fail_mcp)

    row = compare_scalability_approaches.run_instance(args, risk="msd", K=2, n=2, seed=0)

    assert not row["mcp_success"]
    assert not row["mcp_has_profile"]
    assert row["mcp_error"] == "no solver available"
    assert np.isfinite(row["mcp_time_s"])


def test_scalability_driver_run_experiment_returns_both_risks():
    methods = ["stochastic_full_batch", "stochastic_minibatch"]
    args = base_args(methods)

    rows = compare_scalability_approaches.run_experiment(args)

    assert {row["risk"] for row in rows} == {"msd", "cvar"}
    assert len(rows) == 2


def test_scalability_driver_default_rep_range_preserves_seed_formula(monkeypatch):
    args = base_args(["stochastic_full_batch"])
    args.risk = ["msd"]
    args.K = [2]
    args.n = [3]
    args.reps = 3

    def fake_run_instance(args, risk, K, n, seed):
        return {"risk": risk, "K": K, "n": n, "seed": seed}

    monkeypatch.setattr(compare_scalability_approaches, "run_instance", fake_run_instance)

    rows = compare_scalability_approaches.run_experiment(args)

    assert [row["seed"] for row in rows] == [20_423, 20_424, 20_425]


def test_scalability_driver_rep_shard_uses_absolute_rep_indices(monkeypatch):
    args = base_args(["stochastic_full_batch"])
    args.risk = ["cvar"]
    args.K = [2]
    args.n = [3]
    args.reps = 5
    args.rep_start = 2
    args.rep_stop = 4

    def fake_run_instance(args, risk, K, n, seed):
        return {"risk": risk, "K": K, "n": n, "seed": seed}

    monkeypatch.setattr(compare_scalability_approaches, "run_instance", fake_run_instance)

    rows = compare_scalability_approaches.run_experiment(args)

    assert [row["seed"] for row in rows] == [1_020_425, 1_020_426]


def test_scalability_driver_streams_valid_csv_rows(tmp_path, monkeypatch):
    args = base_args(["stochastic_full_batch"])
    args.risk = ["msd"]
    args.K = [2]
    args.n = [3]
    args.reps = 2
    csv_path = tmp_path / "shard.csv"

    def fake_run_instance(args, risk, K, n, seed):
        return {"risk": risk, "K": K, "n": n, "seed": seed, "status": "done"}

    monkeypatch.setattr(compare_scalability_approaches, "run_instance", fake_run_instance)

    with compare_scalability_approaches.StreamingCsvWriter(csv_path) as writer:
        rows = compare_scalability_approaches.run_experiment(args, row_callback=writer.write_row)

    with csv_path.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert [row["seed"] for row in csv_rows] == ["20423", "20424"]
    assert {row["status"] for row in csv_rows} == {"done"}


def test_scalability_driver_rejects_invalid_rep_shard():
    args = base_args(["stochastic_full_batch"])
    args.reps = 3
    args.rep_start = 2
    args.rep_stop = 4

    with pytest.raises(ValueError, match="rep-stop"):
        compare_scalability_approaches.run_experiment(args)
