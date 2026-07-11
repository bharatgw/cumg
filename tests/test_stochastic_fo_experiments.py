import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "compare_stochastic_fo_msd.py"
SPEC = importlib.util.spec_from_file_location("compare_stochastic_fo_msd", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_stochastic_fo_msd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_stochastic_fo_msd)


def test_stochastic_fo_scalability_experiment_records_pairwise_metrics():
    args = argparse.Namespace(
        K=[2],
        n=[2],
        reps=1,
        seed_base=123,
        gamma=0.0,
        entropy_kappa=0.05,
        entropy_kappa_grid=None,
        smoothing_tau=0.1,
        smoothing_tau_grid=None,
        max_iter=0,
        batch_size=1,
        step_size=0.05,
        step_size_grid=None,
        step_decay=0.0,
        logit_bound=20.0,
        gradient_clip_norm=None,
        record_every=0,
        certify_every=0,
        regret_tolerance=1e-8,
        methods=["full_batch", "minibatch"],
        low=0.0,
        high=1.0,
        csv=None,
        history_csv=None,
        quiet=True,
    )

    row = compare_stochastic_fo_msd.run_instance(args, K=2, n=2, seed=0)

    assert row["full_batch_has_profile"]
    assert row["minibatch_has_profile"]
    assert np.isfinite(row["full_batch_eta"])
    assert np.isfinite(row["minibatch_eta"])
    assert np.isfinite(row["time_ratio_minibatch_over_full_batch"])
    assert np.isfinite(row["x_l1_minibatch_minus_full_batch"])
    assert np.isfinite(row["y_l1_minibatch_minus_full_batch"])


def test_stochastic_fo_tuning_grid_and_history_rows():
    args = argparse.Namespace(
        K=[2],
        n=[2],
        reps=1,
        seed_base=123,
        gamma=0.0,
        entropy_kappa=0.05,
        entropy_kappa_grid=[0.05, 0.01],
        smoothing_tau=0.1,
        smoothing_tau_grid=[0.1, 0.02],
        max_iter=0,
        batch_size=1,
        step_size=0.05,
        step_size_grid=[0.05, 0.2],
        step_decay=0.0,
        logit_bound=20.0,
        gradient_clip_norm=None,
        record_every=1,
        certify_every=1,
        regret_tolerance=1e-8,
        methods=["full_batch"],
        low=0.0,
        high=1.0,
        csv=None,
        history_csv=None,
        quiet=True,
    )

    configs = list(compare_stochastic_fo_msd.iter_tuning_configs(args))

    assert len(configs) == 8
    history_rows = []
    row = compare_stochastic_fo_msd.run_instance(
        configs[0], K=2, n=2, seed=0, history_callback=history_rows.append
    )
    assert row["full_batch_best_certificate_iteration"] == 0
    assert len(history_rows) == 1
    assert history_rows[0]["iteration"] == 0
    assert np.isfinite(history_rows[0]["eta"])
