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
        smoothing_tau=0.1,
        max_iter=0,
        batch_size=1,
        step_size=0.05,
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
