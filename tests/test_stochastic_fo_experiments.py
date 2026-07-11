import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "compare_stochastic_fo.py"
SPEC = importlib.util.spec_from_file_location("compare_stochastic_fo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_stochastic_fo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_stochastic_fo)


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_stochastic_fo_scalability_experiment_records_pairwise_metrics(risk):
    args = argparse.Namespace(
        K=[2],
        n=[2],
        reps=1,
        seed_base=123,
        gamma=0.0,
        alpha=1.0,
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

    row = compare_stochastic_fo.run_instance(args, risk, K=2, n=2, seed=0)

    assert row["risk"] == risk
    if risk == "cvar":
        assert row["alpha"] == 1.0
        assert np.isfinite(row["full_batch_theta1"])
        assert np.isfinite(row["full_batch_theta2"])
    else:
        assert np.isnan(row["alpha"])
        assert np.isnan(row["full_batch_theta1"])
        assert np.isnan(row["full_batch_theta2"])
    assert row["full_batch_has_profile"]
    assert row["minibatch_has_profile"]
    assert np.isfinite(row["full_batch_eta"])
    assert np.isfinite(row["minibatch_eta"])
    assert np.isfinite(row["time_ratio_minibatch_over_full_batch"])
    assert np.isfinite(row["x_l1_minibatch_minus_full_batch"])
    assert np.isfinite(row["y_l1_minibatch_minus_full_batch"])


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_stochastic_fo_tuning_grid_and_history_rows(risk):
    args = argparse.Namespace(
        K=[2],
        n=[2],
        reps=1,
        seed_base=123,
        gamma=0.0,
        alpha=1.0,
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

    configs = list(compare_stochastic_fo.iter_tuning_configs(args))

    assert len(configs) == 8
    history_rows = []
    row = compare_stochastic_fo.run_instance(configs[0], risk, K=2, n=2, seed=0, history_callback=history_rows.append)
    assert row["full_batch_best_certificate_iteration"] == 0
    assert len(history_rows) == 1
    assert history_rows[0]["risk"] == risk
    if risk == "cvar":
        assert history_rows[0]["alpha"] == 1.0
        assert np.isfinite(history_rows[0]["theta1"])
        assert np.isfinite(history_rows[0]["theta2"])
    else:
        assert np.isnan(history_rows[0]["alpha"])
        assert np.isnan(history_rows[0]["theta1"])
        assert np.isnan(history_rows[0]["theta2"])
    assert history_rows[0]["iteration"] == 0
    assert np.isfinite(history_rows[0]["eta"])


def test_stochastic_fo_experiment_does_not_swallow_invalid_cvar_parameters():
    args = argparse.Namespace(
        gamma=0.0,
        alpha=0.0,
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
        methods=["full_batch"],
        low=0.0,
        high=1.0,
    )

    with pytest.raises(ValueError, match="alpha must be in"):
        compare_stochastic_fo.run_instance(args, "cvar", K=2, n=2, seed=0)


def test_stochastic_fo_continuation_warm_starts_and_keeps_best_stage(monkeypatch):
    args = argparse.Namespace(
        gamma=0.0,
        alpha=1.0,
        entropy_kappa=0.1,
        entropy_kappa_grid=None,
        smoothing_tau=0.02,
        smoothing_tau_grid=None,
        continuation_kappa=[0.1, 0.03],
        continuation_tau=[0.02, 0.01],
        continuation_max_iter=[10, 20],
        max_iter=100,
        batch_size=1,
        step_size=2.0,
        step_size_grid=None,
        step_decay=0.5,
        logit_bound=20.0,
        gradient_clip_norm=None,
        record_every=10,
        certify_every=10,
        regret_tolerance=1e-3,
        methods=["full_batch"],
        low=0.0,
        high=1.0,
    )
    configs = []
    profiles = [np.array([0.75, 0.25]), np.array([0.6, 0.4])]
    etas = [0.2, 0.1]

    def fake_solver(A, B, p, gamma, config):
        del A, B, p, gamma
        stage = len(configs)
        configs.append(config)
        profile = profiles[stage]
        eta = etas[stage]
        certificate = {"eta": eta, "regret1": eta, "regret2": eta / 2}
        history = [
            {
                "iteration": 0,
                "residual_norm": 1.0,
                "objective": 0.5,
                **certificate,
            },
            {
                "iteration": config.max_iter,
                "residual_norm": 0.5,
                "objective": 0.125,
                **certificate,
            },
        ]
        return SimpleNamespace(
            success=False,
            x=profile,
            y=profile[::-1],
            theta=None,
            certificate=certificate,
            residual_norm=0.5,
            objective=0.125,
            iterations=config.max_iter,
            solve_time_s=1.0,
            best_iterate={"residual_norm": 0.5, "objective": 0.125},
            best_certificate={"eta": eta, "iteration": config.max_iter},
            history=history,
        )

    monkeypatch.setattr(compare_stochastic_fo, "solve_msd_stochastic_fo", fake_solver)
    history_rows = []

    row = compare_stochastic_fo.run_instance(
        args,
        "msd",
        K=2,
        n=2,
        seed=0,
        history_callback=history_rows.append,
    )

    assert len(configs) == 2
    assert configs[0].kappa == 0.1
    assert configs[1].kappa == 0.03
    assert configs[1].tau == 0.01
    assert configs[1].max_iter == 20
    np.testing.assert_allclose(configs[1].x0, profiles[0])
    np.testing.assert_allclose(configs[1].y0, profiles[0][::-1])
    assert row["continuation"]
    assert row["continuation_total_max_iter"] == 30
    assert row["full_batch_eta"] == 0.1
    assert row["full_batch_iterations"] == 30
    assert row["full_batch_selected_stage"] == 1
    assert row["full_batch_selected_kappa"] == 0.03
    assert row["full_batch_selected_tau"] == 0.01
    assert row["full_batch_stages_completed"] == 2
    assert row["full_batch_best_certificate_iteration"] == 30
    assert [history["iteration"] for history in history_rows] == [0, 10, 10, 30]
    assert [history["continuation_stage"] for history in history_rows] == [0, 0, 1, 1]
    assert [history["selected_stage"] for history in history_rows] == [False, False, True, True]


def test_stochastic_fo_continuation_rejects_mismatched_schedules():
    args = argparse.Namespace(
        entropy_kappa=0.1,
        entropy_kappa_grid=None,
        smoothing_tau=0.02,
        smoothing_tau_grid=None,
        max_iter=100,
        continuation_kappa=[0.1, 0.03],
        continuation_tau=[0.02],
        continuation_max_iter=None,
    )

    with pytest.raises(ValueError, match="same length"):
        compare_stochastic_fo._continuation_stages(args)
