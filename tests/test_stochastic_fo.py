import numpy as np
import pytest
from sample_games import dominant_action_game, matching_pennies_game

pytest.importorskip("jax")

from cumg import StochasticFOConfig, solve_cvar_stochastic_fo, solve_msd_stochastic_fo
from cumg.stochastic_fo import varphi_tau


def assert_mixed_strategy(strategy):
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0.0)
    assert np.sum(strategy) == pytest.approx(1.0, abs=1e-10)


def test_varphi_tau_is_stable_and_approximates_positive_part():
    values = varphi_tau(np.array([-3.0, 0.0, 2.0, 1000.0]), tau=1e-3)

    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(0.0, abs=1e-10)
    assert values[1] == pytest.approx(1e-3 * np.log(2.0), abs=1e-12)
    assert values[2] == pytest.approx(2.0, abs=1e-10)
    assert values[3] == pytest.approx(1000.0, abs=1e-10)


def test_stochastic_fo_rejects_invalid_parameters():
    A, B, p = matching_pennies_game()

    with pytest.raises(ValueError, match="gamma"):
        solve_msd_stochastic_fo(A, B, p, gamma=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        solve_cvar_stochastic_fo(A, B, p, gamma=0.0, alpha=0.0)
    with pytest.raises(ValueError, match="kappa"):
        solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=StochasticFOConfig(kappa=0.0))
    with pytest.raises(ValueError, match="tau"):
        solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=StochasticFOConfig(tau=0.0))
    with pytest.raises(ValueError, match="batch_size"):
        solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=StochasticFOConfig(batch_size=0))
    with pytest.raises(ValueError, match="requires certify_every"):
        solve_msd_stochastic_fo(
            A,
            B,
            p,
            gamma=0.0,
            config=StochasticFOConfig(
                stagnation_window=10,
                stagnation_rtol=0.01,
            ),
        )


def test_msd_stochastic_fo_finds_matching_pennies_equilibrium_with_gamma_zero():
    A, B, p = matching_pennies_game()
    config = StochasticFOConfig(
        kappa=0.05,
        tau=0.1,
        max_iter=20,
        batch_size=None,
        step_size=0.1,
        step_decay=0.0,
        record_every=5,
        regret_tolerance=1e-8,
    )

    result = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)

    assert result.success
    assert result.certificate["eta"] == pytest.approx(0.0, abs=1e-8)
    assert_mixed_strategy(result.x)
    assert_mixed_strategy(result.y)
    np.testing.assert_allclose(result.x, np.array([0.5, 0.5]), atol=1e-8)
    np.testing.assert_allclose(result.y, np.array([0.5, 0.5]), atol=1e-8)
    assert result.history
    assert result.best_iterate["objective"] == pytest.approx(0.0, abs=1e-12)


def test_cvar_stochastic_fo_finds_matching_pennies_equilibrium_with_gamma_zero():
    A, B, p = matching_pennies_game()
    config = StochasticFOConfig(
        kappa=0.05,
        tau=0.1,
        max_iter=20,
        batch_size=None,
        step_size=0.1,
        step_decay=0.0,
        record_every=5,
        regret_tolerance=1e-8,
    )

    result = solve_cvar_stochastic_fo(A, B, p, gamma=0.0, alpha=1.0, config=config)

    assert result.success
    assert result.theta is not None
    assert result.certificate["eta"] == pytest.approx(0.0, abs=1e-8)
    assert_mixed_strategy(result.x)
    assert_mixed_strategy(result.y)
    np.testing.assert_allclose(result.x, np.array([0.5, 0.5]), atol=1e-8)
    np.testing.assert_allclose(result.y, np.array([0.5, 0.5]), atol=1e-8)
    assert result.history
    assert result.best_iterate["objective"] == pytest.approx(0.0, abs=1e-12)


def test_stochastic_fo_full_batch_is_deterministic_for_fixed_seed():
    A, B, p = matching_pennies_game()
    config = StochasticFOConfig(
        kappa=0.05,
        tau=0.1,
        max_iter=5,
        batch_size=None,
        step_size=0.1,
        step_decay=0.0,
        seed=13,
        record_every=1,
        regret_tolerance=1e-8,
    )

    first = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)
    second = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)

    np.testing.assert_allclose(first.x, second.x, atol=1e-12)
    np.testing.assert_allclose(first.y, second.y, atol=1e-12)
    assert first.residual_norm == pytest.approx(second.residual_norm, abs=1e-12)
    assert [row["objective"] for row in first.history] == pytest.approx(
        [row["objective"] for row in second.history],
        abs=1e-12,
    )


def test_certify_every_records_best_certificate_and_stops_when_regret_is_small():
    A, B, p = matching_pennies_game()
    config = StochasticFOConfig(
        kappa=0.05,
        tau=0.1,
        max_iter=20,
        batch_size=None,
        step_size=0.1,
        step_decay=0.0,
        record_every=1,
        certify_every=1,
        regret_tolerance=1e-8,
    )

    result = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)

    assert result.success
    assert result.iterations == 0
    assert result.termination_reason == "regret_tolerance"
    assert result.best_certificate is not None
    assert result.best_certificate["eta"] == pytest.approx(0.0, abs=1e-8)
    assert result.best_certificate["certificate"]["eta"] == pytest.approx(0.0, abs=1e-8)
    assert result.best_certificate["iteration"] == 0
    assert result.history[0]["iteration"] == 0
    assert result.history[0]["eta"] == pytest.approx(0.0, abs=1e-8)
    assert result.history[0]["regret1"] == pytest.approx(0.0, abs=1e-8)
    assert result.history[0]["regret2"] == pytest.approx(0.0, abs=1e-8)


def test_stochastic_fo_stops_on_certificate_stagnation():
    A, B, p = dominant_action_game()
    config = StochasticFOConfig(
        kappa=0.05,
        tau=0.1,
        max_iter=10,
        batch_size=None,
        step_size=0.01,
        step_decay=0.0,
        record_every=1,
        certify_every=1,
        regret_tolerance=0.0,
        stagnation_window=1,
        stagnation_rtol=1.0,
    )

    result = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)

    assert not result.success
    assert result.iterations == 1
    assert result.termination_reason == "stagnation"
    assert result.best_certificate is not None
