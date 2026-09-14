from dataclasses import replace

import numpy as np
import pytest
from sample_games import dominant_action_game, matching_pennies_game

pytest.importorskip("jax")

from cumg import StochasticFOConfig, solve_cvar_stochastic_fo, solve_msd_stochastic_fo
from cumg.small_support import full_cvar_regret, full_msd_regret
from cumg.stochastic_fo import StochasticFOResult, _solve_with_restarts, varphi_tau


def assert_mixed_strategy(strategy):
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0.0)
    assert np.sum(strategy) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("solver", [solve_msd_stochastic_fo, solve_cvar_stochastic_fo])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_compiled_updates_match_eager_trajectory(solver, batch_size):
    A, B = np.random.default_rng(7).random((2, 4, 3, 2))
    config = StochasticFOConfig(
        kappa=0.003,
        tau=0.002,
        max_iter=20,
        batch_size=batch_size,
        step_size=0.01,
        seed=13,
        x0=np.array([0.2, 0.3, 0.5]),
        y0=np.array([0.7, 0.3]),
        n_random_starts=0,
        record_every=5,
        certify_every=5,
        regret_tolerance=0,
    )
    eager = solver(A, B, [0.1, 0.2, 0.3, 0.4], gamma=0.5, config=config)
    compiled = solver(A, B, [0.1, 0.2, 0.3, 0.4], gamma=0.5, config=replace(config, jit_updates=True))
    assert eager.iterations == compiled.iterations == 20
    assert eager.termination_reason == compiled.termination_reason
    for first, second in zip(eager.history, compiled.history, strict=True):
        for key in ("x", "y", "eta", "objective", "residual_norm"):
            np.testing.assert_allclose(first[key], second[key], rtol=1e-8, atol=1e-10)
        if "theta" in first:
            np.testing.assert_allclose(first["theta"], second["theta"], rtol=1e-8, atol=1e-10)


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
    assert len(result.start_summaries) == 1  # Success at uniform must skip all random starts.


@pytest.mark.parametrize("n_random_starts", [0, 4])
def test_stochastic_fo_stops_on_certificate_stagnation(n_random_starts):
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
        n_random_starts=n_random_starts,
    )

    result = solve_msd_stochastic_fo(A, B, p, gamma=0.0, config=config)

    assert not result.success
    assert result.iterations == 1 + n_random_starts
    assert result.termination_reason == "stagnation"
    assert result.best_certificate is not None
    assert all(start["termination_reason"] == "stagnation" for start in result.start_summaries)


@pytest.mark.parametrize("solver", [solve_msd_stochastic_fo, solve_cvar_stochastic_fo])
@pytest.mark.parametrize("batch_size", [None, 1])
def test_failed_fo_tries_four_reproducible_random_profiles(solver, batch_size):
    A, B, p = dominant_action_game()
    config = StochasticFOConfig(max_iter=0, regret_tolerance=0, seed=13, batch_size=batch_size)
    first = solver(A, B, p, gamma=0.0, config=config)
    repeat = solver(A, B, p, gamma=0.0, config=config)
    short = solver(A, B, p, gamma=0.0, config=replace(config, n_random_starts=2))
    changed = solver(A, B, p, gamma=0.0, config=replace(config, seed=14))

    assert len(first.start_summaries) == 5
    assert len(short.start_summaries) == 3
    assert not first.success
    assert first.certificate["eta"] == min(r["eta"] for r in first.start_summaries)
    assert first.selected_start == int(np.argmin([r["eta"] for r in first.start_summaries]))
    for index, start in enumerate(first.start_summaries):
        for field in ("initial_x", "initial_y"):
            assert_mixed_strategy(np.array(start[field]))
            np.testing.assert_array_equal(start[field], repeat.start_summaries[index][field])
            if index < 3:
                np.testing.assert_array_equal(start[field], short.start_summaries[index][field])
            if index == 0:
                np.testing.assert_allclose(start[field], [0.5, 0.5])
            else:
                assert not np.allclose(start[field], [0.5, 0.5])
        assert start["termination_reason"] == "max_iter"
        assert start["time_s"] > 0
    assert first.start_summaries[1]["initial_x"] != changed.start_summaries[1]["initial_x"]
    np.testing.assert_array_equal(first.x, repeat.x)
    assert first.solve_time_s >= sum(r["time_s"] for r in first.start_summaries)


@pytest.mark.parametrize("solver", [solve_msd_stochastic_fo, solve_cvar_stochastic_fo])
def test_fo_honors_supplied_profile_before_random_restarts(solver):
    A, B, p = matching_pennies_game()
    config = StochasticFOConfig(x0=np.array([0.8, 0.2]), y0=np.array([0.3, 0.7]), max_iter=0, regret_tolerance=10)
    result = solver(A, B, p, gamma=0, config=config)
    assert result.success and len(result.start_summaries) == 1
    np.testing.assert_allclose(result.start_summaries[0]["initial_x"], config.x0)
    np.testing.assert_allclose(result.start_summaries[0]["initial_y"], config.y0)


@pytest.mark.parametrize("solver", [solve_msd_stochastic_fo, solve_cvar_stochastic_fo])
def test_minibatch_restarts_reproduce_updated_profiles_and_certificates(solver):
    A, B, p = dominant_action_game()
    config = StochasticFOConfig(
        max_iter=2,
        batch_size=1,
        n_random_starts=1,
        seed=13,
        regret_tolerance=0,
        certify_every=1,
        record_every=1,
    )
    first = solver(A, B, p, gamma=0.5, config=config)
    repeat = solver(A, B, p, gamma=0.5, config=config)
    assert first.iterations == 4
    assert len(first.start_summaries) == 2
    np.testing.assert_allclose(first.x, repeat.x, atol=1e-12)
    np.testing.assert_allclose(first.y, repeat.y, atol=1e-12)
    assert [r["eta"] for r in first.history] == pytest.approx([r["eta"] for r in repeat.history], abs=1e-12)
    if solver is solve_msd_stochastic_fo:
        certificate = full_msd_regret(A, B, p, 0.5, first.x, first.y)
    else:
        certificate = full_cvar_regret(A, B, p, 0.5, 0.5, first.x, first.y)
    assert first.certificate["eta"] == pytest.approx(certificate["eta"], abs=1e-12)


@pytest.mark.parametrize("etas, expected", [([0.2, 0.05, 0.01, 0.0, 0.0], 3), ([0.2, 0.04, 0.08, 0.1, 0.06], 5)])
def test_restart_selection_uses_regret_and_stops_at_first_threshold(etas, expected):
    configs = []

    def solve_one(config):
        index = len(configs)
        configs.append(config)
        eta = etas[index]
        checkpoint = {"iteration": 1, "eta": eta, "objective": 5 - index}
        return StochasticFOResult(
            success=eta <= 0.01,
            x=np.array([index / 5, 1 - index / 5]),
            y=np.full(2, 0.5),
            model="test",
            certificate={"eta": eta},
            residual_norm=5 - index,
            objective=5 - index,
            iterations=1,
            solve_time_s=0,
            termination_reason="regret_tolerance" if eta <= 0.01 else "stagnation",
            history=[checkpoint],
            best_certificate=checkpoint,
            best_iterate=checkpoint,
            config=config,
        )

    result = _solve_with_restarts(solve_one, StochasticFOConfig(regret_tolerance=0.01, seed=5), 2, 2)
    assert len(configs) == expected
    assert result.iterations == expected
    assert result.selected_start == int(np.argmin(etas[:expected]))
    assert result.certificate["eta"] == min(etas[:expected])
    assert result.success == (expected == 3)
    assert result.best_certificate["iteration"] == result.selected_start + 1
    assert [r["start_index"] for r in result.history] == list(range(expected))
    assert [r["iteration"] for r in result.history] == list(range(1, expected + 1))
    assert all(c.n_random_starts == 0 for c in configs)
    assert len({c.seed for c in configs}) == expected


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_fo_rejects_invalid_random_restart_count(value):
    A, B, p = matching_pennies_game()
    with pytest.raises(ValueError, match="n_random_starts"):
        solve_msd_stochastic_fo(A, B, p, config=StochasticFOConfig(n_random_starts=value))
