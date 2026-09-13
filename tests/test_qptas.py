from itertools import islice, product
from math import comb, prod
from types import SimpleNamespace

import numpy as np
import pytest
from sample_games import dominant_action_game, matching_pennies_game

import cumg.qptas as qptas
from cumg import (
    enumerate_kappa_uniform_profiles,
    enumerate_kappa_uniform_strategies,
    sample_kappa_uniform_profiles,
    solve_cvar_qptas,
    solve_msd_qptas,
)
from cumg.small_support import full_cvar_regret, full_msd_regret


@pytest.mark.parametrize("n,kappa", [(1, 4), (2, 1), (2, 3), (3, 2), (4, 3)])
def test_strategy_enumeration_matches_all_integer_compositions(n, kappa):
    strategies = list(enumerate_kappa_uniform_strategies(n, kappa))
    counts = {tuple(np.rint(kappa * strategy).astype(int)) for strategy in strategies}
    expected = {c for c in product(range(kappa + 1), repeat=n) if sum(c) == kappa}

    assert len(strategies) == comb(kappa + n - 1, n - 1)
    assert counts == expected
    for strategy in strategies:
        assert np.all(strategy >= 0)
        np.testing.assert_allclose(strategy.sum(), 1.0)
        np.testing.assert_allclose(kappa * strategy, np.rint(kappa * strategy))


def test_kappa_is_denominator_and_can_exceed_number_of_actions():
    np.testing.assert_allclose(
        list(enumerate_kappa_uniform_strategies(2, 3)),
        [[1, 0], [2 / 3, 1 / 3], [1 / 3, 2 / 3], [0, 1]],
    )


def test_joint_enumeration_is_complete_deterministic_and_owns_arrays():
    sizes = (2, 3, 1)
    profiles = list(enumerate_kappa_uniform_profiles(sizes, 2))
    keys = {tuple(tuple(strategy) for strategy in profile) for profile in profiles}
    expected = prod(comb(2 + n - 1, n - 1) for n in sizes)

    assert len(profiles) == len(keys) == expected
    repeated = list(enumerate_kappa_uniform_profiles(sizes, 2))
    for profile, other in zip(profiles, repeated, strict=True):
        for strategy, other_strategy in zip(profile, other, strict=True):
            np.testing.assert_array_equal(strategy, other_strategy)
    profiles[0][0][0] = -1
    assert profiles[1][0][0] == 1


def test_large_grid_is_streamed_without_materializing_strategy_pools():
    profiles = list(islice(enumerate_kappa_uniform_profiles((2, 2), 10**9), 2))

    assert len(profiles) == 2
    np.testing.assert_array_equal(profiles[0][0], [1, 0])
    np.testing.assert_allclose(profiles[1][1], [1 - 1e-9, 1e-9], rtol=0, atol=1e-16)


@pytest.mark.parametrize("n,kappa", [(1, 5), (2, 1), (2, 7), (3, 3), (4, 4)])
def test_strategy_rank_decoding_matches_complete_enumeration(n, kappa):
    for rank, strategy in enumerate(enumerate_kappa_uniform_strategies(n, kappa)):
        decoded = qptas._kappa_uniform_strategy_from_rank(n, kappa, rank)
        np.testing.assert_array_equal(decoded, strategy)


@pytest.mark.parametrize("sizes,kappa", [((1,), 3), ((2, 3, 1), 2), ((3, 2), 3)])
def test_sampling_entire_grid_matches_exhaustive_set_without_duplicates(sizes, kappa):
    expected = {tuple(tuple(x) for x in profile) for profile in enumerate_kappa_uniform_profiles(sizes, kappa)}
    sampled = list(sample_kappa_uniform_profiles(sizes, kappa, len(expected) + 10, seed=17))
    actual = {tuple(tuple(x) for x in profile) for profile in sampled}

    assert len(sampled) == len(actual) == len(expected)
    assert actual == expected


def test_sampling_is_uniform_over_all_ordered_pairs_on_small_grid(monkeypatch):
    # Three strategies, two draws: enumerate all 3*2 equiprobable RNG paths.
    # Each distinct ordered pair must occur once. This detects multinomial
    # weighting or replacement errors without a statistical/flaky tolerance.
    outcomes = []
    for choices in product(range(3), range(2)):
        draws = iter(choices)
        monkeypatch.setattr(
            qptas, "Random", lambda _seed, draws=draws: SimpleNamespace(randrange=lambda _n: next(draws))
        )
        profiles = list(sample_kappa_uniform_profiles((2,), 2, 2, seed=0))
        outcomes.append(tuple(tuple(profile[0]) for profile in profiles))
    strategies = [tuple(x) for x in enumerate_kappa_uniform_strategies(2, 2)]
    expected = {(x, y) for x in strategies for y in strategies if x != y}

    assert len(outcomes) == 6
    assert set(outcomes) == expected


def test_sampling_is_reproducible_and_larger_budget_extends_prefix():
    first = list(sample_kappa_uniform_profiles((5, 4), 3, 10, seed=123))
    longer = list(sample_kappa_uniform_profiles((5, 4), 3, 20, seed=np.int64(123)))
    other = list(sample_kappa_uniform_profiles((5, 4), 3, 10, seed=124))
    keys = [tuple(tuple(x) for x in p) for p in first]

    assert keys == [tuple(tuple(x) for x in p) for p in longer[:10]]
    assert keys != [tuple(tuple(x) for x in p) for p in other]
    assert len(set(keys)) == 10
    first[0][0][0] = -1
    assert all(np.all(x >= 0) for p in first[1:] for x in p)


def test_sampling_does_not_enumerate_or_require_64_bit_grid_size(monkeypatch):
    def fail_if_enumerated(*args, **kwargs):
        pytest.fail("Random sampling must not traverse the exhaustive grid.")

    monkeypatch.setattr(qptas, "enumerate_kappa_uniform_profiles", fail_if_enumerated)
    monkeypatch.setattr(qptas, "enumerate_kappa_uniform_strategies", fail_if_enumerated)
    assert comb(50 + 23 - 1, 49) ** 2 > 2**64
    profiles = list(sample_kappa_uniform_profiles((50, 50), 23, 20, seed=1))

    assert len({tuple(tuple(x) for x in p) for p in profiles}) == 20
    for profile in profiles:
        for strategy in profile:
            assert strategy.shape == (50,)
            assert np.all(strategy >= 0)
            np.testing.assert_allclose(strategy.sum(), 1)
            np.testing.assert_allclose(strategy * 23, np.rint(strategy * 23))


def test_sampling_large_denominator_does_not_scan_all_counts():
    profiles = list(sample_kappa_uniform_profiles((2, 2), 10**9, 3, seed=0))

    assert len(profiles) == 3
    for profile in profiles:
        for strategy in profile:
            np.testing.assert_allclose(strategy.sum(), 1)
            np.testing.assert_allclose(strategy * 10**9, np.rint(strategy * 10**9), atol=1e-6)


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_sample_exhaustion_does_not_claim_entire_grid_failed(solver):
    A, B, p = matching_pennies_game()
    result = solver([A, B], p, gamma=0.5, kappa=1, epsilon=0.01, max_candidates=2, seed=3)

    assert not result.success
    assert result.termination_reason == "sample_exhausted"
    assert result.profiles_checked == result.max_candidates == 2
    assert result.total_profiles == 4
    assert result.seed == 3
    assert result.certificate is None


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_randomized_search_can_cover_whole_grid(solver):
    A, B, p = matching_pennies_game()
    result = solver([A, B], p, gamma=0.5, kappa=1, epsilon=0.01, max_candidates=20, seed=4)

    assert not result.success
    assert result.termination_reason == "grid_exhausted"
    assert result.profiles_checked == result.total_profiles == 4
    assert result.best_response_solves == 6


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_randomized_search_finds_and_certifies_mixed_equilibrium(solver):
    A, B, p = matching_pennies_game()
    result = solver([A, B], p, gamma=0.5, kappa=2, epsilon=1e-8, max_candidates=9, seed=0)

    assert result.success
    assert result.termination_reason == "epsilon_reached"
    assert result.profiles_checked <= 9
    np.testing.assert_allclose(result.strategies, [[0.5, 0.5], [0.5, 0.5]])
    cert = full_msd_regret(A, B, p, 0.5, *result.strategies)
    assert result.certificate["eta"] == pytest.approx(cert["eta"], abs=1e-8)
    assert result.max_candidates == 9
    assert result.seed == 0


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_sampling_still_stops_immediately_on_certificate(solver):
    A = np.zeros((1, 2, 2))
    result = solver([A, A], kappa=2, epsilon=0.0, max_candidates=9, seed=5)

    assert result.success
    assert result.profiles_checked == 1
    assert result.best_response_solves == 2


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.inf])
def test_invalid_sampling_budget_is_rejected(bad):
    with pytest.raises(ValueError, match="n_samples"):
        list(sample_kappa_uniform_profiles((2,), 2, bad))
    with pytest.raises(ValueError, match="max_candidates"):
        solve_msd_qptas([np.zeros((1, 2))], kappa=2, max_candidates=bad)


@pytest.mark.parametrize("seed", [1.5, True, "123"])
def test_invalid_sampling_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        list(sample_kappa_uniform_profiles((2,), 2, 1, seed=seed))
    with pytest.raises(ValueError, match="seed"):
        solve_cvar_qptas([np.zeros((1, 2))], kappa=2, max_candidates=1, seed=seed)


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_matching_pennies_finds_mixed_equilibrium(solver):
    A, B, p = matching_pennies_game()
    result = solver([A, B], p, gamma=0.0, kappa=2, epsilon=1e-8)

    assert result.success
    assert result.termination_reason == "epsilon_reached"
    assert result.profiles_checked == 5
    assert result.total_profiles == 9
    np.testing.assert_allclose(result.strategies, [[0.5, 0.5], [0.5, 0.5]])
    assert result.certificate["eta"] == pytest.approx(0.0, abs=1e-8)
    assert len(result.certificate["best_responses"]) == 2


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_grid_exhaustion_and_early_rejection(solver):
    A, B, p = matching_pennies_game()
    result = solver([A, B], p, gamma=0.5, kappa=1, epsilon=0.01)

    assert not result.success
    assert result.termination_reason == "grid_exhausted"
    assert result.profiles_checked == result.total_profiles == 4
    # Two profiles are rejected after player 1, saving player 2's LP.
    assert result.best_response_solves == 6
    assert result.strategies is None
    assert result.certificate is None


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_best_response_checks_mixed_deviations_outside_candidate_grid(solver):
    # Both pure actions have the same risk-adjusted value, but mixing hedges
    # scenario risk. Comparing only pure deviations would incorrectly accept.
    payoffs = [np.array([[1.0, -1.0], [-1.0, 1.0]])]
    failure = solver(payoffs, [0.5, 0.5], gamma=1.0, kappa=1, epsilon=0.1)
    success = solver(payoffs, [0.5, 0.5], gamma=1.0, kappa=2, epsilon=1e-8)

    assert not failure.success
    assert failure.profiles_checked == 2
    assert success.success
    np.testing.assert_allclose(success.strategies[0], [0.5, 0.5])
    assert success.certificate["eta"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_all_samples_and_nonuniform_zero_probability_weights_are_used(solver):
    payoffs = [np.array([[1.0, 0.0], [-9.0, 0.0], [1000.0, -1000.0]])]
    # The first state alone favors action 0; the second makes action 1 optimal.
    # The last state has zero weight, and scaling probability weights is benign.
    result = solver(payoffs, [9.0, 1.0, 0.0], gamma=1.0, kappa=1, epsilon=1e-8)

    assert result.success
    assert result.profiles_checked == 2
    np.testing.assert_array_equal(result.strategies[0], [0, 1])
    assert result.certificate["eta"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("solver", [solve_msd_qptas, solve_cvar_qptas])
def test_three_player_rectangular_game_with_mixed_equilibrium(solver):
    # Players 0 and 1 play matching pennies, scaled by player 2's action.
    # Player 1 also has a dominated third action. Player 2 hedges two scenarios.
    A = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    B = np.array([[-1.0, 1.0, -4.0], [1.0, -1.0, -4.0]])
    scale = np.array([1.0, 2.0])
    U0 = np.broadcast_to(A[None, :, :, None] * scale, (2, 2, 3, 2)).copy()
    U1 = np.broadcast_to(B[None, :, :, None] * scale, (2, 2, 3, 2)).copy()
    U2 = np.broadcast_to(np.array([[1.0, -1.0], [-1.0, 1.0]])[:, None, None, :], (2, 2, 3, 2)).copy()
    result = solver([U0, U1, U2], gamma=[0.0, 0.5, 1.0], kappa=2, epsilon=1e-8)

    assert result.success
    assert result.total_profiles == 54
    for strategy, expected in zip(result.strategies, ([0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5]), strict=True):
        np.testing.assert_allclose(strategy, expected)
    np.testing.assert_allclose(result.certificate["regrets"], [0, 0, 0], atol=1e-8)


@pytest.mark.parametrize("risk", ["MSD", "CVaR"])
def test_returned_certificate_matches_existing_two_player_regret(risk):
    rng = np.random.default_rng(7)
    A, B = rng.random((2, 4, 2, 3))
    p = np.array([0.1, 0.2, 0.3, 0.4])
    # A coarse grid and positive tolerance exercise acceptance with nonzero regret.
    if risk == "MSD":
        result = solve_msd_qptas([A, B], p, gamma=0.7, kappa=3, epsilon=0.1)
        assert result.success
        cert = full_msd_regret(A, B, p, 0.7, *result.strategies)
    else:
        result = solve_cvar_qptas([A, B], p, gamma=0.7, alpha=0.4, kappa=3, epsilon=0.1)
        assert result.success
        cert = full_cvar_regret(A, B, p, 0.7, 0.4, *result.strategies)

    assert result.certificate["eta"] == pytest.approx(cert["eta"], abs=1e-8)
    np.testing.assert_allclose(result.certificate["regrets"], [cert["regret1"], cert["regret2"]], atol=1e-8)
    assert cert["eta"] <= 0.1


def test_player_specific_cvar_alpha_and_gamma():
    # Verify each player's independently specified risk parameters against
    # its own known fixed-profile payoff calculation.
    A, B, p = dominant_action_game()
    result = solve_cvar_qptas([A, B], p, gamma=[0.0, 1.0], alpha=[1.0, 0.4], kappa=1, epsilon=0.0)

    assert result.success
    np.testing.assert_allclose(result.certificate["current_values"], [4.6, 3.0])


@pytest.mark.parametrize(
    "solver,lp_name", [(solve_msd_qptas, "_maximize_msd_on_simplex"), (solve_cvar_qptas, "_maximize_cvar_on_simplex")]
)
def test_lp_failure_does_not_masquerade_as_grid_exhaustion(monkeypatch, solver, lp_name):
    def failed_lp(*args, **kwargs):
        raise RuntimeError("LP failed")

    monkeypatch.setattr(qptas, lp_name, failed_lp)
    with pytest.raises(RuntimeError, match="LP failed"):
        solver([np.ones((1, 1))], kappa=1)


@pytest.mark.parametrize(
    "solver,lp_name", [(solve_msd_qptas, "_maximize_msd_on_simplex"), (solve_cvar_qptas, "_maximize_cvar_on_simplex")]
)
def test_nonfinite_best_response_is_never_accepted(monkeypatch, solver, lp_name):
    monkeypatch.setattr(qptas, lp_name, lambda *args, **kwargs: {"value": np.nan})
    with pytest.raises(RuntimeError, match="non-finite"):
        solver([np.ones((1, 1))], kappa=1)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.inf])
def test_invalid_grid_parameters_are_rejected(bad):
    with pytest.raises(ValueError, match="kappa"):
        solve_msd_qptas([np.ones((1, 2))], kappa=bad)
    with pytest.raises(ValueError, match="n"):
        list(enumerate_kappa_uniform_strategies(bad, 2))


@pytest.mark.parametrize("epsilon", [-1, np.nan, np.inf])
def test_invalid_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        solve_msd_qptas([np.ones((1, 2))], kappa=1, epsilon=epsilon)


@pytest.mark.parametrize(
    "payoffs,p,match",
    [
        ([], None, "player"),
        ([np.ones((2, 2))] * 2, None, "shape"),
        ([np.ones((1, 2, 2)), np.ones((1, 2, 3))], None, "same shape"),
        ([np.ones((0, 2))], None, "nonempty"),
        ([np.array([[np.nan]])], None, "finite"),
        ([np.ones((2, 2))], [1.0], "shape"),
        ([np.ones((2, 2))], [0.0, 0.0], "positive total mass"),
        ([np.ones((2, 2))], [1.0, -0.5], "nonnegative"),
    ],
)
def test_invalid_game_inputs_are_rejected(payoffs, p, match):
    with pytest.raises(ValueError, match=match):
        solve_msd_qptas(payoffs, p, kappa=1)


@pytest.mark.parametrize(
    "solver,kwargs",
    [
        (solve_msd_qptas, {"gamma": -0.1}),
        (solve_msd_qptas, {"gamma": np.nan}),
        (solve_msd_qptas, {"gamma": [0.1, 0.2]}),
        (solve_cvar_qptas, {"gamma": 1.1}),
        (solve_cvar_qptas, {"alpha": 0.0}),
        (solve_cvar_qptas, {"alpha": 1.1}),
        (solve_cvar_qptas, {"alpha": np.nan}),
    ],
)
def test_invalid_risk_parameters_are_rejected(solver, kwargs):
    with pytest.raises(ValueError, match="gamma|alpha"):
        solver([np.ones((1, 2))], kappa=1, **kwargs)
