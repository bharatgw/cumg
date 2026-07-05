import numpy as np
import pytest

from cumgSolver.results import SolverResult, SupportSearchConfig
from cumgSolver.small_support import (
    _restricted_data,
    _search_with_solver,
    candidate_support_pairs,
    expand_support_probs,
    full_msd_regret,
    restricted_profile_gap_msd,
    sample_supports,
)


def test_sample_supports_caps_support_size_at_number_of_actions():
    supports = sample_supports(3, 5, np.random.default_rng(0))

    assert supports == [(0, 1, 2)]


def test_sample_supports_uses_random_sampling_for_large_combinatorial_space():
    supports = sample_supports(20, 10, np.random.default_rng(0), max_exact=7)

    assert len(supports) == 7
    assert len(set(supports)) == 7
    assert all(len(support) == 10 for support in supports)


def test_candidate_support_pairs_respects_candidate_count_and_support_sizes():
    pairs = list(candidate_support_pairs(5, 6, 7, 2, 3, 4, np.random.default_rng(0)))

    assert len(pairs) == 4
    for (s1, s2), scenarios in pairs:
        assert len(s1) == 2
        assert len(s2) == 2
        assert len(scenarios) == 2
        assert len(scenarios[0]) == 3
        assert len(scenarios[1]) == 3


def test_expand_support_probs_places_probabilities_at_support_indices():
    full = expand_support_probs(np.array([0.1, 0.9]), (2, 0), 4)

    np.testing.assert_allclose(full, np.array([0.9, 0.0, 0.1, 0.0]))


def test_restricted_data_slices_actions_and_scenarios():
    A = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    B = -A
    p = np.array([0.2, 0.3, 0.5])

    A_sub, B_sub, p_sub, s1, s2, scenarios = _restricted_data(A, B, p, ((1, 3), (0, 4)), (2, 0))

    assert s1 == (1, 3)
    assert s2 == (0, 4)
    assert scenarios == (2, 0)
    np.testing.assert_allclose(p_sub, np.array([5.0 / 7.0, 2.0 / 7.0]))
    np.testing.assert_allclose(A_sub[0], A[2][np.ix_((1, 3), (0, 4))])
    np.testing.assert_allclose(B_sub[1], B[0][np.ix_((1, 3), (0, 4))])


def test_restricted_data_uses_union_of_player_scenario_supports():
    A = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    B = -A
    p = np.array([0.2, 0.3, 0.5])

    _, _, p_sub, _, _, scenarios = _restricted_data(A, B, p, ((1, 3), (0, 4)), ((2,), (0, 2)))

    assert scenarios == (0, 2)
    np.testing.assert_allclose(p_sub, np.array([2.0 / 7.0, 5.0 / 7.0]))


def test_full_msd_regret_is_zero_for_dominant_action_profile():
    A = [np.array([[4.0, 4.0], [0.0, 0.0]])]
    B = [np.array([[4.0, 0.0], [4.0, 0.0]])]

    cert = full_msd_regret(
        A,
        B,
        np.array([1.0]),
        gamma=0.5,
        x=np.array([1.0, 0.0]),
        y=np.array([1.0, 0.0]),
    )

    assert cert["eta"] == pytest.approx(0.0, abs=1e-8)


def test_restricted_profile_gap_msd_returns_screen_certificate():
    A = [np.array([[4.0, 4.0], [0.0, 0.0]])]
    B = [np.array([[4.0, 0.0], [4.0, 0.0]])]

    screen = restricted_profile_gap_msd(
        A,
        B,
        np.array([1.0]),
        gamma=0.5,
        S=((0, 1), (0, 1)),
        T=((0,), (0,)),
        n_starts=1,
        seed=0,
        maxiter=200,
    )

    assert screen["success"]
    assert screen["eta"] <= 1e-6
    np.testing.assert_allclose(screen["x"], np.array([1.0, 0.0]), atol=1e-5)


def test_search_with_solver_expands_restricted_solution_to_full_profile():
    A = np.ones((2, 3, 4))
    B = -A
    p = np.array([0.25, 0.75])
    config = SupportSearchConfig(kappa=2, tau=1, max_candidates=1, seed=2, solver="fake")

    def fake_solver(A_sub, B_sub, p, **kwargs):
        assert len(A_sub) == 1
        assert A_sub[0].shape == (2, 2)
        return SolverResult(
            x=np.array([0.25, 0.75]),
            y=np.array([0.4, 0.6]),
            alpha1=1.0,
            alpha2=-1.0,
            model="fake",
            solver=kwargs["solver"],
        )

    result = _search_with_solver(A, B, p, fake_solver, {}, config)

    assert result.success
    assert result.candidate_index == 1
    assert result.x.shape == (3,)
    assert result.y.shape == (4,)
    assert result.x.sum() == 1.0
    assert result.y.sum() == 1.0
    assert result.solver_result.solver == "fake"


def test_search_with_solver_reports_best_error_when_all_candidates_fail():
    A = np.ones((1, 2, 2))
    B = -A
    config = SupportSearchConfig(kappa=1, tau=1, max_candidates=2, seed=0)

    def failing_solver(*args, **kwargs):
        raise RuntimeError("solver unavailable")

    result = _search_with_solver(A, B, np.array([1.0]), failing_solver, {}, config)

    assert not result.success
    assert result.best_error == "solver unavailable"
