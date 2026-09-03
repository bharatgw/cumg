import numpy as np
import pytest
from sample_games import dominant_action_game, matching_pennies_game

import cumg.small_support as small_support_module
from cumg.cvar import cvar_profile_values, cvar_value_from_state_payoffs
from cumg.msd import msd_profile_values, msd_value_from_state_payoffs
from cumg.results import SupportSearchConfig
from cumg.small_support import (
    full_cvar_regret,
    full_msd_regret,
    restricted_profile_gap_msd,
    small_support_search_cvar,
    small_support_search_msd,
)


def test_msd_profile_gamma_zero_matches_expected_payoff():
    A, B, p = matching_pennies_game()
    x = np.array([0.3, 0.7])
    y = np.array([0.4, 0.6])

    values = msd_profile_values(A, B, p, gamma=0.0, x=x, y=y)

    assert values["rho1"] == pytest.approx(float(x @ A[0] @ y))
    assert values["rho2"] == pytest.approx(float(x @ B[0] @ y))


def test_cvar_profile_gamma_zero_matches_expected_payoff():
    A, B, p = dominant_action_game()
    x = np.array([0.25, 0.75])
    y = np.array([0.6, 0.4])
    A_arr = np.asarray(A)
    B_arr = np.asarray(B)

    values = cvar_profile_values(A, B, p, gamma=0.0, alpha=0.5, x=x, y=y)

    expected1 = float(p @ np.einsum("i,kij,j->k", x, A_arr, y))
    expected2 = float(p @ np.einsum("i,kij,j->k", x, B_arr, y))
    assert values["rho1"] == pytest.approx(expected1)
    assert values["rho2"] == pytest.approx(expected2)


def test_identical_scenarios_do_not_create_msd_risk_penalty():
    payoffs = np.array([3.25, 3.25, 3.25])
    p = np.array([0.2, 0.3, 0.5])

    assert msd_value_from_state_payoffs(payoffs, p, gamma=0.9) == pytest.approx(3.25)


def test_identical_scenarios_do_not_change_cvar_value():
    payoffs = np.array([-1.5, -1.5, -1.5])
    p = np.array([0.2, 0.3, 0.5])

    assert cvar_value_from_state_payoffs(payoffs, p, gamma=0.7, alpha=0.4) == pytest.approx(-1.5)


def test_probability_rescaling_does_not_change_msd_profile_values():
    A, B, _ = dominant_action_game()
    x = np.array([0.4, 0.6])
    y = np.array([0.8, 0.2])

    normalized = msd_profile_values(A, B, np.array([2.0 / 3.0, 1.0 / 3.0]), gamma=0.4, x=x, y=y)
    unnormalized = msd_profile_values(A, B, np.array([2.0, 1.0]), gamma=0.4, x=x, y=y)

    assert unnormalized["rho1"] == pytest.approx(normalized["rho1"])
    assert unnormalized["rho2"] == pytest.approx(normalized["rho2"])


def test_matching_pennies_equilibrium_has_zero_msd_regret():
    A, B, p = matching_pennies_game()

    cert = full_msd_regret(A, B, p, gamma=0.0, x=np.array([0.5, 0.5]), y=np.array([0.5, 0.5]))

    assert cert["eta"] == pytest.approx(0.0, abs=1e-8)
    assert cert["regret1"] >= 0
    assert cert["regret2"] >= 0


def test_matching_pennies_bad_profile_has_positive_msd_regret():
    A, B, p = matching_pennies_game()

    cert = full_msd_regret(A, B, p, gamma=0.0, x=np.array([1.0, 0.0]), y=np.array([1.0, 0.0]))

    assert cert["eta"] == pytest.approx(2.0, abs=1e-6)


def test_matching_pennies_equilibrium_has_zero_cvar_regret_when_gamma_zero():
    A, B, p = matching_pennies_game()

    cert = full_cvar_regret(
        A,
        B,
        p,
        gamma=0.0,
        alpha=1.0,
        x=np.array([0.5, 0.5]),
        y=np.array([0.5, 0.5]),
    )

    assert cert["eta"] == pytest.approx(0.0, abs=1e-8)


def test_full_regret_rejects_invalid_mixed_profiles():
    A, B, p = matching_pennies_game()

    with pytest.raises(ValueError, match="x must contain nonnegative"):
        full_msd_regret(A, B, p, gamma=0.0, x=np.array([2.0, -1.0]), y=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="y must sum to 1"):
        full_cvar_regret(A, B, p, gamma=0.0, alpha=1.0, x=np.array([0.5, 0.5]), y=np.array([0.2, 0.2]))


def test_full_regret_accepts_near_simplex_numerical_profiles():
    A, B, p = matching_pennies_game()

    cert = full_msd_regret(
        A,
        B,
        p,
        gamma=0.0,
        x=np.array([0.5000003, 0.4999998]),
        y=np.array([0.4999997, 0.5000002]),
    )

    assert cert["eta"] == pytest.approx(0.0, abs=1e-6)


def test_small_support_search_rejects_invalid_risk_parameters():
    A, B, p = dominant_action_game()
    config = SupportSearchConfig(max_candidates=1)

    with pytest.raises(ValueError, match="gamma"):
        small_support_search_msd(A, B, p, gamma=-0.1, config=config)
    with pytest.raises(ValueError, match="alpha"):
        small_support_search_cvar(A, B, p, gamma=0.5, alpha=2.0, config=config)


def test_restricted_screen_returns_feasible_simplex_profile():
    A, B, p = dominant_action_game()

    screen = restricted_profile_gap_msd(
        A,
        B,
        p,
        gamma=0.6,
        S=((0, 1), (0, 1)),
        T=((0, 1), (0, 1)),
        n_starts=2,
        seed=0,
        maxiter=300,
    )

    assert screen["success"]
    assert np.isfinite(screen["eta"])
    np.testing.assert_allclose(screen["x"].sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(screen["y"].sum(), 1.0, atol=1e-8)
    assert np.all(screen["x"] >= -1e-8)
    assert np.all(screen["y"] >= -1e-8)


def test_small_support_search_retains_rejected_screen_candidate_index(monkeypatch):
    A, B, p = dominant_action_game()

    def rejected_screen(_A, _B, _p, *, S, T, **_kwargs):
        return {
            "eta": 0.03,
            "violation": 1e-12,
            "success": True,
            "message": "Optimization terminated successfully",
            "x": np.array([0.5, 0.5]),
            "y": np.array([0.5, 0.5]),
            "S": S,
            "T": T,
        }

    monkeypatch.setattr(small_support_module, "restricted_profile_gap_msd", rejected_screen)
    config = SupportSearchConfig(
        epsilon=0.01,
        epsilon_scr=0.01,
        kappa=2,
        tau=2,
        max_candidates=1,
        seed=0,
    )

    result = small_support_module.small_support_search_msd(A, B, p, gamma=0.5, config=config)

    assert not result.success
    assert result.metadata["best_screen"]["eta"] == pytest.approx(0.03)
    assert result.metadata["best_screen"]["candidate_index"] == 1


def test_dominant_action_regret_is_nonnegative_for_mixed_profile():
    A, B, p = dominant_action_game()

    cert = full_msd_regret(A, B, p, gamma=0.6, x=np.array([0.25, 0.75]), y=np.array([0.8, 0.2]))

    assert cert["eta"] >= 0.0
    assert cert["regret1"] >= 0.0
    assert cert["regret2"] >= 0.0
