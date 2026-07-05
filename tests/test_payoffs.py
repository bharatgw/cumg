import numpy as np
import pytest

from cumgSolver.cvar import cvar_profile_values, cvar_tail_weights, cvar_value_from_state_payoffs
from cumgSolver.msd import msd_profile_values, msd_value_from_state_payoffs


def test_msd_value_is_mean_when_gamma_is_zero():
    payoffs = np.array([-1.0, 2.0, 4.0])
    p = np.array([0.2, 0.3, 0.5])

    assert msd_value_from_state_payoffs(payoffs, p, gamma=0.0) == pytest.approx(float(p @ payoffs))


def test_msd_profile_values_returns_state_payoffs_and_risk_values():
    A = [
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[0.0, 3.0], [1.0, 0.0]]),
    ]
    B = [-A[0], -A[1]]
    p = np.array([0.25, 0.75])
    x = np.array([0.5, 0.5])
    y = np.array([0.25, 0.75])

    values = msd_profile_values(A, B, p, gamma=0.4, x=x, y=y)

    np.testing.assert_allclose(values["u1_states"], np.array([0.875, 1.25]))
    np.testing.assert_allclose(values["u2_states"], -values["u1_states"])
    assert values["rho1"] == pytest.approx(msd_value_from_state_payoffs(values["u1_states"], p, 0.4))


def test_cvar_tail_weights_zero_when_gamma_is_zero():
    weights = cvar_tail_weights(np.array([3.0, 1.0]), np.array([0.25, 0.75]), gamma=0.0, alpha=0.5)

    np.testing.assert_allclose(weights, np.array([0.0, 0.0]))


def test_cvar_tail_weights_rejects_invalid_alpha():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        cvar_tail_weights(np.array([1.0, 2.0]), np.array([0.5, 0.5]), gamma=0.5, alpha=0.0)


def test_cvar_profile_values_matches_state_utility_function():
    A = [
        np.array([[2.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [3.0, 0.0]]),
    ]
    B = [A[0] + 1.0, A[1] - 1.0]
    p = np.array([0.4, 0.6])
    x = np.array([0.75, 0.25])
    y = np.array([0.2, 0.8])

    values = cvar_profile_values(A, B, p, gamma=0.3, alpha=0.5, x=x, y=y)

    assert values["rho1"] == pytest.approx(cvar_value_from_state_payoffs(values["u1_states"], p, 0.3, 0.5))
    assert values["rho2"] == pytest.approx(cvar_value_from_state_payoffs(values["u2_states"], p, 0.3, 0.5))
