import numpy as np
import pytest

from cumg.cvar import cvar_tail_weights, cvar_value_from_state_payoffs
from cumg.msd import msd_value_from_state_payoffs
from cumg.mystic_seed import fb
from cumg.small_support import (
    expand_support_probs,
    sample_supports,
    support_sizes,
)
from cumg.validation import normalize_game_inputs


def test_normalize_game_inputs_normalizes_probabilities():
    A = [np.eye(2), np.ones((2, 2))]
    B = [np.eye(2), np.ones((2, 2))]

    A_out, B_out, p = normalize_game_inputs(A, B, np.array([2.0, 1.0]))

    assert A_out.shape == (2, 2, 2)
    assert B_out.shape == (2, 2, 2)
    np.testing.assert_allclose(p, np.array([2.0 / 3.0, 1.0 / 3.0]))


def test_normalize_game_inputs_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        normalize_game_inputs([np.ones((2, 2))], [np.ones((2, 3))], np.array([1.0]))


def test_fb_is_zero_on_complementary_positive_pair():
    np.testing.assert_allclose(fb(np.array([2.0]), np.array([0.0])), np.array([0.0]))
    np.testing.assert_allclose(fb(np.array([0.0]), np.array([3.0])), np.array([0.0]))


def test_support_helpers():
    rng = np.random.default_rng(0)

    supports = sample_supports(4, 2, rng)
    expanded = expand_support_probs(np.array([0.25, 0.75]), supports[0], 4)

    exp = np.zeros(4)
    exp[list(supports[0])] = [0.25, 0.75]

    assert len(supports) == 6
    assert expanded == pytest.approx(exp)
    assert support_sizes(30, 10) == (4, 6)


def test_msd_value_from_state_payoffs():
    payoffs = np.array([1.0, 3.0, 4.0, 7.0, 10.0])
    p = np.array([0.25, 0.30, 0.10, 0.30, 0.05])

    assert msd_value_from_state_payoffs(payoffs, p, gamma=0.5) == pytest.approx(3.57625)


def test_cvar_tail_weights_and_value():
    payoffs = np.array([10.0, 0.0, 5.0])
    p = np.array([1.0, 1.0, 2.0])

    weights = cvar_tail_weights(payoffs, p, gamma=0.5, alpha=0.5)

    assert weights.sum() == pytest.approx(0.5)
    assert weights[1] > 0
    assert cvar_value_from_state_payoffs(payoffs, p, gamma=0.5, alpha=0.5) == pytest.approx(3.75)
