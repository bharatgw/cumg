import numpy as np
import pytest

from cumgSolver.validation import as_matrix_lists, normalize_game_inputs


def test_normalize_game_inputs_accepts_single_matrix_and_default_probabilities():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = -A

    A_out, B_out, p = normalize_game_inputs(A, B)

    assert A_out.shape == (1, 2, 2)
    assert B_out.shape == (1, 2, 2)
    np.testing.assert_allclose(p, np.array([1.0]))


def test_normalize_game_inputs_rejects_probability_length_mismatch():
    A = [np.eye(2), np.ones((2, 2))]
    B = [np.eye(2), np.ones((2, 2))]

    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        normalize_game_inputs(A, B, np.array([1.0, 0.0, 0.0]))


def test_normalize_game_inputs_rejects_negative_probabilities():
    A = [np.eye(2), np.ones((2, 2))]
    B = [np.eye(2), np.ones((2, 2))]

    with pytest.raises(ValueError, match="nonnegative"):
        normalize_game_inputs(A, B, np.array([1.0, -0.5]))


def test_normalize_game_inputs_rejects_zero_probability_mass():
    A = [np.eye(2), np.ones((2, 2))]
    B = [np.eye(2), np.ones((2, 2))]

    with pytest.raises(ValueError, match="positive total mass"):
        normalize_game_inputs(A, B, np.array([0.0, 0.0]))


def test_normalize_game_inputs_rejects_non_matrix_inputs():
    with pytest.raises(ValueError, match="2D payoff matrices"):
        normalize_game_inputs(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_as_matrix_lists_preserves_scenarios():
    A = np.arange(8, dtype=float).reshape(2, 2, 2)
    B = -A

    A_list, B_list = as_matrix_lists(A, B)

    assert len(A_list) == 2
    assert len(B_list) == 2
    np.testing.assert_allclose(A_list[1], A[1])
    np.testing.assert_allclose(B_list[0], B[0])

