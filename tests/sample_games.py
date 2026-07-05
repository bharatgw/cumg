import numpy as np


def dominant_action_game():
    """Two-scenario game where action 0 strictly dominates for both players."""

    A = [
        np.array([[4.0, 4.0], [0.0, 0.0]]),
        np.array([[5.0, 5.0], [1.0, 1.0]]),
    ]
    B = [
        np.array([[4.0, 0.0], [4.0, 0.0]]),
        np.array([[3.0, 1.0], [3.0, 1.0]]),
    ]
    p = np.array([0.4, 0.6])
    return A, B, p


def matching_pennies_game():
    """Single-scenario zero-sum game with the unique mixed equilibrium (0.5, 0.5)."""

    A = [np.array([[1.0, -1.0], [-1.0, 1.0]])]
    B = [-A[0]]
    p = np.array([1.0])
    return A, B, p

