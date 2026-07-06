"""Input validation and payoff helpers."""

from __future__ import annotations

import numpy as np


def normalize_probabilities(p, K: int | None = None) -> np.ndarray:
    """Return a finite probability vector normalized to sum to one."""

    probs = np.asarray(p, dtype=float)
    if K is not None and probs.shape != (K,):
        raise ValueError(f"p must have shape ({K},); got {probs.shape}.")
    if probs.ndim != 1:
        raise ValueError("p must be a one-dimensional probability vector.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("p must contain finite probabilities.")
    if np.any(probs < 0):
        raise ValueError("p must contain nonnegative probabilities.")
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p must have positive total mass and all probabilities must be finite.")
    return probs / total


def _validate_mixed_strategy(strategy, n: int, name: str, atol: float = 1e-6) -> np.ndarray:
    strategy = np.asarray(strategy, dtype=float)
    if strategy.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {strategy.shape}.")
    if not np.all(np.isfinite(strategy)):
        raise ValueError(f"{name} must contain finite probabilities.")
    if np.any(strategy < -atol):
        raise ValueError(f"{name} must contain nonnegative probabilities.")
    total = float(strategy.sum())
    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1.")
    strategy = np.maximum(strategy, 0.0)
    return strategy / strategy.sum()


def normalize_game_inputs(
    A_list: list[np.ndarray] | np.ndarray,
    B_list: list[np.ndarray] | np.ndarray,
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate payoff scenarios and return arrays with shape ``(K, n1, n2)``."""

    A = np.asarray(A_list, dtype=float)
    B = np.asarray(B_list, dtype=float)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("A_list and B_list must contain finite payoff values.")
    if A.ndim == 2:
        A = A[None, :, :]
    if B.ndim == 2:
        B = B[None, :, :]
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A_list and B_list must be arrays/lists of 2D payoff matrices.")
    if A.shape != B.shape:
        raise ValueError(f"A_list and B_list must have the same shape; got {A.shape} and {B.shape}.")
    if A.shape[0] == 0 or A.shape[1] == 0 or A.shape[2] == 0:
        raise ValueError("At least one scenario and one action per player are required.")

    K = A.shape[0]
    if p is None:
        probs = np.ones(K, dtype=float) / K
    else:
        probs = normalize_probabilities(p, K)
    return A, B, probs


def as_matrix_lists(A: np.ndarray, B: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert normalized payoff arrays to lists of scenario matrices."""

    return [A[k] for k in range(A.shape[0])], [B[k] for k in range(B.shape[0])]
