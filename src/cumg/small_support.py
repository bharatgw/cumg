"""Small-support screening and certified randomized support search."""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

from .cvar import (
    cvar_profile_values,
    cvar_value_from_state_payoffs,
    solve_cvar_mcp,
)
from .msd import msd_profile_values, msd_value_from_state_payoffs, solve_msd_mcp
from .results import SupportSearchConfig, SupportSearchResult
from .validation import normalize_game_inputs, normalize_probabilities


def expand_support_probs(probs, support, n: int) -> np.ndarray:
    """Expand probabilities on a support to a full strategy vector."""

    probs = np.asarray(probs, dtype=float)
    support = tuple(int(i) for i in support)
    if probs.shape != (len(support),):
        raise ValueError(f"probs must have shape ({len(support)},); got {probs.shape}.")
    if any(i < 0 or i >= n for i in support):
        raise ValueError("support indices must be in range.")
    out = np.zeros(n, dtype=float)
    out[np.asarray(support, dtype=int)] = probs
    return out


def sample_supports(
    n: int, size: int, rng: np.random.Generator, max_exact: int = 200
) -> list[tuple[int, ...]]:
    """Enumerate small support sets exactly, otherwise sample without replacement."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if size <= 0:
        raise ValueError("size must be positive.")
    if max_exact <= 0:
        raise ValueError("max_exact must be positive.")
    size = min(size, n)
    total = comb(n, size)
    if total <= max_exact:
        return list(combinations(range(n), size))
    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_exact:
        seen.add(tuple(sorted(int(i) for i in rng.choice(n, size=size, replace=False))))
    return list(seen)


def candidate_support_pairs(
    n1: int,
    n2: int,
    K: int,
    kappa: int,
    tau: int,
    max_candidates: int,
    rng: np.random.Generator,
):
    """Yield randomized action supports and separate scenario supports for each player."""

    action1 = sample_supports(n1, kappa, rng, max_exact=max_candidates)
    action2 = sample_supports(n2, kappa, rng, max_exact=max_candidates)
    data_supports = sample_supports(K, min(tau, K), rng, max_exact=max_candidates)
    for _ in range(max_candidates):
        S = (
            action1[int(rng.integers(len(action1)))],
            action2[int(rng.integers(len(action2)))],
        )
        T = (
            data_supports[int(rng.integers(len(data_supports)))],
            data_supports[int(rng.integers(len(data_supports)))],
        )
        yield S, T


def candidate_action_support_pairs(
    n1: int,
    n2: int,
    kappa: int,
    max_candidates: int,
    rng: np.random.Generator,
):
    """Yield action-support pairs without scenario supports."""

    action1 = sample_supports(n1, kappa, rng, max_exact=max_candidates)
    action2 = sample_supports(n2, kappa, rng, max_exact=max_candidates)
    total_pairs = len(action1) * len(action2)
    if total_pairs <= max_candidates:
        for s1 in action1:
            for s2 in action2:
                yield s1, s2
        return

    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    while len(seen) < min(max_candidates, total_pairs):
        S = (
            action1[int(rng.integers(len(action1)))],
            action2[int(rng.integers(len(action2)))],
        )
        if S in seen:
            continue
        seen.add(S)
        yield S


def support_sizes(K: int, n: int) -> tuple[int, int]:
    """Default support sizes used in the experiments."""

    kappa = min(n, max(2, int(np.ceil(np.sqrt(n)))))
    tau = min(K, max(5, int(np.ceil(np.sqrt(K)))))
    return kappa, tau


def _scenario_union(T) -> tuple[int, ...]:
    if len(T) == 2 and all(isinstance(part, tuple | list | np.ndarray) for part in T):
        return tuple(sorted(set(int(k) for part in T for k in part)))
    return tuple(int(k) for k in T)


def _restricted_data(A, B, p, S, T):
    p1_support, p2_support = tuple(S[0]), tuple(S[1])
    scenario_support = _scenario_union(T)
    p_sub = normalize_probabilities(p[list(scenario_support)])
    A_sub = [A[k][np.ix_(p1_support, p2_support)] for k in scenario_support]
    B_sub = [B[k][np.ix_(p1_support, p2_support)] for k in scenario_support]
    return A_sub, B_sub, p_sub, p1_support, p2_support, scenario_support


def _validate_mixed_strategy(
    strategy, n: int, name: str, atol: float = 1e-6
) -> np.ndarray:
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


def _strategy_from_lp_result(res, dim: int) -> np.ndarray:
    strategy = np.asarray(res.x[:dim], dtype=float)
    strategy[np.abs(strategy) <= 1e-12] = 0.0
    strategy = np.maximum(strategy, 0.0)
    total = float(strategy.sum())
    if total <= 0.0:
        raise RuntimeError("LP best response returned a zero strategy.")
    return strategy / total


def _lp_best_response_from_result(
    res, dim: int, value_fn: Callable[[np.ndarray], float]
) -> dict[str, Any]:
    if not res.success:
        raise RuntimeError(f"LP best response failed: {res.message}")
    strategy = _strategy_from_lp_result(res, dim)
    return {
        "value": float(value_fn(strategy)),
        "strategy": strategy,
        "success": bool(res.success),
        "message": res.message,
    }


def _maximize_linear_on_simplex(payoff_by_action: np.ndarray) -> dict[str, Any]:
    payoff_by_action = np.asarray(payoff_by_action, dtype=float)
    dim = payoff_by_action.shape[0]
    res = linprog(
        -payoff_by_action,
        A_eq=np.ones((1, dim)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * dim,
        method="highs",
    )
    return _lp_best_response_from_result(
        res, dim, lambda q: float(payoff_by_action @ q)
    )


def _maximize_msd_on_simplex(
    payoff_by_action: np.ndarray, p: np.ndarray, gamma: float
) -> dict[str, Any]:
    payoff_by_action = np.asarray(payoff_by_action, dtype=float)
    p = np.asarray(p, dtype=float)
    n_states, dim = payoff_by_action.shape
    expected_payoff_by_action = p @ payoff_by_action

    if gamma <= 1e-12:
        return _maximize_linear_on_simplex(expected_payoff_by_action)

    # Variables are [strategy, downside]. At optimum downside[k] is
    # min(0, state_payoff[k] - mean_payoff).
    c = np.concatenate([-expected_payoff_by_action, -gamma * p])
    A_ub = np.zeros((n_states, dim + n_states), dtype=float)
    A_ub[:, :dim] = expected_payoff_by_action - payoff_by_action
    A_ub[:, dim:] = np.eye(n_states)
    b_ub = np.zeros(n_states, dtype=float)
    A_eq = np.zeros((1, dim + n_states), dtype=float)
    A_eq[0, :dim] = 1.0
    bounds = [(0.0, 1.0)] * dim + [(None, 0.0)] * n_states

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    return _lp_best_response_from_result(
        res,
        dim,
        lambda q: msd_value_from_state_payoffs(payoff_by_action @ q, p, gamma),
    )


def _maximize_cvar_on_simplex(
    payoff_by_action: np.ndarray,
    p: np.ndarray,
    gamma: float,
    alpha: float,
) -> dict[str, Any]:
    payoff_by_action = np.asarray(payoff_by_action, dtype=float)
    p = np.asarray(p, dtype=float)
    n_states, dim = payoff_by_action.shape
    expected_payoff_by_action = p @ payoff_by_action

    if gamma <= 1e-12:
        return _maximize_linear_on_simplex(expected_payoff_by_action)

    # Variables are [strategy, eta, tail_shortfall]. This is the LP form of
    # gamma * lower-tail CVaR_alpha(state_payoff).
    cap = gamma * p / alpha
    c = np.concatenate(
        [-(1.0 - gamma) * expected_payoff_by_action, np.array([-gamma]), cap]
    )
    A_ub = np.zeros((n_states, dim + 1 + n_states), dtype=float)
    A_ub[:, :dim] = -payoff_by_action
    A_ub[:, dim] = 1.0
    A_ub[:, dim + 1 :] = -np.eye(n_states)
    b_ub = np.zeros(n_states, dtype=float)
    A_eq = np.zeros((1, dim + 1 + n_states), dtype=float)
    A_eq[0, :dim] = 1.0
    bounds = [(0.0, 1.0)] * dim + [(None, None)] + [(0.0, None)] * n_states

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    return _lp_best_response_from_result(
        res,
        dim,
        lambda q: cvar_value_from_state_payoffs(payoff_by_action @ q, p, gamma, alpha),
    )


def _keep_current_strategy_if_better(
    best: dict[str, Any], strategy: np.ndarray, value: float
) -> dict[str, Any]:
    if value > best["value"]:
        return best | {
            "value": float(value),
            "strategy": strategy,
            "message": "Current profile matched LP value safeguard.",
        }
    return best


def full_msd_regret(A_list, B_list, p, gamma: float, x, y) -> dict[str, Any]:
    """Compute max MSD best-response gain for a mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _, n1, n2 = A.shape
    x = _validate_mixed_strategy(x, n1, "x")
    y = _validate_mixed_strategy(y, n2, "y")
    base = msd_profile_values(A, B, p, gamma, x, y)
    p1_payoff_by_action = np.einsum("kij,j->ki", A, y)
    p2_payoff_by_action = np.einsum("i,kij->kj", x, B)

    best1 = _maximize_msd_on_simplex(p1_payoff_by_action, p, gamma)
    best2 = _maximize_msd_on_simplex(p2_payoff_by_action, p, gamma)
    best1 = _keep_current_strategy_if_better(best1, x, float(base["rho1"]))
    best2 = _keep_current_strategy_if_better(best2, y, float(base["rho2"]))
    regret1 = max(0.0, best1["value"] - base["rho1"])
    regret2 = max(0.0, best2["value"] - base["rho2"])
    return {
        "eta": float(max(regret1, regret2)),
        "regret1": float(regret1),
        "regret2": float(regret2),
        "rho1": base["rho1"],
        "rho2": base["rho2"],
        "best_dev1": best1,
        "best_dev2": best2,
    }


def full_cvar_regret(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    x,
    y,
) -> dict[str, Any]:
    """Compute max CVaR best-response gain for a mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _, n1, n2 = A.shape
    x = _validate_mixed_strategy(x, n1, "x")
    y = _validate_mixed_strategy(y, n2, "y")
    base = cvar_profile_values(A, B, p, gamma, alpha, x, y)
    p1_payoff_by_action = np.einsum("kij,j->ki", A, y)
    p2_payoff_by_action = np.einsum("i,kij->kj", x, B)

    best1 = _maximize_cvar_on_simplex(p1_payoff_by_action, p, gamma, alpha)
    best2 = _maximize_cvar_on_simplex(p2_payoff_by_action, p, gamma, alpha)
    best1 = _keep_current_strategy_if_better(best1, x, float(base["rho1"]))
    best2 = _keep_current_strategy_if_better(best2, y, float(base["rho2"]))
    regret1 = max(0.0, best1["value"] - base["rho1"])
    regret2 = max(0.0, best2["value"] - base["rho2"])
    return {
        "eta": float(max(regret1, regret2)),
        "regret1": float(regret1),
        "regret2": float(regret2),
        "rho1": base["rho1"],
        "rho2": base["rho2"],
        "best_dev1": best1,
        "best_dev2": best2,
    }


def _screen_profile(
    A,
    B,
    p,
    value_fn: Callable,
    value_args: tuple[Any, ...],
    S,
    T,
    n_starts: int,
    seed: int,
    maxiter: int,
):
    rng = np.random.default_rng(seed)
    _, n1, n2 = A.shape
    S1, S2 = tuple(S[0]), tuple(S[1])
    T1, T2 = tuple(T[0]), tuple(T[1])
    if not S1 or not S2 or not T1 or not T2:
        raise ValueError("All action and scenario supports must be nonempty.")

    s1, s2, t1, t2 = len(S1), len(S2), len(T1), len(T2)
    S1_idx = np.asarray(S1, dtype=int)
    S2_idx = np.asarray(S2, dtype=int)
    T1_idx = np.asarray(T1, dtype=int)
    T2_idx = np.asarray(T2, dtype=int)
    A_SS = A[:, S1_idx, :][:, :, S2_idx]
    B_SS = B[:, S1_idx, :][:, :, S2_idx]
    A_T1_dev = A[T1_idx, :, :][:, :, S2_idx]
    B_T2_dev = B[T2_idx, :, :][:, S1_idx, :]
    p_T1 = normalize_probabilities(p[T1_idx])
    p_T2 = normalize_probabilities(p[T2_idx])
    offsets = np.cumsum([0, s1, s2, t1, t2])
    payoff_scale = max(1.0, float(max(np.ptp(A), np.ptp(B))))
    eta_bounds = (-10.0 * payoff_scale, 10.0 * payoff_scale)

    def unpack(v):
        x_s = v[offsets[0] : offsets[1]]
        y_s = v[offsets[1] : offsets[2]]
        q1_s = v[offsets[2] : offsets[3]]
        q2_s = v[offsets[3] : offsets[4]]
        eta = v[offsets[4]]
        return x_s, y_s, q1_s, q2_s, eta

    def eq_constraints(v):
        return np.array(
            [
                np.sum(v[offsets[0] : offsets[1]]) - 1.0,
                np.sum(v[offsets[1] : offsets[2]]) - 1.0,
                np.sum(v[offsets[2] : offsets[3]]) - 1.0,
                np.sum(v[offsets[3] : offsets[4]]) - 1.0,
            ]
        )

    def ineq_constraints(v):
        x_s, y_s, q1_s, q2_s, eta = unpack(v)
        u1_states = np.einsum("a,kab,b->k", x_s, A_SS, y_s)
        u2_states = np.einsum("a,kab,b->k", x_s, B_SS, y_s)
        rho1 = value_fn(u1_states, p, *value_args)
        rho2 = value_fn(u2_states, p, *value_args)
        p1_dev_values = np.einsum("k,kib,b->i", q1_s, A_T1_dev, y_s)
        p2_dev_values = np.einsum("k,a,kaj->j", q2_s, x_s, B_T2_dev)
        return np.concatenate([rho1 + eta - p1_dev_values, rho2 + eta - p2_dev_values])

    constraints = [
        {"type": "eq", "fun": eq_constraints},
        {"type": "ineq", "fun": ineq_constraints},
    ]
    bounds = [(0.0, 1.0)] * (s1 + s2 + t1 + t2) + [eta_bounds]
    starts = [
        np.concatenate(
            [np.ones(s1) / s1, np.ones(s2) / s2, p_T1, p_T2, np.array([0.0])]
        )
    ]
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(
            np.concatenate(
                [
                    rng.dirichlet(np.ones(s1)),
                    rng.dirichlet(np.ones(s2)),
                    rng.dirichlet(np.ones(t1)),
                    rng.dirichlet(np.ones(t2)),
                    np.array([0.0]),
                ]
            )
        )

    best = None
    for start in starts:
        res = minimize(
            lambda v: v[offsets[4]],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
        )
        violation = max(
            float(np.max(np.abs(eq_constraints(res.x)))),
            0.0,
            float(-np.min(ineq_constraints(res.x))),
        )
        candidate = {
            "eta": float(res.fun),
            "violation": float(violation),
            "success": bool(res.success),
            "message": res.message,
        }
        if best is None:
            best = candidate | {"result_x": res.x}
        elif (candidate["eta"], candidate["violation"]) < (
            best["eta"],
            best["violation"],
        ):
            best = candidate | {"result_x": res.x}

    assert best is not None
    x_s, y_s, q1, q2, eta = unpack(best["result_x"])
    best.update(
        {
            "x": expand_support_probs(x_s, S1, n1),
            "y": expand_support_probs(y_s, S2, n2),
            "q1": q1,
            "q2": q2,
            "S": (S1, S2),
            "T": (T1, T2),
            "eta": float(eta),
        }
    )
    best["success"] = bool(best["success"] and best["violation"] <= 1e-6)
    return best


def restricted_profile_gap_msd(
    A_list,
    B_list,
    p,
    gamma: float,
    S,
    T,
    n_starts: int = 10,
    seed: int = 0,
    maxiter: int = 1000,
):
    """Solve the notebook-style restricted MSD screening problem."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    return _screen_profile(
        A, B, p, msd_value_from_state_payoffs, (gamma,), S, T, n_starts, seed, maxiter
    )


def restricted_profile_gap_cvar(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    S,
    T,
    n_starts: int = 10,
    seed: int = 0,
    maxiter: int = 1000,
):
    """Solve the notebook-style restricted CVaR screening problem."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    return _screen_profile(
        A,
        B,
        p,
        cvar_value_from_state_payoffs,
        (gamma, alpha),
        S,
        T,
        n_starts,
        seed,
        maxiter,
    )


def _normalize_simplex_candidate(strategy: np.ndarray) -> np.ndarray | None:
    strategy = np.asarray(strategy, dtype=float)
    if not np.all(np.isfinite(strategy)):
        return None
    strategy = np.maximum(strategy, 0.0)
    total = float(strategy.sum())
    if total <= 0.0:
        return None
    return strategy / total


def _support_start_from_profile(
    profile: np.ndarray | None, support: tuple[int, ...]
) -> np.ndarray | None:
    if profile is None:
        return None
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 1 or any(i < 0 or i >= profile.shape[0] for i in support):
        return None
    return _normalize_simplex_candidate(profile[np.asarray(support, dtype=int)])


def _minimize_supported_profile_gap(
    A,
    B,
    p,
    S,
    regret_fn: Callable,
    regret_kwargs: dict[str, Any],
    n_starts: int,
    seed: int,
    x0=None,
    y0=None,
    maxiter: int = 1000,
) -> dict[str, Any]:
    A, B, p = normalize_game_inputs(A, B, p)
    _, n1, n2 = A.shape
    S1, S2 = tuple(S[0]), tuple(S[1])
    if not S1 or not S2:
        raise ValueError("Action supports must be nonempty.")
    if any(i < 0 or i >= n1 for i in S1) or any(j < 0 or j >= n2 for j in S2):
        raise ValueError("Action support indices must be in range.")

    rng = np.random.default_rng(seed)
    s1, s2 = len(S1), len(S2)
    offsets = np.cumsum([0, s1, s2])

    def unpack(v):
        return v[offsets[0] : offsets[1]], v[offsets[1] : offsets[2]]

    def expand(v):
        x_s, y_s = unpack(v)
        x_s = _normalize_simplex_candidate(x_s)
        y_s = _normalize_simplex_candidate(y_s)
        if x_s is None or y_s is None:
            return None, None
        return expand_support_probs(x_s, S1, n1), expand_support_probs(y_s, S2, n2)

    def eq_constraints(v):
        return np.array(
            [
                np.sum(v[offsets[0] : offsets[1]]) - 1.0,
                np.sum(v[offsets[1] : offsets[2]]) - 1.0,
            ]
        )

    def objective(v):
        x, y = expand(v)
        if x is None or y is None:
            return np.inf
        try:
            cert = regret_fn(A, B, p, x, y, **regret_kwargs)
        except (ArithmeticError, RuntimeError):
            return np.inf
        return float(cert["eta"]) if np.isfinite(cert["eta"]) else np.inf

    starts = []
    x_start = _support_start_from_profile(x0, S1)
    y_start = _support_start_from_profile(y0, S2)
    if x_start is not None and y_start is not None:
        starts.append(np.concatenate([x_start, y_start]))
    starts.append(np.concatenate([np.ones(s1) / s1, np.ones(s2) / s2]))
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(
            np.concatenate([rng.dirichlet(np.ones(s1)), rng.dirichlet(np.ones(s2))])
        )

    constraints = [{"type": "eq", "fun": eq_constraints}]
    bounds = [(0.0, 1.0)] * (s1 + s2)
    best = None
    for start in starts:
        res = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
        )
        eta = objective(res.x)
        violation = float(np.max(np.abs(eq_constraints(res.x))))
        candidate = {
            "eta": eta,
            "violation": violation,
            "success": bool(res.success),
            "message": res.message,
            "result_x": res.x,
        }
        if np.isfinite(eta):
            if best is None:
                best = candidate
            elif (eta, violation) < (best["eta"], best["violation"]):
                best = candidate

    if best is None:
        raise RuntimeError("Supported full-regret minimization failed.")

    x, y = expand(best["result_x"])
    cert = regret_fn(A, B, p, x, y, **regret_kwargs)
    best.update(
        {
            "eta": float(cert["eta"]),
            "x": x,
            "y": y,
            "S": (S1, S2),
            "certificate": cert,
        }
    )
    best["success"] = bool(
        best["success"] and best["violation"] <= 1e-6 and np.isfinite(best["eta"])
    )
    return best


def supported_profile_gap_msd(
    A_list,
    B_list,
    p,
    gamma: float,
    S,
    n_starts: int = 20,
    seed: int = 0,
    x0=None,
    y0=None,
    maxiter: int = 1000,
):
    """Minimize the full-game MSD regret over profiles supported on ``S``."""

    return _minimize_supported_profile_gap(
        A_list,
        B_list,
        p,
        S,
        _msd_regret_wrapper,
        {"gamma": gamma},
        n_starts,
        seed,
        x0=x0,
        y0=y0,
        maxiter=maxiter,
    )


def supported_profile_gap_cvar(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    S,
    n_starts: int = 20,
    seed: int = 0,
    x0=None,
    y0=None,
    maxiter: int = 1000,
):
    """Minimize the full-game CVaR regret over profiles supported on ``S``."""

    return _minimize_supported_profile_gap(
        A_list,
        B_list,
        p,
        S,
        _cvar_regret_wrapper,
        {"gamma": gamma, "alpha": alpha},
        n_starts,
        seed,
        x0=x0,
        y0=y0,
        maxiter=maxiter,
    )


def _restricted_action_data(A, B, S):
    _, n1, n2 = A.shape
    S1, S2 = tuple(S[0]), tuple(S[1])
    if not S1 or not S2:
        raise ValueError("Action supports must be nonempty.")
    if any(i < 0 or i >= n1 for i in S1) or any(j < 0 or j >= n2 for j in S2):
        raise ValueError("Action support indices must be in range.")
    S1_idx = np.asarray(S1, dtype=int)
    S2_idx = np.asarray(S2, dtype=int)
    return A[:, S1_idx, :][:, :, S2_idx], B[:, S1_idx, :][:, :, S2_idx], S1, S2


def supported_profile_gap_msd_mcp(
    A_list,
    B_list,
    p,
    gamma: float,
    S,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
):
    """Solve the full-sample restricted MSD MCP on ``S`` and certify full-game regret."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _, n1, n2 = A.shape
    A_sub, B_sub, S1, S2 = _restricted_action_data(A, B, S)
    start = perf_counter()
    solver_result = solve_msd_mcp(
        A_sub,
        B_sub,
        p,
        gamma=gamma,
        solver=solver,
        fallback_solver=fallback_solver,
        solver_options=solver_options,
        tee=tee,
    )
    elapsed = perf_counter() - start
    x_sub = _validate_mixed_strategy(solver_result.x, len(S1), "restricted x", atol=1e-5)
    y_sub = _validate_mixed_strategy(solver_result.y, len(S2), "restricted y", atol=1e-5)
    x = expand_support_probs(x_sub, S1, n1)
    y = expand_support_probs(y_sub, S2, n2)
    cert = full_msd_regret(A, B, p, gamma, x, y)
    return {
        "eta": float(cert["eta"]),
        "x": x,
        "y": y,
        "S": (S1, S2),
        "certificate": cert,
        "solver_result": solver_result,
        "restricted_x": x_sub,
        "restricted_y": y_sub,
        "success": bool(np.isfinite(cert["eta"])),
        "time_s": elapsed,
        "mcp_time_s": solver_result.solve_time_s,
    }


def supported_profile_gap_cvar_mcp(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    S,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
):
    """Solve the full-sample restricted CVaR MCP on ``S`` and certify full-game regret."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _, n1, n2 = A.shape
    A_sub, B_sub, S1, S2 = _restricted_action_data(A, B, S)
    start = perf_counter()
    solver_result = solve_cvar_mcp(
        A_sub,
        B_sub,
        p,
        gamma=gamma,
        alpha=alpha,
        solver=solver,
        fallback_solver=fallback_solver,
        solver_options=solver_options,
        tee=tee,
    )
    elapsed = perf_counter() - start
    x_sub = _validate_mixed_strategy(solver_result.x, len(S1), "restricted x", atol=1e-5)
    y_sub = _validate_mixed_strategy(solver_result.y, len(S2), "restricted y", atol=1e-5)
    x = expand_support_probs(x_sub, S1, n1)
    y = expand_support_probs(y_sub, S2, n2)
    cert = full_cvar_regret(A, B, p, gamma, alpha, x, y)
    return {
        "eta": float(cert["eta"]),
        "x": x,
        "y": y,
        "S": (S1, S2),
        "certificate": cert,
        "solver_result": solver_result,
        "restricted_x": x_sub,
        "restricted_y": y_sub,
        "success": bool(np.isfinite(cert["eta"])),
        "time_s": elapsed,
        "mcp_time_s": solver_result.solve_time_s,
    }


def _payoff_abs_scale(A: np.ndarray, B: np.ndarray) -> float:
    return max(1.0, float(np.max(np.abs(A))), float(np.max(np.abs(B))))


def supported_profile_gap_msd_dual(
    A_list,
    B_list,
    p,
    gamma: float,
    S,
    n_starts: int = 20,
    seed: int = 0,
    x0=None,
    y0=None,
    maxiter: int = 1000,
):
    """Minimize supported MSD regret using dualized best-response constraints."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative.")
    _, n1, n2 = A.shape
    S1, S2 = tuple(S[0]), tuple(S[1])
    if not S1 or not S2:
        raise ValueError("Action supports must be nonempty.")
    if any(i < 0 or i >= n1 for i in S1) or any(j < 0 or j >= n2 for j in S2):
        raise ValueError("Action support indices must be in range.")

    rng = np.random.default_rng(seed)
    s1, s2, K = len(S1), len(S2), A.shape[0]
    S1_idx = np.asarray(S1, dtype=int)
    S2_idx = np.asarray(S2, dtype=int)
    A_SS = A[:, S1_idx, :][:, :, S2_idx]
    B_SS = B[:, S1_idx, :][:, :, S2_idx]
    A_dev = A[:, :, S2_idx]
    B_dev = B[:, S1_idx, :]
    offsets = np.cumsum([0, s1, s2, 1, 1, 1, K, K, K, K])
    eta_idx = offsets[2]
    payoff_scale = _payoff_abs_scale(A, B)
    value_bound = 10.0 * (1.0 + 2.0 * gamma) * payoff_scale
    downside_bound = 4.0 * payoff_scale

    def unpack(v):
        x_s = v[offsets[0] : offsets[1]]
        y_s = v[offsets[1] : offsets[2]]
        eta = v[offsets[2]]
        beta1 = v[offsets[3]]
        beta2 = v[offsets[4]]
        d1 = v[offsets[5] : offsets[6]]
        d2 = v[offsets[6] : offsets[7]]
        lam1 = v[offsets[7] : offsets[8]]
        lam2 = v[offsets[8] : offsets[9]]
        return x_s, y_s, eta, beta1, beta2, d1, d2, lam1, lam2

    def values(v):
        x_s, y_s, eta, beta1, beta2, d1, d2, lam1, lam2 = unpack(v)
        u1 = np.einsum("a,kab,b->k", x_s, A_SS, y_s)
        u2 = np.einsum("a,kab,b->k", x_s, B_SS, y_s)
        mean1 = float(p @ u1)
        mean2 = float(p @ u2)
        rho1 = mean1 + gamma * float(p @ d1)
        rho2 = mean2 + gamma * float(p @ d2)
        p1_payoff_by_action = np.einsum("kib,b->ki", A_dev, y_s)
        p2_payoff_by_action = np.einsum("a,kaj->kj", x_s, B_dev)
        p1_mean_by_action = p @ p1_payoff_by_action
        p2_mean_by_action = p @ p2_payoff_by_action
        p1_dual_values = p1_mean_by_action + lam1 @ (
            p1_payoff_by_action - p1_mean_by_action
        )
        p2_dual_values = p2_mean_by_action + lam2 @ (
            p2_payoff_by_action - p2_mean_by_action
        )
        return {
            "eta": eta,
            "beta1": beta1,
            "beta2": beta2,
            "u1": u1,
            "u2": u2,
            "mean1": mean1,
            "mean2": mean2,
            "rho1": rho1,
            "rho2": rho2,
            "p1_dual_values": p1_dual_values,
            "p2_dual_values": p2_dual_values,
        }

    def eq_constraints(v):
        x_s, y_s, *_ = unpack(v)
        return np.array([np.sum(x_s) - 1.0, np.sum(y_s) - 1.0])

    def ineq_constraints(v):
        _, _, eta, beta1, beta2, d1, d2, *_ = unpack(v)
        vals = values(v)
        return np.concatenate(
            [
                vals["u1"] - vals["mean1"] - d1,
                vals["u2"] - vals["mean2"] - d2,
                beta1 - vals["p1_dual_values"],
                beta2 - vals["p2_dual_values"],
                np.array([vals["rho1"] + eta - beta1, vals["rho2"] + eta - beta2]),
            ]
        )

    def make_start(x_s, y_s):
        x_s = _normalize_simplex_candidate(x_s)
        y_s = _normalize_simplex_candidate(y_s)
        u1 = np.einsum("a,kab,b->k", x_s, A_SS, y_s)
        u2 = np.einsum("a,kab,b->k", x_s, B_SS, y_s)
        mean1 = float(p @ u1)
        mean2 = float(p @ u2)
        d1 = np.minimum(0.0, u1 - mean1)
        d2 = np.minimum(0.0, u2 - mean2)
        lam1 = 0.5 * gamma * p
        lam2 = 0.5 * gamma * p
        p1_payoff_by_action = np.einsum("kib,b->ki", A_dev, y_s)
        p2_payoff_by_action = np.einsum("a,kaj->kj", x_s, B_dev)
        p1_mean_by_action = p @ p1_payoff_by_action
        p2_mean_by_action = p @ p2_payoff_by_action
        beta1 = float(
            np.max(p1_mean_by_action + lam1 @ (p1_payoff_by_action - p1_mean_by_action))
        )
        beta2 = float(
            np.max(p2_mean_by_action + lam2 @ (p2_payoff_by_action - p2_mean_by_action))
        )
        rho1 = mean1 + gamma * float(p @ d1)
        rho2 = mean2 + gamma * float(p @ d2)
        eta = max(0.0, beta1 - rho1, beta2 - rho2)
        return np.concatenate(
            [x_s, y_s, np.array([eta, beta1, beta2]), d1, d2, lam1, lam2]
        )

    starts = []
    x_start = _support_start_from_profile(x0, S1)
    y_start = _support_start_from_profile(y0, S2)
    if x_start is not None and y_start is not None:
        starts.append(make_start(x_start, y_start))
    starts.append(make_start(np.ones(s1) / s1, np.ones(s2) / s2))
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(
            make_start(rng.dirichlet(np.ones(s1)), rng.dirichlet(np.ones(s2)))
        )

    constraints = [
        {"type": "eq", "fun": eq_constraints},
        {"type": "ineq", "fun": ineq_constraints},
    ]
    bounds = (
        [(0.0, 1.0)] * (s1 + s2)
        + [(0.0, value_bound), (-value_bound, value_bound), (-value_bound, value_bound)]
        + [(-downside_bound, 0.0)] * (2 * K)
        + [(0.0, float(gamma * pk)) for pk in p]
        + [(0.0, float(gamma * pk)) for pk in p]
    )

    best = None
    for start in starts:
        res = minimize(
            lambda v: v[eta_idx],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
        )
        violation = max(
            float(np.max(np.abs(eq_constraints(res.x)))),
            0.0,
            float(-np.min(ineq_constraints(res.x))),
        )
        candidate = {
            "dual_eta": float(res.x[eta_idx]),
            "violation": violation,
            "success": bool(res.success),
            "message": res.message,
            "result_x": res.x,
        }
        if best is None:
            best = candidate
        elif (candidate["dual_eta"], violation) < (
            best["dual_eta"],
            best["violation"],
        ):
            best = candidate

    assert best is not None
    x_s, y_s, *_ = unpack(best["result_x"])
    x = expand_support_probs(_normalize_simplex_candidate(x_s), S1, n1)
    y = expand_support_probs(_normalize_simplex_candidate(y_s), S2, n2)
    cert = full_msd_regret(A, B, p, gamma, x, y)
    best.update(
        {"eta": float(cert["eta"]), "x": x, "y": y, "S": (S1, S2), "certificate": cert}
    )
    best["optimizer_success"] = best["success"]
    best["success"] = bool(
        best["violation"] <= 1e-6 and np.isfinite(best["eta"])
    )
    return best


def supported_profile_gap_cvar_dual(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    S,
    n_starts: int = 20,
    seed: int = 0,
    x0=None,
    y0=None,
    maxiter: int = 1000,
):
    """Minimize supported CVaR regret using dualized best-response constraints."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    _, n1, n2 = A.shape
    S1, S2 = tuple(S[0]), tuple(S[1])
    if not S1 or not S2:
        raise ValueError("Action supports must be nonempty.")
    if any(i < 0 or i >= n1 for i in S1) or any(j < 0 or j >= n2 for j in S2):
        raise ValueError("Action support indices must be in range.")

    rng = np.random.default_rng(seed)
    s1, s2, K = len(S1), len(S2), A.shape[0]
    S1_idx = np.asarray(S1, dtype=int)
    S2_idx = np.asarray(S2, dtype=int)
    A_SS = A[:, S1_idx, :][:, :, S2_idx]
    B_SS = B[:, S1_idx, :][:, :, S2_idx]
    A_dev = A[:, :, S2_idx]
    B_dev = B[:, S1_idx, :]
    cap = gamma * p / alpha
    offsets = np.cumsum([0, s1, s2, 1, 1, 1, 1, 1, K, K, K, K])
    eta_idx = offsets[2]
    payoff_scale = _payoff_abs_scale(A, B)
    value_bound = 10.0 * payoff_scale
    tail_bound = 4.0 * payoff_scale

    def unpack(v):
        x_s = v[offsets[0] : offsets[1]]
        y_s = v[offsets[1] : offsets[2]]
        eta = v[offsets[2]]
        beta1 = v[offsets[3]]
        beta2 = v[offsets[4]]
        z1 = v[offsets[5]]
        z2 = v[offsets[6]]
        w1 = v[offsets[7] : offsets[8]]
        w2 = v[offsets[8] : offsets[9]]
        lam1 = v[offsets[9] : offsets[10]]
        lam2 = v[offsets[10] : offsets[11]]
        return x_s, y_s, eta, beta1, beta2, z1, z2, w1, w2, lam1, lam2

    def values(v):
        x_s, y_s, eta, beta1, beta2, z1, z2, w1, w2, lam1, lam2 = unpack(v)
        u1 = np.einsum("a,kab,b->k", x_s, A_SS, y_s)
        u2 = np.einsum("a,kab,b->k", x_s, B_SS, y_s)
        mean1 = float(p @ u1)
        mean2 = float(p @ u2)
        rho1 = (1.0 - gamma) * mean1 + gamma * z1 - float(cap @ w1)
        rho2 = (1.0 - gamma) * mean2 + gamma * z2 - float(cap @ w2)
        p1_payoff_by_action = np.einsum("kib,b->ki", A_dev, y_s)
        p2_payoff_by_action = np.einsum("a,kaj->kj", x_s, B_dev)
        p1_mean_by_action = p @ p1_payoff_by_action
        p2_mean_by_action = p @ p2_payoff_by_action
        p1_dual_values = (1.0 - gamma) * p1_mean_by_action + lam1 @ p1_payoff_by_action
        p2_dual_values = (1.0 - gamma) * p2_mean_by_action + lam2 @ p2_payoff_by_action
        return {
            "eta": eta,
            "beta1": beta1,
            "beta2": beta2,
            "z1": z1,
            "z2": z2,
            "u1": u1,
            "u2": u2,
            "rho1": rho1,
            "rho2": rho2,
            "p1_dual_values": p1_dual_values,
            "p2_dual_values": p2_dual_values,
        }

    def eq_constraints(v):
        x_s, y_s, *_, lam1, lam2 = unpack(v)
        return np.array(
            [
                np.sum(x_s) - 1.0,
                np.sum(y_s) - 1.0,
                np.sum(lam1) - gamma,
                np.sum(lam2) - gamma,
            ]
        )

    def ineq_constraints(v):
        _, _, eta, beta1, beta2, z1, z2, w1, w2, *_ = unpack(v)
        vals = values(v)
        return np.concatenate(
            [
                w1 - z1 + vals["u1"],
                w2 - z2 + vals["u2"],
                beta1 - vals["p1_dual_values"],
                beta2 - vals["p2_dual_values"],
                np.array([vals["rho1"] + eta - beta1, vals["rho2"] + eta - beta2]),
            ]
        )

    def make_start(x_s, y_s):
        x_s = _normalize_simplex_candidate(x_s)
        y_s = _normalize_simplex_candidate(y_s)
        u1 = np.einsum("a,kab,b->k", x_s, A_SS, y_s)
        u2 = np.einsum("a,kab,b->k", x_s, B_SS, y_s)
        mean1 = float(p @ u1)
        mean2 = float(p @ u2)
        z1 = float(np.min(u1))
        z2 = float(np.min(u2))
        w1 = np.maximum(z1 - u1, 0.0)
        w2 = np.maximum(z2 - u2, 0.0)
        lam1 = gamma * p
        lam2 = gamma * p
        p1_payoff_by_action = np.einsum("kib,b->ki", A_dev, y_s)
        p2_payoff_by_action = np.einsum("a,kaj->kj", x_s, B_dev)
        p1_mean_by_action = p @ p1_payoff_by_action
        p2_mean_by_action = p @ p2_payoff_by_action
        beta1 = float(
            np.max((1.0 - gamma) * p1_mean_by_action + lam1 @ p1_payoff_by_action)
        )
        beta2 = float(
            np.max((1.0 - gamma) * p2_mean_by_action + lam2 @ p2_payoff_by_action)
        )
        rho1 = (1.0 - gamma) * mean1 + gamma * z1 - float(cap @ w1)
        rho2 = (1.0 - gamma) * mean2 + gamma * z2 - float(cap @ w2)
        eta = max(0.0, beta1 - rho1, beta2 - rho2)
        return np.concatenate(
            [x_s, y_s, np.array([eta, beta1, beta2, z1, z2]), w1, w2, lam1, lam2]
        )

    starts = []
    x_start = _support_start_from_profile(x0, S1)
    y_start = _support_start_from_profile(y0, S2)
    if x_start is not None and y_start is not None:
        starts.append(make_start(x_start, y_start))
    starts.append(make_start(np.ones(s1) / s1, np.ones(s2) / s2))
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(
            make_start(rng.dirichlet(np.ones(s1)), rng.dirichlet(np.ones(s2)))
        )

    constraints = [
        {"type": "eq", "fun": eq_constraints},
        {"type": "ineq", "fun": ineq_constraints},
    ]
    bounds = (
        [(0.0, 1.0)] * (s1 + s2)
        + [(0.0, value_bound), (-value_bound, value_bound), (-value_bound, value_bound)]
        + [(-payoff_scale, payoff_scale)] * 2
        + [(0.0, tail_bound)] * (2 * K)
        + [(0.0, float(ck)) for ck in cap]
        + [(0.0, float(ck)) for ck in cap]
    )

    best = None
    for start in starts:
        res = minimize(
            lambda v: v[eta_idx],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
        )
        violation = max(
            float(np.max(np.abs(eq_constraints(res.x)))),
            0.0,
            float(-np.min(ineq_constraints(res.x))),
        )
        candidate = {
            "dual_eta": float(res.x[eta_idx]),
            "violation": violation,
            "success": bool(res.success),
            "message": res.message,
            "result_x": res.x,
        }
        if best is None:
            best = candidate
        elif (candidate["dual_eta"], violation) < (
            best["dual_eta"],
            best["violation"],
        ):
            best = candidate

    assert best is not None
    x_s, y_s, *_ = unpack(best["result_x"])
    x = expand_support_probs(_normalize_simplex_candidate(x_s), S1, n1)
    y = expand_support_probs(_normalize_simplex_candidate(y_s), S2, n2)
    cert = full_cvar_regret(A, B, p, gamma, alpha, x, y)
    best.update(
        {"eta": float(cert["eta"]), "x": x, "y": y, "S": (S1, S2), "certificate": cert}
    )
    best["optimizer_success"] = best["success"]
    best["success"] = bool(
        best["violation"] <= 1e-6 and np.isfinite(best["eta"])
    )
    return best


def _certified_search(
    A_list,
    B_list,
    p,
    screen_fn,
    support_gap_fn,
    screen_kwargs,
    regret_kwargs,
    config: SupportSearchConfig,
):
    A, B, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    epsilon_scr = config.epsilon_scr
    if epsilon_scr is None:
        epsilon_scr = 2.0 * config.epsilon / 3.0
    rng = np.random.default_rng(config.seed)
    best_screen = None
    best_regret = None
    best_regret_v = np.inf
    best_error = None

    for idx, (S, T) in enumerate(
        candidate_support_pairs(
            n1, n2, K, config.kappa, config.tau, config.max_candidates, rng
        ),
        start=1,
    ):
        try:
            start = perf_counter()
            screen = screen_fn(
                A,
                B,
                p,
                S=S,
                T=T,
                n_starts=config.n_screen_starts,
                seed=config.seed + idx,
                maxiter=config.screen_maxiter,
                **screen_kwargs,
            )
            screen["time_s"] = perf_counter() - start
            if best_screen is None:
                best_screen = screen
            elif screen["eta"] < best_screen["eta"]:
                best_screen = screen
            if screen["success"] and screen["eta"] <= epsilon_scr:
                # Screening certifies only the support pair; the returned profile
                # must minimize full-game regret over the action support.
                support_cert = support_gap_fn(
                    A,
                    B,
                    p,
                    S=screen["S"],
                    n_starts=config.n_regret_starts,
                    seed=config.seed + idx,
                    x0=screen["x"],
                    y0=screen["y"],
                    **regret_kwargs,
                )
                candidate = {
                    "screen": screen,
                    "support_certificate": support_cert,
                    "certificate": support_cert["certificate"],
                    "candidate_index": idx,
                    "success": bool(
                        support_cert["success"]
                        and np.isfinite(support_cert["eta"])
                        and support_cert["eta"] <= config.epsilon
                    ),
                }
                if support_cert["eta"] < best_regret_v:
                    best_regret = candidate
                    best_regret_v = support_cert["eta"]
                if candidate["success"]:
                    return SupportSearchResult(
                        success=True,
                        x=support_cert["x"],
                        y=support_cert["y"],
                        support=screen["S"],
                        scenarios=screen["T"],
                        candidate_index=idx,
                        metadata=candidate,
                    )
        except (
            ArithmeticError,
            RuntimeError,
        ) as exc:  # pragma: no cover - depends on numerical optimizer failures
            best_error = str(exc)
            continue

    if best_regret is not None:
        support_cert = best_regret["support_certificate"]
        return SupportSearchResult(
            success=False,
            x=support_cert["x"],
            y=support_cert["y"],
            support=best_regret["screen"]["S"],
            scenarios=best_regret["screen"]["T"],
            candidate_index=best_regret["candidate_index"],
            best_error=best_error,
            metadata={"best_screen": best_screen, "best_regret": best_regret},
        )
    return SupportSearchResult(
        success=False, best_error=best_error, metadata={"best_screen": best_screen}
    )


def _accepted_keyword_names(fn: Callable) -> set[str] | None:
    try:
        sig = signature(fn)
    except (TypeError, ValueError):
        return None
    if any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return None
    return {
        name
        for name, param in sig.parameters.items()
        if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }


def _call_support_gap(
    support_gap_fn: Callable,
    A,
    B,
    p,
    kwargs: dict[str, Any],
):
    accepted = _accepted_keyword_names(support_gap_fn)
    if accepted is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return support_gap_fn(A, B, p, **kwargs)


def _action_support_search(
    A_list,
    B_list,
    p,
    support_gap_fn: Callable,
    risk_kwargs: dict[str, Any],
    config: SupportSearchConfig,
    support_gap_kwargs: dict[str, Any] | None = None,
):
    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _, n1, n2 = A.shape
    rng = np.random.default_rng(config.seed)
    best_candidate = None
    best_eta = np.inf
    best_error = None
    extra_gap_kwargs = dict(support_gap_kwargs or {})

    for idx, S in enumerate(
        candidate_action_support_pairs(n1, n2, config.kappa, config.max_candidates, rng),
        start=1,
    ):
        try:
            start = perf_counter()
            call_kwargs = {
                **risk_kwargs,
                "S": S,
                "n_starts": config.n_regret_starts,
                "seed": config.seed + idx,
                "solver": config.solver,
                "fallback_solver": config.fallback_solver,
                "solver_options": config.solver_options,
                **extra_gap_kwargs,
            }
            support_cert = _call_support_gap(support_gap_fn, A, B, p, call_kwargs)
            support_cert.setdefault("time_s", perf_counter() - start)
            certificate = support_cert["certificate"]
            eta = float(support_cert["eta"])
            candidate = {
                "support_certificate": support_cert,
                "certificate": certificate,
                "candidate_index": idx,
                "success": bool(
                    support_cert["success"] and np.isfinite(eta) and eta <= config.epsilon
                ),
            }
            if eta < best_eta:
                best_candidate = candidate
                best_eta = eta
            if candidate["success"]:
                return SupportSearchResult(
                    success=True,
                    x=support_cert["x"],
                    y=support_cert["y"],
                    support=support_cert["S"],
                    candidate_index=idx,
                    metadata=candidate,
                )
        except (
            ArithmeticError,
            RuntimeError,
        ) as exc:  # pragma: no cover - depends on numerical optimizer failures
            best_error = str(exc)
            continue

    if best_candidate is not None:
        support_cert = best_candidate["support_certificate"]
        return SupportSearchResult(
            success=False,
            x=support_cert["x"],
            y=support_cert["y"],
            support=support_cert["S"],
            candidate_index=best_candidate["candidate_index"],
            best_error=best_error,
            metadata={"best_candidate": best_candidate},
        )
    return SupportSearchResult(
        success=False, best_error=best_error, metadata={"best_candidate": None}
    )


def _msd_regret_wrapper(A, B, p, x, y, gamma: float, **kwargs):
    return full_msd_regret(A, B, p, gamma, x, y, **kwargs)


def _cvar_regret_wrapper(A, B, p, x, y, gamma: float, alpha: float, **kwargs):
    return full_cvar_regret(A, B, p, gamma, alpha, x, y, **kwargs)


def small_support_action_search_msd(
    A_list,
    B_list,
    p,
    gamma: float,
    config: SupportSearchConfig | None = None,
    support_gap_func: Callable = supported_profile_gap_msd_dual,
    support_gap_kwargs: dict[str, Any] | None = None,
):
    """Search action supports only and certify the best supported MSD profile."""

    config = config or SupportSearchConfig()
    return _action_support_search(
        A_list,
        B_list,
        p,
        support_gap_func,
        {"gamma": gamma},
        config,
        support_gap_kwargs=support_gap_kwargs,
    )


def small_support_action_search_cvar(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    config: SupportSearchConfig | None = None,
    support_gap_func: Callable = supported_profile_gap_cvar_dual,
    support_gap_kwargs: dict[str, Any] | None = None,
):
    """Search action supports only and certify the best supported CVaR profile."""

    config = config or SupportSearchConfig()
    return _action_support_search(
        A_list,
        B_list,
        p,
        support_gap_func,
        {"gamma": gamma, "alpha": alpha},
        config,
        support_gap_kwargs=support_gap_kwargs,
    )


def small_support_search_msd(
    A_list,
    B_list,
    p,
    gamma: float,
    config: SupportSearchConfig | None = None,
    supported_profile_gap_func: Callable = supported_profile_gap_msd_dual,
):
    """Search small supports and return success only with a full-game MSD regret certificate."""

    config = config or SupportSearchConfig()
    return _certified_search(
        A_list,
        B_list,
        p,
        restricted_profile_gap_msd,
        supported_profile_gap_func,
        {"gamma": gamma},
        {"gamma": gamma},
        config,
    )


def small_support_search_cvar(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    config: SupportSearchConfig | None = None,
    supported_profile_gap_func: Callable = supported_profile_gap_cvar_dual,
):
    """Search small supports and return success only with a full-game CVaR regret certificate."""

    config = config or SupportSearchConfig()
    return _certified_search(
        A_list,
        B_list,
        p,
        restricted_profile_gap_cvar,
        supported_profile_gap_func,
        {"gamma": gamma, "alpha": alpha},
        {"gamma": gamma, "alpha": alpha},
        config,
    )
