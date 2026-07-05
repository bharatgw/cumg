"""Small-support screening and certified randomized support search."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .cvar import cvar_profile_values, cvar_value_from_state_payoffs
from .msd import msd_profile_values, msd_value_from_state_payoffs
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


def sample_supports(n: int, size: int, rng: np.random.Generator, max_exact: int = 200) -> list[tuple[int, ...]]:
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
        S = (action1[int(rng.integers(len(action1)))], action2[int(rng.integers(len(action2)))])
        T = (
            data_supports[int(rng.integers(len(data_supports)))],
            data_supports[int(rng.integers(len(data_supports)))],
        )
        yield S, T


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


def _simplex_starts(dim: int, rng: np.random.Generator, n_random: int):
    starts = [np.ones(dim) / dim]
    starts.extend(rng.dirichlet(np.ones(dim), size=n_random))
    return starts


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


def _maximize_on_simplex(
    objective: Callable[[np.ndarray], float],
    dim: int,
    rng: np.random.Generator,
    n_starts: int = 20,
    maxiter: int = 1000,
) -> dict[str, Any]:
    best = None
    constraints = [{"type": "eq", "fun": lambda v: np.sum(v) - 1.0}]
    bounds = [(0.0, 1.0)] * dim
    for start in _simplex_starts(dim, rng, max(0, n_starts - 1)):
        res = minimize(
            lambda v: -objective(v),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
        )
        value = objective(res.x)
        if best is None or value > best["value"]:
            best = {"value": float(value), "strategy": res.x, "success": bool(res.success), "message": res.message}
    return best


def full_msd_regret(A_list, B_list, p, gamma: float, x, y, n_starts: int = 20, seed: int = 0) -> dict[str, Any]:
    """Compute max MSD best-response gain for a mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    rng = np.random.default_rng(seed)
    _, n1, n2 = A.shape
    x = _validate_mixed_strategy(x, n1, "x")
    y = _validate_mixed_strategy(y, n2, "y")
    base = msd_profile_values(A, B, p, gamma, x, y)
    p1_payoff_by_action = np.einsum("kij,j->ki", A, y)
    p2_payoff_by_action = np.einsum("i,kij->kj", x, B)

    def p1_dev_value(x_dev):
        return msd_value_from_state_payoffs(np.einsum("i,ki->k", x_dev, p1_payoff_by_action), p, gamma)

    def p2_dev_value(y_dev):
        return msd_value_from_state_payoffs(np.einsum("j,kj->k", y_dev, p2_payoff_by_action), p, gamma)

    best1 = _maximize_on_simplex(p1_dev_value, n1, rng, n_starts=n_starts)
    best2 = _maximize_on_simplex(p2_dev_value, n2, rng, n_starts=n_starts)
    current1 = p1_dev_value(x)
    current2 = p2_dev_value(y)
    best1["value"] = max(best1["value"], current1)
    best2["value"] = max(best2["value"], current2)
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
    n_starts: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute max CVaR best-response gain for a mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    rng = np.random.default_rng(seed)
    _, n1, n2 = A.shape
    x = _validate_mixed_strategy(x, n1, "x")
    y = _validate_mixed_strategy(y, n2, "y")
    base = cvar_profile_values(A, B, p, gamma, alpha, x, y)
    p1_payoff_by_action = np.einsum("kij,j->ki", A, y)
    p2_payoff_by_action = np.einsum("i,kij->kj", x, B)

    def p1_dev_value(x_dev):
        return cvar_value_from_state_payoffs(np.einsum("i,ki->k", x_dev, p1_payoff_by_action), p, gamma, alpha)

    def p2_dev_value(y_dev):
        return cvar_value_from_state_payoffs(np.einsum("j,kj->k", y_dev, p2_payoff_by_action), p, gamma, alpha)

    best1 = _maximize_on_simplex(p1_dev_value, n1, rng, n_starts=n_starts)
    best2 = _maximize_on_simplex(p2_dev_value, n2, rng, n_starts=n_starts)
    current1 = p1_dev_value(x)
    current2 = p2_dev_value(y)
    best1["value"] = max(best1["value"], current1)
    best2["value"] = max(best2["value"], current2)
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

    constraints = [{"type": "eq", "fun": eq_constraints}, {"type": "ineq", "fun": ineq_constraints}]
    bounds = [(0.0, 1.0)] * (s1 + s2 + t1 + t2) + [eta_bounds]
    starts = [np.concatenate([np.ones(s1) / s1, np.ones(s2) / s2, p_T1, p_T2, np.array([0.0])])]
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
            np.max(np.abs(eq_constraints(res.x))),
            max(0.0, -np.min(ineq_constraints(res.x))),
        )
        candidate = {
            "eta": float(res.fun),
            "violation": float(violation),
            "success": bool(res.success),
            "message": res.message,
        }
        if best is None or (candidate["eta"], candidate["violation"]) < (best["eta"], best["violation"]):
            best = candidate | {"result_x": res.x}

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
    return _screen_profile(A, B, p, msd_value_from_state_payoffs, (gamma,), S, T, n_starts, seed, maxiter)


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
    return _screen_profile(A, B, p, cvar_value_from_state_payoffs, (gamma, alpha), S, T, n_starts, seed, maxiter)


def _search_with_solver(A_list, B_list, p, solver_fn, solver_kwargs, config: SupportSearchConfig):
    """Legacy restricted-MCP helper retained for tests and direct callers."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    rng = np.random.default_rng(config.seed)
    best_error = None

    for idx, (S, T) in enumerate(
        candidate_support_pairs(n1, n2, K, config.kappa, config.tau, config.max_candidates, rng),
        start=1,
    ):
        A_sub, B_sub, p_sub, p1_support, p2_support, scenario_support = _restricted_data(A, B, p, S, T)
        try:
            result = solver_fn(
                A_sub,
                B_sub,
                p=p_sub,
                solver=config.solver,
                fallback_solver=config.fallback_solver,
                solver_options=config.solver_options,
                **solver_kwargs,
            )
        except Exception as exc:  # pragma: no cover - exercised only when solvers fail
            best_error = str(exc)
            continue

        x = expand_support_probs(result.x, p1_support, n1)
        y = expand_support_probs(result.y, p2_support, n2)
        return SupportSearchResult(
            success=True,
            x=x,
            y=y,
            support=(p1_support, p2_support),
            scenarios=scenario_support,
            candidate_index=idx,
            solver_result=result,
        )

    return SupportSearchResult(success=False, best_error=best_error)


def _certified_search(
    A_list,
    B_list,
    p,
    screen_fn,
    regret_fn,
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
        candidate_support_pairs(n1, n2, K, config.kappa, config.tau, config.max_candidates, rng),
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
            if best_screen is None or screen["eta"] < best_screen["eta"]:
                best_screen = screen
            if screen["success"] and screen["eta"] <= epsilon_scr:
                cert = regret_fn(
                    A,
                    B,
                    p,
                    screen["x"],
                    screen["y"],
                    n_starts=config.n_regret_starts,
                    seed=config.seed + idx,
                    **regret_kwargs,
                )
                candidate = {
                    "screen": screen,
                    "certificate": cert,
                    "candidate_index": idx,
                    "success": bool(np.isfinite(cert["eta"]) and cert["eta"] <= config.epsilon),
                }
                if cert["eta"] < best_regret_v:
                    best_regret = candidate
                    best_regret_v = cert["eta"]
                if candidate["success"]:
                    return SupportSearchResult(
                        success=True,
                        x=screen["x"],
                        y=screen["y"],
                        support=screen["S"],
                        scenarios=screen["T"],
                        candidate_index=idx,
                        metadata=candidate,
                    )
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - depends on numerical optimizer failures
            best_error = str(exc)
            continue

    if best_regret is not None:
        return SupportSearchResult(
            success=False,
            x=best_regret["screen"]["x"],
            y=best_regret["screen"]["y"],
            support=best_regret["screen"]["S"],
            scenarios=best_regret["screen"]["T"],
            candidate_index=best_regret["candidate_index"],
            best_error=best_error,
            metadata={"best_screen": best_screen, "best_regret": best_regret},
        )
    return SupportSearchResult(success=False, best_error=best_error, metadata={"best_screen": best_screen})


def _msd_regret_wrapper(A, B, p, x, y, gamma: float, **kwargs):
    return full_msd_regret(A, B, p, gamma, x, y, **kwargs)


def _cvar_regret_wrapper(A, B, p, x, y, gamma: float, alpha: float, **kwargs):
    return full_cvar_regret(A, B, p, gamma, alpha, x, y, **kwargs)


def small_support_search_msd(A_list, B_list, p, gamma: float, config: SupportSearchConfig | None = None):
    """Search small supports and return success only with a full-game MSD regret certificate."""

    config = config or SupportSearchConfig()
    return _certified_search(
        A_list,
        B_list,
        p,
        restricted_profile_gap_msd,
        _msd_regret_wrapper,
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
):
    """Search small supports and return success only with a full-game CVaR regret certificate."""

    config = config or SupportSearchConfig()
    return _certified_search(
        A_list,
        B_list,
        p,
        restricted_profile_gap_cvar,
        _cvar_regret_wrapper,
        {"gamma": gamma, "alpha": alpha},
        {"gamma": gamma, "alpha": alpha},
        config,
    )
