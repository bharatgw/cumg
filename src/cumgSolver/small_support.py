"""Small-support helpers and randomized restricted MCP search."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .cvar import solve_cvar_mcp
from .msd import solve_msd_mcp
from .results import SupportSearchConfig, SupportSearchResult
from .validation import normalize_game_inputs


def expand_support_probs(probs, support, n: int) -> np.ndarray:
    """Expand probabilities on a support to a full strategy vector."""

    out = np.zeros(n, dtype=float)
    out[np.asarray(support, dtype=int)] = np.asarray(probs, dtype=float)
    return out


def sample_supports(n: int, size: int, rng: np.random.Generator, max_exact: int = 200) -> list[tuple[int, ...]]:
    """Enumerate small support sets exactly, otherwise sample without replacement."""

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
    """Yield randomized action/scenario support candidates."""

    action1 = sample_supports(n1, kappa, rng, max_exact=max_candidates)
    action2 = sample_supports(n2, kappa, rng, max_exact=max_candidates)
    data_supports = sample_supports(K, min(tau, K), rng, max_exact=max_candidates)
    for _ in range(max_candidates):
        S = (action1[int(rng.integers(len(action1)))], action2[int(rng.integers(len(action2)))])
        T = data_supports[int(rng.integers(len(data_supports)))]
        yield S, T


def support_sizes(K: int, n: int) -> tuple[int, int]:
    """Default support sizes used in the experiments."""

    kappa = min(n, max(2, int(np.ceil(np.sqrt(n)))))
    tau = min(K, max(5, int(np.ceil(np.sqrt(K)))))
    return kappa, tau


def _restricted_data(A, B, p, S, T):
    p1_support, p2_support = tuple(S[0]), tuple(S[1])
    scenario_support = tuple(T)
    p_sub = p[list(scenario_support)]
    p_sub = p_sub / p_sub.sum()
    A_sub = [A[k][np.ix_(p1_support, p2_support)] for k in scenario_support]
    B_sub = [B[k][np.ix_(p1_support, p2_support)] for k in scenario_support]
    return A_sub, B_sub, p_sub, p1_support, p2_support, scenario_support


def _search_with_solver(A_list, B_list, p, solver_fn, solver_kwargs, config: SupportSearchConfig):
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


def small_support_search_msd(A_list, B_list, p, gamma: float, config: SupportSearchConfig | None = None):
    """Search randomized small supports and solve restricted MSD MCPs."""

    config = config or SupportSearchConfig()
    return _search_with_solver(A_list, B_list, p, solve_msd_mcp, {"gamma": gamma}, config)


def small_support_search_cvar(
    A_list,
    B_list,
    p,
    gamma: float,
    alpha: float,
    config: SupportSearchConfig | None = None,
):
    """Search randomized small supports and solve restricted CVaR MCPs."""

    config = config or SupportSearchConfig()
    return _search_with_solver(A_list, B_list, p, solve_cvar_mcp, {"gamma": gamma, "alpha": alpha}, config)

