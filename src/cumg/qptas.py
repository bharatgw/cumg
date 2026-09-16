"""Exhaustive or sampled kappa-uniform search, with optional scenario screening."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import comb, prod
from random import Random
from time import perf_counter

import numpy as np

from .cvar import cvar_value_from_state_payoffs
from .msd import msd_value_from_state_payoffs
from .results import QPTASResult
from .small_support import (
    _keep_current_strategy_if_better,
    _maximize_cvar_on_simplex,
    _maximize_msd_on_simplex,
    support_sizes,
)
from .validation import normalize_probabilities


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _profile_denominators(kappa: int | Sequence[int], m: int) -> tuple[int, ...]:
    if np.isscalar(kappa) or kappa is None:
        return (_positive_integer(kappa, "kappa"),) * m
    denominators = tuple(_positive_integer(k, "kappa") for k in kappa)
    if len(denominators) != m:
        raise ValueError(f"kappa must be a scalar or a sequence of length {m}.")
    return denominators


def enumerate_kappa_uniform_strategies(n: int, kappa: int) -> Iterator[np.ndarray]:
    """Yield every ``c / kappa`` with nonnegative integer counts summing to kappa.

    Enumeration is deterministic, beginning with the pure strategy on action 0.
    The number of strategies is ``comb(kappa + n - 1, n - 1)``. Only the current
    count vector is stored; kappa is a denominator and is never capped at n.
    """

    n = _positive_integer(n, "n")
    kappa = _positive_integer(kappa, "kappa")
    counts = [kappa] + [0] * (n - 1)
    while True:
        yield np.asarray(counts, dtype=float) / kappa
        for i in range(n - 2, -1, -1):
            if counts[i] > 0:
                break
        else:
            return
        remaining = counts[-1] + 1
        counts[i] -= 1
        counts[-1] = 0
        counts[i + 1] = remaining


def enumerate_kappa_uniform_profiles(action_sizes: Sequence[int], kappa: int) -> Iterator[tuple[np.ndarray, ...]]:
    """Lazily enumerate the Cartesian product of all players' kappa-uniform grids.

    Unlike ``itertools.product``, this does not cache each player's entire grid.
    Each yielded profile owns its arrays, so changing one cannot affect another.
    """

    kappa = _positive_integer(kappa, "kappa")
    sizes = tuple(_positive_integer(n, "action size") for n in action_sizes)
    if not sizes:
        raise ValueError("At least one player is required.")
    profile: list[np.ndarray] = []

    def visit(player: int) -> Iterator[tuple[np.ndarray, ...]]:
        if player == len(sizes):
            yield tuple(strategy.copy() for strategy in profile)
            return
        for strategy in enumerate_kappa_uniform_strategies(sizes[player], kappa):
            profile.append(strategy)
            yield from visit(player + 1)
            profile.pop()

    yield from visit(0)


def _kappa_uniform_strategy_from_rank(n: int, kappa: int, rank: int) -> np.ndarray:
    """Decode a valid zero-based strategy rank in the exhaustive order."""

    counts = []
    remaining = kappa
    for actions_left in range(n, 1, -1):
        # With t tokens left after this action, there are
        # comb(t + actions_left - 2, actions_left - 2) completions.
        # The cumulative count through t is comb(t + actions_left - 1,
        # actions_left - 1), so binary search locates the rank's block.
        lo, hi = 0, remaining
        while lo < hi:
            mid = (lo + hi) // 2
            if rank < comb(mid + actions_left - 1, actions_left - 1):
                hi = mid
            else:
                lo = mid + 1
        counts.append(remaining - lo)
        if lo:
            rank -= comb(lo + actions_left - 2, actions_left - 1)
        remaining = lo
    counts.append(remaining)
    return np.asarray(counts, dtype=float) / kappa


def sample_kappa_uniform_profiles(
    action_sizes: Sequence[int],
    kappa: int | Sequence[int],
    n_samples: int,
    seed: int | None = 0,
) -> Iterator[tuple[np.ndarray, ...]]:
    """Yield distinct joint profiles uniformly without replacement from the grid.

    Draw at most ``n_samples`` profiles, or the whole grid if it is smaller.
    A fixed seed gives a reproducible random order; None uses a fresh seed.
    Uniformity is over grid profiles (integer count vectors), rather than over
    ordered lists of kappa actions. The latter would favor some count vectors.

    Integer ranks are drawn using a sparse Fisher-Yates shuffle, then decoded
    without enumerating preceding profiles. Python integers support grids larger
    than 64-bit limits. Storage grows with the number of profiles drawn, not the
    total grid size. Increasing n_samples with the same seed extends the prefix.
    kappa may be a scalar or one positive integer denominator per player.
    """

    n_samples = _positive_integer(n_samples, "n_samples")
    sizes = tuple(_positive_integer(n, "action size") for n in action_sizes)
    if not sizes:
        raise ValueError("At least one player is required.")
    denominators = _profile_denominators(kappa, len(sizes))
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer or None.")
        seed = int(seed)
    rng = Random(seed)
    grid_sizes = tuple(comb(k + n - 1, n - 1) for n, k in zip(sizes, denominators, strict=True))
    remaining = prod(grid_sizes)
    swaps: dict[int, int] = {}
    for _ in range(min(n_samples, remaining)):
        position = rng.randrange(remaining)
        rank = swaps.get(position, position)
        last = remaining - 1
        replacement = swaps.pop(last, last)
        if position != last:
            swaps[position] = replacement
        remaining -= 1
        profile = []
        for n, k, grid_size in zip(reversed(sizes), reversed(denominators), reversed(grid_sizes), strict=True):
            rank, strategy_rank = divmod(rank, grid_size)
            profile.append(_kappa_uniform_strategy_from_rank(n, k, strategy_rank))
        yield tuple(reversed(profile))


def _player_parameters(value, m: int, name: str) -> np.ndarray:
    parameters = np.asarray(value, dtype=float)
    if parameters.ndim == 0:
        parameters = np.full(m, float(parameters))
    if parameters.shape != (m,) or not np.all(np.isfinite(parameters)):
        raise ValueError(f"{name} must be finite and either a scalar or a vector of length {m}.")
    return parameters


def _solve_qptas(
    payoffs,
    p,
    *,
    model: str,
    gamma,
    alpha,
    kappa: int | Sequence[int] | None,
    epsilon: float,
    max_candidates: int | None,
    seed: int | None,
    screened: bool = False,
    tau: int | None = None,
    epsilon_scr: float | None = None,
) -> QPTASResult:
    start = perf_counter()
    if not screened:
        kappa = _positive_integer(kappa, "kappa")
    elif max_candidates is None:
        raise ValueError("max_candidates must be a positive integer for screened sampling.")
    if max_candidates is not None:
        max_candidates = _positive_integer(max_candidates, "max_candidates")
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    tensors = tuple(np.asarray(payoff, dtype=float) for payoff in payoffs)
    m = len(tensors)
    if not m:
        raise ValueError("At least one player is required.")
    shape = tensors[0].shape
    if len(shape) != m + 1 or any(size == 0 for size in shape):
        raise ValueError("Each payoff tensor must have nonempty shape (K, n_1, ..., n_m).")
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("All players' payoff tensors must have the same shape (K, n_1, ..., n_m).")
    if any(not np.all(np.isfinite(tensor)) for tensor in tensors):
        raise ValueError("Payoff tensors must contain finite values.")
    K, *action_sizes = shape
    probabilities = normalize_probabilities(np.ones(K) if p is None else p, K)
    gammas = _player_parameters(gamma, m, "gamma")
    if np.any(gammas < 0) or np.any(gammas > 1):
        domain = "in [0, 1]"
        raise ValueError(f"gamma must be {domain}.")
    if model == "CVaR":
        alphas = _player_parameters(alpha, m, "alpha")
        if np.any(alphas <= 0) or np.any(alphas > 1):
            raise ValueError("alpha must be in (0, 1].")

    if screened and kappa is None:
        kappa = tuple(support_sizes(K, n)[0] for n in action_sizes)
    kappas = _profile_denominators(kappa, m)
    total_profiles = prod(comb(k + n - 1, n - 1) for n, k in zip(action_sizes, kappas, strict=True))
    total_pairs = None
    if screened:
        kappa = kappas
        tau = support_sizes(K, action_sizes[0])[1] if tau is None else _positive_integer(tau, "tau")
        epsilon_scr = 2.0 * epsilon / 3.0 if epsilon_scr is None else epsilon_scr
        if not np.isscalar(epsilon_scr) or not np.isfinite(epsilon_scr) or epsilon_scr < 0:
            raise ValueError("epsilon_scr must be finite and nonnegative.")
        total_pairs = total_profiles * comb(tau + K - 1, K - 1) ** m
        # Treat q_i as an additional grid factor. Distinct pairs can share x;
        # a failed screen must not remove that x with all other choices of q.
        candidates = sample_kappa_uniform_profiles(
            (*action_sizes, *((K,) * m)), (*kappas, *((tau,) * m)), max_candidates, seed
        )
    else:
        candidates = (
            enumerate_kappa_uniform_profiles(action_sizes, kappa)
            if max_candidates is None
            else sample_kappa_uniform_profiles(action_sizes, kappa, max_candidates, seed)
        )

    def evaluate_profile(profile):
        # Stream evaluations so unscreened search still rejects early. Screened
        # search evaluates all current payoffs before doing any best-response LP.
        for player, tensor in enumerate(tensors):
            # Contract opponents in descending axis order, preserving scenario
            # axis 0 and all actions of the deviating player: result is (K, n_i).
            payoff_by_action = tensor
            for opponent in range(m - 1, -1, -1):
                if opponent != player:
                    payoff_by_action = np.tensordot(payoff_by_action, profile[opponent], axes=([opponent + 1], [0]))
            state_payoffs = payoff_by_action @ profile[player]
            if model == "MSD":
                current = msd_value_from_state_payoffs(state_payoffs, probabilities, gammas[player])
            else:
                current = cvar_value_from_state_payoffs(state_payoffs, probabilities, gammas[player], alphas[player])
            if not np.isfinite(current):
                raise RuntimeError("QPTAS regret evaluation returned a non-finite payoff.")
            yield payoff_by_action, current

    best_response_solves = 0
    profiles_checked = 0
    screen_rejections = 0
    screen_passes = 0
    for candidate in candidates:
        profiles_checked += 1
        profile = candidate[:m]
        evaluations = evaluate_profile(profile)
        if screened:
            evaluations = list(evaluations)
            if any(
                np.any(q @ payoff_by_action > current + epsilon_scr)
                for q, (payoff_by_action, current) in zip(candidate[m:], evaluations, strict=True)
            ):
                screen_rejections += 1
                continue
            screen_passes += 1
        current_values = []
        regrets = []
        best_responses = []
        for player, (payoff_by_action, current) in enumerate(evaluations):
            if model == "MSD":
                best = _maximize_msd_on_simplex(payoff_by_action, probabilities, gammas[player])
            else:
                best = _maximize_cvar_on_simplex(payoff_by_action, probabilities, gammas[player], alphas[player])
            best_response_solves += 1
            if not np.isfinite(current) or not np.isfinite(best["value"]):
                raise RuntimeError("QPTAS regret evaluation returned a non-finite payoff.")
            best = _keep_current_strategy_if_better(best, profile[player], current)
            regret = max(0.0, float(best["value"] - current))
            if regret > epsilon:
                break
            current_values.append(float(current))
            regrets.append(regret)
            best_responses.append(best)
        else:
            return QPTASResult(
                success=True,
                model=model,
                kappa=kappa,
                epsilon=float(epsilon),
                profiles_checked=profiles_checked,
                total_profiles=total_profiles,
                best_response_solves=best_response_solves,
                solve_time_s=perf_counter() - start,
                strategies=profile,
                certificate={
                    "eta": max(regrets),
                    "regrets": tuple(regrets),
                    "current_values": tuple(current_values),
                    "best_responses": tuple(best_responses),
                },
                termination_reason="epsilon_reached",
                max_candidates=max_candidates,
                seed=int(seed) if max_candidates is not None and seed is not None else None,
                tau=tau,
                epsilon_scr=epsilon_scr,
                total_pairs=total_pairs,
                screen_rejections=screen_rejections,
                screen_passes=screen_passes,
                screening_distributions=candidate[m:] if screened else None,
            )
    exhausted_reason = "pair_grid_exhausted" if screened else "grid_exhausted"
    grid_size = total_pairs if screened else total_profiles
    return QPTASResult(
        success=False,
        model=model,
        kappa=kappa,
        epsilon=float(epsilon),
        profiles_checked=profiles_checked,
        total_profiles=total_profiles,
        best_response_solves=best_response_solves,
        solve_time_s=perf_counter() - start,
        termination_reason=exhausted_reason if profiles_checked == grid_size else "sample_exhausted",
        max_candidates=max_candidates,
        seed=int(seed) if max_candidates is not None and seed is not None else None,
        tau=tau,
        epsilon_scr=epsilon_scr,
        total_pairs=total_pairs,
        screen_rejections=screen_rejections,
        screen_passes=screen_passes,
    )


def solve_msd_qptas(
    payoffs,
    p=None,
    *,
    gamma=0.0,
    kappa: int,
    epsilon: float = 1e-3,
    max_candidates: int | None = None,
    seed: int | None = 0,
) -> QPTASResult:
    """Enumerate kappa-uniform profiles and return the first epsilon-MSD-DRE.

    ``payoffs[i]`` has shape ``(K, n_1, ..., n_m)`` and contains player i's
    scenario payoffs. For two players pass ``[A, B]`` with shape ``(K, n1, n2)``
    each. The number of players and action sizes are inferred from these tensors.
    ``p`` is a length-K vector of nonnegative weights (normalized internally),
    or None for uniform weights. ``gamma`` is a scalar in [0, 1] or a length-m
    vector. MSD payoff is ``E[u] - gamma * E[max(E[u] - u, 0)]``.

    Every best-response LP uses all K scenarios and all the player's actions,
    allowing arbitrary mixed deviations. Reject a profile as soon as any regret
    exceeds epsilon. By default the full grid is streamed in deterministic order.
    Set ``max_candidates=N`` to check up to N distinct, uniformly sampled joint
    profiles in random order, with ``seed`` controlling reproducibility. Stop
    immediately on finding an epsilon-DRE in either mode. No external MCP solver
    is needed. Certificates use floating-point LP
    solves, with the same numerical conventions as ``full_msd_regret``.

    A supplied kappa need not contain an epsilon-DRE. Exhausting the grid returns
    ``success=False`` and ``termination_reason='grid_exhausted'``; this makes no
    claim about profiles outside that grid. If a random sample covers only part
    of the grid and finds no solution, the reason is ``'sample_exhausted'``;
    untested grid profiles may still contain an epsilon-DRE. LP failures raise
    RuntimeError. ``max_candidates=None`` ignores seed and searches exhaustively.
    """

    return _solve_qptas(
        payoffs,
        p,
        model="MSD",
        gamma=gamma,
        alpha=None,
        kappa=kappa,
        epsilon=epsilon,
        max_candidates=max_candidates,
        seed=seed,
    )


def solve_cvar_qptas(
    payoffs,
    p=None,
    *,
    gamma=0.0,
    alpha=0.5,
    kappa: int,
    epsilon: float = 1e-3,
    max_candidates: int | None = None,
    seed: int | None = 0,
) -> QPTASResult:
    """Enumerate kappa-uniform profiles and return the first epsilon-CVaR-DRE.

    Input layout, enumeration, sampling and termination follow ``solve_msd_qptas``.
    ``max_candidates=N`` samples up to N distinct grid profiles uniformly without
    replacement using seed; None retains the exhaustive deterministic search.
    ``gamma`` (in [0, 1]) and ``alpha`` (in (0, 1]) may each be a scalar or a
    length-m vector. Payoff is ``(1-gamma)*E[u] + gamma*lower_CVaR_alpha(u)``,
    where alpha is the lower-tail probability mass. Every best-response LP
    uses all K samples and arbitrary mixed deviations. Certificates follow
    the floating-point conventions of ``full_cvar_regret``.
    """

    return _solve_qptas(
        payoffs,
        p,
        model="CVaR",
        gamma=gamma,
        alpha=alpha,
        kappa=kappa,
        epsilon=epsilon,
        max_candidates=max_candidates,
        seed=seed,
    )


def solve_msd_qptas_screened(
    payoffs,
    p=None,
    *,
    gamma=0.0,
    kappa: int | Sequence[int] | None = None,
    tau: int | None = None,
    epsilon: float = 0.01,
    epsilon_scr: float | None = None,
    max_candidates: int = 1000,
    seed: int | None = 0,
) -> QPTASResult:
    """Sample joint (x, q) pairs, screen, and certify the first epsilon-MSD-DRE.

    Payoff tensors, probabilities and risk parameters follow ``solve_msd_qptas``.
    Defaults are kappa_i=min(n_i, max(2, ceil(sqrt(n_i)))),
    tau=min(K, max(5, ceil(sqrt(K)))), and epsilon_scr=2*epsilon/3.
    Override kappa with a scalar or one positive integer per player, and tau
    with a positive integer denominator (overrides need not be capped).

    Draw up to max_candidates distinct joint pairs uniformly without replacement
    from the product of the players' kappa_i-uniform strategy grids and their
    tau-uniform scenario grids. Each q_i is independent across players within
    a draw. A profile x can recur with a different q. Sampling is uniform over
    integer count vectors; it does not draw scenarios according to p or project
    q into a risk ambiguity set. Seed controls the reproducible random order.

    Compute current robust payoffs using all K scenarios and p. Reject the pair
    if any pure-action expected payoff under q_i exceeds current_i+epsilon_scr.
    A screen failure does not prove that x fails epsilon. Only screen survivors
    undergo full-sample, arbitrary mixed best-response LP checks. Return at the
    first full certificate with every regret <= epsilon, using the existing
    floating-point LP conventions. LP failures raise RuntimeError.

    This is a heuristic, not the theorem's sufficient support bound.
    profiles_checked counts joint pairs, total_profiles counts only x's grid,
    and total_pairs counts the full (x,q) grid. Failure returns no strategies or
    certificate and reason 'sample_exhausted' (or 'pair_grid_exhausted' if every
    pair was checked); neither reason establishes nonexistence of an epsilon-DRE.
    """

    return _solve_qptas(
        payoffs,
        p,
        model="MSD",
        gamma=gamma,
        alpha=None,
        kappa=kappa,
        epsilon=epsilon,
        max_candidates=max_candidates,
        seed=seed,
        screened=True,
        tau=tau,
        epsilon_scr=epsilon_scr,
    )


def solve_cvar_qptas_screened(
    payoffs,
    p=None,
    *,
    gamma=0.0,
    alpha=0.5,
    kappa: int | Sequence[int] | None = None,
    tau: int | None = None,
    epsilon: float = 0.01,
    epsilon_scr: float | None = None,
    max_candidates: int = 1000,
    seed: int | None = 0,
) -> QPTASResult:
    """Screen sampled (x,q) pairs and certify the first epsilon-CVaR-DRE.

    Sampling, screening, defaults, counters, and termination follow
    ``solve_msd_qptas_screened``. Gamma and alpha have the domains and scalar or
    per-player forms documented in ``solve_cvar_qptas``. Both current payoffs
    and best-response LPs use all K scenarios and p; q is used only in screening.
    """

    return _solve_qptas(
        payoffs,
        p,
        model="CVaR",
        gamma=gamma,
        alpha=alpha,
        kappa=kappa,
        epsilon=epsilon,
        max_candidates=max_candidates,
        seed=seed,
        screened=True,
        tau=tau,
        epsilon_scr=epsilon_scr,
    )
