"""Stochastic first-order residual minimization for smoothed CUMGs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .small_support import full_cvar_regret, full_msd_regret
from .validation import normalize_game_inputs


def varphi_tau(a, tau: float):
    """Stable smooth approximation ``tau * log(1 + exp(a / tau))``."""

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    return tau * np.logaddexp(0.0, np.asarray(a, dtype=float) / tau)


@dataclass(frozen=True)
class StochasticFOConfig:
    """Configuration for stochastic first-order residual minimization."""

    kappa: float = 1e-2
    tau: float = 1e-2
    max_iter: int = 1000
    batch_size: int | None = None
    step_size: float = 1e-1
    step_decay: float = 0.5
    seed: int = 0
    x0: np.ndarray | None = None
    y0: np.ndarray | None = None
    theta0: tuple[float, float] | float | None = None
    logit_bound: float | None = 20.0
    theta_bounds: tuple[float | None, float | None] | None = None
    gradient_clip_norm: float | None = None
    record_every: int | None = None
    certify_every: int | None = None
    regret_tolerance: float = 1e-3


@dataclass(frozen=True)
class StochasticFOResult:
    """Result from stochastic first-order residual minimization."""

    success: bool
    x: np.ndarray
    y: np.ndarray
    model: str
    certificate: dict[str, Any]
    residual_norm: float
    objective: float
    iterations: int
    solve_time_s: float
    theta: tuple[float, float] | None = None
    best_iterate: dict[str, Any] = field(default_factory=dict)
    best_certificate: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    config: StochasticFOConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a notebook-friendly dictionary representation."""

        return {
            "success": self.success,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "model": self.model,
            "certificate": self.certificate,
            "residual_norm": self.residual_norm,
            "objective": self.objective,
            "iterations": self.iterations,
            "time": self.solve_time_s,
            "best_iterate": self.best_iterate,
            "best_certificate": self.best_certificate,
            "history": self.history,
        }


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "Stochastic first-order solvers require JAX. Install cumg with the "
            "'stochastic' extra, for example: pip install 'cumg[stochastic]'."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _validate_config(config: StochasticFOConfig, K: int) -> int:
    if not np.isfinite(config.kappa) or config.kappa <= 0:
        raise ValueError("kappa must be finite and positive.")
    if not np.isfinite(config.tau) or config.tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not isinstance(config.max_iter, int) or config.max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer.")
    if not np.isfinite(config.step_size) or config.step_size <= 0:
        raise ValueError("step_size must be finite and positive.")
    if not np.isfinite(config.step_decay) or config.step_decay < 0:
        raise ValueError("step_decay must be finite and nonnegative.")
    if config.batch_size is None:
        batch_size = K
    elif not isinstance(config.batch_size, int) or config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None.")
    else:
        batch_size = config.batch_size
    if config.logit_bound is not None and (
        not np.isfinite(config.logit_bound) or config.logit_bound <= 0
    ):
        raise ValueError("logit_bound must be finite and positive when provided.")
    if config.gradient_clip_norm is not None and (
        not np.isfinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be finite and positive when provided.")
    if config.theta_bounds is not None:
        try:
            lo, hi = config.theta_bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("theta_bounds must be a pair of lower and upper bounds or None.") from exc
        if lo is not None and not np.isfinite(lo):
            raise ValueError("theta_bounds lower bound must be finite or None.")
        if hi is not None and not np.isfinite(hi):
            raise ValueError("theta_bounds upper bound must be finite or None.")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("theta_bounds lower bound cannot exceed upper bound.")
    if config.record_every is not None and (
        not isinstance(config.record_every, int) or config.record_every <= 0
    ):
        raise ValueError("record_every must be a positive integer or None.")
    if config.certify_every is not None and (
        not isinstance(config.certify_every, int) or config.certify_every <= 0
    ):
        raise ValueError("certify_every must be a positive integer or None.")
    if not np.isfinite(config.regret_tolerance) or config.regret_tolerance < 0:
        raise ValueError("regret_tolerance must be finite and nonnegative.")
    return batch_size


def _validate_gamma_msd(gamma: float) -> None:
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative.")


def _validate_gamma_cvar(gamma: float, alpha: float) -> None:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")


def _initial_logits(strategy, dim: int, name: str) -> np.ndarray:
    if strategy is None:
        return np.zeros(dim, dtype=float)
    arr = np.asarray(strategy, dtype=float)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must contain nonnegative values.")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    probs = arr / total
    probs = np.maximum(probs, 1e-12)
    probs = probs / float(np.sum(probs))
    logits = np.log(probs)
    return logits - float(np.mean(logits))


def _theta0_pair(theta0, default1: float, default2: float) -> tuple[float, float]:
    if theta0 is None:
        return default1, default2
    if np.isscalar(theta0):
        theta = float(theta0)
        if not np.isfinite(theta):
            raise ValueError("theta0 must contain finite values.")
        return theta, theta
    theta_arr = np.asarray(theta0, dtype=float)
    if theta_arr.shape != (2,) or not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta0 must be a finite scalar or a pair of finite values.")
    return float(theta_arr[0]), float(theta_arr[1])


def _draw_batch(rng: np.random.Generator, K: int, batch_size: int) -> np.ndarray:
    if batch_size >= K:
        return np.arange(K, dtype=int)
    return rng.integers(0, K, size=batch_size, endpoint=False, dtype=int)


def _tree_norm(jnp, tree) -> Any:
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in tree))


def _clip_gradient(jnp, grad, max_norm: float | None):
    if max_norm is None:
        return grad
    norm = _tree_norm(jnp, grad)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return tuple(scale * leaf for leaf in grad)


def _project_msd_params(jnp, params, config: StochasticFOConfig):
    w1, w2 = params
    return (
        _project_logits(jnp, w1, config.logit_bound),
        _project_logits(jnp, w2, config.logit_bound),
    )


def _project_cvar_params(jnp, params, config: StochasticFOConfig):
    w1, w2, theta1, theta2 = params
    if config.theta_bounds is not None:
        lo, hi = config.theta_bounds
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("theta_bounds lower bound cannot exceed upper bound.")
        lower = -jnp.inf if lo is None else float(lo)
        upper = jnp.inf if hi is None else float(hi)
        theta1 = jnp.clip(theta1, lower, upper)
        theta2 = jnp.clip(theta2, lower, upper)
    return (
        _project_logits(jnp, w1, config.logit_bound),
        _project_logits(jnp, w2, config.logit_bound),
        theta1,
        theta2,
    )


def _project_logits(jnp, logits, bound: float | None):
    logits = logits - jnp.mean(logits)
    if bound is not None:
        logits = jnp.clip(logits, -float(bound), float(bound))
        logits = logits - jnp.mean(logits)
    return logits


def _jax_varphi(jnp, a, tau: float):
    return tau * jnp.logaddexp(0.0, a / tau)


def _batch_weighted_sum(jnp, values, p, batch):
    return (p.shape[0] / batch.shape[0]) * jnp.sum(p[batch] * values[batch])


def _msd_state_payoffs(jnp, A, B, x, y):
    u1 = jnp.einsum("i,kij,j->k", x, A, y)
    u2 = jnp.einsum("i,kij,j->k", x, B, y)
    return u1, u2


def _cvar_state_payoffs(jnp, A, B, x, y):
    u1 = jnp.einsum("i,kij,j->k", x, A, y)
    u2 = jnp.einsum("i,kij,j->k", x, B, y)
    return u1, u2


def _msd_rho_batch(jnp, state_payoffs, p, batch, gamma: float, tau: float):
    mean = jnp.sum(p * state_payoffs)
    sampled_smooth_downside = _batch_weighted_sum(jnp, _jax_varphi(jnp, mean - state_payoffs, tau), p, batch)
    return mean - gamma * sampled_smooth_downside


def _cvar_rho_batch(jnp, state_payoffs, p, batch, gamma: float, alpha: float, tau: float, theta):
    mean = jnp.sum(p * state_payoffs)
    sampled_tail = _batch_weighted_sum(jnp, _jax_varphi(jnp, theta - state_payoffs, tau), p, batch)
    return (1.0 - gamma) * mean + gamma * theta - (gamma / alpha) * sampled_tail


def _centered_strategy_residual(jnp, x, grad, kappa: float):
    q = grad - kappa * (jnp.log(x) + 1.0)
    return q - jnp.dot(q, x) * jnp.ones_like(q)


def _make_msd_residual_fn(jax, jnp, A, B, p, gamma: float, config: StochasticFOConfig):
    def residual(params, batch):
        w1, w2 = params
        x = jax.nn.softmax(w1)
        y = jax.nn.softmax(w2)

        def rho1_from_x(x_var):
            u1, _ = _msd_state_payoffs(jnp, A, B, x_var, y)
            return _msd_rho_batch(jnp, u1, p, batch, gamma, config.tau)

        def rho2_from_y(y_var):
            _, u2 = _msd_state_payoffs(jnp, A, B, x, y_var)
            return _msd_rho_batch(jnp, u2, p, batch, gamma, config.tau)

        g1 = jax.grad(rho1_from_x)(x)
        g2 = jax.grad(rho2_from_y)(y)
        r1 = _centered_strategy_residual(jnp, x, g1, config.kappa)
        r2 = _centered_strategy_residual(jnp, y, g2, config.kappa)
        return jnp.concatenate([r1, r2])

    return residual


def _make_cvar_residual_fn(jax, jnp, A, B, p, gamma: float, alpha: float, config: StochasticFOConfig):
    def residual(params, batch):
        w1, w2, theta1, theta2 = params
        x = jax.nn.softmax(w1)
        y = jax.nn.softmax(w2)

        def rho1_from_x(x_var):
            u1, _ = _cvar_state_payoffs(jnp, A, B, x_var, y)
            return _cvar_rho_batch(jnp, u1, p, batch, gamma, alpha, config.tau, theta1)

        def rho2_from_y(y_var):
            _, u2 = _cvar_state_payoffs(jnp, A, B, x, y_var)
            return _cvar_rho_batch(jnp, u2, p, batch, gamma, alpha, config.tau, theta2)

        def rho1_from_theta(theta_var):
            u1, _ = _cvar_state_payoffs(jnp, A, B, x, y)
            return _cvar_rho_batch(jnp, u1, p, batch, gamma, alpha, config.tau, theta_var)

        def rho2_from_theta(theta_var):
            _, u2 = _cvar_state_payoffs(jnp, A, B, x, y)
            return _cvar_rho_batch(jnp, u2, p, batch, gamma, alpha, config.tau, theta_var)

        g1 = jax.grad(rho1_from_x)(x)
        g2 = jax.grad(rho2_from_y)(y)
        h1 = jax.grad(rho1_from_theta)(theta1)
        h2 = jax.grad(rho2_from_theta)(theta2)
        r1 = _centered_strategy_residual(jnp, x, g1, config.kappa)
        r2 = _centered_strategy_residual(jnp, y, g2, config.kappa)
        return jnp.concatenate([r1, r2, jnp.array([h1, h2])])

    return residual


def _params_to_profile(jax, params):
    return np.asarray(jax.nn.softmax(params[0])), np.asarray(jax.nn.softmax(params[1]))


def _params_to_theta(params) -> tuple[float, float]:
    return float(np.asarray(params[2])), float(np.asarray(params[3]))


def _full_residual_stats(residual_fn, params, full_batch):
    residual = np.asarray(residual_fn(params, full_batch), dtype=float)
    residual_norm = float(np.linalg.norm(residual))
    objective = 0.5 * residual_norm**2
    return residual_norm, objective


def _record_history(
    history: list[dict[str, Any]],
    iteration: int,
    params,
    residual_fn,
    full_batch,
    jax,
    include_theta: bool,
) -> tuple[float, float]:
    x, y = _params_to_profile(jax, params)
    residual_norm, objective = _full_residual_stats(residual_fn, params, full_batch)
    row: dict[str, Any] = {
        "iteration": iteration,
        "residual_norm": residual_norm,
        "objective": objective,
        "x": x,
        "y": y,
    }
    if include_theta:
        row["theta"] = _params_to_theta(params)
    history.append(row)
    return residual_norm, objective


def _maybe_update_best(best, iteration: int, params, residual_norm: float, objective: float, jax, include_theta: bool):
    if best is not None and objective >= best["objective"]:
        return best
    x, y = _params_to_profile(jax, params)
    out = {
        "iteration": iteration,
        "residual_norm": residual_norm,
        "objective": objective,
        "x": x,
        "y": y,
    }
    if include_theta:
        out["theta"] = _params_to_theta(params)
    return out


def _certify_checkpoint(
    certifier: Callable[[Any], dict[str, Any]] | None,
    iteration: int,
    params,
    residual_norm: float,
    objective: float,
    jax,
    include_theta: bool,
) -> dict[str, Any] | None:
    if certifier is None:
        return None
    x, y = _params_to_profile(jax, params)
    cert = certifier(params)
    out: dict[str, Any] = {
        "iteration": iteration,
        "residual_norm": residual_norm,
        "objective": objective,
        "x": x,
        "y": y,
        "certificate": cert,
        "eta": float(cert["eta"]),
    }
    if include_theta:
        out["theta"] = _params_to_theta(params)
    return out


def _maybe_update_best_certificate(best, checkpoint: dict[str, Any] | None):
    if checkpoint is None:
        return best
    if best is None or checkpoint["eta"] < best["eta"]:
        return checkpoint
    return best


def _run_stochastic_fo(
    jax,
    jnp,
    params,
    residual_fn,
    project_fn,
    config: StochasticFOConfig,
    K: int,
    certifier: Callable[[Any], dict[str, Any]] | None = None,
):
    rng = np.random.default_rng(config.seed)
    batch_size = _validate_config(config, K)
    full_batch = jnp.arange(K)
    history: list[dict[str, Any]] = []
    residual_norm, objective = _full_residual_stats(residual_fn, params, full_batch)
    include_theta = len(params) == 4
    best = _maybe_update_best(None, 0, params, residual_norm, objective, jax, include_theta)
    best_certificate = None
    completed_iterations = 0
    if config.certify_every is not None:
        checkpoint = _certify_checkpoint(certifier, 0, params, residual_norm, objective, jax, include_theta)
        best_certificate = _maybe_update_best_certificate(best_certificate, checkpoint)
        if checkpoint is not None and checkpoint["eta"] <= config.regret_tolerance:
            if config.record_every is not None:
                _record_history(history, 0, params, residual_fn, full_batch, jax, include_theta)
            return params, best, best_certificate, history, completed_iterations
    if config.record_every is not None:
        _record_history(history, 0, params, residual_fn, full_batch, jax, include_theta)

    for iteration in range(config.max_iter):
        batch1 = jnp.asarray(_draw_batch(rng, K, batch_size))
        batch2 = jnp.asarray(_draw_batch(rng, K, batch_size))
        residual2 = residual_fn(params, batch2)
        _, pullback = jax.vjp(lambda z, batch=batch1: residual_fn(z, batch), params)
        grad = pullback(residual2)[0]
        grad = _clip_gradient(jnp, grad, config.gradient_clip_norm)
        step = config.step_size / ((iteration + 1) ** config.step_decay)
        params = tuple(param - step * update for param, update in zip(params, grad, strict=True))
        params = project_fn(jnp, params, config)

        should_check = (
            iteration + 1 == config.max_iter
            or (config.record_every is not None and (iteration + 1) % config.record_every == 0)
            or (config.certify_every is not None and (iteration + 1) % config.certify_every == 0)
        )
        if should_check:
            residual_norm, objective = _full_residual_stats(residual_fn, params, full_batch)
            completed_iterations = iteration + 1
            best = _maybe_update_best(best, completed_iterations, params, residual_norm, objective, jax, include_theta)
            if config.record_every is not None and (iteration + 1) % config.record_every == 0:
                _record_history(history, completed_iterations, params, residual_fn, full_batch, jax, include_theta)
            should_certify = config.certify_every is not None and (
                completed_iterations % config.certify_every == 0 or completed_iterations == config.max_iter
            )
            if should_certify:
                checkpoint = _certify_checkpoint(
                    certifier,
                    completed_iterations,
                    params,
                    residual_norm,
                    objective,
                    jax,
                    include_theta,
                )
                best_certificate = _maybe_update_best_certificate(best_certificate, checkpoint)
                if checkpoint is not None and checkpoint["eta"] <= config.regret_tolerance:
                    break

    assert best is not None
    return params, best, best_certificate, history, completed_iterations


def solve_msd_stochastic_fo(
    A_list,
    B_list,
    p=None,
    gamma: float = 0.0,
    config: StochasticFOConfig | None = None,
) -> StochasticFOResult:
    """Run stochastic first-order error minimization for a smoothed MSD game."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _validate_gamma_msd(gamma)
    config = config or StochasticFOConfig()
    batch_size = _validate_config(config, A.shape[0])
    del batch_size
    jax, jnp = _require_jax()
    w1 = jnp.asarray(_initial_logits(config.x0, A.shape[1], "x0"))
    w2 = jnp.asarray(_initial_logits(config.y0, A.shape[2], "y0"))
    params = _project_msd_params(jnp, (w1, w2), config)
    A_j = jnp.asarray(A)
    B_j = jnp.asarray(B)
    p_j = jnp.asarray(p)
    residual_fn = _make_msd_residual_fn(jax, jnp, A_j, B_j, p_j, gamma, config)

    def certifier(candidate_params):
        candidate_x, candidate_y = _params_to_profile(jax, candidate_params)
        return full_msd_regret(A, B, p, gamma, candidate_x, candidate_y)

    started = time.perf_counter()
    params, best, best_certificate, history, completed_iterations = _run_stochastic_fo(
        jax,
        jnp,
        params,
        residual_fn,
        _project_msd_params,
        config,
        A.shape[0],
        certifier,
    )
    elapsed = time.perf_counter() - started
    x, y = _params_to_profile(jax, params)
    residual_norm, objective = _full_residual_stats(residual_fn, params, jnp.arange(A.shape[0]))
    cert = full_msd_regret(A, B, p, gamma, x, y)
    return StochasticFOResult(
        success=bool(np.isfinite(cert["eta"]) and cert["eta"] <= config.regret_tolerance),
        x=x,
        y=y,
        model="MSD stochastic FO",
        certificate=cert,
        residual_norm=residual_norm,
        objective=objective,
        iterations=completed_iterations,
        solve_time_s=elapsed,
        best_iterate=best,
        best_certificate=best_certificate,
        history=history,
        config=config,
    )


def solve_cvar_stochastic_fo(
    A_list,
    B_list,
    p=None,
    gamma: float = 0.0,
    alpha: float = 0.5,
    config: StochasticFOConfig | None = None,
) -> StochasticFOResult:
    """Run stochastic first-order error minimization for a smoothed CVaR game."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    _validate_gamma_cvar(gamma, alpha)
    config = config or StochasticFOConfig()
    batch_size = _validate_config(config, A.shape[0])
    del batch_size
    jax, jnp = _require_jax()
    w1_np = _initial_logits(config.x0, A.shape[1], "x0")
    w2_np = _initial_logits(config.y0, A.shape[2], "y0")
    x0 = np.exp(w1_np) / float(np.sum(np.exp(w1_np)))
    y0 = np.exp(w2_np) / float(np.sum(np.exp(w2_np)))
    u1_init = np.einsum("i,kij,j->k", x0, A, y0)
    u2_init = np.einsum("i,kij,j->k", x0, B, y0)
    theta1, theta2 = _theta0_pair(config.theta0, float(p @ u1_init), float(p @ u2_init))
    params = _project_cvar_params(
        jnp,
        (jnp.asarray(w1_np), jnp.asarray(w2_np), jnp.asarray(theta1), jnp.asarray(theta2)),
        config,
    )
    A_j = jnp.asarray(A)
    B_j = jnp.asarray(B)
    p_j = jnp.asarray(p)
    residual_fn = _make_cvar_residual_fn(jax, jnp, A_j, B_j, p_j, gamma, alpha, config)

    def certifier(candidate_params):
        candidate_x, candidate_y = _params_to_profile(jax, candidate_params)
        return full_cvar_regret(A, B, p, gamma, alpha, candidate_x, candidate_y)

    started = time.perf_counter()
    params, best, best_certificate, history, completed_iterations = _run_stochastic_fo(
        jax,
        jnp,
        params,
        residual_fn,
        _project_cvar_params,
        config,
        A.shape[0],
        certifier,
    )
    elapsed = time.perf_counter() - started
    x, y = _params_to_profile(jax, params)
    theta = _params_to_theta(params)
    residual_norm, objective = _full_residual_stats(residual_fn, params, jnp.arange(A.shape[0]))
    cert = full_cvar_regret(A, B, p, gamma, alpha, x, y)
    return StochasticFOResult(
        success=bool(np.isfinite(cert["eta"]) and cert["eta"] <= config.regret_tolerance),
        x=x,
        y=y,
        theta=theta,
        model="CVaR stochastic FO",
        certificate=cert,
        residual_norm=residual_norm,
        objective=objective,
        iterations=completed_iterations,
        solve_time_s=elapsed,
        best_iterate=best,
        best_certificate=best_certificate,
        history=history,
        config=config,
    )
