"""CVaR MCP formulation and payoff utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements

from .mcp import solve_pyomo_mcp_model
from .results import SolverResult
from .validation import as_matrix_lists, normalize_game_inputs, normalize_probabilities


def build_cvar_mcp_model(A_list, B_list, p=None, gamma: float = 0.0, alpha: float = 0.5) -> pyo.ConcreteModel:
    """Build the Pyomo MCP model for a lower-tail CVaR risk-aware bimatrix game."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    A_list, B_list = as_matrix_lists(A, B)
    K, n1, n2 = A.shape

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, n1 - 1)
    model.J = pyo.RangeSet(0, n2 - 1)
    model.K = pyo.RangeSet(0, K - 1)

    def to_param_3d(mats):
        return {(k, i, j): float(mats[k][i, j]) for k in range(K) for i in range(n1) for j in range(n2)}

    model.p = pyo.Param(model.K, initialize={k: float(p[k]) for k in range(K)}, within=pyo.NonNegativeReals)
    model.gamma = pyo.Param(initialize=float(gamma), within=pyo.NonNegativeReals)
    model.alpha = pyo.Param(initialize=float(alpha), within=pyo.PositiveReals)
    model.A = pyo.Param(model.K, model.I, model.J, initialize=to_param_3d(A_list))
    model.B = pyo.Param(model.K, model.I, model.J, initialize=to_param_3d(B_list))
    model.Abar = pyo.Param(
        model.I,
        model.J,
        initialize={(i, j): sum(p[k] * A[k, i, j] for k in range(K)) for i in range(n1) for j in range(n2)},
    )
    model.Bbar = pyo.Param(
        model.I,
        model.J,
        initialize={(i, j): sum(p[k] * B[k, i, j] for k in range(K)) for i in range(n1) for j in range(n2)},
    )

    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=1.0 / n1)
    model.y = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0 / n2)
    model.alpha1 = pyo.Var(domain=pyo.Reals)
    model.alpha2 = pyo.Var(domain=pyo.Reals)
    model.lam1 = pyo.Var(
        model.K,
        bounds=lambda m, k: (0.0, m.gamma * m.p[k] / m.alpha),
        initialize=lambda m, k: pyo.value(m.gamma * m.p[k]),
    )
    model.lam2 = pyo.Var(
        model.K,
        bounds=lambda m, k: (0.0, m.gamma * m.p[k] / m.alpha),
        initialize=lambda m, k: pyo.value(m.gamma * m.p[k]),
    )
    model.nu1 = pyo.Var(model.K, domain=pyo.NonNegativeReals)
    model.nu2 = pyo.Var(model.K, domain=pyo.NonNegativeReals)
    model.z1 = pyo.Var(domain=pyo.Reals)
    model.z2 = pyo.Var(domain=pyo.Reals)

    model.simplex_x = pyo.Constraint(expr=sum(model.x[i] for i in model.I) == 1.0)
    model.simplex_y = pyo.Constraint(expr=sum(model.y[j] for j in model.J) == 1.0)
    model.cvar_x = pyo.Constraint(expr=sum(model.lam1[k] for k in model.K) == model.gamma)
    model.cvar_y = pyo.Constraint(expr=sum(model.lam2[k] for k in model.K) == model.gamma)

    def v1_rule(m, i):
        base = sum(m.Abar[i, j] * m.y[j] for j in m.J)
        tail = sum(m.lam1[k] * sum(m.A[k, i, j] * m.y[j] for j in m.J) for k in m.K)
        return (1 - m.gamma) * base + tail

    def v2_rule(m, j):
        base = sum(m.x[i] * m.Bbar[i, j] for i in m.I)
        tail = sum(m.lam2[k] * sum(m.x[i] * m.B[k, i, j] for i in m.I) for k in m.K)
        return (1 - m.gamma) * base + tail

    def r1_rule(m, k):
        return sum(m.x[i] * sum(m.A[k, i, j] * m.y[j] for j in m.J) for i in m.I) - m.z1

    def r2_rule(m, k):
        return sum(m.x[i] * sum(m.B[k, i, j] * m.y[j] for j in m.J) for i in m.I) - m.z2

    model.v1 = pyo.Expression(model.I, rule=v1_rule)
    model.v2 = pyo.Expression(model.J, rule=v2_rule)
    model.r1 = pyo.Expression(model.K, rule=r1_rule)
    model.r2 = pyo.Expression(model.K, rule=r2_rule)

    model.comp_x = Complementarity(model.I, rule=lambda m, i: complements(m.x[i] >= 0, m.alpha1 - m.v1[i] >= 0))
    model.comp_y = Complementarity(model.J, rule=lambda m, j: complements(m.y[j] >= 0, m.alpha2 - m.v2[j] >= 0))
    model.comp_r1 = Complementarity(model.K, rule=lambda m, k: complements(m.lam1[k] >= 0, m.r1[k] + m.nu1[k] >= 0))
    model.comp_z1 = Complementarity(
        model.K, rule=lambda m, k: complements(m.gamma * m.p[k] / m.alpha - m.lam1[k] >= 0, m.nu1[k] >= 0)
    )
    model.comp_r2 = Complementarity(model.K, rule=lambda m, k: complements(m.lam2[k] >= 0, m.r2[k] + m.nu2[k] >= 0))
    model.comp_z2 = Complementarity(
        model.K, rule=lambda m, k: complements(m.gamma * m.p[k] / m.alpha - m.lam2[k] >= 0, m.nu2[k] >= 0)
    )
    return model


def _extract_cvar_result(model, raw_result, solve_time_s, solver_name: str) -> SolverResult:
    return SolverResult(
        x=np.array([pyo.value(model.x[i]) for i in model.I], dtype=float),
        y=np.array([pyo.value(model.y[j]) for j in model.J], dtype=float),
        alpha1=float(pyo.value(model.alpha1)),
        alpha2=float(pyo.value(model.alpha2)),
        lam1=np.array([pyo.value(model.lam1[k]) for k in model.K], dtype=float),
        lam2=np.array([pyo.value(model.lam2[k]) for k in model.K], dtype=float),
        z1=float(pyo.value(model.z1)),
        z2=float(pyo.value(model.z2)),
        model="CVaR",
        solver=solver_name,
        raw_result=raw_result,
        solve_time_s=solve_time_s,
        extra={
            "alpha": float(pyo.value(model.alpha)),
            "nu1": np.array([pyo.value(model.nu1[k]) for k in model.K], dtype=float),
            "nu2": np.array([pyo.value(model.nu2[k]) for k in model.K], dtype=float),
        },
    )


def solve_cvar_mcp(
    A_list,
    B_list,
    p=None,
    gamma: float = 0.0,
    alpha: float = 0.5,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
) -> SolverResult:
    """Build and solve the CVaR MCP formulation."""

    model = build_cvar_mcp_model(A_list, B_list, p, gamma, alpha)
    raw, elapsed, used_solver, solved_model = solve_pyomo_mcp_model(
        model,
        solver=solver,
        fallback_solver=fallback_solver,
        solver_options=solver_options,
        tee=tee,
    )
    return _extract_cvar_result(solved_model, raw, elapsed, used_solver)


def cvar_tail_weights(payoffs, p, gamma: float, alpha: float) -> np.ndarray:
    """Lower-tail CVaR weights used by the MCP model."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    payoffs = np.asarray(payoffs, dtype=float)
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("payoffs must contain finite values.")
    p = normalize_probabilities(p, payoffs.shape[0])
    lam = np.zeros_like(p)
    if gamma <= 1e-12:
        return lam

    cap = gamma * p / alpha
    remaining = float(gamma)
    for k in np.argsort(payoffs):
        take = min(float(cap[k]), remaining)
        lam[k] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    return lam


def cvar_value_from_state_payoffs(payoffs, p, gamma: float, alpha: float) -> float:
    """Evaluate the lower-tail CVaR-adjusted payoff."""

    payoffs = np.asarray(payoffs, dtype=float)
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("payoffs must contain finite values.")
    p = normalize_probabilities(p, payoffs.shape[0])
    return (1.0 - gamma) * float(p @ payoffs) + float(cvar_tail_weights(payoffs, p, gamma, alpha) @ payoffs)


def cvar_profile_values(A_list, B_list, p, gamma: float, alpha: float, x, y) -> dict[str, np.ndarray | float]:
    """Evaluate CVaR payoffs for a fixed mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != (A.shape[1],) or y.shape != (A.shape[2],):
        raise ValueError(f"x and y must have shapes ({A.shape[1]},) and ({A.shape[2]},).")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain finite strategy values.")
    u1_states = np.einsum("i,kij,j->k", x, A, y)
    u2_states = np.einsum("i,kij,j->k", x, B, y)
    return {
        "rho1": cvar_value_from_state_payoffs(u1_states, p, gamma, alpha),
        "rho2": cvar_value_from_state_payoffs(u2_states, p, gamma, alpha),
        "u1_states": u1_states,
        "u2_states": u2_states,
    }
