"""Mean-semi-deviation MCP formulation for two-player bimatrix games."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements

from .mcp import solve_pyomo_mcp_model
from .results import SolverResult
from .validation import as_matrix_lists, normalize_game_inputs


def build_msd_mcp_model(A_list, B_list, p=None, gamma: float = 0.0) -> pyo.ConcreteModel:
    """Build the Pyomo MCP model for an MSD risk-aware bimatrix game."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    if gamma < 0:
        raise ValueError("gamma must be nonnegative.")
    A_list, B_list = as_matrix_lists(A, B)
    K, n1, n2 = A.shape

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, n1 - 1)
    model.J = pyo.RangeSet(0, n2 - 1)
    model.K = pyo.RangeSet(0, K - 1)

    def to_param_3d(mats):
        return {
            (k, i, j): float(mats[k][i, j])
            for k in range(K)
            for i in range(n1)
            for j in range(n2)
        }

    model.p = pyo.Param(model.K, initialize={k: float(p[k]) for k in range(K)}, within=pyo.NonNegativeReals)
    model.gamma = pyo.Param(initialize=float(gamma), within=pyo.NonNegativeReals)
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
    model.lam1 = pyo.Var(model.K, bounds=lambda m, k: (0.0, m.gamma * m.p[k]))
    model.lam2 = pyo.Var(model.K, bounds=lambda m, k: (0.0, m.gamma * m.p[k]))
    model.z1 = pyo.Var(model.K, bounds=(None, 0.0))
    model.z2 = pyo.Var(model.K, bounds=(None, 0.0))

    model.simplex_x = pyo.Constraint(expr=sum(model.x[i] for i in model.I) == 1.0)
    model.simplex_y = pyo.Constraint(expr=sum(model.y[j] for j in model.J) == 1.0)

    def v1_rule(m, i):
        base = sum(m.Abar[i, j] * m.y[j] for j in m.J)
        dev = sum(m.lam1[k] * sum((m.A[k, i, j] - m.Abar[i, j]) * m.y[j] for j in m.J) for k in m.K)
        return base + dev

    def v2_rule(m, j):
        base = sum(m.x[i] * m.Bbar[i, j] for i in m.I)
        dev = sum(m.lam2[k] * sum(m.x[i] * (m.B[k, i, j] - m.Bbar[i, j]) for i in m.I) for k in m.K)
        return base + dev

    def r1_rule(m, k):
        return sum(m.x[i] * sum((m.A[k, i, j] - m.Abar[i, j]) * m.y[j] for j in m.J) for i in m.I)

    def r2_rule(m, k):
        return sum(m.x[i] * sum((m.B[k, i, j] - m.Bbar[i, j]) * m.y[j] for j in m.J) for i in m.I)

    model.v1 = pyo.Expression(model.I, rule=v1_rule)
    model.v2 = pyo.Expression(model.J, rule=v2_rule)
    model.r1 = pyo.Expression(model.K, rule=r1_rule)
    model.r2 = pyo.Expression(model.K, rule=r2_rule)

    model.comp_x = Complementarity(model.I, rule=lambda m, i: complements(m.x[i] >= 0, m.alpha1 - m.v1[i] >= 0))
    model.comp_y = Complementarity(model.J, rule=lambda m, j: complements(m.y[j] >= 0, m.alpha2 - m.v2[j] >= 0))
    model.comp_r1 = Complementarity(model.K, rule=lambda m, k: complements(m.lam1[k] >= 0, m.r1[k] - m.z1[k] >= 0))
    model.comp_z1 = Complementarity(
        model.K, rule=lambda m, k: complements(m.gamma * m.p[k] - m.lam1[k] >= 0, -m.z1[k] >= 0)
    )
    model.comp_r2 = Complementarity(model.K, rule=lambda m, k: complements(m.lam2[k] >= 0, m.r2[k] - m.z2[k] >= 0))
    model.comp_z2 = Complementarity(
        model.K, rule=lambda m, k: complements(m.gamma * m.p[k] - m.lam2[k] >= 0, -m.z2[k] >= 0)
    )
    return model


def _extract_msd_result(model, raw_result, solve_time_s, solver_name: str) -> SolverResult:
    return SolverResult(
        x=np.array([pyo.value(model.x[i]) for i in model.I], dtype=float),
        y=np.array([pyo.value(model.y[j]) for j in model.J], dtype=float),
        alpha1=float(pyo.value(model.alpha1)),
        alpha2=float(pyo.value(model.alpha2)),
        lam1=np.array([pyo.value(model.lam1[k]) for k in model.K], dtype=float),
        lam2=np.array([pyo.value(model.lam2[k]) for k in model.K], dtype=float),
        z1=np.array([pyo.value(model.z1[k]) for k in model.K], dtype=float),
        z2=np.array([pyo.value(model.z2[k]) for k in model.K], dtype=float),
        model="MSD",
        solver=solver_name,
        raw_result=raw_result,
        solve_time_s=solve_time_s,
    )


def solve_msd_mcp(
    A_list,
    B_list,
    p=None,
    gamma: float = 0.0,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
) -> SolverResult:
    """Build and solve the MSD MCP formulation."""

    model = build_msd_mcp_model(A_list, B_list, p, gamma)
    raw, elapsed, used_solver, solved_model = solve_pyomo_mcp_model(
        model,
        solver=solver,
        fallback_solver=fallback_solver,
        solver_options=solver_options,
        tee=tee,
    )
    return _extract_msd_result(solved_model, raw, elapsed, used_solver)


def msd_value_from_state_payoffs(state_payoffs, p, gamma: float) -> float:
    """Evaluate ``mean + gamma * E[min(0, payoff - mean)]``."""

    state_payoffs = np.asarray(state_payoffs, dtype=float)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    mean = float(p @ state_payoffs)
    downside = np.minimum(0.0, state_payoffs - mean)
    return mean + gamma * float(p @ downside)


def msd_profile_values(A_list, B_list, p, gamma: float, x, y) -> dict[str, np.ndarray | float]:
    """Evaluate MSD payoffs for a fixed mixed profile."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u1_states = np.einsum("i,kij,j->k", x, A, y)
    u2_states = np.einsum("i,kij,j->k", x, B, y)
    return {
        "rho1": msd_value_from_state_payoffs(u1_states, p, gamma),
        "rho2": msd_value_from_state_payoffs(u2_states, p, gamma),
        "u1_states": u1_states,
        "u2_states": u2_states,
    }

