"""Optional Mystic-based seeding for the MSD MCP residual."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from .mcp import solve_pyomo_mcp_model
from .msd import build_msd_mcp_model
from .results import SolverResult
from .validation import normalize_game_inputs


def fb(a, b):
    """Fischer-Burmeister complementarity residual."""

    return np.sqrt(a * a + b * b) - (a + b)


def pack_vars(x, y, lam1, z1, lam2, z2, alpha1, alpha2):
    """Pack MSD residual variables into one vector."""

    return np.concatenate([x, y, lam1, z1, lam2, z2, np.array([alpha1, alpha2])])


def unpack_vars(v, n1: int, n2: int, K: int):
    """Unpack an MSD residual vector."""

    x = v[0:n1]
    y = v[n1 : n1 + n2]
    lam1 = v[n1 + n2 : n1 + n2 + K]
    z1 = v[n1 + n2 + K : n1 + n2 + 2 * K]
    lam2 = v[n1 + n2 + 2 * K : n1 + n2 + 3 * K]
    z2 = v[n1 + n2 + 3 * K : n1 + n2 + 4 * K]
    alpha1 = v[-2]
    alpha2 = v[-1]
    return x, y, lam1, z1, lam2, z2, alpha1, alpha2


def mcp_residual(v, A_list, B_list, p, gamma: float) -> np.ndarray:
    """Fischer-Burmeister residual for the MSD MCP."""

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    x, y, lam1, z1, lam2, z2, alpha1, alpha2 = unpack_vars(v, n1, n2, K)
    Abar = np.einsum("k,kij->ij", p, A)
    Bbar = np.einsum("k,kij->ij", p, B)
    V1 = Abar.copy()
    V2 = Bbar.copy()
    for k in range(K):
        V1 += lam1[k] * (A[k] - Abar)
        V2 += lam2[k] * (B[k] - Bbar)
    v1 = V1 @ y
    v2 = V2.T @ x
    r1 = np.array([x @ ((A[k] - Abar) @ y) for k in range(K)])
    r2 = np.array([x @ ((B[k] - Bbar) @ y) for k in range(K)])
    res = []
    res.extend(fb(x, alpha1 - v1))
    res.extend(fb(y, alpha2 - v2))
    res.extend(fb(lam1, r1 - z1))
    res.extend(fb(gamma * p - lam1, -z1))
    res.extend(fb(lam2, r2 - z2))
    res.extend(fb(gamma * p - lam2, -z2))
    res.append(x.sum() - 1.0)
    res.append(y.sum() - 1.0)
    return np.array(res, dtype=float)


def residual_objective(v, A_list, B_list, p, gamma: float, w_eq: float = 10.0) -> float:
    """Squared residual objective with simple bound penalties."""

    A, _, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    x, y, lam1, z1, lam2, z2, _, _ = unpack_vars(v, n1, n2, K)
    residual = mcp_residual(v, A_list, B_list, p, gamma)
    penalty = 0.0
    penalty += np.sum(np.clip(-x, 0, None) ** 2) + np.sum(np.clip(-y, 0, None) ** 2)
    penalty += np.sum(np.clip(-lam1, 0, None) ** 2) + np.sum(np.clip(lam1 - gamma * p, 0, None) ** 2)
    penalty += np.sum(np.clip(-lam2, 0, None) ** 2) + np.sum(np.clip(lam2 - gamma * p, 0, None) ** 2)
    penalty += np.sum(np.clip(z1, 0, None) ** 2) + np.sum(np.clip(z2, 0, None) ** 2)
    return float(residual @ residual + w_eq * penalty)


def run_mystic_seed(A_list, B_list, p, gamma: float, maxiter: int = 2000, popsize: int = 80, seed: int = 13):
    """Generate an MSD MCP seed with Mystic's differential evolution solver."""

    try:
        from mystic.monitors import VerboseMonitor
        from mystic.solvers import diffev2
        from mystic.termination import ChangeOverGeneration
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install cumg[mystic] to use run_mystic_seed.") from exc

    A, B, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    payoff_scale = max(1.0, float(max(np.max(np.abs(A)), np.max(np.abs(B)))))
    alpha_bound = (1.0 + max(0.0, gamma)) * payoff_scale
    lb = np.array(
        [0.0] * n1
        + [0.0] * n2
        + [0.0] * K
        + [-2.0 * payoff_scale] * K
        + [0.0] * K
        + [-2.0 * payoff_scale] * K
        + [-alpha_bound, -alpha_bound]
    )
    ub = np.array(
        [1.0] * n1 + [1.0] * n2 + list(gamma * p) + [0.0] * K + list(gamma * p) + [0.0] * K + [alpha_bound, alpha_bound]
    )
    rng = np.random.default_rng(seed)
    x0 = lb + (ub - lb) * rng.random(lb.size)

    def cost(v):
        return residual_objective(v, A_list, B_list, p, gamma)

    return diffev2(
        cost,
        x0=x0,
        bounds=list(zip(lb, ub, strict=True)),
        maxiter=maxiter,
        popsize=popsize,
        ftol=1e-10,
        gtol=50,
        itermon=VerboseMonitor(50),
        termination=ChangeOverGeneration(tolerance=1e-10, generations=50),
    )


def run_mystic_then_pyomo(A_list, B_list, p, gamma: float, **solve_kwargs):
    """Generate a Mystic seed, then solve the Pyomo MSD MCP."""

    seed = run_mystic_seed(A_list, B_list, p, gamma)
    return refine_with_pyomo_seed(seed, A_list, B_list, p, gamma, **solve_kwargs)


def refine_with_pyomo_seed(
    v_seed,
    A_list,
    B_list,
    p,
    gamma: float,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict | None = None,
    tee: bool = False,
) -> SolverResult:
    """Initialize the MSD Pyomo model from a packed seed vector, then solve it."""

    A, _, p = normalize_game_inputs(A_list, B_list, p)
    K, n1, n2 = A.shape
    x, y, lam1, z1, lam2, z2, alpha1, alpha2 = unpack_vars(v_seed, n1, n2, K)
    model = build_msd_mcp_model(A_list, B_list, p, gamma)

    for i in range(n1):
        model.x[i].value = float(np.clip(x[i], 0.0, 1.0))
    for j in range(n2):
        model.y[j].value = float(np.clip(y[j], 0.0, 1.0))
    for k in range(K):
        model.lam1[k].value = float(np.clip(lam1[k], 0.0, gamma * p[k]))
        model.lam2[k].value = float(np.clip(lam2[k], 0.0, gamma * p[k]))
        model.z1[k].value = float(min(z1[k], 0.0))
        model.z2[k].value = float(min(z2[k], 0.0))
    model.alpha1.value = float(alpha1)
    model.alpha2.value = float(alpha2)

    raw, elapsed, used_solver, solved_model = solve_pyomo_mcp_model(
        model,
        solver=solver,
        fallback_solver=fallback_solver,
        solver_options=solver_options,
        tee=tee,
    )
    return SolverResult(
        x=np.array([pyo.value(solved_model.x[i]) for i in solved_model.I], dtype=float),
        y=np.array([pyo.value(solved_model.y[j]) for j in solved_model.J], dtype=float),
        alpha1=float(pyo.value(solved_model.alpha1)),
        alpha2=float(pyo.value(solved_model.alpha2)),
        lam1=np.array([pyo.value(solved_model.lam1[k]) for k in solved_model.K], dtype=float),
        lam2=np.array([pyo.value(solved_model.lam2[k]) for k in solved_model.K], dtype=float),
        z1=np.array([pyo.value(solved_model.z1[k]) for k in solved_model.K], dtype=float),
        z2=np.array([pyo.value(solved_model.z2[k]) for k in solved_model.K], dtype=float),
        model="MSD",
        solver=used_solver,
        raw_result=raw,
        solve_time_s=elapsed,
    )
