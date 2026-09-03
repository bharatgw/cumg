# pyomo_msd_mcp.py
# Two-player mean-semi-deviation (MSD) risk-aware bimatrix game as a Multilinear Complementarity Problem (MCP)
# Requires: pyomo (>=6), plus PATH ('path' or 'pathmcp') or IPOPT with mpec.simple_nonlinear transform.
# Usage:
#   python pyomo_msd_mcp.py   # runs the demo instance
# Or import build_and_solve(A_list,B_list,p,gamma,solver='path') in your project.

import numpy as np
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements


def build_model(A_list: list[np.ndarray], B_list: list[np.ndarray], p: np.ndarray, gamma: float) -> pyo.ConcreteModel:
    K = len(A_list)
    n1, n2 = A_list[0].shape
    assert all(Ak.shape == (n1, n2) for Ak in A_list)
    assert all(Bk.shape == (n1, n2) for Bk in B_list)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()

    m = pyo.ConcreteModel()

    # Sets
    m.I = pyo.RangeSet(0, n1 - 1)
    m.J = pyo.RangeSet(0, n2 - 1)
    m.K = pyo.RangeSet(0, K - 1)

    # Params (store as dicts for Pyomo)
    def to_param_3d(lst):
        data = {}
        for k, M in enumerate(lst):
            for i in range(n1):
                for j in range(n2):
                    data[(k, i, j)] = float(M[i, j])
        return data

    m.p = pyo.Param(m.K, initialize={k: float(p[k]) for k in range(K)}, within=pyo.NonNegativeReals)
    m.gamma = pyo.Param(initialize=float(gamma), within=pyo.NonNegativeReals)
    m.A = pyo.Param(m.K, m.I, m.J, initialize=to_param_3d(A_list))
    m.B = pyo.Param(m.K, m.I, m.J, initialize=to_param_3d(B_list))

    # Abar, Bbar
    def Abar_rule(m, i, j):
        return sum(m.p[k] * m.A[k, i, j] for k in m.K)

    def Bbar_rule(m, i, j):
        return sum(m.p[k] * m.B[k, i, j] for k in m.K)

    m.Abar = pyo.Param(m.I, m.J, initialize={(i, j): Abar_rule(m, i, j) for i in range(n1) for j in range(n2)})
    m.Bbar = pyo.Param(m.I, m.J, initialize={(i, j): Bbar_rule(m, i, j) for i in range(n1) for j in range(n2)})

    # Variables
    m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals)  # strategies
    m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    m.alpha1 = pyo.Var(domain=pyo.Reals)  # values
    m.alpha2 = pyo.Var(domain=pyo.Reals)
    m.lam1 = pyo.Var(m.K, bounds=lambda m, k: (0.0, pyo.value(m.gamma) * pyo.value(m.p[k])))
    m.lam2 = pyo.Var(m.K, bounds=lambda m, k: (0.0, pyo.value(m.gamma) * pyo.value(m.p[k])))
    m.z1 = pyo.Var(m.K, bounds=(None, 0.0))  # z <= 0
    m.z2 = pyo.Var(m.K, bounds=(None, 0.0))

    # Simplex equalities
    m.simplex_x = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1.0)
    m.simplex_y = pyo.Constraint(expr=sum(m.y[j] for j in m.J) == 1.0)

    # Helper expressions
    # Player 1 row values: v1[i] = Abar[i,:] y + sum_k lam1[k]*(A^k - Abar)[i,:] y
    def v1_rule(m, i):
        base = sum(m.Abar[i, j] * m.y[j] for j in m.J)
        dev = sum(m.lam1[k] * sum((m.A[k, i, j] - m.Abar[i, j]) * m.y[j] for j in m.J) for k in m.K)
        return base + dev

    m.v1 = pyo.Expression(m.I, rule=v1_rule)

    # Player 2 column values: v2[j] = x^T Bbar[:,j] + sum_k lam2[k]* x^T (B^k - Bbar)[:,j]
    def v2_rule(m, j):
        base = sum(m.x[i] * m.Bbar[i, j] for i in m.I)
        dev = sum(m.lam2[k] * sum(m.x[i] * (m.B[k, i, j] - m.Bbar[i, j]) for i in m.I) for k in m.K)
        return base + dev

    m.v2 = pyo.Expression(m.J, rule=v2_rule)

    # Scenario residuals:
    # r1[k] = x^T (A^k - Abar) y
    def r1_rule(m, k):
        return sum(m.x[i] * sum((m.A[k, i, j] - m.Abar[i, j]) * m.y[j] for j in m.J) for i in m.I)

    m.r1 = pyo.Expression(m.K, rule=r1_rule)

    # r2[k] = x^T (B^k - Bbar) y
    def r2_rule(m, k):
        return sum(m.x[i] * sum((m.B[k, i, j] - m.Bbar[i, j]) * m.y[j] for j in m.J) for i in m.I)

    m.r2 = pyo.Expression(m.K, rule=r2_rule)

    # Complementarity conditions
    # 0 <= x_i ⟂ alpha1 - v1_i >= 0
    m.comp_x = Complementarity(m.I, rule=lambda m, i: complements(m.x[i] >= 0, m.alpha1 - m.v1[i] >= 0))
    # 0 <= y_j ⟂ alpha2 - v2_j >= 0
    m.comp_y = Complementarity(m.J, rule=lambda m, j: complements(m.y[j] >= 0, m.alpha2 - m.v2[j] >= 0))
    # 0 <= lam1_k ⟂ r1_k - z1_k >= 0 ; 0 <= gamma p_k - lam1_k ⟂ -z1_k >= 0
    m.comp_r1 = Complementarity(m.K, rule=lambda m, k: complements(m.lam1[k] >= 0, m.r1[k] - m.z1[k] >= 0))
    m.comp_z1 = Complementarity(m.K, rule=lambda m, k: complements(m.gamma * m.p[k] - m.lam1[k] >= 0, -m.z1[k] >= 0))
    # Player 2 analogues
    m.comp_r2 = Complementarity(m.K, rule=lambda m, k: complements(m.lam2[k] >= 0, m.r2[k] - m.z2[k] >= 0))
    m.comp_z2 = Complementarity(m.K, rule=lambda m, k: complements(m.gamma * m.p[k] - m.lam2[k] >= 0, -m.z2[k] >= 0))

    return m


def solve_model(m: pyo.ConcreteModel, solver: str = "path"):
    opt = pyo.SolverFactory(solver)
    if opt.available(exception_flag=False):
        res = opt.solve(m, tee=True)
        return res
    # Fallback NLP transform
    pyo.TransformationFactory("mpec.simple_nonlinear").apply_to(m)
    opt = pyo.SolverFactory("ipopt")
    assert opt.available(exception_flag=False), "Neither PATH nor IPOPT available."
    res = opt.solve(m, tee=True)
    return res


def build_and_solve(A_list, B_list, p, gamma, solver="path"):
    m = build_model(A_list, B_list, p, gamma)
    solve_model(m, solver=solver)
    # extract
    x = np.array([pyo.value(m.x[i]) for i in m.I])
    y = np.array([pyo.value(m.y[j]) for j in m.J])
    alpha1 = float(pyo.value(m.alpha1))
    alpha2 = float(pyo.value(m.alpha2))
    lam1 = np.array([pyo.value(m.lam1[k]) for k in m.K])
    lam2 = np.array([pyo.value(m.lam2[k]) for k in m.K])
    z1 = np.array([pyo.value(m.z1[k]) for k in m.K])
    z2 = np.array([pyo.value(m.z2[k]) for k in m.K])
    return {"x": x, "y": y, "alpha1": alpha1, "alpha2": alpha2, "lam1": lam1, "z1": z1, "lam2": lam2, "z2": z2}


if __name__ == "__main__":
    # Demo
    A1 = np.array([[0.8, 0.1], [0.2, 0.6]])
    A2 = np.array([[0.3, 0.9], [0.7, 0.4]])
    B1 = np.array([[0.4, 0.7], [0.9, 0.2]])
    B2 = np.array([[0.6, 0.3], [0.1, 0.8]])
    A_list = [A1, A2]
    B_list = [B1, B2]
    p = np.array([0.5, 0.5])
    gamma = 0.8
    res = build_and_solve(A_list, B_list, p, gamma, solver="path")
    print("x* =", res["x"], " y* =", res["y"])
    print("alpha1 =", res["alpha1"], " alpha2 =", res["alpha2"])
    print("lam1 =", res["lam1"], " z1 =", res["z1"])
    print("lam2 =", res["lam2"], " z2 =", res["z2"])
