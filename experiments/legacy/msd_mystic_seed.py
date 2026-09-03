# msd_mystic_seed.py
# Globalize with mystic (Fischer–Burmeister residual) then refine with Pyomo PATH.
# Requires: mystic, numpy; and for refinement: pyomo + PATH (or IPOPT via mpec transform).

import numpy as np


def fb(a, b):
    return np.sqrt(a * a + b * b) - (a + b)


def pack_vars(x, y, lam1, z1, lam2, z2, alpha1, alpha2):
    return np.concatenate([x, y, lam1, z1, lam2, z2, np.array([alpha1, alpha2])])


def unpack_vars(v, n1, n2, K):
    x = v[0:n1]
    y = v[n1 : n1 + n2]
    lam1 = v[n1 + n2 : n1 + n2 + K]
    z1 = v[n1 + n2 + K : n1 + n2 + 2 * K]
    lam2 = v[n1 + n2 + 2 * K : n1 + n2 + 3 * K]
    z2 = v[n1 + n2 + 3 * K : n1 + n2 + 4 * K]
    alpha1 = v[-2]
    alpha2 = v[-1]
    return x, y, lam1, z1, lam2, z2, alpha1, alpha2


def mcp_residual(v, A_list, B_list, p, gamma):
    n1, n2 = A_list[0].shape
    K = len(A_list)
    x, y, lam1, z1, lam2, z2, alpha1, alpha2 = unpack_vars(v, n1, n2, K)
    Abar = sum(pk * Ak for pk, Ak in zip(p, A_list, strict=True))
    Bbar = sum(pk * Bk for pk, Bk in zip(p, B_list, strict=True))
    V1 = Abar.copy()
    for k in range(K):
        V1 += lam1[k] * (A_list[k] - Abar)
    v1 = V1 @ y
    V2 = Bbar.copy()
    for k in range(K):
        V2 += lam2[k] * (B_list[k] - Bbar)
    v2 = V2.T @ x
    r1 = np.array([x @ ((A_list[k] - Abar) @ y) for k in range(K)])
    r2 = np.array([x @ ((B_list[k] - Bbar) @ y) for k in range(K)])
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


def objective(v, A_list, B_list, p, gamma, w_eq=10.0):
    r = mcp_residual(v, A_list, B_list, p, gamma)
    n1, n2 = A_list[0].shape
    K = len(A_list)
    x, y, lam1, z1, lam2, z2, alpha1, alpha2 = unpack_vars(v, n1, n2, K)
    pen = 0.0
    pen += np.sum(np.clip(-x, 0, None) ** 2) + np.sum(np.clip(-y, 0, None) ** 2)
    pen += np.sum(np.clip(-lam1, 0, None) ** 2) + np.sum(np.clip(lam1 - gamma * p, 0, None) ** 2)
    pen += np.sum(np.clip(-lam2, 0, None) ** 2) + np.sum(np.clip(lam2 - gamma * p, 0, None) ** 2)
    pen += np.sum(np.clip(z1, 0, None) ** 2) + np.sum(np.clip(z2, 0, None) ** 2)
    return np.dot(r, r) + w_eq * pen


def _bounds(A_list, B_list, p, gamma):
    n1, n2 = A_list[0].shape
    K = len(A_list)
    lb = [0.0] * n1 + [0.0] * n2
    ub = [1.0] * n1 + [1.0] * n2
    lb += [0.0] * K + [-1.0] * K + [0.0] * K + [-1.0] * K
    ub += list(gamma * p) + [0.0] * K + list(gamma * p) + [0.0] * K
    lb += [-1.0, -1.0]
    ub += [2.0, 2.0]
    return np.array(lb, float), np.array(ub, float)


def run_mystic_seed(A_list, B_list, p, gamma, maxiter=2000, popsize=80, seed=13):
    try:
        from mystic.monitors import VerboseMonitor
        from mystic.solvers import diffev2
        from mystic.termination import ChangeOverGeneration
    except Exception as e:
        raise RuntimeError("mystic is required for this function") from e
    n1, n2 = A_list[0].shape
    lb, ub = _bounds(A_list, B_list, p, gamma)
    dim = lb.size
    rng = np.random.default_rng(seed)
    x0 = lb + (ub - lb) * rng.random(dim)
    mon = VerboseMonitor(50)

    def cost(v):
        return objective(v, A_list, B_list, p, gamma)

    sol = diffev2(
        cost,
        x0=x0,
        bounds=list(zip(lb, ub, strict=True)),
        maxiter=maxiter,
        popsize=popsize,
        ftol=1e-10,
        gtol=50,
        itermon=mon,
        termination=ChangeOverGeneration(tolerance=1e-10, generations=50),
    )
    return sol


def refine_with_pyomo(v_seed, A_list, B_list, p, gamma, solver="path"):
    import pyomo.environ as pyo
    from pyomo_msd_mcp import build_model, solve_model

    n1, n2 = A_list[0].shape
    K = len(A_list)
    x, y, lam1, z1, lam2, z2, alpha1, alpha2 = unpack_vars(v_seed, n1, n2, K)
    m = build_model(A_list, B_list, p, gamma)
    for i in range(n1):
        m.x[i].value = float(x[i])
    for j in range(n2):
        m.y[j].value = float(y[j])
    for k in range(K):
        m.lam1[k].value = float(np.clip(lam1[k], 0.0, float(gamma * p[k])))
        m.lam2[k].value = float(np.clip(lam2[k], 0.0, float(gamma * p[k])))
        m.z1[k].value = float(min(z1[k], 0.0))
        m.z2[k].value = float(min(z2[k], 0.0))
    m.alpha1.value = float(alpha1)
    m.alpha2.value = float(alpha2)
    res = solve_model(m, solver=solver)
    xs = np.array([pyo.value(m.x[i]) for i in m.I])
    ys = np.array([pyo.value(m.y[j]) for j in m.J])
    a1 = float(pyo.value(m.alpha1))
    a2 = float(pyo.value(m.alpha2))
    l1 = np.array([pyo.value(m.lam1[k]) for k in m.K])
    l2 = np.array([pyo.value(m.lam2[k]) for k in m.K])
    z1s = np.array([pyo.value(m.z1[k]) for k in m.K])
    z2s = np.array([pyo.value(m.z2[k]) for k in m.K])
    return {
        "x": xs,
        "y": ys,
        "alpha1": a1,
        "alpha2": a2,
        "lam1": l1,
        "z1": z1s,
        "lam2": l2,
        "z2": z2s,
        "pyomo_result": str(res),
    }


def run_mystic_then_pyomo(A_list, B_list, p, gamma, solver="path"):
    v_seed = run_mystic_seed(A_list, B_list, p, gamma)
    out = refine_with_pyomo(v_seed, A_list, B_list, p, gamma, solver=solver)
    return out


def _demo():
    A1 = np.array([[0.8, 0.1], [0.2, 0.6]])
    A2 = np.array([[0.3, 0.9], [0.7, 0.4]])
    B1 = np.array([[0.4, 0.7], [0.9, 0.2]])
    B2 = np.array([[0.6, 0.3], [0.1, 0.8]])
    A_list = [A1, A2]
    B_list = [B1, B2]
    p = np.array([0.5, 0.5])
    gamma = 0.8
    try:
        v = run_mystic_seed(A_list, B_list, p, gamma, maxiter=500, popsize=60)
    except RuntimeError as e:
        print("Mystic not available in this environment:", e)
        return
    print("Seed residual norm:", np.linalg.norm(mcp_residual(v, A_list, B_list, p, gamma)))
    try:
        out = refine_with_pyomo(v, A_list, B_list, p, gamma, solver="path")
        print("Refined:", out)
    except Exception as e:
        print("Pyomo/solver not available or path import failed:", e)


if __name__ == "__main__":
    _demo()
