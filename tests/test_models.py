import numpy as np
import pyomo.environ as pyo
import pytest

from cumgSolver import build_msd_mcp_model, solve_msd_mcp
from cumgSolver.cvar import build_cvar_mcp_model


def demo_game():
    A = [
        np.array([[0.8, 0.1], [0.2, 0.6]]),
        np.array([[0.3, 0.9], [0.7, 0.4]]),
    ]
    B = [
        np.array([[0.4, 0.7], [0.9, 0.2]]),
        np.array([[0.6, 0.3], [0.1, 0.8]]),
    ]
    return A, B, np.array([0.5, 0.5])


def test_build_msd_mcp_model_has_expected_sets():
    A, B, p = demo_game()

    model = build_msd_mcp_model(A, B, p, gamma=0.8)

    assert len(model.I) == 2
    assert len(model.J) == 2
    assert len(model.K) == 2


def test_build_cvar_mcp_model_has_expected_sets():
    A, B, p = demo_game()

    model = build_cvar_mcp_model(A, B, p, gamma=0.5, alpha=0.5)

    assert len(model.I) == 2
    assert len(model.J) == 2
    assert len(model.K) == 2


@pytest.mark.solver
def test_solve_msd_mcp_with_available_solver():
    if not (
        pyo.SolverFactory("pathampl").available(exception_flag=False)
        or pyo.SolverFactory("path").available(exception_flag=False)
        or pyo.SolverFactory("ipopt").available(exception_flag=False)
    ):
        pytest.skip("No PATH/PATHAMPL or IPOPT solver available.")

    A, B, p = demo_game()
    result = solve_msd_mcp(A, B, p, gamma=0.8, solver="pathampl", fallback_solver="ipopt")

    assert np.isfinite(result.alpha1)
    assert np.isfinite(result.alpha2)
    assert result.x.sum() == pytest.approx(1.0, abs=1e-5)
    assert result.y.sum() == pytest.approx(1.0, abs=1e-5)

