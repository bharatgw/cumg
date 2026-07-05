import numpy as np
import pyomo.environ as pyo
import pytest
from sample_games import dominant_action_game, matching_pennies_game

from cumgSolver import solve_msd_mcp
from cumgSolver.cvar import solve_cvar_mcp
from cumgSolver.results import SupportSearchConfig
from cumgSolver.small_support import small_support_search_cvar, small_support_search_msd


def available_solver():
    for solver in ("pathampl", "path", "ipopt"):
        if pyo.SolverFactory(solver).available(exception_flag=False):
            return solver
    return None


def solver_kwargs():
    solver = available_solver()
    if solver is None:
        pytest.skip("No PATH/PATHAMPL or IPOPT solver available.")
    return {
        "solver": solver,
        "fallback_solver": None if solver == "ipopt" else "ipopt",
        "tee": False,
    }


def assert_pure_first_action(x, y, tol=1e-4):
    np.testing.assert_allclose(x, np.array([1.0, 0.0]), atol=tol)
    np.testing.assert_allclose(y, np.array([1.0, 0.0]), atol=tol)


def assert_matching_pennies_equilibrium(x, y, tol=1e-4):
    np.testing.assert_allclose(x, np.array([0.5, 0.5]), atol=tol)
    np.testing.assert_allclose(y, np.array([0.5, 0.5]), atol=tol)


@pytest.mark.solver
def test_msd_solver_finds_dominant_action_equilibrium():
    A, B, p = dominant_action_game()

    result = solve_msd_mcp(A, B, p, gamma=0.6, **solver_kwargs())

    assert_pure_first_action(result.x, result.y)
    assert result.model == "MSD"


@pytest.mark.solver
def test_cvar_solver_finds_dominant_action_equilibrium():
    A, B, p = dominant_action_game()

    result = solve_cvar_mcp(A, B, p, gamma=0.5, alpha=0.5, **solver_kwargs())

    assert_pure_first_action(result.x, result.y)
    assert result.model == "CVaR"


@pytest.mark.solver
def test_small_support_msd_finds_dominant_action_equilibrium_with_full_support_candidate():
    A, B, p = dominant_action_game()
    kwargs = solver_kwargs()
    config = SupportSearchConfig(
        kappa=2,
        tau=2,
        max_candidates=1,
        seed=0,
        solver=kwargs["solver"],
        fallback_solver=kwargs["fallback_solver"],
    )

    result = small_support_search_msd(A, B, p, gamma=0.6, config=config)

    assert result.success, result.best_error
    assert_pure_first_action(result.x, result.y)
    assert result.support == ((0, 1), (0, 1))
    assert result.scenarios == ((0, 1), (0, 1))
    assert result.metadata["certificate"]["eta"] <= config.epsilon


@pytest.mark.solver
def test_small_support_cvar_finds_dominant_action_equilibrium_with_full_support_candidate():
    A, B, p = dominant_action_game()
    kwargs = solver_kwargs()
    config = SupportSearchConfig(
        kappa=2,
        tau=2,
        max_candidates=1,
        seed=0,
        solver=kwargs["solver"],
        fallback_solver=kwargs["fallback_solver"],
    )

    result = small_support_search_cvar(A, B, p, gamma=0.5, alpha=0.5, config=config)

    assert result.success, result.best_error
    assert_pure_first_action(result.x, result.y)
    assert result.support == ((0, 1), (0, 1))
    assert result.scenarios == ((0, 1), (0, 1))
    assert result.metadata["certificate"]["eta"] <= config.epsilon


@pytest.mark.solver
def test_msd_solver_finds_matching_pennies_mixed_equilibrium():
    A, B, p = matching_pennies_game()

    result = solve_msd_mcp(A, B, p, gamma=0.0, **solver_kwargs())

    assert_matching_pennies_equilibrium(result.x, result.y)
    assert result.alpha1 == pytest.approx(0.0, abs=1e-4)
    assert result.alpha2 == pytest.approx(0.0, abs=1e-4)


@pytest.mark.solver
def test_cvar_solver_finds_matching_pennies_mixed_equilibrium():
    A, B, p = matching_pennies_game()

    result = solve_cvar_mcp(A, B, p, gamma=0.0, alpha=1.0, **solver_kwargs())

    assert_matching_pennies_equilibrium(result.x, result.y)
    assert result.alpha1 == pytest.approx(0.0, abs=1e-4)
    assert result.alpha2 == pytest.approx(0.0, abs=1e-4)
