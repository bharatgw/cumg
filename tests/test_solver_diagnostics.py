from types import SimpleNamespace

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import cumg.mcp as mcp
from cumg import available_solvers, format_solver_availability, solver_available


def test_solver_available_reports_boolean_for_unknown_solver():
    assert solver_available("definitely-not-a-real-cumg-solver") is False


def test_available_solvers_preserves_requested_names():
    statuses = available_solvers(["definitely-not-a-real-cumg-solver", "also-missing"])

    assert statuses == {
        "definitely-not-a-real-cumg-solver": False,
        "also-missing": False,
    }


def test_format_solver_availability_is_human_readable():
    summary = format_solver_availability(["definitely-not-a-real-cumg-solver"])

    assert summary == "definitely-not-a-real-cumg-solver=missing"


def test_solve_pyomo_mcp_model_scopes_nested_solver_options(monkeypatch):
    calls = []

    class FakeSolver:
        def __init__(self, name):
            self.name = name
            self.options = {}

        def solve(self, model, tee=False):
            calls.append((self.name, dict(self.options)))
            if self.name == "pathampl":
                termination = TerminationCondition.maxIterations
            else:
                termination = TerminationCondition.optimal
            return SimpleNamespace(
                solver=SimpleNamespace(
                    status=SolverStatus.ok,
                    termination_condition=termination,
                )
            )

    monkeypatch.setattr(mcp, "_solver_factory", lambda name: FakeSolver(name))

    mcp.solve_pyomo_mcp_model(
        pyo.ConcreteModel(),
        solver="pathampl",
        fallback_solver="fallback",
        solver_options={
            "pathampl": {"time_limit": 300},
            "fallback": {"max_iter": 100},
        },
    )

    assert calls == [
        ("pathampl", {"time_limit": 300}),
        ("fallback", {"max_iter": 100}),
    ]
