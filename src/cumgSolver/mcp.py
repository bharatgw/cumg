"""Shared Pyomo MCP solve helpers."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


def _solver_factory(name: str):
    # ``validate=False`` avoids noisy Pyomo warnings when optional ASL-backed
    # executables such as PATHAMPL are not installed.
    opt = pyo.SolverFactory(name, validate=False)
    if not opt.available(exception_flag=False):
        return None
    return opt


def solver_available(name: str) -> bool:
    """Return whether a Pyomo solver executable is available on this machine."""

    return _solver_factory(name) is not None


def available_solvers(names: Iterable[str] = ("pathampl", "path", "ipopt")) -> dict[str, bool]:
    """Check availability for common CUMG solver backends.

    The package can build MCP models without external binaries, but solving them
    requires a Pyomo-compatible backend. This helper lets examples, notebooks,
    and users report environment readiness before attempting a solve.
    """

    return {name: solver_available(name) for name in names}


def format_solver_availability(names: Iterable[str] = ("pathampl", "path", "ipopt")) -> str:
    """Return a compact human-readable solver availability summary."""

    statuses = available_solvers(names)
    parts = [f"{name}={'available' if ok else 'missing'}" for name, ok in statuses.items()]
    return ", ".join(parts)


def solve_pyomo_mcp_model(
    model: pyo.ConcreteModel,
    solver: str = "pathampl",
    fallback_solver: str | None = "ipopt",
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
):
    """Solve a Pyomo MCP model, optionally falling back through an NLP transform."""

    solver_options = dict(solver_options or {})
    attempted: list[str] = []
    failures: list[str] = []

    def apply_options(opt):
        for key, value in solver_options.items():
            opt.options[key] = value

    def prepare_model_for_solver(solver_name: str):
        if solver_name == "ipopt":
            pyo.TransformationFactory("mpec.simple_nonlinear").apply_to(model)

    def solved_successfully(result) -> bool:
        status = result.solver.status
        termination = result.solver.termination_condition
        accepted_terminations = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
        }
        return status == SolverStatus.ok and termination in accepted_terminations

    for solver_name in [solver, fallback_solver]:
        if solver_name is None:
            continue
        attempted.append(solver_name)
        opt = _solver_factory(solver_name)
        if opt is None:
            failures.append(f"{solver_name}: unavailable")
            continue
        prepare_model_for_solver(solver_name)
        apply_options(opt)
        start = perf_counter()
        result = opt.solve(model, tee=tee)
        elapsed = perf_counter() - start
        if solved_successfully(result):
            return result, elapsed, solver_name, model
        failures.append(
            f"{solver_name}: status={result.solver.status}, termination={result.solver.termination_condition}"
        )

    tried = ", ".join(attempted or [solver])
    details = "; ".join(failures)
    raise RuntimeError(
        f"No available solver found or acceptable solution produced. Tried: {tried}. {details}. "
        "Install PATH/PATHAMPL or IPOPT and make it visible to Pyomo."
    )
