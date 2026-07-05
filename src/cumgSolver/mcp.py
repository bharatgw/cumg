"""Shared Pyomo MCP solve helpers."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pyomo.environ as pyo


def _solver_factory(name: str):
    opt = pyo.SolverFactory(name)
    if not opt.available(exception_flag=False):
        return None
    return opt


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

    def apply_options(opt):
        for key, value in solver_options.items():
            opt.options[key] = value

    def prepare_model_for_solver(solver_name: str):
        if solver_name == "ipopt":
            pyo.TransformationFactory("mpec.simple_nonlinear").apply_to(model)

    opt = _solver_factory(solver)
    if opt is not None:
        attempted.append(solver)
        prepare_model_for_solver(solver)
        apply_options(opt)
        start = perf_counter()
        result = opt.solve(model, tee=tee)
        return result, perf_counter() - start, solver, model

    if fallback_solver is not None:
        attempted.append(fallback_solver)
        fallback = _solver_factory(fallback_solver)
        if fallback is not None:
            prepare_model_for_solver(fallback_solver)
            apply_options(fallback)
            start = perf_counter()
            result = fallback.solve(model, tee=tee)
            return result, perf_counter() - start, fallback_solver, model

    tried = ", ".join(attempted or [solver])
    raise RuntimeError(
        f"No available solver found. Tried: {tried}. Install PATH/PATHAMPL or IPOPT "
        "and make it visible to Pyomo."
    )
