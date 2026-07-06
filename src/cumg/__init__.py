"""Tools for risk-aware complete-uncertainty matrix games."""

from .cvar import build_cvar_mcp_model, solve_cvar_mcp
from .mcp import available_solvers, format_solver_availability, solve_pyomo_mcp_model, solver_available
from .msd import build_msd_mcp_model, solve_msd_mcp
from .results import SolverConfig, SolverResult, SupportSearchConfig, SupportSearchResult
from .small_support import (
    full_cvar_regret,
    full_msd_regret,
    small_support_action_search_cvar,
    small_support_action_search_msd,
    small_support_search_cvar,
    small_support_search_msd,
)

__all__ = [
    "SolverConfig",
    "SolverResult",
    "SupportSearchConfig",
    "SupportSearchResult",
    "build_cvar_mcp_model",
    "build_msd_mcp_model",
    "available_solvers",
    "format_solver_availability",
    "full_cvar_regret",
    "full_msd_regret",
    "small_support_action_search_cvar",
    "small_support_action_search_msd",
    "small_support_search_cvar",
    "small_support_search_msd",
    "solve_cvar_mcp",
    "solve_msd_mcp",
    "solve_pyomo_mcp_model",
    "solver_available",
]
