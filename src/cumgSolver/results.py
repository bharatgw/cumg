"""Public result and configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for Pyomo MCP solves."""

    solver: str = "pathampl"
    fallback_solver: str | None = "ipopt"
    tee: bool = False
    solver_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverResult:
    """Common result container for MSD and CVaR MCP solves."""

    x: np.ndarray
    y: np.ndarray
    alpha1: float
    alpha2: float
    model: str
    solver: str
    raw_result: Any | None = None
    solve_time_s: float | None = None
    lam1: np.ndarray | None = None
    lam2: np.ndarray | None = None
    z1: np.ndarray | float | None = None
    z2: np.ndarray | float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary compatible with the original notebooks."""

        out = {
            "x": self.x,
            "y": self.y,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "model": self.model,
            "solver": self.solver,
            "time": self.solve_time_s,
            "lam1": self.lam1,
            "lam2": self.lam2,
            "z1": self.z1,
            "z2": self.z2,
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class SupportSearchConfig:
    """Configuration for randomized small-support screening."""

    epsilon: float = 1e-3
    kappa: int = 2
    tau: int = 10
    max_candidates: int = 100
    seed: int = 0
    solver: str = "pathampl"
    fallback_solver: str | None = "ipopt"
    solver_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportSearchResult:
    """Result from a small-support search."""

    success: bool
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    support: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    scenarios: tuple[int, ...] | None = None
    candidate_index: int | None = None
    solver_result: SolverResult | None = None
    best_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

