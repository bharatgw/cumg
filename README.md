# cumgSolver

`cumgSolver` contains Python tools for risk-aware complete-uncertainty matrix games. The library code is separated from the experiment notebooks and committed result CSVs so the package can be installed cleanly while the research workflow remains reproducible.

## What It Supports

- Mean-semi-deviation (MSD) risk-aware two-player bimatrix games.
- Lower-tail CVaR risk-aware two-player bimatrix games.
- Pyomo MCP model builders for direct solver use.
- Solver wrappers for PATH/PATHAMPL with optional IPOPT fallback.
- Randomized small-support restricted MCP searches.
- Optional Mystic-based seeding utilities for MSD residuals.

## Installation

From this repository:

```bash
pip install -e ".[dev,notebooks]"
```

For optional Mystic seeding:

```bash
pip install -e ".[mystic]"
```

## External Solvers

The package does not vendor PATH, PATHAMPL, IPOPT, or other solver binaries. Install and license those tools separately, then make the solver executable visible to Pyomo.

- PATH/PATHAMPL is the preferred backend for MCP models.
- IPOPT can be used as a fallback through Pyomo's `mpec.simple_nonlinear` transform.
- Solver-dependent tests and examples skip or fail clearly when no solver is available.

## Quickstart

```python
import numpy as np

from cumgSolver import build_msd_mcp_model, solve_msd_mcp

A = [
    np.array([[0.8, 0.1], [0.2, 0.6]]),
    np.array([[0.3, 0.9], [0.7, 0.4]]),
]
B = [
    np.array([[0.4, 0.7], [0.9, 0.2]]),
    np.array([[0.6, 0.3], [0.1, 0.8]]),
]
p = np.array([0.5, 0.5])

model = build_msd_mcp_model(A, B, p, gamma=0.8)
result = solve_msd_mcp(A, B, p, gamma=0.8, solver="pathampl")

print(result.x, result.y)
```

## Repository Layout

- `src/cumgSolver/`: installable package source.
- `tests/`: solver-free and solver-gated tests.
- `examples/`: small runnable examples.
- `experiments/`: reproducibility notebooks and experiment notes.
- `experiments/results/`: committed CSV outputs from experiments; these files are not part of the installed package.
- `notebooks/`: exploratory notebooks preserved for reference.

## Experiments

The experiment notebooks and CSV outputs are preserved under `experiments/`. See `experiments/README.md` for the result files, solver expectations, and suggested rerun workflow.

## Development Checks

```bash
python -m compileall src tests examples
pytest
ruff check src tests examples
ruff format --check src tests examples
```

Solver-gated tests are marked with `pytest.mark.solver` and skip when PATH/PATHAMPL or IPOPT is unavailable.

## License

The Python package code is released under the MIT License. Third-party solver binaries and reference PDFs are not included in the publishable repository; install or obtain them from their original sources and follow their licenses.

