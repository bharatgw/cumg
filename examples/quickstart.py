"""Minimal cumgSolver example."""

import numpy as np

from cumgSolver import build_msd_mcp_model, format_solver_availability, solve_msd_mcp


def main() -> None:
    A = [
        np.array([[0.8, 0.1], [0.2, 0.6]]),
        np.array([[0.3, 0.9], [0.7, 0.4]]),
    ]
    B = [
        np.array([[0.4, 0.7], [0.9, 0.2]]),
        np.array([[0.6, 0.3], [0.1, 0.8]]),
    ]
    p = np.array([0.5, 0.5])
    gamma = 0.8

    model = build_msd_mcp_model(A, B, p, gamma)
    print(f"Built MSD MCP with {len(model.I)}x{len(model.J)} actions and {len(model.K)} scenarios.")
    print("Solver status:", format_solver_availability())

    try:
        result = solve_msd_mcp(A, B, p, gamma, solver="pathampl", fallback_solver="ipopt")
    except RuntimeError as exc:
        print(f"Install PATH/PATHAMPL or IPOPT to solve this model: {exc}")
        return

    print("x* =", np.round(result.x, 6))
    print("y* =", np.round(result.y, 6))
    print("alpha1 =", round(result.alpha1, 6))
    print("alpha2 =", round(result.alpha2, 6))


if __name__ == "__main__":
    main()
