"""Reproduce the paired 100-step normalization-sensitivity diagnostic."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cumg import StochasticFOConfig, solve_cvar_stochastic_fo  # noqa: E402
from cumg.validation import normalize_game_inputs  # noqa: E402
from experiments.common.experiment import simulate_population_payoffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new diagnostic file")
    A, B, p, _ = simulate_population_payoffs(K=500, n=50, seed=15328123)
    _, _, p1 = normalize_game_inputs(A, B, p)
    _, _, p2 = normalize_game_inputs(A, B, p1)
    records = []
    for name, kappa, step, theta_step, decay in (
        ("split_decay", 0.01, 500, 1e-5, 0.5),
        ("split_constant", 0.003, 30, 1e-4, 0.0),
        ("shared_control", 0.01, 500, None, 0.5),
    ):
        config = StochasticFOConfig(
            kappa=kappa,
            tau=0.002,
            max_iter=100,
            step_size=step,
            theta_step_size=theta_step,
            step_decay=decay,
            seed=15328123,
            logit_bound=20,
            record_every=100,
            certify_every=100,
            regret_tolerance=0.01,
            n_random_starts=0,
            jit_updates=True,
        )
        single = solve_cvar_stochastic_fo(A, B, p, gamma=0.5, alpha=0.5, config=config)
        restarted = solve_cvar_stochastic_fo(A, B, p, gamma=0.5, alpha=0.5, config=replace(config, n_random_starts=1))
        first = single.history[-1]
        second = [r for r in restarted.history if r["start_index"] == 0][-1]
        records.append(
            {
                "setting": name,
                "max_iter": 100,
                "single_eta": first["eta"],
                "restart_first_start_eta": second["eta"],
                "strategy_l1_difference": float(
                    np.abs(first["x"] - second["x"]).sum() + np.abs(first["y"] - second["y"]).sum()
                ),
                "single_theta": first["theta"],
                "restart_first_start_theta": second["theta"],
            }
        )
    output = {
        "diagnostic_only": True,
        "description": (
            "The same uniform full-batch start with and without a restart wrapper. "
            "The wrapper normalizes p once more before solving. "
            "No random batch sampling is used in either uniform start."
        ),
        "seed": 15328123,
        "K": 500,
        "n": 50,
        "normalization_max_absolute_difference": float(np.max(np.abs(p1 - p2))),
        "cases": records,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
