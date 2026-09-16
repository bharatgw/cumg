"""Recheck calibration profiles and independently price deviations.

Reuse the existing generator and LP certifiers; recompute payoff arithmetic with
einsum and sorted tails to check the local NumPy matmul warnings independently.
The reference formulas here are deliberately specific to uniform p and alpha=0.5.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

DEFAULT_RESULT_DIR = Path(__file__).resolve().parent
REPO = DEFAULT_RESULT_DIR.parents[4]
sys.path[:0] = [str(REPO / "experiments"), str(REPO / "src")]

from compare_scalability_approaches import simulate_population_payoffs  # noqa: E402

from cumg.small_support import full_cvar_regret, full_msd_regret  # noqa: E402

default_phases = (
    "validation_msd",
    "validation_cvar_full",
    "validation_cvar_mini",
    "larger_msd",
    "larger_cvar_full",
    "larger_cvar_mini",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
parser.add_argument("--phases", nargs="+", default=default_phases)
parser.add_argument("--expected-profiles", type=int, default=16)
args = parser.parse_args()
RESULT_DIR = args.result_dir
games = {}
checks = []
with warnings.catch_warnings(record=True) as observed_warnings:
    warnings.simplefilter("always", RuntimeWarning)
    for phase in args.phases:
        with (RESULT_DIR / f"{phase}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows, f"No completed rows for {phase}"
        for row in rows:
            K, n, seed = (int(row[key]) for key in ("K", "n", "seed"))
            key = (K, n, seed)
            if key not in games:
                games[key] = simulate_population_payoffs(K=K, n=n, seed=seed)
            A, B, p, population_ids = games[key]
            np.testing.assert_array_equal(population_ids, json.loads(row["payoff_population_ids"]))
            np.testing.assert_allclose(p, 1 / K, atol=1e-15, rtol=0)
            assert float(row["gamma"]) == 0.5 and K % 2 == 0
            assert row["risk"] != "cvar" or float(row["alpha"]) == 0.5
            for method in row["methods"].split(","):
                x = np.asarray(json.loads(row[f"{method}_x"]))
                y = np.asarray(json.loads(row[f"{method}_y"]))
                cert = (
                    full_msd_regret(A, B, p, 0.5, x, y)
                    if row["risk"] == "msd"
                    else full_cvar_regret(A, B, p, 0.5, 0.5, x, y)
                )
                independent_regrets, pure_lower_bounds, payoff_errors = [], [], []
                for player, own, h in (
                    (1, x, np.einsum("kij,j->ki", A, y)),
                    (2, y, np.einsum("i,kij->kj", x, B)),
                ):
                    best = cert[f"best_dev{player}"]
                    assert best["success"]
                    states = np.column_stack(
                        [
                            np.einsum("ki,i->k", h, own),
                            np.einsum("ki,i->k", h, best["strategy"]),
                            h,
                        ]
                    )
                    assert np.all(np.isfinite(states))
                    means = np.mean(states, axis=0)
                    if row["risk"] == "msd":
                        values = means - 0.5 * np.mean(np.maximum(means - states, 0), axis=0)
                    else:
                        values = 0.5 * means + 0.5 * np.mean(np.sort(states, axis=0)[: K // 2], axis=0)
                    independent_regrets.append(max(0.0, float(values[1] - values[0])))
                    pure_lower_bounds.append(max(0.0, float(np.max(values[2:]) - values[0])))
                    payoff_errors.extend([abs(values[0] - cert[f"rho{player}"]), abs(values[1] - best["value"])])
                eta = max(independent_regrets)
                saved_eta = float(row[f"{method}_eta"])
                assert abs(eta - saved_eta) < 1e-9
                assert max(payoff_errors) < 1e-9
                assert max(pure_lower_bounds) <= eta + 1e-9
                assert (eta <= 0.01) == (row[f"{method}_success"] == "True")
                checks.append(
                    {
                        "phase": phase,
                        "risk": row["risk"],
                        "K": K,
                        "seed": seed,
                        "method": method,
                        "saved_eta": saved_eta,
                        "recomputed_eta": cert["eta"],
                        "independent_payoff_eta": eta,
                        "max_payoff_difference": float(max(payoff_errors)),
                        "pure_deviation_lower_bound": max(pure_lower_bounds),
                        "success": eta <= 0.01,
                    }
                )
assert len(checks) == args.expected_profiles, (len(checks), args.expected_profiles)
output = {
    "scope": "LPs rerun; current and returned deviation payoffs independently evaluated. Not a separate LP solver.",
    "numpy_version": np.__version__,
    "profiles_checked": len(checks),
    "warnings_count": len(observed_warnings),
    "warning_messages": sorted({str(w.message) for w in observed_warnings}),
    "checks": checks,
}
(RESULT_DIR / "certificate_audit.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    f"Verified {len(checks)} profiles; maximum payoff difference {max(c['max_payoff_difference'] for c in checks):.3g}."
)
