"""Evaluate the uniform-profile epsilon-equilibrium baseline on random games."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from compare_scalability_approaches import (  # noqa: E402
    RISKS,
    StreamingCsvWriter,
    experiment_seed,
    rep_indices,
    simulate_random_payoffs,
)

from cumg.small_support import full_cvar_regret, full_msd_regret  # noqa: E402

DEFAULT_RESULT_DIR = ROOT / "experiments" / "results" / "uniform"
DEFAULT_CSV = DEFAULT_RESULT_DIR / "uniform_profile_baseline.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_RESULT_DIR / "uniform_profile_baseline_summary.csv"


def run_instance(args: argparse.Namespace, risk: str, K: int, n: int, rep: int) -> dict[str, Any]:
    seed = experiment_seed(risk, K, n, rep, args.seed_base)
    A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
    uniform = np.ones(n, dtype=float) / n
    started = perf_counter()
    if risk == "msd":
        cert = full_msd_regret(A, B, p, args.gamma, uniform, uniform)
    elif risk == "cvar":
        cert = full_cvar_regret(A, B, p, args.gamma, args.alpha, uniform, uniform)
    else:
        raise ValueError(f"risk must be one of {RISKS}; got {risk!r}.")
    elapsed = perf_counter() - started
    eta = float(cert["eta"])
    return {
        "risk": risk,
        "K": K,
        "n": n,
        "rep": rep,
        "seed": seed,
        "gamma": args.gamma,
        "alpha": args.alpha if risk == "cvar" else np.nan,
        "epsilon": args.epsilon,
        "low": args.low,
        "high": args.high,
        "profile": "uniform",
        "uniform_success": bool(np.isfinite(eta) and eta <= args.epsilon),
        "uniform_eta": eta,
        "uniform_regret1": float(cert["regret1"]),
        "uniform_regret2": float(cert["regret2"]),
        "uniform_rho1": float(cert["rho1"]),
        "uniform_rho2": float(cert["rho2"]),
        "uniform_time_s": elapsed,
    }


def run_experiment(
    args: argparse.Namespace,
    row_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for risk in args.risk:
        for K in args.K:
            for n in args.n:
                for rep in rep_indices(args):
                    row = run_instance(args, risk, K, n, rep)
                    rows.append(row)
                    if row_callback is not None:
                        row_callback(row)
                    if not args.quiet:
                        print(
                            f"{risk} K={K:>3} n={n:>3} rep={rep:>3} "
                            f"uniform: ok={row['uniform_success']} "
                            f"eta={row['uniform_eta']:.4g} time={row['uniform_time_s']:.4f}s"
                        )
    return rows


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([row.get(key, np.nan) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["risk"], row["K"], row["n"]), []).append(row)

    summary = []
    for (risk, K, n), group in sorted(groups.items()):
        eta = _finite_values(group, "uniform_eta")
        times = _finite_values(group, "uniform_time_s")
        pass_count = sum(bool(row["uniform_success"]) for row in group)
        summary.append(
            {
                "risk": risk,
                "K": K,
                "n": n,
                "reps": len(group),
                "uniform_successes": pass_count,
                "uniform_success_rate": pass_count / len(group) if group else np.nan,
                "uniform_eta_q05": float(np.quantile(eta, 0.05)) if eta.size else np.nan,
                "uniform_eta_median": float(np.median(eta)) if eta.size else np.nan,
                "uniform_eta_q95": float(np.quantile(eta, 0.95)) if eta.size else np.nan,
                "uniform_eta_max": float(np.max(eta)) if eta.size else np.nan,
                "uniform_time_median_s": float(np.median(times)) if times.size else np.nan,
            }
        )
    return summary


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSummary")
    print("-------")
    for row in summarize(rows):
        print(
            f"{row['risk'].upper()} K={row['K']:>3} n={row['n']:>3}: "
            f"success_rate={row['uniform_success_rate']:.3f}, "
            f"median_eta={row['uniform_eta_median']:.4g}, "
            f"max_eta={row['uniform_eta_max']:.4g}, "
            f"median_time={row['uniform_time_median_s']:.4g}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", nargs="+", choices=RISKS, default=["msd", "cvar"])
    parser.add_argument("--K", type=int, nargs="+", default=[250, 500])
    parser.add_argument("--n", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--no-summary-csv", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.no_summary_csv:
        args.summary_csv = None
    try:
        rep_indices(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    with StreamingCsvWriter(args.csv) as writer:
        rows = run_experiment(args, row_callback=writer.write_row)
    print_summary(rows)
    print(f"\nwrote {args.csv}")
    if args.summary_csv is not None:
        with StreamingCsvWriter(args.summary_csv) as writer:
            for row in summarize(rows):
                writer.write_row(row)
        print(f"wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
