"""Shared experiment helpers, promoted without changing numerical behavior."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_METHODS = (
    "mcp",
    "screened_dual",
    "action_dual",
    "restricted_mcp",
    "stochastic_full_batch",
    "stochastic_minibatch",
)


METHODS = (*DEFAULT_METHODS, "qptas", "qptas_screened")


RISKS = ("msd", "cvar")


PAYOFF_MODELS = ("uniform", "cell_beta_uniform_v1")


PAYOFF_POPULATIONS = (
    {"family": "beta", "a": 1, "b": 4},
    {"family": "beta", "a": 2, "b": 3},
    {"family": "beta", "a": 3, "b": 2},
    {"family": "beta", "a": 4, "b": 1},
    {"family": "uniform", "low": 0, "high": 1},
)


DEFAULT_PATH_OPTIONS = {
    "major_iteration_limit": 50_000_000,
    "minor_iteration_limit": 50_000_000,
    "cumulative_iteration_limit": 100_000_000,
    "time_limit": 300,
    "nms_memory_size": 50,
    "restart_limit": 100,
    "convergence_tolerance": 1e-8,
}


def simulate_random_payoffs(
    K: int,
    n: int,
    seed: int,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.uniform(low, high, size=(K, n, n))
    B = rng.uniform(low, high, size=(K, n, n))
    p = np.ones(K, dtype=float) / K
    return A, B, p


def simulate_population_payoffs(
    K: int,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a game with a fixed, uniformly chosen population per payoff entry.

    Return A, B, empirical probabilities, and population IDs shaped (2, n, n).
    Mapping and sampling use separate RNG streams; the two players' maps and
    samples are independent. For a fixed seed/n, increasing K preserves the map
    and the previously drawn samples. All five populations have support [0, 1].
    """
    for name, value in (("K", K), ("n", n)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    population_count = len(PAYOFF_POPULATIONS)
    streams = np.random.SeedSequence(seed).spawn(1 + 2 * population_count)
    population_ids = np.random.default_rng(streams[0]).integers(0, population_count, size=(2, n, n))
    payoffs = []
    for player in range(2):
        samples = np.empty((K, n, n), dtype=float)
        for population, specification in enumerate(PAYOFF_POPULATIONS):
            mask = population_ids[player] == population
            rng = np.random.default_rng(streams[1 + player * population_count + population])
            size = (K, int(mask.sum()))
            if specification["family"] == "beta":
                samples[:, mask] = rng.beta(specification["a"], specification["b"], size=size)
            else:
                samples[:, mask] = rng.uniform(specification["low"], specification["high"], size=size)
        payoffs.append(samples)
    return payoffs[0], payoffs[1], np.full(K, 1.0 / K), population_ids


def experiment_seed(risk: str, K: int, n: int, rep: int, seed_base: int) -> int:
    return seed_base + 1_000_000 * RISKS.index(risk) + 10_000 * K + 100 * n + rep


def optional_positive_float(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value


def optional_positive_int(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def stochastic_minibatch_size(args: argparse.Namespace, K: int) -> int:
    if args.batch_size is not None:
        return max(1, min(K, args.batch_size))
    return max(1, min(K, int(np.ceil(np.sqrt(K)))))


def stochastic_regret_tolerance(args: argparse.Namespace) -> float:
    tolerance = getattr(args, "stochastic_regret_tolerance", None)
    return args.epsilon if tolerance is None else tolerance


def rep_indices(args: argparse.Namespace) -> range:
    rep_stop = args.rep_stop if args.rep_stop is not None else args.reps
    if args.reps < 0:
        raise ValueError("reps must be nonnegative.")
    if args.rep_start < 0:
        raise ValueError("rep-start must be nonnegative.")
    if rep_stop < args.rep_start:
        raise ValueError("rep-stop must be greater than or equal to rep-start.")
    if rep_stop > args.reps:
        raise ValueError("rep-stop must be less than or equal to reps.")
    return range(args.rep_start, rep_stop)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([row.get(key, np.nan) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def certificate_success(status, eta, epsilon: float):
    """Common reporting rule for scalar rows and arrays; preserve native flags separately."""
    eta = np.asarray(eta, dtype=float)
    return (np.asarray(status) == "completed") & np.isfinite(eta) & (eta <= epsilon)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> StreamingCsvWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()

    def write_row(self, row: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("StreamingCsvWriter must be used as a context manager.")
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
