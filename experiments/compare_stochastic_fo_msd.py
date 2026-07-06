"""Compare MSD stochastic first-order scalability settings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cumg import StochasticFOConfig, solve_msd_stochastic_fo  # noqa: E402


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


def _optional_positive_float(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value


def _optional_positive_int(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _minibatch_size(args: argparse.Namespace, K: int) -> int:
    if args.batch_size is not None:
        return max(1, min(K, args.batch_size))
    return max(1, min(K, int(np.ceil(np.sqrt(K)))))


def _method_config(
    args: argparse.Namespace, K: int, seed: int, method: str
) -> StochasticFOConfig:
    if method == "full_batch":
        batch_size = None
    elif method == "minibatch":
        batch_size = _minibatch_size(args, K)
    else:
        raise ValueError(f"Unknown method: {method}")

    return StochasticFOConfig(
        kappa=args.entropy_kappa,
        tau=args.smoothing_tau,
        max_iter=args.max_iter,
        batch_size=batch_size,
        step_size=args.step_size,
        step_decay=args.step_decay,
        seed=seed,
        logit_bound=_optional_positive_float(args.logit_bound),
        gradient_clip_norm=_optional_positive_float(args.gradient_clip_norm),
        record_every=_optional_positive_int(args.record_every),
        certify_every=_optional_positive_int(args.certify_every),
        regret_tolerance=args.regret_tolerance,
    )


def _run_method(name: str, A, B, p, gamma: float, args: argparse.Namespace, seed: int):
    start = perf_counter()
    try:
        config = _method_config(args, A.shape[0], seed, name)
        result = solve_msd_stochastic_fo(A, B, p, gamma=gamma, config=config)
        return result, perf_counter() - start, None
    except Exception as exc:  # pragma: no cover - experiment diagnostics
        return None, perf_counter() - start, str(exc)


def _method_metrics(
    prefix: str, result, elapsed_s: float, error: str | None = None
) -> dict[str, Any]:
    cert = result.certificate if result is not None else {}
    best_iterate = result.best_iterate if result is not None else {}
    best_certificate = (
        result.best_certificate
        if result is not None and result.best_certificate is not None
        else {}
    )
    method_error = error

    return {
        f"{prefix}_success": bool(result.success) if result is not None else False,
        f"{prefix}_time_s": elapsed_s,
        f"{prefix}_solve_time_s": (
            float(result.solve_time_s) if result is not None else np.nan
        ),
        f"{prefix}_eta": float(cert.get("eta", np.nan)),
        f"{prefix}_regret1": float(cert.get("regret1", np.nan)),
        f"{prefix}_regret2": float(cert.get("regret2", np.nan)),
        f"{prefix}_residual_norm": (
            float(result.residual_norm) if result is not None else np.nan
        ),
        f"{prefix}_objective": (
            float(result.objective) if result is not None else np.nan
        ),
        f"{prefix}_iterations": result.iterations if result is not None else None,
        f"{prefix}_has_profile": result is not None
        and result.x is not None
        and result.y is not None,
        f"{prefix}_history_len": len(result.history) if result is not None else 0,
        f"{prefix}_best_residual_norm": float(
            best_iterate.get("residual_norm", np.nan)
        ),
        f"{prefix}_best_objective": float(best_iterate.get("objective", np.nan)),
        f"{prefix}_best_certificate_eta": float(best_certificate.get("eta", np.nan)),
        f"{prefix}_error": method_error,
    }


def _pairwise_metrics(row: dict[str, Any], results: dict[str, Any]) -> None:
    if "full_batch" not in results or "minibatch" not in results:
        return
    row["eta_diff_minibatch_minus_full_batch"] = (
        row["minibatch_eta"] - row["full_batch_eta"]
    )
    row["residual_diff_minibatch_minus_full_batch"] = (
        row["minibatch_residual_norm"] - row["full_batch_residual_norm"]
    )
    row["time_ratio_minibatch_over_full_batch"] = (
        row["minibatch_time_s"] / row["full_batch_time_s"]
        if row["full_batch_time_s"] > 0
        else np.nan
    )

    full_batch = results["full_batch"]
    minibatch = results["minibatch"]
    if (
        full_batch is not None
        and minibatch is not None
        and full_batch.x is not None
        and minibatch.x is not None
    ):
        row["x_l1_minibatch_minus_full_batch"] = float(
            np.sum(np.abs(minibatch.x - full_batch.x))
        )
        row["y_l1_minibatch_minus_full_batch"] = float(
            np.sum(np.abs(minibatch.y - full_batch.y))
        )
    else:
        row["x_l1_minibatch_minus_full_batch"] = np.nan
        row["y_l1_minibatch_minus_full_batch"] = np.nan


def run_instance(args: argparse.Namespace, K: int, n: int, seed: int) -> dict[str, Any]:
    A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
    row: dict[str, Any] = {
        "K": K,
        "n": n,
        "seed": seed,
        "gamma": args.gamma,
        "entropy_kappa": args.entropy_kappa,
        "smoothing_tau": args.smoothing_tau,
        "max_iter": args.max_iter,
        "minibatch_size": _minibatch_size(args, K),
        "step_size": args.step_size,
        "step_decay": args.step_decay,
        "regret_tolerance": args.regret_tolerance,
        "record_every": args.record_every,
        "certify_every": args.certify_every,
        "methods": ",".join(args.methods),
    }

    results = {}
    for method in args.methods:
        result, elapsed, error = _run_method(method, A, B, p, args.gamma, args, seed)
        results[method] = result
        row.update(_method_metrics(method, result, elapsed, error))
    _pairwise_metrics(row, results)
    return row


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([row[key] for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def print_summary(rows: list[dict[str, Any]], methods: list[str]) -> None:
    print("\nSummary")
    print("-------")
    for prefix in methods:
        success = np.array(
            [bool(row[f"{prefix}_success"]) for row in rows], dtype=float
        )
        times = _finite_values(rows, f"{prefix}_time_s")
        etas = _finite_values(rows, f"{prefix}_eta")
        residuals = _finite_values(rows, f"{prefix}_residual_norm")
        time_text = f"{np.median(times):.4g}s" if times.size else "nan"
        eta_text = f"{np.median(etas):.4g}" if etas.size else "nan"
        residual_text = f"{np.median(residuals):.4g}" if residuals.size else "nan"
        print(
            f"{prefix:>10}: success_rate={success.mean():.3f}, "
            f"median_time={time_text}, "
            f"median_eta={eta_text}, "
            f"median_residual={residual_text}"
        )

    if "full_batch" in methods and "minibatch" in methods:
        ratios = _finite_values(rows, "time_ratio_minibatch_over_full_batch")
        eta_diffs = _finite_values(rows, "eta_diff_minibatch_minus_full_batch")
        x_diffs = _finite_values(rows, "x_l1_minibatch_minus_full_batch")
        y_diffs = _finite_values(rows, "y_l1_minibatch_minus_full_batch")
        if ratios.size:
            print(f"minibatch/full_batch median time ratio: {np.median(ratios):.4g}")
        if eta_diffs.size:
            print(
                f"median eta difference, minibatch - full_batch: {np.median(eta_diffs):.4g}"
            )
        if x_diffs.size and y_diffs.size:
            print(
                f"median strategy L1 difference: x={np.median(x_diffs):.4g}, y={np.median(y_diffs):.4g}"
            )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--n", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--entropy-kappa", type=float, default=0.2)
    parser.add_argument("--smoothing-tau", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--logit-bound", type=float, default=20.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--certify-every", type=int, default=0)
    parser.add_argument("--regret-tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["full_batch", "minibatch"],
        default=["full_batch", "minibatch"],
    )
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for K in args.K:
        for n in args.n:
            for rep in range(args.reps):
                seed = args.seed_base + 10_000 * K + 100 * n + rep
                row = run_instance(args, K, n, seed)
                rows.append(row)
                if not args.quiet:
                    method_parts = [
                        f"{method}: success={row[f'{method}_success']} eta={row[f'{method}_eta']:.4g} "
                        f"resid={row[f'{method}_residual_norm']:.4g} time={row[f'{method}_time_s']:.3f}s"
                        for method in args.methods
                    ]
                    print(f"K={K:>3} n={n:>3} rep={rep:>2} " + " | ".join(method_parts))

    print_summary(rows, args.methods)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
