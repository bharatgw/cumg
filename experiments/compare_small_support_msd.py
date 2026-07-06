"""Compare nested-LP and dualized support-stage MSD small-support searches."""

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

from cumg.results import SupportSearchConfig  # noqa: E402
from cumg.small_support import (  # noqa: E402
    _certified_search,
    restricted_profile_gap_msd,
    support_sizes,
    supported_profile_gap_msd,
    supported_profile_gap_msd_dual,
)


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


def _search_with_support_stage(
    A,
    B,
    p,
    gamma: float,
    config: SupportSearchConfig,
    support_gap_fn,
    support_maxiter: int,
):
    def support_gap_with_maxiter(*args, **kwargs):
        return support_gap_fn(*args, maxiter=support_maxiter, **kwargs)

    return _certified_search(
        A,
        B,
        p,
        restricted_profile_gap_msd,
        support_gap_with_maxiter,
        {"gamma": gamma},
        {"gamma": gamma},
        config,
    )


def _candidate(result) -> dict[str, Any] | None:
    if result.success:
        return result.metadata
    return result.metadata.get("best_regret")


def _method_metrics(prefix: str, result, elapsed_s: float, error: str | None = None) -> dict[str, Any]:
    candidate = _candidate(result) if result is not None else None
    screen = candidate.get("screen", {}) if candidate is not None else {}
    support_cert = candidate.get("support_certificate", {}) if candidate is not None else {}
    certificate = candidate.get("certificate", support_cert.get("certificate", {})) if candidate is not None else {}
    best_dev1 = certificate.get("best_dev1", {})
    best_dev2 = certificate.get("best_dev2", {})

    return {
        f"{prefix}_success": bool(result.success) if result is not None else False,
        f"{prefix}_time_s": elapsed_s,
        f"{prefix}_eta": float(certificate.get("eta", np.nan)),
        f"{prefix}_regret1": float(certificate.get("regret1", np.nan)),
        f"{prefix}_regret2": float(certificate.get("regret2", np.nan)),
        f"{prefix}_screen_eta": float(screen.get("eta", np.nan)),
        f"{prefix}_support_eta": float(support_cert.get("eta", np.nan)),
        f"{prefix}_dual_eta": float(support_cert.get("dual_eta", np.nan)),
        f"{prefix}_support_violation": float(support_cert.get("violation", np.nan)),
        f"{prefix}_candidate_index": result.candidate_index if result is not None else None,
        f"{prefix}_has_profile": result is not None and result.x is not None and result.y is not None,
        f"{prefix}_best_dev1_success": best_dev1.get("success", np.nan),
        f"{prefix}_best_dev2_success": best_dev2.get("success", np.nan),
        f"{prefix}_error": error,
    }


def _run_method(name: str, A, B, p, gamma, config, support_maxiter):
    support_gap_fn = {
        "nested": supported_profile_gap_msd,
        "dual": supported_profile_gap_msd_dual,
    }[name]
    start = perf_counter()
    try:
        result = _search_with_support_stage(A, B, p, gamma, config, support_gap_fn, support_maxiter)
        return result, perf_counter() - start, None
    except Exception as exc:  # pragma: no cover - experiment diagnostics
        return None, perf_counter() - start, str(exc)


def run_instance(args, K: int, n: int, seed: int) -> dict[str, Any]:
    A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
    kappa, tau = support_sizes(K, n)
    config = SupportSearchConfig(
        epsilon=args.epsilon,
        epsilon_scr=args.epsilon_scr,
        kappa=kappa,
        tau=tau,
        max_candidates=args.max_candidates,
        n_screen_starts=args.n_screen_starts,
        n_regret_starts=args.n_support_starts,
        screen_maxiter=args.screen_maxiter,
        seed=seed,
    )

    nested_result, nested_time, nested_error = _run_method(
        "nested", A, B, p, args.gamma, config, args.support_maxiter
    )
    dual_result, dual_time, dual_error = _run_method("dual", A, B, p, args.gamma, config, args.support_maxiter)

    row: dict[str, Any] = {
        "K": K,
        "n": n,
        "seed": seed,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_scr": args.epsilon_scr,
        "kappa": kappa,
        "tau": tau,
        "max_candidates": args.max_candidates,
        "n_screen_starts": args.n_screen_starts,
        "n_support_starts": args.n_support_starts,
        "screen_maxiter": args.screen_maxiter,
        "support_maxiter": args.support_maxiter,
    }
    row.update(_method_metrics("nested", nested_result, nested_time, nested_error))
    row.update(_method_metrics("dual", dual_result, dual_time, dual_error))

    if (
        nested_result is not None
        and dual_result is not None
        and nested_result.x is not None
        and dual_result.x is not None
    ):
        row["x_l1_dual_minus_nested"] = float(np.sum(np.abs(dual_result.x - nested_result.x)))
        row["y_l1_dual_minus_nested"] = float(np.sum(np.abs(dual_result.y - nested_result.y)))
    else:
        row["x_l1_dual_minus_nested"] = np.nan
        row["y_l1_dual_minus_nested"] = np.nan

    row["eta_diff_dual_minus_nested"] = row["dual_eta"] - row["nested_eta"]
    row["time_ratio_dual_over_nested"] = (
        row["dual_time_s"] / row["nested_time_s"] if row["nested_time_s"] > 0 else np.nan
    )
    return row


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([row[key] for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSummary")
    print("-------")
    for prefix in ("nested", "dual"):
        success = np.array([bool(row[f"{prefix}_success"]) for row in rows], dtype=float)
        times = _finite_values(rows, f"{prefix}_time_s")
        etas = _finite_values(rows, f"{prefix}_eta")
        violations = _finite_values(rows, f"{prefix}_support_violation")
        print(
            f"{prefix:>6}: success_rate={success.mean():.3f}, "
            f"median_time={np.median(times):.4g}s, "
            f"median_eta={np.median(etas):.4g}, "
            f"median_support_violation={np.median(violations):.4g}"
        )

    ratios = _finite_values(rows, "time_ratio_dual_over_nested")
    eta_diffs = _finite_values(rows, "eta_diff_dual_minus_nested")
    x_diffs = _finite_values(rows, "x_l1_dual_minus_nested")
    y_diffs = _finite_values(rows, "y_l1_dual_minus_nested")
    if ratios.size:
        print(f"dual/nested median time ratio: {np.median(ratios):.4g}")
    if eta_diffs.size:
        print(f"median eta difference, dual - nested: {np.median(eta_diffs):.4g}")
    if x_diffs.size and y_diffs.size:
        print(f"median strategy L1 difference: x={np.median(x_diffs):.4g}, y={np.median(y_diffs):.4g}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--epsilon-scr", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--n-screen-starts", type=int, default=1)
    parser.add_argument("--n-support-starts", type=int, default=5)
    parser.add_argument("--screen-maxiter", type=int, default=300)
    parser.add_argument("--support-maxiter", type=int, default=500)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.epsilon_scr is None:
        args.epsilon_scr = 2.0 * args.epsilon / 3.0
    return args


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
                    print(
                        f"K={K:>3} n={n:>3} rep={rep:>2} "
                        f"nested: success={row['nested_success']} eta={row['nested_eta']:.4g} "
                        f"time={row['nested_time_s']:.3f}s | "
                        f"dual: success={row['dual_success']} eta={row['dual_eta']:.4g} "
                        f"dual_eta={row['dual_dual_eta']:.4g} time={row['dual_time_s']:.3f}s"
                    )

    print_summary(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
