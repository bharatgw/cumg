"""Compare scalability of package approaches for MSD and CVaR games."""

from __future__ import annotations

import argparse
import csv
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

from cumg import (  # noqa: E402
    StochasticFOConfig,
    solve_cvar_mcp,
    solve_cvar_stochastic_fo,
    solve_msd_mcp,
    solve_msd_stochastic_fo,
)
from cumg.results import SupportSearchConfig  # noqa: E402
from cumg.small_support import (  # noqa: E402
    full_cvar_regret,
    full_msd_regret,
    small_support_action_search_cvar,
    small_support_action_search_msd,
    small_support_search_cvar,
    small_support_search_msd,
    support_sizes,
    supported_profile_gap_cvar_dual,
    supported_profile_gap_cvar_mcp,
    supported_profile_gap_msd_dual,
    supported_profile_gap_msd_mcp,
)

METHODS = (
    "mcp",
    "screened_dual",
    "action_dual",
    "restricted_mcp",
    "stochastic_full_batch",
    "stochastic_minibatch",
)
RISKS = ("msd", "cvar")
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


def _solver_options(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path_options = getattr(args, "path_options", DEFAULT_PATH_OPTIONS)
    if args.solver != "pathampl" or path_options is None:
        return {}
    return {"pathampl": dict(path_options)}


def _support_config(
    args: argparse.Namespace, K: int, n: int, seed: int
) -> SupportSearchConfig:
    kappa, tau = support_sizes(K, n)
    epsilon_scr = args.epsilon_scr
    if epsilon_scr is None:
        epsilon_scr = 2.0 * args.epsilon / 3.0
    return SupportSearchConfig(
        epsilon=args.epsilon,
        epsilon_scr=epsilon_scr,
        kappa=kappa,
        tau=tau,
        max_candidates=args.max_candidates,
        n_screen_starts=args.n_screen_starts,
        n_regret_starts=args.n_support_starts,
        screen_maxiter=args.screen_maxiter,
        seed=seed,
        solver=args.solver,
        fallback_solver=args.fallback_solver,
        solver_options=_solver_options(args),
    )


def _stochastic_config(
    args: argparse.Namespace, K: int, seed: int, method: str
) -> StochasticFOConfig:
    if method == "stochastic_full_batch":
        batch_size = None
    elif method == "stochastic_minibatch":
        batch_size = stochastic_minibatch_size(args, K)
    else:
        raise ValueError(f"Unknown stochastic method: {method}")
    return StochasticFOConfig(
        kappa=args.entropy_kappa,
        tau=args.smoothing_tau,
        max_iter=args.max_iter,
        batch_size=batch_size,
        step_size=args.step_size,
        step_decay=args.step_decay,
        seed=seed,
        logit_bound=optional_positive_float(args.logit_bound),
        gradient_clip_norm=optional_positive_float(args.gradient_clip_norm),
        record_every=optional_positive_int(args.record_every),
        certify_every=optional_positive_int(args.certify_every),
        regret_tolerance=args.epsilon,
    )


def _candidate_from_support_result(result) -> dict[str, Any] | None:
    if result is None:
        return None
    if result.success:
        return result.metadata
    return result.metadata.get("best_candidate") or result.metadata.get("best_regret")


def _profile_from_result(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    if result is None:
        return None, None
    x = getattr(result, "x", None)
    y = getattr(result, "y", None)
    if x is None or y is None:
        return None, None
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _empty_method_metrics(
    prefix: str, elapsed_s: float, error: str | None
) -> dict[str, Any]:
    return {
        f"{prefix}_success": False,
        f"{prefix}_time_s": elapsed_s,
        f"{prefix}_eta": np.nan,
        f"{prefix}_regret1": np.nan,
        f"{prefix}_regret2": np.nan,
        f"{prefix}_has_profile": False,
        f"{prefix}_error": error,
        f"{prefix}_solver": None,
        f"{prefix}_candidate_index": None,
        f"{prefix}_support_eta": np.nan,
        f"{prefix}_screen_eta": np.nan,
        f"{prefix}_support_violation": np.nan,
        f"{prefix}_residual_norm": np.nan,
        f"{prefix}_objective": np.nan,
        f"{prefix}_iterations": None,
        f"{prefix}_best_certificate_eta": np.nan,
        f"{prefix}_best_certificate_iteration": None,
    }


def _certificate_metrics(prefix: str, cert: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_eta": float(cert.get("eta", np.nan)),
        f"{prefix}_regret1": float(cert.get("regret1", np.nan)),
        f"{prefix}_regret2": float(cert.get("regret2", np.nan)),
    }


def _mcp_result_metrics(
    prefix: str,
    result,
    cert: dict[str, Any],
    elapsed_s: float,
    error: str | None,
    eps: float,
):
    x, y = _profile_from_result(result)
    out = _empty_method_metrics(prefix, elapsed_s, error)
    out.update(_certificate_metrics(prefix, cert))
    out.update(
        {
            f"{prefix}_success": bool(
                np.isfinite(cert.get("eta", np.nan)) & (cert.get("eta", np.nan) <= eps)
            ),
            f"{prefix}_has_profile": x is not None and y is not None,
            f"{prefix}_solver": getattr(result, "solver", None),
            f"{prefix}_error": error,
        }
    )
    return out


def _support_result_metrics(prefix: str, result, elapsed_s: float, error: str | None):
    out = _empty_method_metrics(
        prefix,
        elapsed_s,
        error if error is not None else getattr(result, "best_error", None),
    )
    candidate = _candidate_from_support_result(result)
    if candidate is None:
        return out
    screen = candidate.get("screen", {})
    support_cert = candidate.get("support_certificate", {})
    certificate = candidate.get("certificate", support_cert.get("certificate", {}))
    solver_result = support_cert.get("solver_result")
    x, y = _profile_from_result(result)
    out.update(_certificate_metrics(prefix, certificate))
    out.update(
        {
            f"{prefix}_success": bool(result.success),
            f"{prefix}_has_profile": x is not None and y is not None,
            f"{prefix}_solver": getattr(solver_result, "solver", None),
            f"{prefix}_candidate_index": result.candidate_index,
            f"{prefix}_support_eta": float(support_cert.get("eta", np.nan)),
            f"{prefix}_screen_eta": float(screen.get("eta", np.nan)),
            f"{prefix}_support_violation": float(support_cert.get("violation", np.nan)),
        }
    )
    return out


def _stochastic_result_metrics(
    prefix: str, result, elapsed_s: float, error: str | None
):
    out = _empty_method_metrics(prefix, elapsed_s, error)
    cert = result.certificate if result is not None else {}
    x, y = _profile_from_result(result)
    best_certificate = (
        result.best_certificate
        if result is not None and result.best_certificate is not None
        else {}
    )
    out.update(_certificate_metrics(prefix, cert))
    out.update(
        {
            f"{prefix}_success": bool(result.success) if result is not None else False,
            f"{prefix}_has_profile": x is not None and y is not None,
            f"{prefix}_residual_norm": (
                float(result.residual_norm) if result is not None else np.nan
            ),
            f"{prefix}_objective": (
                float(result.objective) if result is not None else np.nan
            ),
            f"{prefix}_iterations": result.iterations if result is not None else None,
            f"{prefix}_best_certificate_eta": float(
                best_certificate.get("eta", np.nan)
            ),
            f"{prefix}_best_certificate_iteration": best_certificate.get("iteration"),
        }
    )
    return out


def _solve_mcp(risk: str, A, B, p, args: argparse.Namespace):
    if risk == "msd":
        result = solve_msd_mcp(
            A,
            B,
            p,
            gamma=args.gamma,
            solver=args.solver,
            fallback_solver=args.fallback_solver,
            solver_options=_solver_options(args),
        )
        cert = full_msd_regret(A, B, p, args.gamma, result.x, result.y)
    else:
        result = solve_cvar_mcp(
            A,
            B,
            p,
            gamma=args.gamma,
            alpha=args.alpha,
            solver=args.solver,
            fallback_solver=args.fallback_solver,
            solver_options=_solver_options(args),
        )
        cert = full_cvar_regret(A, B, p, args.gamma, args.alpha, result.x, result.y)
    return result, cert


def _run_support_method(
    method: str,
    risk: str,
    A,
    B,
    p,
    args: argparse.Namespace,
    config: SupportSearchConfig,
):
    if risk == "msd":
        if method == "screened_dual":
            return small_support_search_msd(
                A,
                B,
                p,
                args.gamma,
                config=config,
                supported_profile_gap_func=supported_profile_gap_msd_dual,
            )
        if method == "action_dual":
            return small_support_action_search_msd(
                A,
                B,
                p,
                args.gamma,
                config=config,
                support_gap_func=supported_profile_gap_msd_dual,
                support_gap_kwargs={"maxiter": args.support_maxiter},
            )
        if method == "restricted_mcp":
            return small_support_action_search_msd(
                A,
                B,
                p,
                args.gamma,
                config=config,
                support_gap_func=supported_profile_gap_msd_mcp,
            )
    else:
        if method == "screened_dual":
            return small_support_search_cvar(
                A,
                B,
                p,
                args.gamma,
                args.alpha,
                config=config,
                supported_profile_gap_func=supported_profile_gap_cvar_dual,
            )
        if method == "action_dual":
            return small_support_action_search_cvar(
                A,
                B,
                p,
                args.gamma,
                args.alpha,
                config=config,
                support_gap_func=supported_profile_gap_cvar_dual,
                support_gap_kwargs={"maxiter": args.support_maxiter},
            )
        if method == "restricted_mcp":
            return small_support_action_search_cvar(
                A,
                B,
                p,
                args.gamma,
                args.alpha,
                config=config,
                support_gap_func=supported_profile_gap_cvar_mcp,
            )
    raise ValueError(f"Unsupported support method/risk pair: {method}/{risk}")


def _run_stochastic_method(
    method: str, risk: str, A, B, p, args: argparse.Namespace, seed: int
):
    config = _stochastic_config(args, A.shape[0], seed, method)
    if risk == "msd":
        return solve_msd_stochastic_fo(A, B, p, gamma=args.gamma, config=config)
    return solve_cvar_stochastic_fo(
        A, B, p, gamma=args.gamma, alpha=args.alpha, config=config
    )


def _run_method(
    method: str, risk: str, A, B, p, args: argparse.Namespace, seed: int, support_config
):
    start = perf_counter()
    try:
        if method == "mcp":
            result, cert = _solve_mcp(risk, A, B, p, args)
            return (
                result,
                _mcp_result_metrics(
                    method,
                    result,
                    cert,
                    perf_counter() - start,
                    None,
                    support_config.epsilon,
                ),
                None,
            )
        if method in {"screened_dual", "action_dual", "restricted_mcp"}:
            result = _run_support_method(method, risk, A, B, p, args, support_config)
            return (
                result,
                _support_result_metrics(method, result, perf_counter() - start, None),
                None,
            )
        if method in {"stochastic_full_batch", "stochastic_minibatch"}:
            result = _run_stochastic_method(method, risk, A, B, p, args, seed)
            return (
                result,
                _stochastic_result_metrics(
                    method, result, perf_counter() - start, None
                ),
                None,
            )
        raise ValueError(f"Unknown method: {method}")
    except Exception as exc:  # pragma: no cover - experiment diagnostics
        elapsed = perf_counter() - start
        return None, _empty_method_metrics(method, elapsed, str(exc)), str(exc)


def _result_profile(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    return _profile_from_result(result)


def _add_pairwise_metrics(
    row: dict[str, Any], results: dict[str, Any], methods: list[str]
) -> None:
    base = results.get("mcp")
    base_x, base_y = _result_profile(base)
    for method in methods:
        if method == "mcp":
            continue
        row[f"eta_diff_{method}_minus_mcp"] = (
            row[f"{method}_eta"] - row["mcp_eta"] if "mcp" in methods else np.nan
        )
        row[f"time_ratio_{method}_over_mcp"] = (
            row[f"{method}_time_s"] / row["mcp_time_s"]
            if "mcp" in methods and row["mcp_time_s"] > 0
            else np.nan
        )
        x, y = _result_profile(results.get(method))
        if (
            base_x is not None
            and base_y is not None
            and x is not None
            and y is not None
        ):
            row[f"x_l1_{method}_minus_mcp"] = float(np.sum(np.abs(x - base_x)))
            row[f"y_l1_{method}_minus_mcp"] = float(np.sum(np.abs(y - base_y)))
        else:
            row[f"x_l1_{method}_minus_mcp"] = np.nan
            row[f"y_l1_{method}_minus_mcp"] = np.nan

    if "action_dual" in methods and "screened_dual" in methods:
        row["eta_diff_action_dual_minus_screened_dual"] = (
            row["action_dual_eta"] - row["screened_dual_eta"]
        )
        row["time_ratio_action_dual_over_screened_dual"] = (
            row["action_dual_time_s"] / row["screened_dual_time_s"]
            if row["screened_dual_time_s"] > 0
            else np.nan
        )
    if "restricted_mcp" in methods and "action_dual" in methods:
        row["eta_diff_restricted_mcp_minus_action_dual"] = (
            row["restricted_mcp_eta"] - row["action_dual_eta"]
        )
        row["time_ratio_restricted_mcp_over_action_dual"] = (
            row["restricted_mcp_time_s"] / row["action_dual_time_s"]
            if row["action_dual_time_s"] > 0
            else np.nan
        )
    if "stochastic_minibatch" in methods and "stochastic_full_batch" in methods:
        row["eta_diff_stochastic_minibatch_minus_full_batch"] = (
            row["stochastic_minibatch_eta"] - row["stochastic_full_batch_eta"]
        )
        row["time_ratio_stochastic_minibatch_over_full_batch"] = (
            row["stochastic_minibatch_time_s"] / row["stochastic_full_batch_time_s"]
            if row["stochastic_full_batch_time_s"] > 0
            else np.nan
        )


def run_instance(
    args: argparse.Namespace, risk: str, K: int, n: int, seed: int
) -> dict[str, Any]:
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}; got {risk!r}.")
    A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
    kappa, tau = support_sizes(K, n)
    support_config = _support_config(args, K, n, seed)
    row: dict[str, Any] = {
        "risk": risk,
        "K": K,
        "n": n,
        "seed": seed,
        "gamma": args.gamma,
        "alpha": args.alpha if risk == "cvar" else np.nan,
        "epsilon": args.epsilon,
        "epsilon_scr": support_config.epsilon_scr,
        "support_kappa": kappa,
        "support_tau": tau,
        "kappa": kappa,
        "tau": tau,
        "max_candidates": args.max_candidates,
        "n_screen_starts": args.n_screen_starts,
        "n_support_starts": args.n_support_starts,
        "screen_maxiter": args.screen_maxiter,
        "support_maxiter": args.support_maxiter,
        "stochastic_max_iter": args.max_iter,
        "stochastic_minibatch_size": stochastic_minibatch_size(args, K),
        "stochastic_entropy_kappa": args.entropy_kappa,
        "stochastic_smoothing_tau": args.smoothing_tau,
        "stochastic_step_size": args.step_size,
        "stochastic_step_decay": args.step_decay,
        "stochastic_logit_bound": args.logit_bound,
        "stochastic_gradient_clip_norm": args.gradient_clip_norm,
        "stochastic_record_every": args.record_every,
        "stochastic_certify_every": args.certify_every,
        "methods": ",".join(args.methods),
    }
    results: dict[str, Any] = {}
    for method in args.methods:
        result, metrics, _ = _run_method(
            method, risk, A, B, p, args, seed, support_config
        )
        results[method] = result
        row.update(metrics)
    _add_pairwise_metrics(row, results, args.methods)
    return row


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


def run_experiment(
    args: argparse.Namespace,
    row_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rep_range = rep_indices(args)
    for risk in args.risk:
        for K in args.K:
            for n in args.n:
                for rep in rep_range:
                    seed = experiment_seed(risk, K, n, rep, args.seed_base)
                    row = run_instance(args, risk, K, n, seed)
                    rows.append(row)
                    if row_callback is not None:
                        row_callback(row)
                    if not args.quiet:
                        parts = [
                            f"{method}: ok={row[f'{method}_success']} eta={row[f'{method}_eta']:.4g} "
                            f"time={row[f'{method}_time_s']:.3f}s"
                            for method in args.methods
                        ]
                        print(
                            f"{risk} K={K:>3} n={n:>3} rep={rep:>2} "
                            + " | ".join(parts)
                        )
    return rows


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([row.get(key, np.nan) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def print_summary(rows: list[dict[str, Any]], methods: list[str]) -> None:
    print("\nSummary")
    print("-------")
    for risk in sorted({row["risk"] for row in rows}):
        print(risk.upper())
        risk_rows = [row for row in rows if row["risk"] == risk]
        for method in methods:
            success = np.array(
                [bool(row[f"{method}_success"]) for row in risk_rows], dtype=float
            )
            times = _finite_values(risk_rows, f"{method}_time_s")
            etas = _finite_values(risk_rows, f"{method}_eta")
            time_text = f"{np.median(times):.4g}s" if times.size else "nan"
            eta_text = f"{np.median(etas):.4g}" if etas.size else "nan"
            print(
                f"  {method:>22}: success_rate={success.mean():.3f}, median_time={time_text}, median_eta={eta_text}"
            )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", nargs="+", choices=RISKS, default=["msd", "cvar"])
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--epsilon-scr", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--n-screen-starts", type=int, default=1)
    parser.add_argument("--n-support-starts", type=int, default=5)
    parser.add_argument("--screen-maxiter", type=int, default=300)
    parser.add_argument("--support-maxiter", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--entropy-kappa", type=float, default=0.05)
    parser.add_argument("--smoothing-tau", type=float, default=0.1)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--logit-bound", type=float, default=20.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--certify-every", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--solver", default="pathampl")
    parser.add_argument("--fallback-solver", default="ipopt")
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.fallback_solver is not None and args.fallback_solver.lower() == "none":
        args.fallback_solver = None
    args.path_options = DEFAULT_PATH_OPTIONS.copy()
    try:
        rep_indices(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    if args.csv is not None:
        with StreamingCsvWriter(args.csv) as writer:
            rows = run_experiment(args, row_callback=writer.write_row)
    else:
        rows = run_experiment(args)
    print_summary(rows, args.methods)
    if args.csv is not None:
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
