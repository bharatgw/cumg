"""Compare stochastic first-order settings for MSD and CVaR games."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for import_path in (SRC, EXPERIMENTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from compare_scalability_approaches import (  # noqa: E402
    RISKS,
    StreamingCsvWriter,
    experiment_seed,
    optional_positive_float,
    optional_positive_int,
    simulate_random_payoffs,
    stochastic_minibatch_size,
)

from cumg import (  # noqa: E402
    StochasticFOConfig,
    solve_cvar_stochastic_fo,
    solve_msd_stochastic_fo,
)


def _continuation_stages(args: argparse.Namespace) -> list[tuple[float, float, int]]:
    kappas = getattr(args, "continuation_kappa", None)
    taus = getattr(args, "continuation_tau", None)
    max_iters = getattr(args, "continuation_max_iter", None)
    if kappas is None and taus is None and max_iters is None:
        return [(args.entropy_kappa, args.smoothing_tau, args.max_iter)]
    if kappas is None or taus is None:
        raise ValueError("continuation requires both --continuation-kappa and --continuation-tau")
    if args.entropy_kappa_grid is not None or args.smoothing_tau_grid is not None:
        raise ValueError("continuation cannot be combined with kappa or tau tuning grids")
    if len(kappas) != len(taus):
        raise ValueError("continuation kappa and tau schedules must have the same length")
    if max_iters is None:
        max_iters = [args.max_iter] * len(kappas)
    if len(max_iters) != len(kappas):
        raise ValueError("continuation max-iter schedule must match the kappa and tau schedules")
    if not kappas:
        raise ValueError("continuation schedules cannot be empty")
    if any(not np.isfinite(value) or value <= 0 for value in (*kappas, *taus)):
        raise ValueError("continuation kappa and tau values must be finite and positive")
    if any(value < 0 for value in max_iters):
        raise ValueError("continuation max-iter values must be nonnegative")
    return list(zip(kappas, taus, max_iters, strict=True))


def _method_config(
    args: argparse.Namespace,
    K: int,
    seed: int,
    method: str,
    *,
    kappa: float | None = None,
    tau: float | None = None,
    max_iter: int | None = None,
    x0=None,
    y0=None,
    theta0=None,
) -> StochasticFOConfig:
    if method == "full_batch":
        batch_size = None
    elif method == "minibatch":
        batch_size = stochastic_minibatch_size(args, K)
    else:
        raise ValueError(f"Unknown method: {method}")

    return StochasticFOConfig(
        kappa=args.entropy_kappa if kappa is None else kappa,
        tau=args.smoothing_tau if tau is None else tau,
        max_iter=args.max_iter if max_iter is None else max_iter,
        batch_size=batch_size,
        step_size=args.step_size,
        step_decay=args.step_decay,
        seed=seed,
        logit_bound=optional_positive_float(args.logit_bound),
        gradient_clip_norm=optional_positive_float(args.gradient_clip_norm),
        x0=x0,
        y0=y0,
        theta0=theta0,
        record_every=optional_positive_int(args.record_every),
        certify_every=optional_positive_int(args.certify_every),
        regret_tolerance=args.regret_tolerance,
    )


def _run_method(
    risk: str,
    name: str,
    A,
    B,
    p,
    gamma: float,
    alpha: float,
    args: argparse.Namespace,
    seed: int,
):
    start = perf_counter()
    stage_runs = []
    iteration_offset = 0
    x0 = None
    y0 = None
    theta0 = None
    for stage, (kappa, tau, max_iter) in enumerate(_continuation_stages(args)):
        config = _method_config(
            args,
            A.shape[0],
            seed,
            name,
            kappa=kappa,
            tau=tau,
            max_iter=max_iter,
            x0=x0,
            y0=y0,
            theta0=theta0,
        )
        if risk == "msd":
            result = solve_msd_stochastic_fo(A, B, p, gamma=gamma, config=config)
        else:
            result = solve_cvar_stochastic_fo(A, B, p, gamma=gamma, alpha=alpha, config=config)
        stage_runs.append(
            {
                "stage": stage,
                "kappa": kappa,
                "tau": tau,
                "max_iter": max_iter,
                "iteration_offset": iteration_offset,
                "result": result,
            }
        )
        iteration_offset += result.iterations
        if result.success:
            break
        x0 = result.x
        y0 = result.y
        theta0 = result.theta

    selected_run = min(
        stage_runs,
        key=lambda run: np.nan_to_num(
            float(run["result"].certificate.get("eta", np.inf)),
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        ),
    )
    selected_run["selected"] = True
    return selected_run["result"], perf_counter() - start, None, stage_runs


def _method_metrics(
    prefix: str,
    result,
    elapsed_s: float,
    error: str | None = None,
    stage_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cert = result.certificate if result is not None else {}
    best_iterate = result.best_iterate if result is not None else {}
    best_certificate = result.best_certificate if result is not None and result.best_certificate is not None else {}
    theta = result.theta if result is not None and result.theta is not None else (np.nan, np.nan)
    stage_runs = stage_runs or []
    selected_run = next((run for run in stage_runs if run.get("selected")), None)
    total_iterations = sum(run["result"].iterations for run in stage_runs)
    total_solve_time = sum(run["result"].solve_time_s for run in stage_runs)
    total_history_len = sum(len(run["result"].history) for run in stage_runs)
    selected_stage = selected_run["stage"] if selected_run is not None else 0
    selected_offset = selected_run["iteration_offset"] if selected_run is not None else 0
    selected_kappa = selected_run["kappa"] if selected_run is not None else np.nan
    selected_tau = selected_run["tau"] if selected_run is not None else np.nan
    certificate_iteration = best_certificate.get("iteration")
    if certificate_iteration is not None:
        certificate_iteration += selected_offset
    method_error = error

    return {
        f"{prefix}_success": bool(result.success) if result is not None else False,
        f"{prefix}_time_s": elapsed_s,
        f"{prefix}_solve_time_s": float(total_solve_time) if stage_runs else float(result.solve_time_s),
        f"{prefix}_eta": float(cert.get("eta", np.nan)),
        f"{prefix}_regret1": float(cert.get("regret1", np.nan)),
        f"{prefix}_regret2": float(cert.get("regret2", np.nan)),
        f"{prefix}_residual_norm": (float(result.residual_norm) if result is not None else np.nan),
        f"{prefix}_objective": (float(result.objective) if result is not None else np.nan),
        f"{prefix}_iterations": total_iterations if stage_runs else result.iterations,
        f"{prefix}_has_profile": result is not None and result.x is not None and result.y is not None,
        f"{prefix}_history_len": total_history_len if stage_runs else len(result.history),
        f"{prefix}_best_residual_norm": float(best_iterate.get("residual_norm", np.nan)),
        f"{prefix}_best_objective": float(best_iterate.get("objective", np.nan)),
        f"{prefix}_best_certificate_eta": float(best_certificate.get("eta", np.nan)),
        f"{prefix}_best_certificate_iteration": certificate_iteration,
        f"{prefix}_selected_stage": selected_stage,
        f"{prefix}_selected_kappa": selected_kappa,
        f"{prefix}_selected_tau": selected_tau,
        f"{prefix}_stages_completed": len(stage_runs) if stage_runs else 1,
        f"{prefix}_theta1": float(theta[0]),
        f"{prefix}_theta2": float(theta[1]),
        f"{prefix}_error": method_error,
    }


def _pairwise_metrics(row: dict[str, Any], results: dict[str, Any]) -> None:
    if "full_batch" not in results or "minibatch" not in results:
        return
    row["eta_diff_minibatch_minus_full_batch"] = row["minibatch_eta"] - row["full_batch_eta"]
    row["residual_diff_minibatch_minus_full_batch"] = row["minibatch_residual_norm"] - row["full_batch_residual_norm"]
    row["time_ratio_minibatch_over_full_batch"] = (
        row["minibatch_time_s"] / row["full_batch_time_s"] if row["full_batch_time_s"] > 0 else np.nan
    )

    full_batch = results["full_batch"]
    minibatch = results["minibatch"]
    if full_batch is not None and minibatch is not None and full_batch.x is not None and minibatch.x is not None:
        row["x_l1_minibatch_minus_full_batch"] = float(np.sum(np.abs(minibatch.x - full_batch.x)))
        row["y_l1_minibatch_minus_full_batch"] = float(np.sum(np.abs(minibatch.y - full_batch.y)))
    else:
        row["x_l1_minibatch_minus_full_batch"] = np.nan
        row["y_l1_minibatch_minus_full_batch"] = np.nan


def _history_rows(
    args: argparse.Namespace,
    risk: str,
    K: int,
    n: int,
    seed: int,
    method: str,
    stage_run: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    result = stage_run["result"]
    if result is None:
        return
    for checkpoint in result.history:
        theta = checkpoint.get("theta", (np.nan, np.nan))
        yield {
            "risk": risk,
            "K": K,
            "n": n,
            "seed": seed,
            "gamma": args.gamma,
            "alpha": args.alpha if risk == "cvar" else np.nan,
            "entropy_kappa": stage_run["kappa"],
            "smoothing_tau": stage_run["tau"],
            "step_size": args.step_size,
            "step_decay": args.step_decay,
            "max_iter": stage_run["max_iter"],
            "minibatch_size": stochastic_minibatch_size(args, K),
            "regret_tolerance": args.regret_tolerance,
            "record_every": args.record_every,
            "certify_every": args.certify_every,
            "method": method,
            "continuation_stage": stage_run["stage"],
            "stage_iteration": checkpoint["iteration"],
            "iteration": stage_run["iteration_offset"] + checkpoint["iteration"],
            "selected_stage": bool(stage_run.get("selected", False)),
            "residual_norm": checkpoint["residual_norm"],
            "objective": checkpoint["objective"],
            "eta": checkpoint.get("eta", np.nan),
            "regret1": checkpoint.get("regret1", np.nan),
            "regret2": checkpoint.get("regret2", np.nan),
            "theta1": float(theta[0]),
            "theta2": float(theta[1]),
        }


def run_instance(
    args: argparse.Namespace,
    risk: str,
    K: int,
    n: int,
    seed: int,
    history_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}; got {risk!r}.")
    stages = _continuation_stages(args)
    A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
    row: dict[str, Any] = {
        "risk": risk,
        "K": K,
        "n": n,
        "seed": seed,
        "gamma": args.gamma,
        "alpha": args.alpha if risk == "cvar" else np.nan,
        "entropy_kappa": stages[0][0],
        "smoothing_tau": stages[0][1],
        "max_iter": stages[0][2],
        "minibatch_size": stochastic_minibatch_size(args, K),
        "step_size": args.step_size,
        "step_decay": args.step_decay,
        "regret_tolerance": args.regret_tolerance,
        "record_every": args.record_every,
        "certify_every": args.certify_every,
        "continuation": getattr(args, "continuation_kappa", None) is not None,
        "continuation_stages": len(stages),
        "continuation_kappa": ",".join(f"{kappa:g}" for kappa, _, _ in stages),
        "continuation_tau": ",".join(f"{tau:g}" for _, tau, _ in stages),
        "continuation_max_iter": ",".join(str(max_iter) for _, _, max_iter in stages),
        "continuation_total_max_iter": sum(max_iter for _, _, max_iter in stages),
        "methods": ",".join(args.methods),
    }

    results = {}
    for method in args.methods:
        result, elapsed, error, stage_runs = _run_method(risk, method, A, B, p, args.gamma, args.alpha, args, seed)
        results[method] = result
        row.update(_method_metrics(method, result, elapsed, error, stage_runs))
        if history_callback is not None:
            for stage_run in stage_runs:
                for history_row in _history_rows(args, risk, K, n, seed, method, stage_run):
                    history_callback(history_row)
    _pairwise_metrics(row, results)
    return row


def _finite_values(rows: list[dict[str, Any]], key: str, risk: str) -> np.ndarray:
    vals = np.array([row[key] for row in rows if row["risk"] == risk], dtype=float)
    return vals[np.isfinite(vals)]


def print_summary(rows: list[dict[str, Any]], risks: list[str], methods: list[str]) -> None:
    print("\nSummary")
    print("-------")
    for risk in risks:
        print(f"\n{risk}")
        print("-------")
        for prefix in methods:
            success = np.array(
                [bool(row[f"{prefix}_success"]) for row in rows if row["risk"] == risk],
                dtype=float,
            )
            times = _finite_values(rows, f"{prefix}_time_s", risk)
            etas = _finite_values(rows, f"{prefix}_eta", risk)
            residuals = _finite_values(rows, f"{prefix}_residual_norm", risk)
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
            ratios = _finite_values(rows, "time_ratio_minibatch_over_full_batch", risk)
            eta_diffs = _finite_values(rows, "eta_diff_minibatch_minus_full_batch", risk)
            x_diffs = _finite_values(rows, "x_l1_minibatch_minus_full_batch", risk)
            y_diffs = _finite_values(rows, "y_l1_minibatch_minus_full_batch", risk)
            if ratios.size:
                print(f"minibatch/full_batch median time ratio: {np.median(ratios):.4g}")
            if eta_diffs.size:
                print(f"median eta difference, minibatch - full_batch: {np.median(eta_diffs):.4g}")
            if x_diffs.size and y_diffs.size:
                print(f"median strategy L1 difference: x={np.median(x_diffs):.4g}, y={np.median(y_diffs):.4g}")


def iter_tuning_configs(args: argparse.Namespace) -> Iterator[argparse.Namespace]:
    _continuation_stages(args)
    if getattr(args, "continuation_kappa", None) is not None:
        step_sizes = args.step_size_grid or [args.step_size]
        for step_size in step_sizes:
            config_args = argparse.Namespace(**vars(args))
            config_args.step_size = step_size
            yield config_args
        return
    kappas = args.entropy_kappa_grid or [args.entropy_kappa]
    taus = args.smoothing_tau_grid or [args.smoothing_tau]
    step_sizes = args.step_size_grid or [args.step_size]
    for kappa, tau, step_size in product(kappas, taus, step_sizes):
        config_args = argparse.Namespace(**vars(args))
        config_args.entropy_kappa = kappa
        config_args.smoothing_tau = tau
        config_args.step_size = step_size
        yield config_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", nargs="+", choices=RISKS, default=["msd", "cvar"])
    parser.add_argument("--K", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--n", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--entropy-kappa", type=float, default=0.2)
    parser.add_argument("--entropy-kappa-grid", type=float, nargs="+", default=None)
    parser.add_argument("--smoothing-tau", type=float, default=0.05)
    parser.add_argument("--smoothing-tau-grid", type=float, nargs="+", default=None)
    parser.add_argument("--continuation-kappa", type=float, nargs="+", default=None)
    parser.add_argument("--continuation-tau", type=float, nargs="+", default=None)
    parser.add_argument("--continuation-max-iter", type=int, nargs="+", default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--step-size-grid", type=float, nargs="+", default=None)
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
    parser.add_argument("--history-csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.history_csv is not None and args.record_every <= 0:
        parser.error("--history-csv requires a positive --record-every")
    try:
        _continuation_stages(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    rows = []
    with ExitStack() as stack:
        result_writer = stack.enter_context(StreamingCsvWriter(args.csv)) if args.csv is not None else None
        history_writer = (
            stack.enter_context(StreamingCsvWriter(args.history_csv)) if args.history_csv is not None else None
        )
        history_callback = history_writer.write_row if history_writer is not None else None

        for config_args in iter_tuning_configs(args):
            for risk in args.risk:
                for K in args.K:
                    for n in args.n:
                        for rep in range(args.reps):
                            seed = experiment_seed(risk, K, n, rep, args.seed_base)
                            row = run_instance(
                                config_args,
                                risk,
                                K,
                                n,
                                seed,
                                history_callback=history_callback,
                            )
                            rows.append(row)
                            if result_writer is not None:
                                result_writer.write_row(row)
                            if not args.quiet:
                                if row["continuation"]:
                                    setting_text = (
                                        f"continuation_kappa={row['continuation_kappa']} "
                                        f"continuation_tau={row['continuation_tau']} "
                                    )
                                else:
                                    setting_text = (
                                        f"kappa={config_args.entropy_kappa:g} tau={config_args.smoothing_tau:g} "
                                    )
                                method_parts = [
                                    f"{risk}, {method}: "
                                    f"success={row[f'{method}_success']} "
                                    f"eta={row[f'{method}_eta']:.4g} "
                                    f"resid={row[f'{method}_residual_norm']:.4g} time={row[f'{method}_time_s']:.3f}s"
                                    for method in args.methods
                                ]
                                print(
                                    setting_text + f"step={config_args.step_size:g} "
                                    f"K={K:>3} n={n:>3} rep={rep:>2} " + " | ".join(method_parts)
                                )

    print_summary(rows, args.risk, args.methods)
    if args.csv is not None:
        print(f"\nwrote {args.csv}")
    if args.history_csv is not None:
        print(f"wrote {args.history_csv}")


if __name__ == "__main__":
    main()
