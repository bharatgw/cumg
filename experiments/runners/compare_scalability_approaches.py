"""Compare scalability of package approaches for MSD and CVaR games."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.common.experiment import (
    DEFAULT_METHODS,
    DEFAULT_PATH_OPTIONS,
    METHODS,
    PAYOFF_MODELS,
    PAYOFF_POPULATIONS,
    RISKS,
    StreamingCsvWriter,
    _finite_values,
    experiment_seed,
    optional_positive_float,
    optional_positive_int,
    rep_indices,
    simulate_population_payoffs,
    simulate_random_payoffs,
    stochastic_minibatch_size,
    stochastic_regret_tolerance,
)
from experiments.common.provenance import save_profile

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cumg import (  # noqa: E402
    StochasticFOConfig,
    solve_cvar_mcp,
    solve_cvar_qptas,
    solve_cvar_qptas_screened,
    solve_cvar_stochastic_fo,
    solve_msd_mcp,
    solve_msd_qptas,
    solve_msd_qptas_screened,
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

# Versioned catalogue: IDs 0..3 are Beta(a, b); ID 4 is Uniform[0, 1].


def _solver_options(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path_options = getattr(args, "path_options", DEFAULT_PATH_OPTIONS)
    if args.solver != "pathampl" or path_options is None:
        return {}
    return {"pathampl": dict(path_options)}


def _support_config(args: argparse.Namespace, K: int, n: int, seed: int) -> SupportSearchConfig:
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


def _stochastic_config(args: argparse.Namespace, K: int, seed: int, method: str) -> StochasticFOConfig:
    if method == "stochastic_full_batch":
        batch_size = None
    elif method == "stochastic_minibatch":
        batch_size = stochastic_minibatch_size(args, K)
    else:
        raise ValueError(f"Unknown stochastic method: {method}")
    return StochasticFOConfig(
        capture_certificates=getattr(args, "save_profiles", False),
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
        regret_tolerance=stochastic_regret_tolerance(args),
        n_random_starts=getattr(args, "stochastic_n_random_starts", 4),
        stagnation_window=optional_positive_int(getattr(args, "stagnation_window", None)),
        stagnation_rtol=getattr(args, "stagnation_rtol", 0.0),
        stagnation_atol=getattr(args, "stagnation_atol", 0.0),
        jit_updates=getattr(args, "jit_updates", False),
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
    strategies = getattr(result, "strategies", None)
    if strategies is not None:
        x, y = strategies
    if x is None or y is None:
        return None, None
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _empty_method_metrics(prefix: str, elapsed_s: float, error: str | None) -> dict[str, Any]:
    out = {
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
        f"{prefix}_best_screen_eta": np.nan,
        f"{prefix}_best_screen_success": None,
        f"{prefix}_best_screen_violation": np.nan,
        f"{prefix}_best_screen_candidate_index": None,
        f"{prefix}_best_screen_message": None,
        f"{prefix}_support_violation": np.nan,
        f"{prefix}_residual_norm": np.nan,
        f"{prefix}_objective": np.nan,
        f"{prefix}_iterations": None,
        f"{prefix}_best_certificate_eta": np.nan,
        f"{prefix}_best_certificate_iteration": None,
    }
    if prefix in {"qptas", "qptas_screened"}:
        out.update(
            {
                f"{prefix}_profiles_checked": None,
                f"{prefix}_total_profiles": None,
                f"{prefix}_best_response_solves": None,
                f"{prefix}_sampling_seed": None,
                f"{prefix}_termination_reason": "error" if error is not None else None,
            }
        )
    if prefix == "qptas_screened":
        out.update({f"{prefix}_{key}": None for key in ("total_pairs", "screen_rejections", "screen_passes")})
    if prefix in ("stochastic_full_batch", "stochastic_minibatch"):
        out.update(
            {
                f"{prefix}_starts_attempted": 0,
                f"{prefix}_selected_start": None,
                f"{prefix}_start_summaries": "[]",
                f"{prefix}_termination_reason": "error" if error is not None else None,
            }
        )
    return out


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
            f"{prefix}_success": bool(np.isfinite(cert.get("eta", np.nan)) & (cert.get("eta", np.nan) <= eps)),
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
    best_screen = result.metadata.get("best_screen", {}) if result is not None else {}
    if not best_screen and candidate is not None:
        best_screen = candidate.get("best_screen", candidate.get("screen", {}))
    out.update(
        {
            f"{prefix}_best_screen_eta": float(best_screen.get("eta", np.nan)),
            f"{prefix}_best_screen_success": best_screen.get("success"),
            f"{prefix}_best_screen_violation": float(best_screen.get("violation", np.nan)),
            f"{prefix}_best_screen_candidate_index": best_screen.get("candidate_index"),
            f"{prefix}_best_screen_message": best_screen.get("message"),
        }
    )
    if candidate is None:
        out[f"{prefix}_screen_eta"] = float(best_screen.get("eta", np.nan))
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


def _stochastic_result_metrics(prefix: str, result, elapsed_s: float, error: str | None):
    out = _empty_method_metrics(prefix, elapsed_s, error)
    cert = result.certificate if result is not None else {}
    x, y = _profile_from_result(result)
    best_certificate = result.best_certificate if result is not None and result.best_certificate is not None else {}
    out.update(_certificate_metrics(prefix, cert))
    out.update(
        {
            f"{prefix}_success": bool(result.success) if result is not None else False,
            f"{prefix}_has_profile": x is not None and y is not None,
            f"{prefix}_residual_norm": (float(result.residual_norm) if result is not None else np.nan),
            f"{prefix}_objective": (float(result.objective) if result is not None else np.nan),
            f"{prefix}_iterations": result.iterations if result is not None else None,
            f"{prefix}_best_certificate_eta": float(best_certificate.get("eta", np.nan)),
            f"{prefix}_best_certificate_iteration": best_certificate.get("iteration"),
            f"{prefix}_starts_attempted": (len(result.start_summaries) or 1) if result is not None else 0,
            f"{prefix}_selected_start": result.selected_start if result is not None else None,
            f"{prefix}_start_summaries": json.dumps(result.start_summaries) if result is not None else "[]",
            f"{prefix}_termination_reason": result.termination_reason if result is not None else "error",
        }
    )
    return out


def _qptas_result_metrics(result, elapsed_s: float, prefix: str = "qptas") -> dict[str, Any]:
    out = _empty_method_metrics(prefix, elapsed_s, None)
    cert = result.certificate or {}
    regrets = cert.get("regrets", (np.nan, np.nan))
    out.update(
        _certificate_metrics(prefix, {"eta": cert.get("eta", np.nan), "regret1": regrets[0], "regret2": regrets[1]})
    )
    x, y = _profile_from_result(result)
    out.update(
        {
            f"{prefix}_success": result.success,
            f"{prefix}_has_profile": x is not None and y is not None,
            f"{prefix}_solver": "highs",
            f"{prefix}_profiles_checked": result.profiles_checked,
            f"{prefix}_total_profiles": result.total_profiles,
            f"{prefix}_best_response_solves": result.best_response_solves,
            f"{prefix}_sampling_seed": result.seed,
            f"{prefix}_termination_reason": result.termination_reason,
        }
    )
    if prefix == "qptas_screened":
        out.update(
            {f"{prefix}_{key}": getattr(result, key) for key in ("total_pairs", "screen_rejections", "screen_passes")}
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


def _run_stochastic_method(method: str, risk: str, A, B, p, args: argparse.Namespace, seed: int):
    config = _stochastic_config(args, A.shape[0], seed, method)
    if risk == "msd":
        return solve_msd_stochastic_fo(A, B, p, gamma=args.gamma, config=config)
    return solve_cvar_stochastic_fo(A, B, p, gamma=args.gamma, alpha=args.alpha, config=config)


def _run_method(method: str, risk: str, A, B, p, args: argparse.Namespace, seed: int, support_config):
    start = perf_counter()
    try:
        if method in {"qptas", "qptas_screened"}:
            kwargs = {
                "gamma": args.gamma,
                "kappa": support_config.kappa,
                "epsilon": args.epsilon,
                "max_candidates": args.max_candidates,
                "seed": seed,
            }
            if method == "qptas_screened":
                kwargs.update(tau=support_config.tau, epsilon_scr=support_config.epsilon_scr)
                solve = solve_msd_qptas_screened if risk == "msd" else solve_cvar_qptas_screened
            else:
                solve = solve_msd_qptas if risk == "msd" else solve_cvar_qptas
            if risk == "cvar":
                kwargs["alpha"] = args.alpha
            result = solve([A, B], p, **kwargs)
            return result, _qptas_result_metrics(result, perf_counter() - start, method), None
        if method == "mcp":
            result, cert = _solve_mcp(risk, A, B, p, args)
            elapsed = perf_counter() - start
            if getattr(args, "save_profiles", False) and hasattr(result, "extra"):
                result.extra["certificate"] = cert
            return (
                result,
                _mcp_result_metrics(
                    method,
                    result,
                    cert,
                    elapsed,
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
                _stochastic_result_metrics(method, result, perf_counter() - start, None),
                None,
            )
        raise ValueError(f"Unknown method: {method}")
    except Exception as exc:  # pragma: no cover - experiment diagnostics
        elapsed = perf_counter() - start
        return None, _empty_method_metrics(method, elapsed, str(exc)), str(exc)


def _result_profile(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    return _profile_from_result(result)


def _add_pairwise_metrics(row: dict[str, Any], results: dict[str, Any], methods: list[str]) -> None:
    base = results.get("mcp")
    base_x, base_y = _result_profile(base)
    for method in methods:
        if method == "mcp":
            continue
        row[f"eta_diff_{method}_minus_mcp"] = row[f"{method}_eta"] - row["mcp_eta"] if "mcp" in methods else np.nan
        row[f"time_ratio_{method}_over_mcp"] = (
            row[f"{method}_time_s"] / row["mcp_time_s"] if "mcp" in methods and row["mcp_time_s"] > 0 else np.nan
        )
        x, y = _result_profile(results.get(method))
        if base_x is not None and base_y is not None and x is not None and y is not None:
            row[f"x_l1_{method}_minus_mcp"] = float(np.sum(np.abs(x - base_x)))
            row[f"y_l1_{method}_minus_mcp"] = float(np.sum(np.abs(y - base_y)))
        else:
            row[f"x_l1_{method}_minus_mcp"] = np.nan
            row[f"y_l1_{method}_minus_mcp"] = np.nan

    if "action_dual" in methods and "screened_dual" in methods:
        row["eta_diff_action_dual_minus_screened_dual"] = row["action_dual_eta"] - row["screened_dual_eta"]
        row["time_ratio_action_dual_over_screened_dual"] = (
            row["action_dual_time_s"] / row["screened_dual_time_s"] if row["screened_dual_time_s"] > 0 else np.nan
        )
    if "restricted_mcp" in methods and "action_dual" in methods:
        row["eta_diff_restricted_mcp_minus_action_dual"] = row["restricted_mcp_eta"] - row["action_dual_eta"]
        row["time_ratio_restricted_mcp_over_action_dual"] = (
            row["restricted_mcp_time_s"] / row["action_dual_time_s"] if row["action_dual_time_s"] > 0 else np.nan
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


def run_instance(args: argparse.Namespace, risk: str, K: int, n: int, seed: int) -> dict[str, Any]:
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}; got {risk!r}.")
    payoff_model = getattr(args, "payoff_model", "uniform")
    if payoff_model == "uniform":
        A, B, p = simulate_random_payoffs(K=K, n=n, seed=seed, low=args.low, high=args.high)
        population_ids = None
    elif payoff_model == "cell_beta_uniform_v1":
        if args.low != 0.0 or args.high != 1.0:
            raise ValueError("cell_beta_uniform_v1 requires low=0 and high=1.")
        A, B, p, population_ids = simulate_population_payoffs(K=K, n=n, seed=seed)
    else:
        raise ValueError(f"Unknown payoff model: {payoff_model}")
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
        "payoff_model": payoff_model,
        "payoff_populations": json.dumps(PAYOFF_POPULATIONS) if population_ids is not None else "",
        "payoff_population_ids": json.dumps(population_ids.tolist()) if population_ids is not None else "",
        "payoff_numpy_version": np.__version__,
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
        "stochastic_theta_step_size": args.step_size if risk == "cvar" else None,
        "stochastic_step_decay": args.step_decay,
        "stochastic_logit_bound": args.logit_bound,
        "stochastic_gradient_clip_norm": args.gradient_clip_norm,
        "stochastic_record_every": args.record_every,
        "stochastic_certify_every": args.certify_every,
        "stochastic_regret_tolerance": stochastic_regret_tolerance(args),
        "stochastic_n_random_starts": getattr(args, "stochastic_n_random_starts", 4),
        "stochastic_stagnation_window": optional_positive_int(getattr(args, "stagnation_window", None)),
        "stochastic_stagnation_rtol": getattr(args, "stagnation_rtol", 0.0),
        "stochastic_stagnation_atol": getattr(args, "stagnation_atol", 0.0),
        "stochastic_jit_updates": getattr(args, "jit_updates", False),
        "methods": ",".join(args.methods),
    }
    results: dict[str, Any] = {}
    if getattr(args, "save_profiles", False) and not any(method.startswith("stochastic_") for method in args.methods):
        for key in row:
            if key.startswith("stochastic_"):
                row[key] = None
    for method in args.methods:
        result, metrics, _ = _run_method(method, risk, A, B, p, args, seed, support_config)
        results[method] = result
        row.update(metrics)
        if getattr(args, "save_profiles", False) and args.csv is not None:
            stem = args.csv.stem.removesuffix(".partial")
            path = args.csv.parent / "profiles" / f"{stem}__{risk}_K{K}_n{n}_seed{seed}__{method}.json"
            save_profile(
                path,
                args=args,
                row=row,
                method=method,
                result=result,
                profile=_profile_from_result(result),
                arrays=(A, B, p),
            )
            row[f"{method}_profile_artifact"] = str(path.relative_to(args.csv.parent))
    _add_pairwise_metrics(row, results, args.methods)
    return row


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
                        print(f"{risk} K={K:>3} n={n:>3} rep={rep:>2} " + " | ".join(parts))
    return rows


def print_summary(rows: list[dict[str, Any]], methods: list[str]) -> None:
    print("\nSummary")
    print("-------")
    for risk in sorted({row["risk"] for row in rows}):
        print(risk.upper())
        risk_rows = [row for row in rows if row["risk"] == risk]
        for method in methods:
            success = np.array([bool(row[f"{method}_success"]) for row in risk_rows], dtype=float)
            times = _finite_values(risk_rows, f"{method}_time_s")
            etas = _finite_values(risk_rows, f"{method}_eta")
            time_text = f"{np.median(times):.4g}s" if times.size else "nan"
            eta_text = f"{np.median(etas):.4g}" if etas.size else "nan"
            print(f"  {method:>22}: success_rate={success.mean():.3f}, median_time={time_text}, median_eta={eta_text}")


def parse_args(argv=None) -> argparse.Namespace:
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
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--stochastic-regret-tolerance",
        "--stochastic-epsilon",
        dest="stochastic_regret_tolerance",
        type=float,
        default=None,
        help="Stochastic-method regret tolerance; defaults to --epsilon.",
    )
    parser.add_argument("--epsilon-scr", type=float, default=None)
    parser.add_argument(
        "--stochastic-n-random-starts",
        "--n-random-starts",
        type=int,
        default=4,
        help="Additional FO starts after failure (0 keeps the single uniform start).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Support-search budget, random profiles for qptas, or joint (x,q) pairs for qptas_screened.",
    )
    parser.add_argument("--n-screen-starts", type=int, default=1)
    parser.add_argument("--n-support-starts", type=int, default=5)
    parser.add_argument("--screen-maxiter", type=int, default=300)
    parser.add_argument("--support-maxiter", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--entropy-kappa", type=float, default=0.1)
    parser.add_argument("--smoothing-tau", type=float, default=0.02)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--logit-bound", type=float, default=20.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--certify-every", type=int, default=0)
    parser.add_argument("--stagnation-window", type=int, default=None, help="FO stagnation window; 0 disables it.")
    parser.add_argument("--stagnation-rtol", type=float, default=0.0)
    parser.add_argument("--stagnation-atol", type=float, default=0.0)
    parser.add_argument("--jit-updates", action="store_true", help="Compile FO updates with JAX, as in the pilot.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--solver", default="pathampl")
    parser.add_argument("--fallback-solver", default="ipopt")
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--payoff-model", choices=PAYOFF_MODELS, default="uniform")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--save-profiles", action="store_true", help="Write versioned profile/provenance sidecars.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit nonzero after writing diagnostics if any method raises an error; search exhaustion is not an error.",
    )
    args = parser.parse_args(argv)
    if args.fallback_solver is not None and args.fallback_solver.lower() == "none":
        args.fallback_solver = None
    args.path_options = DEFAULT_PATH_OPTIONS.copy()
    if args.payoff_model != "uniform" and (args.low != 0.0 or args.high != 1.0):
        parser.error("cell_beta_uniform_v1 requires --low 0 and --high 1")
    if args.stochastic_n_random_starts < 0:
        parser.error("stochastic-n-random-starts must be nonnegative")
    if args.stochastic_regret_tolerance is not None and (
        not np.isfinite(args.stochastic_regret_tolerance) or args.stochastic_regret_tolerance < 0
    ):
        parser.error("stochastic regret tolerance must be finite and nonnegative")
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
    if args.fail_on_error:
        errors = [
            f"{row['risk']} K={row['K']} n={row['n']} seed={row['seed']} {method}: {row[f'{method}_error']}"
            for row in rows
            for method in args.methods
            if row.get(f"{method}_error")
        ]
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
