"""Time the existing QPTAS search on paired random games, with a per-run cap."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import cumg.qptas as qptas  # noqa: E402
from experiments.common.experiment import StreamingCsvWriter, simulate_random_payoffs  # noqa: E402


def time_search(A, B, p, risk, args):
    """Count completed work; stop only between profiles, never truncate an LP."""

    original_profiles = qptas.enumerate_kappa_uniform_profiles
    lp_name = f"_maximize_{risk}_on_simplex"
    original_lp = getattr(qptas, lp_name)
    counts = {"profiles": 0, "lp_solves": 0, "cap_hit": False}
    start = perf_counter()

    def timed_profiles(action_sizes, kappa):
        for profile in original_profiles(action_sizes, kappa):
            if perf_counter() - start >= args.seconds:
                counts["cap_hit"] = True
                raise TimeoutError("Benchmark time budget reached between profiles.")
            yield profile
            counts["profiles"] += 1

    def counted_lp(*lp_args, **lp_kwargs):
        result = original_lp(*lp_args, **lp_kwargs)
        counts["lp_solves"] += 1
        return result

    result = None
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("once", RuntimeWarning)
        with patch.object(qptas, "enumerate_kappa_uniform_profiles", timed_profiles):
            with patch.object(qptas, lp_name, counted_lp):
                try:
                    kwargs = {"gamma": args.gamma, "kappa": args.kappa, "epsilon": args.epsilon}
                    if risk == "cvar":
                        result = qptas.solve_cvar_qptas([A, B], p, alpha=args.alpha, **kwargs)
                    else:
                        result = qptas.solve_msd_qptas([A, B], p, **kwargs)
                except TimeoutError:
                    if not counts["cap_hit"]:
                        raise
    elapsed = perf_counter() - start
    if result is not None:
        counts["profiles"] = result.profiles_checked
        assert counts["lp_solves"] == result.best_response_solves
    total_profiles = comb(args.n + args.kappa - 1, args.n - 1) ** 2
    rate = counts["profiles"] / elapsed
    row = {
        "risk": risk,
        "n": args.n,
        "K": args.K,
        "kappa": args.kappa,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "alpha": args.alpha if risk == "cvar" else "",
        "status": "time_cap" if counts["cap_hit"] else result.termination_reason,
        "success": bool(result is not None and result.success),
        "elapsed_s": elapsed,
        "time_cap_s": args.seconds,
        "profiles_checked": counts["profiles"],
        "lp_solves": counts["lp_solves"],
        "profiles_per_s": rate,
        "seconds_per_profile": 1.0 / rate if rate else None,
        "total_profiles": total_profiles,
        "full_grid_years_at_observed_rate": total_profiles / rate / (365.25 * 24 * 3600) if rate else None,
        "eta": result.certificate["eta"] if result is not None and result.success else None,
        "warnings": json.dumps(sorted({str(w.message) for w in seen})),
    }
    if result is not None and result.success:
        row["strategies"] = json.dumps([strategy.tolist() for strategy in result.strategies])
    else:
        row["strategies"] = ""
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--kappa", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "payoffs": "iid Uniform[0,1]; same A and B for both risks at each seed",
        "probabilities": "uniform over all K scenarios",
        "timing": "sequential runs; excludes payoff generation and imports; cap checked between profiles",
        "extrapolation": "full-grid traversal at measured prefix throughput, not estimated time to first solution",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/cumg/qptas.py",
                "src/cumg/small_support.py",
                "experiments/runners/benchmark_qptas_runtime.py",
            )
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with StreamingCsvWriter(args.output_dir / "timings.csv") as writer:
        for seed in args.seeds:
            A, B, p = simulate_random_payoffs(args.K, args.n, seed)
            # Alternate order to reduce systematic warm-up/order effects.
            risks = ("msd", "cvar") if seed % 2 == 0 else ("cvar", "msd")
            for risk in risks:
                print(f"START seed={seed} risk={risk}", flush=True)
                row = {"seed": seed, **time_search(A, B, p, risk, args)}
                writer.write_row(row)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
