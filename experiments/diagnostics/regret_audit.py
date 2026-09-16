"""Recheck saved profiles and inventory missing certificates in presented results.

All LPs use the full game. Agreement is a fresh certificate computation using
the package LPs, not an independent LP implementation. Regenerated inputs with
unknown provenance are explicitly conditional evidence.
"""

import argparse
import csv
import gzip
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from cumg.small_support import full_cvar_regret, full_msd_regret
from experiments.common.experiment import simulate_population_payoffs, simulate_random_payoffs
from experiments.common.paths import REPO, result_path
from experiments.common.provenance import array_hash, environment, json_value

ATOL, RTOL = 1e-8, 1e-6
# These are the actual README/notebook input datasets. Calibration profiles are
# included as additional evidence; archived unrelated experiments are not pooled.
PRESENTATIONS = {
    "scalability/msd_cvar_original": "*K*_n*.csv",
    "scalability/cvar_capped_24h_v1": "capped_method_results.csv",
    "scalability/equal_screen_v1": "*K*_n*.csv",
    "qptas_scalability/sampled_1000_v1": "capped_method_results.csv",
    "population_qptas_fo/beta_uniform_v2": "capped_method_results.csv",
    "uniform_baseline/v1": "uniform_profile_baseline.csv",
    "stochastic_fo/continuation_v2": "*_summary.csv",
    "stochastic_fo/continuation_v3": "*_summary.csv",
    "population_fo_calibration/pilot_v1": "*.csv",
    "population_fo_calibration/cvar_steps_v2": "*.csv",
    "archive/notebook_results": "prisonersDilemmaMSD.csv",
}


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_profile(A, B, p, risk, gamma, alpha, x, y, saved, epsilon=0.01) -> dict:
    """Validate without repairs, solve full BRs, and separately check classification."""
    started = perf_counter()
    for name, value, size in (("x", x, A.shape[1]), ("y", y, A.shape[2])):
        value = np.asarray(value, dtype=float)
        if (
            value.shape != (size,)
            or not np.isfinite(value).all()
            or (value < 0).any()
            or not np.isclose(value.sum(), 1.0, atol=1e-10, rtol=0)
        ):
            return {"status": "invalid_simplex", "detail": name}
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", RuntimeWarning)
            cert = (
                full_msd_regret(A, B, p, gamma, x, y)
                if risk == "msd"
                else full_cvar_regret(A, B, p, gamma, alpha, x, y)
            )
        if not all(finite(cert.get(key)) for key in ("eta", "regret1", "regret2", "rho1", "rho2")):
            return {"status": "lp_failure", "detail": "Non-finite certificate"}
        if not all(cert[f"best_dev{i}"]["success"] for i in (1, 2)):
            return {"status": "lp_failure", "certificate": json_value(cert)}
        # Independent payoff arithmetic, generalized from the saved calibration auditor.
        weights = p / np.sum(p)
        arithmetic_errors = []
        for i, own, h in ((1, x, np.einsum("kij,j->ki", A, y)), (2, y, np.einsum("i,kij->kj", x, B))):
            for strategy, expected in (
                (own, cert[f"rho{i}"]),
                (cert[f"best_dev{i}"]["strategy"], cert[f"best_dev{i}"]["value"]),
            ):
                states = np.einsum("ki,i->k", h, strategy)
                mean = float(np.sum(weights * states))
                if risk == "msd":
                    value = mean - gamma * float(np.sum(weights * np.maximum(mean - states, 0)))
                else:
                    order = np.argsort(states)
                    cumulative = np.cumsum(weights[order])
                    tail_weights = np.minimum(weights[order], np.maximum(0, alpha - (cumulative - weights[order])))
                    value = (1 - gamma) * mean + gamma * float(np.sum(tail_weights * states[order]) / alpha)
                arithmetic_errors.append(abs(value - expected))
        compared = {
            key: abs(float(saved[key]) - cert[key]) for key in ("eta", "regret1", "regret2") if finite(saved.get(key))
        }
        agreement = all(np.isclose(float(saved[key]), cert[key], atol=ATOL, rtol=RTOL) for key in compared)
        classification = (float(saved["eta"]) <= epsilon) == (cert["eta"] <= epsilon) if "eta" in compared else None
        status = (
            "classification_flip"
            if classification is False
            else "mismatch"
            if not agreement or max(arithmetic_errors) > ATOL
            else "match"
            if "eta" in compared
            else "no_recorded_regret"
        )
        return {
            "status": status,
            "warnings": sorted({str(w.message) for w in observed}),
            "certificate": json_value(cert),
            "absolute_differences": compared,
            "classification_agrees": classification,
            "max_independent_payoff_error": max(arithmetic_errors),
            "elapsed_s": perf_counter() - started,
            "profile_hashes": {"x": array_hash(x), "y": array_hash(y)},
            "input_hashes": {name: array_hash(value) for name, value in zip(("A", "B", "p"), (A, B, p), strict=True)},
        }
    except Exception as exc:
        return {"status": "lp_failure", "detail": str(exc), "elapsed_s": perf_counter() - started}


def presented_rows():
    for campaign, pattern in PRESENTATIONS.items():
        for path in sorted(result_path(campaign).glob(pattern)):
            if "history" in path.name or "summary" == path.stem:
                continue
            with path.open(newline="") as stream:
                for row_number, row in enumerate(csv.DictReader(stream), start=2):
                    if path.name == "prisonersDilemmaMSD.csv":
                        # The notebook explicitly rounded these strategies to three decimals.
                        # Certify what was stored, without calling it the original full-precision iterate.
                        row = {
                            **row,
                            "risk": "msd",
                            "K": "3",
                            "n": "2",
                            "seed": "444",
                            "payoff_model": "notebook_prisoners_dilemma",
                            "methods": "rounded_notebook",
                        }
                        for key in ("x", "y"):
                            row["rounded_notebook_" + key] = json.dumps(
                                np.fromstring(row[key].strip("[]"), sep=" ").tolist()
                            )
                    if "seed" not in row or "risk" not in row:
                        continue
                    if campaign == "scalability/msd_cvar_original" and row["risk"] != "msd":
                        continue
                    methods = (
                        [row["method"]]
                        if "method" in row
                        else ["uniform"]
                        if row.get("profile") == "uniform"
                        else row.get("methods", "").split(",")
                    )
                    for method in methods:
                        if not method:
                            continue
                        metrics = (
                            row
                            if "method" in row
                            else {k.removeprefix(method + "_"): v for k, v in row.items() if k.startswith(method + "_")}
                        )
                        yield {
                            "campaign": campaign,
                            "source": str(path.relative_to(REPO)),
                            "row_number": row_number,
                            "method": method,
                            "row": row,
                            "metrics": metrics,
                            "key": f"{campaign}:{path.name}:{row_number}:{method}",
                        }


def profile_state(item: dict) -> tuple[str, object, object]:
    row, metrics = item["row"], item["metrics"]
    if metrics.get("profile_artifact"):
        base = (REPO / item["source"]).parent
        candidates = [
            base / metrics["profile_artifact"],
            base / "method_shards" / item["method"] / metrics["profile_artifact"],
        ]
        for path in candidates:
            if path.is_file():
                sidecar = json.loads(path.read_text())
                item["profile_sidecar"] = str(path)
                if sidecar.get("x") is not None and sidecar.get("y") is not None:
                    return "saved_profile_sidecar", np.asarray(sidecar["x"]), np.asarray(sidecar["y"])
    if item["method"] == "uniform":
        strategy = np.ones(int(row["n"])) / int(row["n"])
        return "fixed_uniform_profile", strategy, strategy.copy()
    if metrics.get("x") and metrics.get("y"):
        try:
            return "saved_profile", np.asarray(json.loads(metrics["x"])), np.asarray(json.loads(metrics["y"]))
        except (ValueError, TypeError):
            return "invalid_serialized_profile", None, None
    if str(metrics.get("has_profile", "")).lower() in ("false", "0") or not finite(metrics.get("eta")):
        return "intentionally_absent", None, None
    return "missing_profile", None, None


def game_for_row(row):
    if row.get("payoff_model") == "notebook_prisoners_dilemma":
        scenarios = np.random.default_rng(444).uniform(0, 1, (100, 5))[int(row["sample"])]
        p = (
            np.array(
                [(scenarios < 0.5).sum(), ((scenarios >= 0.5) & (scenarios < 0.75)).sum(), (scenarios >= 0.75).sum()]
            )
            / 5
        )
        A = np.array([[[3.0, 0.0], [5.0, 1.0]], [[3.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
        return A, A.transpose(0, 2, 1).copy(), p
    K, n, seed = (int(row[key]) for key in ("K", "n", "seed"))
    if row.get("payoff_model") == "cell_beta_uniform_v1":
        A, B, p, mapping = simulate_population_payoffs(K, n, seed)
        if row.get("payoff_population_ids"):
            np.testing.assert_array_equal(mapping, json.loads(row["payoff_population_ids"]))
        return A, B, p
    if row.get("payoff_model", "uniform") not in ("uniform", ""):
        raise ValueError("Unknown payoff generator")
    return simulate_random_payoffs(K, n, seed, float(row.get("low") or 0), float(row.get("high") or 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", choices=tuple(PRESENTATIONS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    items = [item for item in presented_rows() if args.campaign is None or item["campaign"] == args.campaign]
    with gzip.open(args.output_dir / "row_inventory.json.gz", "wt") as stream:
        json.dump(items, stream)
    (args.output_dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    counts = {}
    # Keep one game resident at a time; cache a certificate only under identical inputs/profile/risk.
    last_game_key, game = None, None
    cache = {}
    with (args.output_dir / "profile_checks.jsonl").open("x") as output:
        for index, item in enumerate(items):
            state, x, y = profile_state(item)
            record = {k: item[k] for k in ("key", "campaign", "source", "row_number", "method")}
            record.update(evidence_type=state, status=state)
            if x is not None:
                row = item["row"]
                game_key = tuple(row.get(k, "") for k in ("K", "n", "seed", "payoff_model", "low", "high", "sample"))
                try:
                    original_hashes = None
                    if game_key != last_game_key:
                        game = game_for_row(row)
                        last_game_key = game_key
                    if item.get("profile_sidecar"):
                        original_hashes = json.loads(Path(item["profile_sidecar"]).read_text()).get("inputs")
                        if original_hashes is not None:
                            current_hashes = {
                                name: array_hash(a) for name, a in zip(("A", "B", "p"), game, strict=True)
                            }
                            if original_hashes != current_hashes:
                                raise ValueError("Regenerated input hashes differ from the original sidecar")
                    gamma, alpha = float(row["gamma"]), float(row.get("alpha") or 0.5)
                    cache_key = (
                        *[array_hash(a)["sha256"] for a in game],
                        row["risk"],
                        gamma,
                        alpha if row["risk"] == "cvar" else None,
                        array_hash(x)["sha256"],
                        array_hash(y)["sha256"],
                    )
                    # Saved values still differ across rows; only duplicate identical comparisons are reused.
                    cache_key += tuple(str(item["metrics"].get(k)) for k in ("eta", "regret1", "regret2"))
                    if cache_key not in cache:
                        cache[cache_key] = audit_profile(*game, row["risk"], gamma, alpha, x, y, item["metrics"])
                    record.update(cache[cache_key])
                    record["input_identity"] = (
                        "regenerated; all original array hashes match"
                        if original_hashes is not None
                        else "regenerated; original array hashes unavailable"
                    )
                    record["numpy_agreement"] = (
                        row.get("payoff_numpy_version") == np.__version__ if row.get("payoff_numpy_version") else None
                    )
                    record["scope"] = (
                        "original-profile recheck on verified inputs"
                        if original_hashes is not None
                        else "conditional original-profile recheck"
                    )
                    if item["method"] == "rounded_notebook":
                        record["scope"] = "conditional check of rounded stored strategies; no original regret recorded"
                except Exception as exc:
                    record.update(status="unavailable_inputs", detail=str(exc))
            output.write(json.dumps(json_value(record), allow_nan=False) + "\n")
            output.flush()
            counts.setdefault(item["campaign"], Counter())[record["status"]] += 1
            if index % 200 == 0:
                print(f"Audited {index + 1}/{len(items)} presentation rows", flush=True)
    report = {
        "atol": ATOL,
        "rtol": RTOL,
        "epsilon": 0.01,
        "counts": counts,
        "scope": "Fresh package LPs and independent payoff arithmetic; regenerated inputs remain conditional",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
