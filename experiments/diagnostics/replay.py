"""Three frozen game seeds per study/run, with hard process-group wall limits."""

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import numpy as np

from experiments.common.experiment import write_csv
from experiments.common.paths import REPO
from experiments.common.provenance import json_value
from experiments.diagnostics.regret_audit import audit_profile, finite, game_for_row, profile_state


def select_seeds(items: list[dict]) -> dict:
    """Choose once, never by eta/success; prefer small cells then cheap workloads."""
    runs = defaultdict(lambda: defaultdict(list))
    for item in items:
        if profile_state(item)[0] == "missing_profile":
            row = item["row"]
            runs[item["campaign"]][(row["risk"], int(row["n"]), int(row["K"]), int(row["seed"]))].append(item)
    selection = {}
    rng = np.random.default_rng(0)
    for run, games in sorted(runs.items()):
        risks = sorted({key[0] for key in games})
        chosen = []
        for index in range(3):
            risk = risks[index % len(risks)]
            pool = [key for key in games if key[0] == risk and key[3] not in {k[3] for k in chosen}]
            if not pool:
                pool = [key for key in games if key[3] not in {k[3] for k in chosen}]
            if not pool:
                break
            smallest = min((key[1], key[2]) for key in pool)
            pool = sorted(key for key in pool if (key[1], key[2]) == smallest)
            chosen.append(pool[int(rng.integers(len(pool)))])
        selection[run] = [
            [
                item
                for item in sorted(
                    games[key],
                    key=lambda item: (
                        float(item["metrics"]["time_s"]) if finite(item["metrics"].get("time_s")) else float("inf")
                    ),
                )
            ]
            for key in chosen
        ]
    return selection


def run_bounded(command: list[str], seconds: float, log: Path, env: dict | None = None) -> dict:
    started = monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            code = process.wait(timeout=max(0.001, seconds - (monotonic() - started)))
            status = "completed" if code == 0 else "error"
        except subprocess.TimeoutExpired:
            # Kill the entire group at the deadline, not just the shell parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
            status = "timeout"
    return {"status": status, "exit_code": code, "elapsed_s": monotonic() - started}


def arguments_for(item: dict, output: Path):
    """Recover recorded numerical options; refuse unknown FO configurations."""
    from experiments.runners import compare_scalability_approaches as general
    from experiments.runners import compare_stochastic_fo as tuning

    row, method = item["row"], item["method"]
    is_tuning = method in ("full_batch", "minibatch")
    driver = tuning if is_tuning else general
    args = driver.parse_args(["--methods", method])
    args.csv = output / "replay.csv"
    args.save_profiles = True
    args.quiet = True
    args.gamma = float(row["gamma"])
    args.alpha = float(row["alpha"]) if row["risk"] == "cvar" else 0.5
    args.payoff_model = row.get("payoff_model") or "uniform"
    args.low, args.high = float(row.get("low") or 0), float(row.get("high") or 1)
    if is_tuning:
        mapping = {
            key: key
            for key in (
                "entropy_kappa",
                "smoothing_tau",
                "max_iter",
                "step_size",
                "step_decay",
                "logit_bound",
                "record_every",
                "certify_every",
                "regret_tolerance",
                "stagnation_window",
                "stagnation_rtol",
                "stagnation_atol",
                "continuation_stage_rtol",
                "continuation_stage_atol",
            )
        }
        # Continuation histories predate random restarts and used the uniform start.
        args.n_random_starts = int(row.get("n_random_starts") or 0)
        args.jit_updates = row.get("jit_updates") == "True"
        for key in ("continuation_kappa", "continuation_tau", "continuation_max_iter", "continuation_step_size"):
            if row.get(key):
                cast = int if key == "continuation_max_iter" else float
                setattr(args, key, [cast(x) for x in row[key].split(",")])
        if not row.get("continuation_step_size") and row.get("step_size"):
            args.continuation_step_size = None
        # Historical tuning runner used this documented bound before recording it.
        if not row.get("logit_bound"):
            args.logit_bound = 20.0
    else:
        args.epsilon = float(row["epsilon"])
        mapping = {
            key: key
            for key in (
                "epsilon_scr",
                "max_candidates",
                "n_screen_starts",
                "n_support_starts",
                "screen_maxiter",
                "support_maxiter",
            )
        }
        if method.startswith("stochastic_"):
            mapping.update(
                {
                    "stochastic_" + key: key
                    for key in (
                        "max_iter",
                        "entropy_kappa",
                        "smoothing_tau",
                        "step_size",
                        "step_decay",
                        "logit_bound",
                        "gradient_clip_norm",
                        "record_every",
                        "certify_every",
                        "stagnation_window",
                        "stagnation_rtol",
                        "stagnation_atol",
                    )
                }
            )
            required = (
                "stochastic_entropy_kappa",
                "stochastic_smoothing_tau",
                "stochastic_step_size",
                "stochastic_max_iter",
                "stochastic_regret_tolerance",
                "stochastic_certify_every",
            )
            if any(not row.get(key) for key in required):
                raise ValueError("Historical FO numerical configuration is not fully recorded")
            args.stochastic_regret_tolerance = float(row["stochastic_regret_tolerance"])
            args.stochastic_n_random_starts = int(row.get("stochastic_n_random_starts") or 0)
            args.jit_updates = row.get("stochastic_jit_updates") == "True"
        args.solver = item["metrics"].get("solver") or "pathampl"
        args.fallback_solver = None
    for source, target in mapping.items():
        if finite(row.get(source)):
            value = float(row[source])
            if target in (
                "max_iter",
                "max_candidates",
                "n_screen_starts",
                "n_support_starts",
                "screen_maxiter",
                "support_maxiter",
                "record_every",
                "certify_every",
                "stagnation_window",
            ):
                value = int(value)
            setattr(args, target, value)
    return driver, args


def worker(path: Path, directory: Path) -> None:
    items = json.loads(path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "comparisons.jsonl").open("x") as stream:
        for number, item in enumerate(items):
            row = item["row"]
            record = {
                "key": item["key"],
                "evidence_type": "sampled_seed_replay",
                "input_identity": "conditional: original hashes/full environment unavailable",
            }
            try:
                driver, args = arguments_for(item, directory / str(number))
                replayed = driver.run_instance(args, row["risk"], int(row["K"]), int(row["n"]), int(row["seed"]))
                method = item["method"]
                fresh = {
                    key.removeprefix(method + "_"): value
                    for key, value in replayed.items()
                    if key.startswith(method + "_")
                }
                record["replayed"] = fresh
                record["original"] = item["metrics"]
                profiles = list((args.csv.parent / "profiles").glob("*.json"))
                profile = json.loads(profiles[0].read_text())
                if profile["x"] is None:
                    record["status"] = "no_replayed_profile"
                else:
                    game = game_for_row(row)
                    record["fresh_certificate"] = audit_profile(
                        *game, row["risk"], args.gamma, args.alpha, profile["x"], profile["y"], fresh
                    )
                    record["original_comparison"] = audit_profile(
                        *game, row["risk"], args.gamma, args.alpha, profile["x"], profile["y"], item["metrics"]
                    )
                    record["status"] = record["original_comparison"]["status"]
                    record["work_agreement"] = {
                        key: str(fresh.get(key)) == str(item["metrics"][key])
                        for key in ("selected_start", "iterations", "termination_reason")
                        if item["metrics"].get(key)
                    }
            except Exception as exc:
                record.update(status="unavailable_configuration_or_solver", detail=str(exc))
            stream.write(json.dumps(json_value(record), allow_nan=False) + "\n")
            stream.flush()


def resume_adapter_errors(directory: Path) -> None:
    """Retry only adapter failures, using the frozen keys and unspent wall budgets."""
    selection = json.loads((directory / "selection.json").read_text())
    reports = json.loads((directory / "execution.json").read_text())
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    for campaign, groups in selection.items():
        for index, items in enumerate(groups):
            previous = reports[campaign][index]
            target = directory / campaign / str(index)
            source = target / "output/comparisons.jsonl"
            if not source.exists() or (target / "adapter_retry").exists():
                continue
            errors = {
                row["key"]
                for row in (json.loads(line) for line in source.read_text().splitlines())
                if "not JSON serializable" in row.get("detail", "")
                or "continuation step-size schedule must match" in row.get("detail", "")
            }
            retry = [item for item in items if item["key"] in errors]
            # Reserve one second for parent setup and cleanup; original attempt time is charged.
            seconds = min(99 - previous["elapsed_s"], 299 - sum(row["elapsed_s"] for row in reports[campaign]))
            if not retry or seconds <= 0:
                continue
            target = target / "adapter_retry"
            target.mkdir()
            task_file = target / "tasks.json"
            task_file.write_text(json.dumps(retry, indent=2) + "\n")
            print(f"Repair adapter for {campaign} seed {previous['seed']}; remaining cap {seconds:.1f}s", flush=True)
            command = [
                sys.executable,
                "-m",
                "experiments.diagnostics.replay",
                "--worker",
                str(task_file),
                "--output-dir",
                str(target / "output"),
            ]
            attempt = run_bounded(command, seconds, target / "worker.log", env)
            reports[campaign][index] = {
                **previous,
                "status": attempt["status"],
                "exit_code": attempt["exit_code"],
                "elapsed_s": previous["elapsed_s"] + attempt["elapsed_s"],
                "attempts": [previous, attempt],
            }
            (directory / "execution.json").write_text(json.dumps(reports, indent=2) + "\n")


def summarize(directory: Path) -> dict:
    """Join every frozen task to its latest completed comparison or explicit gap."""
    selection = json.loads((directory / "selection.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    rows, counts = [], {}
    for campaign, groups in selection.items():
        for index, items in enumerate(groups):
            target = directory / campaign / str(index)
            report = execution[campaign][index]
            records = {}
            for attempt in (target, target / "adapter_retry"):
                if attempt.name == "adapter_retry" and (attempt / "tasks.json").exists():
                    for task in json.loads((attempt / "tasks.json").read_text()):
                        previous = records.get(task["key"], {})
                        records[task["key"]] = {"initial_adapter_error": previous.get("detail")}
                path = attempt / "output/comparisons.jsonl"
                if path.exists():
                    for line in path.read_text().splitlines():
                        record = json.loads(line)
                        records[record["key"]] = {
                            **records.get(record["key"], {}),
                            **record,
                            "comparison_source": str(path.relative_to(directory)),
                        }
            for item in items:
                record = records.get(item["key"], {})
                status = record.get("status", "timeout_or_unattempted")
                if record.get("detail") and report["status"] == "timeout":
                    if (
                        "not JSON serializable" in record["detail"]
                        or "continuation step-size schedule must match" in record["detail"]
                    ):
                        status = "adapter_error_unresolved_at_deadline"
                fresh = record.get("replayed", {})
                row = {
                    "campaign": campaign,
                    "key": item["key"],
                    "source": item["source"],
                    "row_number": item["row_number"],
                    "risk": item["row"]["risk"],
                    "n": item["row"]["n"],
                    "K": item["row"]["K"],
                    "seed": item["row"]["seed"],
                    "method": item["method"],
                    "status": status,
                    "original_eta": item["metrics"].get("eta"),
                    "replayed_eta": fresh.get("eta"),
                    "fresh_certificate_status": record.get("fresh_certificate", {}).get("status"),
                    "seed_execution_status": report["status"],
                    "seed_elapsed_s": report["elapsed_s"],
                    "work_agreement": json.dumps(record.get("work_agreement", {}), sort_keys=True),
                    "comparison_source": record.get("comparison_source"),
                    "detail": record.get("detail"),
                    "initial_adapter_error": record.get("initial_adapter_error"),
                    "evidence_type": "conditional sampled-seed replay; original profile unavailable",
                }
                rows.append(row)
        counts[campaign] = {
            "selected_seeds": [group[0]["row"]["seed"] for group in groups],
            "insufficient_distinct_seeds": len(groups) < 3,
            "elapsed_s": sum(report["elapsed_s"] for report in execution[campaign]),
            "statuses": dict(Counter(row["status"] for row in rows if row["campaign"] == campaign)),
        }
    write_csv(rows, directory / "comparisons.csv")
    (directory / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-adapter-errors", action="store_true")
    parser.add_argument(
        "--summarize", action="store_true", help="Report the existing frozen replays without running solvers"
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output_dir)
        return
    directory = args.audit_dir / "replays"
    if args.summarize:
        print(json.dumps(summarize(directory), indent=2))
        return
    if args.resume_adapter_errors:
        resume_adapter_errors(directory)
        return
    directory.mkdir(exist_ok=False)
    inventory = args.audit_dir / "row_inventory.json.gz"
    if inventory.exists():
        with gzip.open(inventory, "rt") as stream:
            items = json.load(stream)
    else:
        items = json.loads((args.audit_dir / "row_inventory.json").read_text())
    selection = select_seeds(items)
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    reports = {}
    for campaign, groups in selection.items():
        start = monotonic()
        reports[campaign] = []
        for index, items in enumerate(groups):
            target = directory / campaign / str(index)
            target.mkdir(parents=True)
            input_path = target / "tasks.json"
            input_path.write_text(json.dumps(items, indent=2) + "\n")
            remaining = 300 - (monotonic() - start)
            if remaining <= 0:
                report = {"status": "unattempted", "elapsed_s": 0}
            else:
                command = [
                    sys.executable,
                    "-m",
                    "experiments.diagnostics.replay",
                    "--worker",
                    str(input_path),
                    "--output-dir",
                    str(target / "output"),
                ]
                print(f"Replay {campaign} seed {items[0]['row']['seed']} ({len(items)} method rows)", flush=True)
                report = run_bounded(command, min(100, remaining), target / "worker.log", env)
            reports[campaign].append(
                {
                    **report,
                    "seed": items[0]["row"]["seed"],
                    "risk": items[0]["row"]["risk"],
                    "task_keys": [item["key"] for item in items],
                }
            )
            (directory / "execution.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
