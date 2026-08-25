"""Plan, record, and collect method-capped scalability resume runs.

The uncapped scalability campaign writes one wide CSV row after every method in
an instance has finished.  This utility lets a replacement runner execute one
method at a time while reusing completed method measurements from that legacy
row.  Measurements exceeding the new uniform cap are retained as provenance
but classified as timeouts in the capped dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_scalability_approaches import METHODS, experiment_seed

DEFAULT_K_GRID = (5, 10, 30, 100, 250, 500)
DEFAULT_N_GRID = (5, 10, 20, 50)
DEFAULT_REPS = 20
DEFAULT_SEED_BASE = 123
DEFAULT_TIME_LIMIT_SECONDS = 24 * 60 * 60

COMMON_RESULT_FIELDS = (
    "gamma",
    "alpha",
    "epsilon",
    "epsilon_scr",
    "support_kappa",
    "support_tau",
    "max_candidates",
    "n_screen_starts",
    "n_support_starts",
    "screen_maxiter",
    "support_maxiter",
    "stochastic_max_iter",
    "stochastic_minibatch_size",
    "stochastic_entropy_kappa",
    "stochastic_smoothing_tau",
    "stochastic_step_size",
    "stochastic_step_decay",
    "stochastic_logit_bound",
    "stochastic_gradient_clip_norm",
    "stochastic_record_every",
    "stochastic_certify_every",
    "stochastic_regret_tolerance",
)

METHOD_RESULT_FIELDS = (
    "success",
    "eta",
    "regret1",
    "regret2",
    "has_profile",
    "error",
    "solver",
    "candidate_index",
    "support_eta",
    "screen_eta",
    "best_screen_eta",
    "best_screen_success",
    "best_screen_violation",
    "best_screen_candidate_index",
    "best_screen_message",
    "support_violation",
    "residual_norm",
    "objective",
    "iterations",
    "best_certificate_eta",
    "best_certificate_iteration",
)

OUTPUT_FIELDS = (
    "risk",
    "K",
    "n",
    "rep",
    "seed",
    "method",
    "status",
    "success",
    "censored",
    "time_limit_s",
    "time_s",
    "observed_time_s",
    "source",
    "exit_code",
    "status_message",
    *COMMON_RESULT_FIELDS,
    *(field for field in METHOD_RESULT_FIELDS if field != "success"),
    "uncapped_success",
    "uncapped_eta",
)


@dataclass(frozen=True)
class Task:
    risk: str
    K: int
    n: int
    rep: int
    method: str

    @property
    def replicate_stem(self) -> str:
        return f"{self.risk}_K{self.K}_n{self.n}_rep{self.rep:03d}"

    @property
    def method_stem(self) -> str:
        return f"{self.replicate_stem}__{self.method}"


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _read_single_result(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result row in {path}; found {len(rows)}.")
    return rows[0]


def legacy_result_path(legacy_dir: Path, task: Task) -> Path:
    return legacy_dir / f"{task.replicate_stem}.csv"


def method_shard_path(result_dir: Path, task: Task) -> Path:
    return result_dir / "method_shards" / task.method / f"{task.method_stem}.csv"


def method_status_path(result_dir: Path, task: Task) -> Path:
    return result_dir / "method_shards" / task.method / f"{task.method_stem}.status.json"


def iter_tasks(args: argparse.Namespace):
    for risk in args.risk:
        for K in args.K:
            for n in args.n:
                for rep in range(args.reps):
                    for method in args.methods:
                        yield Task(risk=risk, K=K, n=n, rep=rep, method=method)


def _base_record(task: Task, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "risk": task.risk,
        "K": task.K,
        "n": task.n,
        "rep": task.rep,
        "seed": experiment_seed(task.risk, task.K, task.n, task.rep, args.seed_base),
        "method": task.method,
        "status": "pending",
        "success": False,
        "censored": False,
        "time_limit_s": args.time_limit_seconds,
        "time_s": "",
        "observed_time_s": "",
        "source": "",
        "exit_code": "",
        "status_message": "",
        "uncapped_success": "",
        "uncapped_eta": "",
    }


def _result_record(
    task: Task,
    args: argparse.Namespace,
    row: dict[str, str],
    source: str,
) -> dict[str, Any] | None:
    observed_time = _finite_float(row.get(f"{task.method}_time_s"))
    if observed_time is None:
        return None

    record = _base_record(task, args)
    record.update({field: row.get(field, "") for field in COMMON_RESULT_FIELDS})
    method_values = {
        field: row.get(f"{task.method}_{field}", "") for field in METHOD_RESULT_FIELDS
    }
    timed_out = observed_time > args.time_limit_seconds
    record.update(
        {
            "status": "timeout" if timed_out else "completed",
            "success": False if timed_out else (_parse_bool(method_values["success"]) or False),
            "censored": timed_out,
            "time_s": args.time_limit_seconds if timed_out else observed_time,
            "observed_time_s": observed_time,
            "source": source,
        }
    )

    if timed_out:
        record["uncapped_success"] = _parse_bool(method_values["success"])
        record["uncapped_eta"] = method_values["eta"]
        return record

    record.update(
        {field: value for field, value in method_values.items() if field != "success"}
    )
    return record


def _status_record(
    task: Task,
    args: argparse.Namespace,
    status_path: Path,
) -> dict[str, Any] | None:
    if not status_path.is_file():
        return None
    with status_path.open() as f:
        status = json.load(f)
    marker_limit = int(status["time_limit_s"])
    if marker_limit != args.time_limit_seconds:
        raise ValueError(
            f"Time-limit mismatch in {status_path}: {marker_limit} != {args.time_limit_seconds}."
        )
    record = _base_record(task, args)
    marker_status = status.get("status", "error")
    elapsed = _finite_float(status.get("elapsed_s"))
    record.update(
        {
            "status": marker_status,
            "success": False,
            "censored": marker_status == "timeout",
            "time_s": args.time_limit_seconds if marker_status == "timeout" else "",
            "observed_time_s": elapsed if elapsed is not None else "",
            "source": "capped_run",
            "exit_code": status.get("exit_code", ""),
            "status_message": status.get("message", ""),
        }
    )
    return record


def resolve_task(task: Task, args: argparse.Namespace) -> dict[str, Any]:
    new_row = _read_single_result(method_shard_path(args.result_dir, task))
    if new_row is not None:
        record = _result_record(task, args, new_row, source="capped_run")
        if record is not None:
            return record

    status_record = _status_record(task, args, method_status_path(args.result_dir, task))
    if status_record is not None:
        return status_record

    legacy_row = _read_single_result(legacy_result_path(args.legacy_dir, task))
    if legacy_row is not None:
        record = _result_record(task, args, legacy_row, source="legacy_reused")
        if record is not None:
            return record

    return _base_record(task, args)


def write_manifest(args: argparse.Namespace) -> dict[str, int]:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    counts: dict[str, int] = {"total": 0, "scheduled": 0}
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for task in iter_tasks(args):
            record = resolve_task(task, args)
            counts["total"] += 1
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
            if status == "pending" or (status == "error" and args.retry_errors):
                counts["scheduled"] += 1
                writer.writerow(
                    [
                        task.risk,
                        task.K,
                        task.n,
                        task.rep,
                        task.method,
                        record["seed"],
                    ]
                )
    os.replace(tmp_path, args.manifest)
    return counts


def collect_results(args: argparse.Namespace) -> dict[str, int]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: dict[str, int] = {"total": 0}
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for task in iter_tasks(args):
            record = resolve_task(task, args)
            status = str(record["status"])
            counts["total"] += 1
            counts[status] = counts.get(status, 0) + 1
            writer.writerow(record)
    os.replace(tmp_path, args.output)
    return counts


def record_status(args: argparse.Namespace) -> Path:
    task = Task(args.risk, args.K, args.n, args.rep, args.method)
    path = method_status_path(args.result_dir, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "risk": task.risk,
        "K": task.K,
        "n": task.n,
        "rep": task.rep,
        "seed": experiment_seed(task.risk, task.K, task.n, task.rep, args.seed_base),
        "method": task.method,
        "status": args.status,
        "elapsed_s": args.elapsed_s,
        "time_limit_s": args.time_limit_seconds,
        "exit_code": args.exit_code,
        "message": args.message,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    return path


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--risk", nargs="+", choices=("msd", "cvar"), default=["cvar"])
    parser.add_argument("--K", type=int, nargs="+", default=list(DEFAULT_K_GRID))
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write unresolved method tasks to a TSV manifest.")
    _add_grid_arguments(plan_parser)
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--retry-errors", action="store_true")

    collect_parser = subparsers.add_parser(
        "collect", help="Collect legacy, capped, timeout, and pending tasks into one long CSV."
    )
    _add_grid_arguments(collect_parser)
    collect_parser.add_argument("--output", type=Path, required=True)

    record_parser = subparsers.add_parser("record-status", help="Record a timeout or process error.")
    record_parser.add_argument("--result-dir", type=Path, required=True)
    record_parser.add_argument("--risk", choices=("msd", "cvar"), required=True)
    record_parser.add_argument("--K", type=int, required=True)
    record_parser.add_argument("--n", type=int, required=True)
    record_parser.add_argument("--rep", type=int, required=True)
    record_parser.add_argument("--method", choices=METHODS, required=True)
    record_parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    record_parser.add_argument("--status", choices=("timeout", "error"), required=True)
    record_parser.add_argument("--elapsed-s", type=float, required=True)
    record_parser.add_argument(
        "--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS
    )
    record_parser.add_argument("--exit-code", type=int, required=True)
    record_parser.add_argument("--message", default="")

    args = parser.parse_args()
    if getattr(args, "time_limit_seconds", 1) <= 0:
        parser.error("time-limit-seconds must be positive")
    if getattr(args, "reps", 1) <= 0:
        parser.error("reps must be positive")
    return args


def _format_counts(counts: dict[str, int]) -> str:
    order = ("total", "completed", "timeout", "error", "pending", "scheduled")
    return " ".join(f"{key}={counts.get(key, 0)}" for key in order)


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        counts = write_manifest(args)
        print(f"Wrote {args.manifest}: {_format_counts(counts)}")
    elif args.command == "collect":
        counts = collect_results(args)
        print(f"Wrote {args.output}: {_format_counts(counts)}")
    else:
        path = record_status(args)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
