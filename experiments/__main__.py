"""Plan, run, inspect, collect, and sync named research campaigns."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from experiments.common.campaign import check_saved_config, load_campaign, resolver_args
from experiments.common.experiment import certificate_success
from experiments.common.paths import REPO, ROOT, catalog, relocated_path, result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status", "collect", "sync"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--config", type=Path, help="Explicit preset for a new run ID")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--remote")
    parser.add_argument("--remote-root", default="/root/cumg")
    parser.add_argument("--remote-path", help="Explicit legacy remote directory, if it has not been migrated")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", help="Named calibration phase")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "sync":
        if not args.remote or args.remote.startswith("-"):
            parser.error("sync requires --remote USER@HOST")
        if args.campaign in catalog()["runs"] and args.config is None:
            entry = catalog()["runs"][args.campaign]
            destination = result_path(args.campaign)
            remote_path = args.remote_path or str(Path(args.remote_root) / entry["path"])
        else:
            config = load_campaign(args.campaign, args.config)
            destination = (REPO / config["result_dir"]).resolve()
            if not destination.is_relative_to(REPO) and not args.remote_path:
                parser.error("A destination outside the repository requires an explicit --remote-path")
            remote_path = args.remote_path or str(Path(args.remote_root) / destination.relative_to(REPO))
        rsync_help = subprocess.run(["rsync", "--help"], check=True, capture_output=True, text=True)
        help_text = rsync_help.stdout + rsync_help.stderr
        command = ["rsync", "-av", "--exclude=*.lock/", "--exclude=*.partial*", "--exclude=*.tmp"]
        remote_source = remote_path.rstrip("/") + "/"
        if "--secluded-args" in help_text:
            command.append("--secluded-args")
        elif "--protect-args" in help_text:
            command.append("--protect-args")
        else:
            # macOS OpenRSYNC and rsync 2.x pass paths through the remote shell.
            remote_source = shlex.quote(remote_source)
        if args.preview:
            command += ["--dry-run"]
        else:
            destination.mkdir(parents=True, exist_ok=True)
        command += [args.remote + ":" + remote_source, str(destination) + "/"]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        return

    preset = ROOT / "configs" / (args.campaign + ".json")
    if args.config is None and not preset.is_file():
        entry = catalog()["runs"][args.campaign]
        if args.command in ("plan", "status"):
            print(json.dumps(entry, indent=2))
            return
        parser.error("This historical run has no complete execution preset; its artifacts remain available via sync")
    config = load_campaign(args.campaign, args.config)
    result_dir = REPO / config["result_dir"]
    snapshot = result_dir / "campaign_config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        parser.error("Saved campaign preset differs; use a new campaign ID")
    env = config.get("historical_env", {})
    check_saved_config(result_dir / "run_config.env", env)
    if config["engine"] == "continuation":
        from experiments.runners.tuning_campaign import completed, run

        pending = [task["name"] for task in config["tasks"] if not completed(REPO / task["summary"])]
        print(
            json.dumps(
                {
                    "campaign": args.campaign,
                    "expected": len(config["tasks"]),
                    "completed": len(config["tasks"]) - len(pending),
                    "pending": pending,
                },
                indent=2,
            )
        )
        if args.command == "run" and not args.preview:
            result_dir.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                snapshot.write_text(json.dumps(config, indent=2) + "\n")
            print(Counter(run(config, args.workers)))
        return
    if config["engine"] == "tuning":
        if args.phase is None:
            if args.command in ("plan", "status"):
                print(json.dumps(config, indent=2))
                return
            parser.error("Specify --phase from the named preset")
        phase = config["phases"][args.phase]
        command = [
            sys.executable,
            "-m",
            "experiments.runners.compare_stochastic_fo",
            *phase["arguments"],
            "--save-profiles",
        ]
        if args.command in ("plan", "status") or args.preview:
            print(json.dumps({"campaign": args.campaign, "command": command, "outputs": phase["outputs"]}, indent=2))
            return
        if args.command != "run":
            parser.error("Tuning outputs are written by the runner; use sync to transfer them")
        if any((REPO / path).exists() for path in phase["outputs"]):
            parser.error("Tuning output already exists; use a new run ID")
        result_dir.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_text(json.dumps(config, indent=2) + "\n")
        subprocess.run(command, cwd=REPO, check=True)
        return

    from experiments.runners.capped_scalability_resume import collect_results, iter_tasks, resolve_task

    options = resolver_args(config)
    options.retry_errors = args.retry_errors
    tasks = list(iter_tasks(options))
    rows = [resolve_task(task, options) for task in tasks]
    counts = Counter(row["status"] for row in rows)
    successes = int(
        sum(
            certificate_success(
                row["status"], row["eta"] if row.get("eta") not in (None, "") else "nan", float(env["EPSILON"])
            )
            for row in rows
        )
    )
    pending = [
        task.method_stem
        for task, row in zip(tasks, rows, strict=True)
        if row["status"] == "pending" or (args.retry_errors and row["status"] == "error")
    ]
    report = {
        "campaign": args.campaign,
        "expected": len(tasks),
        "counts": dict(counts),
        "certificate_successes": successes,
        "pending": len(pending),
    }
    if args.command == "plan":
        report.update(pending_tasks=pending, configuration=config)
    print(json.dumps(report, indent=2), flush=True)
    if args.command in ("plan", "status"):
        return
    if args.command == "collect":
        options.output = args.output or result_dir / "capped_method_results.csv"
        # Preserve the bytes of completed historical aggregates when logically identical.
        if options.output.exists():
            with options.output.open(newline="") as stream:
                saved = list(csv.DictReader(stream))
            if len(saved) == len(rows) and all(
                all(str(row.get(k, "")) == v for k, v in old.items()) for old, row in zip(saved, rows, strict=True)
            ):
                print("Aggregate already matches the shards; left unchanged.")
                return
            if catalog()["runs"].get(args.campaign, {}).get("preset_sha256"):
                parser.error("Frozen aggregate differs from the shards; collect to a new --output path for review")
        collect_results(options)
        return
    if not pending:
        print("No pending work; no processes launched.")
        return
    if args.preview:
        print(json.dumps({"pending_tasks": pending}, indent=2))
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        with snapshot.open("x") as stream:
            stream.write(json.dumps(config, indent=2) + "\n")
    runtime_env = os.environ.copy()
    runtime_env.update(config["environment"])
    runtime_env.update(
        {
            ("RISK_GRID" if key == "risk" else key): relocated_path(value) if key == "LEGACY_RESULT_DIR" else value
            for key, value in env.items()
        }
    )
    runtime_env.update(
        PYTHON_BIN=sys.executable,
        RESULT_DIR=str(result_dir),
        RETRY_ERRORS=str(int(args.retry_errors)),
        DRY_RUN=str(int(args.preview)),
    )
    if args.workers is not None:
        # Worker count is execution provenance, but old env files include it;
        # use the existing guard rather than bypassing a saved configuration.
        if (result_dir / "run_config.env").exists() and str(args.workers) != env["WORKERS"]:
            parser.error("Saved worker count differs; create a new campaign preset")
        runtime_env["WORKERS"] = str(args.workers)
    if config["engine"] == "capped":
        Path(runtime_env["LEGACY_RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", str(ROOT / "runners" / (config["engine"] + ".sh"))], cwd=REPO, env=runtime_env, check=True)


if __name__ == "__main__":
    main()
