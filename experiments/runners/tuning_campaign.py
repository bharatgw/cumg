"""Schedule recorded continuation shards without changing their solver loop."""

import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.common.paths import REPO


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="") as stream:
        return any(csv.DictReader(stream))


def run_task(task: dict) -> str:
    summary, history = (REPO / task[key] for key in ("summary", "history"))
    if completed(summary):
        return "completed"
    lock = summary.with_name("." + task["name"] + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        return "locked"
    try:
        command = [
            sys.executable,
            "-m",
            "experiments.runners.compare_stochastic_fo",
            *task["arguments"],
            "--csv",
            str(summary.with_suffix(".partial.csv")),
            "--history-csv",
            str(history.with_suffix(".partial.csv")),
            "--save-profiles",
        ]
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        )
        log = summary.parent / "logs" / (task["name"] + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            return "error"
        if not completed(summary.with_suffix(".partial.csv")):
            return "error"
        history.with_suffix(".partial.csv").replace(history)
        summary.with_suffix(".partial.csv").replace(summary)
        return "completed"
    finally:
        lock.rmdir()


def run(config: dict, workers: int | None = None) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers or config["workers"]) as pool:
        return list(pool.map(run_task, config["tasks"]))
