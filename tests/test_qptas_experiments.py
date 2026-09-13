import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import compare_scalability_approaches as driver  # noqa: E402

from cumg.small_support import full_cvar_regret, full_msd_regret  # noqa: E402

RUNNER = ROOT / "experiments" / "run_qptas_scalability_remote.sh"


@pytest.fixture
def qptas_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(driver.__file__), "--methods", "qptas", "--max-candidates", "3", "--reps", "1", "--quiet"],
    )
    return driver.parse_args()


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_qptas_driver_uses_full_game_and_records_certificate(qptas_args, monkeypatch, risk):
    args = qptas_args
    args.epsilon = 1.0  # Every profile of this bounded game passes, exposing its certificate.
    solve = getattr(driver, f"solve_{risk}_qptas")
    captured = {}

    def capture(payoffs, p, **kwargs):
        captured.update(payoffs=payoffs, p=p, kwargs=kwargs)
        captured["result"] = solve(payoffs, p, **kwargs)
        return captured["result"]

    monkeypatch.setattr(driver, f"solve_{risk}_qptas", capture)
    row = driver.run_instance(args, risk, K=4, n=3, seed=123)
    A, B, p = driver.simulate_random_payoffs(K=4, n=3, seed=123)
    np.testing.assert_array_equal(captured["payoffs"], [A, B])
    np.testing.assert_array_equal(captured["p"], p)
    assert captured["kwargs"] == {
        "gamma": 0.5,
        "kappa": 2,
        "epsilon": 1.0,
        "max_candidates": 3,
        "seed": 123,
        **({"alpha": 0.5} if risk == "cvar" else {}),
    }
    x, y = driver._profile_from_result(captured["result"])
    if risk == "msd":
        cert = full_msd_regret(A, B, p, args.gamma, x, y)
    else:
        cert = full_cvar_regret(A, B, p, args.gamma, args.alpha, x, y)
    assert row["qptas_success"]
    assert row["qptas_has_profile"]
    assert row["qptas_error"] is None
    assert row["qptas_termination_reason"] == "epsilon_reached"
    assert row["qptas_profiles_checked"] == 1
    assert row["qptas_total_profiles"] == 36
    assert row["qptas_best_response_solves"] == 2
    assert row["qptas_sampling_seed"] == 123
    for key in ("eta", "regret1", "regret2"):
        assert row[f"qptas_{key}"] == pytest.approx(cert[key], abs=1e-10)


@pytest.mark.parametrize("risk", ["msd", "cvar"])
@pytest.mark.parametrize("budget,checked,reason", [(3, 3, "sample_exhausted"), (1000, 9, "grid_exhausted")])
def test_qptas_driver_records_exhaustion(qptas_args, monkeypatch, risk, budget, checked, reason):
    # This zero-sum game's unique equilibrium (.6, .4) is outside the kappa=2 grid.
    A = np.tile([[1.0, -1.0], [-1.0, 2.0]], (2, 1, 1))
    monkeypatch.setattr(driver, "simulate_random_payoffs", lambda **kwargs: (A, -A, np.array([0.5, 0.5])))
    qptas_args.max_candidates = budget
    row = driver.run_instance(qptas_args, risk, K=2, n=2, seed=123)
    assert not row["qptas_success"]
    assert not row["qptas_has_profile"]
    assert row["qptas_error"] is None
    assert np.isnan(row["qptas_eta"])
    assert row["qptas_termination_reason"] == reason
    assert row["qptas_profiles_checked"] == checked
    assert row["qptas_total_profiles"] == 9
    assert checked <= row["qptas_best_response_solves"] <= 2 * checked


def test_qptas_error_rows_have_stable_csv_schema(qptas_args, monkeypatch, tmp_path):
    solve = driver.solve_msd_qptas

    def fail(*args, **kwargs):
        raise RuntimeError("LP failed")

    monkeypatch.setattr(driver, "solve_msd_qptas", fail)
    failed = driver.run_instance(qptas_args, "msd", K=2, n=2, seed=123)
    monkeypatch.setattr(driver, "solve_msd_qptas", solve)
    qptas_args.epsilon = 1.0
    passed = driver.run_instance(qptas_args, "msd", K=2, n=2, seed=123)
    assert failed["qptas_termination_reason"] == "error"
    assert failed["qptas_error"] == "LP failed"
    assert passed["qptas_success"]
    assert failed.keys() == passed.keys()
    path = tmp_path / "rows.csv"
    with driver.StreamingCsvWriter(path) as writer:
        writer.write_row(failed)
        writer.write_row(passed)
    with path.open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_fail_on_error_keeps_diagnostics_and_exits_nonzero(qptas_args, monkeypatch, tmp_path):
    qptas_args.risk = ["msd"]
    qptas_args.K = [2]
    qptas_args.n = [2]
    qptas_args.gamma = -1.0
    qptas_args.fail_on_error = True
    qptas_args.csv = tmp_path / "partial.csv"
    monkeypatch.setattr(driver, "parse_args", lambda: qptas_args)
    with pytest.raises(SystemExit, match="gamma must be"):
        driver.main()
    with qptas_args.csv.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["qptas_termination_reason"] == "error"
    assert "gamma must be" in row["qptas_error"]


def test_qptas_is_opt_in_for_existing_driver(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(driver.__file__)])
    args = driver.parse_args()
    assert args.methods == list(driver.DEFAULT_METHODS)
    assert "qptas" not in args.methods
    assert not args.fail_on_error


@pytest.fixture
def runner_env(tmp_path):
    # Use explicit settings so user shell overrides cannot enlarge a test run.
    return {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
        "RESULT_DIR": str(tmp_path / "results"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_qptas_shell_default_plan(runner_env):
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env={**runner_env, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    directory = Path(runner_env["RESULT_DIR"])
    rows = [line.split("\t") for line in (directory / "pending_methods.tsv").read_text().splitlines()]
    assert len(rows) == 960
    assert {row[0] for row in rows} == {"msd", "cvar"}
    assert {int(row[1]) for row in rows} == {5, 10, 30, 100, 250, 500}
    assert {int(row[2]) for row in rows} == {5, 10, 20, 50}
    assert {int(row[3]) for row in rows} == set(range(20))
    assert {row[4] for row in rows} == {"qptas"}
    for risk, K, n, rep, _, seed in rows:
        assert int(seed) == driver.experiment_seed(risk, int(K), int(n), int(rep), 123)
    config = (directory / "run_config.env").read_text().splitlines()
    assert {"EPSILON=0.01", "MAX_CANDIDATES=1000", "WORKERS=8", "METHOD_TIME_LIMIT_SECONDS=86400"} <= set(config)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="Requires GNU timeout")
def test_qptas_shell_runs_both_risks_and_resumes(runner_env):
    env = {**runner_env, "K_GRID": "2", "N_GRID": "2", "REPS": "1", "WORKERS": "2", "MAX_CANDIDATES": "3"}
    run = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    directory = Path(env["RESULT_DIR"])
    with (directory / "capped_method_results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["risk"] for row in rows} == {"msd", "cvar"}
    for row in rows:
        assert row["status"] == "completed"
        assert row["censored"] == "False"
        assert row["method"] == "qptas"
        assert row["sampling_seed"] == row["seed"]
        assert row["support_kappa"] == "2"
        assert row["total_profiles"] == "9"
        assert 1 <= int(row["profiles_checked"]) <= 3
        assert row["error"] == ""
        if row["success"] == "True":
            assert float(row["eta"]) <= 0.01
            assert row["termination_reason"] == "epsilon_reached"
        else:
            assert row["termination_reason"] == "sample_exhausted"
            assert row["profiles_checked"] == "3"
    shards = {p: p.stat().st_mtime_ns for p in (directory / "method_shards").rglob("*.csv")}
    assert len(shards) == 2
    resumed = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (directory / "pending_methods.tsv").read_text() == ""
    assert {p: p.stat().st_mtime_ns for p in shards} == shards
    changed = subprocess.run(
        ["bash", str(RUNNER)],
        env={**env, "EPSILON": "0.02"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert changed.returncode == 2
    assert "Configuration differs" in changed.stderr


@pytest.mark.skipif(shutil.which("timeout") is None, reason="Requires GNU timeout")
def test_qptas_shell_errors_are_retryable(runner_env):
    env = {
        **runner_env,
        "RISK_GRID": "msd",
        "K_GRID": "1",
        "N_GRID": "1",
        "REPS": "1",
        "WORKERS": "1",
        "GAMMA": "-1",  # Exercise the real Python error path and process exit status.
    }
    directory = Path(env["RESULT_DIR"])
    for retry, expected_attempts in [("0", 1), ("0", 1), ("1", 2)]:
        result = subprocess.run(
            ["bash", str(RUNNER)],
            env={**env, "RETRY_ERRORS": retry},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with (directory / "capped_method_results.csv").open(newline="") as f:
            row = next(csv.DictReader(f))
        assert row["status"] == "error"
        assert row["exit_code"] == "1"
        assert row["success"] == "False"
        assert row["censored"] == "False"
        assert len(list((directory / "logs").glob("*_attempt*.log"))) == expected_attempts
    assert len(list((directory / "logs").glob("*_status_before_attempt*.json"))) == 1
    assert not (directory / "method_shards/qptas/msd_K1_n1_rep000__qptas.csv").exists()


@pytest.mark.skipif(shutil.which("timeout") is None, reason="Requires GNU timeout")
def test_qptas_shell_timeout_is_censored_and_not_retried(runner_env, tmp_path):
    # Keep real planning/collection, but make the numerical subprocess exceed its cap.
    launcher = tmp_path / "slow-python"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "experiments/compare_scalability_approaches.py" ]]; then\n'
        "  exec sleep 10\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    launcher.chmod(0o755)
    env = {
        **runner_env,
        "PYTHON_BIN": str(launcher),
        "RISK_GRID": "msd",
        "K_GRID": "1",
        "N_GRID": "1",
        "REPS": "1",
        "WORKERS": "1",
        "METHOD_TIME_LIMIT_SECONDS": "1",
    }
    directory = Path(env["RESULT_DIR"])
    for retry in ("0", "1"):
        result = subprocess.run(
            ["bash", str(RUNNER)],
            env={**env, "RETRY_ERRORS": retry},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with (directory / "capped_method_results.csv").open(newline="") as f:
            row = next(csv.DictReader(f))
        assert row["status"] == "timeout"
        assert row["censored"] == "True"
        assert row["time_s"] == "1"
        assert row["exit_code"] == "124"
        assert len(list((directory / "logs").glob("*_attempt*.log"))) == 1


@pytest.mark.parametrize("rsync_exit", [0, 23])
def test_qptas_sync_destination_success_counts_and_transfer_failure(runner_env, tmp_path, rsync_exit):
    # Copy the standalone script to prove that its destination follows its checkout,
    # including spaces, regardless of the caller's working directory.
    checkout = tmp_path / "local checkout"
    script = checkout / "experiments/sync_qptas_results.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments/sync_qptas_results.sh", script)
    destination = checkout / "experiments/results/remote/qptas_scalability/sampled_1000_v1"
    destination.mkdir(parents=True)
    notes = destination / "local_notes.txt"
    notes.write_text("keep this")
    source_csv = tmp_path / "source.csv"
    with source_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["risk", "method", "status", "success"])
        writer.writerows(
            [
                ["msd", "qptas", "completed", "True"],
                ["msd", "qptas", "completed", "False"],
                ["msd", "qptas", "error", "False"],
                ["cvar", "qptas", "completed", "True"],
                ["cvar", "qptas", "completed", "False"],
                ["cvar", "qptas", "timeout", "True"],  # Non-completed rows cannot count as successes.
                ["msd", "mcp", "completed", "True"],  # Ignore other methods.
            ]
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_rsync = bin_dir / "rsync"
    fake_rsync.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$RSYNC_ARGS_FILE"\n'
        'if (( RSYNC_EXIT_CODE != 0 )); then exit "$RSYNC_EXIT_CODE"; fi\n'
        'cp "$FIXTURE_CSV" "${@: -1}/capped_method_results.csv"\n'
    )
    fake_rsync.chmod(0o755)
    args_file = tmp_path / "rsync_args.txt"
    env = {
        **runner_env,
        "PATH": str(bin_dir) + os.pathsep + runner_env["PATH"],
        "RSYNC_ARGS_FILE": str(args_file),
        "RSYNC_EXIT_CODE": str(rsync_exit),
        "FIXTURE_CSV": str(source_csv),
    }
    result = subprocess.run(
        ["bash", str(script), "root@example.invalid"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == rsync_exit, result.stdout + result.stderr
    arguments = args_file.read_text().splitlines()
    assert arguments[-2] == (
        "root@example.invalid:/root/cumg/experiments/results/remote/qptas_scalability/sampled_1000_v1/"
    )
    assert arguments[-1] == str(destination) + "/"
    assert "--delete" not in arguments
    assert "--exclude=*.partial.csv" in arguments
    assert "--exclude=*.lock/" in arguments
    assert notes.read_text() == "keep this"
    if rsync_exit:
        assert "Success rate" not in result.stdout
    else:
        assert "MSD 1 / 3 33.3%" in " ".join(result.stdout.split())
        assert "CVAR 1 / 3 33.3%" in " ".join(result.stdout.split())
        assert "TOTAL 2 / 6 33.3%" in " ".join(result.stdout.split())
        assert "completed=4, error=1, timeout=1" in result.stdout
