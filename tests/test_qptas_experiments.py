import csv
import json
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
POPULATION_RUNNER = ROOT / "experiments" / "run_population_qptas_fo_remote.sh"


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
    assert args.payoff_model == "uniform"


def test_population_game_has_fixed_seeded_cell_assignments_and_expected_moments():
    A, B, p, populations = driver.simulate_population_payoffs(K=5000, n=10, seed=27)
    assert A.shape == B.shape == (5000, 10, 10)
    assert populations.shape == (2, 10, 10)
    assert set(populations.ravel()) == set(range(5))
    np.testing.assert_array_equal(p, np.full(5000, 1 / 5000))
    assert not np.array_equal(populations[0], populations[1])
    means = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
    variances = np.array([4 / 150, 6 / 150, 6 / 150, 4 / 150, 1 / 12])
    for player, payoff in enumerate((A, B)):
        assert np.isfinite(payoff).all() and ((payoff >= 0) & (payoff <= 1)).all()
        # Checking each cell rules out drawing a new population per scenario.
        np.testing.assert_allclose(payoff.mean(axis=0), means[populations[player]], atol=0.02)
        for label in range(5):
            draws = payoff[:, populations[player] == label]
            assert draws.var() == pytest.approx(variances[label], abs=0.002)
    short_A, short_B, _, short_populations = driver.simulate_population_payoffs(K=2, n=10, seed=27)
    np.testing.assert_array_equal(populations, short_populations)
    np.testing.assert_array_equal(A[:2], short_A)
    np.testing.assert_array_equal(B[:2], short_B)
    _, _, _, different = driver.simulate_population_payoffs(K=2, n=10, seed=28)
    assert not np.array_equal(populations, different)
    _, _, _, large_map = driver.simulate_population_payoffs(K=1, n=50, seed=27)
    frequencies = np.bincount(large_map.ravel(), minlength=5) / large_map.size
    np.testing.assert_allclose(frequencies, np.full(5, 0.2), atol=0.025)


@pytest.mark.parametrize("K,n", [(0, 2), (2, 0), (1.5, 2), (2, True)])
def test_population_generator_rejects_invalid_dimensions(K, n):
    with pytest.raises(ValueError, match="positive integer"):
        driver.simulate_population_payoffs(K=K, n=n, seed=0)


def test_legacy_uniform_generator_keeps_original_draws():
    A, B, p = driver.simulate_random_payoffs(K=3, n=2, seed=123, low=-1, high=2)
    rng = np.random.default_rng(123)
    np.testing.assert_array_equal(A, rng.uniform(-1, 2, size=(3, 2, 2)))
    np.testing.assert_array_equal(B, rng.uniform(-1, 2, size=(3, 2, 2)))
    np.testing.assert_array_equal(p, np.full(3, 1 / 3))


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_population_driver_shares_all_samples_across_methods(qptas_args, monkeypatch, risk):
    pytest.importorskip("jax")
    args = qptas_args
    args.payoff_model = "cell_beta_uniform_v1"
    args.methods = ["qptas", "stochastic_full_batch", "stochastic_minibatch"]
    args.max_iter = 0
    args.certify_every = 1
    args.epsilon = 1
    run_method = driver._run_method
    games = []

    def capture(method, risk, A, B, p, *other):
        games.append((A, B, p))
        return run_method(method, risk, A, B, p, *other)

    monkeypatch.setattr(driver, "_run_method", capture)
    row = driver.run_instance(args, risk, K=3, n=2, seed=123)
    A, B, p, populations = driver.simulate_population_payoffs(K=3, n=2, seed=123)
    for game in games:
        for actual, expected in zip(game, (A, B, p), strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert len(games) == 3
    assert row["payoff_model"] == args.payoff_model
    assert json.loads(row["payoff_populations"]) == list(driver.PAYOFF_POPULATIONS)
    np.testing.assert_array_equal(json.loads(row["payoff_population_ids"]), populations)
    assert all(row[f"{method}_success"] for method in args.methods)
    assert row["stochastic_full_batch_starts_attempted"] == row["stochastic_minibatch_starts_attempted"] == 1


def test_population_driver_rejects_rescaling(qptas_args):
    qptas_args.payoff_model = "cell_beta_uniform_v1"
    qptas_args.high = 2
    with pytest.raises(ValueError, match="low=0 and high=1"):
        driver.run_instance(qptas_args, "msd", K=2, n=2, seed=123)


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


def test_population_shell_default_plan(runner_env):
    result = subprocess.run(
        ["bash", str(POPULATION_RUNNER)],
        env={**runner_env, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    directory = Path(runner_env["RESULT_DIR"])
    rows = [line.split("\t") for line in (directory / "pending_methods.tsv").read_text().splitlines()]
    assert len(rows) == 360
    assert {row[0] for row in rows} == {"msd", "cvar"}
    assert {int(row[1]) for row in rows} == {1000, 2000, 4000}
    assert {int(row[2]) for row in rows} == {50}
    assert {int(row[3]) for row in rows} == set(range(20))
    assert {row[4] for row in rows} == {"qptas", "stochastic_full_batch", "stochastic_minibatch"}
    for risk, K, n, rep, _, seed in rows:
        assert int(seed) == driver.experiment_seed(risk, int(K), int(n), int(rep), 123)
    config = set((directory / "run_config.env").read_text().splitlines())
    assert {
        "EPSILON=0.01",
        "STOCHASTIC_REGRET_TOLERANCE=0.01",
        "STOCHASTIC_N_RANDOM_STARTS=4",
        "PAYOFF_MODEL=cell_beta_uniform_v1",
        "MAX_ITER=2000",
        "MAX_CANDIDATES=1000",
        "WORKERS=8",
        "CERTIFY_EVERY=1",
        "STAGNATION_WINDOW=500",
        "STAGNATION_RTOL=0.005",
        "STAGNATION_ATOL=0.00001",
        "GAMMA=0.5",
        "ALPHA=0.5",
        "METHOD_TIME_LIMIT_SECONDS=86400",
    } <= config


@pytest.mark.skipif(shutil.which("timeout") is None, reason="Requires GNU timeout")
@pytest.mark.parametrize("gamma", [0.5, -1])
def test_population_shell_runs_and_resumes_all_three_methods(runner_env, gamma):
    pytest.importorskip("jax")
    env = {
        **runner_env,
        "K_GRID": "2",
        "N_GRID": "2",
        "REPS": "1",
        "WORKERS": "1",
        "MAX_ITER": "3",
        "CERTIFY_EVERY": "1",
        "STAGNATION_WINDOW": "1",
        "STAGNATION_RTOL": "1",
        "STAGNATION_ATOL": "0",
        "MAX_CANDIDATES": "2",
        "GAMMA": str(gamma),
    }
    run = subprocess.run(["bash", str(POPULATION_RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    directory = Path(env["RESULT_DIR"])
    with (directory / "capped_method_results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    for row in rows:
        if gamma < 0:
            assert row["status"] == "error" and row["success"] == "False"
            continue
        assert row["status"] == "completed" and row["error"] == ""
        assert row["payoff_model"] == "cell_beta_uniform_v1"
        *_, populations = driver.simulate_population_payoffs(K=2, n=2, seed=int(row["seed"]))
        np.testing.assert_array_equal(json.loads(row["payoff_population_ids"]), populations)
        if row["method"] == "qptas":
            assert 1 <= int(row["profiles_checked"]) <= 2
        else:
            assert row["stochastic_certify_every"] == "1"
            assert row["stochastic_stagnation_window"] == "1"
            assert float(row["stochastic_stagnation_rtol"]) == 1
            assert float(row["stochastic_stagnation_atol"]) == 0
            starts = json.loads(row["start_summaries"])
            assert 1 <= len(starts) == int(row["starts_attempted"]) <= 5
            assert int(row["iterations"]) == sum(start["iterations"] for start in starts)
            assert all(start["iterations"] <= 1 for start in starts)
            assert all(start["termination_reason"] in {"stagnation", "regret_tolerance"} for start in starts)
            assert all(start["termination_reason"] == "stagnation" for start in starts[:-1])
            assert (row["success"] == "True") == (float(row["eta"]) <= 0.01)
    shards = {p: p.stat().st_mtime_ns for p in (directory / "method_shards").rglob("*") if p.is_file()}
    resumed = subprocess.run(["bash", str(POPULATION_RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (directory / "pending_methods.tsv").read_text() == ""
    assert {p: p.stat().st_mtime_ns for p in shards} == shards

    changed = subprocess.run(
        ["bash", str(POPULATION_RUNNER)],
        env={**env, "STAGNATION_WINDOW": "2", "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert changed.returncode == 2
    assert "Configuration differs" in changed.stderr


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
