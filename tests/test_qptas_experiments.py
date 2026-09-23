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

from cumg.small_support import full_cvar_regret, full_msd_regret  # noqa: E402
from experiments.runners import compare_scalability_approaches as driver  # noqa: E402


@pytest.fixture
def qptas_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(driver.__file__), "--methods", "qptas", "--max-candidates", "3", "--reps", "1", "--quiet"],
    )
    return driver.parse_args()


@pytest.mark.parametrize("risk", ["msd", "cvar"])
@pytest.mark.parametrize("method", ["qptas", "qptas_screened"])
def test_qptas_driver_uses_full_game_and_records_certificate(qptas_args, monkeypatch, risk, method):
    args = qptas_args
    args.methods = [method]
    args.epsilon = 1.0  # Every profile of this bounded game passes, exposing its certificate.
    solve = getattr(driver, f"solve_{risk}_{method}")
    captured = {}

    def capture(payoffs, p, **kwargs):
        captured.update(payoffs=payoffs, p=p, kwargs=kwargs)
        captured["result"] = solve(payoffs, p, **kwargs)
        return captured["result"]

    monkeypatch.setattr(driver, f"solve_{risk}_{method}", capture)
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
        **({"tau": 4, "epsilon_scr": 2 / 3} if method == "qptas_screened" else {}),
    }
    x, y = driver._profile_from_result(captured["result"])
    if risk == "msd":
        cert = full_msd_regret(A, B, p, args.gamma, x, y)
    else:
        cert = full_cvar_regret(A, B, p, args.gamma, args.alpha, x, y)
    assert row[f"{method}_success"]
    assert row[f"{method}_has_profile"]
    assert row[f"{method}_error"] is None
    assert row[f"{method}_termination_reason"] == "epsilon_reached"
    assert row[f"{method}_profiles_checked"] == 1
    assert row[f"{method}_total_profiles"] == 36
    assert row[f"{method}_best_response_solves"] == 2
    assert row[f"{method}_sampling_seed"] == 123
    if method == "qptas_screened":
        assert row[f"{method}_total_pairs"] == 36 * 35**2
        assert row[f"{method}_screen_passes"] == 1
        assert row[f"{method}_screen_rejections"] == 0
    for key in ("eta", "regret1", "regret2"):
        assert row[f"{method}_{key}"] == pytest.approx(cert[key], abs=1e-10)


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


@pytest.mark.parametrize("method", ["qptas", "qptas_screened"])
def test_qptas_error_rows_have_stable_csv_schema(qptas_args, monkeypatch, tmp_path, method):
    qptas_args.methods = [method]
    solve = getattr(driver, f"solve_msd_{method}")

    def fail(*args, **kwargs):
        raise RuntimeError("LP failed")

    monkeypatch.setattr(driver, f"solve_msd_{method}", fail)
    failed = driver.run_instance(qptas_args, "msd", K=2, n=2, seed=123)
    monkeypatch.setattr(driver, f"solve_msd_{method}", solve)
    qptas_args.epsilon = 1.0
    passed = driver.run_instance(qptas_args, "msd", K=2, n=2, seed=123)
    assert failed[f"{method}_termination_reason"] == "error"
    assert failed[f"{method}_error"] == "LP failed"
    assert passed[f"{method}_success"]
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
    assert "qptas_screened" not in args.methods
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
    args.methods = ["qptas", "qptas_screened", "stochastic_full_batch", "stochastic_minibatch"]
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
    assert len(games) == 4
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


def campaign_preset(tmp_path, *, tiny=False, gamma=0.5, source="population_qptas_fo/beta_uniform_v2"):
    data = json.loads((ROOT / "experiments/configs" / (source + ".json")).read_text())
    data["campaign"] = "test/fresh"
    data["result_dir"] = str(tmp_path / "results")
    data["historical_env"]["LEGACY_RESULT_DIR"] = str(tmp_path / "legacy")
    if tiny:
        data["historical_env"].update(
            K_GRID="2",
            N_GRID="2",
            REPS="1",
            WORKERS="1",
            MAX_ITER="3",
            CERTIFY_EVERY="1",
            STAGNATION_WINDOW="1",
            STAGNATION_RTOL="1",
            STAGNATION_ATOL="0",
            MAX_CANDIDATES="2",
            GAMMA=str(gamma),
        )
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data))
    return path, data


def campaign_command(path, command="run", *extra):
    return [sys.executable, "-m", "experiments", command, "--campaign", "test/fresh", "--config", str(path), *extra]


def execute_campaign(path, command="run", *extra):
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    return subprocess.run(
        campaign_command(path, command, *extra), cwd=ROOT, env=env, capture_output=True, text=True, timeout=90
    )


@pytest.mark.parametrize(
    "source,expected", [("population_qptas_fo/beta_uniform_v2", 640), ("qptas_scalability/sampled_1000_v1", 960)]
)
def test_named_preset_plan_does_not_write_or_launch(tmp_path, source, expected):
    path, data = campaign_preset(tmp_path, source=source)
    result = execute_campaign(path, "plan")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["expected"] == report["pending"] == expected
    assert len(report["pending_tasks"]) == expected
    assert not Path(data["result_dir"]).exists()


@pytest.mark.skipif(shutil.which("timeout") is None, reason="Requires GNU timeout")
@pytest.mark.parametrize("gamma", [0.5, -1])
def test_named_campaign_runs_all_methods_and_resumes(tmp_path, gamma):
    pytest.importorskip("jax")
    path, data = campaign_preset(tmp_path, tiny=True, gamma=gamma)
    result = execute_campaign(path)
    assert result.returncode == 0, result.stdout + result.stderr
    directory = Path(data["result_dir"])
    with (directory / "capped_method_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["status"] for row in rows} == ({"completed"} if gamma > 0 else {"error"})
    files = {str(p): p.read_bytes() for p in directory.rglob("*.csv")}
    repeated = execute_campaign(path)
    assert repeated.returncode == 0, repeated.stderr
    assert "No pending work" in repeated.stdout
    assert files == {str(p): p.read_bytes() for p in directory.rglob("*.csv")}
    if gamma > 0:
        profiles = list(directory.glob("method_shards/*/profiles/*.json"))
        assert len(profiles) == 8
        for profile in profiles:
            record = json.loads(profile.read_text())
            assert record["schema_version"] == 2
            assert record["inputs"]["A"]["shape"] == [2, 2, 2]
            assert record["environment"]["packages"]["numpy"]
            if record["method"].startswith("stochastic_"):
                assert len(record["x"]) == 2
                assert record["history"]
            else:
                assert record["configuration"]["step_size"] is None
    else:
        retry = execute_campaign(path, "run", "--retry-errors")
        assert retry.returncode == 0, retry.stderr
        assert len(list((directory / "logs").glob("*_status_before_attempt001.json"))) == 8


def test_configuration_changes_cannot_reuse_results(tmp_path):
    path, data = campaign_preset(tmp_path, tiny=True)
    directory = Path(data["result_dir"])
    directory.mkdir()
    config = directory / "run_config.env"
    config.write_text("\n".join(f"{key}={value}" for key, value in data["historical_env"].items()) + "\n")
    original = config.read_bytes()
    data["historical_env"]["STOCHASTIC_CVAR_STEP_SIZE"] = "1"
    path.write_text(json.dumps(data))
    result = execute_campaign(path)
    assert result.returncode != 0
    assert "Configuration differs" in result.stderr
    assert config.read_bytes() == original


def test_tuning_preview_never_launches_or_writes(tmp_path):
    data = {
        "campaign": "test/fresh",
        "engine": "tuning",
        "result_dir": str(tmp_path / "results"),
        "phases": {"pilot": {"arguments": [], "outputs": [str(tmp_path / "results/summary.csv")]}},
    }
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data))
    result = execute_campaign(path, "run", "--phase", "pilot", "--preview")
    assert result.returncode == 0, result.stderr
    assert not Path(data["result_dir"]).exists()
    assert json.loads(result.stdout)["command"][-1] == "--save-profiles"


@pytest.mark.parametrize("status", [0, 23])
def test_sync_propagates_transfer_status_and_uses_campaign_paths(monkeypatch, status):
    from experiments import __main__ as cli

    captured = []

    def transfer(command, **kwargs):
        if command == ["rsync", "--help"]:
            return subprocess.CompletedProcess(command, 0, stdout="--protect-args", stderr="")
        captured.append((command, kwargs))
        if status:
            raise subprocess.CalledProcessError(status, command)

    monkeypatch.setattr(subprocess, "run", transfer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiments",
            "sync",
            "--campaign",
            "population_qptas_fo/beta_uniform_v2",
            "--remote",
            "user@example",
            "--preview",
        ],
    )
    if status:
        with pytest.raises(subprocess.CalledProcessError):
            cli.main()
    else:
        cli.main()
    command, kwargs = captured[0]
    assert "--dry-run" in command and "--delete" not in command and kwargs["check"]
    assert command[-2].endswith("/population_qptas_fo/beta_uniform_v2/")
    assert command[-1].endswith("/population_qptas_fo/beta_uniform_v2/")


@pytest.mark.parametrize("help_text", ["--protect-args", "--secluded-args", "openrsync usage"])
@pytest.mark.parametrize("remote_path", ["/srv/new-run", "/srv/run with spaces/it's literal;$(echo nope)"])
def test_sync_accepts_new_campaign_config(tmp_path, monkeypatch, help_text, remote_path):
    from experiments import __main__ as cli

    path, data = campaign_preset(tmp_path)
    commands = []

    def transfer(command, **kwargs):
        if command == ["rsync", "--help"]:
            return subprocess.CompletedProcess(command, 0, stdout=help_text, stderr="")
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", transfer)
    monkeypatch.setattr(
        sys,
        "argv",
        campaign_command(path, "sync", "--remote", "user@example", "--remote-path", remote_path, "--preview")[2:],
    )
    cli.main()
    command = commands[0]
    remote_source = command[-2].removeprefix("user@example:")
    if help_text.startswith("--"):
        assert help_text in command
        assert remote_source == remote_path + "/"
    else:
        assert "--protect-args" not in command and "--secluded-args" not in command
        assert shlex.split(remote_source) == [remote_path + "/"]
    assert {"--dry-run", "--exclude=*.lock/", "--exclude=*.partial*", "--exclude=*.tmp"} <= set(command)
    assert "--delete" not in command
    assert commands[0][-1] == data["result_dir"] + "/"
    assert not Path(data["result_dir"]).exists()
