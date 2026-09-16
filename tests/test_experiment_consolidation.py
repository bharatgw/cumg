import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.common.campaign import check_saved_config
from experiments.common.experiment import certificate_success
from experiments.diagnostics import regret_audit as audit
from experiments.diagnostics.migration import migrate
from experiments.diagnostics.replay import arguments_for, run_bounded, select_seeds, summarize


def test_common_success_rule_matches_boundary_for_arrays_and_scalars():
    statuses = ["completed", "completed", "completed", "timeout", "completed", "completed"]
    etas = [0, 0.01, 0.010000001, 0.001, np.nan, np.inf]
    assert certificate_success(statuses, etas, 0.01).tolist() == [True, True, False, False, False, False]
    assert certificate_success("completed", 0, 0.01)


def test_migration_preflight_idempotence_and_rollback(tmp_path):
    source = tmp_path / "old/a.csv"
    source.parent.mkdir()
    source.write_bytes(b"a,b\r\n1,2\r\n")
    row = {
        "old": "old/a.csv",
        "new": "new/a.csv",
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert migrate(tmp_path, [row])["pending"] == 1
    assert source.exists()
    assert migrate(tmp_path, [row], apply=True)["moved"] == 1
    assert migrate(tmp_path, [row], apply=True)["moved"] == 0
    assert migrate(tmp_path, [row], apply=True, reverse=True)["moved"] == 1
    target = tmp_path / "new/a.csv"
    target.write_text("collision")
    with pytest.raises(ValueError, match="collision"):
        migrate(tmp_path, [row], apply=True)
    assert source.exists()


def test_resume_relocation_allows_only_known_path_change(tmp_path):
    path = tmp_path / "config.env"
    path.write_text("LEGACY_RESULT_DIR=experiments/results/remote/cvar_scalability/v1\nGAMMA=0.5\n")
    proposed = {"LEGACY_RESULT_DIR": "experiments/results/scalability/cvar_uncapped_v1", "GAMMA": "0.5"}
    check_saved_config(path, proposed)
    with pytest.raises(ValueError, match="GAMMA"):
        check_saved_config(path, {**proposed, "GAMMA": "0.6"})
    with pytest.raises(ValueError, match="LEGACY"):
        check_saved_config(path, {**proposed, "LEGACY_RESULT_DIR": "unrelated"})


@pytest.fixture
def certificate_game():
    return np.array([[[0.0, 0.0], [0.01, 0.01]]]), np.zeros((1, 2, 2)), np.ones(1)


@pytest.mark.parametrize("saved,status", [(0.01, "match"), (0.001, "mismatch"), (0.010000001, "classification_flip")])
def test_regret_audit_checks_threshold_separately(certificate_game, saved, status):
    report = audit.audit_profile(*certificate_game, "msd", 0, 0.5, [1.0, 0.0], [0.5, 0.5], {"eta": saved})
    assert report["status"] == status
    assert report["max_independent_payoff_error"] < 1e-12


def test_regret_audit_rejects_invalid_simplex_and_lp_failure(certificate_game, monkeypatch):
    report = audit.audit_profile(*certificate_game, "msd", 0, 0.5, [1.0, -1e-12], [0.5, 0.5], {"eta": 0.01})
    assert report["status"] == "invalid_simplex"

    def fail(*args):
        raise RuntimeError("LP failed")

    monkeypatch.setattr(audit, "full_msd_regret", fail)
    report = audit.audit_profile(*certificate_game, "msd", 0, 0.5, [1.0, 0.0], [0.5, 0.5], {"eta": 0.01})
    assert report["status"] == "lp_failure"


def test_missing_profile_is_distinct_from_exhaustion():
    item = {"method": "qptas", "row": {}, "metrics": {"eta": "nan", "has_profile": "False"}}
    assert audit.profile_state(item)[0] == "intentionally_absent"
    item["metrics"] = {"eta": "0.005", "has_profile": "True"}
    assert audit.profile_state(item)[0] == "missing_profile"


def test_replay_selection_uses_three_distinct_seeds_without_outcome_selection():
    items = [
        {
            "campaign": "study/run",
            "method": "stochastic_full_batch",
            "row": {"risk": risk, "n": "5", "K": "30", "seed": str(seed)},
            "metrics": {"eta": "0.001", "has_profile": "True", "time_s": "1"},
        }
        for risk, seeds in [("msd", range(5)), ("cvar", range(5, 10))]
        for seed in seeds
    ]
    selection = select_seeds(items)["study/run"]
    keys = [(g[0]["row"]["risk"], g[0]["row"]["seed"]) for g in selection]
    assert len({seed for _, seed in keys}) == 3
    assert {risk for risk, _ in keys} == {"msd", "cvar"}
    for item in items:
        item["metrics"]["eta"] = "0.1"
    assert keys == [(g[0]["row"]["risk"], g[0]["row"]["seed"]) for g in select_seeds(items)["study/run"]]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group deadline")
def test_replay_deadline_kills_children(tmp_path):
    sentinel = tmp_path / "child_survived"
    child = f"import time,pathlib; time.sleep(0.7); pathlib.Path({str(sentinel)!r}).touch()"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    result = run_bounded([sys.executable, "-c", parent], 0.2, tmp_path / "worker.log")
    assert result["status"] == "timeout" and result["elapsed_s"] < 2
    time.sleep(0.8)
    assert not sentinel.exists()


@pytest.mark.parametrize("risk", ["msd", "cvar"])
def test_checkpoint_capture_preserves_numerical_path(risk):
    pytest.importorskip("jax")
    from cumg import StochasticFOConfig, solve_cvar_stochastic_fo, solve_msd_stochastic_fo
    from experiments.common.experiment import simulate_population_payoffs

    A, B, p, _ = simulate_population_payoffs(5, 3, 17)
    config = StochasticFOConfig(
        max_iter=4, certify_every=2, record_every=None, regret_tolerance=0, n_random_starts=1, seed=17
    )
    solve = solve_msd_stochastic_fo if risk == "msd" else solve_cvar_stochastic_fo
    kwargs = {"gamma": 0.5, **({"alpha": 0.5} if risk == "cvar" else {})}
    original = solve(A, B, p, config=config, **kwargs)
    captured = solve(A, B, p, config=replace(config, capture_certificates=True), **kwargs)
    np.testing.assert_array_equal(original.x, captured.x)
    np.testing.assert_array_equal(original.y, captured.y)
    assert original.certificate["eta"] == captured.certificate["eta"]
    assert original.iterations == captured.iterations
    assert original.selected_start == captured.selected_start
    assert captured.history


def test_committed_v2_summary_uses_all_attempts_and_cross_rule():
    from experiments.analysis import generate_readme_figures as figures

    data = figures.load_population_data()
    summary = figures._summarize_certified_runs(data)
    assert len(data) == 640 and len(summary) == 32
    assert summary.successes.sum() == 135
    assert summary.successes.lt(5).sum() == 18
    assert summary.reps.eq(20).all()
    assert data.loc[data.method.str.startswith("qptas"), "eta"].isna().all()


def test_profile_sidecar_saves_certificate_without_solver_objects(tmp_path, monkeypatch, certificate_game):
    from experiments.common import provenance

    monkeypatch.setattr(provenance, "environment", lambda: {})
    result = SimpleNamespace(
        success=False,
        metadata={"best_candidate": {"solver_result": object(), "certificate": {"eta": np.float64(0.02)}}},
    )
    path = tmp_path / "profile.json"
    provenance.save_profile(
        path,
        args=SimpleNamespace(),
        row={"risk": "msd", "K": 1, "n": 2, "seed": 0},
        method="restricted_mcp",
        result=result,
        profile=([1.0, 0.0], [0.5, 0.5]),
        arrays=certificate_game,
    )
    saved = json.loads(path.read_text())
    assert saved["certificate"] == {"eta": 0.02}
    assert saved["x"] == [1.0, 0.0]


@pytest.mark.parametrize("identity", ["match", "mismatch", "unavailable"])
def test_sidecar_audit_verifies_original_input_hashes(tmp_path, monkeypatch, certificate_game, identity):
    from experiments.common.provenance import array_hash

    inputs = {name: array_hash(value) for name, value in zip(("A", "B", "p"), certificate_game, strict=True)}
    if identity == "mismatch":
        inputs["A"]["sha256"] = "wrong"
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"x": [1.0, 0.0], "y": [0.5, 0.5], "inputs": inputs if identity != "unavailable" else None})
    )
    item = {
        "key": "test/run:row1:method",
        "campaign": "test/run",
        "source": str(tmp_path / "results.csv"),
        "row_number": 2,
        "method": "stochastic_full_batch",
        "row": {"risk": "msd", "gamma": "0", "K": "1", "n": "2", "seed": "0"},
        "metrics": {"eta": "0.01", "profile_artifact": "profile.json"},
    }
    monkeypatch.setattr(audit, "presented_rows", lambda: iter([item]))
    monkeypatch.setattr(audit, "game_for_row", lambda _: certificate_game)
    monkeypatch.setattr(audit, "environment", lambda: {})
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--output-dir", str(output)])
    audit.main()
    record = json.loads((output / "profile_checks.jsonl").read_text())
    assert (output / "row_inventory.json.gz").is_file()
    if identity == "mismatch":
        assert record["status"] == "unavailable_inputs"
        assert "hashes differ" in record["detail"]
    else:
        assert record["status"] == "match"
        assert ("verified inputs" in record["scope"]) == (identity == "match")


def test_replay_recovers_constant_continuation_step(tmp_path):
    pytest.importorskip("jax")
    item = {
        "method": "full_batch",
        "row": {
            "risk": "msd",
            "gamma": "0.5",
            "step_size": "10",
            "continuation_kappa": "1,0.1",
            "continuation_tau": "0.1,0.01",
            "continuation_max_iter": "100,200",
        },
        "metrics": {},
    }
    _, args = arguments_for(item, tmp_path)
    assert args.step_size == 10
    assert args.continuation_step_size is None
    assert args.continuation_max_iter == [100, 200]
    assert args.n_random_starts == 0


def test_replay_summary_keeps_unfinished_retry_inconclusive(tmp_path):
    item = {
        "key": "study/run:row1:method",
        "campaign": "study/run",
        "source": "results.csv",
        "row_number": 2,
        "method": "full_batch",
        "row": {"risk": "msd", "n": 2, "K": 3, "seed": 7},
        "metrics": {"eta": 0.002},
    }
    (tmp_path / "selection.json").write_text(json.dumps({"study/run": [[item]]}))
    (tmp_path / "execution.json").write_text(json.dumps({"study/run": [{"status": "timeout", "elapsed_s": 100}]}))
    original = tmp_path / "study/run/0/output"
    original.mkdir(parents=True)
    (original / "comparisons.jsonl").write_text(
        json.dumps(
            {"key": item["key"], "status": "unavailable_configuration_or_solver", "detail": "not JSON serializable"}
        )
        + "\n"
    )
    retry = original.parent / "adapter_retry"
    retry.mkdir()
    (retry / "tasks.json").write_text(json.dumps([item]))
    report = summarize(tmp_path)["study/run"]
    assert report["statuses"] == {"timeout_or_unattempted": 1}
    assert report["insufficient_distinct_seeds"]
