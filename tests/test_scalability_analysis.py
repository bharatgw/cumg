from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import scalability_analysis as analysis  # noqa: E402


def test_capped_results_to_wide_recovers_metadata_from_completed_method() -> None:
    capped = pd.DataFrame(
        [
            {
                "risk": "cvar",
                "K": 500,
                "n": 20,
                "rep": 5,
                "seed": 123,
                "method": "mcp",
                "gamma": 0.5,
                "alpha": 0.5,
                "success": True,
                "time_s": 12.0,
                "eta": 0.1,
                "status": "completed",
                "censored": False,
                "time_limit_s": 86400,
            },
            {
                "risk": "cvar",
                "K": 500,
                "n": 20,
                "rep": 5,
                "seed": 123,
                "method": "action_dual",
                "gamma": None,
                "alpha": None,
                "success": False,
                "time_s": 86400.0,
                "eta": None,
                "status": "timeout",
                "censored": True,
                "time_limit_s": 86400,
            },
        ]
    )

    wide = analysis.capped_results_to_wide(capped, expected_methods=("mcp", "action_dual"))

    assert len(wide) == 1
    assert wide.loc[0, "gamma"] == pytest.approx(0.5)
    assert wide.loc[0, "alpha"] == pytest.approx(0.5)
    assert wide.loc[0, "action_dual_status"] == "timeout"
    assert wide.loc[0, "action_dual_time_s"] == pytest.approx(86400)


def test_scalability_summary_separates_capped_time_timeout_and_successful_eta() -> None:
    long = pd.DataFrame(
        {
            "risk": ["cvar"] * 4,
            "n": [20] * 4,
            "K": [500] * 4,
            "method": ["mcp"] * 4,
            "status": ["completed", "completed", "completed", "timeout"],
            "success": [True, True, False, False],
            "time_s": [1.0, 3.0, 5.0, 24.0],
            "eta": [0.1, 0.2, 0.9, None],
        }
    )

    result = analysis.summarize_scalability(long).iloc[0]

    assert result["reps"] == 4
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["timeout_rate"] == pytest.approx(0.25)
    assert result["capped_time_median"] == pytest.approx(4.0)
    assert result["eta_reps"] == 2
    assert result["eta_median"] == pytest.approx(0.15)


def test_eta_history_never_crosses_risk_or_hyperparameter_trajectories() -> None:
    rows = []
    for risk, entropy_kappa, values in (
        ("msd", 0.1, (5.0, 4.0)),
        ("cvar", 0.1, (1.0, 2.0)),
        ("cvar", 0.2, (10.0, 9.0)),
    ):
        for iteration, eta in enumerate(values):
            rows.append(
                {
                    "risk": risk,
                    "K": 100,
                    "n": 10,
                    "method": "full_batch",
                    "seed": 7,
                    "entropy_kappa": entropy_kappa,
                    "smoothing_tau": 0.01,
                    "step_size": 10.0,
                    "iteration": iteration,
                    "eta": eta,
                }
            )
    history = pd.DataFrame(rows)

    prepared = analysis.prepare_eta_improvement(history)
    cvar_point = prepared.loc[
        prepared["risk"].eq("cvar") & prepared["entropy_kappa"].eq(0.1) & prepared["iteration"].eq(1)
    ].iloc[0]
    other_entropy = prepared.loc[
        prepared["risk"].eq("cvar") & prepared["entropy_kappa"].eq(0.2) & prepared["iteration"].eq(0)
    ].iloc[0]

    assert cvar_point["best_eta"] == pytest.approx(1.0)
    assert cvar_point["initial_eta"] == pytest.approx(1.0)
    assert cvar_point["eta_improvement_pct"] == pytest.approx(0.0)
    assert other_entropy["initial_eta"] == pytest.approx(10.0)


def test_eta_history_uses_shard_to_separate_repeated_configs() -> None:
    history = pd.DataFrame(
        {
            "shard": ["first", "first", "rerun", "rerun"],
            "risk": ["cvar"] * 4,
            "K": [100] * 4,
            "n": [10] * 4,
            "method": ["full_batch"] * 4,
            "seed": [7] * 4,
            "entropy_kappa": [0.1] * 4,
            "smoothing_tau": [0.01] * 4,
            "step_size": [10.0] * 4,
            "iteration": [0, 1, 0, 1],
            "eta": [1.0, 0.5, 10.0, 9.0],
        }
    )

    prepared = analysis.prepare_best_eta_history(history)

    rerun = prepared.loc[prepared["shard"].eq("rerun")]
    assert rerun["best_eta"].tolist() == [10.0, 9.0]


def test_eta_history_joins_continuation_stages_within_shard() -> None:
    history = pd.DataFrame(
        {
            "shard": ["run"] * 4,
            "risk": ["cvar"] * 4,
            "K": [250] * 4,
            "n": [20] * 4,
            "method": ["full_batch"] * 4,
            "seed": [7] * 4,
            "entropy_kappa": [0.1, 0.1, 0.03, 0.03],
            "smoothing_tau": [0.02, 0.02, 0.01, 0.01],
            "step_size": [10.0] * 4,
            "continuation_stage": [0, 0, 1, 1],
            "stage_iteration": [0, 1, 0, 1],
            "iteration": [0, 1, 1, 2],
            "eta": [10.0, 8.0, 9.0, 6.0],
        }
    )

    prepared = analysis.prepare_eta_improvement(history)

    assert prepared["continuation_stage"].tolist() == [0, 1, 1]
    assert prepared["best_eta"].tolist() == [10.0, 8.0, 6.0]
    assert prepared["initial_eta"].tolist() == [10.0, 10.0, 10.0]
    assert prepared["eta_improvement_pct"].tolist() == pytest.approx([0.0, 20.0, 40.0])


def test_prepare_v3_history_is_safe_to_rerun() -> None:
    summary = pd.DataFrame({"shard": ["a", "b"], "step_size": [10.0, 20.0]})
    history = pd.DataFrame(
        {
            "shard": ["a", "a", "b", "b"],
            "method": ["full_batch"] * 4,
            "iteration": [0, 1, 0, 1],
            "eta": [3.0, 2.0, 5.0, 6.0],
        }
    )

    once = analysis.prepare_v3_history(history, summary)
    twice = analysis.prepare_v3_history(once, summary)

    pd.testing.assert_frame_equal(once, twice)
    assert once["best_eta_so_far"].tolist() == [3.0, 2.0, 5.0, 5.0]


def test_prepare_v3_plot_history_keeps_a_fixed_cohort() -> None:
    summary = pd.DataFrame({"shard": ["short", "long"], "step_size": [10.0, 10.0]})
    history = pd.DataFrame(
        {
            "shard": ["short", "short", "long", "long", "long"],
            "risk": ["cvar"] * 5,
            "K": [250] * 5,
            "n": [20] * 5,
            "method": ["full_batch"] * 5,
            "iteration": [0, 1, 0, 1, 2],
            "eta": [3.0, 1.0, 5.0, 4.0, 3.0],
        }
    )

    prepared = analysis.prepare_v3_plot_history(history, summary)
    counts = prepared.groupby("iteration")["shard"].nunique()
    medians = prepared.groupby("iteration")["best_eta_so_far"].median()

    assert counts.tolist() == [2, 2, 2]
    assert medians.tolist() == pytest.approx([4.0, 2.5, 2.0])
    assert medians.diff().dropna().le(0).all()
