from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import generate_readme_figures as figures  # noqa: E402


def test_common_certificate_success_ignores_native_flags_and_rejects_nonfinite_eta():
    rows = pd.DataFrame(
        {
            "risk": ["msd"] * 8,
            "n": [5] * 8,
            "K": [30] * 8,
            "method": ["qptas"] * 8,
            "seed": range(8),
            "status": ["completed"] * 6 + ["timeout", "error"],
            "success": [False, False, True, True, True, True, True, True],
            "eta": [0, "0.01", 0.010000001, np.nan, np.inf, -np.inf, 0.001, 0.001],
            "time_s": range(1, 9),
        }
    )
    original = rows.copy(deep=True)
    summary = figures._summarize_certified_runs(rows).iloc[0]
    assert summary["successes"] == 2
    assert summary["reps"] == 8
    assert summary["success_rate"] == 0.25
    assert summary["capped_time_median"] == 4.5  # All attempts contribute runtime.
    pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize("method", ["qptas", "uniform"])
@pytest.mark.parametrize("value_column", ["success_rate", "capped_time_median"])
def test_crosses_mark_zero_through_four_successes_but_not_five(method, value_column):
    points = pd.DataFrame({"K": [30, 100, 250, 500], "successes": [0, 4, 5, 20], value_column: [0, 0.2, 0.25, 1]})
    fig, ax = figures.plt.subplots()
    try:
        figures._plot_method_points(ax, points, method, value_column)
        ordinary, crosses = ax.collections
        np.testing.assert_array_equal(ordinary.get_offsets()[:, 0], [250, 500])
        np.testing.assert_array_equal(crosses.get_offsets()[:, 0], [30, 100])
        assert figures.METHOD_MARKERS[method] != "x"
    finally:
        figures.plt.close(fig)


def test_both_graphs_pass_the_same_certificate_counts_to_the_markers(monkeypatch, tmp_path):
    rows = pd.DataFrame(
        [
            {
                "risk": "msd",
                "n": 5,
                "K": K,
                "method": "qptas",
                "seed": seed,
                "status": "completed",
                "success": True,
                "time_s": seed + 1,
                "eta": 0.01 if seed < successes else 0.02,
            }
            for K, successes in [(30, 4), (100, 5)]
            for seed in range(20)
        ]
    )
    calls = []
    plot_points = figures._plot_method_points

    def capture(ax, points, method, value_column):
        calls.append((method, value_column, points["successes"].tolist()))
        plot_points(ax, points, method, value_column)

    monkeypatch.setattr(figures, "_plot_method_points", capture)
    figures.plot_runtime(rows, tmp_path / "runtime.png")
    figures.plot_certificate_rate(rows, tmp_path / "certificates.png")
    assert calls == [("qptas", "capped_time_median", [4, 5]), ("qptas", "success_rate", [4, 5])]
    assert (tmp_path / "runtime.png").stat().st_size > 0
    assert (tmp_path / "certificates.png").stat().st_size > 0


@pytest.mark.parametrize("corruption", [None, "mismatched_seed", "duplicate_seed", "missing_seed"])
def test_loader_adds_qptas_for_both_risks_and_checks_seed_cohorts(monkeypatch, corruption):
    monkeypatch.setattr(figures, "K_GRID", (30,))
    monkeypatch.setattr(figures, "N_GRID", (5,))
    capped = pd.DataFrame(
        [
            {
                "risk": risk,
                "K": 30,
                "n": 5,
                "rep": rep,
                "seed": rep,
                "method": method,
                "status": "completed",
                "success": False,
                "censored": False,
                "time_s": 1.0,
                "eta": 0.01,
                "gamma": 0.5,
                "alpha": 0.5 if risk == "cvar" else np.nan,
            }
            for risk in figures.RISK_GRID
            for method in (*figures.sa.SCALABILITY_METHODS, "qptas")
            for rep in range(20)
        ]
    )
    legacy = capped.loc[capped["method"].ne("qptas")]
    msd_wide = figures.sa.capped_results_to_wide(legacy.loc[legacy["risk"].eq("msd")])
    qptas = capped.loc[capped["method"].eq("qptas")].copy()
    if corruption == "mismatched_seed":
        qptas.loc[qptas.index[0], "seed"] = 999
    elif corruption == "duplicate_seed":
        qptas.loc[qptas.index[0], "seed"] = qptas.iloc[1]["seed"]
    elif corruption == "missing_seed":
        qptas = qptas.iloc[1:]
    uniform = capped.loc[capped["method"].eq("mcp"), ["risk", "K", "n", "seed"]].copy()
    uniform = uniform.assign(uniform_eta=0.01, uniform_success=True, uniform_time_s=0.01)
    files = {
        figures.RESULTS_DIR / "remote/cvar_scalability/capped_24h_v1/capped_method_results.csv": legacy.loc[
            legacy["risk"].eq("cvar")
        ],
        figures.RESULTS_DIR / "remote/qptas_scalability/sampled_1000_v1/capped_method_results.csv": qptas,
        figures.RESULTS_DIR / "uniform/uniform_profile_baseline.csv": uniform,
    }
    monkeypatch.setattr(figures.sa, "load_csv_shards", lambda *args: msd_wide.copy())
    monkeypatch.setattr(figures.pd, "read_csv", lambda path: files[path].copy())
    if corruption:
        with pytest.raises(ValueError, match="seeds"):
            figures.load_scalability_data()
    else:
        long = figures.load_scalability_data()
        assert len(long) == 20 * 2 * len(figures.METHODS)
        assert set(long["method"]) == set(figures.METHODS)
        assert long.loc[long["method"].eq("uniform"), "time_s"].eq(0.01).all()
        assert long.loc[long["method"].eq("qptas")].groupby("risk").size().to_dict() == {"msd": 20, "cvar": 20}
        assert figures._summarize_certified_runs(long)["successes"].eq(20).all()
