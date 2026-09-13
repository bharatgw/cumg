"""Generate the static scalability figures embedded in the repository README."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import scalability_analysis as sa  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "figures"
K_GRID = (30, 100, 250, 500)
N_GRID = (5, 10, 20, 50)
RISK_GRID = ("msd", "cvar")
EPSILON = 1e-2
MIN_SUCCESSFUL_SEEDS = 5
CVaR_CAP_SECONDS = 24 * 60 * 60

METHODS = ("mcp", "screened_dual", "action_dual", "restricted_mcp", "qptas", "uniform")
CERTIFICATE_PLOT_ORDER = (
    "uniform",
    "mcp",
    "screened_dual",
    "action_dual",
    "restricted_mcp",
    "qptas",
)
METHOD_LABELS = {
    "mcp": "MCP",
    "screened_dual": "Screened dual",
    "action_dual": "Action dual",
    "restricted_mcp": "Restricted MCP",
    "qptas": "QPTAS (sampled)",
    "uniform": "Uniform baseline",
}
METHOD_COLORS = {
    "mcp": "#0072B2",
    "screened_dual": "#E69F00",
    "action_dual": "#009E73",
    "restricted_mcp": "#D55E00",
    "qptas": "#7B3294",
    "uniform": "#666666",
}
METHOD_LINESTYLES = {
    "mcp": "-",
    "screened_dual": "-",
    "action_dual": "-",
    "restricted_mcp": "-",
    "qptas": "-.",
    "uniform": "--",
}
METHOD_MARKERS = {
    "mcp": "o",
    "screened_dual": "o",
    "action_dual": "o",
    "restricted_mcp": "o",
    "qptas": "D",
    "uniform": "v",
}


def load_scalability_data() -> pd.DataFrame:
    """Load five algorithms and measured uniform certificates on matched seeds."""

    msd_wide = sa.load_csv_shards(RESULTS_DIR / "remote/msd_cvar_part_scalability", "*K*_n*.csv")
    msd_wide = msd_wide.loc[msd_wide["risk"].eq("msd")].copy()

    capped = pd.read_csv(RESULTS_DIR / "remote/cvar_scalability/capped_24h_v1/capped_method_results.csv")
    cvar_wide = sa.capped_results_to_wide(capped, expected_methods=sa.SCALABILITY_METHODS)
    wide = pd.concat([msd_wide, cvar_wide], ignore_index=True, sort=False)
    long = sa.wide_scalability_to_long(wide, methods=sa.SCALABILITY_METHODS)

    qptas = pd.read_csv(RESULTS_DIR / "remote/qptas_scalability/sampled_1000_v1/capped_method_results.csv")
    qptas_wide = sa.capped_results_to_wide(qptas, expected_methods=("qptas",))
    qptas_long = sa.wide_scalability_to_long(qptas_wide, methods=("qptas",))
    long = pd.concat([long, qptas_long], ignore_index=True, sort=False)

    uniform = pd.read_csv(RESULTS_DIR / "uniform/uniform_profile_baseline.csv")
    long = sa.append_uniform_baseline(long, uniform)
    long = long.loc[
        long["risk"].isin(RISK_GRID) & long["n"].isin(N_GRID) & long["K"].isin(K_GRID) & long["method"].isin(METHODS)
    ].copy()

    baseline_times = pd.to_numeric(long.loc[long["method"].eq("uniform"), "time_s"], errors="coerce")
    if not np.isfinite(baseline_times).all() or not baseline_times.gt(0).all():
        raise ValueError("Uniform baseline must have finite positive certification runtimes")

    primary = long.loc[long["method"].isin(METHODS)]
    cells = primary.groupby(["risk", "n", "K", "method"], dropna=False)
    cell_counts = cells.size()
    expected_cells = len(RISK_GRID) * len(N_GRID) * len(K_GRID) * len(METHODS)
    if len(cell_counts) != expected_cells or not cell_counts.eq(20).all() or not cells["seed"].nunique().eq(20).all():
        raise ValueError("Scalability data do not contain 20 distinct seeds for every plotted method cell")
    matched_methods = primary.groupby(["risk", "n", "K", "seed"], dropna=False)["method"].nunique()
    if not matched_methods.eq(len(METHODS)).all():
        raise ValueError("Scalability methods do not use the same seeds within each plotted cell")

    return long


def _summarize_certified_runs(long: pd.DataFrame) -> pd.DataFrame:
    """Use the same certificate-based success rule for both figures.

    Runtime summaries still include all attempts. Native solver success flags
    do not determine whether a recorded profile meets the common tolerance.
    """

    work = long.copy()
    work["eta"] = pd.to_numeric(work["eta"], errors="coerce")
    work["success"] = work["status"].eq("completed") & np.isfinite(work["eta"]) & work["eta"].le(EPSILON)
    return sa.summarize_scalability(work)


def _style_axes(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)


def _method_legend(methods: tuple[str, ...]) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2,
            markersize=5,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            marker="x",
            linestyle="none",
            markersize=6,
            label=f"Fewer than {MIN_SUCCESSFUL_SEEDS} seeds with η ≤ {EPSILON:g}",
        )
    )
    return handles


def _plot_method_points(ax, points: pd.DataFrame, method: str, value_column: str) -> None:
    """Draw a method curve, replacing its marker when too few seeds certify."""

    color = METHOD_COLORS[method]
    zorder = 1 if method == "uniform" else 2
    ax.plot(
        points["K"],
        points[value_column],
        color=color,
        linestyle=METHOD_LINESTYLES[method],
        linewidth=1.8,
        zorder=zorder,
    )
    low_success = points["successes"].lt(MIN_SUCCESSFUL_SEEDS)
    ordinary = points.loc[~low_success]
    flagged = points.loc[low_success]
    ax.scatter(
        ordinary["K"], ordinary[value_column], color=color, marker=METHOD_MARKERS[method], s=22, zorder=zorder + 1
    )
    ax.scatter(flagged["K"], flagged[value_column], color=color, marker="x", s=48, linewidths=1.8, zorder=zorder + 2)


def plot_runtime(long: pd.DataFrame, output_path: Path) -> None:
    """Plot attempt runtimes and uniform certification times with IQR bands."""

    summary = _summarize_certified_runs(long.loc[long["method"].isin(METHODS)])
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True)
    _style_axes(axes)

    for row, risk in enumerate(RISK_GRID):
        for col, n in enumerate(N_GRID):
            ax = axes[row, col]
            panel = summary.loc[summary["risk"].eq(risk) & summary["n"].eq(n)]
            for method in METHODS:
                points = panel.loc[panel["method"].eq(method)].sort_values("K")
                if points.empty:
                    continue
                color = METHOD_COLORS[method]
                _plot_method_points(ax, points, method, "capped_time_median")
                ax.fill_between(
                    points["K"],
                    points["capped_time_q25"],
                    points["capped_time_q75"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

            if risk == "cvar":
                ax.axhline(CVaR_CAP_SECONDS, color="#555555", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_xticks(K_GRID)
            ax.set_title(f"{risk.upper()} · n={n}", fontsize=10)
            if row == len(RISK_GRID) - 1:
                ax.set_xlabel("Samples K", fontsize=9)
            if col == 0:
                ax.set_ylabel("Median runtime (seconds)", fontsize=9)

    handles = _method_legend(METHODS)
    handles.append(Line2D([0], [0], color="#555555", linestyle="--", label="CVaR 24-hour cap"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Scalability runtime across 20 matched random games per cell", fontsize=14)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_certificate_rate(long: pd.DataFrame, output_path: Path) -> None:
    """Plot the share of runs with a finite exact-regret certificate at eta <= 1e-2."""

    summary = _summarize_certified_runs(long.loc[long["method"].isin(METHODS)])
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True)
    _style_axes(axes)

    for row, risk in enumerate(RISK_GRID):
        for col, n in enumerate(N_GRID):
            ax = axes[row, col]
            panel = summary.loc[summary["risk"].eq(risk) & summary["n"].eq(n)]
            for method in CERTIFICATE_PLOT_ORDER:
                points = panel.loc[panel["method"].eq(method)].sort_values("K")
                if points.empty:
                    continue
                _plot_method_points(ax, points, method, "success_rate")

            ax.set_xticks(K_GRID)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_title(f"{risk.upper()} · n={n}", fontsize=10)
            if row == len(RISK_GRID) - 1:
                ax.set_xlabel("Samples K", fontsize=9)
            if col == 0:
                ax.set_ylabel(r"Share with exact-regret $\eta \leq 10^{-2}$", fontsize=9)

    fig.legend(
        handles=_method_legend(METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Comparable equilibrium-certificate rate", fontsize=14)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long = load_scalability_data()
    plot_runtime(long, OUTPUT_DIR / "scalability_runtime.png")
    plot_certificate_rate(long, OUTPUT_DIR / "scalability_certificate_rate.png")
    print(f"Wrote README figures to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
