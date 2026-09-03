"""Reusable data preparation for the scalability-analysis notebook.

The functions in this module keep the statistical meaning of capped runs
explicit: timeout runtimes are right-censored at the configured cap, success
rates use every replicate, and eta summaries use successful completed runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SCALABILITY_METHODS = (
    "mcp",
    "screened_dual",
    "action_dual",
    "restricted_mcp",
    "stochastic_full_batch",
    "stochastic_minibatch",
)
SCALABILITY_ID_COLUMNS = ("risk", "K", "n", "rep", "seed")
SCALABILITY_METRICS = ("success", "time_s", "eta", "status", "censored")
TERMINAL_STATUSES = frozenset({"completed", "timeout"})
TRAJECTORY_COLUMNS = (
    "risk",
    "K",
    "n",
    "method",
    "seed",
    "entropy_kappa",
    "smoothing_tau",
    "step_size",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_csv_shards(
    directory: str | Path,
    pattern: str,
    *,
    shard_suffix: str | None = None,
) -> pd.DataFrame:
    """Read and concatenate all non-empty CSV shards matching ``pattern``.

    When ``shard_suffix`` is supplied, the filename without that suffix is
    recorded in a ``shard`` column. This makes independent optimization
    trajectories explicit in downstream grouping.
    """

    root = Path(directory)
    paths = [path for path in sorted(root.glob(pattern)) if path.stat().st_size > 0]
    if not paths:
        raise FileNotFoundError(f"No non-empty CSV files match {root / pattern}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if shard_suffix is not None:
            if not path.name.endswith(shard_suffix):
                raise ValueError(f"{path.name!r} does not end with {shard_suffix!r}")
            frame["shard"] = path.name[: -len(shard_suffix)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_stochastic_shards(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load paired stochastic-search summary and history shards."""

    summary = load_csv_shards(directory, "*_summary.csv", shard_suffix="_summary.csv")
    history = load_csv_shards(directory, "*_history.csv", shard_suffix="_history.csv")
    return summary, history


def capped_results_to_wide(
    capped: pd.DataFrame,
    *,
    expected_methods: Sequence[str] = SCALABILITY_METHODS,
) -> pd.DataFrame:
    """Convert terminal capped method rows to the legacy one-row-per-replicate form.

    ``gamma`` and ``alpha`` are deliberately not pivot keys. Timeout-marker rows
    can lack those metadata values, so including them would split one replicate
    into multiple rows.
    """

    required = (*SCALABILITY_ID_COLUMNS, "method", *SCALABILITY_METRICS)
    _require_columns(capped, required, label="capped results")
    work = capped.copy()

    duplicate = work.duplicated([*SCALABILITY_ID_COLUMNS, "method"], keep=False)
    if duplicate.any():
        keys = work.loc[duplicate, [*SCALABILITY_ID_COLUMNS, "method"]].head().to_dict("records")
        raise ValueError(f"capped results contain duplicate method rows, for example: {keys}")

    statuses = set(work["status"].dropna().astype(str))
    unexpected_statuses = sorted(statuses - TERMINAL_STATUSES)
    if unexpected_statuses:
        raise ValueError(f"capped results contain non-terminal statuses: {unexpected_statuses}")

    expected = set(expected_methods)
    unexpected_methods = sorted(set(work["method"].dropna()) - expected)
    if unexpected_methods:
        raise ValueError(f"capped results contain unexpected methods: {unexpected_methods}")

    method_counts = work.groupby(list(SCALABILITY_ID_COLUMNS), dropna=False)["method"].nunique()
    incomplete = method_counts[method_counts.ne(len(expected))]
    if not incomplete.empty:
        raise ValueError(f"{len(incomplete)} capped replicates do not contain every expected method")

    timeout = work["status"].eq("timeout")
    if work.loc[timeout, "success"].eq(True).any():
        raise ValueError("timeout rows cannot be successful")
    if work.loc[timeout, "censored"].ne(True).any():
        raise ValueError("timeout rows must be marked censored")
    if work.loc[~timeout, "censored"].eq(True).any():
        raise ValueError("completed rows cannot be marked censored")
    if "time_limit_s" in work.columns and timeout.any():
        time_s = pd.to_numeric(work.loc[timeout, "time_s"], errors="coerce")
        limit_s = pd.to_numeric(work.loc[timeout, "time_limit_s"], errors="coerce")
        if not np.isclose(time_s, limit_s, equal_nan=False).all():
            raise ValueError("timeout time_s must equal time_limit_s")

    shared_columns = [column for column in ("gamma", "alpha") if column in work.columns]
    if shared_columns:
        shared = work.groupby(list(SCALABILITY_ID_COLUMNS), dropna=False)[shared_columns].first().reset_index()
    else:
        shared = work.loc[:, list(SCALABILITY_ID_COLUMNS)].drop_duplicates()

    wide = work.pivot(
        index=list(SCALABILITY_ID_COLUMNS),
        columns="method",
        values=list(SCALABILITY_METRICS),
    )
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index().merge(shared, on=list(SCALABILITY_ID_COLUMNS), validate="one_to_one")
    wide["methods"] = ",".join(expected_methods)
    return wide.sort_values(list(SCALABILITY_ID_COLUMNS)).reset_index(drop=True)


def merge_higher_epsilon_screened_dual(
    wide: pd.DataFrame,
    higher_epsilon: pd.DataFrame,
    *,
    epsilon_scr: float = 0.01,
) -> pd.DataFrame:
    """Attach the higher-epsilon screened-dual variant to the main wide table."""

    join_columns = ["risk", "K", "n", "seed", "gamma", "alpha"]
    source_columns = [
        "screened_dual_success",
        "screened_dual_time_s",
        "screened_dual_eta",
    ]
    _require_columns(wide, join_columns, label="wide scalability data")
    _require_columns(
        higher_epsilon,
        [*join_columns, "epsilon_scr", *source_columns],
        label="higher-epsilon data",
    )

    epsilon = pd.to_numeric(higher_epsilon["epsilon_scr"], errors="coerce")
    if not np.isclose(epsilon, epsilon_scr, equal_nan=False).all():
        raise ValueError(f"higher-epsilon data must have epsilon_scr={epsilon_scr}")
    if higher_epsilon.duplicated(join_columns).any():
        raise ValueError("higher-epsilon data contain duplicate experiment keys")

    known_keys = wide.loc[:, join_columns].drop_duplicates()
    unmatched = higher_epsilon.loc[:, join_columns].merge(
        known_keys,
        on=join_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if unmatched["_merge"].eq("left_only").any():
        raise ValueError("higher-epsilon data contain keys outside the main scalability grid")

    rename = {
        "screened_dual_success": "higher_eps_screened_dual_success",
        "screened_dual_time_s": "higher_eps_screened_dual_time_s",
        "screened_dual_eta": "higher_eps_screened_dual_eta",
    }
    variant = higher_epsilon.loc[:, [*join_columns, *source_columns]].rename(columns=rename)
    merged = wide.merge(variant, on=join_columns, how="left", validate="one_to_one")
    metric_columns = list(rename.values())
    present = merged[metric_columns].notna().any(axis=1)
    merged["higher_eps_screened_dual_status"] = pd.NA
    merged.loc[present, "higher_eps_screened_dual_status"] = "completed"
    merged["higher_eps_screened_dual_censored"] = pd.NA
    merged.loc[present, "higher_eps_screened_dual_censored"] = False
    return merged


def wide_scalability_to_long(
    wide: pd.DataFrame,
    *,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Reshape method-prefixed scalability columns into one row per method run."""

    identity_columns = [
        column for column in ("risk", "K", "n", "rep", "seed", "gamma", "alpha") if column in wide.columns
    ]
    _require_columns(wide, ["risk", "K", "n", "seed"], label="wide scalability data")

    frames: list[pd.DataFrame] = []
    for method in methods:
        metric_columns = [f"{method}_{metric}" for metric in ("success", "time_s", "eta")]
        _require_columns(wide, metric_columns, label=f"wide scalability data for {method}")
        frame = wide.loc[:, [*identity_columns, *metric_columns]].copy()
        frame = frame.rename(columns={column: column.removeprefix(f"{method}_") for column in metric_columns})

        status_column = f"{method}_status"
        censored_column = f"{method}_censored"
        frame["status"] = wide[status_column] if status_column in wide.columns else "completed"
        frame["censored"] = wide[censored_column] if censored_column in wide.columns else False
        frame["method"] = method
        frame = frame.loc[frame[["success", "time_s", "eta"]].notna().any(axis=1)]
        frame["status"] = frame["status"].astype("string").fillna("completed")
        frame["censored"] = frame["censored"].astype("boolean").fillna(False)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def append_uniform_baseline(long: pd.DataFrame, uniform: pd.DataFrame) -> pd.DataFrame:
    """Append only uniform-baseline rows matching the analyzed experiment grid."""

    keys = ["risk", "K", "n", "seed"]
    uniform_columns = ["uniform_success", "uniform_time_s", "uniform_eta"]
    _require_columns(long, keys, label="long scalability data")
    _require_columns(uniform, [*keys, *uniform_columns], label="uniform baseline")
    if uniform.duplicated(keys).any():
        raise ValueError("uniform baseline contains duplicate experiment keys")

    reference = long.loc[:, keys].drop_duplicates()
    matched = uniform.merge(reference, on=keys, how="inner", validate="one_to_one")
    identity_columns = [
        column for column in ("risk", "K", "n", "rep", "seed", "gamma", "alpha") if column in matched.columns
    ]
    baseline = matched.loc[:, [*identity_columns, *uniform_columns]].rename(
        columns={
            "uniform_success": "success",
            "uniform_time_s": "time_s",
            "uniform_eta": "eta",
        }
    )
    baseline["status"] = "completed"
    baseline["censored"] = False
    baseline["method"] = "uniform"
    return pd.concat([long, baseline], ignore_index=True, sort=False)


def summarize_scalability(long: pd.DataFrame) -> pd.DataFrame:
    """Summarize capped runtime, timeout rate, success rate, and valid eta.

    Runtime summaries include every attempt and therefore treat a timeout's
    ``time_s`` as capped wall time. Eta summaries use only successful rows with
    ``status == 'completed'``.
    """

    group_columns = ["risk", "n", "K", "method"]
    required = [*group_columns, "status", "success", "time_s", "eta"]
    _require_columns(long, required, label="long scalability data")
    work = long.copy()
    work["success"] = work["success"].astype("boolean").fillna(False).astype(bool)
    work["is_timeout"] = work["status"].eq("timeout")

    grouped = work.groupby(group_columns, dropna=False, sort=True)
    summary = grouped.agg(
        reps=("method", "size"),
        successes=("success", "sum"),
        timeouts=("is_timeout", "sum"),
        capped_time_median=("time_s", "median"),
        capped_time_q25=("time_s", lambda values: values.quantile(0.25)),
        capped_time_q75=("time_s", lambda values: values.quantile(0.75)),
    ).reset_index()
    summary["success_rate"] = summary["successes"] / summary["reps"]
    summary["timeout_rate"] = summary["timeouts"] / summary["reps"]

    valid_eta = work.loc[work["status"].eq("completed") & work["success"] & work["eta"].notna()]
    eta_summary = (
        valid_eta.groupby(group_columns, dropna=False, sort=True)
        .agg(
            eta_reps=("eta", "size"),
            eta_median=("eta", "median"),
            eta_q25=("eta", lambda values: values.quantile(0.25)),
            eta_q75=("eta", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    summary = summary.merge(eta_summary, on=group_columns, how="left", validate="one_to_one")
    summary["eta_reps"] = summary["eta_reps"].fillna(0).astype(int)
    return summary


def prepare_best_eta_history(history: pd.DataFrame) -> pd.DataFrame:
    """Compute the running best eta separately for every search trajectory."""

    required = [*TRAJECTORY_COLUMNS, "iteration", "eta"]
    _require_columns(history, required, label="stochastic history")
    work = history.copy()
    trajectory_columns = [*TRAJECTORY_COLUMNS]
    if "shard" in work.columns:
        trajectory_columns.insert(0, "shard")
    order = [*trajectory_columns, "iteration"]
    work = work.sort_values(order).reset_index(drop=True)
    work["best_eta"] = work.groupby(trajectory_columns, dropna=False)["eta"].cummin()
    return work


def prepare_eta_improvement(history: pd.DataFrame) -> pd.DataFrame:
    """Add percentage improvement relative to each trajectory's initial eta."""

    work = prepare_best_eta_history(history)
    trajectory_columns = [*TRAJECTORY_COLUMNS]
    if "shard" in work.columns:
        trajectory_columns.insert(0, "shard")
    grouped = work.groupby(trajectory_columns, dropna=False)["eta"]
    work["initial_eta"] = grouped.transform("first")
    denominator = work["initial_eta"].where(work["initial_eta"].ne(0))
    work["eta_improvement_pct"] = 100 * (work["initial_eta"] - work["best_eta"]) / denominator
    return work


def stochastic_summary_to_long(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] = ("full_batch", "minibatch"),
) -> pd.DataFrame:
    """Reshape stochastic-search summary metrics to one row per method."""

    metric_suffixes = {
        "eta": "eta",
        "success": "success",
        "time_s": "time_s",
        "iterations": "iterations",
        "stages": "stages_completed",
        "selected_stage": "selected_stage",
    }
    frames: list[pd.DataFrame] = []
    prefixes = tuple(f"{method}_" for method in methods)
    base_columns = [column for column in summary.columns if not column.startswith(prefixes)]
    for method in methods:
        columns = [f"{method}_{suffix}" for suffix in metric_suffixes.values()]
        _require_columns(summary, columns, label=f"stochastic summary for {method}")
        frame = summary.loc[:, [*base_columns, *columns]].copy()
        frame = frame.rename(columns={f"{method}_{suffix}": metric for metric, suffix in metric_suffixes.items()})
        frame["method"] = method
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_v3_history(history: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attach each shard's initial step and compute a shard-local running best eta."""

    _require_columns(history, ["shard", "method", "iteration", "eta"], label="v3 history")
    _require_columns(summary, ["shard", "step_size"], label="v3 summary")
    initial_steps = summary.loc[:, ["shard", "step_size"]].drop_duplicates()
    if initial_steps.duplicated("shard").any():
        raise ValueError("v3 summary has multiple initial step sizes for a shard")
    initial_steps = initial_steps.rename(columns={"step_size": "initial_step"})

    work = history.drop(columns=["initial_step", "best_eta_so_far"], errors="ignore").copy()
    work = work.merge(initial_steps, on="shard", how="left", validate="many_to_one")
    if work["initial_step"].isna().any():
        raise ValueError("v3 history contains shards missing from the summary")
    work = work.sort_values(["shard", "method", "iteration"]).reset_index(drop=True)
    work["best_eta_so_far"] = work.groupby(["shard", "method"], dropna=False)["eta"].cummin()
    return work
