# Experiments

Run commands from the repository root with `cumg` and its development/notebook
extras installed. Numerical solvers live in `src/cumg`; this package contains
campaign configuration, scheduling, analysis, and reproducibility evidence.

## Campaign commands

```bash
python -m experiments plan --campaign population_qptas_fo/beta_uniform_v2
python -m experiments status --campaign population_qptas_fo/beta_uniform_v2
python -m experiments run --campaign population_qptas_fo/beta_uniform_v2
python -m experiments collect --campaign population_qptas_fo/beta_uniform_v2
```

`plan` and `status` are read-only. Completion comes from result shards and
status markers, not the original `pending_methods.tsv`. The completed v2
campaign reports 640 completed jobs, 135 recorded certificates at eta <= 0.01,
and zero pending jobs. `run` then launches no processes. `collect` preserves an
existing aggregate byte-for-byte when it already matches the shards.

Presets are explicit JSON files in [`configs/`](configs/). Capped runs retain
per-method deadlines, locks, error/timeout markers, and atomic CSV publication.
Uncapped campaigns retain their per-replicate workflow. Only error markers can
be retried with `--retry-errors`; timeouts remain censored observations.

For a new numerical configuration, copy a preset to a new study/run ID and
change both `campaign` and `result_dir` (and `LEGACY_RESULT_DIR` for an independent
campaign). Invoke it with `--config path/to/preset.json --campaign study/run`.
The saved numerical configuration must match on resume. Known historical result
paths are compared through the explicit relocation map; other changes fail.
Do not use a completed campaign as fresh calibration or validation data.

```bash
nohup python -m experiments run --campaign STUDY/RUN >> campaign.log 2>&1 &
tail -f campaign.log
python -m experiments status --campaign STUDY/RUN
```

Use a new result directory for a smoke test: set both grids and repetitions to
small values in its copied preset. `python -m pytest tests/test_qptas_experiments.py`
exercises an isolated tiny campaign and its resume behavior.

### Uniform-payoff grid with calibrated FO settings

[`uniform_qptas_fo/calibrated_v1`](configs/uniform_qptas_fo/calibrated_v1.json)
transfers the population `beta_uniform_v2` FO settings to the original
`K = 5, 10, 30, 100, 250, 500` and `n = 5, 10, 20, 50` grid, with independent
Uniform[0,1] payoffs and the original 20 game seeds per risk/cell. It runs
screened QPTAS, stochastic full batch, and stochastic minibatch: 2,880 method
attempts across MSD and CVaR, with 8 workers and a 24-hour cap per attempt.

```bash
python -m experiments plan --campaign uniform_qptas_fo/calibrated_v1
python -m experiments run --campaign uniform_qptas_fo/calibrated_v1 --workers 8
python -m experiments status --campaign uniform_qptas_fo/calibrated_v1
python -m experiments collect --campaign uniform_qptas_fo/calibrated_v1
```

FO retains entropy 0.01, smoothing 0.002, steps 1000 (MSD) / 500 (CVaR),
decay 0.5, logit bound 20, JIT, 2,000 updates per start, up to four random
restarts, certification every 100 updates, and the population stagnation rule
(500 updates, rtol 0.005, atol 1e-5). These settings are transferred, not retuned
on uniform games. Both FO and QPTAS target epsilon 0.01; screened QPTAS retains
the 1,000-candidate budget and automatic support sizes for each `(K, n)`.
Rerun the same `run` command to resume. Outputs go to
`experiments/results/uniform_qptas_fo/calibrated_v1/`; GNU `timeout` is required.

## Remote transfer

```bash
python -m experiments sync --campaign population_qptas_fo/beta_uniform_v2 \
  --remote USER@HOST --preview
python -m experiments sync --campaign population_qptas_fo/beta_uniform_v2 \
  --remote USER@HOST
```

The remote repository defaults to `/root/cumg`; override `--remote-root` as
needed. The local consolidation does not move remote files. If the remote still
uses the old layout, supply the exact source directory:

```bash
python -m experiments sync --campaign population_qptas_fo/beta_uniform_v2 \
  --remote USER@HOST \
  --remote-path /root/cumg/experiments/results/remote/population_qptas_fo/beta_uniform_v2 \
  --preview
```

Transfers exclude locks and partial files and never delete local files. A
transfer failure propagates a nonzero exit status. Sync operates on any catalog
study/run, including historical campaigns without a runnable preset. For a new
campaign not yet in the catalog, pass its `--config` to `sync` as well.

## Code and analysis

- [`common/experiment.py`](common/experiment.py): shared payoff generators,
  exact seed formula, numerical option conversions, and CSV writers.
- [`runners/compare_scalability_approaches.py`](runners/compare_scalability_approaches.py):
  general method comparison; run via `python -m experiments.runners.compare_scalability_approaches`.
- [`runners/compare_stochastic_fo.py`](runners/compare_stochastic_fo.py): separate
  FO tuning and continuation loop; run via `python -m experiments.runners.compare_stochastic_fo`.
- [`analysis/`](analysis/): both experiment notebooks, loaders, and figures.
- [`diagnostics/`](diagnostics/): regret checks, bounded replays, stability, and migration verification.
- [`legacy/`](legacy/): superseded source retained for provenance, not active launch commands.

```bash
python -m experiments.analysis.generate_readme_figures
python -m experiments.analysis.generate_readme_figures --study population
python -m experiments plan --campaign population_fo_calibration/cvar_steps_v2
python -m experiments plan --campaign population_fo_calibration/cvar_steps_v2 --phase cvar_validate_500
python -m experiments status --campaign stochastic_fo/continuation_v3
```

Calibration presets preserve the original phase arguments and distinguish
screening, development, and validation. Existing outputs are never overwritten
by calibration `run`; use a new preset/result directory to repeat a phase.
Continuation presets preserve the individual recorded shard schedules.

Population plots remain separate from the older i.i.d. uniform-payoff plots.
Both use finite recorded eta <= 0.01 for success and crosses for fewer than five
successful seeds. Runtime summarizes all attempts, including failures, and
retains the older campaigns' right-censoring at their time limit.

## Profiles and verification

New scheduled runs save versioned JSON sidecars under each shard's `profiles/`
directory, linked by `profile_artifact`. They contain returned x/y, CVaR theta,
certificates and available best responses, selected start/checkpoint, existing
checkpoint diagnostics, effective parameters, input-array hashes, source hashes,
and environment information. These are serialized after the solver timing
boundary. `capture_certificates` retains already computed checkpoints without
additional LPs or changes to update/stopping rules. Exhausted QPTAS legitimately
returns no profile; unused FO settings are null in its new records.

```bash
python -m experiments.diagnostics.regret_audit --output-dir experiments/results/audits/NEW_AUDIT
python -m experiments.diagnostics.replay --audit-dir experiments/results/audits/NEW_AUDIT
python -m experiments.diagnostics.stability --output experiments/scratch/NEW_stability.json
```

The regret audit checks available presented profiles using all samples and
fresh full mixed best responses, with independent payoff arithmetic. Agreement
uses atol=1e-8 and rtol=1e-6; any change in the exact 0.01 success classification
is reported separately. Missing original inputs/environment remain an explicit
limitation, even when regenerated-input checks agree.

For missing final profiles, the replay command freezes at most three distinct
recorded game seeds per study/run, starting with small presented cells and
including both risks where available. It never selects by success or regret.
Each seed has a 100-second process-group deadline, within 300 seconds total per
study/run. Insufficient seeds, unavailable configurations, solver failures, and
timeouts remain inconclusive. Replays do not overwrite the original data or
recertify missing original strategies.

See [`results/README.md`](results/README.md) and the audit report for current
coverage and limitations. The local consolidation plan remains ignored under
`experiments/scratch/`.
