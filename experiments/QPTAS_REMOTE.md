# Remote sampled QPTAS experiments

`run_qptas_scalability_remote.sh` runs **both MSD and CVaR** using the same
two-player random-game grid and seeds as `run_cvar_scalability_capped_resume.sh`.
It reuses that runner's worker pool, per-job timeouts, logs, locks, and resume
protocol. QPTAS is also available as `--methods qptas` in
`compare_scalability_approaches.py`; the other methods' defaults are unchanged.

| Setting | Default |
| --- | --- |
| Risks | MSD and CVaR |
| Pure actions per player, `n` | 5, 10, 20, 50 |
| Payoff samples, `K` | 5, 10, 30, 100, 250, 500 |
| Repetitions per risk / K / n | 20 (960 jobs in total) |
| Kappa | `ceil(sqrt(n))`: 3, 4, 5, 8 |
| Candidate budget | Up to 1,000 distinct joint kappa-uniform profiles |
| Epsilon | 0.01 |
| Gamma / CVaR alpha | 0.5 / 0.5 |
| Workers | 8 |
| Time cap | 86,400 seconds (24 hours) per job |
| Seed base | 123 |
| Payoffs / probabilities | Independent Uniform[0, 1] draws / uniform over K samples |

Here the budget counts **joint profiles**, not support sets. Sampling is uniform
without replacement over the joint kappa-uniform grid. Every regret check solves
a full mixed best-response LP using **all K samples**. A run stops immediately
when all players' regrets are at most epsilon. The game seed and candidate seed
are both `123 + 1_000_000 * risk_index + 10_000 * K + 100 * n + rep`, where
MSD has index 0, CVaR has index 1, and `rep` starts at 0. This matches games
across methods within each risk; the two risks use different game seeds.

## Run on the remote machine

Copy or pull the updated checkout onto the remote machine, activate its Python
environment (Python 3.10+), and install the package with `python -m pip install -e .`.
GNU `timeout` must be on PATH; it is normally provided by Ubuntu's coreutils.
The QPTAS jobs use SciPy/HiGHS and do not require PATHAMPL, IPOPT, or JAX.
Run these commands from the repository root:

```bash
# Preview the 960 jobs; write the configuration and pending manifest.
DRY_RUN=1 bash experiments/run_qptas_scalability_remote.sh

# Run in the background. Repeating this command resumes the same campaign.
nohup bash experiments/run_qptas_scalability_remote.sh > qptas_runner.log 2>&1 &
```

An optional small check uses a separate results directory:

```bash
RESULT_DIR=experiments/results/remote/qptas_smoke \
K_GRID="5" N_GRID="5" REPS=1 WORKERS=2 MAX_CANDIDATES=10 \
bash experiments/run_qptas_scalability_remote.sh
```

Settings can be overridden through environment variables shown in the script:
`RISK_GRID`, `K_GRID`, `N_GRID`, `REPS`, `WORKERS`, `MAX_CANDIDATES`, `EPSILON`,
`GAMMA`, `ALPHA`, `SEED_BASE`, `METHOD_TIME_LIMIT_SECONDS`, `PYTHON_BIN`, and
`RESULT_DIR`. Use a new `RESULT_DIR` (or `VERSION`) when changing campaign
settings; the runner refuses to mix configurations in an existing directory.

## Results and resuming

The default output directory is
`experiments/results/remote/qptas_scalability/sampled_1000_v1/`:

- `run_config.env` records the configuration.
- `method_shards/qptas/` holds one CSV per finished run, or a JSON status marker
  for an error or timeout. Failed attempts may also leave a diagnostic partial CSV.
- `logs/` contains per-attempt logs.
- `capped_method_results.csv` collects results at startup and when the runner finishes.
- `pending_methods.tsv` lists the jobs selected at startup.

The collected CSV includes elapsed solver time, success, regret certificates,
profiles checked, total grid size, number of best-response LPs, sampling seed,
and `termination_reason`. `support_kappa` is the QPTAS probability denominator;
scenario-support and stochastic-method fields are unused by QPTAS.

`epsilon_reached` means a profile passed all regret checks. `sample_exhausted`
means the sampled budget found none; `grid_exhausted` means the entire selected
grid found none. Exhaustion is a completed search with `success=False`, and its
runtime is time to exhaust the search rather than time to find an equilibrium.
The arbitrary kappa and finite sample budget do not give the exhaustive QPTAS
existence guarantee. Certificates and profiles are only returned on success.

`status=timeout` is censored at the wall-clock cap. `status=error` records a
failed subprocess; inspect its log. Resume skips completed searches and timeouts.
Set `RETRY_ERRORS=1` on the same invocation to retry errors, preserving earlier
attempt logs. A live worker owns a per-job lock; after a forced shutdown, remove
any stale lock only after confirming that its worker is no longer running.

## Download and count successes

After the campaign finishes, run this in your **local checkout**, using your
server's SSH address or host alias:

```bash
bash experiments/sync_qptas_results.sh root@YOUR_SERVER_IP
```

The script downloads the campaign from `/root/cumg` into the matching local
`experiments/results/remote/qptas_scalability/sampled_1000_v1/` directory, then
prints successes, total runs, and success rates for MSD, CVaR, and both combined.
It counts rows with `status=completed` and `success=True`; the denominator includes
all QPTAS rows, and status counts are printed separately. It uses Python 3's
standard library and rsync, preserves local-only files, and skips partial CSVs
and worker locks. A failed transfer stops the script before reporting counts.
Override `VERSION`, `REMOTE_REPO`, `LOCAL_RESULT_DIR`, or `PYTHON_BIN` as needed.
