# QPTAS and stochastic FO with heterogeneous payoff populations

Run `run_population_qptas_fo_remote.sh` for a separate campaign comparing sampled
QPTAS, screened QPTAS, full-batch stochastic FO, and mini-batch stochastic FO. It uses the existing
method scheduler, locks, timeouts, and resume collector.

| Setting | Default |
| --- | --- |
| Risks | MSD and CVaR |
| Actions per player | 50 |
| Payoff samples K | 500; 1,000; 2,000; 4,000 |
| Game seeds per risk / K / n | 20 |
| Total method jobs | 640: 160 games, each with four methods |
| Success | Full-sample exact-regret certificate eta <= 0.01 |
| FO iterations | Up to 2,000 per start |
| FO starts | Uniform, then up to four random starts only after failure |
| FO certification | At initialization, every 100 iterations, and at the iteration limit |
| FO stagnation stopping | 500-iteration window; relative tolerance 0.005; absolute tolerance 1e-5 |
| FO mini-batch size | ceil(sqrt(K)): 23, 32, 45, 64 |
| FO entropy kappa / smoothing tau | 0.1 / 0.02 |
| FO step size / decay / logit bound | 1.0 / 0.5 / 20 |
| QPTAS kappa | ceil(sqrt(50)) = 8 |
| QPTAS candidate budget | Up to 1,000 distinct random joint kappa-uniform profiles |
| Screened QPTAS candidate budget | Up to 1,000 distinct joint (x,q) pairs |
| Screened QPTAS tau | min(K, max(5, ceil(sqrt(K)))): 23, 32, 45, 64 |
| Screened QPTAS epsilon_scr | 2 * epsilon / 3 = 0.006666… |
| Gamma / CVaR alpha | 0.5 / 0.5 |
| Scenario probabilities | p[k] = 1/K |
| Workers / wall-clock cap | 8 / 24 hours per method job, including all starts |
| Seed base | 123 |

## Payoff construction

For **each player separately**, assign every cell (i, j) one of these five
populations with probability 1/5 each. Assignments are independent across cells
and players. The assignment stays fixed across all K scenarios; only the payoff
draws vary across scenarios.

| Population ID | Distribution | Mean |
| --- | --- | --- |
| 0 | Beta(1, 4) | 0.2 |
| 1 | Beta(2, 3) | 0.4 |
| 2 | Beta(3, 2) | 0.6 |
| 3 | Beta(4, 1) | 0.8 |
| 4 | Uniform[0, 1] | 0.5 |

All payoffs are in [0, 1]. The four Beta means are evenly spaced within that
interval. Assignments are random rather than forced to have exactly equal
population counts. Conditional on the assignments, samples are independent
across cells, players, and scenarios.

`simulate_population_payoffs(K, n, seed)` in `compare_scalability_approaches.py`
returns `(A, B, p, population_ids)`, with payoff shapes `(K, n, n)` and map shape
`(2, n, n)`. The versioned model is `cell_beta_uniform_v1`. It uses NumPy
`SeedSequence(seed).spawn(11)`: child 0 generates the maps, and children 1–10
generate each player's five population samples. For the same seed and n,
increasing K preserves both maps and the previous sample prefix.

The runner retains the existing game seed formula:

```text
seed = 123 + 1_000_000 * risk_index + 10_000 * K + 100 * n + rep
risk_index: MSD=0, CVaR=1; rep=0,...,19
```

All four methods receive the same complete game within a risk/K/n/replicate
cell, even though they run in separate processes. Different risk or K cells use
different seeds and hence different maps; this campaign does not use nested
samples across its K grid. The generator's prefix property is available when
explicitly reusing a seed. FO's uniform **initial strategy** is still retained;
the payoff generator is the new heterogeneous population model.

## Run remotely

From the updated repository on the remote machine, activate its Python
environment and install `python -m pip install -e ".[stochastic]"`. JAX and
SciPy/HiGHS are used; these four methods need no PATHAMPL or IPOPT. The shell
runner requires GNU `timeout`, as on Ubuntu.

```bash
# Preview the default 640 jobs without solving them.
DRY_RUN=1 bash experiments/run_population_qptas_fo_remote.sh

# Launch; use the same command to resume a stopped campaign.
nohup bash experiments/run_population_qptas_fo_remote.sh >> population_runner_v2.log 2>&1 &

# Monitor the launcher and per-method completion messages.
tail -f population_runner_v2.log
```

Default results go to
`experiments/results/remote/population_qptas_fo/beta_uniform_v2/`.
Use a separate directory for a small smoke test:

```bash
RESULT_DIR=experiments/scratch/population_smoke \
K_GRID=10 N_GRID=5 REPS=1 WORKERS=1 MAX_ITER=10 CERTIFY_EVERY=1 MAX_CANDIDATES=5 \
bash experiments/run_population_qptas_fo_remote.sh
```

Grid settings, repetitions, worker count, iteration budget, restart budget,
certification frequency, stagnation window/tolerances, candidate budget, epsilon,
gamma, alpha, seed base, cap, and result directory can be overridden using the script's environment
variables. `STOCHASTIC_N_RANDOM_STARTS=4` means four additional starts (five
total). `EPSILON` controls all methods' regret thresholds.
`CERTIFY_EVERY=100` checks full-sample regret at initialization, every 100
updates, and at the iteration limit, including in mini-batch mode. Each start
gets a fresh iteration budget. The first successful certificate ends FO
immediately and skips all remaining starts. An iterate that meets the target
between checkpoints can be missed. Set `CERTIFY_EVERY=1` to check every update.

Stagnation uses `STAGNATION_WINDOW=500`, `STAGNATION_RTOL=0.005`, and
`STAGNATION_ATOL=0.00001`. Let b[t] be the lowest regret observed at certification
checkpoints within the current start. After at least 500 iterations, a start
stops at a certification checkpoint if:

```text
b[t - 500] - b[t] < max(0.00001, 0.005 * abs(b[t - 500]))
```

This is less than the required absolute or 0.5% relative improvement over the
last 500 iterations. Success is checked first. Stagnation permits the next
random start, and its iteration/stagnation counters start afresh. Set
`STAGNATION_WINDOW=0` to disable this rule. If all starts fail, FO returns its
lowest-regret certified profile with `success=False`. Both regret and stagnation
decisions are made every 100 iterations with these defaults.

Choose a new `VERSION` or `RESULT_DIR` whenever changing numerical settings. The runner
rejects a conflicting saved configuration, and the collector rejects completed
shards from a different payoff model. Legacy lookups stay inside the new
campaign's empty `legacy/` directory, so old uniform-game results are not imported.

The population runner permits **additions only** to `K_GRID` and `METHODS` within
the same version. When upgrading an existing v2 directory to add K=500 and
`qptas_screened`, it archives the previous `run_config.env` as
`run_config.before_extension_<timestamp>.env` and retains completed method shards.
All other saved settings must match exactly. The planner adds the new tasks and
retains unfinished old tasks. This also applies to a dry-run preview. Stop any
previous runner and its workers before extending the campaign and relaunching.

### Switching from every-iteration checking

`beta_uniform_v1` used `CERTIFY_EVERY=1`. The new default `beta_uniform_v2`
uses 100 and keeps the same games, seeds, epsilon, restart budget, and stagnation
parameters. Stop the old launcher and its worker processes before launching v2;
editing the script does not change running Python processes. Verify the worker
processes have stopped before starting a new worker pool. Keep the v1 directory
and use the new log filename above to preserve its results and logs.

The new version initially schedules the full grid from scratch. It does not recover
in-progress FO iterates or automatically import completed v1 runs. Analyze v1
and v2 FO timings separately because the certification schedules differ. QPTAS
does not change with this setting, so its completed v1 results remain valid.

## Screened QPTAS

The method ID is `qptas_screened`. Each draw samples a kappa-uniform strategy
for each player and a separate tau-uniform scenario distribution q for each
player. Sampling is uniform over integer count vectors, without replacement
over complete (x,q) pairs. The same x can appear again with a different q.
The default n=50 gives kappa=8 for both players. These are heuristic sparsity
choices, not the sufficient bounds from the small-support theorem.

Current robust payoffs use all K samples and the empirical probabilities p.
Screening compares every pure-action expected payoff under that player's q to
its current robust payoff plus epsilon_scr. q is used only for this screen and
is not required to belong to the risk ambiguity set. Only pairs that pass all
players' screens reach the best-response LPs. Those LPs use all K samples, p,
and arbitrary mixed deviations, rejecting immediately if a regret exceeds
epsilon. The method returns at the first successful full-regret certificate.

A failed screen can discard an equilibrium profile. No success after 1,000
pairs means only that the sampling budget failed; it is not a nonexistence
certificate. Certificates use the package's floating-point LP conventions.

The public functions also support more than two players and unequal action sizes:

```python
from cumg import solve_cvar_qptas_screened, solve_msd_qptas_screened

msd_result = solve_msd_qptas_screened(
    [A, B], p, gamma=0.5, epsilon=0.01, max_candidates=1000, seed=123,
)
cvar_result = solve_cvar_qptas_screened(
    [A, B], p, gamma=0.5, alpha=0.5, epsilon=0.01, max_candidates=1000, seed=123,
)
```

Both select kappa_i, tau, and epsilon_scr automatically. They accept overrides;
kappa may be a scalar or a tuple with one denominator per player. Success
results include `strategies`, `certificate`, and `screening_distributions`.
Failure results have no strategies or certificate. Original `qptas` continues
to sample strategy profiles directly without screening.

## Saved results

- `run_config.env`: grid, methods, payoff model, solver budgets, and thresholds.
- `method_shards/`: one CSV per completed method run; JSON markers for errors
  and timeouts, with per-attempt logs under `logs/`.
- `capped_method_results.csv`: collected at startup and when the runner ends.
- `payoff_populations`, `payoff_population_ids`, `payoff_numpy_version`: JSON
  distribution catalogue, JSON maps in player/row/column order, and sampling
  library version in each completed result row.
- FO `start_summaries`: initial profiles, seeds, regrets, times, iterations, and
  stop reasons; `selected_start` identifies the returned profile. Work totals
  include all attempted starts.
- `stochastic_stagnation_window`, `stochastic_stagnation_rtol`, and
  `stochastic_stagnation_atol`: the stopping settings recorded in completed rows.
- QPTAS `profiles_checked`, `best_response_solves`, `termination_reason`: the
  work performed and why the sampled search stopped.
- Screened QPTAS additionally records `total_pairs`, `screen_passes`, and
  `screen_rejections`. Its `profiles_checked` counts pairs, including rejected
  screens, and equals `screen_passes + screen_rejections`. `total_profiles`
  counts the x grid only; `total_pairs` includes all players' q grids.

Method runtimes exclude payoff generation and include full-sample regret
certification. The external wall-clock cap also covers process startup and
payoff generation. For success counts, require `status=completed`, a finite
`eta`, and `eta <= 0.01`. A QPTAS exhausted search is a completed failure and
returns no certificate; its time measures budget exhaustion. FO may return a
finite certificate above the threshold on failure. Neither a low residual nor
reaching the iteration limit counts as success.

Sampling 1,000 profiles with kappa=8 is a budgeted QPTAS search and does not
provide the exhaustive search's existence guarantee. All certificates and
best-response LPs use all K payoff samples for both players.

For an up-to-date aggregate while workers are running, invoke the collector
directly (do not relaunch just to inspect progress):

```bash
python experiments/capped_scalability_resume.py collect \
  --legacy-dir experiments/results/remote/population_qptas_fo/beta_uniform_v2/legacy \
  --result-dir experiments/results/remote/population_qptas_fo/beta_uniform_v2 \
  --risk msd cvar --K 500 1000 2000 4000 --n 50 --reps 20 \
  --methods qptas qptas_screened stochastic_full_batch stochastic_minibatch \
  --payoff-model cell_beta_uniform_v1 --stochastic-n-random-starts 4 \
  --output experiments/results/remote/population_qptas_fo/beta_uniform_v2/capped_method_results.csv
```
