# QPTAS and stochastic FO with heterogeneous payoff populations

Run `run_population_qptas_fo_remote.sh` for a separate campaign comparing sampled
QPTAS, full-batch stochastic FO, and mini-batch stochastic FO. It uses the existing
method scheduler, locks, timeouts, and resume collector.

| Setting | Default |
| --- | --- |
| Risks | MSD and CVaR |
| Actions per player | 50 |
| Payoff samples K | 1,000; 2,000; 4,000 |
| Game seeds per risk / K / n | 20 |
| Total method jobs | 360: 120 games, each with three methods |
| Success | Full-sample exact-regret certificate eta <= 0.01 |
| FO iterations | Up to 2,000 per start |
| FO starts | Uniform, then up to four random starts only after failure |
| FO certification | At initialization and every iteration |
| FO stagnation stopping | 500-iteration window; relative tolerance 0.005; absolute tolerance 1e-5 |
| FO mini-batch size | ceil(sqrt(K)): 32, 45, 64 |
| FO entropy kappa / smoothing tau | 0.1 / 0.02 |
| FO step size / decay / logit bound | 1.0 / 0.5 / 20 |
| QPTAS kappa | ceil(sqrt(50)) = 8 |
| QPTAS candidate budget | Up to 1,000 distinct random joint kappa-uniform profiles |
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

All three methods receive the same complete game within a risk/K/n/replicate
cell, even though they run in separate processes. Different risk or K cells use
different seeds and hence different maps; this campaign does not use nested
samples across its K grid. The generator's prefix property is available when
explicitly reusing a seed. FO's uniform **initial strategy** is still retained;
the payoff generator is the new heterogeneous population model.

## Run remotely

From the updated repository on the remote machine, activate its Python
environment and install `python -m pip install -e ".[stochastic]"`. JAX and
SciPy/HiGHS are used; these three methods need no PATHAMPL or IPOPT. The shell
runner requires GNU `timeout`, as on Ubuntu.

```bash
# Preview the default 360 jobs without solving them.
DRY_RUN=1 bash experiments/run_population_qptas_fo_remote.sh

# Launch; use the same command to resume a stopped campaign.
nohup bash experiments/run_population_qptas_fo_remote.sh > population_runner.log 2>&1 &

# Monitor the launcher and per-method completion messages.
tail -f population_runner.log
```

Default results go to
`experiments/results/remote/population_qptas_fo/beta_uniform_v1/`.
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
total). `EPSILON` controls both algorithms' regret thresholds.
`CERTIFY_EVERY=1` checks full-sample regret after every update, including in
mini-batch mode. Each start gets a fresh iteration budget. The first successful
certificate ends FO immediately and skips all remaining starts.

Stagnation uses `STAGNATION_WINDOW=500`, `STAGNATION_RTOL=0.005`, and
`STAGNATION_ATOL=0.00001`. Let b[t] be the lowest certified regret found so far
within the current start. After at least 500 iterations, a start stops if:

```text
b[t - 500] - b[t] < max(0.00001, 0.005 * abs(b[t - 500]))
```

This is less than the required absolute or 0.5% relative improvement over the
last 500 iterations. Success is checked first. Stagnation permits the next
random start, and its iteration/stagnation counters start afresh. Set
`STAGNATION_WINDOW=0` to disable this rule. If all starts fail, FO returns its
lowest-regret certified profile with `success=False`. Regret checking every
iteration adds full best-response LP solves even for mini-batch updates.

Choose a new `VERSION` or `RESULT_DIR` whenever changing settings. The runner
rejects a conflicting saved configuration, and the collector rejects completed
shards from a different payoff model. Legacy lookups stay inside the new
campaign's empty `legacy/` directory, so old uniform-game results are not imported.

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
  --legacy-dir experiments/results/remote/population_qptas_fo/beta_uniform_v1/legacy \
  --result-dir experiments/results/remote/population_qptas_fo/beta_uniform_v1 \
  --risk msd cvar --K 1000 2000 4000 --n 50 --reps 20 \
  --methods qptas stochastic_full_batch stochastic_minibatch \
  --payoff-model cell_beta_uniform_v1 --stochastic-n-random-starts 4 \
  --output experiments/results/remote/population_qptas_fo/beta_uniform_v1/capped_method_results.csv
```
