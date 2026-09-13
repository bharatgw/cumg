# QPTAS runtime benchmark: two players, 50 actions, 500 samples

All ten searches reached the 30-second benchmark cap without finding an epsilon-DRE. These are censored runs: time to solution is unknown and exceeds the measured window. They are not completed solves or exhaustive failures.

## Configuration

- kappa = ceil(sqrt(50)) = 8; epsilon = 0.01; gamma = 0.5; CVaR alpha = 0.5.
- Seeds 0, 1, 2, 3, 4. Payoffs are iid Uniform[0,1]; scenario probabilities are 1/500.
- Both risks use the same game at each seed. Runs are sequential, with risk order alternating between seeds.
- Uses the existing QPTAS functions and their exact enumeration order. The benchmark wraps enumeration and LP calls for counting and checks the cap between profiles. It never interrupts an LP or changes the search order.
- Timings include solver initialization within the call and benchmark instrumentation; exclude process imports, payoff generation, file writing, and post-run validation. Small overruns above 30 seconds are intentional because the cap is checked between profiles.
- Environment: macOS-26.5.1-arm64-arm-64bit; Python 3.12.11, NumPy 2.2.6, SciPy 1.16.1.

## Measurements

| Seed | MSD elapsed (s) | MSD profiles | MSD ms/profile | CVaR elapsed (s) | CVaR profiles | CVaR ms/profile |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.011 | 2099 | 14.30 | 30.017 | 1397 | 21.49 |
| 1 | 30.006 | 2298 | 13.06 | 30.021 | 1394 | 21.54 |
| 2 | 30.013 | 1919 | 15.64 | 30.011 | 1297 | 23.14 |
| 3 | 30.011 | 2130 | 14.09 | 30.000 | 1445 | 20.76 |
| 4 | 30.002 | 1981 | 15.14 | 30.004 | 1394 | 21.52 |

Every evaluated profile was rejected by the first player, so each profile required one LP. Throughput applies to this initial segment, not necessarily later profiles requiring two LPs.

## Extrapolation and limits

There are comb(57,49) = 1,652,411,475 strategies per player and 2,730,463,682,711,675,625 joint profiles.
- MSD: mean 14.45 ms/profile across seeds; range 13.06-15.64 ms; pooled throughput 69.49 profiles/s. Full-grid traversal at each seed's measured rate averages 1.250 billion years.
- CVAR: mean 21.69 ms/profile across seeds; range 20.76-23.14 ms; pooled throughput 46.16 profiles/s. Full-grid traversal at each seed's measured rate averages 1.877 billion years.

These full-grid figures are arithmetic extrapolations at observed prefix throughput, not predictions or rigorous bounds on the time to the first acceptable profile. Early stopping could occur much sooner; later LP costs can also change. No mean time-to-solution can be estimated from these ten capped runs.

## Numerical checks

The local NumPy/Accelerate build emitted divide-by-zero, overflow, and invalid-value warnings in matmul despite finite bounded operands. The existing search explicitly rejects non-finite current or best-response values. A standalone bounded matrix-vector check matched einsum to 5.56e-16 despite emitting the same warnings.

After timing finished, the first and last checked profile from each of the ten runs were verified using independent einsum contractions and direct risk formulas. All 20 checks agreed within 1e-12 and independently confirmed first-player regret above epsilon. Details are in numerical_validation.json; the observed warnings are retained in timings.csv.

## Reproduction

```bash
PYENV_VERSION=.venv PYTHONDONTWRITEBYTECODE=1 python -u experiments/benchmark_qptas_runtime.py \
  --n 50 --K 500 --kappa 8 --seeds 0 1 2 3 4 \
  --gamma 0.5 --alpha 0.5 --epsilon 0.01 --seconds 30 \
  --output-dir experiments/results/qptas/n50_K500_kappa8_rerun
```

The output directory must be new. config.json records the settings, environment, and source hashes. timings.csv contains all measured runs.
