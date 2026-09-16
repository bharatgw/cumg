# CVaR pilot with separate threshold learning rates

This follow-up tests whether separating CVaR's strategy and threshold learning
rates improves the full-regret certificate. It reuses the existing population
game generator, solver, restart policy, and best-response LPs. It is a calibration
dataset and is separate from the paper's scalability results.

## Result and recommendation

After **73 completed method solves** (43 screening, 18 development, 12
validation), the best tested settings remained **entropy 0.01, tau 0.002,
shared initial step 500, decay 0.5, and logit bound 20**. Smaller separate
threshold rates did not outperform that configuration in the five-start
development comparison. The optional new threshold-rate parameter should be
left unset for the selected configuration.

| Validation mode | K=500 successes | K=1000 successes | Total | Median regret | Regret range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full batch | 0/3 | 1/3 | 1/6 | 0.01077 | 0.00991–0.01654 |
| Minibatch | 0/3 | 1/3 | 1/6 | 0.01131 | 0.00703–0.01518 |

Both modes succeeded on the same K=1000 game (seed 20428123). Success is
strictly eta <= 0.01: nearby failures are not rounded into successes. All
solves allowed uniform plus four random starts, with at most 2,000 iterations
per start. K=2000 and K=4000 were not tested here.

**For another exploratory run, prefer full batch with the settings above:**
it has the same observed success count and slightly lower median regret.
The six validation games do not establish a precise success rate or a reliable
0.01 solver. In particular, the original shared-step update still suffers from
threshold overshooting, even when its returned strategy has a valid certificate.
These settings are now used for both CVaR batch modes in the
[v2 population campaign](../../../../POPULATION_QPTAS_FO.md).

`summary.json` contains every completed method run, and `selection.json`
records the choice made before validation. All **30 development/validation
profiles** were independently checked: best-response LPs were rerun, and
current/deviation payoffs were recomputed using einsum and sorted tails. The
maximum payoff difference was **4.00e-15**. This checks the certificate arithmetic;
it is not a comparison against an independently implemented LP solver.

## Design and selection

All games have two players, n=50, gamma=0.5, alpha=0.5, and equally weighted
scenarios drawn using `cell_beta_uniform_v1`. The target is a finite full-sample
regret certificate eta <= 0.01. Each start has at most 2,000 iterations, with
certificates every 100 iterations and the existing 500-iteration stagnation
window (rtol 0.005, atol 1e-5). Logit bounds are 20 throughout this follow-up.

1. **Screening:** 43 full-batch solves on one K=500 training game, seed 15328123,
   each from uniform. Entropy weights: 0.01, 0.003, 0.001. Smoothing values:
   0.002, 0.005, 0.01. Strategy steps: 10, 30, 100, 500. Separate threshold
   steps: 1e-5, 1e-4, 1e-3. Both constant steps and 1/sqrt(t) decay were tested.
   These were staged, adaptive grids, not the full Cartesian product; the exact
   combinations are in `calibrate_population_fo.sh`. None reached 0.01.
2. **Development comparison:** three selected configurations on training seeds
   15328123–15328125, in both full-batch and minibatch mode. Each solve allows
   uniform plus four random starts. This gives 18 completed method solves.
3. **Validation:** freeze the choice before inspecting new seeds, then evaluate
   both modes on three K=500 games and three K=1000 games. Seed base 9423123 is
   disjoint from development (9323123), the earlier pilot (9123123/9223123), and
   the main population campaign (123). Selection is recorded in `selection.json`.

The seed formula is `seed_base + 1_000_000*risk_index + 10_000*K + 100*n + rep`,
where CVaR's risk_index is 1. Validation seeds are 15428123–15428125 at K=500
and 20428123–20428125 at K=1000. Mini-batches have ceil(sqrt(K)) observations.
All certificates still solve best responses using the full K scenarios.

Configurations were ranked by the number of successful development games,
then by lower median certified regret. The same configuration won for both
modes: entropy **0.01**, tau **0.002**, initial strategy and threshold steps
**500**, decay **0.5**, logit bound **20**. This is the best of the configurations
tested under this rule, not a global optimum over hyperparameters.

| Development setting | Entropy | Tau | Strategy step | Threshold step | Decay | Full-batch successes / median eta | Minibatch successes / median eta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Shared-step control | 0.01 | 0.002 | 500 | 500 | 0.5 | 0/3; 0.01517 | 1/3; 0.01406 |
| Separate, decaying steps | 0.01 | 0.002 | 500 | 1e-5 | 0.5 | 0/3; 0.02011 | 1/3; 0.01942 |
| Separate, constant steps | 0.003 | 0.002 | 30 | 1e-4 | 0 | 0/3; 0.01657 | 0/3; 0.03060 |

## What changed in the solver

`StochasticFOConfig.theta_step_size` is optional. When omitted, the existing
shared learning rate is used. When supplied, the strategy logits use
`step_size / t**step_decay`, and the CVaR thresholds use
`theta_step_size / t**step_decay`. Both schedules restart at each new start.
MSD's updates are unaffected. This is a scaling of the existing residual-loss
gradient, not a change to the payoff model or regret definition.

The experiment driver exposes `--theta-step-size` and `--theta-step-size-grid`
and records the rates in results and histories. Tests check the exact trajectory
for a quadratic residual, compiled/eager agreement, validation of invalid rates,
and forwarding/recording of the new option.

## Numerical diagnostics

Smaller threshold steps prevented the extreme threshold excursions, but that
did not translate into better development regret overall. The shared-step
control can still send thresholds far outside the payoff range and leave a
large residual. A successful full-regret certificate would remain valid, but
would not establish convergence of the smoothed stationarity equations.

The separate decaying setting was also sensitive to roundoff. The restart
wrapper renormalizes probabilities once more, changing individual probabilities
by at most 1.30e-18. In a paired, full-batch 100-iteration check, this changed
the sum of strategy L1 differences by about 1.98 for the separate decaying
setting. The corresponding differences were 6.83e-11 for the separate constant
setting and 6.26e-13 for the shared-step control. The shared-step thresholds
were nevertheless near -1161 and -1449. See `numerical_stability.json`.
These three short paired diagnostics are excluded from experiment solve counts.

## Reproduction

Activate the repository Python environment with JAX installed, then run:

```bash
RESULT_DIR=experiments/results/calibration/population_fo/cvar_reproduction \
  bash experiments/calibrate_population_fo.sh cvar_step_decay
```

Use the same command with these phase names:

```text
cvar_control
cvar_step_decay
cvar_step_constant
cvar_refine_decay
cvar_refine_constant
cvar_confirm_decay
cvar_confirm_constant
cvar_confirm_control
cvar_validate_500
cvar_validate_1000
```

The runner refuses to overwrite an existing phase. Result rows contain the
settings, returned profiles, population maps, certificates, and per-start
summaries. Histories contain checkpoint regrets and CVaR thresholds. The
`interrupted_*` folder preserves partial histories from the interrupted first
development attempt; those are excluded from all counts. Only the complete
retries in the main directory are analyzed.

These are CPU runs with compiled JAX updates, including compilation time. Some
phases ran concurrently; timings are diagnostic rather than comparable to remote
scalability timings. `metadata.json` records software versions and source hashes.
As in the earlier pilot, the local NumPy version emits matmul warnings on some
finite inputs, so final profile certificates are checked again with independent
payoff arithmetic using einsum and sorted tails.

To rerun the certificate audit and the short numerical diagnostic:

```bash
python experiments/results/calibration/population_fo/pilot_v1/audit_certificates.py \
  --result-dir experiments/results/calibration/population_fo/cvar_steps_v2 \
  --phases cvar_confirm_decay cvar_confirm_constant cvar_confirm_control \
    cvar_validate_500 cvar_validate_1000 --expected-profiles 30

python experiments/results/calibration/population_fo/cvar_steps_v2/check_stability.py
```

Verification: **302 tests passed**, 15 tests requiring external solvers were
deselected. Ruff lint/format, shell syntax, and diff whitespace checks passed.
