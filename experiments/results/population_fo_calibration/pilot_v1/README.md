# Population-game FO calibration

This is a separate exploratory dataset for `cell_beta_uniform_v1`, not part of
the paper's scalability results. The selected MSD settings were subsequently
adopted for the [v2 campaign](../../../../POPULATION_QPTAS_FO.md); CVaR uses the
settings from the [follow-up pilot](../cvar_steps_v2/README.md).
Here, **kappa is the FO entropy weight** and **tau is the FO smoothing parameter**;
neither is the QPTAS strategy denominator or screening sample count.

## Result and decision

The pilot improved MSD performance, but did **not** establish reliable settings
for achieving regret 0.01 across these games. The 16 validation solves produced
three successes, all MSD. Keep these as calibration results; do not treat the
selected settings as validated defaults for the full K=500/1000/2000/4000 grid.

| Method | Entropy | Tau | Initial step | Logit bound | K=500 successes | K=1000 successes | Validation regret range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSD full batch | 0.01 | 0.002 | 1000 | 20 | 1/3 | 1/1 | 0.00856–0.01512 |
| MSD mini-batch | 0.01 | 0.002 | 1000 | 20 | 0/3 | 1/1 | 0.00885–0.01845 |
| CVaR full batch | 0.01 | 0.002 | 500 | 100 | 0/3 | 0/1 | 0.01087–0.02751 |
| CVaR mini-batch | 0.001 | 0.002 | 200 | 20 | 0/3 | 0/1 | 0.01750–0.07594 |

Each validation solve allowed five starts; successful solves stopped early.
These small, deliberately limited checks are not precise success-rate estimates.
K=2000 and K=4000 were not tested in this calibration.

The original settings' training regrets were about 0.0633 (MSD) and 0.0535
(CVaR) in both modes. The best training regrets after tuning were 0.02147 and
0.01783 for MSD full/mini-batch, and 0.01345 and 0.02132 for CVaR full/mini-batch.
Selection used the lowest true regret per method. Bounds 20 and 100 tied exactly
for the selected CVaR full-batch training result; validation used 100. Increasing
the bound did not improve any of the 16 matched training comparisons, and
worsened one CVaR mini-batch result.

**Recommendation:** entropy/tau changes alone are insufficient. Before another
large campaign, address the residual optimization's conditioning, particularly
the relative update scales of CVaR logits and tail thresholds. The large steps
above are exploratory candidates, not a general recommendation. All successful
outputs still passed the original full-regret test.

`summary.json` records every measured run and the grouped results.
`certificate_audit.json` rechecks all 16 validation profiles: LP certificates
were recomputed, and current/deviation payoffs were independently evaluated
using einsum and sorted tails. The maximum payoff discrepancy was 3.77e-15.
This verifies the arithmetic behind the recorded certificates, rather than
claiming comparison against a different LP solver.

## Design

- Two players, 50 actions each; gamma = 0.5 and CVaR alpha = 0.5.
- The existing generator assigns each player's payoff cell uniformly to
  Beta(1,4), Beta(2,3), Beta(3,2), Beta(4,1), or Uniform[0,1]. All K samples are
  retained for the current payoff and mixed best-response LP certificates.
- Success means a finite, full-sample regret certificate **eta <= 0.01**.
- At most 2,000 updates per start; certification and history every 100 updates,
  including initialization and the final update. Stagnation uses a 500-update
  window, relative tolerance 0.005 and absolute tolerance 1e-5.
- Mini-batches contain ceil(sqrt(K)) samples; step sizes decay as 1/sqrt(t).
- Initial screening used K=500 and one uniform start per setting, with seed base
  9123123: MSD seed 14128123 and CVaR seed 15128123. Settings share their risk's
  game and sampling seed. This is one training game per risk, not 72 independent
  games or a success-rate estimate.
- Validation freezes one setting per risk/method and uses seed base 9223123:
  three new games at K=500 and one new game at K=1000 per risk. Each solve uses
  a uniform start plus up to four random restarts, stopping at the first success.
  Validation settings were selected before inspecting these seeds.

The initial screening comprised 144 method solves (72 risk/settings pairs,
each evaluated with full batch and mini-batch). None reached eta <= 0.01.
`baseline.csv` uses the campaign's entropy 0.1, smoothing 0.02, step 1 and
logit bound 20. The remaining screening files vary entropy, smoothing, step
size, and then compare logit bounds 20 and 100 at identical settings.

## Interpretation

Reducing regularization alone did not solve the optimization problem within
this budget. Large steps improved some true-regret certificates, but CVaR's
tail-threshold variables became unstable: on the training game, the baseline
full-batch thresholds moved from about (0.493, 0.498) to (0.131, -0.034).
At entropy 0.01, tau 0.002 and step 500 they ended near (-856, -1603).
The full residual remained near sqrt(0.5), rather than converging to zero.
These are diagnostics of the current update rule, not reliable production
settings just because they produced the lowest observed regrets.

There are two different errors to control: regularization/smoothing bias and
incomplete optimization. For an actual equilibrium of the **unrestricted**
smoothed, entropy-regularized game, the code's one-sided softplus approximation
gives the following sufficient bias bounds (natural logarithms):

```text
MSD:  kappa * log(n) + gamma * tau * log(2)
CVaR: kappa * log(n) + (gamma / alpha) * tau * log(2)
```

The smoothed payoff lies between the true payoff minus the smoothing term and
the true payoff. Entropy-regularized optimality then adds at most kappa*log(n)
to a unilateral gain. These bounds require solving that regularized game;
they do not certify an unconverged iterate or a fixed point caused by logit
clipping. At n=50, kappa=0.001 and tau=0.002, the bias budgets are about
0.00461 for MSD and 0.00530 for CVaR. These values leave room under 0.01 in
principle, but were not successful solver settings in the screening runs.
Always use the full-regret certificate to decide success.

## Reproduction and provenance

Activate the repository's Python environment with JAX installed. For example:

```bash
# Existing files are protected; use a fresh output directory to reproduce.
RESULT_DIR=experiments/results/calibration/population_fo/reproduction \
  bash experiments/calibrate_population_fo.sh baseline
```

Run the same command with each of these phase names:

```text
baseline
regularization_msd
regularization_steps_cvar
steps_msd
large_steps_cvar
logit_steps
logit_control
validation_msd
validation_cvar_full
validation_cvar_mini
larger_msd
larger_cvar_full
larger_cvar_mini
```

The runner reuses `compare_stochastic_fo.py`, the existing payoff generator,
seed formula, solvers and CSV writer. Each phase records its settings and
population assignments; `_history.csv` files retain checkpoint regrets and
CVaR thresholds. Validation result rows also save the returned strategies and
all start summaries, allowing certificates to be checked again.

`metadata.json` records source hashes and software versions. These are local
CPU runs with NumPy 2.2.6, not bitwise replays of the synced NumPy 2.5.3 games.
`jit_updates=True` compiles the same updates; sampling, stopping and certificates
remain in the original loop. Compilation time is included. Some phases ran
concurrently, so the recorded laptop timings are diagnostic and should not be
mixed with remote scalability timings. Logs retain the local NumPy matmul
warnings; independent payoff arithmetic is checked for the validation profiles.

To repeat that arithmetic check against the saved validation results:

```bash
python experiments/results/calibration/population_fo/pilot_v1/audit_certificates.py
```

Verification: 294 tests passed, 15 tests requiring external solvers were
deselected; Ruff lint/format, shell syntax, and diff whitespace checks passed.
