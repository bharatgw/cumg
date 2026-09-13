# Experiments

This directory preserves the research workflows and committed outputs used while developing `cumg`.

## Contents

- `mcpSolvers.ipynb`: main notebook for MCP formulations, small-support experiments, and plots.
- `mcpAlgoAnalysis.ipynb`: analysis and plotting notebook for scalability outputs.
- `compare_scalability_approaches.py`: general MSD/CVaR scalability runner.
- `run_population_qptas_fo_remote.sh`: QPTAS and both FO modes with fixed
  Beta/uniform populations per payoff entry, K=1000/2000/4000 and n=50.
  See [`POPULATION_QPTAS_FO.md`](POPULATION_QPTAS_FO.md) for setup and outputs.
- `compare_small_support_msd.py`: script comparing MSD small-support search backends.
- `compare_stochastic_fo.py`: script comparing full-batch and mini-batch MSD or CVaR stochastic first-order runs.
- `compare_stochastic_fo_msd.py`: superseded MSD-only predecessor retained for
  provenance; use `compare_stochastic_fo.py` for new runs.
- `capped_scalability_resume.py` and
  `run_cvar_scalability_capped_resume.sh`: planner and shell runner for the
  completed per-method capped CVaR campaign.
- `uniform_profile_baseline.py`: reproducible uniform-profile baseline generator.
- `scalability_analysis.py`: validated loading, reshaping, and summarization used by `mcpAlgoAnalysis.ipynb`.
- `legacy/`: original standalone scripts kept as provenance for the packaged implementation.
- `results/`: committed CSV outputs from experiments.

## Result Files

See [`results/README.md`](results/README.md) for the authoritative, legacy, and
incomplete datasets and the exact inputs used by the analysis notebook. Curated
CSV outputs are intentionally tracked for reproducibility, but they are not
package data and are not installed with `cumg`.

## Reproducing

1. Install the package with notebook dependencies:

   ```bash
   pip install -e ".[dev,notebooks]"
   ```

2. Install an external MCP/NLP solver such as PATH/PATHAMPL or IPOPT and make it visible to Pyomo.
3. Open the notebooks from this directory or adapt the code into scripts.
4. Write new intermediate outputs to `experiments/tmp/` or `experiments/scratch/`; those paths are ignored.
5. Promote only curated, documented CSV outputs into `experiments/results/`.

### Stochastic FO Starts and Stopping

Both MSD and CVaR FO solvers, in full-batch and mini-batch mode, start from
uniform strategies unless `x0`/`y0` are supplied. They return immediately when
a certificate satisfies `eta <= regret_tolerance`. Only an unsuccessful start
triggers a random restart, up to four by default. Each random profile draws
the players' strategies independently from Dirichlet(1, ..., 1). The seed
controls both these draws and the mini-batch streams. If all starts fail, the
solver returns the profile with the lowest certified regret.

Set `StochasticFOConfig(n_random_starts=0)` for the original single-start
behavior. In `compare_stochastic_fo.py`, use `--n-random-starts`; in
`compare_scalability_approaches.py`, use `--stochastic-n-random-starts`.
Both default to four additional starts.

The core configuration still defaults to `regret_tolerance=0.001` and
`certify_every=None`. To check the initial profile and then every 100 iterations
against a 0.01 target, use:

```python
config = StochasticFOConfig(
    regret_tolerance=0.01,
    certify_every=100,
    n_random_starts=4,
)
```

Use `certify_every=1` to check after every update. With certification disabled,
the solver checks regret only at the end of each start. Each start has its own
`max_iter` budget and fresh step-size decay. Optional stagnation stopping checks
whether the running-best certified regret improves by less than
`max(stagnation_atol, stagnation_rtol * abs(reference_eta))` over the configured
iteration window. Stagnation is disabled by default and requires periodic
certification. Reaching stagnation or the iteration limit without meeting the
regret target permits another start; it does not count as success.

Results record each attempted start's initial profile, seed, certificate,
runtime, iteration count, and stopping reason in `start_summaries`.
`selected_start` identifies the returned profile. Total iterations and runtime
include all attempted starts; history rows also carry `start_index` and
`start_iteration` so independent trajectories can be distinguished.

### Stochastic Hyperparameter Pilot

Run a small fixed-seed grid before launching the full scalability experiment:

```bash
python experiments/compare_stochastic_fo.py \
  --risk msd cvar \
  --K 30 100 \
  --n 20 \
  --reps 1 \
  --entropy-kappa-grid 0.05 0.01 0.002 \
  --smoothing-tau-grid 0.1 0.02 0.005 \
  --step-size-grid 0.01 0.05 0.2 \
  --max-iter 300 \
  --record-every 25 \
  --certify-every 25 \
  --regret-tolerance 0.001 \
  --csv experiments/results/stochastic/pilot_summary.csv \
  --history-csv experiments/results/stochastic/pilot_history.csv \
  --quiet
```

The summary CSV records the risk model, every stochastic hyperparameter, and
the iteration of the best exact-regret certificate. The history CSV records
the risk model, residual norm, objective, exact eta, player regrets, and CVaR
thresholds at each shared record/certification checkpoint. Grid values form a
Cartesian product and all configurations reuse the same game seeds within each
risk model.

### Stochastic Continuation

Use continuation when a fixed smoothing and entropy level reduces the residual
but stalls above the exact-regret target:

```bash
python experiments/compare_stochastic_fo.py \
  --risk msd cvar \
  --K 30 100 250 \
  --n 20 \
  --reps 3 \
  --continuation-kappa 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 \
  --continuation-tau 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 \
  --continuation-max-iter 500 1500 1500 1000 1000 1000 1000 \
  --continuation-step-size 10 8 6 4 3 2 1 \
  --step-decay 0.5 \
  --record-every 100 \
  --certify-every 100 \
  --stagnation-window 500 \
  --stagnation-rtol 0.005 \
  --stagnation-atol 1e-5 \
  --continuation-stage-rtol 0.005 \
  --continuation-stage-atol 1e-5 \
  --regret-tolerance 0.001 \
  --csv experiments/results/stochastic/continuation_summary.csv \
  --history-csv experiments/results/stochastic/continuation_history.csv \
  --quiet
```

Each stage warm-starts from the preceding stage's best certified profile and
restarts the step-size decay counter. CVaR also carries forward both threshold
variables for its first attempt. Each stage can also use the conditional random
restarts described above, with fresh CVaR thresholds for each random profile.
A successful certificate stops the entire continuation immediately. Otherwise,
a stage finishes after its allowed starts end at stagnation or the iteration
limit. Continuation itself stops when a completed stage contributes less than
the configured stage-level improvement. The summary reports the lowest exact eta
found across all stages and its selected stage, kappa, tau, and step size.
History rows include local and cumulative iterations plus stage settings and
termination reasons.

The historical `run_stochastic_continuation.sh` runner and the superseded
`compare_stochastic_fo_msd.py` explicitly use zero random restarts to preserve
the original experiment protocol. Use `compare_stochastic_fo.py` for new runs.

Large third-party solver distributions, PDFs, and archives were removed from the publishable tree. Local copies, if present, live under `.local/removed_artifacts/` and are ignored by git.
