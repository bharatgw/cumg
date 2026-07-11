# Experiments

This directory preserves the research workflows and committed outputs used while developing `cumg`.

## Contents

- `mcpSolvers.ipynb`: main notebook for MCP formulations, small-support experiments, and plots.
- `mcpAlgoAnalysis.ipynb`: analysis and plotting notebook for scalability outputs.
- `compare_small_support_msd.py`: script comparing MSD small-support search backends.
- `compare_stochastic_fo.py`: script comparing full-batch and mini-batch MSD or CVaR stochastic first-order runs.
- `legacy/`: original standalone scripts kept as provenance for the packaged implementation.
- `results/`: committed CSV outputs from experiments.

## Result Files

- `results/prisonersDilemmaMSD.csv`: MSD prisoner-dilemma style experiment output.
- `results/scalabilityExperimentMSD.csv`: early MSD scalability experiment output.
- `results/scalabilityExperimentMSDv2.csv`: current MSD scalability experiment output.
- `results/scalabilityExperimentcVaR.csv`: CVaR scalability experiment output.
- `results/partial/`: partial or checkpointed experiment outputs.

These CSV files are intentionally tracked in git for reproducibility, but they are not package data and are not installed with `cumg`.

## Reproducing

1. Install the package with notebook dependencies:

   ```bash
   pip install -e ".[dev,notebooks]"
   ```

2. Install an external MCP/NLP solver such as PATH/PATHAMPL or IPOPT and make it visible to Pyomo.
3. Open the notebooks from this directory or adapt the code into scripts.
4. Write new intermediate outputs to `experiments/tmp/` or `experiments/scratch/`; those paths are ignored.
5. Promote only curated, documented CSV outputs into `experiments/results/`.

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

Large third-party solver distributions, PDFs, and archives were removed from the publishable tree. Local copies, if present, live under `.local/removed_artifacts/` and are ignored by git.
