# Experiments

This directory preserves the research workflows and committed outputs used while developing `cumgSolver`.

## Contents

- `mcpSolvers.ipynb`: main notebook for MCP formulations, small-support experiments, and plots.
- `mcpAlgoAnalysis.ipynb`: analysis and plotting notebook for scalability outputs.
- `legacy/`: original standalone scripts kept as provenance for the packaged implementation.
- `results/`: committed CSV outputs from experiments.

## Result Files

- `results/prisonersDilemmaMSD.csv`: MSD prisoner-dilemma style experiment output.
- `results/scalabilityExperimentMSD.csv`: early MSD scalability experiment output.
- `results/scalabilityExperimentMSDv2.csv`: current MSD scalability experiment output.
- `results/scalabilityExperimentcVaR.csv`: CVaR scalability experiment output.
- `results/partial/`: partial or checkpointed experiment outputs.

These CSV files are intentionally tracked in git for reproducibility, but they are not package data and are not installed with `cumgSolver`.

## Reproducing

1. Install the package with notebook dependencies:

   ```bash
   pip install -e ".[dev,notebooks]"
   ```

2. Install an external MCP/NLP solver such as PATH/PATHAMPL or IPOPT and make it visible to Pyomo.
3. Open the notebooks from this directory or adapt the code into scripts.
4. Write new intermediate outputs to `experiments/tmp/` or `experiments/scratch/`; those paths are ignored.
5. Promote only curated, documented CSV outputs into `experiments/results/`.

Large third-party solver distributions, PDFs, and archives were removed from the publishable tree. Local copies, if present, live under `.local/removed_artifacts/` and are ignored by git.

