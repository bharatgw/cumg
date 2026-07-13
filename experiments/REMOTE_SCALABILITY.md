# Remote Scalability Runs

Use this workflow when the full scalability grid is too slow or fragile to run
inside a local Jupyter notebook. The first remote target should be a dedicated
VM or lab machine where PATHAMPL/IPOPT can be installed directly. A custom
Docker image is only needed later if you move to GitLab Docker/shared runners.

## VM Setup

Create a Python environment and install the package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,stochastic,notebooks]"
```

Install PATHAMPL and IPOPT separately, then make their executables visible on
`PATH`. Check solver visibility before launching long jobs:

```bash
python -c "from cumg import format_solver_availability; print(format_solver_availability())"
```

Pin BLAS/OpenMP thread counts so parallel shards do not oversubscribe the VM:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Smoke Test

Run one small instance first:

```bash
python experiments/compare_scalability_approaches.py \
  --risk msd --K 5 --n 5 --reps 1 \
  --max-candidates 1000 \
  --n-screen-starts 3 \
  --n-support-starts 20 \
  --screen-maxiter 1000 \
  --support-maxiter 1000 \
  --max-iter 1000 \
  --solver pathampl \
  --fallback-solver ipopt \
  --csv experiments/results/remote/smoke.csv
```

## Sharded Runs

Run independent shards, typically one `(risk, K, n)` per process. Each completed
row is written and flushed immediately, so interrupted jobs preserve completed
replicates.

```bash
python experiments/compare_scalability_approaches.py \
  --risk msd \
  --K 100 \
  --n 10 \
  --reps 20 \
  --max-candidates 1000 \
  --n-screen-starts 3 \
  --n-support-starts 20 \
  --screen-maxiter 1000 \
  --support-maxiter 1000 \
  --max-iter 1000 \
  --solver pathampl \
  --fallback-solver ipopt \
  --csv experiments/results/remote/msd_K100_n10.csv \
  --quiet
```

To split replicate ranges across machines, use absolute replicate indices:

```bash
python experiments/compare_scalability_approaches.py \
  --risk cvar \
  --K 100 \
  --n 10 \
  --reps 20 \
  --rep-start 0 \
  --rep-stop 10 \
  --max-candidates 1000 \
  --n-screen-starts 3 \
  --n-support-starts 20 \
  --screen-maxiter 1000 \
  --support-maxiter 1000 \
  --max-iter 1000 \
  --solver pathampl \
  --fallback-solver ipopt \
  --csv experiments/results/remote/cvar_K100_n10_rep00_09.csv \
  --quiet
```

`--rep-stop` is exclusive. Seeds are computed from the original absolute
replicate index, so shard outputs recombine consistently.

## Notebook Merge

Copy or sync the CSV shards back into the repo, then merge them in the analysis
notebook:

```python
from pathlib import Path

import pandas as pd

result_dir = Path("experiments/results/remote")
df = pd.concat(
    [pd.read_csv(path) for path in sorted(result_dir.glob("*.csv"))],
    ignore_index=True,
)
```
