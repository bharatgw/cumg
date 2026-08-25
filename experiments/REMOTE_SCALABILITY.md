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

## Resume CVaR With a Per-Method Cap

Use the capped resume runner if an uncapped CVaR campaign has reached its
practical time limit. Stop the old runner and its child workers first. The new
runner refuses to start while a legacy `.cvar_*.lock` directory is present, so
it cannot silently duplicate active work.

The default protocol applies a 24-hour wall-clock cap separately to every
`(risk, K, n, replicate, method)` task. It:

- reuses a legacy method result when its recorded runtime is at most 24 hours;
- classifies a legacy result taking more than 24 hours as a timeout;
- launches only methods missing from incomplete or unstarted legacy rows;
- runs methods in independent processes so one timeout cannot block later
  methods; and
- writes a long-form `capped_method_results.csv` containing completed,
  timed-out, errored, and pending tasks.

From the repository root on the remote machine:

```bash
nohup ./experiments/run_cvar_scalability_capped_resume.sh \
  > capped_24h_runner.log 2>&1 &
```

The defaults resume from
`experiments/results/remote/cvar_scalability/v1` into
`experiments/results/remote/cvar_scalability/capped_24h_v1`. To inspect the
plan without launching methods:

```bash
DRY_RUN=1 ./experiments/run_cvar_scalability_capped_resume.sh
```

The command is restartable. Completed method shards and timeout markers are
skipped on subsequent invocations. A non-timeout process error is recorded and
skipped by default; set `RETRY_ERRORS=1` to retry those tasks after inspecting
their logs.

Do not mix cap values in one result directory. To use a different cap, choose
both a new limit and a new version, for example:

```bash
METHOD_TIME_LIMIT_SECONDS=172800 VERSION=capped_48h_v1 \
  ./experiments/run_cvar_scalability_capped_resume.sh
```

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
