#!/usr/bin/env bash
set -euo pipefail

WORKERS=8
PYTHON_BIN="$(PYENV_VERSION=.venv pyenv which python)"

VERSION="v2"
RESULT_DIR="experiments/results/stochastic/continuation_shards/$VERSION"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

export PYTHON_BIN RESULT_DIR LOG_DIR
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

for risk in msd cvar; do
  for K in 250; do
    for step in 5 10 15; do
      for rep in 0 1 2; do
        printf "%s %s %s %s\n" "$risk" "$K" "$step" "$rep"
      done
    done
  done
done | xargs -n 4 -P "$WORKERS" bash -c '
  set -u

  risk="$1"
  K="$2"
  step="$3"
  rep="$4"

  # This reproduces the original rep=0,1,2 seed sequence while each
  # shard itself runs only one repetition.
  seed_base=$((123 + rep))
  stem="${risk}_K${K}_n20_step${step}_rep${rep}"
  summary="$RESULT_DIR/${stem}_summary.csv"
  history="$RESULT_DIR/${stem}_history.csv"
  log="$LOG_DIR/${stem}.log"

  if [[ -s "$summary" ]] && [[ "$(wc -l < "$summary")" -ge 2 ]]; then
    echo "SKIP complete: $stem"
    exit 0
  fi

  echo "START $(date "+%Y-%m-%dT%H:%M:%S%z") $stem" | tee "$log"

  "$PYTHON_BIN" experiments/compare_stochastic_fo.py \
    --risk "$risk" \
    --K "$K" \
    --n 20 \
    --reps 1 \
    --seed-base "$seed_base" \
    --continuation-kappa 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 \
    --continuation-tau 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 \
    --continuation-max-iter 2000 2000 2000 2000 2000 2000 2000 \
    --step-size "$step" \
    --step-decay 0.5 \
    --record-every 100 \
    --certify-every 100 \
    --regret-tolerance 0.001 \
    --csv "$summary" \
    --history-csv "$history" \
    >> "$log" 2>&1

  status=$?
  echo "END $(date "+%Y-%m-%dT%H:%M:%S%z") status=$status $stem" | tee -a "$log"
  exit "$status"
' _