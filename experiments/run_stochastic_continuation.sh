#!/usr/bin/env bash
set -euo pipefail

WORKERS=8
PYTHON_BIN="$(PYENV_VERSION=.venv pyenv which python)"

VERSION="v3"
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
    for step in 15 20 25; do
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

  case "$step" in
    15) continuation_steps=(15 12 9 6 4 2.5 1.5 1 1) ;;
    20) continuation_steps=(20 19 18 17 16 15 14 13 12) ;;
    25) continuation_steps=(25 20 15 10 5 4 2 1 0.5) ;;
    *) echo "Unsupported initial step: $step" >&2; exit 2 ;;
  esac

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
    --continuation-kappa 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 1e-6 1e-8 \
    --continuation-tau 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 2e-6 2e-8 \
    --continuation-max-iter 500 1500 1500 1000 1000 1000 1000 1000 1000 \
    --continuation-step-size "${continuation_steps[@]}" \
    --step-decay 0.5 \
    --record-every 100 \
    --certify-every 100 \
    --stagnation-window 500 \
    --stagnation-rtol 0.005 \
    --stagnation-atol 0.00001 \
    --continuation-stage-rtol 0.005 \
    --continuation-stage-atol 0.00001 \
    --regret-tolerance 0.001 \
    --csv "$summary" \
    --history-csv "$history" \
    >> "$log" 2>&1

  status=$?
  echo "END $(date "+%Y-%m-%dT%H:%M:%S%z") status=$status $stem" | tee -a "$log"
  exit "$status"
' _
