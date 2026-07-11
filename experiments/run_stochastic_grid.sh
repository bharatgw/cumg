#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: WORKERS=8 $0 {msd|cvar} [{msd|cvar} ...]"
}

if (( $# == 0 )); then
  usage
  exit 2
fi

for risk in "$@"; do
  case "$risk" in
    msd|cvar) ;;
    *)
      echo "Unsupported risk: $risk" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

WORKERS="${WORKERS:-8}"
REPS="${REPS:-20}"
SEED_BASE="${SEED_BASE:-123}"
GAMMA="${GAMMA:-0.5}"
ALPHA="${ALPHA:-0.5}"
EPSILON="${EPSILON:-0.001}"
MAX_ITER="${MAX_ITER:-1000}"
ENTROPY_KAPPA="${ENTROPY_KAPPA:-0.05}"
SMOOTHING_TAU="${SMOOTHING_TAU:-0.1}"
STEP_SIZE="${STEP_SIZE:-0.05}"
STEP_DECAY="${STEP_DECAY:-0.5}"
LOGIT_BOUND="${LOGIT_BOUND:-20.0}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-0}"
RECORD_EVERY="${RECORD_EVERY:-0}"
CERTIFY_EVERY="${CERTIFY_EVERY:-1}"
RESULT_DIR="${RESULT_DIR:-experiments/results/stochastic/grid}"
LOG_DIR="${LOG_DIR:-experiments/results/stochastic/logs/grid}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "pyenv is unavailable; set PYTHON_BIN to the project Python executable." >&2
    exit 2
  fi
  PYTHON_BIN="$(PYENV_VERSION=.venv pyenv which python)"
fi

mkdir -p "$RESULT_DIR" "$LOG_DIR"

run_config="$(printf '%s\n' \
  "reps=$REPS" \
  "seed_base=$SEED_BASE" \
  "gamma=$GAMMA" \
  "alpha=$ALPHA" \
  "epsilon=$EPSILON" \
  "max_iter=$MAX_ITER" \
  "entropy_kappa=$ENTROPY_KAPPA" \
  "smoothing_tau=$SMOOTHING_TAU" \
  "step_size=$STEP_SIZE" \
  "step_decay=$STEP_DECAY" \
  "logit_bound=$LOGIT_BOUND" \
  "gradient_clip_norm=$GRADIENT_CLIP_NORM" \
  "record_every=$RECORD_EVERY" \
  "certify_every=$CERTIFY_EVERY" \
  "methods=stochastic_full_batch,stochastic_minibatch")"
config_path="$RESULT_DIR/run_config.txt"
if [[ -f "$config_path" ]]; then
  if [[ "$(<"$config_path")" != "$run_config" ]]; then
    echo "Configuration differs from $config_path; use a new RESULT_DIR." >&2
    exit 2
  fi
else
  printf '%s\n' "$run_config" > "$config_path"
fi

# Each worker gets one CPU thread so concurrent JAX/BLAS processes do not
# oversubscribe the machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

export REPS SEED_BASE GAMMA ALPHA EPSILON MAX_ITER
export ENTROPY_KAPPA SMOOTHING_TAU STEP_SIZE STEP_DECAY
export LOGIT_BOUND GRADIENT_CLIP_NORM RECORD_EVERY CERTIFY_EVERY
export RESULT_DIR LOG_DIR PYTHON_BIN

printf "Launching with %s workers; results=%s\n" "$WORKERS" "$RESULT_DIR"

for risk in "$@"; do
  for K in 5 10 30 100 250 500; do
    for n in 5 10 20 50; do
      printf "%s %s %s\n" "$risk" "$K" "$n"
    done
  done
done | xargs -n 3 -P "$WORKERS" bash -c '
  set -euo pipefail

  risk="$1"
  K="$2"
  n="$3"
  stem="${risk}_K${K}_n${n}"
  lock_dir="${RESULT_DIR}/.${stem}.lock"

  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "SKIP locked shard: risk=$risk K=$K n=$n" >&2
    exit 0
  fi
  trap '\''rmdir "$lock_dir" 2>/dev/null || true'\'' EXIT INT TERM

  completed="$($PYTHON_BIN -c '\''
import csv
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
stem = sys.argv[2]
count = 0
for path in result_dir.glob(f"{stem}_rep*.csv"):
    with path.open(newline="") as file:
        count += sum(1 for _ in csv.DictReader(file))
print(count)
'\'' "$RESULT_DIR" "$stem")"

  if (( completed > REPS )); then
    echo "ERROR $stem has $completed rows, exceeding REPS=$REPS" >&2
    exit 1
  fi
  if (( completed == REPS )); then
    echo "SKIP complete shard: risk=$risk K=$K n=$n rows=$completed"
    exit 0
  fi

  part=$(printf "%03d_to_%03d" "$completed" "$REPS")
  out="${RESULT_DIR}/${stem}_rep${part}.csv"
  log="${LOG_DIR}/${stem}_rep${part}.log"

  echo "START $(date "+%Y-%m-%dT%H:%M:%S%z") risk=$risk K=$K n=$n rep_start=$completed rep_stop=$REPS" \
    | tee "$log"

  set +e
  "$PYTHON_BIN" experiments/compare_scalability_approaches.py \
    --risk "$risk" \
    --K "$K" \
    --n "$n" \
    --reps "$REPS" \
    --rep-start "$completed" \
    --rep-stop "$REPS" \
    --seed-base "$SEED_BASE" \
    --gamma "$GAMMA" \
    --alpha "$ALPHA" \
    --epsilon "$EPSILON" \
    --max-iter "$MAX_ITER" \
    --entropy-kappa "$ENTROPY_KAPPA" \
    --smoothing-tau "$SMOOTHING_TAU" \
    --step-size "$STEP_SIZE" \
    --step-decay "$STEP_DECAY" \
    --logit-bound "$LOGIT_BOUND" \
    --gradient-clip-norm "$GRADIENT_CLIP_NORM" \
    --record-every "$RECORD_EVERY" \
    --certify-every "$CERTIFY_EVERY" \
    --solver pathampl \
    --fallback-solver none \
    --methods stochastic_full_batch stochastic_minibatch \
    --csv "$out" \
    --quiet >> "$log" 2>&1
  status=$?
  set -e

  echo "END $(date "+%Y-%m-%dT%H:%M:%S%z") status=$status risk=$risk K=$K n=$n" | tee -a "$log"
  exit "$status"
' _
