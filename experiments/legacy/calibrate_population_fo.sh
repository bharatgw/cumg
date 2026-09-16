#!/usr/bin/env bash
set -euo pipefail

# Reproduce the separate population-game FO pilot; never writes campaign shards.
# Usage: bash experiments/runners/calibration.sh baseline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PHASE="${1:-baseline}"
PYTHON="${PYTHON:-python}"
if [[ "$PHASE" == cvar_* ]]; then
  RESULT_DIR="${RESULT_DIR:-experiments/results/population_fo_calibration/cvar_steps_v2}"
else
  RESULT_DIR="${RESULT_DIR:-experiments/results/population_fo_calibration/pilot_v1}"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

args=(
  --risk msd cvar --K 500 --n 50 --reps 1 --seed-base 9123123
  --payoff-model cell_beta_uniform_v1 --jit-updates --gamma 0.5 --alpha 0.5
  --max-iter 2000 --n-random-starts 0 --regret-tolerance 0.01
  --certify-every 100 --record-every 100
  --stagnation-window 500 --stagnation-rtol 0.005 --stagnation-atol 0.00001
  --step-decay 0.5 --logit-bound 20
)
case "$PHASE" in
  cvar_control|cvar_step_decay|cvar_step_constant)
    args+=(--risk cvar --methods full_batch --seed-base 9323123
      --entropy-kappa 0.01 --smoothing-tau 0.002)
    if [[ "$PHASE" == cvar_control ]]; then
      args+=(--step-size 500)
    else
      args+=(--step-size-grid 10 100 500 --theta-step-size-grid 0.00001 0.0001 0.001)
      if [[ "$PHASE" == cvar_step_constant ]]; then
        args+=(--step-decay 0)
      fi
    fi
    ;;
  cvar_refine_decay|cvar_refine_constant)
    args+=(--risk cvar --methods full_batch --seed-base 9323123
      --entropy-kappa-grid 0.003 0.001 --smoothing-tau-grid 0.002 0.005 0.01)
    if [[ "$PHASE" == cvar_refine_decay ]]; then
      args+=(--step-size-grid 100 500 --theta-step-size 0.00001)
    else
      args+=(--step-size-grid 10 30 --theta-step-size 0.0001 --step-decay 0)
    fi
    ;;
  cvar_confirm_decay|cvar_confirm_constant|cvar_confirm_control)
    # Development games: compare five-start performance before selecting settings.
    args+=(--risk cvar --seed-base 9323123 --reps 3 --n-random-starts 4 --smoothing-tau 0.002)
    if [[ "$PHASE" == cvar_confirm_constant ]]; then
      args+=(--entropy-kappa 0.003 --step-size 30 --theta-step-size 0.0001 --step-decay 0)
    else
      args+=(--entropy-kappa 0.01 --step-size 500)
      if [[ "$PHASE" == cvar_confirm_decay ]]; then
        args+=(--theta-step-size 0.00001)
      fi
    fi
    ;;
  cvar_validate_500|cvar_validate_1000)
    # Selected on the three development games, before these held-out seeds.
    # Shared threshold/strategy steps won for both modes by success count,
    # then median certified regret. Compare both modes without further tuning.
    args+=(--risk cvar --seed-base 9423123 --reps 3 --n-random-starts 4
      --entropy-kappa 0.01 --smoothing-tau 0.002 --step-size 500)
    if [[ "$PHASE" == cvar_validate_1000 ]]; then
      args+=(--K 1000)
    fi
    ;;
  baseline)
    args+=(--entropy-kappa 0.1 --smoothing-tau 0.02 --step-size 1)
    ;;
  regularization_msd)
    args+=(--risk msd --entropy-kappa-grid 0.01 0.003 0.001
      --smoothing-tau-grid 0.02 0.002 --step-size 1)
    ;;
  regularization_steps_cvar)
    args+=(--risk cvar --entropy-kappa-grid 0.01 0.003 0.001
      --smoothing-tau-grid 0.02 0.002 --step-size-grid 0.001 0.01 0.1 1)
    ;;
  steps_msd)
    args+=(--risk msd --entropy-kappa-grid 0.01 0.003 0.001
      --smoothing-tau-grid 0.02 0.002 --step-size-grid 10 100 1000)
    ;;
  large_steps_cvar)
    args+=(--risk cvar --entropy-kappa-grid 0.003 0.001
      --smoothing-tau 0.002 --step-size-grid 10 100 1000)
    ;;
  logit_steps|logit_control)
    args+=(--entropy-kappa-grid 0.001 0.01 --smoothing-tau 0.002 --step-size-grid 200 500)
    if [[ "$PHASE" == logit_steps ]]; then
      args+=(--logit-bound 100)
    fi
    ;;
  validation_msd|larger_msd|validation_cvar_full|larger_cvar_full|validation_cvar_mini|larger_cvar_mini)
    # Held-out game seeds. Freeze these settings before looking at their results.
    args+=(--seed-base 9223123 --reps 3 --n-random-starts 4 --smoothing-tau 0.002)
    if [[ "$PHASE" == larger_* ]]; then
      args+=(--K 1000 --reps 1)
    fi
    case "$PHASE" in
      *_msd)
        args+=(--risk msd --entropy-kappa 0.01 --step-size 1000)
        ;;
      *_cvar_full)
        args+=(--risk cvar --methods full_batch --entropy-kappa 0.01
          --step-size 500 --logit-bound 100)
        ;;
      *_cvar_mini)
        args+=(--risk cvar --methods minibatch --entropy-kappa 0.001 --step-size 200)
        ;;
    esac
    ;;
  *)
    printf 'Unknown calibration phase: %s\n' "$PHASE" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULT_DIR"
if [[ -e "$RESULT_DIR/$PHASE.csv" || -e "$RESULT_DIR/${PHASE}_history.csv" ]]; then
  printf 'Results already exist for %s. Choose a fresh RESULT_DIR to rerun.\n' "$PHASE" >&2
  exit 2
fi
exec "$PYTHON" -u -m experiments.runners.compare_stochastic_fo "${args[@]}" \
  --csv "$RESULT_DIR/$PHASE.csv" --history-csv "$RESULT_DIR/${PHASE}_history.csv" \
  > "$RESULT_DIR/$PHASE.log" 2>&1
