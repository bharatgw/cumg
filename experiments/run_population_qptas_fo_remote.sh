#!/usr/bin/env bash
set -euo pipefail

# Fixed populations per payoff entry; sampled QPTAS and both stochastic FO modes.
# Defaults: 360 method jobs, 2000 iterations per FO start, uniform + 4 retries.
# Preview: DRY_RUN=1 bash experiments/run_population_qptas_fo_remote.sh
# Run: nohup bash experiments/run_population_qptas_fo_remote.sh > population_runner.log 2>&1 &
# See experiments/POPULATION_QPTAS_FO.md for the design and result interpretation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PAYOFF_MODEL="cell_beta_uniform_v1"
export METHODS="qptas stochastic_full_batch stochastic_minibatch"
export RISK_GRID="${RISK_GRID:-msd cvar}"
export K_GRID="${K_GRID:-1000 2000 4000}"
export N_GRID="${N_GRID:-50}"
export REPS="${REPS:-20}"
export WORKERS="${WORKERS:-8}"
export SEED_BASE="${SEED_BASE:-123}"
export GAMMA="${GAMMA:-0.5}"
export ALPHA="${ALPHA:-0.5}"
export EPSILON="${EPSILON:-0.01}"
export STOCHASTIC_REGRET_TOLERANCE="$EPSILON"
export STOCHASTIC_N_RANDOM_STARTS="${STOCHASTIC_N_RANDOM_STARTS:-4}"
export MAX_ITER="${MAX_ITER:-2000}"
export CERTIFY_EVERY="${CERTIFY_EVERY:-1}"
export STAGNATION_WINDOW="${STAGNATION_WINDOW:-500}"
export STAGNATION_RTOL="${STAGNATION_RTOL:-0.005}"
export STAGNATION_ATOL="${STAGNATION_ATOL:-0.00001}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
export METHOD_TIME_LIMIT_SECONDS="${METHOD_TIME_LIMIT_SECONDS:-86400}"
export VERSION="${VERSION:-beta_uniform_v1}"
export RESULT_DIR="${RESULT_DIR:-experiments/results/remote/population_qptas_fo/$VERSION}"

# Keep all legacy lookups inside this separate campaign.
export LEGACY_RESULT_DIR="$RESULT_DIR/legacy"
mkdir -p "$LEGACY_RESULT_DIR"
exec bash "$SCRIPT_DIR/run_cvar_scalability_capped_resume.sh"
