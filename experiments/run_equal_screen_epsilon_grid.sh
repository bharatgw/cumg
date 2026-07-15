#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERSION="${VERSION:-v2}"
export VERSION
export WORKERS="${WORKERS:-10}"
export REPS="${REPS:-20}"
export RISK_GRID="${RISK_GRID:-msd cvar}"
export K_GRID="${K_GRID:-5 10 30 100 250 500}"
export N_GRID="${N_GRID:-50}"
export METHODS="${METHODS:-screened_dual}"
export EPSILON="${EPSILON:-0.01}"
export EPSILON_SCR="${EPSILON_SCR:-$EPSILON}"
export RESULT_DIR="${RESULT_DIR:-experiments/results/remote/higher_eps/$VERSION}"
export LOG_DIR="${LOG_DIR:-$RESULT_DIR/logs}"

exec "$SCRIPT_DIR/run_cvar_scalability_remote.sh"
