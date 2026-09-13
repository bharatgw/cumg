#!/usr/bin/env bash
set -euo pipefail

# Sampled QPTAS on the capped scalability grid, for both MSD and CVaR.
# MAX_CANDIDATES counts distinct joint kappa-uniform profiles (not support
# sets). kappa=ceil(sqrt(n)); every regret check uses all K payoff samples.
# Defaults: 960 jobs, 8 workers, 24 hours per job, epsilon=0.01, 1000 profiles.
#
# Preview: DRY_RUN=1 bash experiments/run_qptas_scalability_remote.sh
# Run:     nohup bash experiments/run_qptas_scalability_remote.sh > qptas_runner.log 2>&1 &
# Resume with the same command; use a new RESULT_DIR when changing settings.
# See experiments/QPTAS_REMOTE.md for overrides and result interpretation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export RISK_GRID="${RISK_GRID:-msd cvar}"
export METHODS="qptas"
export K_GRID="${K_GRID:-5 10 30 100 250 500}"
export N_GRID="${N_GRID:-5 10 20 50}"
export REPS="${REPS:-20}"
export WORKERS="${WORKERS:-8}"
export SEED_BASE="${SEED_BASE:-123}"
export GAMMA="${GAMMA:-0.5}"
export ALPHA="${ALPHA:-0.5}"
export EPSILON="${EPSILON:-0.01}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
export METHOD_TIME_LIMIT_SECONDS="${METHOD_TIME_LIMIT_SECONDS:-86400}"
export VERSION="${VERSION:-sampled_1000_v1}"
export RESULT_DIR="${RESULT_DIR:-experiments/results/remote/qptas_scalability/$VERSION}"

# Reuse the existing planner, locks, timeouts, and collector. QPTAS has no
# legacy campaign to import; keep its legacy lookup inside its own results.
export LEGACY_RESULT_DIR="$RESULT_DIR/legacy"
mkdir -p "$LEGACY_RESULT_DIR"

exec bash "$SCRIPT_DIR/run_cvar_scalability_capped_resume.sh"
