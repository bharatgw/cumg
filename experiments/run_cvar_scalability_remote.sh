#!/usr/bin/env bash
set -euo pipefail

# Override any of these with environment variables, for example:
#   VERSION=smoke K_GRID="5" N_GRID="5" REPS=1 WORKERS=1 ./experiments/run_cvar_scalability_remote.sh
WORKERS="${WORKERS:-8}"
REPS="${REPS:-20}"
SEED_BASE="${SEED_BASE:-123}"
RISK_GRID="${RISK_GRID:-cvar}"
K_GRID="${K_GRID:-5 10 30 100 250 500}"
N_GRID="${N_GRID:-5 10 20 50}"
METHODS="${METHODS:-mcp screened_dual action_dual restricted_mcp stochastic_full_batch stochastic_minibatch}"

GAMMA="${GAMMA:-0.5}"
ALPHA="${ALPHA:-0.5}"
EPSILON="${EPSILON:-0.01}"
EPSILON_SCR="${EPSILON_SCR:-}"
STOCHASTIC_REGRET_TOLERANCE="${STOCHASTIC_REGRET_TOLERANCE:-0.001}"
MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
N_SCREEN_STARTS="${N_SCREEN_STARTS:-3}"
N_SUPPORT_STARTS="${N_SUPPORT_STARTS:-20}"
SCREEN_MAXITER="${SCREEN_MAXITER:-1000}"
SUPPORT_MAXITER="${SUPPORT_MAXITER:-1000}"
MAX_ITER="${MAX_ITER:-1000}"
CERTIFY_EVERY="${CERTIFY_EVERY:-100}"

SOLVER="${SOLVER:-pathampl}"
FALLBACK_SOLVER="${FALLBACK_SOLVER:-none}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-0}"
DRY_RUN="${DRY_RUN:-0}"
VERSION="${VERSION:-v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-experiments/results/remote/cvar_scalability/$VERSION}"
LOG_DIR="${LOG_DIR:-$RESULT_DIR/logs}"
CONFIG_FILE="$RESULT_DIR/run_config.env"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got: $value" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a nonnegative integer; got: $value" >&2
    exit 2
  fi
}

require_positive_integer WORKERS "$WORKERS"
require_positive_integer REPS "$REPS"
require_nonnegative_integer RUN_TIMEOUT_SECONDS "$RUN_TIMEOUT_SECONDS"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if (( RUN_TIMEOUT_SECONDS > 0 )) && ! command -v timeout >/dev/null 2>&1; then
  echo "RUN_TIMEOUT_SECONDS requires GNU timeout (available by default on Ubuntu)." >&2
  exit 2
fi

mkdir -p "$RESULT_DIR" "$LOG_DIR"

run_config="$(printf '%s\n' \
  "risk=$RISK_GRID" \
  "K_GRID=$K_GRID" \
  "N_GRID=$N_GRID" \
  "REPS=$REPS" \
  "SEED_BASE=$SEED_BASE" \
  "METHODS=$METHODS" \
  "GAMMA=$GAMMA" \
  "ALPHA=$ALPHA" \
  "EPSILON=$EPSILON" \
  "STOCHASTIC_REGRET_TOLERANCE=$STOCHASTIC_REGRET_TOLERANCE" \
  "MAX_CANDIDATES=$MAX_CANDIDATES" \
  "N_SCREEN_STARTS=$N_SCREEN_STARTS" \
  "N_SUPPORT_STARTS=$N_SUPPORT_STARTS" \
  "SCREEN_MAXITER=$SCREEN_MAXITER" \
  "SUPPORT_MAXITER=$SUPPORT_MAXITER" \
  "MAX_ITER=$MAX_ITER" \
  "CERTIFY_EVERY=$CERTIFY_EVERY" \
  "SOLVER=$SOLVER" \
  "FALLBACK_SOLVER=$FALLBACK_SOLVER" \
  "WORKERS=$WORKERS" \
  "RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS")"
if [[ -n "$EPSILON_SCR" ]]; then
  run_config+=$'\n'"EPSILON_SCR=$EPSILON_SCR"
fi

if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(cat "$CONFIG_FILE")" != "$run_config" ]]; then
    echo "Configuration differs from $CONFIG_FILE." >&2
    echo "Use a new VERSION or RESULT_DIR instead of mixing experiment settings." >&2
    exit 2
  fi
else
  printf '%s\n' "$run_config" > "$CONFIG_FILE"
  {
    printf 'created_at=%s\n' "$(date "+%Y-%m-%dT%H:%M:%S%z")"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    printf 'python=%s\n' "$($PYTHON_BIN --version 2>&1)"
    uname -a | sed 's/^/host=/'
  } > "$RESULT_DIR/run_environment.txt"
fi

if (( DRY_RUN == 0 )) && [[ " $METHODS " == *" mcp "* || " $METHODS " == *" restricted_mcp "* ]]; then
  "$PYTHON_BIN" -c '
import sys
from cumg import format_solver_availability, solver_available

solver = sys.argv[1]
fallback = sys.argv[2]
names = [solver] + ([] if fallback.lower() == "none" else [fallback])
print(format_solver_availability(names))
if not solver_available(solver):
    raise SystemExit(f"Required solver is unavailable: {solver}")
' "$SOLVER" "$FALLBACK_SOLVER"
fi

# Keep every worker single-threaded so concurrent processes do not each create
# their own BLAS/JAX thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

export PYTHON_BIN RESULT_DIR LOG_DIR REPS SEED_BASE METHODS
export GAMMA ALPHA EPSILON EPSILON_SCR STOCHASTIC_REGRET_TOLERANCE
export MAX_CANDIDATES N_SCREEN_STARTS N_SUPPORT_STARTS
export SCREEN_MAXITER SUPPORT_MAXITER MAX_ITER CERTIFY_EVERY
export SOLVER FALLBACK_SOLVER RUN_TIMEOUT_SECONDS DRY_RUN

printf 'Launching scalability run for risks [%s] with %s workers; results=%s\n' \
  "$RISK_GRID" "$WORKERS" "$RESULT_DIR"

for risk in $RISK_GRID; do
  case "$risk" in
    msd|cvar) ;;
    *)
      echo "Unsupported risk: $risk" >&2
      exit 2
      ;;
  esac
  for K in $K_GRID; do
    require_positive_integer K "$K"
    for n in $N_GRID; do
      require_positive_integer n "$n"
      for ((rep = 0; rep < REPS; rep++)); do
        printf '%s %s %s %s\n' "$risk" "$K" "$n" "$rep"
      done
    done
  done
done | xargs -n 4 -P "$WORKERS" bash -c '
  set -euo pipefail

  risk="$1"
  K="$2"
  n="$3"
  rep="$4"
  rep_stop=$((rep + 1))
  stem=$(printf "%s_K%s_n%s_rep%03d" "$risk" "$K" "$n" "$rep")
  out="$RESULT_DIR/${stem}.csv"
  lock_dir="$RESULT_DIR/.${stem}.lock"

  if [[ -s "$out" ]]; then
    line_count=$(wc -l < "$out")
    if (( line_count == 2 )); then
      echo "SKIP complete: $stem"
      exit 0
    fi
    if (( lƒine_count > 2 )); then
      echo "ERROR $out has $line_count lines; expected exactly 2" >&2
      exit 1
    fi
  fi

  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "SKIP locked: $stem" >&2
    exit 0
  fi
  trap '\''rmdir "$lock_dir" 2>/dev/null || true'\'' EXIT INT TERM

  attempt=0
  while :; do
    log=$(printf "%s/%s_attempt%03d.log" "$LOG_DIR" "$stem" "$attempt")
    [[ -e "$log" ]] || break
    attempt=$((attempt + 1))
  done

  read -r -a method_args <<< "$METHODS"
  command=(
    "$PYTHON_BIN" experiments/compare_scalability_approaches.py
    --risk "$risk"
    --K "$K"
    --n "$n"
    --reps "$REPS"
    --rep-start "$rep"
    --rep-stop "$rep_stop"
    --seed-base "$SEED_BASE"
    --gamma "$GAMMA"
    --alpha "$ALPHA"
    --epsilon "$EPSILON"
    --stochastic-regret-tolerance "$STOCHASTIC_REGRET_TOLERANCE"
    --max-candidates "$MAX_CANDIDATES"
    --n-screen-starts "$N_SCREEN_STARTS"
    --n-support-starts "$N_SUPPORT_STARTS"
    --screen-maxiter "$SCREEN_MAXITER"
    --support-maxiter "$SUPPORT_MAXITER"
    --max-iter "$MAX_ITER"
    --certify-every "$CERTIFY_EVERY"
    --solver "$SOLVER"
    --fallback-solver "$FALLBACK_SOLVER"
    --methods "${method_args[@]}"
    --csv "$out"
    --quiet
  )
  if [[ -n "$EPSILON_SCR" ]]; then
    command+=(--epsilon-scr "$EPSILON_SCR")
  fi

  if (( DRY_RUN != 0 )); then
    printf "DRY RUN:"
    printf " %q" "${command[@]}"
    printf "\n"
    exit 0
  fi

  echo "START $(date "+%Y-%m-%dT%H:%M:%S%z") risk=$risk K=$K n=$n rep=$rep" | tee "$log"
  set +e
  if (( RUN_TIMEOUT_SECONDS > 0 )); then
    timeout --signal=TERM --kill-after=60s "$RUN_TIMEOUT_SECONDS" \
      "${command[@]}" >> "$log" 2>&1
  else
    "${command[@]}" >> "$log" 2>&1
  fi
  status=$?
  set -e
  echo "END $(date "+%Y-%m-%dT%H:%M:%S%z") status=$status risk=$risk K=$K n=$n rep=$rep" \
    | tee -a "$log"
  exit "$status"
' _
