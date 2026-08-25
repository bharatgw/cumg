#!/usr/bin/env bash
set -euo pipefail

# Resume the original CVaR grid one method at a time.  Completed legacy method
# measurements are reused, measurements over the cap are classified as
# timeouts, and only unresolved method/replicate pairs are launched.
#
# Typical remote invocation after stopping the uncapped runner:
#   nohup ./experiments/run_cvar_scalability_capped_resume.sh \
#     > capped_24h_runner.log 2>&1 &

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
METHOD_TIME_LIMIT_SECONDS="${METHOD_TIME_LIMIT_SECONDS:-86400}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
DRY_RUN="${DRY_RUN:-0}"
VERSION="${VERSION:-capped_24h_v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
LEGACY_RESULT_DIR="${LEGACY_RESULT_DIR:-experiments/results/remote/cvar_scalability/v1}"
RESULT_DIR="${RESULT_DIR:-experiments/results/remote/cvar_scalability/$VERSION}"
LOG_DIR="${LOG_DIR:-$RESULT_DIR/logs}"
MANIFEST="${MANIFEST:-$RESULT_DIR/pending_methods.tsv}"
COLLECTED_CSV="${COLLECTED_CSV:-$RESULT_DIR/capped_method_results.csv}"
CONFIG_FILE="$RESULT_DIR/run_config.env"
RESUME_TOOL="$SCRIPT_DIR/capped_scalability_resume.py"

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
require_positive_integer METHOD_TIME_LIMIT_SECONDS "$METHOD_TIME_LIMIT_SECONDS"
require_nonnegative_integer RETRY_ERRORS "$RETRY_ERRORS"
require_nonnegative_integer DRY_RUN "$DRY_RUN"

if [[ ! -d "$LEGACY_RESULT_DIR" ]]; then
  echo "Legacy result directory not found: $LEGACY_RESULT_DIR" >&2
  exit 2
fi
if [[ ! -f "$RESUME_TOOL" ]]; then
  echo "Resume utility not found: $RESUME_TOOL" >&2
  exit 2
fi
if (( DRY_RUN == 0 )) && ! command -v timeout >/dev/null 2>&1; then
  echo "Method caps require GNU timeout (available by default on Ubuntu)." >&2
  exit 2
fi

# Do not compete with the old uncapped workers.  Their output files are empty
# until all methods finish, so concurrent capped jobs would duplicate work.
active_legacy_lock="$(find "$LEGACY_RESULT_DIR" -maxdepth 1 -type d -name '.cvar_*.lock' -print -quit)"
if [[ -n "$active_legacy_lock" ]]; then
  echo "An uncapped legacy worker lock is still present: $active_legacy_lock" >&2
  echo "Stop the old runner and its workers before starting the capped resume." >&2
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
  "METHOD_TIME_LIMIT_SECONDS=$METHOD_TIME_LIMIT_SECONDS" \
  "LEGACY_RESULT_DIR=$LEGACY_RESULT_DIR")"
if [[ -n "$EPSILON_SCR" ]]; then
  run_config+=$'\n'"EPSILON_SCR=$EPSILON_SCR"
fi

if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(cat "$CONFIG_FILE")" != "$run_config" ]]; then
    echo "Configuration differs from $CONFIG_FILE." >&2
    echo "Use a new VERSION or RESULT_DIR instead of mixing capped settings." >&2
    exit 2
  fi
else
  printf '%s\n' "$run_config" > "$CONFIG_FILE"
fi

read -r -a risk_args <<< "$RISK_GRID"
read -r -a K_args <<< "$K_GRID"
read -r -a n_args <<< "$N_GRID"
read -r -a method_args <<< "$METHODS"

common_plan_args=(
  --legacy-dir "$LEGACY_RESULT_DIR"
  --result-dir "$RESULT_DIR"
  --risk "${risk_args[@]}"
  --K "${K_args[@]}"
  --n "${n_args[@]}"
  --reps "$REPS"
  --seed-base "$SEED_BASE"
  --methods "${method_args[@]}"
  --time-limit-seconds "$METHOD_TIME_LIMIT_SECONDS"
)

plan_command=("$PYTHON_BIN" "$RESUME_TOOL" plan "${common_plan_args[@]}" --manifest "$MANIFEST")
if (( RETRY_ERRORS != 0 )); then
  plan_command+=(--retry-errors)
fi
"${plan_command[@]}"
"$PYTHON_BIN" "$RESUME_TOOL" collect "${common_plan_args[@]}" --output "$COLLECTED_CSV"

pending_count="$(wc -l < "$MANIFEST")"
printf 'Capped resume: %s unresolved method runs; %s workers; cap=%ss; results=%s\n' \
  "$pending_count" "$WORKERS" "$METHOD_TIME_LIMIT_SECONDS" "$RESULT_DIR"

if (( DRY_RUN != 0 )); then
  sed -n '1,40p' "$MANIFEST"
  exit 0
fi
if (( pending_count == 0 )); then
  exit 0
fi

if [[ " $METHODS " == *" mcp "* || " $METHODS " == *" restricted_mcp "* ]]; then
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

# Keep each worker single-threaded.  Parallelism is across independent methods.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

export PYTHON_BIN RESULT_DIR LOG_DIR REPS SEED_BASE
export GAMMA ALPHA EPSILON EPSILON_SCR STOCHASTIC_REGRET_TOLERANCE
export MAX_CANDIDATES N_SCREEN_STARTS N_SUPPORT_STARTS
export SCREEN_MAXITER SUPPORT_MAXITER MAX_ITER CERTIFY_EVERY
export SOLVER FALLBACK_SOLVER METHOD_TIME_LIMIT_SECONDS RESUME_TOOL

xargs -n 6 -P "$WORKERS" bash -c '
  set -euo pipefail

  risk="$1"
  K="$2"
  n="$3"
  rep="$4"
  method="$5"
  seed="$6"
  rep_stop=$((rep + 1))
  replicate_stem=$(printf "%s_K%s_n%s_rep%03d" "$risk" "$K" "$n" "$rep")
  method_stem="${replicate_stem}__${method}"
  shard_dir="$RESULT_DIR/method_shards/$method"
  out="$shard_dir/${method_stem}.csv"
  partial="$shard_dir/${method_stem}.partial.csv"
  status_marker="$shard_dir/${method_stem}.status.json"
  lock_dir="$shard_dir/.${method_stem}.lock"
  mkdir -p "$shard_dir"

  if [[ -s "$out" ]] && [[ "$(wc -l < "$out")" -eq 2 ]]; then
    echo "SKIP complete: $method_stem"
    exit 0
  fi
  if [[ -s "$status_marker" ]]; then
    echo "SKIP status recorded: $method_stem"
    exit 0
  fi
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "SKIP locked: $method_stem" >&2
    exit 0
  fi
  trap '\''rmdir "$lock_dir" 2>/dev/null || true'\'' EXIT INT TERM

  attempt=0
  while :; do
    log=$(printf "%s/%s_attempt%03d.log" "$LOG_DIR" "$method_stem" "$attempt")
    [[ -e "$log" ]] || break
    attempt=$((attempt + 1))
  done

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
    --methods "$method"
    --csv "$partial"
    --quiet
  )
  if [[ -n "$EPSILON_SCR" ]]; then
    command+=(--epsilon-scr "$EPSILON_SCR")
  fi

  started_epoch=$(date +%s)
  echo "START $(date "+%Y-%m-%dT%H:%M:%S%z") risk=$risk K=$K n=$n rep=$rep method=$method seed=$seed" \
    | tee "$log"
  set +e
  timeout --signal=TERM --kill-after=60s "$METHOD_TIME_LIMIT_SECONDS" \
    "${command[@]}" >> "$log" 2>&1
  status=$?
  set -e
  ended_epoch=$(date +%s)
  elapsed_s=$((ended_epoch - started_epoch))

  if (( status == 0 )); then
    line_count=0
    [[ -f "$partial" ]] && line_count=$(wc -l < "$partial")
    if (( line_count == 2 )); then
      mv "$partial" "$out"
      outcome="completed"
    else
      status=70
      outcome="error"
      "$PYTHON_BIN" "$RESUME_TOOL" record-status \
        --result-dir "$RESULT_DIR" --risk "$risk" --K "$K" --n "$n" --rep "$rep" \
        --method "$method" --seed-base "$SEED_BASE" --status error \
        --elapsed-s "$elapsed_s" --time-limit-seconds "$METHOD_TIME_LIMIT_SECONDS" \
        --exit-code "$status" --message "Process exited successfully but did not write one CSV row." \
        >> "$log" 2>&1
    fi
  elif (( status == 124 )); then
    outcome="timeout"
    "$PYTHON_BIN" "$RESUME_TOOL" record-status \
      --result-dir "$RESULT_DIR" --risk "$risk" --K "$K" --n "$n" --rep "$rep" \
      --method "$method" --seed-base "$SEED_BASE" --status timeout \
      --elapsed-s "$elapsed_s" --time-limit-seconds "$METHOD_TIME_LIMIT_SECONDS" \
      --exit-code "$status" --message "GNU timeout reached the per-method wall-clock cap." \
      >> "$log" 2>&1
  else
    outcome="error"
    "$PYTHON_BIN" "$RESUME_TOOL" record-status \
      --result-dir "$RESULT_DIR" --risk "$risk" --K "$K" --n "$n" --rep "$rep" \
      --method "$method" --seed-base "$SEED_BASE" --status error \
      --elapsed-s "$elapsed_s" --time-limit-seconds "$METHOD_TIME_LIMIT_SECONDS" \
      --exit-code "$status" --message "Method process exited nonzero; inspect its attempt log." \
      >> "$log" 2>&1
  fi

  echo "END $(date "+%Y-%m-%dT%H:%M:%S%z") status=$status outcome=$outcome risk=$risk K=$K n=$n rep=$rep method=$method" \
    | tee -a "$log"
' _ < "$MANIFEST"

"$PYTHON_BIN" "$RESUME_TOOL" collect "${common_plan_args[@]}" --output "$COLLECTED_CSV"
