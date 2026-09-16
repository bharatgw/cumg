#!/usr/bin/env bash
set -euo pipefail

# Run locally after the remote campaign finishes:
#   bash experiments/sync_qptas_results.sh root@YOUR_SERVER_IP
# Optional overrides: VERSION, RESULT_PATH, REMOTE_REPO, LOCAL_RESULT_DIR, PYTHON_BIN.
if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 USER@HOST (or an SSH host alias)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="${VERSION:-sampled_1000_v1}"
RESULT_PATH="${RESULT_PATH:-experiments/results/remote/qptas_scalability/$VERSION}"
REMOTE_REPO="${REMOTE_REPO:-/root/cumg}"
LOCAL_RESULT_DIR="${LOCAL_RESULT_DIR:-$REPO_ROOT/$RESULT_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v "$PYTHON_BIN" >/dev/null
mkdir -p "$LOCAL_RESULT_DIR"
rsync -avzc --progress --exclude='*.partial.csv' --exclude='*.lock/' -- \
  "$1:${REMOTE_REPO%/}/$RESULT_PATH/" "${LOCAL_RESULT_DIR%/}/"

"$PYTHON_BIN" - "$LOCAL_RESULT_DIR/capped_method_results.csv" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["method"] == "qptas"]
if not rows:
    raise SystemExit(f"No QPTAS rows found in {path}")

print(f"\nDownloaded results: {path.parent}")
print("Risk    Successes / runs    Success rate")
for risk in ("msd", "cvar", "total"):
    group = rows if risk == "total" else [row for row in rows if row["risk"] == risk]
    # The collector writes Boolean fields as True/False. Completion alone is insufficient.
    successes = sum(row["status"] == "completed" and row["success"].strip().lower() == "true" for row in group)
    rate = f"{successes / len(group):.1%}" if group else "n/a"
    print(f"{risk.upper():<7} {successes:>4} / {len(group):<7}   {rate}")
statuses = Counter(row["status"] for row in rows)
print("Statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())))
PY
