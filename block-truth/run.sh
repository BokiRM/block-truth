#!/usr/bin/env bash
# Start block-truth. Seeds a database first if there is not one already.
#
#   ./run.sh                 serve on 8100
#   ./run.sh 9000            serve on 9000
#   ./run.sh --reseed        throw the database away and build a fresh one
#
# Nothing here reaches the network: the database is built locally.
set -euo pipefail
cd "$(dirname "$0")"

PORT=8100
RESEED=0
DAYS=21
for arg in "$@"; do
  case "$arg" in
    --reseed) RESEED=1 ;;
    --days=*) DAYS="${arg#*=}" ;;
    ''|*[!0-9]*) echo "unknown argument: $arg" >&2; exit 2 ;;
    *) PORT="$arg" ;;
  esac
done

DB="${BLOCK_TRUTH_DB:-data/blocks.db}"

python3 - <<'PY' || { echo; echo "Install the dependencies first:  pip install -r requirements.txt" >&2; exit 1; }
import importlib.util, sys
missing = [m for m in ("fastapi", "uvicorn", "openpyxl", "multipart")
           if importlib.util.find_spec(m) is None]
if missing:
    names = {"multipart": "python-multipart"}
    print("missing: " + ", ".join(names.get(m, m) for m in missing), file=sys.stderr)
    sys.exit(1)
PY

if [ "$RESEED" = "1" ]; then
  rm -f "$DB"
fi

if [ ! -f "$DB" ]; then
  echo "No database at $DB - building one ($DAYS days of recorded rounds)."
  BLOCK_TRUTH_DB="$DB" python3 -m block_truth.seed --db "$DB" --days "$DAYS"
  echo "Planning the month ahead from the fleet and volumes in that history."
  BLOCK_TRUTH_DB="$DB" python3 -m block_truth.schedule --db "$DB" --days 30
fi

echo
echo "  block-truth   http://127.0.0.1:$PORT"
echo "  api docs      http://127.0.0.1:$PORT/api/docs"
echo "  database      $DB"
echo
exec python3 -m uvicorn block_truth.api:app --host 127.0.0.1 --port "$PORT"
