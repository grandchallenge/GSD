#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

RUNNER_ROOT="${GCL_COLAB_RUNNER_ROOT:-/mnt/f/_codex/MATH/QUANTUM-TECHNOLOGIES}"
GDSUITE_DIR="${GDSUITE_DIR:-/home/jim/GDsuite}"
JOB="${1:-$ROOT/experiments/gsd-001/colab/jobs/wp01_olmo2_1b_crt_coarse20_t4.json}"

VENV_PY="${GCL_COLAB_PY:-/usr/bin/python3}"
COLAB_BIN="${GCL_COLAB_BIN:-$HOME/.local/bin/colab}"
COLAB_AUTH="${COLAB_AUTH:-oauth2}"
GH_BIN="${GCL_GH_BIN:-$(command -v gh || true)}"
STATUS_REPO="${GSD_STATUS_REPO:-grandchallenge/GSD}"
STATUS_ISSUE="${GSD_STATUS_ISSUE:-1}"

STATUS_TOOL="$ROOT/experiments/gsd-001/colab/update_run_status.py"
COMMENT_TOOL="$ROOT/experiments/gsd-001/colab/format_run_status_comment.py"

[[ -x "$VENV_PY" ]] || { echo "missing host python: $VENV_PY" >&2; exit 2; }
[[ -x "$COLAB_BIN" ]] || { echo "missing existing Colab CLI: $COLAB_BIN" >&2; exit 2; }
[[ -f "$JOB" ]] || { echo "missing job: $JOB" >&2; exit 2; }
[[ -d "$GDSUITE_DIR" ]] || { echo "missing locked GDsuite checkout: $GDSUITE_DIR" >&2; exit 2; }
[[ -f "$STATUS_TOOL" ]] || { echo "missing run-status tool: $STATUS_TOOL" >&2; exit 2; }
[[ -f "$COMMENT_TOOL" ]] || { echo "missing status-comment tool: $COMMENT_TOOL" >&2; exit 2; }

EXPERIMENT_ID="$("$VENV_PY" - "$JOB" <<'PY'
import json,sys
job=json.load(open(sys.argv[1],encoding="utf-8"))
print(job["experiment_id"])
PY
)"
REMOTE_TIMEOUT="$("$VENV_PY" - "$JOB" <<'PY'
import json,sys
job=json.load(open(sys.argv[1],encoding="utf-8"))
print(int(job["remote_timeout_seconds"]))
PY
)"
ACCELERATOR="$("$VENV_PY" - "$JOB" <<'PY'
import json,sys
job=json.load(open(sys.argv[1],encoding="utf-8"))
print(job["resource"]["accelerator"])
PY
)"
EXPECTED_REVISIONS="$("$VENV_PY" - "$JOB" <<'PY'
import json,sys
job=json.load(open(sys.argv[1],encoding="utf-8"))
print(len(job.get("revisions", [])))
PY
)"
SOURCE_COMMIT="$(git rev-parse HEAD)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SESSION="gcl-gsd-$RANDOM-$$"
LOCAL_RUN="$ROOT/experiments/gsd-001/runs/hosted/$EXPERIMENT_ID/$RUN_ID"
mkdir -p "$LOCAL_RUN"

PAYLOAD="$LOCAL_RUN/gcl_source.tar.gz"
MANIFEST="$LOCAL_RUN/gcl_manifest.json"
STATUS_FILE="$LOCAL_RUN/RUN_STATUS.json"
COMMENT_FILE="$LOCAL_RUN/github-terminal-comment.md"
NOTIFIED_MARKER="$LOCAL_RUN/.github-terminal-notified"
RESULT_FILE="$LOCAL_RUN/gcl_result.json"

cp "$JOB" "$LOCAL_RUN/gcl_job.json"

ALLOCATED=0
FINAL_STATE=""
FINAL_MESSAGE=""
REMOTE_RC=99

update_status() {
  local state="$1"
  local message="${2:-}"
  local exit_code="${3:-}"
  local args=(
    "$VENV_PY" "$STATUS_TOOL" "$STATUS_FILE"
    --state "$state"
    --experiment-id "$EXPERIMENT_ID"
    --run-id "$RUN_ID"
    --session "$SESSION"
    --accelerator "$ACCELERATOR"
    --source-commit "$SOURCE_COMMIT"
    --message "$message"
  )
  if [[ -n "$exit_code" ]]; then
    args+=(--exit-code "$exit_code")
  fi
  if [[ -f "$LOCAL_RUN/experiment_receipt.json" ]]; then
    args+=(--receipt "$LOCAL_RUN/experiment_receipt.json")
  fi
  if [[ -f "$RESULT_FILE" ]]; then
    args+=(--result "$RESULT_FILE")
  fi
  "${args[@]}" >/dev/null
}

post_terminal_status() {
  [[ -f "$NOTIFIED_MARKER" ]] && return 0
  if [[ -z "$GH_BIN" || ! -x "$GH_BIN" ]]; then
    echo "gh CLI unavailable; terminal notification not posted" >"$LOCAL_RUN/github-notification-error.txt"
    return 0
  fi
  "$VENV_PY" "$COMMENT_TOOL" "$STATUS_FILE" --output "$COMMENT_FILE"
  local attempt
  for attempt in 1 2 3; do
    if "$GH_BIN" issue comment "$STATUS_ISSUE" --repo "$STATUS_REPO" --body-file "$COMMENT_FILE"       >"$LOCAL_RUN/github-notification.txt" 2>"$LOCAL_RUN/github-notification-error.txt"; then
      touch "$NOTIFIED_MARKER"
      return 0
    fi
    sleep "$attempt"
  done
  return 0
}

cleanup() {
  local rc=$?
  set +e
  local stop_ok=1
  if [[ "$ALLOCATED" -eq 1 ]]; then
    "$COLAB_BIN" --auth="$COLAB_AUTH" log -s "$SESSION"       -o "$LOCAL_RUN/colab-execution.md" >"$LOCAL_RUN/colab-log-command.txt" 2>&1 || true
    if ! "$COLAB_BIN" --auth="$COLAB_AUTH" stop -s "$SESSION"       >"$LOCAL_RUN/colab-stop.txt" 2>&1; then
      stop_ok=0
    fi
  fi
  "$COLAB_BIN" --auth="$COLAB_AUTH" sessions     >"$LOCAL_RUN/colab-sessions-after.txt" 2>&1 || true

  if [[ "$stop_ok" -ne 1 ]]; then
    FINAL_STATE="RED"
    FINAL_MESSAGE="Scientific execution returned, but Colab session cleanup failed."
    rc=13
  elif [[ -z "$FINAL_STATE" ]]; then
    FINAL_STATE="RED"
    FINAL_MESSAGE="Host launcher exited before verified completion."
  fi

  update_status "$FINAL_STATE" "$FINAL_MESSAGE" "$rc" || true
  post_terminal_status || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

update_status "QUEUED" "Frozen job accepted by host launcher."

"$VENV_PY" experiments/gsd-001/colab/build_payload.py   --upstream-dir "$GDSUITE_DIR"   --output "$PAYLOAD"   --manifest "$MANIFEST"

update_status "ALLOCATING" "Deterministic payload built; requesting exact accelerator."

echo "[GSD] allocating session=$SESSION accelerator=$ACCELERATOR"
"$COLAB_BIN" --auth="$COLAB_AUTH" new -s "$SESSION" --gpu "$ACCELERATOR"
ALLOCATED=1

"$COLAB_BIN" --auth="$COLAB_AUTH" status -s "$SESSION" >"$LOCAL_RUN/colab-status.txt"
update_status "RUNNING" "Colab session ready; payload upload and remote evaluation in progress."

"$COLAB_BIN" --auth="$COLAB_AUTH" upload -s "$SESSION" "$PAYLOAD" /content/gcl_source.tar.gz
"$COLAB_BIN" --auth="$COLAB_AUTH" upload -s "$SESSION" "$LOCAL_RUN/gcl_job.json" /content/gcl_job.json
"$COLAB_BIN" --auth="$COLAB_AUTH" upload -s "$SESSION" "$MANIFEST" /content/gcl_manifest.json

set +e
"$COLAB_BIN" --auth="$COLAB_AUTH" exec -s "$SESSION"   -f "$ROOT/experiments/gsd-001/colab/gsd_remote_job.py"   --timeout "$REMOTE_TIMEOUT"   > >(tee "$LOCAL_RUN/remote-stdout.txt")   2> >(tee "$LOCAL_RUN/remote-stderr.txt" >&2)
REMOTE_RC=$?
set -e

update_status "VERIFYING" "Remote evaluation returned; downloading and verifying terminal artifacts."

"$COLAB_BIN" --auth="$COLAB_AUTH" download -s "$SESSION"   /content/experiment_receipt.json "$LOCAL_RUN/experiment_receipt.json"   >"$LOCAL_RUN/download-receipt.txt" 2>&1 || true
"$COLAB_BIN" --auth="$COLAB_AUTH" download -s "$SESSION"   /content/gcl_output_bundle.tar.gz "$LOCAL_RUN/gcl_output_bundle.tar.gz"   >"$LOCAL_RUN/download-bundle.txt" 2>&1 || true
"$COLAB_BIN" --auth="$COLAB_AUTH" download -s "$SESSION"   /content/gcl_result.json "$RESULT_FILE"   >"$LOCAL_RUN/download-result.txt" 2>&1 || true

[[ -f "$LOCAL_RUN/experiment_receipt.json" ]] || {
  echo "[GSD] missing experiment receipt; evidence retained at $LOCAL_RUN" >&2
  exit 11
}
[[ -f "$LOCAL_RUN/gcl_output_bundle.tar.gz" ]] || {
  echo "[GSD] missing output bundle; evidence retained at $LOCAL_RUN" >&2
  exit 12
}

"$VENV_PY" -   "$LOCAL_RUN/experiment_receipt.json"   "$MANIFEST"   "$LOCAL_RUN/gcl_job.json"   "$RESULT_FILE"   "$EXPECTED_REVISIONS" <<'PY'
import hashlib,json,sys
from pathlib import Path

receipt=json.load(open(sys.argv[1],encoding="utf-8"))
manifest=json.load(open(sys.argv[2],encoding="utf-8"))
job_path=Path(sys.argv[3])
result_path=Path(sys.argv[4])
expected=int(sys.argv[5])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("[GSD] receipt status:", receipt.get("status"))
print("[GSD] observed runtime:", receipt.get("runtime"))

if receipt.get("status") != "GREEN_ENGINEERING":
    raise SystemExit(10)
if receipt.get("source_commit") != manifest.get("source_commit"):
    raise SystemExit("receipt source commit mismatch")
if receipt.get("source_payload_sha256") != manifest.get("payload_sha256"):
    raise SystemExit("receipt source payload mismatch")
if receipt.get("job_sha256") != sha(job_path):
    raise SystemExit("receipt job digest mismatch")
if not result_path.exists():
    raise SystemExit("missing gcl_result.json for green run")
if receipt.get("result_sha256") != sha(result_path):
    raise SystemExit("receipt result digest mismatch")

result=json.loads(result_path.read_text(encoding="utf-8"))
for key in ("revision_count", "summary_count", "manifest_count"):
    if int(result.get(key, -1)) != expected:
        raise SystemExit(f"{key} mismatch: {result.get(key)} != {expected}")
PY

if [[ "$REMOTE_RC" -ne 0 ]]; then
  echo "[GSD] remote execution failed rc=$REMOTE_RC; evidence retained at $LOCAL_RUN" >&2
  exit "$REMOTE_RC"
fi

FINAL_STATE="GREEN"
FINAL_MESSAGE="Receipt, payload/job/result digests, summary count, and manifest count verified."
echo "[GSD] hosted job verified GREEN: $LOCAL_RUN"
