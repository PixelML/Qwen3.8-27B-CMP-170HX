#!/usr/bin/env bash
# run_matrix.sh — fail-safe orchestration for the Qwen3.8-27B Q8 PCIe scaling
# baseline (experiments/qwen38-q8-pcie-baseline).
#
# Contract: one invocation handles ONE card-count argument (1, 2, or 4) with
# even tensor parallelism, a single pinned OpenAI-compatible vLLM server, no
# speculative decoding, and fixed client sampling. Topologies are run
# strictly sequentially (one invocation at a time; ascending card count) —
# see PLAN.md. Raw run output stays in the local (uncommitted) run
# directory; only sanitized receipts are ever copied into receipts/.
#
# Usage:
#   OPERATOR_LEASE_ACK=exclusive-node-lease-acknowledged \
#   MODEL_DIR=/path/to/model/on/canonical/shared/model/storage \
#   ./run_matrix.sh 1
#
# Exit codes:
#   0 success   2 preflight failure   3 server start failure
#   4 benchmark failed (classified, receipts preserved)
#   5 safety stop (watchdog threshold tripped, receipts preserved)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Fixed contract constants (must match PLAN.md / MANIFEST.md — do not tune)
# ---------------------------------------------------------------------------
readonly CONTRACT_SCHEMA="qwen38-q8-pcie-baseline.v1"
readonly SERVED_MODEL_NAME="qwen3.8-27b-q8"
readonly MAX_MODEL_LEN=36864          # 32768-token bucket + output + margin
readonly MAX_NUM_SEQS=1               # single stream
readonly GPU_MEMORY_UTILIZATION=0.90
readonly SERVER_SEED=1234
readonly POWER_POLICY_W=180           # benchmark power policy for the round
readonly GPU_MEMORY_CLASS_MIB=60000   # 64 GiB-class card check
readonly LEASE_ACK_TOKEN="exclusive-node-lease-acknowledged"
readonly HEALTH_TIMEOUT_S=1800        # 4-card weight load over PCIe is slow
readonly HEALTH_POLL_S=10
readonly SETTLE_S=20                  # settle delay after health gate
readonly WATCHDOG_INTERVAL_S=10
readonly WATCHDOG_STRIKES=3
readonly TEMP_STOP_C=88
readonly POWER_STOP_W=200             # sustained; transients below this are known

# ---------------------------------------------------------------------------
# Operator-tunable environment (documented in METHODOLOGY.md)
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-}"
MODEL_STORAGE_ROOT="${MODEL_STORAGE_ROOT:-/library/models}"  # canonical shared model storage
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs-local}"
BENCH_PORT="${BENCH_PORT:-18400}"
BENCH_API_KEY="${BENCH_API_KEY:-}"    # generated per run when unset; never logged or committed
GPU_CLASS_OVERRIDE="${GPU_CLASS_OVERRIDE:-0}"
POWER_POLICY_OVERRIDE="${POWER_POLICY_OVERRIDE:-0}"

CARDS="${1:-}"
die() { printf '[run_matrix] FATAL: %s\n' "$1" >&2; exit "${2:-2}"; }
info() { printf '[run_matrix] %s\n' "$1" >&2; }

[[ "$CARDS" =~ ^([124])$ ]] || die "usage: run_matrix.sh {1|2|4}  (got: '${CARDS}')"
CARDS="$BASH_REMATCH"

# ---------------------------------------------------------------------------
# Preflight 1/5 — operator lease acknowledgement (explicit, not implicit)
# ---------------------------------------------------------------------------
[[ "${OPERATOR_LEASE_ACK:-}" == "$LEASE_ACK_TOKEN" ]] || die \
  "operator lease not acknowledged: export OPERATOR_LEASE_ACK=${LEASE_ACK_TOKEN} to confirm exclusive use of the node for this sequential run"

[[ -n "$MODEL_DIR" ]] || die "MODEL_DIR is required (path under canonical shared model storage)"

# ---------------------------------------------------------------------------
# Preflight 2/5 — required GPU visibility, class, and power policy
# ---------------------------------------------------------------------------
for dep in nvidia-smi curl python3 df awk; do
  command -v "$dep" >/dev/null 2>&1 || die "required tool missing: $dep"
done

GPU_TABLE="$(nvidia-smi --query-gpu=index,name,memory.total,power.limit,driver_version \
  --format=csv,noheader,nounits)" || die "nvidia-smi query failed"
GPU_COUNT="$(printf '%s\n' "$GPU_TABLE" | wc -l)"
[[ "$GPU_COUNT" -ge "$CARDS" ]] || die \
  "GPU visibility: need ${CARDS} visible GPUs for this topology, found ${GPU_COUNT}"

GPU_INDICES=()
while IFS= read -r row; do
  idx="$(printf '%s' "$row" | cut -d, -f1 | tr -d ' ')"
  mem="$(printf '%s' "$row" | cut -d, -f3 | tr -d ' ')"
  plimit="$(printf '%s' "$row" | cut -d, -f4 | tr -d ' ')"
  if [[ "$mem" =~ ^[0-9]+$ ]]; then
    if (( mem < GPU_MEMORY_CLASS_MIB )); then
      if [[ "$GPU_CLASS_OVERRIDE" == "1" ]]; then
        info "WARNING: GPU ${idx} reports ${mem} MiB (< ${GPU_MEMORY_CLASS_MIB}); override active, recording deviation"
      else
        die "GPU ${idx} reports ${mem} MiB total — expected ${GPU_MEMORY_CLASS_MIB}+ MiB (64 GiB class); refusing (GPU_CLASS_OVERRIDE=1 to deviate)"
      fi
    fi
  else
    die "GPU ${idx} reported a non-numeric memory total ('${mem}') — cannot verify the 64 GiB class; refusing (GPU_CLASS_OVERRIDE=1 to deviate)"
  fi
  if [[ "$plimit" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    if awk -v p="$plimit" -v cap="$POWER_POLICY_W" 'BEGIN{exit !(p > cap + 1)}'; then
      if [[ "$POWER_POLICY_OVERRIDE" == "1" ]]; then
        info "WARNING: GPU ${idx} power limit ${plimit} W exceeds the ${POWER_POLICY_W} W policy; override active, recording deviation"
      else
        die "GPU ${idx} power limit is ${plimit} W — the ${POWER_POLICY_W} W benchmark power policy is not applied; refusing (POWER_POLICY_OVERRIDE=1 to deviate)"
      fi
    fi
  else
    info "WARNING: GPU ${idx} power limit unreadable ('${plimit}') — power-policy preflight skipped for this card"
  fi
  GPU_INDICES+=("$idx")
done < <(printf '%s\n' "$GPU_TABLE" | head -n "$CARDS")

CUDA_VISIBLE_DEVICES_LIST="$(IFS=,; echo "${GPU_INDICES[*]}")"
info "topology cards=${CARDS} tp=${CARDS} gpu_indices=${CUDA_VISIBLE_DEVICES_LIST}"

# ---------------------------------------------------------------------------
# Preflight 3/5 — no active compute processes on any visible GPU
# ---------------------------------------------------------------------------
COMPUTE_APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${COMPUTE_APPS//[[:space:]]/}" ]]; then
  printf '[run_matrix] FATAL: active compute processes present; refusing to start:\n%s\n' "$COMPUTE_APPS" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Preflight 4/5 — canonical model storage mounted/readable + free space
# ---------------------------------------------------------------------------
[[ -d "$MODEL_STORAGE_ROOT" && -r "$MODEL_STORAGE_ROOT" ]] || die \
  "canonical model storage root not mounted/readable: ${MODEL_STORAGE_ROOT}"
[[ -d "$MODEL_DIR" && -r "$MODEL_DIR" ]] || die \
  "model directory not present/readable: ${MODEL_DIR}"
[[ -n "$(ls -A "$MODEL_DIR" 2>/dev/null | head -n 1)" ]] || die \
  "model directory is empty: ${MODEL_DIR}"

free_gib() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1048576}'; }
OUT_FREE="$(free_gib "$OUTPUT_ROOT")"
HOME_FREE="$(free_gib "${HOME:-/tmp}")"
STOR_FREE="$(free_gib "$MODEL_STORAGE_ROOT")"
awk -v v="$OUT_FREE" 'BEGIN{exit !(v < 20)}' && die "output root has only ${OUT_FREE} GiB free (need >= 20): ${OUTPUT_ROOT}"
awk -v v="$HOME_FREE" 'BEGIN{exit !(v < 10)}' && die "home/cache filesystem has only ${HOME_FREE} GiB free (need >= 10)"
awk -v v="$STOR_FREE" 'BEGIN{exit !(v < 5)}' && die "model storage has only ${STOR_FREE} GiB free (need >= 5): ${MODEL_STORAGE_ROOT}"

# ---------------------------------------------------------------------------
# Preflight 5/5 — port free, single-flight lock
# ---------------------------------------------------------------------------
if (exec 3<>"/dev/tcp/127.0.0.1/${BENCH_PORT}") 2>/dev/null; then
  exec 3>&- 3<&- || true
  die "port ${BENCH_PORT} already in use"
fi

LOCK_DIR="${OUTPUT_ROOT}/.lock"
mkdir -p "$OUTPUT_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another invocation holds the lock (${LOCK_DIR}); topologies run strictly sequentially"
fi
cleanup_lock() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
cleanup_key() { rm -f "$KEY_FILE" 2>/dev/null || true; }

# ---------------------------------------------------------------------------
# Run directory (local only — never committed; see receipts/README.md)
# ---------------------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="q8pcie-cards${CARDS}-${STAMP}"
RUN_DIR="${OUTPUT_ROOT}/cards${CARDS}/${STAMP}"
mkdir -p "$RUN_DIR"
info "run_id=${RUN_ID} run_dir=${RUN_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "schema=${CONTRACT_SCHEMA}"
  echo "cards=${CARDS}"
  echo "tensor_parallel=${CARDS}"
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_dir=${MODEL_DIR}"
  echo "model_storage_root=${MODEL_STORAGE_ROOT}"
  echo "served_model_name=${SERVED_MODEL_NAME}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  echo "server_seed=${SERVER_SEED}"
  echo "speculative_decoding=off"
  echo "power_policy_w=${POWER_POLICY_W}"
  echo "gpu_indices=${CUDA_VISIBLE_DEVICES_LIST}"
  printf '%s\n' "$GPU_TABLE" | sed 's/^/gpu_table: /'
  echo "health_timeout_s=${HEALTH_TIMEOUT_S}"
} >"$RUN_DIR/metadata.txt"

# API key: generated per run when unset; kept in a 0600 file outside receipts.
KEY_FILE="$(mktemp)"
chmod 600 "$KEY_FILE"
if [[ -n "$BENCH_API_KEY" ]]; then
  printf '%s' "$BENCH_API_KEY" >"$KEY_FILE"
else
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24 >"$KEY_FILE"
  else
    python3 -c 'import secrets; print(secrets.token_hex(24), end="")' >"$KEY_FILE"
  fi
fi
BENCH_API_KEY_VALUE="$(cat "$KEY_FILE")"

SERVER_PID=""
MONITOR_PID=""
WATCHDOG_PID=""
CLIENT_PID=""
SAFETY_STOP=0

shutdown_all() {
  # Clean shutdown, order: client (graceful, lets it finalize receipts),
  # watchdog, monitor, then the whole server process group.
  [[ -n "$CLIENT_PID" ]] && kill -INT "$CLIENT_PID" 2>/dev/null || true
  [[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
  if [[ -n "$SERVER_PID" ]]; then
    kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 -- -"$SERVER_PID" 2>/dev/null || true
  fi
  [[ -n "$CLIENT_PID" ]] && wait "$CLIENT_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true
  [[ -n "$WATCHDOG_PID" ]] && wait "$WATCHDOG_PID" 2>/dev/null || true
  # Verify the port actually closed.
  if (exec 3<>"/dev/tcp/127.0.0.1/${BENCH_PORT}") 2>/dev/null; then
    exec 3>&- 3<&- || true
    info "WARNING: port ${BENCH_PORT} still accepting after shutdown"
  fi
}
trap 'shutdown_all; cleanup_lock; cleanup_key' EXIT

# ---------------------------------------------------------------------------
# Launch one pinned OpenAI-compatible vLLM server for this topology
# ---------------------------------------------------------------------------
info "launching vLLM server (tp=${CARDS}) on port ${BENCH_PORT} ..."
(
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST"
  export HF_HOME="${MODEL_STORAGE_ROOT}/hf-cache"
  export VLLM_NO_USAGE_STATS=1
  export DO_NOT_TRACK=1
  # API key via environment, not argv, so it never appears in process listings.
  export VLLM_API_KEY="$BENCH_API_KEY_VALUE"
  exec vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$CARDS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --seed "$SERVER_SEED" \
    --port "$BENCH_PORT" \
    --distributed-executor-backend mp
  # Note: no speculative decoding flags of any kind — absence is the contract.
) >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
info "server_pid=${SERVER_PID}"

# ---------------------------------------------------------------------------
# Health gate
# ---------------------------------------------------------------------------
HEALTHY=0
for _ in $(seq 1 $((HEALTH_TIMEOUT_S / HEALTH_POLL_S))); do
  if curl -sf -m 3 -H "Authorization: Bearer ${BENCH_API_KEY_VALUE}" \
      "http://127.0.0.1:${BENCH_PORT}/health" >/dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep "$HEALTH_POLL_S"
done
if [[ "$HEALTHY" -ne 1 ]]; then
  info "server failed its health gate (see ${RUN_DIR}/server.log — local only, never committed)"
  exit 3
fi
info "server healthy; settling ${SETTLE_S}s"
sleep "$SETTLE_S"

# ---------------------------------------------------------------------------
# GPU monitor + safety watchdog (both fail-safe, both local-only output)
# ---------------------------------------------------------------------------
python3 "$SCRIPT_DIR/monitor_gpu.py" \
  --gpus "$CUDA_VISIBLE_DEVICES_LIST" \
  --interval 2.0 \
  --schema "$CONTRACT_SCHEMA" \
  --output "$RUN_DIR/monitor.jsonl" &
MONITOR_PID=$!

(
  # Watchdog: temp / sustained power / ECC growth → abort the round.
  # Returns non-zero from the subshell when a stop is tripped.
  declare -A ECC_BASE=()
  strikes=0
  ecc_available=1
  nvidia-smi -i "${GPU_INDICES[0]}" \
    --query-gpu=ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits >/dev/null 2>&1 || ecc_available=0
  if [[ "$ecc_available" -eq 0 ]]; then
    echo "ecc_watch=unavailable_on_this_stack" >"$RUN_DIR/watchdog.txt"
  fi
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    tripped=0
    for idx in "${GPU_INDICES[@]}"; do
      row="$(nvidia-smi -i "$idx" \
        --query-gpu=temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' | tr -d '[' | tr -d ']' || true)"
      temp="${row%%,*}"
      power="${row##*,}"
      if [[ "$temp" =~ ^[0-9]+([.][0-9]+)?$ ]] \
         && awk -v t="$temp" -v lim="$TEMP_STOP_C" 'BEGIN{exit !(t >= lim)}'; then
        echo "trip=temp gpu=${idx} value=${temp}C limit=${TEMP_STOP_C}C" >>"$RUN_DIR/watchdog.txt"
        tripped=1
      fi
      if [[ "$power" =~ ^[0-9]+([.][0-9]+)?$ ]] \
         && awk -v p="$power" -v lim="$POWER_STOP_W" 'BEGIN{exit !(p > lim)}'; then
        echo "trip=power gpu=${idx} value=${power}W limit=${POWER_STOP_W}W" >>"$RUN_DIR/watchdog.txt"
        tripped=1
      fi
      if [[ "$ecc_available" -eq 1 ]]; then
        ecc="$(nvidia-smi -i "$idx" \
          --query-gpu=ecc.errors.uncorrected.volatile.total \
          --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
        if [[ "$ecc" =~ ^[0-9]+$ ]]; then
          if [[ -n "${ECC_BASE[$idx]:-}" ]] && (( ecc > ECC_BASE[$idx] )); then
            echo "trip=ecc gpu=${idx} baseline=${ECC_BASE[$idx]} now=${ecc}" >>"$RUN_DIR/watchdog.txt"
            tripped=1
          fi
          [[ -n "${ECC_BASE[$idx]:-}" ]] || ECC_BASE[$idx]="$ecc"
        fi
      fi
    done
    if [[ "$tripped" -eq 1 ]]; then
      strikes=$((strikes + 1))
    else
      strikes=0
    fi
    if (( strikes >= WATCHDOG_STRIKES )); then
      echo "safety_stop=1 at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_DIR/watchdog.txt"
      touch "$RUN_DIR/SAFETY_STOP"
      exit 5
    fi
    sleep "$WATCHDOG_INTERVAL_S"
  done
  exit 0
) &
WATCHDOG_PID=$!

# ---------------------------------------------------------------------------
# Benchmark client — cold + 3 warm per bucket, sanitized JSONL receipts
# ---------------------------------------------------------------------------
info "starting benchmark client (cold + 3 warm per bucket)"
set +e
python3 "$SCRIPT_DIR/bench_client.py" \
  --base-url "http://127.0.0.1:${BENCH_PORT}" \
  --api-key "$BENCH_API_KEY_VALUE" \
  --card-count "$CARDS" \
  --run-id "$RUN_ID" \
  --schema "$CONTRACT_SCHEMA" \
  --warm-repetitions 3 \
  --output "$RUN_DIR/receipt.jsonl" &
CLIENT_PID=$!
wait "$CLIENT_PID"
CLIENT_RC=$?
set -e
CLIENT_PID=""

if [[ -f "$RUN_DIR/SAFETY_STOP" ]]; then
  SAFETY_STOP=1
fi

# ---------------------------------------------------------------------------
# Classified outcome — receipts are always preserved
# ---------------------------------------------------------------------------
echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_DIR/metadata.txt"
info "receipt (sanitized by construction): ${RUN_DIR}/receipt.jsonl"
info "monitor telemetry:                   ${RUN_DIR}/monitor.jsonl"
info "Before committing receipts, follow receipts/README.md and the"
info "publication checklist in METHODOLOGY.md."

if [[ "$SAFETY_STOP" -eq 1 ]]; then
  info "SAFETY STOP: watchdog threshold tripped (see ${RUN_DIR}/watchdog.txt)"
  exit 5
fi
if [[ "$CLIENT_RC" -ne 0 ]]; then
  info "benchmark client exited ${CLIENT_RC}; classified failure preserved in receipt.jsonl"
  exit 4
fi
info "topology cards=${CARDS} completed"
exit 0
