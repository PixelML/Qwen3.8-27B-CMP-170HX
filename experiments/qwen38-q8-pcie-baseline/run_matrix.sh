#!/usr/bin/env bash
# run_matrix.sh — fail-safe orchestration for the Qwen3.8-27B Q8 whole-node
# throughput baseline (experiments/qwen38-q8-pcie-baseline).
#
# Contract: one invocation handles ONE phase. "control" runs one short
# single-card TP1 server with the pinned client. "node4" launches four
# card-local TP1 servers simultaneously (one per physical card, identical
# settings) and measures them in one shared time window for whole-node
# aggregate throughput and fairness/interference evidence. No speculative
# decoding anywhere. The former 1/2/4 TP scaling sweep and TP4 tuning are
# out of scope (parked) — see PLAN.md. Raw run output stays in the local
# (uncommitted) run directory; only sanitized receipts are ever copied
# into receipts/.
#
# Usage:
#   1) export MODEL_DIR=<path under the canonical shared model storage>
#   2) export OPERATOR_LEASE_ACK=exclusive-node-lease-acknowledged
#   3) ./run_matrix.sh control   # then: ./run_matrix.sh node4
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

PHASE="${1:-}"
die() { printf '[run_matrix] FATAL: %s\n' "$1" >&2; exit "${2:-2}"; }
info() { printf '[run_matrix] %s\n' "$1" >&2; }

[[ "$PHASE" =~ ^(control|node4)$ ]] || die "usage: run_matrix.sh {control|node4}  (got: '${PHASE}')"

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
EXPECTED_CARDS=1
[[ "$PHASE" == "node4" ]] && EXPECTED_CARDS=4
[[ "$GPU_COUNT" -ge "$EXPECTED_CARDS" ]] || die \
  "GPU visibility: need ${EXPECTED_CARDS} visible GPUs for this phase, found ${GPU_COUNT}"

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
done < <(printf '%s\n' "$GPU_TABLE" | head -n "$EXPECTED_CARDS")

CUDA_VISIBLE_DEVICES_LIST="$(IFS=,; echo "${GPU_INDICES[*]}")"
info "phase=${PHASE} expected_cards=${EXPECTED_CARDS} gpu_indices=${CUDA_VISIBLE_DEVICES_LIST}"

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
  die "another invocation holds the lock (${LOCK_DIR}); phases run strictly sequentially"
fi
cleanup_lock() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
cleanup_key() { rm -f "$KEY_FILE" 2>/dev/null || true; }

# ---------------------------------------------------------------------------
# Run directory (local only — never committed; see receipts/README.md)
# ---------------------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$PHASE" == "node4" ]]; then
  RUN_ID="q8node4-${STAMP}"
  RUN_DIR="${OUTPUT_ROOT}/node4/${STAMP}"
else
  RUN_ID="q8control-${STAMP}"
  RUN_DIR="${OUTPUT_ROOT}/control/${STAMP}"
fi
mkdir -p "$RUN_DIR"
info "run_id=${RUN_ID} run_dir=${RUN_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "schema=${CONTRACT_SCHEMA}"
  echo "phase=${PHASE}"
  echo "tensor_parallel=1"
  if [[ "$PHASE" == "node4" ]]; then
    echo "workers=4 card_local=true"
  else
    echo "workers=1 card_local=control"
  fi
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

SERVER_PIDS=()
WORKER_PORTS=()
launch_worker() {
  local worker_id="$1" port="$2" gpu_index="$3" log_file="$4"
  info "launching card-local vLLM worker ${worker_id} on port ${port} (single card)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_index"
    export HF_HOME="${MODEL_STORAGE_ROOT}/hf-cache"
    export VLLM_NO_USAGE_STATS=1
    export DO_NOT_TRACK=1
    export VLLM_API_KEY="$BENCH_API_KEY_VALUE"
    exec vllm serve "$MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 1 \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --seed "$SERVER_SEED" \
      --port "$port" \
      --distributed-executor-backend mp
  ) >"$log_file" 2>&1 &
  SERVER_PIDS+=("$!")
  WORKER_PORTS+=("$port")
}
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
  [[ -n "$CLIENT_PID" ]] && wait "$CLIENT_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true
  [[ -n "$WATCHDOG_PID" ]] && wait "$WATCHDOG_PID" 2>/dev/null || true
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 -- -"$pid" 2>/dev/null || true
  done
  # Verify every worker port actually closed.
  for port in "${WORKER_PORTS[@]:-}"; do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      exec 3>&- 3<&- || true
      info "WARNING: port ${port} still accepting after shutdown"
    fi
  done
}
trap 'shutdown_all; cleanup_lock; cleanup_key' EXIT

if [[ "$PHASE" == "node4" ]]; then
  for worker_id in 1 2 3 4; do
    gpu_index="${GPU_INDICES[$((worker_id - 1))]}"
    port=$((NODE4_PORT_BASE + worker_id - 1))
    launch_worker "$worker_id" "$port" "$gpu_index" "$RUN_DIR/worker-${worker_id}.log"
  done
  urls=()
  for port in "${WORKER_PORTS[@]}"; do
    urls+=("http://127.0.0.1:${port}")
  done
  ALL_URLS="$(IFS=,; echo "${urls[*]}")"
else
  launch_worker 1 "$BENCH_PORT" "${GPU_INDICES[0]}" "$RUN_DIR/worker-1.log"
  ALL_URLS="http://127.0.0.1:${BENCH_PORT}"
fi
wait_all_healthy() {
  local total="${#SERVER_PIDS[@]}"
  local elapsed=0
  while (( elapsed < HEALTH_TIMEOUT_S )); do
    local healthy=0
    local i
    for i in "${!SERVER_PIDS[@]}"; do
      local pid="${SERVER_PIDS[$i]}"
      local port="${WORKER_PORTS[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        info "worker $((i + 1)) exited during startup (log: ${RUN_DIR}/worker-$((i + 1)).log, local only)"
        return 3
      fi
      if curl -sf -m 3 -H "Authorization: Bearer ${BENCH_API_KEY_VALUE}" "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        healthy=$((healthy + 1))
      fi
    done
    if (( healthy == total )); then
      return 0
    fi
    sleep "$HEALTH_POLL_S"
    elapsed=$((elapsed + HEALTH_POLL_S))
  done
  info "health gate timed out before all workers were healthy"
  return 3
}
if ! wait_all_healthy; then
  exit 3
fi
info "all workers healthy; settling ${SETTLE_S}s"
sleep "$SETTLE_S"


# ---------------------------------------------------------------------------


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
  server_alive=0
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      server_alive=1
      break
    fi
  done
  while [[ "$server_alive" -eq 1 ]]; do
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
    server_alive=0
    for pid in "${SERVER_PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        server_alive=1
        break
      fi
    done
  done
  exit 0
) &
WATCHDOG_PID=$!

# ---------------------------------------------------------------------------
# Benchmark client — cold + 3 warm per bucket, sanitized JSONL receipts
# ---------------------------------------------------------------------------
info "starting benchmark client (${PHASE})"
set +e
if [[ "$PHASE" == "node4" ]]; then
  python3 "$SCRIPT_DIR/bench_client.py" \
    --mode node4 --node4-urls "$ALL_URLS" \
    --api-key "$BENCH_API_KEY_VALUE" \
    --card-count 4 \
    --run-id "$RUN_ID" --schema "$CONTRACT_SCHEMA" \
    --warm-repetitions 3 --output "$RUN_DIR/receipt.jsonl" &
else
  python3 "$SCRIPT_DIR/bench_client.py" \
    --mode control --base-url "$ALL_URLS" \
    --api-key "$BENCH_API_KEY_VALUE" \
    --card-count 1 \
    --run-id "$RUN_ID" --schema "$CONTRACT_SCHEMA" \
    --warm-repetitions 3 --output "$RUN_DIR/receipt.jsonl" &
fi
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
info "phase=${PHASE} completed"
exit 0
