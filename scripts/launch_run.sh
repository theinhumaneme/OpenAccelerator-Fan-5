#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/experiment.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${1:?Usage: scripts/launch_run.sh <run_id>}"

set -o allexport
eval "$("$PYTHON_BIN" src/benchmark.py --config "$CONFIG" --skip-vram-validation --emit-env "$RUN_ID")"
set +o allexport

args=(
  vllm serve "$VERIFIER_MODEL"
  --host 0.0.0.0
  --port "$VLLM_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$VERIFIER_DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --disable-log-requests
  --enable-metrics
)

if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

exec "${args[@]}"
