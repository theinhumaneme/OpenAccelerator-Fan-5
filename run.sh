#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/experiment.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-local}"
RUN_ID="${1:-}"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh --list-runs
  ./run.sh <run_id>
  ./run.sh                 # runs every expanded config, if VRAM validation passes

Modes:
  MODE=local   ./run.sh <run_id>   # benchmark.py starts/stops vLLM locally
  MODE=compose ./run.sh <run_id>   # Docker/Podman starts vLLM, benchmark uses --external-vllm

Environment:
  CONFIG=configs/experiment.yaml
  PYTHON_BIN=python3
EOF
}

if [[ "${RUN_ID}" == "-h" || "${RUN_ID}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${RUN_ID}" == "--list-runs" || "${RUN_ID}" == "list" ]]; then
    "$PYTHON_BIN" src/benchmark.py --config "$CONFIG" --skip-vram-validation --list-runs
    exit 0
fi

if [[ "$MODE" == "local" ]]; then
    args=(src/benchmark.py --config "$CONFIG")
    if [[ -n "$RUN_ID" ]]; then
        args+=(--run-id "$RUN_ID")
    fi
    exec "$PYTHON_BIN" "${args[@]}"
fi

if [[ "$MODE" != "compose" ]]; then
    echo "[run.sh] ERROR: MODE must be 'local' or 'compose'." >&2
    exit 1
fi

if [[ -z "$RUN_ID" ]]; then
    echo "[run.sh] ERROR: compose mode requires a run_id. Use ./run.sh --list-runs first." >&2
    exit 1
fi

if command -v podman &>/dev/null; then
    RUNTIME=podman
    COMPOSE="podman compose"
    echo "[run.sh] Using Podman"
elif command -v docker &>/dev/null; then
    RUNTIME=docker
    COMPOSE="docker compose"
    echo "[run.sh] Using Docker"
else
    echo "[run.sh] ERROR: neither docker nor podman found in PATH" >&2
    exit 1
fi

if [[ -f .env ]]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[run.sh] ERROR: HF_TOKEN is required for gated Gemma models. Set it in .env." >&2
    exit 1
fi

if [[ "$RUNTIME" == "docker" ]]; then
    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L \
        || { echo "[run.sh] ERROR: Docker GPU access failed."; exit 1; }
else
    podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
        nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L \
        || { echo "[run.sh] ERROR: Podman GPU access failed. Generate CDI with nvidia-ctk if needed."; exit 1; }
fi

set -o allexport
eval "$("$PYTHON_BIN" src/benchmark.py --config "$CONFIG" --skip-vram-validation --emit-env "$RUN_ID")"
set +o allexport

echo "[run.sh] Starting run: $RUN_ID"
echo "[run.sh] Verifier: $VERIFIER_MODEL"
if [[ -n "$SPECULATIVE_CONFIG" ]]; then
    echo "[run.sh] Speculative config: $SPECULATIVE_CONFIG"
else
    echo "[run.sh] Baseline run: speculation disabled"
fi

$COMPOSE down --remove-orphans 2>/dev/null || true
$COMPOSE up --build --abort-on-container-exit --exit-code-from benchmark
