#!/usr/bin/env bash
# Convenience wrapper for running one experiment end-to-end.
# Detects Docker vs Podman and sets the correct GPU flags.
#
# Usage:
#   ./run.sh dense_same_family
#   ./run.sh moe_cross_family
#
# Prerequisites:
#   cp .env.example .env && vim .env    (set HF_TOKEN and model names)
set -euo pipefail

EXPERIMENT="${1:-dense_same_family}"

# ── runtime detection ──────────────────────────────────────────────────────────
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

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[run.sh] ERROR: .env not found. Run: cp .env.example .env && edit it" >&2
    exit 1
fi

# Verify GPU access
if [[ "$RUNTIME" == "docker" ]]; then
    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L \
        || { echo "[run.sh] ERROR: Docker GPU access failed (is nvidia-container-toolkit installed?)"; exit 1; }
else
    podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
        nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L \
        || { echo "[run.sh] ERROR: Podman GPU access failed (is CDI configured? Run: nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml)"; exit 1; }
fi

# ── run ───────────────────────────────────────────────────────────────────────
export EXPERIMENT="$EXPERIMENT"

# Source .env so model names are available for the compose file
set -o allexport; source .env; set +o allexport

echo "[run.sh] Starting experiment: $EXPERIMENT"
echo "[run.sh] Verifier: ${VERIFIER_MODEL:-google/gemma-2-27b-it}"
echo "[run.sh] Draft   : ${DRAFT_MODEL:-google/gemma-2-2b-it}"
echo ""

# Tear down any previous run (different model pair may have been loaded).
$COMPOSE down --remove-orphans 2>/dev/null || true

# Build benchmark image if needed, then start both services.
# vLLM health-check gates benchmark startup automatically.
$COMPOSE up --build --abort-on-container-exit --exit-code-from benchmark

echo ""
echo "[run.sh] Done. Results in ./results/${EXPERIMENT}.json"
echo "[run.sh] Run: python src/analyze.py   to generate charts"
