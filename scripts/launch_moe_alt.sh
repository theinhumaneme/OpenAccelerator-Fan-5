#!/usr/bin/env bash
# Config: moe_alt_draft
# Draft  : google/gemma-2-2b-it               (same draft as dense_same_family)
# Verifier: mistralai/Mixtral-8x7B-Instruct-v0.1  (MoE)
# Purpose: tests whether MoE acceptance rate is draft-family-dependent.
#          Pair with dense_same_family to see if Gemma draft fares better with
#          dense Gemma vs MoE Mixtral.
set -euo pipefail

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --speculative-model google/gemma-2-2b-it \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
