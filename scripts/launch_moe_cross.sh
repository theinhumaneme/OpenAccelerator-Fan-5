#!/usr/bin/env bash
# Config: moe_cross_family
# Draft  : Qwen/Qwen2.5-3B-Instruct          (same draft as dense_cross_family)
# Verifier: mistralai/Mixtral-8x7B-Instruct-v0.1  (MoE, ~13B active params)
# Purpose: compare directly against dense_cross_family to isolate architecture effect.
#          Same draft model, only the verifier architecture changes.
set -euo pipefail

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --speculative-model Qwen/Qwen2.5-3B-Instruct \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
