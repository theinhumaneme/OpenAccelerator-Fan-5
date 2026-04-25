#!/usr/bin/env bash
# Config: dense_cross_family
# Draft  : Qwen/Qwen2.5-3B-Instruct (different family from verifier)
# Verifier: google/gemma-2-27b-it    (dense)
# Purpose: isolates the family-mismatch penalty with architecture held constant.
set -euo pipefail

python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --speculative-model Qwen/Qwen2.5-3B-Instruct \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
