#!/usr/bin/env bash
# Config: dense_same_family
# Draft  : google/gemma-2-2b-it   (same family as verifier)
# Verifier: google/gemma-2-27b-it  (dense)
# Expected: highest acceptance rate — both models share the same learned distribution.
set -euo pipefail

python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --speculative-model google/gemma-2-2b-it \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
