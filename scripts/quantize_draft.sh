#!/usr/bin/env bash
# Quantize a draft model to INT4 (W4A16) using LLM Compressor.
# Usage: ./scripts/quantize_draft.sh [model_id] [output_dir]
#   model_id   : HuggingFace model ID (default: google/gemma-4-31B-it)
#   output_dir : where to save the quantized model (default: ./quantized-draft)
set -euo pipefail

MODEL_ID="${1:-google/gemma-4-31B-it}"
OUTPUT_DIR="${2:-./quantized-draft}"

pip install llmcompressor --quiet

python - <<EOF
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer

model = SparseAutoModelForCausalLM.from_pretrained("${MODEL_ID}", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("${MODEL_ID}")

recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe, output_dir="${OUTPUT_DIR}")
tokenizer.save_pretrained("${OUTPUT_DIR}")
print(f"Quantized model saved to ${OUTPUT_DIR}")
EOF
