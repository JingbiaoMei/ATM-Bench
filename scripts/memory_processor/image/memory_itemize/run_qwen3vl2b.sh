#!/bin/bash
set -euo pipefail

OUTPUT_DIR="output/image/qwen3vl2b"

python memqa/mem_processor/image/batch_processor.py "./data/raw_memory/image" \
  --output_dir "${OUTPUT_DIR}" \
  --provider "vllm" \
  --vllm-endpoint "http://localhost:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --max-concurrent 512 \
  --chunk-size 2048 \
  --timeout 1200
