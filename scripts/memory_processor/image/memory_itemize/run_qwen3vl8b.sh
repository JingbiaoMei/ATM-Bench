#!/bin/bash
set -euo pipefail

python memqa/mem_processor/image/batch_processor.py "./data/raw_memory/image" \
  --output_dir "output/image/qwen3vl8b" \
  --provider "vllm" \
  --vllm-endpoint "http://localhost:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-concurrent 500 \
  --chunk-size 1000 \
  --timeout 1000
