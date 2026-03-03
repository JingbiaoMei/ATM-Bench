#!/bin/bash
set -euo pipefail

python memqa/mem_processor/video/batch_processor.py "./data/raw_memory/video" \
  --output_dir "output/video/qwen3vl8b" \
  --provider "vllm" \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-concurrent 2 \
  --chunk-size 50 \
  --timeout 1000
