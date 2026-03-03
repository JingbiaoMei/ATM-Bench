#!/bin/bash
set -euo pipefail

python memqa/mem_processor/image/batch_processor.py "./data/raw_memory/image" \
  --output_dir "output/image/qwen3vl2b" \
  --provider "vllm" \
  --vllm-endpoint "http://localhost:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --max-concurrent 256 \
  --chunk-size 512 \
  --timeout 1200
