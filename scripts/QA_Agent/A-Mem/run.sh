#!/usr/bin/env bash

TOP_K="${TOP_K:-10}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

MEMORY_PROVIDER="${MEMORY_PROVIDER:-vllm}"
MEMORY_ENDPOINT="${MEMORY_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MEMORY_MODEL="${MEMORY_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
MEMORY_WORKERS="${MEMORY_WORKERS:-16}"

ANSWER_PROVIDER="${ANSWER_PROVIDER:-vllm}"
ANSWER_ENDPOINT="${ANSWER_ENDPOINT:-${MEMORY_ENDPOINT}}"
ANSWER_MODEL="${ANSWER_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
MAX_WORKERS="${MAX_WORKERS:-8}"

QA_ATMBENCH="./data/atm-bench/atm-bench.json"
QA_HARD="./data/atm-bench/atm-bench-hard.json"

EMAIL_FILE="./data/raw_memory/email/merged_emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT="./data/raw_memory/image"
VIDEO_ROOT="./data/raw_memory/video"

CACHE_DIR="output/QA_Agent/A-Mem/index_cache"
OUTPUT_BASE="output/QA_Agent/A-Mem/main_table/topk${TOP_K}"

echo "=============================================="
echo "A-Mem (ATMBench)"
echo "  TOP_K:      ${TOP_K}"
echo "  Embedding:  ${EMBEDDING_MODEL}"
echo "  Memory LLM: ${MEMORY_MODEL}"
echo "  Answer LLM: ${ANSWER_MODEL}"
echo "  Endpoint:   ${MEMORY_ENDPOINT}"
echo "=============================================="

python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage build \
  --qa-file "${QA_ATMBENCH}" \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --index-cache "${CACHE_DIR}" \
  --memory-provider "${MEMORY_PROVIDER}" \
  --memory-vllm-endpoint "${MEMORY_ENDPOINT}" \
  --memory-model "${MEMORY_MODEL}" \
  --memory-workers "${MEMORY_WORKERS}" \
  --timeout 1200 \
  --output-dir-base "${OUTPUT_BASE}"

python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage answer \
  --qa-file "${QA_ATMBENCH}" \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --index-cache "${CACHE_DIR}" \
  --memory-provider "${MEMORY_PROVIDER}" \
  --memory-vllm-endpoint "${MEMORY_ENDPOINT}" \
  --memory-model "${MEMORY_MODEL}" \
  --provider "${ANSWER_PROVIDER}" \
  --vllm-endpoint "${ANSWER_ENDPOINT}" \
  --model "${ANSWER_MODEL}" \
  --retrieve-k "${TOP_K}" \
  --follow-links \
  --max-workers "${MAX_WORKERS}" \
  --timeout 1200 \
  --output-dir-base "${OUTPUT_BASE}" \
  --method-name "atmbench/amem"

python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage answer \
  --qa-file "${QA_HARD}" \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --index-cache "${CACHE_DIR}" \
  --memory-provider "${MEMORY_PROVIDER}" \
  --memory-vllm-endpoint "${MEMORY_ENDPOINT}" \
  --memory-model "${MEMORY_MODEL}" \
  --provider "${ANSWER_PROVIDER}" \
  --vllm-endpoint "${ANSWER_ENDPOINT}" \
  --model "${ANSWER_MODEL}" \
  --retrieve-k "${TOP_K}" \
  --follow-links \
  --max-workers "${MAX_WORKERS}" \
  --timeout 1200 \
  --output-dir-base "${OUTPUT_BASE}" \
  --method-name "hard/amem"
