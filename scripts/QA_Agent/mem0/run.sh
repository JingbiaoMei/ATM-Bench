#!/usr/bin/env bash

TOP_K="${TOP_K:-10}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
MEM0_LLM_BASE_URL="${MEM0_LLM_BASE_URL:-${VLLM_ENDPOINT%/chat/completions}}"
MEM0_INFER="${MEM0_INFER:-0}"
USE_CUSTOM_EXTRACTION_PROMPT="${USE_CUSTOM_EXTRACTION_PROMPT:-1}"

MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-local}"
MEM0_LOCAL_RETRIEVER="${MEM0_LOCAL_RETRIEVER:-sentence_transformer}"
TEXT_EMBEDDING_MODEL="${TEXT_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
MEM0_LOCAL_DEVICE="${MEM0_LOCAL_DEVICE:-cpu}"
MEM0_USER_ID="${MEM0_USER_ID:-atmbench}"

EMBED_TAG="${TEXT_EMBEDDING_MODEL//\//_}"
EMBED_TAG="${EMBED_TAG//-/_}"
METHOD="${METHOD:-mem0_${MEM0_LOCAL_RETRIEVER}_${EMBED_TAG}_infer${MEM0_INFER}}"
INDEX_BASE="${INDEX_BASE:-output/QA_Agent/Mem0/index_cache/${METHOD}}"

MEM0_COLLECTION_NAME="${MEM0_COLLECTION_NAME:-${METHOD}}"
MEM0_VECTOR_PATH="${MEM0_VECTOR_PATH:-${INDEX_BASE}/mem0_chroma}"
MEM0_HISTORY_DB_PATH="${MEM0_HISTORY_DB_PATH:-${INDEX_BASE}/mem0_history.db}"
MEM0_PROGRESS_PATH="${MEM0_PROGRESS_PATH:-${INDEX_BASE}/mem0_index_progress.jsonl}"

MEM0_INFER_FLAG="--no-mem0-infer"
if [ "${MEM0_INFER}" = "1" ]; then
  MEM0_INFER_FLAG=""
fi

CUSTOM_PROMPT_FLAG=""
if [ "${MEM0_INFER}" = "1" ] && [ "${USE_CUSTOM_EXTRACTION_PROMPT}" = "1" ]; then
  CUSTOM_PROMPT_FLAG="--use-custom-extraction-prompt"
fi

MAX_WORKERS="${MAX_WORKERS:-32}"

QA_ATMBENCH="./data/atm-bench/atm-bench.json"
QA_HARD="./data/atm-bench/atm-bench-hard.json"

EMAIL_FILE="./data/raw_memory/email/merged_emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT="./data/raw_memory/image"
VIDEO_ROOT="./data/raw_memory/video"

OUTPUT_BASE="output/QA_Agent/Mem0/main_table/topk${TOP_K}"

echo "=============================================="
echo "Mem0 (ATMBench)"
echo "  TOP_K:     ${TOP_K}"
echo "  Index:     ${MEM0_LOCAL_RETRIEVER} (${TEXT_EMBEDDING_MODEL})"
echo "  Infer:     ${MEM0_INFER} (custom_prompt=${USE_CUSTOM_EXTRACTION_PROMPT})"
echo "  Memory LLM:${MEM0_LLM_MODEL} (base=${MEM0_LLM_BASE_URL})"
echo "  Answer LLM:${MODEL}"
echo "  Endpoint:  ${VLLM_ENDPOINT}"
echo "  Method:    ${METHOD}"
echo "  Index dir: ${INDEX_BASE}"
echo "=============================================="

python memqa/qa_agent_baselines/mem0/mem0_baseline.py \
  --qa-file "${QA_ATMBENCH}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --mem0-user-id "${MEM0_USER_ID}" \
  --mem0-collection-name "${MEM0_COLLECTION_NAME}" \
  --mem0-vector-path "${MEM0_VECTOR_PATH}" \
  --mem0-history-db-path "${MEM0_HISTORY_DB_PATH}" \
  --mem0-progress-path "${MEM0_PROGRESS_PATH}" \
  --mem0-embedder-provider "${MEM0_EMBEDDER_PROVIDER}" \
  --mem0-local-retriever "${MEM0_LOCAL_RETRIEVER}" \
  --mem0-local-device "${MEM0_LOCAL_DEVICE}" \
  --text-embedding-model "${TEXT_EMBEDDING_MODEL}" \
  --no-include-caption \
  --no-include-ocr-text \
  --no-include-tags \
  --mem0-llm-model "${MEM0_LLM_MODEL}" \
  --mem0-llm-base-url "${MEM0_LLM_BASE_URL}" \
  ${CUSTOM_PROMPT_FLAG} \
  --mem0-top-k "${TOP_K}" \
  ${MEM0_INFER_FLAG} \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}/atmbench" \
  --method-name "${METHOD}"

python memqa/qa_agent_baselines/mem0/mem0_baseline.py \
  --qa-file "${QA_HARD}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --mem0-user-id "${MEM0_USER_ID}" \
  --mem0-collection-name "${MEM0_COLLECTION_NAME}" \
  --mem0-vector-path "${MEM0_VECTOR_PATH}" \
  --mem0-history-db-path "${MEM0_HISTORY_DB_PATH}" \
  --mem0-progress-path "${MEM0_PROGRESS_PATH}" \
  --mem0-embedder-provider "${MEM0_EMBEDDER_PROVIDER}" \
  --mem0-local-retriever "${MEM0_LOCAL_RETRIEVER}" \
  --mem0-local-device "${MEM0_LOCAL_DEVICE}" \
  --text-embedding-model "${TEXT_EMBEDDING_MODEL}" \
  --no-include-caption \
  --no-include-ocr-text \
  --no-include-tags \
  --mem0-llm-model "${MEM0_LLM_MODEL}" \
  --mem0-llm-base-url "${MEM0_LLM_BASE_URL}" \
  ${CUSTOM_PROMPT_FLAG} \
  --mem0-top-k "${TOP_K}" \
  ${MEM0_INFER_FLAG} \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}/hard" \
  --method-name "${METHOD}"
