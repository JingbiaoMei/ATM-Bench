#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_eval.sh"

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MEMORYOS_INDEX_LLM_MODEL="${MEMORYOS_INDEX_LLM_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
MEMORYOS_ANSWER_LLM_MODEL="${MEMORYOS_ANSWER_LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
MEMORYOS_EMBEDDING_MODEL="${MEMORYOS_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

MAX_WORKERS="${MAX_WORKERS:-16}"
TOP_K="${TOP_K:-10}"
TOP_K_SESSIONS="${TOP_K_SESSIONS:-50}"
EVAL_NO_UPDATE="${EVAL_NO_UPDATE:-1}"

# MemoryOS full-history benchmark mode.
# Note: this is not paper-default behavior, but is required for large memory banks
# to avoid aggressive summarization/eviction and degraded indexing performance.
FULL_HISTORY_MODE="${FULL_HISTORY_MODE:-1}"
FULL_HISTORY_MID_TERM_CAPACITY="${FULL_HISTORY_MID_TERM_CAPACITY:-20000}"
FULL_HISTORY_TOP_K_SESSIONS="${FULL_HISTORY_TOP_K_SESSIONS:-200}"
FULL_HISTORY_HEAT_THRESHOLD="${FULL_HISTORY_HEAT_THRESHOLD:-1000000000}"

QA_ATMBENCH="./data/atm-bench/atm-bench.json"
QA_HARD="./data/atm-bench/atm-bench-hard.json"

EMAIL_FILE="./data/raw_memory/email/merged_emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT="./data/raw_memory/image"
VIDEO_ROOT="./data/raw_memory/video"

METHOD_BASE="memoryos"
METHOD="${METHOD_BASE}"
FULL_HISTORY_ARGS=""
if [ "${FULL_HISTORY_MODE}" = "1" ]; then
  METHOD="${METHOD_BASE}_fullhistory"
  FULL_HISTORY_ARGS="--memoryos-full-history-mode \
    --memoryos-full-history-mid-term-capacity ${FULL_HISTORY_MID_TERM_CAPACITY} \
    --memoryos-full-history-top-k-sessions ${FULL_HISTORY_TOP_K_SESSIONS} \
    --memoryos-full-history-heat-threshold ${FULL_HISTORY_HEAT_THRESHOLD}"
fi

EVAL_NO_UPDATE_FLAG=""
if [ "${EVAL_NO_UPDATE}" = "1" ]; then
  EVAL_NO_UPDATE_FLAG="--memoryos-eval-no-update"
fi

OUTPUT_BASE="output/QA_Agent/MemoryOS/main_table/topk${TOP_K}"
ATM_DIR="${OUTPUT_BASE}/atmbench/${METHOD}"
HARD_DIR="${OUTPUT_BASE}/hard/${METHOD}"

echo "=============================================="
echo "MemoryOS (ATMBench)"
echo "  Embedding:  ${MEMORYOS_EMBEDDING_MODEL}"
echo "  Index LLM:  ${MEMORYOS_INDEX_LLM_MODEL}"
echo "  Answer LLM: ${MEMORYOS_ANSWER_LLM_MODEL}"
echo "  Retrieve:   topk=${TOP_K} sessions=${TOP_K_SESSIONS}"
echo "  Full hist:  ${FULL_HISTORY_MODE} (mtm>=${FULL_HISTORY_MID_TERM_CAPACITY}, sessions>=${FULL_HISTORY_TOP_K_SESSIONS})"
echo "  Eval noop:  ${EVAL_NO_UPDATE}"
echo "  Endpoint:   ${VLLM_ENDPOINT}"
echo "=============================================="

python memqa/qa_agent_baselines/MemoryOS/memoryos_baseline.py \
  --qa-file "${QA_ATMBENCH}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  ${EVAL_NO_UPDATE_FLAG} \
  ${FULL_HISTORY_ARGS} \
  --memoryos-llm-model "${MEMORYOS_INDEX_LLM_MODEL}" \
  --memoryos-index-llm-model "${MEMORYOS_INDEX_LLM_MODEL}" \
  --memoryos-answer-llm-model "${MEMORYOS_ANSWER_LLM_MODEL}" \
  --memoryos-embedding-model "${MEMORYOS_EMBEDDING_MODEL}" \
  --memoryos-retrieval-queue-capacity "${TOP_K}" \
  --memoryos-top-k-sessions "${TOP_K_SESSIONS}" \
  --memoryos-use-per-qa-instance \
  --memoryos-answer-prompt-style baseline \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}/atmbench" \
  --method-name "${METHOD}"

run_eval_bundle \
  "${QA_ATMBENCH}" \
  "${ATM_DIR}/memoryos_answers.jsonl" \
  "${ATM_DIR}/eval" \
  "${ATM_DIR}/retrieval_recall_details.json"

python memqa/qa_agent_baselines/MemoryOS/memoryos_baseline.py \
  --qa-file "${QA_HARD}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  ${EVAL_NO_UPDATE_FLAG} \
  ${FULL_HISTORY_ARGS} \
  --memoryos-llm-model "${MEMORYOS_INDEX_LLM_MODEL}" \
  --memoryos-index-llm-model "${MEMORYOS_INDEX_LLM_MODEL}" \
  --memoryos-answer-llm-model "${MEMORYOS_ANSWER_LLM_MODEL}" \
  --memoryos-embedding-model "${MEMORYOS_EMBEDDING_MODEL}" \
  --memoryos-retrieval-queue-capacity "${TOP_K}" \
  --memoryos-top-k-sessions "${TOP_K_SESSIONS}" \
  --memoryos-use-per-qa-instance \
  --memoryos-answer-prompt-style baseline \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}/hard" \
  --method-name "${METHOD}"

run_eval_bundle \
  "${QA_HARD}" \
  "${HARD_DIR}/memoryos_answers.jsonl" \
  "${HARD_DIR}/eval" \
  "${HARD_DIR}/retrieval_recall_details.json"
