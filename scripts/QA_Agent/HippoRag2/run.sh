#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_eval.sh"

TOP_K="${TOP_K:-10}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"

ANSWERER_ENDPOINT="${ANSWERER_ENDPOINT:-${VLLM_ENDPOINT}}"
ANSWERER_MODEL="${ANSWERER_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
AUGMENTATION_LEVEL="${AUGMENTATION_LEVEL:-short_caption_only}"

MAX_WORKERS="${MAX_WORKERS:-32}"
OPENIE_WORKERS="${OPENIE_WORKERS:-64}"

QA_ATMBENCH="./data/atm-bench/atm-bench.json"
QA_HARD="./data/atm-bench/atm-bench-hard.json"

EMAIL_FILE="./data/raw_memory/email/emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT="./data/raw_memory/image"
VIDEO_ROOT="./data/raw_memory/video"

OUTPUT_BASE="output/QA_Agent/HippoRag2/main_table/topk${TOP_K}/text_embed/allminilm_l6"
ATM_DIR="${OUTPUT_BASE}/atmbench/hipporag2"
HARD_DIR="${OUTPUT_BASE}/hard/hipporag2"

echo "=============================================="
echo "HippoRAG 2 (ATMBench)"
echo "  TOP_K:         ${TOP_K}"
echo "  Embedding:     ${EMBEDDING_MODEL}"
echo "  Augment:       ${AUGMENTATION_LEVEL}"
echo "  Memory/Index:  ${MODEL}"
echo "  Answerer LLM:  ${ANSWERER_MODEL}"
echo "  Endpoint:      ${VLLM_ENDPOINT}"
echo "=============================================="

python memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py \
  --stage build \
  --qa-file "${QA_ATMBENCH}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --augmentation-level "${AUGMENTATION_LEVEL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --checkpoint-interval 50 \
  --openie-workers "${OPENIE_WORKERS}"

python memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py \
  --stage answer \
  --qa-file "${QA_ATMBENCH}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --augmentation-level "${AUGMENTATION_LEVEL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --answerer-endpoint "${ANSWERER_ENDPOINT}" \
  --answerer-model "${ANSWERER_MODEL}" \
  --checkpoint-interval 50 \
  --method-name "atmbench/hipporag2" \
  --qa-top-k "${TOP_K}" \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}"

run_eval_bundle \
  "${QA_ATMBENCH}" \
  "${ATM_DIR}/hipporag2_answers.jsonl" \
  "${ATM_DIR}/eval" \
  "${ATM_DIR}/retrieval_recall_details.json"

python memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py \
  --stage answer \
  --qa-file "${QA_HARD}" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --augmentation-level "${AUGMENTATION_LEVEL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --answerer-endpoint "${ANSWERER_ENDPOINT}" \
  --answerer-model "${ANSWERER_MODEL}" \
  --checkpoint-interval 50 \
  --method-name "hard/hipporag2" \
  --qa-top-k "${TOP_K}" \
  --max-workers "${MAX_WORKERS}" \
  --output-dir-base "${OUTPUT_BASE}"

run_eval_bundle \
  "${QA_HARD}" \
  "${HARD_DIR}/hipporag2_answers.jsonl" \
  "${HARD_DIR}/eval" \
  "${HARD_DIR}/retrieval_recall_details.json"
