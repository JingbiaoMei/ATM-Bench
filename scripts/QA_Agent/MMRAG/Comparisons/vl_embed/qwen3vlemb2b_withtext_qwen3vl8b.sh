#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ANSWERER_MODEL="${ANSWERER_MODEL_8B}"
ANSWERER_TAG="qwen3vl8b_answerer"
METHOD_NAME="withtext"
OUTPUT_DIR="${OUTPUT_BASE}/vl_embed/qwen3vlemb_2b/${ANSWERER_TAG}/${METHOD_NAME}"

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
    --qa-file "${QA_FILE}" \
    --media-source raw \
    --image-root "${IMAGE_ROOT}" \
    --video-root "${VIDEO_ROOT}" \
    --image-batch-results "${IMAGE_BATCH}" \
    --video-batch-results "${VIDEO_BATCH}" \
    --email-file "${EMAIL_FILE}" \
    --retriever qwen3_vl_embedding \
    --vl-embedding-model "${MODEL_QWEN3_VL_EMB_2B}" \
    --retriever-batch-size 4 \
    --provider vllm \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --model "${ANSWERER_MODEL}" \
    --max-workers 4 \
    --timeout 1200 \
    --retrieval-top-k "${RETRIEVAL_TOP_K}" \
    --retrieval-max-k "${RETRIEVAL_MAX_K}" \
    --output-dir-base "${OUTPUT_BASE}/vl_embed/qwen3vlemb_2b/${ANSWERER_TAG}" \
    --method-name "${METHOD_NAME}" \
    --vl-text-augment \
    --include-id --include-timestamp --include-location --include-short-caption

run_eval_dual \
    "${OUTPUT_DIR}/mmrag_answers.jsonl" \
    "${OUTPUT_DIR}/eval" \
    "${OUTPUT_DIR}/retrieval_recall_details.json"
