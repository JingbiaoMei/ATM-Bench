#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ANSWERER_MODEL="${ANSWERER_MODEL_8B}"
ANSWERER_TAG="qwen3vl8b_answerer"
METHOD_NAME="short_caption"
OUTPUT_DIR="${OUTPUT_BASE}/text_embed_rerank/qwen3emb_0.6b_qwen3rerank_0.6b/${ANSWERER_TAG}/${METHOD_NAME}"

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
    --stage retrieve \
    --qa-file "${QA_FILE}" \
    --media-source batch_results \
    --image-batch-results "${IMAGE_BATCH}" \
    --video-batch-results "${VIDEO_BATCH}" \
    --email-file "${EMAIL_FILE}" \
    --retriever text \
    --text-embedding-model "${MODEL_QWEN3_EMB_06B}" \
    --retriever-batch-size 8 \
    --reranker qwen3_reranker \
    --text-reranker-model "${MODEL_QWEN3_RERANK_06B}" \
    --reranker-batch-size 8 \
    --provider vllm \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --model "${ANSWERER_MODEL}" \
    --max-workers 4 \
    --timeout 1200 \
    --retrieval-max-k "${RETRIEVAL_MAX_K}" \
    --rerank-input-k "${RERANK_INPUT_K}" \
    --rerank-top-k "${RERANK_TOP_K}" \
    --output-dir-base "${OUTPUT_BASE}/text_embed_rerank/qwen3emb_0.6b_qwen3rerank_0.6b/${ANSWERER_TAG}" \
    --method-name "${METHOD_NAME}" \
    ${TEXT_AUGMENT_ARGS}

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
    --stage rerank \
    --qa-file "${QA_FILE}" \
    --media-source batch_results \
    --image-batch-results "${IMAGE_BATCH}" \
    --video-batch-results "${VIDEO_BATCH}" \
    --email-file "${EMAIL_FILE}" \
    --retriever text \
    --text-embedding-model "${MODEL_QWEN3_EMB_06B}" \
    --retriever-batch-size 8 \
    --reranker qwen3_reranker \
    --text-reranker-model "${MODEL_QWEN3_RERANK_06B}" \
    --reranker-batch-size 8 \
    --provider vllm \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --model "${ANSWERER_MODEL}" \
    --max-workers 4 \
    --timeout 1200 \
    --retrieval-max-k "${RETRIEVAL_MAX_K}" \
    --rerank-input-k "${RERANK_INPUT_K}" \
    --rerank-top-k "${RERANK_TOP_K}" \
    --output-dir-base "${OUTPUT_BASE}/text_embed_rerank/qwen3emb_0.6b_qwen3rerank_0.6b/${ANSWERER_TAG}" \
    --method-name "${METHOD_NAME}" \
    ${TEXT_AUGMENT_ARGS}

run_eval_dual \
    "${OUTPUT_DIR}/mmrag_answers.jsonl" \
    "${OUTPUT_DIR}/eval" \
    "${OUTPUT_DIR}/retrieval_recall_details.json"
