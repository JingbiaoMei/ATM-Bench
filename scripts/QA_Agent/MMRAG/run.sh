#!/usr/bin/env bash

TOP_K="${TOP_K:-10}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
ANSWERER_MODEL="${ANSWERER_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
TEXT_EMBED_MODEL="${TEXT_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

EMAIL_FILE="./data/raw_memory/email/merged_emails.json"
IMAGE_BATCH="./data/raw_memory/image/batch_results.json"
VIDEO_BATCH="./data/raw_memory/video/batch_results.json"

OUTPUT_BASE="output/QA_Agent/MMRAG/main_table/topk${TOP_K}"

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --retriever sentence_transformer \
  --text-embedding-model "${TEXT_EMBED_MODEL}" \
  --retriever-batch-size 64 \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${ANSWERER_MODEL}" \
  --max-workers 32 \
  --retrieval-top-k "${TOP_K}" \
  --output-dir-base "${OUTPUT_BASE}/atmbench/text_embed/allminilm_l6/qwen3vl8b_answerer" \
  --method-name MMRAG

python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file "./data/atm-bench/atm-bench-hard.json" \
  --media-source batch_results \
  --image-batch-results "${IMAGE_BATCH}" \
  --video-batch-results "${VIDEO_BATCH}" \
  --email-file "${EMAIL_FILE}" \
  --retriever sentence_transformer \
  --text-embedding-model "${TEXT_EMBED_MODEL}" \
  --retriever-batch-size 64 \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${ANSWERER_MODEL}" \
  --max-workers 32 \
  --retrieval-top-k "${TOP_K}" \
  --output-dir-base "${OUTPUT_BASE}/hard/text_embed/allminilm_l6/qwen3vl8b_answerer" \
  --method-name MMRAG
