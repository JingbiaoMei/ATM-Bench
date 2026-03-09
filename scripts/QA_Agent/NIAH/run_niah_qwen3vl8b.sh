#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"

GT_FILE="./data/atm-bench/atm-bench-hard.json"
NIAH_DIR="./data/atm-bench/niah"

IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
EMAIL_FILE="./data/raw_memory/email/emails.json"

ANSWERER_MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"
JUDGE_MODEL="gpt-5-mini"

for k in 25 50 100 200; do
  NIAH_FILE="${NIAH_DIR}/atm-bench-hard-niah${k}.json"
  OUT_DIR="output/QA_Agent/NIAH/hard/qwen3vl8b/batch_results/niah${k}"
  PRED_FILE="${OUT_DIR}/niah_answers.jsonl"

  python memqa/qa_agent_baselines/NIAH/niah_evaluate.py \
    --qa-file "${NIAH_FILE}" \
    --media-source batch_results \
    --image-batch-results "${IMAGE_BATCH}" \
    --video-batch-results "${VIDEO_BATCH}" \
    --email-file "${EMAIL_FILE}" \
    --provider vllm \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --model "${ANSWERER_MODEL}" \
    --max-workers 1 \
    --timeout 1600 \
    --output-file "${PRED_FILE}"

  python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth "${GT_FILE}" \
    --predictions "${PRED_FILE}" \
    --output-dir "${OUT_DIR}/gpt5_mini" \
    --metrics em atm \
    --judge-provider openai \
    --judge-model "${JUDGE_MODEL}" \
    --judge-reasoning-effort minimal \
    --max-workers 2

done
