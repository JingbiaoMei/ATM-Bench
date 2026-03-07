#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

GT_FILE="./data/atm-bench/atm-bench-hard.json"
NIAH_DIR="./data/atm-bench/niah"

IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
EMAIL_FILE="./data/raw_memory/email/merged_emails.json"

ANSWERER_MODEL="gpt-5"
JUDGE_MODEL="gpt-5-mini"

for k in 25 50 100 200; do
  NIAH_FILE="${NIAH_DIR}/atm-bench-hard-niah${k}.json"
  OUT_DIR="output/QA_Agent/NIAH/hard/gpt5/batch_results/niah${k}"
  PRED_FILE="${OUT_DIR}/niah_answers.jsonl"

  python memqa/qa_agent_baselines/NIAH/niah_evaluate.py \
    --qa-file "${NIAH_FILE}" \
    --media-source batch_results \
    --image-batch-results "${IMAGE_BATCH}" \
    --video-batch-results "${VIDEO_BATCH}" \
    --email-file "${EMAIL_FILE}" \
    --provider openai \
    --model "${ANSWERER_MODEL}" \
    --max-workers 8 \
    --timeout 120 \
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
