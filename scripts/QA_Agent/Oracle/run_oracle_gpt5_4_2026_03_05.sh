#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

MODEL_NAME="gpt-5.4-2026-03-05"
MODEL_TAG="gpt5_4_2026_03_05"
ANSWER_REASONING_EFFORT="${ANSWER_REASONING_EFFORT:-medium}"

if [[ "${ANSWER_REASONING_EFFORT}" == "none" ]]; then
  RUN_TAG="${MODEL_TAG}_no_reasoning_effort"
else
  RUN_TAG="${MODEL_TAG}_reasoning_${ANSWER_REASONING_EFFORT}"
fi

ATM_PREDICTIONS="output/QA_Agent/Oracle/${RUN_TAG}/atmbench/oracle_${RUN_TAG}.jsonl"
ATM_EVAL_DIR="output/QA_Agent/Oracle/${RUN_TAG}/atmbench/eval"
HARD_PREDICTIONS="output/QA_Agent/Oracle/${RUN_TAG}/hard/oracle_${RUN_TAG}.jsonl"
HARD_EVAL_DIR="output/QA_Agent/Oracle/${RUN_TAG}/hard/eval"

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model "${MODEL_NAME}" \
  --reasoning-effort "${ANSWER_REASONING_EFFORT}" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "${ATM_PREDICTIONS}"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "${ATM_PREDICTIONS}" \
  --output-dir "${ATM_EVAL_DIR}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench-hard.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model "${MODEL_NAME}" \
  --reasoning-effort "${ANSWER_REASONING_EFFORT}" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "${HARD_PREDICTIONS}"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "${HARD_PREDICTIONS}" \
  --output-dir "${HARD_EVAL_DIR}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
