#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"

# Descriptive Memory (D): ID + short_caption only

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source batch_results \
  --batch-fields "short_caption" \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/qwen3vl8b_D/atmbench/oracle_qwen3vl8b_D.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "output/QA_Agent/Oracle/qwen3vl8b_D/atmbench/oracle_qwen3vl8b_D.jsonl" \
  --output-dir "output/QA_Agent/Oracle/qwen3vl8b_D/atmbench/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench-hard.json" \
  --media-source batch_results \
  --batch-fields "short_caption" \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/qwen3vl8b_D/hard/oracle_qwen3vl8b_D.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "output/QA_Agent/Oracle/qwen3vl8b_D/hard/oracle_qwen3vl8b_D.jsonl" \
  --output-dir "output/QA_Agent/Oracle/qwen3vl8b_D/hard/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
