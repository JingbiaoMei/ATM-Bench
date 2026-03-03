#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"

# Schema-Guided Memory (SGM): type,timestamp,location,short_caption,caption,ocr,tags

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source batch_results \
  --batch-fields "type,timestamp,location,short_caption,caption,ocr,tags" \
  --image-batch-results "./data/raw_memory/image/batch_results.json" \
  --video-batch-results "./data/raw_memory/video/batch_results.json" \
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/qwen3vl8b_SGM/atmbench/oracle_qwen3vl8b_SGM.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "output/QA_Agent/Oracle/qwen3vl8b_SGM/atmbench/oracle_qwen3vl8b_SGM.jsonl" \
  --output-dir "output/QA_Agent/Oracle/qwen3vl8b_SGM/atmbench/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench-hard.json" \
  --media-source batch_results \
  --batch-fields "type,timestamp,location,short_caption,caption,ocr,tags" \
  --image-batch-results "./data/raw_memory/image/batch_results.json" \
  --video-batch-results "./data/raw_memory/video/batch_results.json" \
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/qwen3vl8b_SGM/hard/oracle_qwen3vl8b_SGM.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "output/QA_Agent/Oracle/qwen3vl8b_SGM/hard/oracle_qwen3vl8b_SGM.jsonl" \
  --output-dir "output/QA_Agent/Oracle/qwen3vl8b_SGM/hard/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
