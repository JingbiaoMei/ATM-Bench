#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://100.122.172.110:8317/v1/chat/completions}"
MODEL="${MODEL:-claude-sonnet-4.5}"
TAG="${TAG:-sonnet45}"
MAX_WORKERS="${MAX_WORKERS:-1}"

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --max-workers "${MAX_WORKERS}" \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/${TAG}/atmbench/oracle_${TAG}.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "output/QA_Agent/Oracle/${TAG}/atmbench/oracle_${TAG}.jsonl" \
  --output-dir "output/QA_Agent/Oracle/${TAG}/atmbench/eval" \
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
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --max-workers "${MAX_WORKERS}" \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/${TAG}/hard/oracle_${TAG}.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "output/QA_Agent/Oracle/${TAG}/hard/oracle_${TAG}.jsonl" \
  --output-dir "output/QA_Agent/Oracle/${TAG}/hard/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
