#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model gpt-5 \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/gpt5/atmbench/oracle_gpt5.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "output/QA_Agent/Oracle/gpt5/atmbench/oracle_gpt5.jsonl" \
  --output-dir "output/QA_Agent/Oracle/gpt5/atmbench/eval" \
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
  --model gpt-5 \
  --max-workers 8 \
  --timeout 120 \
  --output-file "output/QA_Agent/Oracle/gpt5/hard/oracle_gpt5.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "output/QA_Agent/Oracle/gpt5/hard/oracle_gpt5.jsonl" \
  --output-dir "output/QA_Agent/Oracle/gpt5/hard/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
