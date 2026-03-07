#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --no-evidence \
  --output-file "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/atmbench/oracle_qwen3vl8b_no_evidence.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/atmbench/oracle_qwen3vl8b_no_evidence.jsonl" \
  --output-dir "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/atmbench/eval" \
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
  --email-file "./data/raw_memory/email/merged_emails.json" \
  --provider vllm \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --model "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --max-workers 8 \
  --timeout 120 \
  --no-evidence \
  --output-file "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/hard/oracle_qwen3vl8b_no_evidence.jsonl"

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/hard/oracle_qwen3vl8b_no_evidence.jsonl" \
  --output-dir "output/QA_Agent/Oracle/no_evidence_qwen3vl8b/hard/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
