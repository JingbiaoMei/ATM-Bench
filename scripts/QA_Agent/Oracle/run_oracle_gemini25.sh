#!/usr/bin/env bash

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://100.122.172.110:8317/v1/chat/completions}"
MAX_WORKERS="${MAX_WORKERS:-4}"

run_oracle() {
  local tag="$1"
  local model_name="$2"

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
    --model "${model_name}" \
    --max-workers "${MAX_WORKERS}" \
    --timeout 120 \
    --output-file "output/QA_Agent/Oracle/${tag}/atmbench/oracle_${tag}.jsonl"

  python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth "./data/atm-bench/atm-bench.json" \
    --predictions "output/QA_Agent/Oracle/${tag}/atmbench/oracle_${tag}.jsonl" \
    --output-dir "output/QA_Agent/Oracle/${tag}/atmbench/eval" \
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
    --model "${model_name}" \
    --max-workers "${MAX_WORKERS}" \
    --timeout 120 \
    --output-file "output/QA_Agent/Oracle/${tag}/hard/oracle_${tag}.jsonl"

  python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth "./data/atm-bench/atm-bench-hard.json" \
    --predictions "output/QA_Agent/Oracle/${tag}/hard/oracle_${tag}.jsonl" \
    --output-dir "output/QA_Agent/Oracle/${tag}/hard/eval" \
    --metrics em atm \
    --judge-provider openai \
    --judge-model gpt-5-mini \
    --judge-reasoning-effort minimal \
    --max-workers 2
}

run_oracle "gemini25pro" "gemini-2.5-pro"
run_oracle "gemini25flash" "gemini-2.5-flash"
