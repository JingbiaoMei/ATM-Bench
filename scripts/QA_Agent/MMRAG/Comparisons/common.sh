#!/usr/bin/env bash
# Common configurations for ATMBench MMRAG comparison scripts.

# Top-K configuration
TOP_K="${TOP_K:-10}"
RETRIEVAL_MAX_K="${RETRIEVAL_MAX_K:-200}"

if [ "${TOP_K}" -eq 5 ]; then
    RETRIEVAL_TOP_K=5
    RERANK_INPUT_K=10
    RERANK_TOP_K=5
else
    RETRIEVAL_TOP_K=10
    RERANK_INPUT_K=20
    RERANK_TOP_K=10
fi

# Paths (public placeholders)
OUTPUT_BASE="${OUTPUT_BASE:-output/QA_Agent/MMRAG/table5/topk${TOP_K}}"
QA_FILE="${QA_FILE:-./data/atm-bench/atm-bench.json}"
EMAIL_FILE="${EMAIL_FILE:-./data/raw_memory/email/merged_emails.json}"
IMAGE_ROOT="${IMAGE_ROOT:-./data/raw_memory/image}"
VIDEO_ROOT="${VIDEO_ROOT:-./data/raw_memory/video}"
IMAGE_BATCH="${IMAGE_BATCH:-./output/image/qwen3vl2b/batch_results.json}"
VIDEO_BATCH="${VIDEO_BATCH:-./output/video/qwen3vl2b/batch_results.json}"

# Model endpoints
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"

# Answerer models
ANSWERER_MODEL_2B="Qwen/Qwen3-VL-2B-Instruct"
ANSWERER_MODEL_8B="Qwen/Qwen3-VL-8B-Instruct-FP8"

# Embedding models
MODEL_ALLMINILM_L6="sentence-transformers/all-MiniLM-L6-v2"
MODEL_QWEN3_EMB_06B="Qwen/Qwen3-Embedding-0.6B"
MODEL_QWEN3_EMB_4B="Qwen/Qwen3-Embedding-4B"
MODEL_QWEN3_VL_EMB_2B="Qwen/Qwen3-VL-Embedding-2B"

# Reranker models
MODEL_QWEN3_RERANK_06B="Qwen/Qwen3-Reranker-0.6B"
MODEL_QWEN3_RERANK_4B="Qwen/Qwen3-Reranker-4B"

# Evaluation (GPT-only)
JUDGE_MODEL_GPT="${JUDGE_MODEL_GPT:-gpt-5-mini}"

# Text embedding: include essential metadata + short caption
TEXT_AUGMENT_ARGS="--include-id --include-timestamp --include-location --include-short-caption"

# Single-judge evaluation: OpenAI judge only
run_eval_dual() {
    local predictions="$1"
    local eval_dir="$2"
    local retrieval_details="$3"

    echo "=============================================="
    echo "Starting evaluation: ${eval_dir}"
    echo "=============================================="

    mkdir -p "${eval_dir}"

    python memqa/utils/evaluator/evaluate_qa.py \
        --ground-truth "${QA_FILE}" \
        --predictions "${predictions}" \
        --output-dir "${eval_dir}" \
        --metrics em atm \
        --judge-provider openai \
        --judge-model "${JUDGE_MODEL_GPT}" \
        --judge-reasoning-effort minimal \
        --max-workers 8

    if [ -f "${retrieval_details}" ]; then
        python memqa/utils/evaluator/evaluate_retrieval/comprehensive_eval.py \
            --details "${retrieval_details}"

        local atm_details=""
        atm_details="$(ls "${eval_dir}"/atm_*.json 2>/dev/null | head -n 1)"
        if [ -n "${atm_details}" ] && [ -f "${atm_details}" ]; then
            python memqa/utils/evaluator/evaluate_retrieval/joint_accuracy.py \
                --retrieval-details "${retrieval_details}" \
                --atm-details "${atm_details}"
        else
            echo "Skipping joint accuracy (no atm_*.json found in ${eval_dir})"
        fi
    else
        echo "Skipping retrieval/joint evaluation (missing details): ${retrieval_details}"
    fi

    echo "=============================================="
    echo "Evaluation complete: ${eval_dir}"
    echo "=============================================="
}

export -f run_eval_dual
export QA_FILE JUDGE_MODEL_GPT
