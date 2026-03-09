#!/usr/bin/env bash

JUDGE_MODEL_GPT="${JUDGE_MODEL_GPT:-gpt-5-mini}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-8}"

run_eval_bundle() {
    local ground_truth="$1"
    local predictions="$2"
    local eval_dir="$3"
    local retrieval_details="${4:-}"

    echo "=============================================="
    echo "Starting evaluation: ${eval_dir}"
    echo "=============================================="

    mkdir -p "${eval_dir}"

    python memqa/utils/evaluator/evaluate_qa.py \
        --ground-truth "${ground_truth}" \
        --predictions "${predictions}" \
        --output-dir "${eval_dir}" \
        --metrics em atm \
        --judge-provider openai \
        --judge-model "${JUDGE_MODEL_GPT}" \
        --judge-reasoning-effort minimal \
        --max-workers "${JUDGE_MAX_WORKERS}"

    if [ -n "${retrieval_details}" ] && [ -f "${retrieval_details}" ]; then
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

export -f run_eval_bundle
