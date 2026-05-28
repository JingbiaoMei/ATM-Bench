#!/usr/bin/env bash
# SimpleMem Baseline for ATMBench
#
# SimpleMem: Efficient Lifelong Memory for LLM Agents
# Paper: https://arxiv.org/abs/2601.02553
# Code:  https://github.com/aiming-lab/SimpleMem
#
# Prerequisites:
#   1. Clone SimpleMem:
#        git clone https://github.com/aiming-lab/SimpleMem.git ../SimpleMem
#        export SIMPLEMEM_DIR=../SimpleMem
#   2. Install SimpleMem deps:
#        pip install -r "${SIMPLEMEM_DIR}/requirements.txt"
#   3. Create ${SIMPLEMEM_DIR}/config.py from config.py.example and set API key.
#
# By default this builds the SimpleMem index once (the build cache key is
# derived from the corpus, not the QA set) and answers + evaluates against
# both the easy and hard QA splits:
#   - ./data/atm-bench/atm-bench.json       (easy split only; NOT a superset)
#   - ./data/atm-bench/atm-bench-hard.json  (disjoint hard split)
#
# Usage:
#   bash scripts/QA_Agent/SimpleMem/run.sh
#
# Useful overrides:
#   STAGE=build|answer|all                   (default: all)
#   SIMPLEMEM_FORCE_REBUILD=0|1              (default: 0 — resume from checkpoint)
#   QA_ATMBENCH=...   QA_HARD=...            (override per-split QA file)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_eval.sh"

# --- Provider + model ---
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
PROVIDER="${PROVIDER:-vllm}"

# Build (indexing) model — smaller/cheaper companion to MODEL.
BUILD_MODEL="${BUILD_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
BUILD_ENDPOINT="${BUILD_ENDPOINT:-${VLLM_ENDPOINT}}"

# Answer mode: "atm" aligns with HippoRAG2/A-Mem/MemoryOS evidence-only prompt.
SIMPLEMEM_ANSWER_MODE="${SIMPLEMEM_ANSWER_MODE:-atm}"

# Default to 0 so an interrupted build resumes from the per-window checkpoint.
SIMPLEMEM_FORCE_REBUILD="${SIMPLEMEM_FORCE_REBUILD:-0}"

# SimpleMem retrieval / build hyperparameters (paper defaults).
SEMANTIC_TOP_K="${SEMANTIC_TOP_K:-25}"
KEYWORD_TOP_K="${KEYWORD_TOP_K:-5}"
STRUCTURED_TOP_K="${STRUCTURED_TOP_K:-5}"
WINDOW_SIZE="${WINDOW_SIZE:-40}"
OVERLAP_SIZE="${OVERLAP_SIZE:-2}"
MAX_REFLECTION_ROUNDS="${MAX_REFLECTION_ROUNDS:-2}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS:-4}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# Pipeline stage: build | answer | all.
STAGE="${STAGE:-all}"
case "${STAGE}" in
    build|answer|all) ;;
    *)
        echo "ERROR: STAGE must be one of: build, answer, all"
        exit 1
        ;;
esac

# --- SimpleMem repository pin ---
SIMPLEMEM_DIR="${SIMPLEMEM_DIR:-../SimpleMem}"
if [ -z "${SIMPLEMEM_DIR}" ]; then
    echo "ERROR: Set SIMPLEMEM_DIR to the cloned SimpleMem repository path."
    exit 1
fi
SIMPLEMEM_EXPECTED_COMMIT="${SIMPLEMEM_EXPECTED_COMMIT-094027eca4c890dc9912be8cee1da04428de8076}"
if [ -n "${SIMPLEMEM_EXPECTED_COMMIT}" ]; then
    if ! actual_commit="$(git -C "${SIMPLEMEM_DIR}" rev-parse HEAD 2>/dev/null)"; then
        echo "ERROR: SIMPLEMEM_EXPECTED_COMMIT is set, but ${SIMPLEMEM_DIR} is not a git checkout."
        exit 1
    fi
    if [ "${actual_commit}" != "${SIMPLEMEM_EXPECTED_COMMIT}" ]; then
        echo "ERROR: SimpleMem commit mismatch."
        echo "  Expected: ${SIMPLEMEM_EXPECTED_COMMIT}"
        echo "  Actual:   ${actual_commit}"
        exit 1
    fi
fi

# --- Data paths (public-repo layout) ---
QA_ATMBENCH="${QA_ATMBENCH:-./data/atm-bench/atm-bench.json}"
QA_HARD="${QA_HARD:-./data/atm-bench/atm-bench-hard.json}"

EMAIL_FILE="./data/raw_memory/email/emails.json"
IMAGE_BATCH="./output/image/qwen3vl2b/batch_results.json"
VIDEO_BATCH="./output/video/qwen3vl2b/batch_results.json"
IMAGE_ROOT="./data/raw_memory/image"
VIDEO_ROOT="./data/raw_memory/video"

# --- Output layout (matches HippoRag2/MemoryOS) ---
TOP_K_LABEL="${TOP_K_LABEL:-topk$((SEMANTIC_TOP_K + KEYWORD_TOP_K + STRUCTURED_TOP_K))}"
OUTPUT_BASE="${OUTPUT_BASE:-output/QA_Agent/SimpleMem/main_table/${TOP_K_LABEL}}"
METHOD_BASE="${METHOD_BASE:-simplemem}"

ATM_METHOD="atmbench/${METHOD_BASE}"
HARD_METHOD="hard/${METHOD_BASE}"
ATM_DIR="${OUTPUT_BASE}/${ATM_METHOD}"
HARD_DIR="${OUTPUT_BASE}/${HARD_METHOD}"

# --- Python path: include local repo and the cloned SimpleMem repo. ---
export PYTHONPATH=".:${SIMPLEMEM_DIR}:${PYTHONPATH:-}"

VLLM_ENDPOINT_ARGS=()
if [ -n "${VLLM_ENDPOINT}" ]; then
    VLLM_ENDPOINT_ARGS=(--vllm-endpoint "${VLLM_ENDPOINT}")
fi

FORCE_REBUILD_ARGS=()
if [ "${SIMPLEMEM_FORCE_REBUILD}" = "1" ]; then
    FORCE_REBUILD_ARGS=(--simplemem-force-rebuild)
else
    FORCE_REBUILD_ARGS=(--no-simplemem-force-rebuild)
fi

echo "=============================================="
echo "SimpleMem (ATMBench)"
echo "  Build model:  ${BUILD_MODEL}"
echo "  Answer model: ${MODEL}"
echo "  Endpoint:     ${VLLM_ENDPOINT}"
echo "  Top-K (S/K/X):${SEMANTIC_TOP_K} / ${KEYWORD_TOP_K} / ${STRUCTURED_TOP_K}"
echo "  Window:       size=${WINDOW_SIZE} overlap=${OVERLAP_SIZE}"
echo "  Answer mode:  ${SIMPLEMEM_ANSWER_MODE}"
echo "  Force rebuild:${SIMPLEMEM_FORCE_REBUILD}"
echo "  SimpleMem:    ${SIMPLEMEM_DIR}"
echo "  Output base:  ${OUTPUT_BASE}"
echo "=============================================="

run_simplemem_stage() {
    local stage="$1"
    local qa_file="$2"
    local method_name="$3"

    python memqa/qa_agent_baselines/SimpleMem/simplemem_baseline.py \
        --qa-file "${qa_file}" \
        --media-source batch_results \
        --image-batch-results "${IMAGE_BATCH}" \
        --video-batch-results "${VIDEO_BATCH}" \
        --email-file "${EMAIL_FILE}" \
        --image-root "${IMAGE_ROOT}" \
        --video-root "${VIDEO_ROOT}" \
        --provider "${PROVIDER}" \
        --model "${MODEL}" \
        --output-dir-base "${OUTPUT_BASE}" \
        --method-name "${method_name}" \
        --overwrite-output \
        --max-workers "${MAX_WORKERS}" \
        --simplemem-semantic-top-k "${SEMANTIC_TOP_K}" \
        --simplemem-keyword-top-k "${KEYWORD_TOP_K}" \
        --simplemem-structured-top-k "${STRUCTURED_TOP_K}" \
        --simplemem-window-size "${WINDOW_SIZE}" \
        --simplemem-overlap-size "${OVERLAP_SIZE}" \
        --simplemem-enable-planning \
        --simplemem-enable-reflection \
        --simplemem-max-reflection-rounds "${MAX_REFLECTION_ROUNDS}" \
        --no-simplemem-enable-parallel-processing \
        --simplemem-max-parallel-workers "${PARALLEL_WORKERS}" \
        --simplemem-enable-parallel-retrieval \
        --simplemem-max-retrieval-workers "${RETRIEVAL_WORKERS}" \
        --simplemem-answer-mode "${SIMPLEMEM_ANSWER_MODE}" \
        --no-simplemem-quiet \
        --stage "${stage}" \
        --retrieval-log-k 100 \
        --simplemem-dir "${SIMPLEMEM_DIR}" \
        "${VLLM_ENDPOINT_ARGS[@]}" \
        --build-model "${BUILD_MODEL}" \
        --build-endpoint "${BUILD_ENDPOINT}" \
        "${FORCE_REBUILD_ARGS[@]}"
}

# --- Build stage (shared index, runs once) ---
if [ "${STAGE}" = "build" ] || [ "${STAGE}" = "all" ]; then
    echo ""
    echo "=== Build stage (shared index across both QA splits) ==="
    run_simplemem_stage build "${QA_ATMBENCH}" "${ATM_METHOD}"
    build_status=$?
    if [ "${build_status}" -ne 0 ]; then
        echo "SimpleMem build failed with exit code ${build_status}; skipping QA + evaluation."
        exit "${build_status}"
    fi
fi

if [ "${STAGE}" = "build" ]; then
    echo ""
    echo "=== Build stage complete ==="
    echo "Switch vLLM on ${VLLM_ENDPOINT} to ${MODEL}, then run:"
    echo "  STAGE=answer bash scripts/QA_Agent/SimpleMem/run.sh"
    exit 0
fi

# --- Easy split ---
echo ""
echo "================================================================"
echo "QA file: ${QA_ATMBENCH}"
echo "Method:  ${ATM_METHOD}"
echo "================================================================"

run_simplemem_stage answer "${QA_ATMBENCH}" "${ATM_METHOD}"

run_eval_bundle \
    "${QA_ATMBENCH}" \
    "${ATM_DIR}/simplemem_answers.jsonl" \
    "${ATM_DIR}/eval" \
    "${ATM_DIR}/retrieval_recall_details.json"

# --- Hard split ---
echo ""
echo "================================================================"
echo "QA file: ${QA_HARD}"
echo "Method:  ${HARD_METHOD}"
echo "================================================================"

run_simplemem_stage answer "${QA_HARD}" "${HARD_METHOD}"

run_eval_bundle \
    "${QA_HARD}" \
    "${HARD_DIR}/simplemem_answers.jsonl" \
    "${HARD_DIR}/eval" \
    "${HARD_DIR}/retrieval_recall_details.json"
