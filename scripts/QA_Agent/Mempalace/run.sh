#!/usr/bin/env bash

# MemPalace baseline runner.
#
# From the repository root, after activating the dedicated `mempalace` conda
# environment (see memqa/qa_agent_baselines/Mempalace/README.md):
#
#   conda activate mempalace
#   bash scripts/QA_Agent/Mempalace/run.sh
#
# Useful env overrides:
#   TOP_K=10           # evidence items passed to the answerer
#   N_RESULTS=100      # MemPalace search candidates (over-fetch for recall)
#   VLLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
#   ANSWERER_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
#   REBUILD_INDEX=1    # force-rebuild the palace cache
#   OVERWRITE=1        # regenerate existing answer/eval files
#   MAX_WORKERS=8      # answer-stage thread workers
#   RETRIEVAL_K_VALUES=1,5,10,25,50,100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common_eval.sh"

TOP_K="${TOP_K:-10}"
N_RESULTS="${N_RESULTS:-100}"

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
ANSWERER_PROVIDER="${ANSWERER_PROVIDER:-vllm}"
ANSWERER_ENDPOINT="${ANSWERER_ENDPOINT:-${VLLM_ENDPOINT}}"
ANSWERER_MODEL="${ANSWERER_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"

MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
TIMEOUT="${TIMEOUT:-120}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"
OVERWRITE="${OVERWRITE:-0}"

QA_ATMBENCH="${QA_ATMBENCH:-./data/atm-bench/atm-bench.json}"
QA_HARD="${QA_HARD:-./data/atm-bench/atm-bench-hard.json}"

EMAIL_FILE="${EMAIL_FILE:-./data/raw_memory/email/emails.json}"
IMAGE_BATCH="${IMAGE_BATCH:-./output/image/qwen3vl2b/batch_results.json}"
VIDEO_BATCH="${VIDEO_BATCH:-./output/video/qwen3vl2b/batch_results.json}"
IMAGE_ROOT="${IMAGE_ROOT:-./data/raw_memory/image}"
VIDEO_ROOT="${VIDEO_ROOT:-./data/raw_memory/video}"

OUTPUT_BASE="${OUTPUT_BASE:-output/QA_Agent/Mempalace/main_table/topk${TOP_K}}"
CACHE_DIR="${CACHE_DIR:-output/QA_Agent/Mempalace/index_cache}"
ATM_DIR="${OUTPUT_BASE}/atmbench/mempalace"
HARD_DIR="${OUTPUT_BASE}/hard/mempalace"

read_key_file() {
  local path="$1"
  if [ -f "${path}" ]; then
    tr -d '\r\n' < "${path}"
  fi
}

if [ -z "${ANSWERER_API_KEY+x}" ]; then
  if [ "${ANSWERER_PROVIDER}" = "openai" ]; then
    ANSWERER_API_KEY="${OPENAI_API_KEY:-$(read_key_file api_keys/.openai_key)}"
  else
    ANSWERER_API_KEY="${VLLM_API_KEY:-$(read_key_file api_keys/.vllm_key)}"
  fi
fi

if [ -z "${OPENAI_API_KEY+x}" ]; then
  OPENAI_API_KEY="$(read_key_file api_keys/.openai_key)"
fi
export OPENAI_API_KEY

echo "=============================================="
echo "MemPalace (ATM-Bench)"
echo "  TOP_K:        ${TOP_K}"
echo "  N_RESULTS:    ${N_RESULTS}"
echo "  Embedding:    all-MiniLM-L6-v2 (MemPalace ONNX default)"
echo "  Index:        drawers + regex closets"
echo "  Retrieval:    search_memories(candidate_strategy=vector)"
echo "  Cache:        ${CACHE_DIR}"
echo "  Output base:  ${OUTPUT_BASE}"
echo "  Answerer:     ${ANSWERER_PROVIDER} / ${ANSWERER_MODEL}"
echo "  Endpoint:     ${ANSWERER_ENDPOINT}"
echo "=============================================="

BUILD_ARGS=(
  --stage build
  --qa-file "${QA_ATMBENCH}"
  --image-batch-results "${IMAGE_BATCH}"
  --video-batch-results "${VIDEO_BATCH}"
  --email-file "${EMAIL_FILE}"
  --image-root "${IMAGE_ROOT}"
  --video-root "${VIDEO_ROOT}"
  --n-results "${N_RESULTS}"
  --index-cache "${CACHE_DIR}"
  --output-dir-base "${OUTPUT_BASE}"
)
if [ "${REBUILD_INDEX}" = "1" ]; then
  BUILD_ARGS+=(--force-rebuild)
fi

python memqa/qa_agent_baselines/Mempalace/mempalace_baseline.py "${BUILD_ARGS[@]}"
if [ "$?" -ne 0 ]; then
  echo "ERROR: MemPalace build failed."
  exit 1
fi

run_case() {
  local qa_file="$1"
  local method="$2"
  local out_dir="$3"

  echo ""
  echo "=============================================="
  echo "MemPalace Answer Stage: ${method}"
  echo "  QA file: ${qa_file}"
  echo "=============================================="

  ANSWER_ARGS=(
    --stage answer
    --qa-file "${qa_file}"
    --image-batch-results "${IMAGE_BATCH}"
    --video-batch-results "${VIDEO_BATCH}"
    --email-file "${EMAIL_FILE}"
    --image-root "${IMAGE_ROOT}"
    --video-root "${VIDEO_ROOT}"
    --candidate-strategy vector
    --n-results "${N_RESULTS}"
    --provider "${ANSWERER_PROVIDER}"
    --model "${ANSWERER_MODEL}"
    --retrieve-k "${TOP_K}"
    --max-workers "${MAX_WORKERS}"
    --max-tokens "${MAX_TOKENS}"
    --timeout "${TIMEOUT}"
    --index-cache "${CACHE_DIR}"
    --output-dir-base "${OUTPUT_BASE}"
    --method-name "${method}"
  )

  if [ "${ANSWERER_PROVIDER}" = "vllm" ]; then
    ANSWER_ARGS+=(--vllm-endpoint "${ANSWERER_ENDPOINT}")
  fi
  if [ -n "${ANSWERER_API_KEY}" ]; then
    ANSWER_ARGS+=(--api-key "${ANSWERER_API_KEY}")
  fi
  if [ "${OVERWRITE}" = "1" ]; then
    ANSWER_ARGS+=(--overwrite)
    rm -f "${out_dir}/retrieval_recall_comprehensive_summary.json" \
          "${out_dir}/retrieval_recall_joint_accuracy_"*.json
    rm -rf "${out_dir}/eval"
  fi

  python memqa/qa_agent_baselines/Mempalace/mempalace_baseline.py "${ANSWER_ARGS[@]}"
  if [ "$?" -ne 0 ]; then
    echo "ERROR: MemPalace answer stage failed for ${method}."
    return 1
  fi

  run_eval_bundle \
    "${qa_file}" \
    "${out_dir}/mempalace_answers.jsonl" \
    "${out_dir}/eval" \
    "${out_dir}/retrieval_recall_details.json"
}

FAILURES=0
run_case "${QA_ATMBENCH}" "atmbench/mempalace" "${ATM_DIR}" || FAILURES=1
run_case "${QA_HARD}"     "hard/mempalace"     "${HARD_DIR}" || FAILURES=1

echo ""
echo "=============================================="
if [ "${FAILURES}" = "0" ]; then
  echo "MEMPALACE RUN COMPLETE"
else
  echo "MEMPALACE RUN COMPLETE WITH FAILURES"
fi
echo "=============================================="
exit "${FAILURES}"
