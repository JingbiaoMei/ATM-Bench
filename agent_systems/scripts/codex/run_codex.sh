#!/usr/bin/env bash
#
# Run Codex CLI on ATM-hard questions (one process per question).
#
# Usage:
#   bash agent_systems/scripts/codex/run_codex.sh              # all questions
#   bash agent_systems/scripts/codex/run_codex.sh <question_id>
#
# Env overrides (see agent_systems/config.py):
#   AGSYS_RUN_TAG, AGSYS_CODEX_BIN, AGSYS_CODEX_MODEL, AGSYS_CODEX_MODEL_TAG
#   AGSYS_MEMORY_MODE=sgm|raw|descriptive
#   AGSYS_CODEX_REASONING_EFFORT
#   AGSYS_EVAL_EXTRA_FLAGS="--request-delay 1.0"
#   AGSYS_EVAL_GLM_FLAGS="--judge-endpoint http://... --max-workers 1"
#   AGSYS_EVAL_GPT_FLAGS="--judge-model gpt-5-mini --judge-reasoning-effort minimal"
#   AGSYS_SKIP_EVAL=1
#

set -o pipefail

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

agsys_require_eval_root

codex_finalize_answer() {
  local qid="$1"
  local workspace_answer="$2"
  local final_message="$3"
  local answer_file="$4"

  if [[ -f "${workspace_answer}" ]] && agsys_answer_valid "${workspace_answer}" "${qid}"; then
    cp "${workspace_answer}" "${answer_file}" || return 1
    echo "workspace-output/answer.json"
    return 0
  fi
  if [[ -f "${final_message}" ]] && agsys_answer_valid "${final_message}" "${qid}"; then
    cp "${final_message}" "${answer_file}" || return 1
    echo "assistant_final_message.txt"
    return 0
  fi
  return 1
}

if [[ ! -x "$(command -v "${AGSYS_CODEX_BIN}")" ]] && [[ ! -x "${AGSYS_CODEX_BIN}" ]]; then
  agsys_die "Codex binary not found: ${AGSYS_CODEX_BIN}. Set AGSYS_CODEX_BIN to the desired codex executable."
fi

CODEX_HELP="$("${AGSYS_CODEX_BIN}" exec --help 2>&1 || true)"
if [[ "${CODEX_HELP}" != *"--ephemeral"* ]] || [[ "${CODEX_HELP}" != *"--output-schema"* ]] || [[ "${CODEX_HELP}" != *"--output-last-message"* ]]; then
  CODEX_VERSION="$("${AGSYS_CODEX_BIN}" --version 2>/dev/null | head -n 1 || true)"
  agsys_die "Codex CLI is too old for this harness (${AGSYS_CODEX_BIN}: ${CODEX_VERSION:-unknown version}). Expected codex-cli >= 0.104.0-alpha.1 with --ephemeral, --output-schema, and --output-last-message. Check 'which codex', 'codex --version', or set AGSYS_CODEX_BIN=/path/to/newer/codex."
fi

SINGLE_QID="${1:-}"
MEMORY_MODE="$(agsys_normalize_memory_mode "${AGSYS_MEMORY_MODE:-sgm}")"
MODEL_TAG="$(agsys_tag_with_memory_mode "${AGSYS_CODEX_MODEL_TAG:-${AGSYS_CODEX_MODEL:-default}}" "${MEMORY_MODE}")"

COMPLETED=0
FAILED=0

if [[ "${MEMORY_MODE}" == "raw" ]]; then
  mkdir -p "${AGSYS_EVAL_ROOT}/.codex-runtime"
  SESSION_TMP_ROOT="$(mktemp -d "${AGSYS_EVAL_ROOT}/.codex-runtime/session-XXXXXXXX")"
else
  SESSION_TMP_ROOT="$(mktemp -d "/tmp/agsys-codex-XXXXXXXX")"
fi
MEMORY_CACHE_DIR="${SESSION_TMP_ROOT}/memory-cache"
mkdir -p "${MEMORY_CACHE_DIR}"
cp "${AGSYS_EVAL_ROOT}/memory/image_metadata.json" "${MEMORY_CACHE_DIR}/image_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/video_metadata.json" "${MEMORY_CACHE_DIR}/video_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/emails.json" "${MEMORY_CACHE_DIR}/emails.json"
cp "${AGSYS_EVAL_ROOT}/memory/memory_variant.json" "${MEMORY_CACHE_DIR}/memory_variant.json" 2>/dev/null || true

cleanup() {
  rm -rf "${SESSION_TMP_ROOT}"
}
trap cleanup EXIT

while IFS= read -r QID; do
  RUN_DIR="$(agsys_setup_run_dir "codex" "${MODEL_TAG}" "${QID}")"
  ANSWER_FILE="${RUN_DIR}/output/answer.json"
  TRACE_FILE="${RUN_DIR}/output/trace.jsonl"
  USAGE_FILE="${RUN_DIR}/output/usage.json"
  STDERR_LOG="${RUN_DIR}/output/stderr.log"
  LAST_MESSAGE_FILE="${RUN_DIR}/output/assistant_final_message.txt"
  RUNTIME_MANIFEST="${RUN_DIR}/output/runtime_manifest.json"

  if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    if [[ ! -f "${USAGE_FILE}" ]] && [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_usage.py \
        --format codex \
        --trace "${TRACE_FILE}" \
        --out "${USAGE_FILE}" \
        --agent codex \
        --model-tag "${MODEL_TAG}" \
        --model "${AGSYS_CODEX_MODEL:-}" \
        >> "${STDERR_LOG}" 2>&1 || true
    fi
    echo "SKIP: ${QID} already answered (${ANSWER_FILE})"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  rm -f "${ANSWER_FILE}"
  rm -f "${TRACE_FILE}"
  rm -f "${USAGE_FILE}"
  rm -f "${STDERR_LOG}"
  rm -f "${LAST_MESSAGE_FILE}"
  rm -f "${RUNTIME_MANIFEST}"
  rm -rf "${RUN_DIR}/output/codex-log"
  rm -rf "${RUN_DIR}/workspace-output"
  rm -f "${RUN_DIR}/output/codex-config.toml"

  SYSTEM_PROMPT="$(cat "${AGSYS_SYSTEM_PROMPT}")"
  QTMP_ROOT="$(mktemp -d "${SESSION_TMP_ROOT}/${QID}.XXXXXXXX")"
  WORKSPACE_DIR="${QTMP_ROOT}/workspace"
  WORKSPACE_OUTPUT_DIR="${WORKSPACE_DIR}/output"
  WORKSPACE_ANSWER_FILE="${WORKSPACE_OUTPUT_DIR}/answer.json"
  mkdir -p "${WORKSPACE_DIR}/memory" "${WORKSPACE_OUTPUT_DIR}"
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.json" "${WORKSPACE_DIR}/question.json"
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.txt" "${WORKSPACE_DIR}/question.txt"
  cp "${MEMORY_CACHE_DIR}/image_metadata.json" "${WORKSPACE_DIR}/memory/image_metadata.json"
  cp "${MEMORY_CACHE_DIR}/video_metadata.json" "${WORKSPACE_DIR}/memory/video_metadata.json"
  cp "${MEMORY_CACHE_DIR}/emails.json" "${WORKSPACE_DIR}/memory/emails.json"
  if [[ -f "${MEMORY_CACHE_DIR}/memory_variant.json" ]]; then
    cp "${MEMORY_CACHE_DIR}/memory_variant.json" "${WORKSPACE_DIR}/memory/memory_variant.json"
  fi
  if [[ "${MEMORY_MODE}" == "raw" ]]; then
    mkdir -p "${WORKSPACE_DIR}/memory/raw_images" "${WORKSPACE_DIR}/memory/raw_videos"
    cp -al "${AGSYS_EVAL_ROOT}/memory/raw_images/." "${WORKSPACE_DIR}/memory/raw_images/"
    cp -al "${AGSYS_EVAL_ROOT}/memory/raw_videos/." "${WORKSPACE_DIR}/memory/raw_videos/"
  fi
  QUESTION_TEXT="$(cat "${WORKSPACE_DIR}/question.txt")"
  mkdir -p "${AGSYS_EVAL_ROOT}/.codex-runtime"
  CODEX_TMPDIR="$(mktemp -d "${AGSYS_EVAL_ROOT}/.codex-runtime/${MODEL_TAG}-${QID}-XXXXXXXX")"
  CODEX_HOME_DIR="${CODEX_TMPDIR}/home"
  mkdir -p "${CODEX_HOME_DIR}"
  if [[ -f "${HOME}/.codex/auth.json" ]]; then
    cp "${HOME}/.codex/auth.json" "${CODEX_HOME_DIR}/auth.json"
  fi
  if ! python3 agent_systems/runtime_artifacts.py write-codex-config \
    --out "${CODEX_HOME_DIR}/config.toml" \
    --reasoning-effort "${AGSYS_CODEX_REASONING_EFFORT:-}" \
    --model "${AGSYS_CODEX_MODEL:-}"; then
    echo "FAILED: ${QID} (could not build isolated Codex config)"
    rm -rf "${QTMP_ROOT}"
    rm -rf "${CODEX_TMPDIR}"
    FAILED=$((FAILED + 1))
    echo ""
    continue
  fi
  cp "${CODEX_HOME_DIR}/config.toml" "${RUN_DIR}/output/codex-config.toml"

  echo "── [codex/${MODEL_TAG}] ${QID}"

  CMD=(
    "${AGSYS_CODEX_BIN}" exec
    --sandbox workspace-write
    --ephemeral
    --skip-git-repo-check
    --json
    -C "${WORKSPACE_DIR}"
    --output-schema "${AGSYS_QA_SCHEMA}"
    --output-last-message "${LAST_MESSAGE_FILE}"
  )
  if [[ -n "${AGSYS_CODEX_MODEL}" ]]; then
    CMD+=(--model "${AGSYS_CODEX_MODEL}")
  fi

  # Read prompt from stdin so we don't fight shell quoting on large prompts.
  {
    echo "${SYSTEM_PROMPT}"
    echo ""
    echo "Question:"
    echo "${QUESTION_TEXT}"
  } | env CODEX_HOME="${CODEX_HOME_DIR}" "${CMD[@]}" - > "${TRACE_FILE}" 2> "${STDERR_LOG}"
  EXIT_CODE=$?

  if [[ -d "${CODEX_HOME_DIR}/log" ]]; then
    mkdir -p "${RUN_DIR}/output/codex-log"
    cp -r "${CODEX_HOME_DIR}/log/." "${RUN_DIR}/output/codex-log/" 2>/dev/null || true
  fi

  if compgen -G "${WORKSPACE_OUTPUT_DIR}/*" >/dev/null; then
    mkdir -p "${RUN_DIR}/workspace-output"
    cp -r "${WORKSPACE_OUTPUT_DIR}/." "${RUN_DIR}/workspace-output/" 2>/dev/null || true
  fi

  if [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_usage.py \
      --format codex \
      --trace "${TRACE_FILE}" \
      --out "${USAGE_FILE}" \
      --agent codex \
      --model-tag "${MODEL_TAG}" \
      --model "${AGSYS_CODEX_MODEL:-}" \
      >> "${STDERR_LOG}" 2>&1 || true
  fi

  ANSWER_SOURCE=""
  if ANSWER_SOURCE="$(codex_finalize_answer "${QID}" "${WORKSPACE_ANSWER_FILE}" "${LAST_MESSAGE_FILE}" "${ANSWER_FILE}")"; then
    :
  else
    ANSWER_SOURCE=""
  fi

  python3 - <<'PY' \
    "${RUNTIME_MANIFEST}" \
    "${QID}" \
    "${RUN_DIR}" \
    "${WORKSPACE_DIR}" \
    "${AGSYS_EVAL_ROOT}/qas/${QID}" \
    "${AGSYS_CODEX_BIN}" \
    "${AGSYS_CODEX_MODEL:-}" \
    "${MODEL_TAG}" \
    "${CODEX_HOME_DIR}" \
    "${RUN_DIR}/output/codex-config.toml" \
    "${LAST_MESSAGE_FILE}" \
    "${ANSWER_SOURCE}" \
    "${MEMORY_MODE}"
import json, sys
out = sys.argv[1]
payload = {
    "agent": "codex",
    "question_id": sys.argv[2],
    "run_dir": sys.argv[3],
    "workspace_mode": "copied question files + copied temp memory cache",
    "workspace_host_path": sys.argv[4],
    "question_source_dir": sys.argv[5],
    "codex_bin": sys.argv[6],
    "model": sys.argv[7],
    "model_tag": sys.argv[8],
    "memory_mode": sys.argv[13],
    "codex_home": sys.argv[9],
    "runtime_config_snapshot": sys.argv[10],
    "assistant_final_message": sys.argv[11],
    "answer_source": sys.argv[12],
    "tmp_workspace_deleted_after_run": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  rm -rf "${QTMP_ROOT}"
  rm -rf "${CODEX_TMPDIR}"

  if [[ ${EXIT_CODE} -eq 0 ]] && [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    PREVIEW="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1],"r",encoding="utf-8")); print(d.get("answer","")[:120])' "${ANSWER_FILE}" 2>/dev/null || true)"
    echo "A: ${PREVIEW}"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "FAILED: ${QID} (exit=${EXIT_CODE})"
    if [[ -f "${STDERR_LOG}" ]]; then
      echo "--- stderr (last 10 lines) ---"
      tail -10 "${STDERR_LOG}"
      echo "---"
    fi
    FAILED=$((FAILED + 1))
  fi
  echo ""
done < <(agsys_list_qids "${SINGLE_QID}")

echo "=== codex summary: ${COMPLETED} completed, ${FAILED} failed ==="

if [[ -z "${SINGLE_QID}" ]]; then
  echo "=== Collecting: codex/${MODEL_TAG} ==="
  python3 agent_systems/collect_results.py --agent codex --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"
  python3 agent_systems/collect_usage.py --agent codex --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"

  if [[ -n "${AGSYS_SKIP_EVAL:-}" ]]; then
    echo "SKIP: evaluation (AGSYS_SKIP_EVAL=1)"
  else
    PREDICTIONS="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/codex/${MODEL_TAG}/answers.jsonl"
    EVAL_DIR="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/codex/${MODEL_TAG}/eval"
    agsys_evaluate_predictions "${PREDICTIONS}" "${EVAL_DIR}"
    EVAL_EXIT=$?
    if [[ ${EVAL_EXIT} -ne 0 ]]; then
      echo "WARN: evaluate_qa.py failed (exit=${EVAL_EXIT}). Set AGSYS_SKIP_EVAL=1 to skip."
    fi
  fi
fi
