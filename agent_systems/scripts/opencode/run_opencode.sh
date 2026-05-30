#!/usr/bin/env bash
#
# Run opencode on ATM-hard questions (one process per question).
#
# Usage:
#   bash agent_systems/scripts/opencode/run_opencode.sh              # all questions
#   bash agent_systems/scripts/opencode/run_opencode.sh <question_id>
#
# Env overrides (see agent_systems/config.py):
#   AGSYS_RUN_TAG, AGSYS_OPENCODE_BIN, AGSYS_OPENCODE_MODEL, AGSYS_OPENCODE_VARIANT
#   AGSYS_MEMORY_MODE=sgm|raw|descriptive
#   AGSYS_OPENCODE_CONFIG_SOURCE, AGSYS_OPENCODE_CONFIG_DIR
#   AGSYS_OPENCODE_SANDBOX=bwrap|auto|off
#   AGSYS_EVAL_EXTRA_FLAGS="--request-delay 1.0"
#   AGSYS_EVAL_GPT_FLAGS="--judge-model gpt-5-mini --judge-reasoning-effort minimal"
#   AGSYS_OPENCODE_FORWARD_ENV="VAR1 VAR2 ..."  (extra env vars to forward into the
#                              sandbox; OPENAI_COMPATIBLE_API_KEY is always forwarded)
#   AGSYS_OPENCODE_TIMEOUT=900  (default 15 min; 0 to disable)
#   AGSYS_SKIP_EVAL=1
#

set -o pipefail

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

agsys_require_eval_root

SINGLE_QID="${1:-}"
MEMORY_MODE="$(agsys_normalize_memory_mode "${AGSYS_MEMORY_MODE:-sgm}")"
MODEL_TAG="$(agsys_tag_with_memory_mode "${AGSYS_OPENCODE_MODEL:-default}" "${MEMORY_MODE}")"

COMPLETED=0
FAILED=0

OPENCODE_BIN="${AGSYS_OPENCODE_BIN}"
if [[ "${OPENCODE_BIN}" != /* ]]; then
  OPENCODE_BIN="$(command -v "${OPENCODE_BIN}" 2>/dev/null || true)"
fi
[[ -n "${OPENCODE_BIN}" ]] || agsys_die "OpenCode binary not found: ${AGSYS_OPENCODE_BIN}"
[[ -x "${OPENCODE_BIN}" ]] || agsys_die "OpenCode binary is not executable: ${OPENCODE_BIN}"

SESSION_TMP_ROOT="$(mktemp -d "/tmp/agsys-opencode-XXXXXXXX")"
MEMORY_CACHE_DIR="${SESSION_TMP_ROOT}/memory-cache"
SESSION_BIN_DIR="${SESSION_TMP_ROOT}/bin"
mkdir -p "${MEMORY_CACHE_DIR}" "${SESSION_BIN_DIR}"

cleanup() {
  rm -rf "${SESSION_TMP_ROOT}"
}
trap cleanup EXIT

cp "${AGSYS_EVAL_ROOT}/memory/image_metadata.json" "${MEMORY_CACHE_DIR}/image_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/video_metadata.json" "${MEMORY_CACHE_DIR}/video_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/emails.json" "${MEMORY_CACHE_DIR}/emails.json"
cp "${AGSYS_EVAL_ROOT}/memory/memory_variant.json" "${MEMORY_CACHE_DIR}/memory_variant.json" 2>/dev/null || true
cp "${OPENCODE_BIN}" "${SESSION_BIN_DIR}/opencode"
chmod +x "${SESSION_BIN_DIR}/opencode"
RAW_IMAGES_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${AGSYS_EVAL_ROOT}/memory/raw_images")"
RAW_VIDEOS_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${AGSYS_EVAL_ROOT}/memory/raw_videos")"

REAL_AUTH="${XDG_DATA_HOME:-${HOME}/.local/share}/opencode/auth.json"
SANDBOX_REQUESTED="${AGSYS_OPENCODE_SANDBOX:-bwrap}"
SANDBOX_ACTIVE="off"
SANDBOX_STATUS="disabled"
BWRAP_BIN=""

if [[ "${SANDBOX_REQUESTED}" != "off" ]]; then
  BWRAP_BIN="$(command -v bwrap 2>/dev/null || true)"
  if [[ -z "${BWRAP_BIN}" ]]; then
    SANDBOX_STATUS="bwrap-not-found"
    if [[ "${SANDBOX_REQUESTED}" == "bwrap" ]]; then
      agsys_die "AGSYS_OPENCODE_SANDBOX=bwrap but bwrap is not installed"
    fi
  elif "${BWRAP_BIN}" \
    --ro-bind /usr /usr \
    --ro-bind /bin /bin \
    --ro-bind /lib /lib \
    --ro-bind /lib64 /lib64 \
    --proc /proc \
    --dev /dev \
    /bin/sh -c 'exit 0' >/dev/null 2>&1; then
    SANDBOX_ACTIVE="bwrap"
    SANDBOX_STATUS="available"
  else
    SANDBOX_STATUS="bwrap-unavailable"
    if [[ "${SANDBOX_REQUESTED}" == "bwrap" ]]; then
      agsys_die "AGSYS_OPENCODE_SANDBOX=bwrap but bwrap could not start on this machine"
    fi
  fi
fi

# ── Forward selected env vars into the isolated runtime so opencode.json
# `{env:...}` placeholders (e.g. an OpenAI-compatible API key) resolve inside the
# sandbox. OPENAI_COMPATIBLE_API_KEY is always forwarded when set; add more via
# AGSYS_OPENCODE_FORWARD_ENV="VAR1 VAR2 ...". ─────────────────────────────────
OPENCODE_FORWARD_ENV_NAMES=()
declare -A _OC_FORWARD_SEEN=()
_oc_register_forward() {
  local var="$1"
  [[ -n "${var}" ]] || return 0
  [[ -n "${!var:-}" ]] || return 0
  [[ -z "${_OC_FORWARD_SEEN[$var]:-}" ]] || return 0
  _OC_FORWARD_SEEN[$var]=1
  OPENCODE_FORWARD_ENV_NAMES+=("${var}")
}
_oc_register_forward "OPENAI_COMPATIBLE_API_KEY"
if [[ -n "${AGSYS_OPENCODE_FORWARD_ENV:-}" ]]; then
  for extra in ${AGSYS_OPENCODE_FORWARD_ENV}; do
    _oc_register_forward "${extra}"
  done
fi

echo "INFO: opencode memory_mode=${MEMORY_MODE} sandbox requested=${SANDBOX_REQUESTED} active=${SANDBOX_ACTIVE} status=${SANDBOX_STATUS}"
if [[ ${#OPENCODE_FORWARD_ENV_NAMES[@]} -gt 0 ]]; then
  echo "INFO: opencode forwarding env vars=${OPENCODE_FORWARD_ENV_NAMES[*]}"
fi

while IFS= read -r QID; do
  RUN_DIR="$(agsys_setup_run_dir "opencode" "${MODEL_TAG}" "${QID}")"
  TRACE_FILE="${RUN_DIR}/output/trace.jsonl"
  ANSWER_FILE="${RUN_DIR}/output/answer.json"
  USAGE_FILE="${RUN_DIR}/output/usage.json"
  STDERR_LOG="${RUN_DIR}/output/stderr.log"
  RUNTIME_MANIFEST="${RUN_DIR}/output/runtime_manifest.json"
  RUNTIME_CONFIG_SNAPSHOT="${RUN_DIR}/output/opencode_runtime_config.json"

  if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    if [[ ! -f "${USAGE_FILE}" ]] && [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_usage.py \
        --format opencode \
        --trace "${TRACE_FILE}" \
        --out "${USAGE_FILE}" \
        --agent opencode \
        --model-tag "${MODEL_TAG}" \
        --model "${AGSYS_OPENCODE_MODEL:-}" \
        >> "${STDERR_LOG}" 2>&1 || true
    fi
    echo "SKIP: ${QID} already answered (${ANSWER_FILE})"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  rm -f "${ANSWER_FILE}"
  rm -f "${TRACE_FILE}"
  rm -f "${USAGE_FILE}"
  rm -f "${RUNTIME_MANIFEST}"
  rm -f "${RUNTIME_CONFIG_SNAPSHOT}"
  rm -rf "${RUN_DIR}/opencode-logs" "${RUN_DIR}/workspace-output"

  QTMP_ROOT="$(mktemp -d "${SESSION_TMP_ROOT}/${QID}.XXXXXXXX")"
  WORKSPACE_DIR="${QTMP_ROOT}/workspace"
  RUNTIME_HOME="${QTMP_ROOT}/home"
  mkdir -p \
    "${WORKSPACE_DIR}/memory" \
    "${WORKSPACE_DIR}/output" \
    "${RUNTIME_HOME}/bin" \
    "${RUNTIME_HOME}/.config/opencode" \
    "${RUNTIME_HOME}/.local/share" \
    "${RUNTIME_HOME}/.local/state" \
    "${RUNTIME_HOME}/.cache"

  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.json" "${WORKSPACE_DIR}/question.json"
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.txt" "${WORKSPACE_DIR}/question.txt"
  cp "${MEMORY_CACHE_DIR}/image_metadata.json" "${WORKSPACE_DIR}/memory/image_metadata.json"
  cp "${MEMORY_CACHE_DIR}/video_metadata.json" "${WORKSPACE_DIR}/memory/video_metadata.json"
  cp "${MEMORY_CACHE_DIR}/emails.json" "${WORKSPACE_DIR}/memory/emails.json"
  if [[ -f "${MEMORY_CACHE_DIR}/memory_variant.json" ]]; then
    cp "${MEMORY_CACHE_DIR}/memory_variant.json" "${WORKSPACE_DIR}/memory/memory_variant.json"
  fi
  if [[ "${MEMORY_MODE}" == "raw" ]]; then
    if [[ "${SANDBOX_ACTIVE}" == "bwrap" ]]; then
      mkdir -p "${WORKSPACE_DIR}/memory/raw_images" "${WORKSPACE_DIR}/memory/raw_videos"
    else
      ln -sfn "${RAW_IMAGES_DIR}" "${WORKSPACE_DIR}/memory/raw_images"
      ln -sfn "${RAW_VIDEOS_DIR}" "${WORKSPACE_DIR}/memory/raw_videos"
    fi
  fi
  cp "${SESSION_BIN_DIR}/opencode" "${RUNTIME_HOME}/bin/opencode"

  if [[ -n "${AGSYS_OPENCODE_CONFIG_DIR}" ]]; then
    [[ -d "${AGSYS_OPENCODE_CONFIG_DIR}" ]] || agsys_die "Missing AGSYS_OPENCODE_CONFIG_DIR=${AGSYS_OPENCODE_CONFIG_DIR}"
    cp -R "${AGSYS_OPENCODE_CONFIG_DIR}/." "${RUNTIME_HOME}/.config/opencode/"
  else
    if ! python3 agent_systems/runtime_artifacts.py write-opencode-config \
      --source "${AGSYS_OPENCODE_CONFIG_SOURCE}" \
      --model "${AGSYS_OPENCODE_MODEL:-}" \
      --out "${RUNTIME_HOME}/.config/opencode/opencode.json" \
      >> "${STDERR_LOG}" 2>&1; then
      echo "FAILED: ${QID} (could not build isolated OpenCode config)"
      FAILED=$((FAILED + 1))
      rm -rf "${QTMP_ROOT}"
      echo ""
      continue
    fi
  fi

  if [[ -f "${REAL_AUTH}" ]]; then
    mkdir -p "${RUNTIME_HOME}/.local/share/opencode"
    cp "${REAL_AUTH}" "${RUNTIME_HOME}/.local/share/opencode/auth.json"
  fi

  if [[ -f "${RUNTIME_HOME}/.config/opencode/opencode.json" ]]; then
    cp "${RUNTIME_HOME}/.config/opencode/opencode.json" "${RUNTIME_CONFIG_SNAPSHOT}"
  fi

  python3 - <<'PY' \
    "${RUNTIME_MANIFEST}" \
    "${QID}" \
    "${RUN_DIR}" \
    "${WORKSPACE_DIR}" \
    "${AGSYS_EVAL_ROOT}/qas/${QID}" \
    "${AGSYS_OPENCODE_MODEL:-}" \
    "${AGSYS_OPENCODE_VARIANT:-}" \
    "${OPENCODE_BIN}" \
    "${SANDBOX_REQUESTED}" \
    "${SANDBOX_ACTIVE}" \
    "${SANDBOX_STATUS}" \
    "${AGSYS_OPENCODE_CONFIG_SOURCE}" \
    "${AGSYS_OPENCODE_CONFIG_DIR}" \
    "${REAL_AUTH}" \
    "${RUNTIME_CONFIG_SNAPSHOT}" \
    "${MEMORY_MODE}" \
    "${MODEL_TAG}"
import json, sys
out = sys.argv[1]
payload = {
    "agent": "opencode",
    "question_id": sys.argv[2],
    "run_dir": sys.argv[3],
    "workspace_mode": "copied question files + copied temp memory cache",
    "workspace_host_path": sys.argv[4],
    "question_source_dir": sys.argv[5],
    "model": sys.argv[6],
    "variant": sys.argv[7],
    "model_tag": sys.argv[17],
    "memory_mode": sys.argv[16],
    "opencode_bin": sys.argv[8],
    "sandbox_requested": sys.argv[9],
    "sandbox_active": sys.argv[10],
    "sandbox_status": sys.argv[11],
    "config_source": sys.argv[12],
    "config_override_dir": sys.argv[13],
    "auth_source": sys.argv[14],
    "runtime_config_snapshot": sys.argv[15],
    "tmp_workspace_deleted_after_run": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  SYSTEM_PROMPT="$(cat "${AGSYS_SYSTEM_PROMPT}")"
  QUESTION_TEXT="$(cat "${WORKSPACE_DIR}/question.txt")"
  MESSAGE="$(printf '%s\n\nQuestion:\n%s' "${SYSTEM_PROMPT}" "${QUESTION_TEXT}")"

  echo "── [opencode/${MODEL_TAG}] ${QID}"

  COMMON_ENV=(
    HOME="${RUNTIME_HOME}"
    XDG_CONFIG_HOME="${RUNTIME_HOME}/.config"
    XDG_DATA_HOME="${RUNTIME_HOME}/.local/share"
    XDG_STATE_HOME="${RUNTIME_HOME}/.local/state"
    XDG_CACHE_HOME="${RUNTIME_HOME}/.cache"
    OPENCODE_CONFIG_DIR="${RUNTIME_HOME}/.config/opencode"
    OPENCODE_DISABLE_MODELS_FETCH=1
    OPENCODE_DISABLE_CLAUDE_CODE=1
    OPENCODE_ENABLE_EXA=0
  )
  for fwd in "${OPENCODE_FORWARD_ENV_NAMES[@]}"; do
    [[ -n "${!fwd:-}" ]] || continue
    COMMON_ENV+=("${fwd}=${!fwd}")
  done

  CMD=()
  if [[ "${SANDBOX_ACTIVE}" == "bwrap" ]]; then
    CMD=(
      "${BWRAP_BIN}"
      --die-with-parent
      --proc /proc
      --dev /dev
      --tmpfs /tmp
    )
    for SYS_PATH in /usr /bin /lib /lib64 /etc /run /sbin; do
      if [[ -e "${SYS_PATH}" ]]; then
        CMD+=(--ro-bind "${SYS_PATH}" "${SYS_PATH}")
      fi
    done
    CMD+=(
      --bind "${RUNTIME_HOME}" /home/opencode
      --bind "${WORKSPACE_DIR}" /workspace
    )
    if [[ "${MEMORY_MODE}" == "raw" ]]; then
      CMD+=(
        --ro-bind "${RAW_IMAGES_DIR}" /workspace/memory/raw_images
        --ro-bind "${RAW_VIDEOS_DIR}" /workspace/memory/raw_videos
      )
    fi
    CMD+=(
      --chdir /workspace
      env
      HOME=/home/opencode
      XDG_CONFIG_HOME=/home/opencode/.config
      XDG_DATA_HOME=/home/opencode/.local/share
      XDG_STATE_HOME=/home/opencode/.local/state
      XDG_CACHE_HOME=/home/opencode/.cache
      OPENCODE_CONFIG_DIR=/home/opencode/.config/opencode
      OPENCODE_DISABLE_MODELS_FETCH=1
      OPENCODE_DISABLE_CLAUDE_CODE=1
      OPENCODE_ENABLE_EXA=0
    )
    for fwd in "${OPENCODE_FORWARD_ENV_NAMES[@]}"; do
      [[ -n "${!fwd:-}" ]] || continue
      CMD+=("${fwd}=${!fwd}")
    done
    CMD+=(
      /home/opencode/bin/opencode
      run
      --format
      json
      --dir
      /workspace
    )
  else
    CMD=(
      env
      "${COMMON_ENV[@]}"
      "${RUNTIME_HOME}/bin/opencode"
      run
      --format
      json
      --dir
      "${WORKSPACE_DIR}"
    )
  fi

  if [[ -n "${AGSYS_OPENCODE_MODEL}" ]]; then
    CMD+=(--model "${AGSYS_OPENCODE_MODEL}")
  fi
  if [[ -n "${AGSYS_OPENCODE_VARIANT}" ]]; then
    CMD+=(--variant "${AGSYS_OPENCODE_VARIANT}")
  fi
  CMD+=("${MESSAGE}")

  AGSYS_TIMEOUT="${AGSYS_OPENCODE_TIMEOUT:-900}"
  if [[ "${AGSYS_TIMEOUT}" == "0" ]]; then
    "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
    EXIT_CODE=$?
  else
    timeout --kill-after=30 "${AGSYS_TIMEOUT}" \
      "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 124 ]] || [[ ${EXIT_CODE} -eq 137 ]]; then
      echo "TIMEOUT: opencode exceeded ${AGSYS_TIMEOUT}s limit (exit=${EXIT_CODE})"
    fi
  fi

  if [[ -d "${RUNTIME_HOME}/.local/share/opencode/log" ]]; then
    mkdir -p "${RUN_DIR}/opencode-logs"
    cp -r "${RUNTIME_HOME}/.local/share/opencode/log/." "${RUN_DIR}/opencode-logs/" 2>/dev/null || true
  fi

  if compgen -G "${WORKSPACE_DIR}/output/*" >/dev/null; then
    mkdir -p "${RUN_DIR}/workspace-output"
    cp -r "${WORKSPACE_DIR}/output/." "${RUN_DIR}/workspace-output/" 2>/dev/null || true
  fi

  if [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_usage.py \
      --format opencode \
      --trace "${TRACE_FILE}" \
      --out "${USAGE_FILE}" \
      --agent opencode \
      --model-tag "${MODEL_TAG}" \
      --model "${AGSYS_OPENCODE_MODEL:-}" \
      >> "${STDERR_LOG}" 2>&1 || true
  fi

  TRACE_ERROR=""
  if [[ -f "${TRACE_FILE}" ]]; then
    TRACE_ERROR="$(python3 -c '
import json, sys
for line in open(sys.argv[1], "r", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    if ev.get("type") == "error":
        err = ev.get("error", {})
        msg = err.get("data", {}).get("message", "") or err.get("name", "unknown error")
        print(msg)
        break
' "${TRACE_FILE}" 2>/dev/null || true)"
  fi

  if [[ -n "${TRACE_ERROR}" ]]; then
    echo "ERROR (opencode): ${TRACE_ERROR}"
    echo "FAILED: ${QID} (exit=${EXIT_CODE})"
    FAILED=$((FAILED + 1))
  elif [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: opencode exited with code ${EXIT_CODE}"
    if [[ -f "${STDERR_LOG}" ]]; then
      echo "--- stderr (last 5 lines) ---"
      tail -5 "${STDERR_LOG}"
      echo "---"
    fi
    echo "FAILED: ${QID} (exit=${EXIT_CODE})"
    FAILED=$((FAILED + 1))
  else
    if [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_answer.py \
        --format opencode \
        --trace "${TRACE_FILE}" \
        --out "${ANSWER_FILE}" \
        --expected-id "${QID}" \
        --question-file "${WORKSPACE_DIR}/question.json" \
        >> "${STDERR_LOG}" 2>&1
      EXTRACT_EXIT=$?
      if [[ ${EXTRACT_EXIT} -ne 0 ]]; then
        echo "ERROR: extract_answer.py failed (exit=${EXTRACT_EXIT})"
        if [[ -f "${STDERR_LOG}" ]]; then
          echo "--- stderr (last 5 lines) ---"
          tail -5 "${STDERR_LOG}"
          echo "---"
        fi
      fi
    fi

    if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
      PREVIEW="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1],"r",encoding="utf-8")); print(d.get("answer","")[:120])' "${ANSWER_FILE}" 2>/dev/null || true)"
      echo "A: ${PREVIEW}"
      COMPLETED=$((COMPLETED + 1))
    else
      echo "FAILED: ${QID} (exit=${EXIT_CODE})"
      if [[ -f "${STDERR_LOG}" ]] && [[ ! -f "${ANSWER_FILE}" ]]; then
        echo "--- stderr (last 5 lines) ---"
        tail -5 "${STDERR_LOG}"
        echo "---"
      fi
      FAILED=$((FAILED + 1))
    fi
  fi

  rm -rf "${QTMP_ROOT}"
  echo ""
done < <(agsys_list_qids "${SINGLE_QID}")

echo "=== opencode summary: ${COMPLETED} completed, ${FAILED} failed ==="

if [[ -z "${SINGLE_QID}" ]]; then
  echo "=== Collecting: opencode/${MODEL_TAG} ==="
  python3 agent_systems/collect_results.py --agent opencode --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"
  python3 agent_systems/collect_usage.py --agent opencode --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"

  if [[ -n "${AGSYS_SKIP_EVAL:-}" ]]; then
    echo "SKIP: evaluation (AGSYS_SKIP_EVAL=1)"
  else
    PREDICTIONS="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/opencode/${MODEL_TAG}/answers.jsonl"
    EVAL_DIR="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/opencode/${MODEL_TAG}/eval"
    agsys_evaluate_predictions "${PREDICTIONS}" "${EVAL_DIR}"
    EVAL_EXIT=$?
    if [[ ${EVAL_EXIT} -ne 0 ]]; then
      echo "WARN: evaluate_qa.py failed (exit=${EVAL_EXIT}). Set AGSYS_SKIP_EVAL=1 to skip."
    fi
  fi
fi
