#!/usr/bin/env bash
#
# Run pi (@earendil-works/pi-coding-agent) on ATM-hard questions
# (one process per question).
#
# Usage:
#   bash agent_systems/scripts/pi/run_pi.sh              # all questions
#   bash agent_systems/scripts/pi/run_pi.sh <question_id>
#
# Env overrides (see agent_systems/config.py):
#   AGSYS_RUN_TAG, AGSYS_PI_BIN, AGSYS_PI_MODEL, AGSYS_PI_MODEL_TAG
#   AGSYS_PI_PROVIDER, AGSYS_PI_THINKING (off|minimal|low|medium|high|xhigh)
#   AGSYS_PI_TOOLS=read,bash,grep,find,ls
#   AGSYS_MEMORY_MODE=sgm|raw|descriptive
#   AGSYS_PI_SANDBOX=bwrap|auto|off
#   AGSYS_PI_OFFLINE=1
#   AGSYS_PI_CONFIG_SOURCE_DIR=~/.pi/agent
#   AGSYS_PI_TIMEOUT_S=900  (0 to disable)
#   AGSYS_PI_EXTENSION_PATHS=path1:path2   (colon-separated, passed as -e to pi)
#   AGSYS_PI_API_KEYS_DIR=api_keys         (sourced for *_key files → env vars)
#   AGSYS_PI_FORWARD_ENV="VAR1 VAR2 ..."   (extra env vars to forward into bwrap)
#   AGSYS_EVAL_EXTRA_FLAGS, AGSYS_EVAL_GLM_FLAGS, AGSYS_EVAL_GPT_FLAGS
#   AGSYS_SKIP_EVAL=1
#

set -o pipefail

# Default to GPT-only evaluation for pi runs. The GLM judge endpoint depends on
# a private LAN vLLM box that isn't always reachable; the GPT judge is the
# authoritative baseline anyway. Override with AGSYS_EVAL_ENABLE_GLM_JUDGE=1
# in the environment if you also want the GLM pass.
export AGSYS_EVAL_ENABLE_GLM_JUDGE="${AGSYS_EVAL_ENABLE_GLM_JUDGE:-0}"

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

agsys_require_eval_root

SINGLE_QID="${1:-}"
MEMORY_MODE="$(agsys_normalize_memory_mode "${AGSYS_MEMORY_MODE:-sgm}")"
MODEL_TAG_BASE="${AGSYS_PI_MODEL_TAG:-${AGSYS_PI_MODEL:-default}}"
MODEL_TAG="$(agsys_tag_with_memory_mode "${MODEL_TAG_BASE}" "${MEMORY_MODE}")"

COMPLETED=0
FAILED=0

PI_BIN="${AGSYS_PI_BIN}"
if [[ "${PI_BIN}" != /* ]]; then
  PI_BIN="$(command -v "${PI_BIN}" 2>/dev/null || true)"
fi
[[ -n "${PI_BIN}" ]] || agsys_die "pi binary not found: ${AGSYS_PI_BIN}"
[[ -x "${PI_BIN}" ]] || agsys_die "pi binary is not executable: ${PI_BIN}"

# Pi is typically an npm-global symlink that points into a sibling `lib/node_modules/...`
# tree (~175 MB). For the bwrap sandbox we need to bind-mount the smallest ancestor of
# `pi` that also contains the resolved package files, so the symlink resolves inside
# the sandbox. Computed once per session and reused for every per-question invocation.
PI_BIN_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${PI_BIN}")"
PI_REAL="$(readlink -f "${PI_BIN}" 2>/dev/null || echo "${PI_BIN_ABS}")"
PI_PACKAGE_ROOT="$(python3 - "${PI_BIN_ABS}" "${PI_REAL}" <<'PY'
import os, sys
bin_path = os.path.abspath(sys.argv[1])
real_path = os.path.abspath(sys.argv[2])
candidate = os.path.dirname(bin_path)
real_norm = real_path.rstrip("/") + "/"
while candidate != "/":
    if (real_norm.startswith(candidate.rstrip("/") + "/")) or real_path == candidate:
        break
    parent = os.path.dirname(candidate)
    if parent == candidate:
        break
    candidate = parent
print(candidate)
PY
)"
if [[ -z "${PI_PACKAGE_ROOT}" ]] || [[ ! -d "${PI_PACKAGE_ROOT}" ]]; then
  PI_PACKAGE_ROOT="$(dirname "$(dirname "${PI_BIN_ABS}")")"
fi

PI_VERSION="$("${PI_BIN}" --version 2>&1 | head -n1 | tr -d '\r' || true)"

# ── Provider credentials: source per-key files from api_keys/ ──────────────
# Convention: api_keys/.<provider>_key contents are exported as
# <PROVIDER>_API_KEY (e.g. api_keys/.openai_compatible_key → OPENAI_COMPATIBLE_API_KEY).
# Existing env vars always win — set OPENAI_COMPATIBLE_API_KEY in your shell to
# override the file value.
PI_API_KEYS_DIR="${AGSYS_PI_API_KEYS_DIR:-api_keys}"
declare -A _PI_FORWARD_SEEN=()
PI_FORWARD_ENV_NAMES=()
_pi_register_forward() {
  local var="$1"
  [[ -n "${var}" ]] || return 0
  [[ -n "${!var:-}" ]] || return 0
  [[ -z "${_PI_FORWARD_SEEN[$var]:-}" ]] || return 0
  _PI_FORWARD_SEEN[$var]=1
  PI_FORWARD_ENV_NAMES+=("${var}")
}
if [[ -d "${PI_API_KEYS_DIR}" ]]; then
  shopt -s nullglob
  # Match hidden and visible *_key files separately so dotglob can stay off
  # (otherwise each hidden file matches twice and we'd double-forward env vars).
  for keyfile in "${PI_API_KEYS_DIR}"/.*_key "${PI_API_KEYS_DIR}"/*_key; do
    [[ -f "${keyfile}" ]] || continue
    base="$(basename "${keyfile}")"
    base="${base#.}"               # strip leading dot
    base="${base%_key}"            # strip trailing _key
    [[ -n "${base}" ]] || continue
    upper="$(echo "${base}" | tr '[:lower:]-' '[:upper:]_')"
    var="${upper}_API_KEY"
    if [[ -z "${!var:-}" ]]; then
      val="$(tr -d '\r\n' < "${keyfile}" 2>/dev/null || true)"
      if [[ -n "${val}" ]]; then
        export "${var}=${val}"
      fi
    fi
    _pi_register_forward "${var}"
  done
  shopt -u nullglob
fi
if [[ -n "${AGSYS_PI_FORWARD_ENV:-}" ]]; then
  for extra in ${AGSYS_PI_FORWARD_ENV}; do
    _pi_register_forward "${extra}"
  done
fi

# ── Extension paths (each forwarded as -e to pi, ro-bound into bwrap) ──────
PI_EXTENSION_PATHS_ABS=()
if [[ -n "${AGSYS_PI_EXTENSION_PATHS:-}" ]]; then
  IFS=':' read -r -a _PI_EXT_RAW <<< "${AGSYS_PI_EXTENSION_PATHS}"
  for ext in "${_PI_EXT_RAW[@]}"; do
    [[ -n "${ext}" ]] || continue
    ext_abs="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ext}")"
    if [[ ! -e "${ext_abs}" ]]; then
      agsys_die "AGSYS_PI_EXTENSION_PATHS entry not found: ${ext} (resolved to ${ext_abs})"
    fi
    PI_EXTENSION_PATHS_ABS+=("${ext_abs}")
  done
fi

SESSION_TMP_ROOT="$(mktemp -d "/tmp/agsys-pi-XXXXXXXX")"
MEMORY_CACHE_DIR="${SESSION_TMP_ROOT}/memory-cache"
mkdir -p "${MEMORY_CACHE_DIR}"

cleanup() {
  rm -rf "${SESSION_TMP_ROOT}"
}
trap cleanup EXIT

cp "${AGSYS_EVAL_ROOT}/memory/image_metadata.json" "${MEMORY_CACHE_DIR}/image_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/video_metadata.json" "${MEMORY_CACHE_DIR}/video_metadata.json"
cp "${AGSYS_EVAL_ROOT}/memory/emails.json" "${MEMORY_CACHE_DIR}/emails.json"
cp "${AGSYS_EVAL_ROOT}/memory/memory_variant.json" "${MEMORY_CACHE_DIR}/memory_variant.json" 2>/dev/null || true

RAW_IMAGES_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${AGSYS_EVAL_ROOT}/memory/raw_images")"
RAW_VIDEOS_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${AGSYS_EVAL_ROOT}/memory/raw_videos")"

PI_CONFIG_SOURCE_DIR="${AGSYS_PI_CONFIG_SOURCE_DIR}"
SANDBOX_REQUESTED="${AGSYS_PI_SANDBOX:-bwrap}"
SANDBOX_ACTIVE="off"
SANDBOX_STATUS="disabled"
BWRAP_BIN=""

if [[ "${SANDBOX_REQUESTED}" != "off" ]]; then
  BWRAP_BIN="$(command -v bwrap 2>/dev/null || true)"
  if [[ -z "${BWRAP_BIN}" ]]; then
    SANDBOX_STATUS="bwrap-not-found"
    if [[ "${SANDBOX_REQUESTED}" == "bwrap" ]]; then
      agsys_die "AGSYS_PI_SANDBOX=bwrap but bwrap is not installed"
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
      agsys_die "AGSYS_PI_SANDBOX=bwrap but bwrap could not start on this machine"
    fi
  fi
fi

echo "INFO: pi memory_mode=${MEMORY_MODE} sandbox requested=${SANDBOX_REQUESTED} active=${SANDBOX_ACTIVE} status=${SANDBOX_STATUS} model=${AGSYS_PI_MODEL:-<default>} thinking=${AGSYS_PI_THINKING:-medium}"
echo "INFO: pi bin=${PI_BIN} package_root=${PI_PACKAGE_ROOT} version=${PI_VERSION:-unknown}"
if [[ ${#PI_EXTENSION_PATHS_ABS[@]} -gt 0 ]]; then
  echo "INFO: pi extensions=${PI_EXTENSION_PATHS_ABS[*]}"
fi
if [[ ${#PI_FORWARD_ENV_NAMES[@]} -gt 0 ]]; then
  echo "INFO: pi forwarding env vars=${PI_FORWARD_ENV_NAMES[*]}"
fi

while IFS= read -r QID; do
  RUN_DIR="$(agsys_setup_run_dir "pi" "${MODEL_TAG}" "${QID}")"
  TRACE_FILE="${RUN_DIR}/output/trace.jsonl"
  ANSWER_FILE="${RUN_DIR}/output/answer.json"
  USAGE_FILE="${RUN_DIR}/output/usage.json"
  STDERR_LOG="${RUN_DIR}/output/stderr.log"
  RUNTIME_MANIFEST="${RUN_DIR}/output/runtime_manifest.json"

  if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    if [[ ! -f "${USAGE_FILE}" ]] && [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_usage.py \
        --format pi \
        --trace "${TRACE_FILE}" \
        --out "${USAGE_FILE}" \
        --agent pi \
        --model-tag "${MODEL_TAG}" \
        --model "${AGSYS_PI_MODEL:-}" \
        >> "${STDERR_LOG}" 2>&1 || true
    fi
    echo "SKIP: ${QID} already answered (${ANSWER_FILE})"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  rm -f "${ANSWER_FILE}" "${TRACE_FILE}" "${USAGE_FILE}" "${STDERR_LOG}" "${RUNTIME_MANIFEST}"
  rm -rf "${RUN_DIR}/pi-logs" "${RUN_DIR}/workspace-output"

  QTMP_ROOT="$(mktemp -d "${SESSION_TMP_ROOT}/${QID}.XXXXXXXX")"
  WORKSPACE_DIR="${QTMP_ROOT}/workspace"
  WORKSPACE_OUTPUT_DIR="${WORKSPACE_DIR}/output"
  RUNTIME_HOME="${QTMP_ROOT}/home"
  PI_HOME_DIR="${RUNTIME_HOME}/.pi/agent"
  mkdir -p \
    "${WORKSPACE_DIR}/memory" \
    "${WORKSPACE_OUTPUT_DIR}" \
    "${PI_HOME_DIR}/sessions" \
    "${RUNTIME_HOME}/.config" \
    "${RUNTIME_HOME}/.local/share" \
    "${RUNTIME_HOME}/.local/state" \
    "${RUNTIME_HOME}/.cache"

  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.json" "${WORKSPACE_DIR}/question.json"
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.txt"  "${WORKSPACE_DIR}/question.txt"
  cp "${MEMORY_CACHE_DIR}/image_metadata.json" "${WORKSPACE_DIR}/memory/image_metadata.json"
  cp "${MEMORY_CACHE_DIR}/video_metadata.json" "${WORKSPACE_DIR}/memory/video_metadata.json"
  cp "${MEMORY_CACHE_DIR}/emails.json"         "${WORKSPACE_DIR}/memory/emails.json"
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

  if [[ -d "${PI_CONFIG_SOURCE_DIR}" ]]; then
    for f in auth.json settings.json; do
      if [[ -f "${PI_CONFIG_SOURCE_DIR}/${f}" ]]; then
        cp "${PI_CONFIG_SOURCE_DIR}/${f}" "${PI_HOME_DIR}/${f}"
      fi
    done
  fi

  SYSTEM_PROMPT="$(cat "${AGSYS_SYSTEM_PROMPT}")"
  QUESTION_TEXT="$(cat "${WORKSPACE_DIR}/question.txt")"
  MESSAGE="$(printf '%s\n\nQuestion:\n%s' "${SYSTEM_PROMPT}" "${QUESTION_TEXT}")"

  echo "── [pi/${MODEL_TAG}] ${QID}"

  PI_ARGS=(
    -p
    --mode json
    --no-session
    --no-extensions
    --no-skills
    --no-prompt-templates
    --no-themes
    --no-context-files
  )
  if [[ "${AGSYS_PI_OFFLINE:-1}" != "0" ]]; then
    PI_ARGS+=(--offline)
  fi
  if [[ -n "${AGSYS_PI_TOOLS:-}" ]]; then
    PI_ARGS+=(--tools "${AGSYS_PI_TOOLS}")
  fi
  if [[ -n "${AGSYS_PI_PROVIDER:-}" ]]; then
    PI_ARGS+=(--provider "${AGSYS_PI_PROVIDER}")
  fi
  if [[ -n "${AGSYS_PI_MODEL:-}" ]]; then
    PI_ARGS+=(--model "${AGSYS_PI_MODEL}")
  fi
  if [[ -n "${AGSYS_PI_THINKING:-}" ]]; then
    PI_ARGS+=(--thinking "${AGSYS_PI_THINKING}")
  fi
  for ext_abs in "${PI_EXTENSION_PATHS_ABS[@]}"; do
    PI_ARGS+=(-e "${ext_abs}")
  done
  PI_ARGS+=("${MESSAGE}")

  COMMON_ENV=(
    HOME="${RUNTIME_HOME}"
    XDG_CONFIG_HOME="${RUNTIME_HOME}/.config"
    XDG_DATA_HOME="${RUNTIME_HOME}/.local/share"
    XDG_STATE_HOME="${RUNTIME_HOME}/.local/state"
    XDG_CACHE_HOME="${RUNTIME_HOME}/.cache"
    PI_CODING_AGENT_DIR="${PI_HOME_DIR}"
    PI_CODING_AGENT_SESSION_DIR="${PI_HOME_DIR}/sessions"
    PI_TELEMETRY=0
  )
  if [[ "${AGSYS_PI_OFFLINE:-1}" != "0" ]]; then
    COMMON_ENV+=(PI_OFFLINE=1)
  fi
  for fwd in "${PI_FORWARD_ENV_NAMES[@]}"; do
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
      --bind "${RUNTIME_HOME}" /home/pi
      --bind "${WORKSPACE_DIR}" /workspace
      --ro-bind "${PI_PACKAGE_ROOT}" "${PI_PACKAGE_ROOT}"
    )
    for ext_abs in "${PI_EXTENSION_PATHS_ABS[@]}"; do
      CMD+=(--ro-bind "${ext_abs}" "${ext_abs}")
    done
    if [[ "${MEMORY_MODE}" == "raw" ]]; then
      CMD+=(
        --ro-bind "${RAW_IMAGES_DIR}" /workspace/memory/raw_images
        --ro-bind "${RAW_VIDEOS_DIR}" /workspace/memory/raw_videos
      )
    fi
    CMD+=(
      --chdir /workspace
      env
      HOME=/home/pi
      XDG_CONFIG_HOME=/home/pi/.config
      XDG_DATA_HOME=/home/pi/.local/share
      XDG_STATE_HOME=/home/pi/.local/state
      XDG_CACHE_HOME=/home/pi/.cache
      PI_CODING_AGENT_DIR=/home/pi/.pi/agent
      PI_CODING_AGENT_SESSION_DIR=/home/pi/.pi/agent/sessions
      PI_TELEMETRY=0
    )
    if [[ "${AGSYS_PI_OFFLINE:-1}" != "0" ]]; then
      CMD+=(PI_OFFLINE=1)
    fi
    for fwd in "${PI_FORWARD_ENV_NAMES[@]}"; do
      [[ -n "${!fwd:-}" ]] || continue
      CMD+=("${fwd}=${!fwd}")
    done
    CMD+=("${PI_BIN}" "${PI_ARGS[@]}")
  else
    CMD=(
      env
      "${COMMON_ENV[@]}"
      "${PI_BIN}"
      "${PI_ARGS[@]}"
    )
  fi

  # Build runtime manifest BEFORE running so that audit info exists even on timeout.
  python3 - <<'PY' \
    "${RUNTIME_MANIFEST}" \
    "${QID}" \
    "${RUN_DIR}" \
    "${WORKSPACE_DIR}" \
    "${AGSYS_EVAL_ROOT}/qas/${QID}" \
    "${AGSYS_PI_MODEL:-}" \
    "${AGSYS_PI_PROVIDER:-}" \
    "${AGSYS_PI_THINKING:-}" \
    "${AGSYS_PI_TOOLS:-}" \
    "${PI_BIN}" \
    "${PI_VERSION}" \
    "${SANDBOX_REQUESTED}" \
    "${SANDBOX_ACTIVE}" \
    "${SANDBOX_STATUS}" \
    "${PI_CONFIG_SOURCE_DIR}" \
    "${PI_HOME_DIR}" \
    "${MEMORY_MODE}" \
    "${MODEL_TAG}"
import json, sys
out = sys.argv[1]
payload = {
    "agent": "pi",
    "question_id": sys.argv[2],
    "run_dir": sys.argv[3],
    "workspace_mode": "copied question files + copied temp memory cache",
    "workspace_host_path": sys.argv[4],
    "question_source_dir": sys.argv[5],
    "model": sys.argv[6],
    "provider": sys.argv[7],
    "thinking": sys.argv[8],
    "tools": sys.argv[9],
    "pi_bin": sys.argv[10],
    "pi_version": sys.argv[11],
    "sandbox_requested": sys.argv[12],
    "sandbox_active": sys.argv[13],
    "sandbox_status": sys.argv[14],
    "pi_config_source_dir": sys.argv[15],
    "pi_home_dir": sys.argv[16],
    "memory_mode": sys.argv[17],
    "model_tag": sys.argv[18],
    "tmp_workspace_deleted_after_run": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  AGSYS_TIMEOUT="${AGSYS_PI_TIMEOUT_S:-900}"
  if [[ "${AGSYS_TIMEOUT}" == "0" ]]; then
    "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
    EXIT_CODE=$?
  else
    timeout --kill-after=30 "${AGSYS_TIMEOUT}" \
      "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 124 ]] || [[ ${EXIT_CODE} -eq 137 ]]; then
      echo "TIMEOUT: pi exceeded ${AGSYS_TIMEOUT}s limit (exit=${EXIT_CODE})"
    fi
  fi

  if compgen -G "${WORKSPACE_OUTPUT_DIR}/*" >/dev/null; then
    mkdir -p "${RUN_DIR}/workspace-output"
    cp -r "${WORKSPACE_OUTPUT_DIR}/." "${RUN_DIR}/workspace-output/" 2>/dev/null || true
  fi

  if [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_usage.py \
      --format pi \
      --trace "${TRACE_FILE}" \
      --out "${USAGE_FILE}" \
      --agent pi \
      --model-tag "${MODEL_TAG}" \
      --model "${AGSYS_PI_MODEL:-}" \
      >> "${STDERR_LOG}" 2>&1 || true
  fi

  if [[ ${EXIT_CODE} -ne 0 ]] && [[ ${EXIT_CODE} -ne 124 ]] && [[ ${EXIT_CODE} -ne 137 ]]; then
    echo "ERROR: pi exited with code ${EXIT_CODE}"
    if [[ -f "${STDERR_LOG}" ]]; then
      echo "--- stderr (last 5 lines) ---"
      tail -5 "${STDERR_LOG}"
      echo "---"
    fi
    echo "FAILED: ${QID} (exit=${EXIT_CODE})"
    FAILED=$((FAILED + 1))
    rm -rf "${QTMP_ROOT}"
    echo ""
    continue
  fi

  if [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_answer.py \
      --format pi \
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

  rm -rf "${QTMP_ROOT}"
  echo ""
done < <(agsys_list_qids "${SINGLE_QID}")

echo "=== pi summary: ${COMPLETED} completed, ${FAILED} failed ==="

if [[ -z "${SINGLE_QID}" ]]; then
  echo "=== Collecting: pi/${MODEL_TAG} ==="
  python3 agent_systems/collect_results.py --agent pi --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"
  python3 agent_systems/collect_usage.py   --agent pi --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"

  if [[ -n "${AGSYS_SKIP_EVAL:-}" ]]; then
    echo "SKIP: evaluation (AGSYS_SKIP_EVAL=1)"
  else
    PREDICTIONS="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/pi/${MODEL_TAG}/answers.jsonl"
    EVAL_DIR="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/pi/${MODEL_TAG}/eval"
    agsys_evaluate_predictions "${PREDICTIONS}" "${EVAL_DIR}"
    EVAL_EXIT=$?
    if [[ ${EVAL_EXIT} -ne 0 ]]; then
      echo "WARN: evaluate_qa.py failed (exit=${EVAL_EXIT}). Set AGSYS_SKIP_EVAL=1 to skip."
    fi
  fi
fi
