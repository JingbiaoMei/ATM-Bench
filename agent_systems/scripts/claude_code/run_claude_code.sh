#!/usr/bin/env bash
#
# Run Claude Code on ATM-hard questions (one process per question).
#
# Usage:
#   bash agent_systems/scripts/claude_code/run_claude_code.sh              # all questions
#   bash agent_systems/scripts/claude_code/run_claude_code.sh <question_id>
#
# Env overrides (see agent_systems/config.py):
#   AGSYS_RUN_TAG, AGSYS_CLAUDE_BIN, AGSYS_CLAUDE_MODEL, AGSYS_CLAUDE_MODEL_TAG, AGSYS_CLAUDE_BUDGET_USD
#   AGSYS_MEMORY_MODE=sgm|raw|descriptive
#   AGSYS_CLAUDE_EFFORT=medium|low|high|max
#   AGSYS_CLAUDE_SANDBOX=bwrap|auto|off  (default: bwrap)
#   AGSYS_CLAUDE_TIMEOUT_S=600  (0 to disable watchdog timeout)
#   AGSYS_CLAUDE_MAX_ASSISTANT_EVENTS=50  (0 to disable the turn watchdog)
#   AGSYS_CLAUDE_EXTRA_FLAGS="--verbose"
#   AGSYS_EVAL_EXTRA_FLAGS="--request-delay 1.0"
#   AGSYS_EVAL_GLM_FLAGS="--judge-endpoint http://... --max-workers 1"
#   AGSYS_EVAL_GPT_FLAGS="--judge-model gpt-5-mini --judge-reasoning-effort minimal"
#   AGSYS_SKIP_EVAL=1
#

set -o pipefail

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

agsys_require_eval_root

SINGLE_QID="${1:-}"
MEMORY_MODE="$(agsys_normalize_memory_mode "${AGSYS_MEMORY_MODE:-sgm}")"

agsys_claude_model_tag() {
  local explicit_tag="${AGSYS_CLAUDE_MODEL_TAG:-}"
  local model="${AGSYS_CLAUDE_MODEL:-default}"
  local effort="${AGSYS_CLAUDE_EFFORT:-}"
  local tag="${model}"

  if [[ -n "${explicit_tag}" ]]; then
    tag="${explicit_tag}"
  fi

  # Keep existing medium/default run dirs stable, but suffix alternate effort
  # levels so reruns do not collide with prior results.
  if [[ -z "${explicit_tag}" ]] && [[ -n "${effort}" ]] && [[ "${effort}" != "medium" ]]; then
    case "${tag}" in
      *-"${effort}")
        ;;
      *)
        tag="${tag}-${effort}"
        ;;
    esac
  fi
  agsys_tag_with_memory_mode "${tag}" "${MEMORY_MODE}"
}

MODEL_TAG="$(agsys_claude_model_tag)"

read -r -a EXTRA_FLAGS <<< "${AGSYS_CLAUDE_EXTRA_FLAGS:-}"

for FLAG in "${EXTRA_FLAGS[@]}"; do
  case "${FLAG}" in
    --allow-dangerously-skip-permissions|--dangerously-skip-permissions|\
    --mcp-config|--mcp-config=*|--setting-sources|--setting-sources=*|\
    --settings|--settings=*|--plugin-dir|--plugin-dir=*|--tools|--tools=*|\
    --allowedTools|--allowedTools=*|--allowed-tools|--allowed-tools=*|\
    --disallowedTools|--disallowedTools=*|--disallowed-tools|--disallowed-tools=*|\
    --chrome|--agent|--agent=*|--agents|--agents=*|--permission-mode|--permission-mode=*|\
    --resume|--resume=*|-r|--continue|-c)
      agsys_die "AGSYS_CLAUDE_EXTRA_FLAGS cannot override isolation control: ${FLAG}"
      ;;
  esac
done

# ── Resolve Claude binary to an absolute path (needed for bwrap --ro-bind) ──

CLAUDE_BIN="${AGSYS_CLAUDE_BIN}"
if [[ "${CLAUDE_BIN}" != /* ]]; then
  CLAUDE_BIN="$(command -v "${CLAUDE_BIN}" 2>/dev/null || true)"
fi
[[ -n "${CLAUDE_BIN}" ]] || agsys_die "Claude binary not found: ${AGSYS_CLAUDE_BIN}"
[[ -x "${CLAUDE_BIN}" ]] || agsys_die "Claude binary is not executable: ${CLAUDE_BIN}"
CLAUDE_HOST_CREDENTIALS_FILE="${HOME}/.claude/.credentials.json"
CLAUDE_HOST_STATE_FILE="${HOME}/.claude.json"

# Follow symlinks to get the real binary path (needed for bwrap bind).
CLAUDE_BIN_REAL="$(readlink -f "${CLAUDE_BIN}")"
# Determine the install dir (e.g. $HOME/.claude/local/ or /usr/lib/node_modules/...)
# so we can bind the whole tree that claude may need at runtime.
CLAUDE_INSTALL_DIR="$(dirname "$(dirname "${CLAUDE_BIN_REAL}")")"

# ── Sandbox setup (bwrap) ──

SANDBOX_REQUESTED="${AGSYS_CLAUDE_SANDBOX:-bwrap}"
SANDBOX_ACTIVE="off"
SANDBOX_STATUS="disabled"
BWRAP_BIN=""

if [[ "${SANDBOX_REQUESTED}" != "off" ]]; then
  BWRAP_BIN="$(command -v bwrap 2>/dev/null || true)"
  if [[ -z "${BWRAP_BIN}" ]]; then
    SANDBOX_STATUS="bwrap-not-found"
    if [[ "${SANDBOX_REQUESTED}" == "bwrap" ]]; then
      agsys_die "AGSYS_CLAUDE_SANDBOX=bwrap but bwrap is not installed"
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
      agsys_die "AGSYS_CLAUDE_SANDBOX=bwrap but bwrap could not start on this machine"
    fi
  fi
fi

echo "INFO: claude_code memory_mode=${MEMORY_MODE} sandbox requested=${SANDBOX_REQUESTED} active=${SANDBOX_ACTIVE} status=${SANDBOX_STATUS}"

# ── Pre-cache memory files for copy into per-question workspaces ──

SESSION_TMP_ROOT="$(mktemp -d "/tmp/agsys-claude-session-XXXXXXXX")"
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

claude_has_auth_env() {
  local env_pair
  for env_pair in "$@"; do
    case "${env_pair}" in
      ANTHROPIC_API_KEY=*|ANTHROPIC_AUTH_TOKEN=*|CLAUDE_API_KEY=*|CLAUDE_CODE_OAUTH_TOKEN=*)
        return 0
        ;;
    esac
  done
  return 1
}

run_claude_process() {
  if [[ "${SANDBOX_ACTIVE}" == "bwrap" ]]; then
    "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
  else
    (
      cd "${WORKSPACE_DIR}" && \
      "${CMD[@]}" < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
    )
  fi
}

run_claude_monitored() {
  local timeout_s="$1"
  local max_assistant_events="$2"
  local elapsed=0
  local loop_reason=""
  local auth_fail_count=0
  local assistant_event_count=0

  run_claude_process &
  local claude_pid=$!

  while kill -0 "${claude_pid}" 2>/dev/null; do
    if [[ -f "${TRACE_FILE}" ]]; then
      auth_fail_count="$(rg -c '"error"[[:space:]]*:[[:space:]]*"authentication_failed"' "${TRACE_FILE}" 2>/dev/null || echo 0)"
      if [[ "${auth_fail_count}" =~ ^[0-9]+$ ]] && (( auth_fail_count >= 3 )); then
        loop_reason="authentication_failed"
        echo "ERROR: Claude authentication failed repeatedly; aborting run early." >> "${STDERR_LOG}"
        kill "${claude_pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${claude_pid}" 2>/dev/null || true
        break
      fi
      assistant_event_count="$(rg -c '^\{"type":"assistant"' "${TRACE_FILE}" 2>/dev/null || echo 0)"
      if [[ "${max_assistant_events}" != "0" ]] && [[ "${assistant_event_count}" =~ ^[0-9]+$ ]] && (( assistant_event_count >= max_assistant_events )); then
        loop_reason="assistant_event_cap"
        echo "ERROR: Claude exceeded assistant event cap (${max_assistant_events}); aborting run." >> "${STDERR_LOG}"
        kill "${claude_pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${claude_pid}" 2>/dev/null || true
        break
      fi
    fi
    if [[ "${timeout_s}" != "0" ]] && (( elapsed >= timeout_s )); then
      loop_reason="timeout"
      echo "ERROR: Claude exceeded ${timeout_s}s timeout; aborting run." >> "${STDERR_LOG}"
      kill "${claude_pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${claude_pid}" 2>/dev/null || true
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  wait "${claude_pid}"
  local wait_code=$?
  case "${loop_reason}" in
    authentication_failed)
      return 190
      ;;
    assistant_event_cap)
      return 125
      ;;
    timeout)
      return 124
      ;;
    *)
      return "${wait_code}"
      ;;
  esac
}

COMPLETED=0
FAILED=0

while IFS= read -r QID; do
  RUN_DIR="$(agsys_setup_run_dir "claude_code" "${MODEL_TAG}" "${QID}")"
  RUN_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${RUN_DIR}")"
  TRACE_FILE="${RUN_DIR}/output/trace.jsonl"
  ANSWER_FILE="${RUN_DIR}/output/answer.json"
  USAGE_FILE="${RUN_DIR}/output/usage.json"
  STDERR_LOG="${RUN_DIR}/output/stderr.log"

  if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    if [[ ! -f "${USAGE_FILE}" ]] && [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_usage.py \
        --format claude_code \
        --trace "${TRACE_FILE}" \
        --out "${USAGE_FILE}" \
        --agent claude_code \
        --model-tag "${MODEL_TAG}" \
        --model "${AGSYS_CLAUDE_MODEL:-}" \
        >> "${STDERR_LOG}" 2>&1 || true
    fi
    echo "SKIP: ${QID} already answered (${ANSWER_FILE})"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  rm -f "${ANSWER_FILE}"
  rm -f "${TRACE_FILE}"
  rm -f "${USAGE_FILE}"
  rm -rf "${RUN_DIR}/output/claude-debug"
  rm -rf "${RUN_DIR}/output/claude-transcripts"
  rm -f "${RUN_DIR}/output/claude-settings.json"
  rm -f "${RUN_DIR}/output/claude-mcp-config.json"
  rm -f "${RUN_DIR}/output/runtime_manifest.json"

  # ── Build isolated per-question workspace ──

  QTMP_ROOT="$(mktemp -d "${SESSION_TMP_ROOT}/${QID}.XXXXXXXX")"
  WORKSPACE_DIR="${QTMP_ROOT}/workspace"
  CLAUDE_HOME="${QTMP_ROOT}/home"
  CLAUDE_SETTINGS_FILE="${QTMP_ROOT}/settings.json"
  CLAUDE_MCP_CONFIG_FILE="${QTMP_ROOT}/mcp-config.json"
  CLAUDE_AUTH_SOURCE="none"
  mkdir -p \
    "${WORKSPACE_DIR}/memory" \
    "${WORKSPACE_DIR}/output" \
    "${CLAUDE_HOME}/.claude"

  # Copy question + memory into the isolated workspace.
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

  if [[ -f "${CLAUDE_HOST_CREDENTIALS_FILE}" ]]; then
    cp "${CLAUDE_HOST_CREDENTIALS_FILE}" "${CLAUDE_HOME}/.claude/.credentials.json"
    CLAUDE_AUTH_SOURCE="copied_oauth_credentials"
  fi
  if [[ -f "${CLAUDE_HOST_STATE_FILE}" ]]; then
    cp "${CLAUDE_HOST_STATE_FILE}" "${CLAUDE_HOME}/.claude.json"
    if [[ "${CLAUDE_AUTH_SOURCE}" == "none" ]]; then
      CLAUDE_AUTH_SOURCE="copied_claude_state"
    fi
  fi

  if ! python3 agent_systems/runtime_artifacts.py write-claude-settings --out "${CLAUDE_SETTINGS_FILE}"; then
    echo "FAILED: ${QID} (could not build isolated Claude settings)"
    rm -rf "${QTMP_ROOT}"
    FAILED=$((FAILED + 1))
    echo ""
    continue
  fi
  printf '{"mcpServers":{}}\n' > "${CLAUDE_MCP_CONFIG_FILE}"
  cp "${CLAUDE_SETTINGS_FILE}" "${RUN_DIR}/output/claude-settings.json"
  cp "${CLAUDE_MCP_CONFIG_FILE}" "${RUN_DIR}/output/claude-mcp-config.json"
  if ! mapfile -t CLAUDE_ENV_ARGS < <(
    python3 agent_systems/runtime_artifacts.py print-claude-env --source "${HOME}/.claude/settings.json"
  ); then
    echo "FAILED: ${QID} (could not load Claude auth env)"
    rm -rf "${QTMP_ROOT}"
    FAILED=$((FAILED + 1))
    echo ""
    continue
  fi
  if claude_has_auth_env "${CLAUDE_ENV_ARGS[@]}"; then
    CLAUDE_AUTH_SOURCE="env"
  fi
  if [[ ! -f "${CLAUDE_HOME}/.claude/.credentials.json" ]] && ! claude_has_auth_env "${CLAUDE_ENV_ARGS[@]}"; then
    echo "FAILED: ${QID} (no Claude auth available in isolated runtime; missing ~/.claude/.credentials.json and auth env vars)"
    rm -rf "${QTMP_ROOT}"
    FAILED=$((FAILED + 1))
    echo ""
    continue
  fi

  SYSTEM_PROMPT="$(cat "${AGSYS_SYSTEM_PROMPT}")"
  CLAUDE_APPEND_PROMPT="Claude Code runner note: do not use Write, Edit, or NotebookEdit to create the final answer artifact. Complete the task by calling StructuredOutput exactly once with the final {id, question, answer} payload."
  SCHEMA_JSON="$(cat "${AGSYS_QA_SCHEMA}")"
  QUESTION_TEXT="$(cat "${WORKSPACE_DIR}/question.txt")"

  echo "── [claude_code/${MODEL_TAG}] ${QID}"

  # ── Write runtime manifest for post-hoc auditing ──

  python3 - <<'PY' \
    "${RUN_DIR}/output/runtime_manifest.json" \
    "${QID}" \
    "${RUN_DIR}" \
    "${WORKSPACE_DIR}" \
    "${AGSYS_CLAUDE_MODEL:-}" \
    "${MODEL_TAG}" \
    "${AGSYS_CLAUDE_EFFORT:-}" \
    "${CLAUDE_BIN}" \
    "${SANDBOX_REQUESTED}" \
    "${SANDBOX_ACTIVE}" \
    "${SANDBOX_STATUS}" \
    "${AGSYS_CLAUDE_TIMEOUT_S:-600}" \
    "${AGSYS_CLAUDE_MAX_ASSISTANT_EVENTS:-50}" \
    "${CLAUDE_AUTH_SOURCE}" \
    "${MEMORY_MODE}"
import json, sys
out = sys.argv[1]
payload = {
    "agent": "claude_code",
    "question_id": sys.argv[2],
    "run_dir": sys.argv[3],
    "workspace_mode": "copied question files + copied memory into isolated temp workspace",
    "workspace_host_path": sys.argv[4],
    "model": sys.argv[5],
    "model_tag": sys.argv[6],
    "memory_mode": sys.argv[15],
    "effort": sys.argv[7],
    "claude_bin": sys.argv[8],
    "sandbox_requested": sys.argv[9],
    "sandbox_active": sys.argv[10],
    "sandbox_status": sys.argv[11],
    "timeout_s": sys.argv[12],
    "max_assistant_events": sys.argv[13],
    "auth_source": sys.argv[14],
    "tmp_workspace_deleted_after_run": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

  # ── Build the command ──

  # Claude CLI args (these are the same regardless of sandbox mode).
  # NOTE: --mcp-config takes variadic args (<configs...>), so it must appear
  # before the positional prompt argument.  We insert --settings, --strict-mcp-config,
  # and --mcp-config into this array early (sandbox-specific paths are appended later
  # via CLAUDE_CONFIG_ARGS).
  CLAUDE_ARGS=(
    -p
    --no-session-persistence
    --max-budget-usd "${AGSYS_CLAUDE_BUDGET_USD}"
    --output-format stream-json
    --verbose
    --system-prompt "${SYSTEM_PROMPT}"
    --append-system-prompt "${CLAUDE_APPEND_PROMPT}"
    --json-schema "${SCHEMA_JSON}"
    --setting-sources local
    --disable-slash-commands
    --no-chrome
    --tools "${AGSYS_CLAUDE_TOOLS}"
  )
  if [[ -n "${AGSYS_CLAUDE_MODEL}" ]]; then
    CLAUDE_ARGS+=(--model "${AGSYS_CLAUDE_MODEL}")
  fi
  if [[ -n "${AGSYS_CLAUDE_EFFORT:-}" ]]; then
    CLAUDE_ARGS+=(--effort "${AGSYS_CLAUDE_EFFORT}")
  fi
  if [[ -n "${AGSYS_CLAUDE_ALLOWED_TOOLS:-}" ]]; then
    CLAUDE_ARGS+=(--allowed-tools "${AGSYS_CLAUDE_ALLOWED_TOOLS}")
  fi
  if [[ -n "${AGSYS_CLAUDE_DISALLOWED_TOOLS:-}" ]]; then
    CLAUDE_ARGS+=(--disallowed-tools "${AGSYS_CLAUDE_DISALLOWED_TOOLS}")
  fi
  if [[ -n "${AGSYS_CLAUDE_DEBUG:-}" ]] && [[ "${AGSYS_CLAUDE_DEBUG}" != "0" ]]; then
    CLAUDE_ARGS+=(--debug)
  fi
  if [[ "${#EXTRA_FLAGS[@]}" -gt 0 ]]; then
    CLAUDE_ARGS+=("${EXTRA_FLAGS[@]}")
  fi

  CMD=()
  if [[ "${SANDBOX_ACTIVE}" == "bwrap" ]]; then
    # Copy settings + mcp config into workspace so they are visible inside bwrap.
    cp "${CLAUDE_SETTINGS_FILE}" "${WORKSPACE_DIR}/.claude-settings.json"
    cp "${CLAUDE_MCP_CONFIG_FILE}" "${WORKSPACE_DIR}/.claude-mcp-config.json"

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
    # Bind the Claude install tree read-only so the binary and its node_modules are accessible.
    CMD+=(--ro-bind "${CLAUDE_INSTALL_DIR}" "${CLAUDE_INSTALL_DIR}")
    # Bind the workspace read-write and the home dir read-write.
    CMD+=(
      --bind "${CLAUDE_HOME}" /home/claude
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
      HOME=/home/claude
    )
    for ENV_PAIR in "${CLAUDE_ENV_ARGS[@]}"; do
      CMD+=("${ENV_PAIR}")
    done
    # --mcp-config is variadic (<configs...>), so it MUST come before the
    # positional prompt; use -- to terminate options explicitly.
    CMD+=(
      "${CLAUDE_BIN_REAL}"
      "${CLAUDE_ARGS[@]}"
      --settings /workspace/.claude-settings.json
      --strict-mcp-config
      --mcp-config /workspace/.claude-mcp-config.json
      -- "${QUESTION_TEXT}"
    )
  else
    # Non-sandboxed: run from workspace dir with isolated HOME (original behavior).
    CMD=(
      env
      HOME="${CLAUDE_HOME}"
      "${CLAUDE_ENV_ARGS[@]}"
      "${CLAUDE_BIN}"
      "${CLAUDE_ARGS[@]}"
      --settings "${CLAUDE_SETTINGS_FILE}"
      --strict-mcp-config
      --mcp-config "${CLAUDE_MCP_CONFIG_FILE}"
      -- "${QUESTION_TEXT}"
    )
  fi

  run_claude_monitored "${AGSYS_CLAUDE_TIMEOUT_S:-600}" "${AGSYS_CLAUDE_MAX_ASSISTANT_EVENTS:-50}"
  EXIT_CODE=$?

  if compgen -G "${WORKSPACE_DIR}/output/*" >/dev/null; then
    cp -r "${WORKSPACE_DIR}/output/." "${RUN_DIR}/output/" 2>/dev/null || true
  fi

  if [[ -d "${CLAUDE_HOME}/.claude/debug" ]]; then
    mkdir -p "${RUN_DIR}/output/claude-debug"
    cp -r "${CLAUDE_HOME}/.claude/debug/." "${RUN_DIR}/output/claude-debug/" 2>/dev/null || true
  fi
  if [[ -d "${CLAUDE_HOME}/.claude/transcripts" ]]; then
    mkdir -p "${RUN_DIR}/output/claude-transcripts"
    cp -r "${CLAUDE_HOME}/.claude/transcripts/." "${RUN_DIR}/output/claude-transcripts/" 2>/dev/null || true
  fi
  rm -rf "${QTMP_ROOT}"

  if [[ -f "${TRACE_FILE}" ]] && [[ ${EXIT_CODE} -ne 190 ]]; then
    python3 agent_systems/extract_answer.py \
      --format claude_code \
      --trace "${TRACE_FILE}" \
      --out "${ANSWER_FILE}" \
      --expected-id "${QID}" \
      --question-file "${RUN_DIR}/question.json" \
      >> "${STDERR_LOG}" 2>&1 || true

    python3 agent_systems/extract_usage.py \
      --format claude_code \
      --trace "${TRACE_FILE}" \
      --out "${USAGE_FILE}" \
      --agent claude_code \
      --model-tag "${MODEL_TAG}" \
      --model "${AGSYS_CLAUDE_MODEL:-}" \
      >> "${STDERR_LOG}" 2>&1 || true
  elif [[ ${EXIT_CODE} -eq 190 ]]; then
    echo "SKIP: extract_answer/extract_usage after Claude auth watchdog abort." >> "${STDERR_LOG}"
  fi

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

echo "=== claude_code summary: ${COMPLETED} completed, ${FAILED} failed ==="

if [[ -z "${SINGLE_QID}" ]]; then
  echo "=== Collecting: claude_code/${MODEL_TAG} ==="
  python3 agent_systems/collect_results.py --agent claude_code --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"
  python3 agent_systems/collect_usage.py --agent claude_code --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"

  if [[ -n "${AGSYS_SKIP_EVAL:-}" ]]; then
    echo "SKIP: evaluation (AGSYS_SKIP_EVAL=1)"
  else
    PREDICTIONS="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/claude_code/${MODEL_TAG}/answers.jsonl"
    EVAL_DIR="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/claude_code/${MODEL_TAG}/eval"
    agsys_evaluate_predictions "${PREDICTIONS}" "${EVAL_DIR}"
    EVAL_EXIT=$?
    if [[ ${EVAL_EXIT} -ne 0 ]]; then
      echo "WARN: evaluate_qa.py failed (exit=${EVAL_EXIT}). Set AGSYS_SKIP_EVAL=1 to skip."
    fi
  fi
fi
