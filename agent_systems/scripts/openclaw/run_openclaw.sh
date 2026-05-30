#!/usr/bin/env bash
#
# Run OpenClaw locally on ATM-hard questions (one process per question).
#
# IMPORTANT: This is an internal worker script for the benchmark-user path.
# Do NOT run this directly as your normal user from the main repo. Use
# run_openclaw_bench_user.sh or smoke_test_openclaw.sh.
#
# Usage (from benchmark bundle root as openclaw-bench user):
#   bash agent_systems/scripts/openclaw/run_openclaw.sh              # all questions
#   bash agent_systems/scripts/openclaw/run_openclaw.sh <question_id>
#
# Env overrides (see agent_systems/config.py):
#   AGSYS_RUN_TAG, AGSYS_OPENCLAW_BIN, AGSYS_OPENCLAW_TIMEOUT_S, AGSYS_OPENCLAW_THINKING
#   AGSYS_OPENCLAW_MODEL_TAG="label-for-results"
#   AGSYS_OPENCLAW_AGENT="atmbench"  (passed as --agent to OpenClaw)
#   AGSYS_EVAL_EXTRA_FLAGS="--request-delay 1.0"
#   AGSYS_EVAL_GLM_FLAGS="--judge-endpoint http://... --max-workers 1"
#   AGSYS_EVAL_GPT_FLAGS="--judge-model gpt-5-mini --judge-reasoning-effort minimal"
#   AGSYS_SKIP_EVAL=1
#

set -o pipefail

# ── Enforce benchmark-user isolation ────────────────────────────────────────
if [[ "${AGSYS_OPENCLAW_BENCH_MODE:-}" != "1" ]]; then
  cat >&2 <<'MSG'
ERROR: run_openclaw.sh is an internal benchmark-worker script.

Use one of these entrypoints instead:
  bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh
  bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh
MSG
  exit 1
fi

if [[ -z "${OPENCLAW_HOME:-}" ]]; then
  cat >&2 <<'MSG'
ERROR: OPENCLAW_HOME is not set.

OpenClaw benchmark runs require explicit isolation to prevent ground-truth
leakage and config contamination. Do NOT run this script directly as your
normal user from the main repo checkout.

Instead, use the benchmark-user workflow:
  1. Set up the openclaw-bench user (see agent_systems/openclaw/isolation_setup.md)
  2. Copy the benchmark bundle:
       bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh /target/dir
  3. Run via the launcher:
       bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh

To override this check (NOT recommended for benchmark runs), set:
  OPENCLAW_HOME=/path/to/isolated/home
MSG
  exit 1
fi

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

agsys_require_eval_root

# ── Pre-flight: reject runs if bootstrap files exist in the workspace ───────
_preflight_check_bootstrap_files() {
  local eval_root="${AGSYS_EVAL_ROOT}"
  local found=0
  for f in SOUL.md AGENTS.md TOOLS.md BOOTSTRAP.md; do
    if [[ -f "${eval_root}/${f}" ]]; then
      echo "ERROR: Found ${eval_root}/${f} — remove before running benchmark." >&2
      found=1
    fi
  done
  if [[ ${found} -ne 0 ]]; then
    exit 1
  fi
}
_preflight_check_bootstrap_files

SINGLE_QID="${1:-}"
MODEL_TAG="$(agsys_sanitize_tag "${AGSYS_OPENCLAW_MODEL_TAG:-default}")"
OPENCLAW_AGENT="${AGSYS_OPENCLAW_AGENT:-}"
OPENCLAW_VERSION="$("${AGSYS_OPENCLAW_BIN}" --version 2>/dev/null | head -n1 | tr -d '\r' || true)"
OPENCLAW_PRIMARY_MODEL="$(python3 - "${AGSYS_OPENCLAW_CONFIG_SOURCE}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
try:
    data = json.load(path.open("r", encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
agents = data.get("agents", {}) if isinstance(data, dict) else {}
defaults = agents.get("defaults", {}) if isinstance(agents, dict) else {}
model = defaults.get("model", {}) if isinstance(defaults, dict) else {}
primary = model.get("primary", "") if isinstance(model, dict) else ""
print(primary if isinstance(primary, str) else "")
PY
)"

agsys_openclaw_seed_auth_state() {
  local target_state_dir="$1"
  local source_root=""
  local config_parent=""
  config_parent="$(dirname "${AGSYS_OPENCLAW_CONFIG_SOURCE}")"

  local -a candidates=()
  if [[ -n "${AGSYS_OPENCLAW_AUTH_SOURCE_ROOT:-}" ]]; then
    candidates+=("${AGSYS_OPENCLAW_AUTH_SOURCE_ROOT}")
  fi
  candidates+=("${config_parent}" "${HOME}/.openclaw" "${HOME}/.openclaw-state" "${OPENCLAW_HOME}")

  local root
  for root in "${candidates[@]}"; do
    [[ -n "${root}" ]] || continue
    if [[ -f "${root}/agents/main/agent/auth-profiles.json" ]]; then
      source_root="${root}"
      break
    fi
  done

  if [[ -z "${source_root}" ]]; then
    return 1
  fi

  mkdir -p "${target_state_dir}/agents/main"
  cp -r "${source_root}/agents/main/agent" "${target_state_dir}/agents/main/"

  if [[ -n "${OPENCLAW_AGENT}" ]] && [[ -d "${source_root}/agents/${OPENCLAW_AGENT}/agent" ]]; then
    mkdir -p "${target_state_dir}/agents/${OPENCLAW_AGENT}"
    cp -r "${source_root}/agents/${OPENCLAW_AGENT}/agent" "${target_state_dir}/agents/${OPENCLAW_AGENT}/"
  fi

  AGSYS_OPENCLAW_AUTH_SEED_SOURCE="${source_root}"
  return 0
}

COMPLETED=0
FAILED=0

while IFS= read -r QID; do
  RUN_DIR="$(agsys_setup_run_dir "openclaw" "${MODEL_TAG}" "${QID}")"
  TRACE_FILE="${RUN_DIR}/output/trace.json"
  ANSWER_FILE="${RUN_DIR}/output/answer.json"
  USAGE_FILE="${RUN_DIR}/output/usage.json"
  STDERR_LOG="${RUN_DIR}/output/stderr.log"

  if [[ -f "${ANSWER_FILE}" ]] && agsys_answer_valid "${ANSWER_FILE}" "${QID}"; then
    if [[ ! -f "${USAGE_FILE}" ]] && [[ -f "${TRACE_FILE}" ]]; then
      python3 agent_systems/extract_usage.py \
        --format openclaw \
        --trace "${TRACE_FILE}" \
        --out "${USAGE_FILE}" \
        --agent openclaw \
        --model-tag "${MODEL_TAG}" \
        >> "${STDERR_LOG}" 2>&1 || true
    fi
    echo "SKIP: ${QID} already answered (${ANSWER_FILE})"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  rm -f "${ANSWER_FILE}"
  rm -f "${TRACE_FILE}"
  rm -f "${USAGE_FILE}"
  rm -rf "${RUN_DIR}/output/openclaw-logs"
  rm -f "${RUN_DIR}/output/openclaw-runtime.json"   # legacy name
  rm -f "${RUN_DIR}/output/runtime_manifest.json"

  SYSTEM_PROMPT="$(cat "${AGSYS_SYSTEM_PROMPT}")"
  QUESTION_TEXT="$(cat "${RUN_DIR}/question.txt")"
  MESSAGE="$(printf '%s\n\nQuestion:\n%s' "${SYSTEM_PROMPT}" "${QUESTION_TEXT}")"

  # ── Per-question temp workspace (copied, not symlinked) ──────────────────
  OPENCLAW_TMPDIR="$(mktemp -d "/tmp/agsys-openclaw-XXXXXXXX")"
  WORKSPACE_DIR="${OPENCLAW_TMPDIR}/workspace"
  OPENCLAW_RUNTIME_HOME="${OPENCLAW_TMPDIR}/home"
  mkdir -p "${WORKSPACE_DIR}"
  mkdir -p "${OPENCLAW_RUNTIME_HOME}/.config" "${OPENCLAW_RUNTIME_HOME}/.cache" "${OPENCLAW_RUNTIME_HOME}/.local/share"
  # Copy question files and memory into an isolated workspace to prevent
  # traversal to sibling question dirs or eval_root via relative symlinks.
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.json" "${WORKSPACE_DIR}/question.json"
  cp "${AGSYS_EVAL_ROOT}/qas/${QID}/question.txt"  "${WORKSPACE_DIR}/question.txt"
  cp -r "${AGSYS_EVAL_ROOT}/memory" "${WORKSPACE_DIR}/memory"

  OPENCLAW_STATE_DIR="${OPENCLAW_TMPDIR}/state"
  OPENCLAW_CONFIG_PATH="${OPENCLAW_TMPDIR}/openclaw.json"
  OPENCLAW_SESSION_ID="${QID}-$(date +%s%N)"
  mkdir -p "${OPENCLAW_STATE_DIR}"
  AUTH_SEEDED=false
  AUTH_SEED_SOURCE=""
  if agsys_openclaw_seed_auth_state "${OPENCLAW_STATE_DIR}"; then
    AUTH_SEEDED=true
    AUTH_SEED_SOURCE="${AGSYS_OPENCLAW_AUTH_SEED_SOURCE}"
  fi
  if ! python3 agent_systems/runtime_artifacts.py write-openclaw-config \
    --source "${AGSYS_OPENCLAW_CONFIG_SOURCE}" \
    --workspace "${WORKSPACE_DIR}" \
    --out "${OPENCLAW_CONFIG_PATH}"; then
    echo "FAILED: ${QID} (could not build isolated OpenClaw config)"
    rm -rf "${OPENCLAW_TMPDIR}"
    FAILED=$((FAILED + 1))
    echo ""
    continue
  fi

  # ── Runtime manifest for post-hoc auditing ───────────────────────────────
  cat > "${RUN_DIR}/output/runtime_manifest.json" <<EOF
{
  "agent": "openclaw",
  "question_id": "${QID}",
  "run_dir": "${RUN_DIR}",
  "workspace_mode": "copied question files + copied memory into isolated temp workspace",
  "workspace_host_path": "${WORKSPACE_DIR}",
  "model_tag": "${MODEL_TAG}",
  "configured_primary_model": "${OPENCLAW_PRIMARY_MODEL}",
  "openclaw_version": "${OPENCLAW_VERSION}",
  "openclaw_agent": "${OPENCLAW_AGENT}",
  "openclaw_bin": "${AGSYS_OPENCLAW_BIN}",
  "openclaw_home": "${OPENCLAW_HOME}",
  "runtime_home": "${OPENCLAW_RUNTIME_HOME}",
  "config_source": "${AGSYS_OPENCLAW_CONFIG_SOURCE}",
  "auth_seeded": ${AUTH_SEEDED},
  "auth_seed_source": "${AUTH_SEED_SOURCE}",
  "thinking": "${AGSYS_OPENCLAW_THINKING}",
  "timeout_s": "${AGSYS_OPENCLAW_TIMEOUT_S}",
  "session_id": "${OPENCLAW_SESSION_ID}",
  "web_search_enabled": false,
  "web_fetch_enabled": false,
  "browser_enabled": false,
  "agent_to_agent_enabled": false,
  "plugins_enabled": false,
  "skills_enabled": false,
  "tmp_workspace_deleted_after_run": true
}
EOF

  echo "── [openclaw/${MODEL_TAG}] ${QID}"

  # ── Build CLI args ───────────────────────────────────────────────────────
  OPENCLAW_ARGS=(
    --no-color
    agent --local --json
    --session-id "${OPENCLAW_SESSION_ID}"
    --timeout "${AGSYS_OPENCLAW_TIMEOUT_S}"
    --thinking "${AGSYS_OPENCLAW_THINKING}"
    --verbose "${AGSYS_OPENCLAW_VERBOSE}"
  )
  if [[ -n "${OPENCLAW_AGENT}" ]]; then
    OPENCLAW_ARGS+=(--agent "${OPENCLAW_AGENT}")
  fi
  OPENCLAW_ARGS+=(--message "${MESSAGE}")

  # ── External timeout guard: TERM at timeout, KILL 30s later ───────────────

  HOME="${OPENCLAW_RUNTIME_HOME}" \
  XDG_CONFIG_HOME="${OPENCLAW_RUNTIME_HOME}/.config" \
  XDG_CACHE_HOME="${OPENCLAW_RUNTIME_HOME}/.cache" \
  XDG_DATA_HOME="${OPENCLAW_RUNTIME_HOME}/.local/share" \
  OPENCLAW_HOME="${OPENCLAW_HOME}" \
  OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR}" \
  OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH}" \
    timeout --kill-after=30 "${AGSYS_OPENCLAW_TIMEOUT_S}" \
    "${AGSYS_OPENCLAW_BIN}" "${OPENCLAW_ARGS[@]}" \
      < /dev/null > "${TRACE_FILE}" 2> "${STDERR_LOG}"
  EXIT_CODE=$?

  if [[ -d "${OPENCLAW_STATE_DIR}/logs" ]]; then
    mkdir -p "${RUN_DIR}/output/openclaw-logs"
    cp -r "${OPENCLAW_STATE_DIR}/logs/." "${RUN_DIR}/output/openclaw-logs/" 2>/dev/null || true
  fi
  rm -rf "${OPENCLAW_TMPDIR}"

  if [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_usage.py \
      --format openclaw \
      --trace "${TRACE_FILE}" \
      --out "${USAGE_FILE}" \
      --agent openclaw \
      --model-tag "${MODEL_TAG}" \
      >> "${STDERR_LOG}" 2>&1 || true
  fi

  if [[ ${EXIT_CODE} -eq 0 ]] && [[ -f "${TRACE_FILE}" ]]; then
    python3 agent_systems/extract_answer.py \
      --format openclaw \
      --trace "${TRACE_FILE}" \
      --out "${ANSWER_FILE}" \
      --expected-id "${QID}" \
      --question-file "${RUN_DIR}/question.json" \
      >> "${STDERR_LOG}" 2>&1
    EXTRACT_EXIT=$?
    if [[ ${EXTRACT_EXIT} -ne 0 ]]; then
      echo "ERROR: extract_answer.py failed (exit=${EXTRACT_EXIT})" >> "${STDERR_LOG}"
      EXIT_CODE=${EXTRACT_EXIT}
    fi
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

echo "=== openclaw summary: ${COMPLETED} completed, ${FAILED} failed ==="

if [[ -z "${SINGLE_QID}" ]]; then
  echo "=== Collecting: openclaw/${MODEL_TAG} ==="
  python3 agent_systems/collect_results.py --agent openclaw --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"
  python3 agent_systems/collect_usage.py --agent openclaw --run-tag "${AGSYS_RUN_TAG}" --model-tag "${MODEL_TAG}"

  if [[ -n "${AGSYS_SKIP_EVAL:-}" ]]; then
    echo "SKIP: evaluation (AGSYS_SKIP_EVAL=1)"
  else
    PREDICTIONS="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/openclaw/${MODEL_TAG}/answers.jsonl"
    EVAL_DIR="${AGSYS_RESULTS_ROOT}/${AGSYS_RUN_TAG}/openclaw/${MODEL_TAG}/eval"
    agsys_evaluate_predictions "${PREDICTIONS}" "${EVAL_DIR}"
    EVAL_EXIT=$?
    if [[ ${EVAL_EXIT} -ne 0 ]]; then
      echo "WARN: evaluate_qa.py failed (exit=${EVAL_EXIT}). Set AGSYS_SKIP_EVAL=1 to skip."
    fi
  fi
fi
