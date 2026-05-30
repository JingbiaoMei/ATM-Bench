#!/usr/bin/env bash
#
# Launch the isolated OpenClaw benchmark as the dedicated benchmark user.
#
# Assumes execution from the copied benchmark bundle root, e.g.:
#   /home/openclaw-bench/atmbench
#
# Usage:
#   bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh
#   bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh <question_id>
#

set -o pipefail

[[ -f "agent_systems/config.py" ]] || {
  echo "ERROR: run this script from the benchmark bundle root (the directory containing agent_systems/)." >&2
  exit 1
}

BENCH_HOME="${OPENCLAW_BENCH_HOME:-${HOME}}"
BENCH_ROOT="${OPENCLAW_BENCH_ROOT:-${PWD}}"
OPENCLAW_CONFIG_PRIMARY_DEFAULT="${BENCH_HOME}/.openclaw/openclaw.json"
OPENCLAW_CONFIG_FALLBACK_DEFAULT="${BENCH_HOME}/.config/openclaw/openclaw.json"
OPENCLAW_PROVIDER_ENV_DEFAULT="${BENCH_HOME}/.config/openclaw/provider.env"
OPENCLAW_BIN_DEFAULT="${BENCH_HOME}/.local/openclaw-cli/bin/openclaw"

if [[ -f "${OPENCLAW_CONFIG_PRIMARY_DEFAULT}" ]]; then
  OPENCLAW_CONFIG_DEFAULT="${OPENCLAW_CONFIG_PRIMARY_DEFAULT}"
else
  OPENCLAW_CONFIG_DEFAULT="${OPENCLAW_CONFIG_FALLBACK_DEFAULT}"
fi

agsys_openclaw_primary_model() {
  python3 - "$1" <<'PY'
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
}

export OPENCLAW_HOME="${OPENCLAW_HOME:-${BENCH_HOME}/.openclaw-atmbench}"
export AGSYS_OPENCLAW_BIN="${AGSYS_OPENCLAW_BIN:-${OPENCLAW_BIN_DEFAULT}}"
export AGSYS_OPENCLAW_CONFIG_SOURCE="${AGSYS_OPENCLAW_CONFIG_SOURCE:-${OPENCLAW_CONFIG_DEFAULT}}"
export AGSYS_SKIP_EVAL="${AGSYS_SKIP_EVAL:-1}"
export AGSYS_OPENCLAW_BENCH_MODE=1

PROVIDER_ENV="${OPENCLAW_PROVIDER_ENV:-${OPENCLAW_PROVIDER_ENV_DEFAULT}}"

[[ -x "${AGSYS_OPENCLAW_BIN}" ]] || {
  echo "ERROR: OpenClaw binary not found or not executable: ${AGSYS_OPENCLAW_BIN}" >&2
  exit 1
}
[[ -f "${AGSYS_OPENCLAW_CONFIG_SOURCE}" ]] || {
  echo "ERROR: OpenClaw config source not found: ${AGSYS_OPENCLAW_CONFIG_SOURCE}" >&2
  exit 1
}
if [[ -z "${AGSYS_EVAL_ROOT:-}" ]]; then
  AGSYS_EVAL_ROOT="$(python3 agent_systems/config.py --print-json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("AGSYS_EVAL_ROOT",""))' 2>/dev/null \
    || true)"
  if [[ -z "${AGSYS_EVAL_ROOT}" ]]; then
    AGSYS_EVAL_ROOT="agent_systems/eval_root_sgm"
  fi
  export AGSYS_EVAL_ROOT
fi
[[ -d "${AGSYS_EVAL_ROOT}" ]] || {
  echo "ERROR: Missing ${AGSYS_EVAL_ROOT}. Copy the prepared benchmark bundle first." >&2
  exit 1
}

if [[ -z "${AGSYS_OPENCLAW_MODEL_TAG:-}" ]]; then
  OPENCLAW_PRIMARY_MODEL="$(agsys_openclaw_primary_model "${AGSYS_OPENCLAW_CONFIG_SOURCE}")"
  if [[ -n "${OPENCLAW_PRIMARY_MODEL}" ]]; then
    export AGSYS_OPENCLAW_MODEL_TAG="${OPENCLAW_PRIMARY_MODEL}"
  fi
fi

if [[ -f "${PROVIDER_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PROVIDER_ENV}"
  PROVIDER_ENV_STATUS="loaded"
else
  PROVIDER_ENV_STATUS="missing"
fi

echo "OpenClaw benchmark launcher"
echo "  bundle_root: ${BENCH_ROOT}"
echo "  openclaw_bin: ${AGSYS_OPENCLAW_BIN}"
echo "  openclaw_home: ${OPENCLAW_HOME}"
echo "  config_source: ${AGSYS_OPENCLAW_CONFIG_SOURCE}"
if [[ -n "${AGSYS_OPENCLAW_MODEL_TAG:-}" ]]; then
  echo "  model_tag: ${AGSYS_OPENCLAW_MODEL_TAG}"
fi
if [[ "${PROVIDER_ENV_STATUS}" == "loaded" ]]; then
  echo "  provider_env: ${PROVIDER_ENV}"
else
  echo "  provider_env: not found; assuming OpenClaw-managed auth/config"
fi
echo "  skip_eval: ${AGSYS_SKIP_EVAL}"
if [[ -n "${OPENCLAW_PROFILE:-}" ]]; then
  echo "  profile: ${OPENCLAW_PROFILE}"
fi
echo ""

bash agent_systems/scripts/openclaw/run_openclaw.sh "$@"
