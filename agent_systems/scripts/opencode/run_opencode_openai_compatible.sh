#!/usr/bin/env bash
#
# opencode preset: generic OpenAI-compatible API server (provider-agnostic).
#
# Point opencode at ANY endpoint that speaks the OpenAI /v1/chat/completions API
# through the @ai-sdk/openai-compatible provider. Nothing provider-specific is
# baked in -- configure it entirely via environment variables:
#
#   OPENCODE_BASE_URL           Base URL of your endpoint (default http://localhost:8000/v1)
#   OPENCODE_MODEL_ID           Model id served by your endpoint (default gpt-4o-mini)
#   OPENAI_COMPATIBLE_API_KEY   Bearer token. If unset, loaded from
#                               api_keys/.openai_compatible_key
#
# The API key is never written to disk: the generated opencode.json keeps it as
# a `{env:OPENAI_COMPATIBLE_API_KEY}` placeholder, and run_opencode.sh forwards
# the env var into the isolated runtime where opencode resolves it.
#
# A static reference config lives at configs/openai-compatible.json.
#
# Usage:
#   OPENCODE_BASE_URL=https://my-host/v1 OPENCODE_MODEL_ID=my-model \
#     bash agent_systems/scripts/opencode/run_opencode_openai_compatible.sh
#   bash agent_systems/scripts/opencode/run_opencode_openai_compatible.sh <question_id>
#

set -o pipefail

: "${OPENCODE_BASE_URL:=http://localhost:8000/v1}"
: "${OPENCODE_MODEL_ID:=gpt-4o-mini}"

# Load the API key from api_keys/.openai_compatible_key if not already in env.
if [[ -z "${OPENAI_COMPATIBLE_API_KEY:-}" ]]; then
  _KEY_FILE="${OPENCODE_API_KEYS_DIR:-api_keys}/.openai_compatible_key"
  if [[ -f "${_KEY_FILE}" ]]; then
    OPENAI_COMPATIBLE_API_KEY="$(tr -d '\r\n' < "${_KEY_FILE}" 2>/dev/null || true)"
  fi
fi
export OPENAI_COMPATIBLE_API_KEY

# Render an isolated opencode config from the env values. The api key stays a
# {env:...} placeholder (no secret on disk); run_opencode.sh forwards
# OPENAI_COMPATIBLE_API_KEY into the sandbox where opencode resolves it.
RENDERED_CONFIG="$(mktemp "/tmp/agsys-opencode-oai-XXXXXXXX.json")"
cleanup_oai_cfg() { rm -f "${RENDERED_CONFIG}"; }
trap cleanup_oai_cfg EXIT
cat > "${RENDERED_CONFIG}" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai-compatible": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenAI-compatible API server",
      "options": {
        "baseURL": "${OPENCODE_BASE_URL}",
        "apiKey": "{env:OPENAI_COMPATIBLE_API_KEY}"
      },
      "models": {
        "${OPENCODE_MODEL_ID}": {
          "name": "${OPENCODE_MODEL_ID}"
        }
      }
    }
  }
}
EOF

export AGSYS_OPENCODE_CONFIG_SOURCE="${RENDERED_CONFIG}"
export AGSYS_OPENCODE_MODEL="openai-compatible/${OPENCODE_MODEL_ID}"

bash agent_systems/scripts/opencode/run_opencode.sh "$@"
