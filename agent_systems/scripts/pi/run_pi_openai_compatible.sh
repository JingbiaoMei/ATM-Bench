#!/usr/bin/env bash
#
# pi preset: generic OpenAI-compatible API server (provider-agnostic template).
#
# Point pi at ANY endpoint that speaks the OpenAI /v1/chat/completions API
# (vLLM, llama.cpp, LM Studio, OpenRouter, a cloud gateway, ...). Nothing
# provider-specific is baked in -- configure it entirely via environment vars:
#
#   PI_OPENAI_BASE_URL          Base URL of your endpoint (default http://localhost:8000/v1)
#   PI_OPENAI_MODEL             Model id served by your endpoint (default gpt-4o-mini)
#   PI_OPENAI_CONTEXT_WINDOW    (optional) context window in tokens (default 128000)
#   PI_OPENAI_MAX_TOKENS        (optional) max output tokens (default 8192)
#   OPENAI_COMPATIBLE_API_KEY   Bearer token. If unset, run_pi.sh loads it from
#                               api_keys/.openai_compatible_key
#
# Usage:
#   PI_OPENAI_BASE_URL=https://my-host/v1 PI_OPENAI_MODEL=my-model \
#     bash agent_systems/scripts/pi/run_pi_openai_compatible.sh
#   bash agent_systems/scripts/pi/run_pi_openai_compatible.sh <question_id>
#

set -o pipefail

: "${PI_OPENAI_BASE_URL:=http://localhost:8000/v1}"
: "${PI_OPENAI_MODEL:=gpt-4o-mini}"
export PI_OPENAI_BASE_URL PI_OPENAI_MODEL
[[ -n "${PI_OPENAI_CONTEXT_WINDOW:-}" ]] && export PI_OPENAI_CONTEXT_WINDOW
[[ -n "${PI_OPENAI_MAX_TOKENS:-}" ]] && export PI_OPENAI_MAX_TOKENS

# pi addresses the model as <provider>/<id>; the extension registers provider id
# "openai-compatible" with a single model whose id == ${PI_OPENAI_MODEL}.
export AGSYS_PI_PROVIDER="openai-compatible"
export AGSYS_PI_MODEL="openai-compatible/${PI_OPENAI_MODEL}"
export AGSYS_PI_MODEL_TAG="${AGSYS_PI_MODEL_TAG:-openai-compatible_${PI_OPENAI_MODEL//\//_}}"
export AGSYS_PI_THINKING="${AGSYS_PI_THINKING:-medium}"

# Generic, provider-agnostic extension that registers the endpoint above.
export AGSYS_PI_EXTENSION_PATHS="${AGSYS_PI_EXTENSION_PATHS:-agent_systems/scripts/pi/extensions/openai-compatible}"

# Forward the endpoint config + key into the (offline/bwrap) pi runtime so the
# extension can read them at provider-registration time.
export AGSYS_PI_FORWARD_ENV="${AGSYS_PI_FORWARD_ENV:+${AGSYS_PI_FORWARD_ENV} }PI_OPENAI_BASE_URL PI_OPENAI_MODEL PI_OPENAI_CONTEXT_WINDOW PI_OPENAI_MAX_TOKENS OPENAI_COMPATIBLE_API_KEY"

bash agent_systems/scripts/pi/run_pi.sh "$@"
