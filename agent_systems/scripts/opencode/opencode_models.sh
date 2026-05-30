#!/usr/bin/env bash
#
# Convenience helper to list opencode model IDs using a writable XDG state.
#
# Usage:
#   bash agent_systems/scripts/opencode/opencode_models.sh            # all providers
#   bash agent_systems/scripts/opencode/opencode_models.sh openai-compatible  # filter provider
#   bash agent_systems/scripts/opencode/opencode_models.sh <provider>
#

set -o pipefail

# shellcheck disable=SC1091
source "agent_systems/scripts/common.sh"

PROVIDER="${1:-}"

TMP_ROOT="${TMP_ROOT:-/tmp/agsys_opencode_models}"
mkdir -p "${TMP_ROOT}/data" "${TMP_ROOT}/state" "${TMP_ROOT}/cache"

OPENCODE_DISABLE_MODELS_FETCH=1 \
XDG_DATA_HOME="${TMP_ROOT}/data" \
XDG_STATE_HOME="${TMP_ROOT}/state" \
XDG_CACHE_HOME="${TMP_ROOT}/cache" \
  "${AGSYS_OPENCODE_BIN}" models ${PROVIDER:+${PROVIDER}} --verbose
