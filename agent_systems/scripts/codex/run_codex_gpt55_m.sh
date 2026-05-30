#!/usr/bin/env bash
#
# Codex preset: gpt-5.5 with medium reasoning effort.
#
# Usage:
#   bash agent_systems/scripts/codex/run_codex_gpt55_m.sh
#   bash agent_systems/scripts/codex/run_codex_gpt55_m.sh <question_id>
#

set -o pipefail

export AGSYS_CODEX_MODEL="gpt-5.5"
export AGSYS_CODEX_MODEL_TAG="gpt-5.5-medium"
export AGSYS_CODEX_REASONING_EFFORT="medium"

bash agent_systems/scripts/codex/run_codex.sh "$@"
