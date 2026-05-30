#!/usr/bin/env bash
#
# Claude Code preset: Opus 4.7 with xhigh effort.
#
# Usage:
#   bash agent_systems/scripts/claude_code/run_claude_code_opus47-xhigh.sh
#   bash agent_systems/scripts/claude_code/run_claude_code_opus47-xhigh.sh <question_id>
#

set -o pipefail

export AGSYS_CLAUDE_MODEL="claude-opus-4-7"
export AGSYS_CLAUDE_MODEL_TAG="claude-opus-4-7-xhigh"
export AGSYS_CLAUDE_EFFORT="xhigh"

bash agent_systems/scripts/claude_code/run_claude_code.sh "$@"
