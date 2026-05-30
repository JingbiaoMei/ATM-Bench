#!/usr/bin/env bash
#
# Package a minimal benchmark bundle for the openclaw-bench user.
#
# This copies only the files needed to run the OpenClaw benchmark, excluding
# ground-truth data, the rest of the repo, and your personal config.
#
# Usage (from repo root):
#   sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh /home/openclaw-bench/atmbench
#   bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh ./bundle  # local test
#
# Prerequisites:
#   python3 agent_systems/prepare_sandbox.py --force
#

set -o pipefail

TARGET_DIR="${1:-}"

if [[ -z "${TARGET_DIR}" ]]; then
  echo "Usage: bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh <target_dir>" >&2
  exit 1
fi

if [[ ! -f "agent_systems/config.py" ]]; then
  echo "ERROR: Run this script from the repo root." >&2
  exit 1
fi

if [[ -z "${AGSYS_EVAL_ROOT:-}" ]]; then
  AGSYS_EVAL_ROOT="$(python3 agent_systems/config.py --print-json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("AGSYS_EVAL_ROOT",""))' 2>/dev/null \
    || true)"
  if [[ -z "${AGSYS_EVAL_ROOT}" ]]; then
    AGSYS_EVAL_ROOT="agent_systems/eval_root_sgm"
  fi
fi
EVAL_ROOT_NAME="$(basename "${AGSYS_EVAL_ROOT}")"

if [[ ! -d "${AGSYS_EVAL_ROOT}" ]]; then
  echo "ERROR: Missing ${AGSYS_EVAL_ROOT}. Run first:" >&2
  echo "  python3 agent_systems/prepare_sandbox.py --force" >&2
  exit 1
fi

if [[ ! -d "${AGSYS_EVAL_ROOT}/memory" ]] || [[ ! -d "${AGSYS_EVAL_ROOT}/qas" ]]; then
  echo "ERROR: ${AGSYS_EVAL_ROOT} is incomplete. Run first:" >&2
  echo "  python3 agent_systems/prepare_sandbox.py --force" >&2
  exit 1
fi

TARGET_PARENT="$(dirname "${TARGET_DIR}")"
if [[ -e "${TARGET_DIR}" ]]; then
  if [[ ! -w "${TARGET_DIR}" ]]; then
    cat >&2 <<EOF
ERROR: Target directory is not writable: ${TARGET_DIR}

You are probably packaging directly into the benchmark user's home as your
normal user. That is expected to need sudo because the directory is owned by
the openclaw-bench user.

Use one of these patterns:
  sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh ${TARGET_DIR}
  bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh ./bundle
  sudo rsync -a ./bundle/ ${TARGET_DIR}/
EOF
    exit 1
  fi
else
  if [[ ! -d "${TARGET_PARENT}" ]] || [[ ! -w "${TARGET_PARENT}" ]]; then
    cat >&2 <<EOF
ERROR: Parent directory is not writable: ${TARGET_PARENT}

If you want to package directly into another user's home, rerun with sudo:
  sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh ${TARGET_DIR}
EOF
    exit 1
  fi
fi

# Verify no GT files will be copied.
GT_FILES=(
  "data/atm-bench/atm-bench.json"
  "data/atm-bench/atm-bench-hard.json"
)
for f in "${GT_FILES[@]}"; do
  if [[ -f "${f}" ]]; then
    echo "INFO: Ground-truth file ${f} exists in repo but will NOT be copied."
  fi
done

echo "Packaging benchmark bundle to: ${TARGET_DIR}"

mkdir -p "${TARGET_DIR}/agent_systems"

# Copy agent_systems/ excluding:
#   - eval_root/runs/ (previous run outputs — start fresh)
#   - eval_root/.* (hidden runtime residue like .codex-runtime)
#   - any .git artifacts
#   - __pycache__
rsync -a \
  --exclude="eval_root/" \
  --exclude="eval_root_sgm/runs/" --exclude="eval_root_sgm/.*" \
  --exclude="eval_root_dm/" \
  --exclude="eval_root_raw/" \
  --exclude="eval_root_archive/" \
  --exclude="__pycache__/" \
  --exclude=".git" \
  --exclude="*.pyc" \
  agent_systems/ "${TARGET_DIR}/agent_systems/"

# Re-include the configured sgm eval_root only (other variants are excluded above).
# This is the single source of truth shipped to the benchmark user.
if [[ "${EVAL_ROOT_NAME}" != "eval_root_sgm" ]]; then
  rsync -a \
    --exclude="runs/" \
    --exclude=".*" \
    "${AGSYS_EVAL_ROOT}/" "${TARGET_DIR}/agent_systems/${EVAL_ROOT_NAME}/"
fi

# The packaged benchmark bundle should not contain any symlinks before runs start.
SYMLINKS_FOUND=0
while IFS= read -r link_path; do
  [[ -n "${link_path}" ]] || continue
  echo "ERROR: Unexpected symlink in packaged bundle: ${link_path}" >&2
  SYMLINKS_FOUND=1
done < <(find "${TARGET_DIR}/agent_systems" -type l -print 2>/dev/null)

if [[ ${SYMLINKS_FOUND} -ne 0 ]]; then
  echo "" >&2
  echo "ERROR: Bundle packaging aborted because symlinks were found." >&2
  echo "Recreate eval_root with prepare_sandbox.py and remove hidden runtime residue before packaging." >&2
  exit 1
fi

# Verify the bundle does NOT contain GT data.
_check_no_gt() {
  local bundle="$1"
  local found=0
  for pattern in "atm-bench*.json"; do
    while IFS= read -r match; do
      # eval_root/qas/*/question.json is fine; only flag files that look like
      # the full ground-truth dataset.
      if python3 -c "
import json, sys
d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
if isinstance(d, list) and len(d) > 5:
    first = d[0] if d else {}
    if 'answer' in first or 'ground_truth' in first:
        sys.exit(1)
" "${match}" 2>/dev/null; then
        continue
      fi
      echo "WARNING: Possible GT file in bundle: ${match}" >&2
      found=1
    done < <(find "${bundle}" -name "${pattern}" -type f 2>/dev/null)
  done
  return ${found}
}

if ! _check_no_gt "${TARGET_DIR}"; then
  echo "WARNING: Possible ground-truth files detected in the bundle. Review before distributing." >&2
fi

# Verify bundle has required files.
REQUIRED=(
  "agent_systems/config.py"
  "agent_systems/scripts/common.sh"
  "agent_systems/scripts/openclaw/run_openclaw.sh"
  "agent_systems/scripts/openclaw/run_openclaw_bench_user.sh"
  "agent_systems/scripts/openclaw/smoke_test_openclaw.sh"
  "agent_systems/${EVAL_ROOT_NAME}/question_ids.txt"
  "agent_systems/${EVAL_ROOT_NAME}/prompts/system_prompt.txt"
  "agent_systems/${EVAL_ROOT_NAME}/prompts/qa_schema.json"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "${TARGET_DIR}/${f}" ]]; then
    echo "ERROR: Missing required file in bundle: ${f}" >&2
    MISSING=1
  fi
done

MEMORY_COUNT=$(find "${TARGET_DIR}/agent_systems/${EVAL_ROOT_NAME}/memory" -name "*.json" -type f 2>/dev/null | wc -l)
QAS_COUNT=$(find "${TARGET_DIR}/agent_systems/${EVAL_ROOT_NAME}/qas" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

echo ""
echo "Bundle contents:"
echo "  Memory files:    ${MEMORY_COUNT}"
echo "  Question dirs:   ${QAS_COUNT}"
echo "  Target:          ${TARGET_DIR}"

if [[ ${MISSING} -ne 0 ]]; then
  echo ""
  echo "ERROR: Bundle is incomplete. Fix the missing files above." >&2
  exit 1
fi

echo ""
echo "Bundle is ready. Next steps:"
echo "  1. Transfer to the benchmark user:"
echo "     sudo chown -R openclaw-bench:openclaw-bench ${TARGET_DIR}"
echo "  2. As openclaw-bench, run:"
echo "     cd ${TARGET_DIR}"
echo "     bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh"
echo "     bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh"
