#!/usr/bin/env bash
#
# One-question smoke test for the OpenClaw benchmark runner.
#
# Runs a single question, then validates all expected outputs and isolation
# properties before committing to a full batch run.
#
# Usage (from benchmark bundle root as openclaw-bench user):
#   bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh
#   bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh <question_id>
#
# Prerequisites:
#   - OPENCLAW_HOME must be set (benchmark-user isolation)
#   - eval_root must be prepared
#   - OpenClaw must be installed and authenticated
#

set -o pipefail

if [[ ! -f "agent_systems/config.py" ]]; then
  echo "ERROR: Run this script from the benchmark bundle root." >&2
  exit 1
fi

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
export AGSYS_SKIP_EVAL=1
export AGSYS_OPENCLAW_BENCH_MODE=1

PROVIDER_ENV="${OPENCLAW_PROVIDER_ENV:-${OPENCLAW_PROVIDER_ENV_DEFAULT}}"

# ── Pick a test question ────────────────────────────────────────────────────
eval "$(python3 agent_systems/config.py --print-bash)" || {
  echo "ERROR: Failed to load config." >&2
  exit 1
}

if [[ -z "${AGSYS_OPENCLAW_MODEL_TAG:-}" ]]; then
  OPENCLAW_PRIMARY_MODEL="$(agsys_openclaw_primary_model "${AGSYS_OPENCLAW_CONFIG_SOURCE}")"
  if [[ -n "${OPENCLAW_PRIMARY_MODEL}" ]]; then
    export AGSYS_OPENCLAW_MODEL_TAG="${OPENCLAW_PRIMARY_MODEL}"
  fi
fi

SMOKE_QID="${1:-}"
if [[ -z "${SMOKE_QID}" ]]; then
  SMOKE_QID="$(head -1 "${AGSYS_EVAL_ROOT}/question_ids.txt")"
fi

if [[ -z "${SMOKE_QID}" ]]; then
  echo "ERROR: No question ID available for smoke test." >&2
  exit 1
fi

MODEL_TAG="$(python3 -c "t='${AGSYS_OPENCLAW_MODEL_TAG:-default}'.replace('/','_').replace(' ','_').replace(':','_').replace(',',''); print(t)")"

RUN_DIR="${AGSYS_RUNS_ROOT}/${AGSYS_RUN_TAG}/openclaw/${MODEL_TAG}/${SMOKE_QID}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  OpenClaw Smoke Test                                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Question:     ${SMOKE_QID}"
echo "║  Model tag:    ${MODEL_TAG}"
echo "║  OPENCLAW_HOME: ${OPENCLAW_HOME:-NOT SET}"
echo "║  Bundle root:  ${PWD}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────────────────
PREFLIGHT_OK=1

echo "=== Pre-flight checks ==="

# 1. OPENCLAW_HOME
if [[ -z "${OPENCLAW_HOME:-}" ]]; then
  echo "  FAIL: OPENCLAW_HOME is not set"
  PREFLIGHT_OK=0
else
  echo "  PASS: OPENCLAW_HOME=${OPENCLAW_HOME}"
fi

# 2. OpenClaw binary
OPENCLAW_BIN="${AGSYS_OPENCLAW_BIN:-openclaw}"
if command -v "${OPENCLAW_BIN}" >/dev/null 2>&1 || [[ -x "${OPENCLAW_BIN}" ]]; then
  echo "  PASS: OpenClaw binary found: ${OPENCLAW_BIN}"
else
  echo "  FAIL: OpenClaw binary not found: ${OPENCLAW_BIN}"
  PREFLIGHT_OK=0
fi

# 3. eval_root
if [[ -d "${AGSYS_EVAL_ROOT}/memory" ]] && [[ -d "${AGSYS_EVAL_ROOT}/qas" ]]; then
  echo "  PASS: eval_root is populated"
else
  echo "  FAIL: eval_root is missing or incomplete"
  PREFLIGHT_OK=0
fi

# 4. Question exists
if [[ -f "${AGSYS_EVAL_ROOT}/qas/${SMOKE_QID}/question.json" ]]; then
  echo "  PASS: Question ${SMOKE_QID} exists"
else
  echo "  FAIL: Question ${SMOKE_QID} not found in eval_root/qas/"
  PREFLIGHT_OK=0
fi

# 5. No bootstrap files in workspace
BOOTSTRAP_CLEAN=1
for f in SOUL.md AGENTS.md TOOLS.md BOOTSTRAP.md; do
  if [[ -f "${AGSYS_EVAL_ROOT}/${f}" ]]; then
    echo "  FAIL: Found ${AGSYS_EVAL_ROOT}/${f} — remove before benchmark"
    BOOTSTRAP_CLEAN=0
    PREFLIGHT_OK=0
  fi
done
if [[ ${BOOTSTRAP_CLEAN} -eq 1 ]]; then
  echo "  PASS: No bootstrap files in eval_root"
fi

# 6. No GT files in bundle
GT_FOUND=0
for gt_pattern in "atm-hard-*.json"; do
  while IFS= read -r match; do
    dir="$(dirname "${match}")"
    # eval_root/qas/*/question.json is expected; only flag things outside qas/
    if [[ "${dir}" != *"/qas/"* ]]; then
      python3 -c "
import json, sys
d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
if isinstance(d, list) and len(d) > 5:
    first = d[0] if d else {}
    if 'answer' in first or 'ground_truth' in first:
        sys.exit(1)
" "${match}" 2>/dev/null || {
        echo "  FAIL: Possible GT file: ${match}"
        GT_FOUND=1
        PREFLIGHT_OK=0
      }
    fi
  done < <(find . -name "${gt_pattern}" -type f 2>/dev/null)
done
if [[ ${GT_FOUND} -eq 0 ]]; then
  echo "  PASS: No ground-truth files detected in bundle"
fi

# 7. Config source exists
CONFIG_SRC="${AGSYS_OPENCLAW_CONFIG_SOURCE:-}"
if [[ -n "${CONFIG_SRC}" ]] && [[ -f "${CONFIG_SRC}" ]]; then
  echo "  PASS: Config source exists: ${CONFIG_SRC}"
else
  echo "  FAIL: Config source not found: ${CONFIG_SRC}"
  PREFLIGHT_OK=0
fi

# 8. Provider env exists (optional)
if [[ -f "${PROVIDER_ENV}" ]]; then
  echo "  PASS: Provider env exists: ${PROVIDER_ENV}"
else
  echo "  WARN: Provider env not found: ${PROVIDER_ENV}"
  echo "        Assuming auth is configured via OpenClaw-managed config/state."
fi

echo ""

if [[ ${PREFLIGHT_OK} -eq 0 ]]; then
  echo "Pre-flight checks FAILED. Fix the issues above before running." >&2
  exit 1
fi

if [[ -f "${PROVIDER_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PROVIDER_ENV}"
fi

echo "All pre-flight checks passed."
echo ""

# ── Run the single question ────────────────────────────────────────────────
echo "=== Running smoke test question: ${SMOKE_QID} ==="
echo ""

# Clear any previous run for this question.
rm -rf "${RUN_DIR}/output"

export AGSYS_SKIP_EVAL=1
bash agent_systems/scripts/openclaw/run_openclaw.sh "${SMOKE_QID}"
RUN_EXIT=$?

echo ""
echo "=== Post-run validation ==="

# ── Validate outputs ───────────────────────────────────────────────────────
POSTRUN_OK=1

# 1. answer.json
if [[ -f "${RUN_DIR}/output/answer.json" ]]; then
  if python3 -c "
import json, sys
d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
assert set(d.keys()) == {'id', 'question', 'answer'}, f'Bad keys: {set(d.keys())}'
assert str(d['id']) == sys.argv[2], f'ID mismatch: {d[\"id\"]} != {sys.argv[2]}'
assert isinstance(d['answer'], str) and len(d['answer']) > 0, 'Empty answer'
" "${RUN_DIR}/output/answer.json" "${SMOKE_QID}" 2>/dev/null; then
    ANSWER_PREVIEW="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['answer'][:100])" "${RUN_DIR}/output/answer.json" 2>/dev/null)"
    echo "  PASS: answer.json valid — ${ANSWER_PREVIEW}"
  else
    echo "  FAIL: answer.json exists but has invalid schema"
    POSTRUN_OK=0
  fi
else
  echo "  FAIL: answer.json not created"
  POSTRUN_OK=0
fi

# 2. trace.json
if [[ -f "${RUN_DIR}/output/trace.json" ]]; then
  TRACE_SIZE="$(wc -c < "${RUN_DIR}/output/trace.json")"
  echo "  PASS: trace.json exists (${TRACE_SIZE} bytes)"
else
  echo "  FAIL: trace.json not created"
  POSTRUN_OK=0
fi

# 3. usage.json
if [[ -f "${RUN_DIR}/output/usage.json" ]]; then
  echo "  PASS: usage.json exists"
else
  echo "  WARN: usage.json not created (may indicate OpenClaw didn't report tokens)"
fi

# 4. runtime_manifest.json
if [[ -f "${RUN_DIR}/output/runtime_manifest.json" ]]; then
  echo "  PASS: runtime_manifest.json exists"
  # Verify key fields
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
checks = [
    ('agent', 'openclaw'),
    ('web_search_enabled', False),
    ('web_fetch_enabled', False),
    ('browser_enabled', False),
    ('agent_to_agent_enabled', False),
    ('plugins_enabled', False),
    ('skills_enabled', False),
]
for key, expected in checks:
    actual = d.get(key)
    if actual != expected:
        print(f'  WARN: runtime_manifest.{key} = {actual!r} (expected {expected!r})')
    else:
        print(f'  PASS: runtime_manifest.{key} = {actual!r}')
" "${RUN_DIR}/output/runtime_manifest.json" 2>/dev/null
else
  echo "  FAIL: runtime_manifest.json not created"
  POSTRUN_OK=0
fi

# 4b. trace.json isolation audit
if [[ -f "${RUN_DIR}/output/trace.json" ]]; then
  TRACE_AUDIT="$(python3 - "${RUN_DIR}/output/trace.json" <<'PY'
import json, sys
from pathlib import Path

trace_path = Path(sys.argv[1])
text = trace_path.read_text(encoding="utf-8")

start = text.find("{")
if start < 0:
    print("FAIL:no_json_object")
    raise SystemExit(0)

obj = None
for idx in range(start, len(text)):
    if text[idx] != "}":
        continue
    candidate = text[start:idx + 1]
    try:
        obj = json.loads(candidate)
        break
    except json.JSONDecodeError:
        continue

if not isinstance(obj, dict):
    print("FAIL:no_json_object")
    raise SystemExit(0)

meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
report = meta.get("systemPromptReport", {}) if isinstance(meta, dict) else {}

bad_files = []
for entry in report.get("injectedWorkspaceFiles") or []:
    if not isinstance(entry, dict):
        continue
    name = entry.get("name")
    missing = entry.get("missing")
    if name in {"AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md"} and missing is False:
        bad_files.append(name)

skill_entries = report.get("skills", {}).get("entries") or []
skill_count = len(skill_entries) if isinstance(skill_entries, list) else -1

tool_entries = report.get("tools", {}).get("entries") or []
tool_names = []
if isinstance(tool_entries, list):
    for entry in tool_entries:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                tool_names.append(name)

bad_tools = sorted(set(tool_names) - {"read", "exec"})
missing_tools = sorted({"read", "exec"} - set(tool_names))
plugin_lines = "Registered plugin command:" in text

if bad_files:
    print("FAIL:bootstrap:" + ",".join(sorted(bad_files)))
elif skill_count not in (0,):
    print(f"FAIL:skills:{skill_count}")
elif plugin_lines:
    print("FAIL:plugins:registered_plugin_commands")
elif missing_tools:
    print("FAIL:tools_missing:" + ",".join(missing_tools))
elif bad_tools:
    print("FAIL:tools:" + ",".join(bad_tools))
else:
    print("PASS")
PY
)"
  case "${TRACE_AUDIT}" in
    PASS)
      echo "  PASS: trace.json systemPromptReport shows no bootstrap injection, no skills, and only read/exec tools"
      ;;
    FAIL:bootstrap:*)
      echo "  FAIL: trace.json shows injected bootstrap files: ${TRACE_AUDIT#FAIL:bootstrap:}"
      POSTRUN_OK=0
      ;;
    FAIL:skills:*)
      echo "  FAIL: trace.json shows bundled/local skills still exposed: ${TRACE_AUDIT#FAIL:skills:}"
      POSTRUN_OK=0
      ;;
    FAIL:plugins:*)
      echo "  FAIL: trace.json shows plugin commands still registered"
      POSTRUN_OK=0
      ;;
    FAIL:tools_missing:*)
      echo "  FAIL: trace.json is missing expected tools: ${TRACE_AUDIT#FAIL:tools_missing:}"
      POSTRUN_OK=0
      ;;
    FAIL:tools:*)
      echo "  FAIL: trace.json shows unexpected tools still exposed: ${TRACE_AUDIT#FAIL:tools:}"
      POSTRUN_OK=0
      ;;
    *)
      echo "  WARN: could not audit trace.json systemPromptReport (${TRACE_AUDIT})"
      ;;
  esac
fi

# 5. stderr.log
if [[ -f "${RUN_DIR}/output/stderr.log" ]]; then
  STDERR_SIZE="$(wc -c < "${RUN_DIR}/output/stderr.log")"
  echo "  PASS: stderr.log exists (${STDERR_SIZE} bytes)"
  # Check for suspicious patterns
  for pattern in "browser" "plugin.*load" "SOUL.md" "AGENTS.md" "BOOTSTRAP.md" "skill.*load"; do
    if grep -qi "${pattern}" "${RUN_DIR}/output/stderr.log" 2>/dev/null; then
      echo "  WARN: stderr.log contains suspicious pattern: ${pattern}"
    fi
  done
else
  echo "  WARN: stderr.log not created"
fi

# 6. Temp workspace cleanup
if [[ -d "/tmp/agsys-openclaw-"* ]] 2>/dev/null; then
  echo "  WARN: Leftover temp workspace(s) found in /tmp/agsys-openclaw-*"
else
  echo "  PASS: No leftover temp workspaces"
fi

echo ""

if [[ ${POSTRUN_OK} -eq 1 ]] && [[ ${RUN_EXIT} -eq 0 ]]; then
  echo "══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST PASSED"
  echo ""
  echo "  The single-question run completed successfully."
  echo "  Review the outputs above, then proceed with full batch:"
  echo ""
  echo "    bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh"
  echo "══════════════════════════════════════════════════════════════"
else
  echo "══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST FAILED"
  echo ""
  echo "  Review the failures above. Check:"
  echo "    ${RUN_DIR}/output/stderr.log"
  echo "    ${RUN_DIR}/output/trace.json"
  echo "══════════════════════════════════════════════════════════════"
  exit 1
fi
